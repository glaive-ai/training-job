import copy
import logging
import json
import requests
import os
import utils
import torch
import time
import transformers
import wandb

from dataclasses import dataclass, field, asdict
from utils import folder_exists_on_gcs, upload_blob, GLAIVE_BUCKET
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.functional import cross_entropy
from typing import Dict, Optional, Sequence

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "<pad>"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "<unk>"


@dataclass
class ModelArguments:
    model_id: str = field(default=None, metadata={"help": "Model ID for tracking."})
    model_name_or_path: str = field(default="mistralai/Mistral-7B-v0.1")
    lora_r: Optional[int] = field(default=8)
    lora_alpha: Optional[float] = field(default=32)
    lora_dropout: Optional[float] = field(default=0.0)

@dataclass
class DataArguments:
    data_path: Optional[str] = field(default=None, metadata={"help": "Path to the training data."})
    data_url: Optional[str] = field(default=None, metadata={"help": "URL to download the training data."})
    callback_url: Optional[str] = field(default=None, metadata={"help": "URL for callback notifications."})


@dataclass
class TrainingArguments:
    per_device_train_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU/MPS/NPU core/CPU for training."}
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )
    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for AdamW."})
    num_train_epochs: int = field(default=10, metadata={"help": "Total number of training epochs to perform."})
    cache_dir: Optional[str] = field(default=None)
    optim: Optional[str] = field(default="adamw_torch_fused")
    logging_steps: float = field(
        default=10,
        metadata={
            "help": (
                "Log every X updates steps. "
            )
        },
    )
    save_steps: float = field(
        default=10,
        metadata={
            "help": (
                "Save checkpoint every X updates steps.  "
            )
        },
    )
    eval_steps: int = field(
        default=10,
        metadata={
            "help": (
                "Run an evaluation every X steps."
            )
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    use_amp : Optional[bool] = field(default=True, 
                                     metadata={"help": "use mixed precision?"})

def callback_completion(callback_url:str,model_url:str,failed:bool,error:str=None):
    """
    Sends a callback to the specified URL.
    """
    payload = {'failed': failed, "url": model_url, "error": error}
    response = requests.post(callback_url, json=payload)
    return response


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


class SFTDataset(Dataset):
    """Dataset for Supervised Fine-Tuning (SFT)."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SFTDataset, self).__init__()
        self.tokenizer = tokenizer
        logging.warning("Loading data...")
        self.loaded_dataset = utils.jload(data_path)
        self.input_ids = list()
        self.labels = list()
        self.response_len = list()
        logging.warning("Formatting inputs...")
        self._preprocess()

    def _preprocess(self):
        """Preprocess the data by tokenizing."""
        for i, example in enumerate(self.loaded_dataset):
            if any([key not in example for key in ('prompt', 'response')]):
                raise ValueError('`prompt` or `response` not found in {i} example', i=i)
            instruction = example['prompt'] + example['response']
            ex_input_ids = self.tokenizer(instruction,
                                      return_tensors="pt",
                                      padding="longest",
                                      max_length=self.tokenizer.model_max_length,
                                      truncation=True).input_ids[0]
            ex_prompt_len = len(self.tokenizer(example['prompt'],
                                      return_tensors="pt",
                                      padding="longest",
                                      max_length=self.tokenizer.model_max_length,
                                      truncation=True).input_ids[0])
            ex_response_len = len(ex_input_ids) - ex_prompt_len
            ex_labels = copy.deepcopy(ex_input_ids)
            ex_labels[:ex_prompt_len] = IGNORE_INDEX
            self.input_ids.append(ex_input_ids)
            self.labels.append(ex_labels)
            self.response_len.append(ex_response_len)


    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], 
                    labels=self.labels[i],
                    response_len=self.response_len[i])


@dataclass
class DataCollatorForSFTDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    device: torch.cuda.Device

    def __call__(self, examples: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [example['input_ids'] for example in examples]
        labels = [example['labels'] for example in examples]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        ).to(self.device)
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        ).to(self.device)
        response_len = [example['response_len'] for example in examples]
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            response_len=response_len)

def eval_loop(model, dataloader):
    model.eval()
    loss = 0.0
    num_response_tokens = 0
    with torch.no_grad():
        for batch in dataloader:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                output = model(input_ids=batch['input_ids'], 
                                attention_mask=batch['attention_mask'])
                loss = cross_entropy(output.logits.transpose(1, 2), 
                                        batch['labels'], 
                                        reduction='sum',
                                        ignore_index=IGNORE_INDEX)
                loss += loss.item()
                num_response_tokens += sum(batch['response_len'])
    return loss/num_response_tokens

def train():
    try:
        parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        output_dir = 'output'

        os.makedirs(output_dir, exist_ok=True)

        if folder_exists_on_gcs(model_args.model_id, GLAIVE_BUCKET):
            raise ValueError(f"Folder `{model_args.model_id}` already exists on GCS. Did you run this experiment before?")

        with open(os.path.join(output_dir, 'args.json'), 'w') as f: 
            args_dict = dict(model_args=asdict(model_args), 
                             data_args=asdict(data_args),
                             training_args=asdict(training_args))
            json.dump(args_dict, f)

        if data_args.data_path is None and data_args.data_url is None:
            raise ValueError("Must specify either data_path or data_url.")
        
        if data_args.data_url is not None:
            response = requests.get(data_args.data_url)
            with open('data.jsonl', 'w') as file:
                file.write(response.text)

        data_path = "data.jsonl"
        wandb.init(project="train-lora-job", 
                   tags=[model_args.model_id],
                   dir=output_dir)
        
        print("Loading the model...")
        start_model_load = time.time()
        device = 'cuda:0'
        model = transformers.AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-v0.1"
        ).to(device)
        wandb.log(dict(model_load_time=int(time.time() - start_model_load)))

        logging.info("Loading the tokenizer...")
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            model_max_length=training_args.max_seq_length,
            padding_side="right",
            use_fast=False
        )

        special_tokens_dict = dict()
        if tokenizer.pad_token is None:
            special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
        if tokenizer.eos_token is None:
            special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
        if tokenizer.unk_token is None:
            special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=special_tokens_dict,
            tokenizer=tokenizer,
            model=model,
        )

        lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM, 
                                 r=model_args.lora_r,
                                 lora_alpha=model_args.lora_alpha)
        peft_model = get_peft_model(model, lora_config)

        logging.info("Loading the data...")
        dataset = SFTDataset(tokenizer=tokenizer, data_path=data_path)
        train_dataset, val_dataset = random_split(dataset, [0.9, 0.1])
        collator_fn = DataCollatorForSFTDataset(tokenizer=tokenizer, device=device)

        training_dataloader = DataLoader(train_dataset, 
                                         batch_size=training_args.per_device_train_batch_size,
                                         shuffle=True,
                                         collate_fn=collator_fn)
        val_dataloader = DataLoader(val_dataset, 
                                    batch_size=training_args.per_device_train_batch_size,
                                    collate_fn=collator_fn)                                 
        optimizer = torch.optim.AdamW(peft_model.parameters(), 
                                      lr=training_args.learning_rate)
        scaler = torch.cuda.amp.GradScaler(enabled=training_args.use_amp)

        i = 0
        step = 0
        log = dict(loss=0.0, num_response_tokens=0)
        best_val_loss = 1e10

        start_training = time.time()
        for epoch in range(training_args.num_train_epochs):
            for batch in training_dataloader:
                model.train()
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    output = model(input_ids=batch['input_ids'], 
                                   attention_mask=batch['attention_mask'])
                    loss = cross_entropy(output.logits.transpose(1, 2), 
                                         batch['labels'], 
                                         reduction='sum',
                                         ignore_index=IGNORE_INDEX)
                scaler.scale(loss).backward()
                log['loss'] += loss.item()
                log['num_response_tokens'] += sum(batch['response_len'])

                if (i + 1) % training_args.gradient_accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    step += 1

                if step % training_args.logging_steps == 0:
                    wandb.log(dict(train=dict(
                                   loss=log['loss']/log['num_response_tokens'])), 
                                   step=step)
                    log = dict(loss=0.0, num_response_tokens=0)

                if step % training_args.eval_steps == 0:
                    val_loss = eval_loop(model, val_dataloader)
                    wandb.log(dict(valid=dict(
                                   loss=val_loss)), 
                                   step=step)
                    # save model
                    if val_loss < best_val_loss:
                        ckpt_dir = os.path.join(output_dir, f"CHECKPOINT")
                        peft_model.save_pretrained(ckpt_dir)
                        best_val_loss = val_loss
                i+=1
        
        end_training = time.time()
        wandb.log(dict(training_time=int(end_training - start_training)))

        url = utils.upload_blob(os.path.join(output_dir, "CHECKPOINT", "adapter_model.safetensors"), 
                                os.path.join(model_args.model_id, "CHECKPOINT", "adapter_model.safetensors"))
        if data_args.callback_url is not None: 
             callback_completion(data_args.callback_url,url,False)

    except Exception as e:
        callback_completion(data_args.callback_url,None,True,str(e))

if __name__ == "__main__":
    train()
    # python train.py --model_id test --data_url "https://storage.googleapis.com/glaive-data/train_cc4ee6ee-6f39-44e4-9664-f79a8dc65904.jsonl?Expires=1703359938&GoogleAccessId=storage-admin%40glaive-393514.iam.gserviceaccount.com&Signature=hsTPzGuuntiQpT4lcr%2FU8n0KcPkbHK5HMeft5ek%2BZpTG%2Fley6t5MtIcDVZ%2FBH1%2BCfSbmYQH29%2FSmMxcPl1ewdPkseV5TqIHrPq%2F3ivkm3X4RqTk0kG7TqDe69rey1zC7SQozJWlIhpLL0hD6eIxf82tIhErH8e9qUekzIAxp2KDkRLTdZuwmpDnGvDiiCHqS%2FNXMh7kTuqSKSPOz9g3zsAP3iKdFT3uznUgCTcqhWSpDMNAQO0zxv%2FiTC0DP9HfnAwd2oViQsOn%2BbqF1EvmqjWaY6BFJnsziSGtlH%2BBmyOCzAaxNkfrFlOcmZDNZ8%2BIdl29DL2oYdg5M9xUodtMOgA%3D%3D"