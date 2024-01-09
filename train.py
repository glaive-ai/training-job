import copy
import logging
import json
import requests
import os
import glaive_utils
import torch
import time
import transformers
import wandb
import logging


from dataclasses import dataclass, field, asdict
from dataset import HF_SFTDataset, SFTDataset, DataCollatorForSFTDataset
from glaive_utils import folder_exists_on_gcs, GLAIVE_BUCKET
from logger_config import setup_logging
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from torch.utils.data import DataLoader, random_split
from torch.utils.data import DistributedSampler
from torch.nn.functional import cross_entropy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from typing import Dict, Optional, Sequence
from urllib.parse import urlparse
from wrap_policy import fsdp_auto_wrap_policy
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer

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
    local_data_path: Optional[str] = field(default=None, metadata={"help": "Local path to the training data."})
    gcs_data_url: Optional[str] = field(default=None, metadata={"help": "GCS URL to download the training data."})
    hf_data_path: Optional[str] = field(default=None, metadata={"help": "HF data path"})
    prompt_key: Optional[str] = field(default="prompt", metadata={"help": "Column of the prompt in the dataset"})
    response_key: Optional[str] = field(default="response", metadata={"help": "Column of the response in the dataset"})
    callback_url: Optional[str] = field(default=None, metadata={"help": "URL for callback notifications."})

bfSixteen = MixedPrecision(
    param_dtype=torch.bfloat16,
    # Gradient communication precision.
    reduce_dtype=torch.bfloat16,
    # Buffer precision.
    buffer_dtype=torch.bfloat16,
    cast_forward_inputs=True,
)

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
    enable_fsdp : Optional[bool] = field(default=True, 
                                     metadata={"help": "use fully sharded data parallel?"})

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

def eval_loop(model, dataloader, logger):
    model.eval()
    log = dict(loss=0.0, num_response_tokens=0)

    with torch.no_grad():
        for batch in dataloader:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                output = model(input_ids=batch['input_ids'], 
                                attention_mask=batch['attention_mask'])
                loss = cross_entropy(output.logits.transpose(1, 2), 
                                     batch['labels'], 
                                     reduction='sum',
                                     ignore_index=IGNORE_INDEX)
            log['loss'] += loss.item()
            log['num_response_tokens'] += sum(batch['response_len'])
        logger.info(f"val loss: {log['loss']} toks: {log['num_response_tokens']}")
    return log['loss']/log['num_response_tokens']


def train(model_args, data_args, training_args):
    if training_args.enable_fsdp:
        torch.distributed.init_process_group("nccl")
        # torchrun specific
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
    else:
        rank = 0
        world_size = 1

    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    log_file_path = os.path.join(output_dir, '0.log')
    if training_args.enable_fsdp:
        log_file_path = os.path.join(output_dir, str(rank) + '.log')

    setup_logging(log_file_path, rank)
    logger = logging.getLogger(__name__)

    if rank == 0:
        if folder_exists_on_gcs(model_args.model_id, GLAIVE_BUCKET):
            raise ValueError(f"Folder `{model_args.model_id}` already exists on GCS. Did you run this experiment before?")

        with open(os.path.join(output_dir, 'args.json'), 'w') as f: 
            args_dict = dict(model_args=asdict(model_args), 
                                data_args=asdict(data_args),
                                training_args=asdict(training_args))
            json.dump(args_dict, f)

    if data_args.local_data_path is None and data_args.gcs_data_url is None and data_args.hf_data_path is None:
        raise ValueError("Must specify either `local_data_path`, `gcs_data_url` or `hf_data_path`")
    
    if rank == 0:
        if data_args.gcs_data_url is not None:
            logger.info(f"Downloading `{data_args.gcs_data_url}` to `data.jsonl`")
            response = requests.get(data_args.gcs_data_url)
            downloaded_filename = os.path.basename(urlparse(data_args.gcs_data_url).path) 
            with open(downloaded_filename, 'w') as file:
                file.write(response.text)
                data_args.local_data_path = downloaded_filename
        
    if training_args.enable_fsdp:
        torch.distributed.barrier()

    if rank == 0:
        wandb.init(project="train-lora-job", 
                    tags=[model_args.model_id],
                    dir=output_dir)
    
    logger.info("Loading the model...")
    start_model_load = time.time()
    device = 'cuda'
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path
    ).to(device)

    if rank == 0:
        wandb.log(dict(model_load_time=int(time.time() - start_model_load)))


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
    model = get_peft_model(model, lora_config)

    if training_args.enable_fsdp:
        my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, MistralDecoderLayer)
        model = FSDP(model,
                     sharding_strategy=ShardingStrategy.NO_SHARD,
                     auto_wrap_policy=my_auto_wrapping_policy,
                     mixed_precision=bfSixteen,
                     device_id=torch.cuda.current_device(),
                     limit_all_gathers=True)

    logger.info("Loading the data...")
    if data_args.local_data_path is not None:
        dataset = SFTDataset(tokenizer=tokenizer, data_path=data_args.local_data_path, 
                             prompt_key=data_args.prompt_key, response_key=data_args.response_key,
                             ignore_index=IGNORE_INDEX)
        if data_args.hf_data_path is not None:
            logging.warning("Loading dataset from `local_data_path` and ignoring the HF dataset specified by `hf_data_path`")
    else: 
        dataset = HF_SFTDataset(tokenizer=tokenizer, data_path=data_args.hf_data_path, 
                                prompt_key=data_args.prompt_key, response_key=data_args.response_key,
                                ignore_index=IGNORE_INDEX)
    train_dataset, val_dataset = random_split(dataset, [0.97, 0.03])

    collator_fn = DataCollatorForSFTDataset(tokenizer=tokenizer, device=device, 
                                            ignore_index=IGNORE_INDEX)
    train_dist_sampler = DistributedSampler(train_dataset, 
                                      rank=rank,
                                      num_replicas=world_size,
                                      shuffle=True)
    training_dataloader = DataLoader(train_dataset, 
                                     batch_size=training_args.per_device_train_batch_size,
                                     sampler=train_dist_sampler,
                                     collate_fn=collator_fn)
    val_dist_sampler = DistributedSampler(val_dataset, 
                                      rank=rank,
                                      num_replicas=world_size,
                                      shuffle=True)
    val_dataloader = DataLoader(val_dataset, 
                                batch_size=training_args.per_device_train_batch_size,
                                sampler=val_dist_sampler,
                                collate_fn=collator_fn)                                 
    optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr=training_args.learning_rate)
    if training_args.enable_fsdp:
        scaler = ShardedGradScaler()
    else:
        scaler = torch.cuda.amp.GradScaler()

    i = 0
    step = 0
    log = dict(loss=0.0, num_response_tokens=0)
    best_val_loss = 1e10

    logger.info("Start training...")
    start_training = time.time()
    for epoch in range(training_args.num_train_epochs):
        for batch in training_dataloader:
            logger.info(batch['input_ids'].shape)
            model.train()
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                output = model(input_ids=batch['input_ids'], 
                                    attention_mask=batch['attention_mask'])
                loss = cross_entropy(output.logits.transpose(1, 2), 
                                     batch['labels'], 
                                     reduction='sum',
                                     ignore_index=IGNORE_INDEX)
            
            log['loss'] += loss.item()
            log['num_response_tokens'] += sum(batch['response_len'])

            scaler.scale(loss).backward()
            
            if (i + 1) % training_args.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                step += 1

                if step % training_args.logging_steps == 0:
                    tmp_loss = log['loss']/log['num_response_tokens']
                    logger.info(f"Step {step} - train-loss: {tmp_loss}")
                    if rank == 0:
                        wandb.log(dict(train=dict(
                                        loss=tmp_loss)), 
                                        step=step)
                    log = dict(loss=0.0, num_response_tokens=0)

                if step % training_args.eval_steps == 0:
                    val_loss = eval_loop(model, val_dataloader, logger)
                    model.train()
                    if rank == 0:
                        wandb.log(dict(valid=dict(
                                        loss=val_loss)), 
                                        step=step)
                    logger.info(f"Step {step} - val-loss: {val_loss}")
                    # only save best model (i.e., early stopping)

                    if training_args.enable_fsdp:
                        torch.distributed.barrier()

                    if val_loss < best_val_loss:
                        ckpt_dir = os.path.join(output_dir, f"CHECKPOINT")
                        model.save_pretrained(ckpt_dir)
                        best_val_loss = val_loss
                    
                    if training_args.enable_fsdp:
                        torch.distributed.barrier()
            i+=1
    
    end_training = time.time()
    if rank == 0:
        wandb.log(dict(training_time=int(end_training - start_training)))

    if rank == 0 :
        gcs_urls = dict()
        for file in os.listdir(ckpt_dir):
            filename = os.path.basename(file)
            gcs_urls[filename] = glaive_utils.upload_blob(
                os.path.join(ckpt_dir, filename), 
                os.path.join(model_args.model_id, "CHECKPOINT", filename))
        gcs_urls["args.json"] = glaive_utils.upload_blob(
            os.path.join(output_dir, "args.json"), 
            os.path.join(model_args.model_id, "args.json"))
        gcs_urls["log.txt"] = glaive_utils.upload_blob(os.path.join(output_dir, "log.txt"), 
            os.path.join(model_args.model_id, "log.txt"))
    
        gcs_model_url = [k for k in gcs_urls.keys() if "safetensors" in k][0]
        if data_args.callback_url is not None: 
             glaive_utils.callback_completion(data_args.callback_url,gcs_model_url,False)

if __name__ == "__main__":
    try:
        parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        train(model_args, data_args, training_args)
    except Exception as e:
        callback_completion(data_args.callback_url,None,True,str(e))
    
    # python train.py --model_id test --gcs_data_url "https://storage.googleapis.com/glaive-data/train_cc4ee6ee-6f39-44e4-9664-f79a8dc65904.jsonl?Expires=1703359938&GoogleAccessId=storage-admin%40glaive-393514.iam.gserviceaccount.com&Signature=hsTPzGuuntiQpT4lcr%2FU8n0KcPkbHK5HMeft5ek%2BZpTG%2Fley6t5MtIcDVZ%2FBH1%2BCfSbmYQH29%2FSmMxcPl1ewdPkseV5TqIHrPq%2F3ivkm3X4RqTk0kG7TqDe69rey1zC7SQozJWlIhpLL0hD6eIxf82tIhErH8e9qUekzIAxp2KDkRLTdZuwmpDnGvDiiCHqS%2FNXMh7kTuqSKSPOz9g3zsAP3iKdFT3uznUgCTcqhWSpDMNAQO0zxv%2FiTC0DP9HfnAwd2oViQsOn%2BbqF1EvmqjWaY6BFJnsziSGtlH%2BBmyOCzAaxNkfrFlOcmZDNZ8%2BIdl29DL2oYdg5M9xUodtMOgA%3D%3D"
    # python train.py --model_id test_hf --hf_data_path ise-uiuc/Magicoder-OSS-Instruct-75K --prompt_key problem --response_key solution --num_train_epochs 2 --per_device_train_batch_size 2 --gradient_accumulation_steps 4
        
    # torchrun --nnodes 1 --nproc_per_node 2 train.py --model_id test_hf --hf_data_path ise-uiuc/Magicoder-OSS-Instruct-75K --prompt_key problem --response_key solution --num_train_epochs 2 --per_device_train_batch_size 1 --gradient_accumulation_steps 4 --logging_steps 1 --eval_steps 100 --enable_fsdp