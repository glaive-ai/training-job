import copy
import logging
import torch
import transformers
import datasets
import glaive_utils

from dataclasses import dataclass
from torch.utils.data import Dataset
from typing import List, Dict, Sequence, Optional

logger = logging.getLogger(__name__)

def tokenize_sft_data(data : List[Dict],
                      tokenizer: transformers.PreTrainedTokenizer, 
                      ignore_index : int = -100,
                      prompt_key : str ='prompt',
                      response_key : str ='response',
                      max_seq_len : int = 1900):
    """Tokenize the sft data"""
    input_ids, labels, response_len = list(), list(), list()
    logger.info(tokenizer.model_max_length)

    for i, example in enumerate(data):
        if any([key not in example for key in (prompt_key, response_key)]):
            raise ValueError(f'{prompt_key} or {response_key} not found in {i}-th example')
        instruction = example[prompt_key] + example[response_key]
        ex_input_ids = tokenizer(instruction,
                                 return_tensors="pt",
                                 padding="longest",
                                 max_length=tokenizer.model_max_length).input_ids[0]
        ex_prompt_len = len(tokenizer(example[prompt_key],
                                      return_tensors="pt",
                                      padding="longest",
                                      max_length=tokenizer.model_max_length).input_ids[0])
        ex_response_len = len(ex_input_ids) - ex_prompt_len
        ex_labels = copy.deepcopy(ex_input_ids)
        ex_labels[:ex_prompt_len] = ignore_index
        if len(ex_input_ids) <= max_seq_len:
            input_ids.append(ex_input_ids)
            labels.append(ex_labels)
            response_len.append(ex_response_len)
    return input_ids, labels, response_len

class SFTDataset(Dataset):
    """Dataset for Supervised Fine-Tuning (SFT)."""

    def __init__(self, 
                 data_path: str, 
                 tokenizer: transformers.PreTrainedTokenizer, 
                 prompt_key : str = 'prompt',
                 response_key : str = 'response',
                 ignore_index : int = -100):
        super(SFTDataset, self).__init__()
        self.tokenizer = tokenizer
        self.data_path = data_path
        self.ignore_index = ignore_index
        logger.info("Loading data...")
        self.loaded_dataset = glaive_utils.jload(data_path)
        logger.info("Tokenizing data...")
        self.input_ids, self.labels, self.response_len = tokenize_sft_data(self.loaded_dataset, self.tokenizer, 
                                                                           ignore_index, prompt_key, response_key)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], 
                    labels=self.labels[i],
                    response_len=self.response_len[i])


class HF_SFTDataset(Dataset):
    """Dataset for Supervised Fine-Tuning (SFT)."""
    def __init__(self, 
                 data_path: str, 
                 tokenizer: transformers.PreTrainedTokenizer, 
                 prompt_key : str = 'prompt',
                 response_key : str = 'response',
                 split : str = "train",
                 ignore_index : int = -100,
                 max_examples : int = None):
        super(HF_SFTDataset, self).__init__()
        self.tokenizer = tokenizer
        self.data_path = data_path
        self.ignore_index = ignore_index
        self.max_examples = max_examples
        logger.info("Loading HF dataset...")
        self.loaded_dataset = datasets.load_dataset(data_path, split=split)
        if max_examples is not None:
            self.loaded_dataset = self.loaded_dataset.select(indices=range(max_examples))
        logger.info("Tokenize dataset...")
        self.input_ids, self.labels, self.response_len = tokenize_sft_data(self.loaded_dataset, self.tokenizer, 
                                                                           ignore_index, prompt_key, response_key)

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
    ignore_index : int = -100

    def __call__(self, examples: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [example['input_ids'] for example in examples]
        labels = [example['labels'] for example in examples]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=self.ignore_index
        )
        response_len = [example['response_len'] for example in examples]
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            response_len=response_len)