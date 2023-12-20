import copy
import logging
import torch
import transformers
import utils

from dataclasses import dataclass
from torch.utils.data import Dataset
from typing import Dict, Sequence, Optional


class SFTDataset(Dataset):
    """Dataset for Supervised Fine-Tuning (SFT)."""

    def __init__(self, data_path: str, 
                 tokenizer: transformers.PreTrainedTokenizer, 
                 ignore_index : int = -100):
        super(SFTDataset, self).__init__()
        self.tokenizer = tokenizer
        self.data_path = data_path
        self.ignore_index = ignore_index
        logging.info("Loading data...")
        self.loaded_dataset = utils.jload(data_path)
        self.input_ids = list()
        self.labels = list()
        self.response_len = list()
        logging.info("Formatting inputs...")
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
            ex_labels[:ex_prompt_len] = self.ignore_index
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
    ignore_index : int = -100

    def __call__(self, examples: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [example['input_ids'] for example in examples]
        labels = [example['labels'] for example in examples]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        ).to(self.device)
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=self.ignore_index
        ).to(self.device)
        response_len = [example['response_len'] for example in examples]
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            response_len=response_len)