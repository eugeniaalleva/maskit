from torch import nn
import torch.nn.functional as F
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


class MaskitModel(nn.Module):
    def __init__(self, model_name, verbalizer_map):
        super().__init__()
        self.backbone = AutoModelForMaskedLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.verbalizer_map = verbalizer_map
        self.verbalizer_id_dict = self.convert_map_to_id()
        self.label_word_ids = list(self.verbalizer_id_dict.values())
        self.n_classes = len(verbalizer_map)

    def verbalizer(self, full_logits, mask_index):
        batch_size = full_logits.size(0)
        # extract logits for mask token
        label_word_logits = full_logits[torch.arange(batch_size), mask_index]
        # extract logits on label words
        label_word_logits = label_word_logits[:, self.label_word_ids]
        # normalize
        label_word_logits = (F.softmax(label_word_logits.reshape(batch_size, -1),dim=-1)
                             .reshape(*label_word_logits.shape)
                             )
        # convert to logits
        label_word_logits = torch.log(label_word_logits+1e-15)
        # aggregate logits
        label_word_logits = label_word_logits.sum(-1)/torch.ones_like(label_word_logits).sum(-1)
        return label_word_logits

    def forward(self, **kwargs):
        inputs = {
            "input_ids": kwargs["input_ids"],
            "attention_mask": kwargs["attention_mask"],
        }
        # apply encoder model
        logits = self.backbone(**inputs).logits
        # extract mask token id
        mask_index = kwargs["mask_token_id"].squeeze(1)
        # apply verbalizer
        label_word_logits = self.verbalizer(logits, mask_index)
        return label_word_logits

    def convert_map_to_id(self):
        # convert tokens to vocabulary ids
        # and assert no special tokens are present
        id_map = {
            key: self.tokenizer.convert_tokens_to_ids(item)
            for key, item in self.verbalizer_map.items()
        }
        label_word_ids = [id for ids in id_map.values() for id in ids]
        assert all(
            x not in self.tokenizer.all_special_ids for x in label_word_ids
        ), "Special Tokens in Verbalizer are not allowed"
        return id_map


class MultiMaskitModel(nn.Module):
    def __init__(self, model_name, verbalizer_map):
        super().__init__()
        self.backbone = AutoModelForMaskedLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.verbalizer_map = verbalizer_map
        self.verbalizer_id_dict = self.convert_map_to_id()
        self.label_word_ids = {task: list(words.values()) for task, words in self.verbalizer_id_dict.items()}
        self.n_classes = len(verbalizer_map)

    def verbalizer(self, full_logits, mask_index, label_word_ids):
        batch_size = full_logits.size(0)
        # extract logits for mask token
        label_word_logits = full_logits[torch.arange(batch_size), mask_index]
        # extract logits on label words
        label_word_logits = label_word_logits[:, label_word_ids]
        # normalize
        label_word_logits = (F.softmax(label_word_logits.reshape(batch_size, -1),dim=-1)
                             .reshape(*label_word_logits.shape)
                             )
        # convert to logits
        label_word_logits = torch.log(label_word_logits+1e-15)
        # aggregate logits
        label_word_logits = label_word_logits.sum(-1)/torch.ones_like(label_word_logits).sum(-1)
        return label_word_logits

    def forward(self, **kwargs):
        inputs = {
            "input_ids": kwargs["input_ids"],
            "attention_mask": kwargs["attention_mask"],
        }
        # apply encoder model
        logits = self.backbone(**inputs).logits
        # extract mask token id
        output = {}
        for task in self.verbalizer_map.keys():
            mask_index = kwargs["mask_token_ids"][task]
            # apply verbalizer
            label_word_ids = self.label_word_ids[task]
            label_word_logits = self.verbalizer(logits, mask_index, label_word_ids)
            output[task] = label_word_logits
        return output

    def convert_map_to_id(self):
        # convert tokens to vocabulary ids
        # and assert no special tokens are present
        id_map = {
            task: {
                key: self.tokenizer.convert_tokens_to_ids(item)
                for key, item in task_map.items()
            }
            for task, task_map in self.verbalizer_map.items()
        }
        label_word_ids = [id for task in id_map.values() for ids in task.values() for id in ids]
        assert all(
            x not in self.tokenizer.all_special_ids for x in label_word_ids
        ), "Special Tokens in Verbalizer are not allowed"
        return id_map


