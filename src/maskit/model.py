from torch import nn
import torch.nn.functional as F
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


class maskitModel(nn.Module):
    def __init__(self, model_name, verbalizer_map):
        super().__init__()
        self.backbone = AutoModelForMaskedLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.verbalizer_map = verbalizer_map
        self.verbalizer_id_dict = self.convert_map_to_id()
        self.label_word_ids = list(self.verbalizer_id_dict.values())
        self.n_classes = len(verbalizer_map)

    def forward(self, **kwargs):
        inputs = {
            "input_ids": kwargs["input_ids"],
            "attention_mask": kwargs["attention_mask"],
        }
        logits = self.backbone(**inputs)["logits"]
        batch_size = logits.size(0)
        # extract mask token id
        mask_index = kwargs["mask_token_id"].squeeze(1)
        # extract logits on label words for mask token
        logits = logits[torch.arange(batch_size), mask_index]
        logits = logits[:, self.label_word_ids]
        # normalize logits
        logits = F.softmax(logits.reshape(batch_size, -1), dim=-1).reshape(
            *logits.shape
        )
        # convert to logits
        logits = torch.log(logits+1e-15)
        # aggregate logits
        logits = logits.sum(-1)/torch.ones_like(logits).sum(-1)
        return logits

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
