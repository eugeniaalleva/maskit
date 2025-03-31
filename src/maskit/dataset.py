import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoConfig


class maskitDataset(Dataset):
    def __init__(self, texts, labels, model_name, template, truncation="tail"):
        self.texts = texts
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = AutoConfig.from_pretrained(
            model_name).max_position_embeddings
        self.template = template
        self.mask_token = self.tokenizer.mask_token
        self.mask_token_id = self.tokenizer.mask_token_id
        self.truncation = truncation

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        if self.mask_token not in self.template:
            raise ValueError(
                "Template must contain the mask token placeholder."
            )

        # Split template into prefix and suffix around the mask token
        template_parts = self.template.split("{text}")
        if len(template_parts) != 2:
            raise ValueError(
                "Template must contain a single '{text}' placeholder."
            )

        prefix = template_parts[0]
        suffix = template_parts[1]

        # Tokenize prefix and suffix with special tokens
        prefix_ids = self.tokenizer(
            prefix, add_special_tokens=True, return_attention_mask=False
        )["input_ids"]
        suffix_ids = self.tokenizer(
            suffix, add_special_tokens=False, return_attention_mask=False
        )["input_ids"]

        # Tokenize the text to be inserted
        text_ids = self.tokenizer(
            text, add_special_tokens=False, return_attention_mask=False
        )["input_ids"]

        # Truncate text_ids from the front if needed
        total_len = len(prefix_ids) + len(text_ids) + len(suffix_ids)
        if total_len > self.max_length:
            max_text_len = self.max_length - len(prefix_ids) - len(suffix_ids)
            if self.truncation == "tail":
                text_ids = text_ids[:max_text_len]
            else:
                text_ids = text_ids[-max_text_len:]  # truncate from front

        # Concatenate all parts
        input_ids = prefix_ids + text_ids + suffix_ids

        #  Pad if needed
        attention_mask = [1] * len(input_ids)
        padding_length = self.max_length - len(input_ids)
        if padding_length > 0:
            input_ids += [self.tokenizer.pad_token_id] * padding_length
            attention_mask += [0] * padding_length

        #  Convert to tensors
        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)

        #  Find mask token index
        mask_token_indices = (input_ids == self.mask_token_id).nonzero(
            as_tuple=True)[0]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label),
            "mask_token_id": mask_token_indices,
        }
