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

class MultiMaskitDataset(Dataset):
    def __init__(self, texts, labels, model_name, template, task_words, truncation="tail"):
        self.texts = texts
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = AutoConfig.from_pretrained(model_name).max_position_embeddings
        self.template = template
        self.truncation = truncation

        self.mask_token = self.tokenizer.mask_token
        self.mask_token_id = self.tokenizer.mask_token_id
        self.task_words = task_words

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = {key:torch.tensor(value[idx]) for key, value in self.labels.items()}

        if "{text}" not in self.template:
            raise ValueError("Template must contain the '{text}' placeholder.")

        # Split the template around {text}
        prefix_str, suffix_str = self.template.split("{text}")

        # Tokenize components
        prefix_ids = self.tokenizer(prefix_str, add_special_tokens=False)["input_ids"]
        suffix_ids = self.tokenizer(suffix_str, add_special_tokens=False)["input_ids"]
        text_ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]

        # Compute available space for all parts
        total_len = len(prefix_ids) + len(text_ids) + len(suffix_ids)
        available = self.max_length - 2  # reserve for [CLS] and [SEP]

        if total_len > available:
            # Prioritize keeping as much text as possible
            excess = total_len - available

            if self.truncation == "tail":
                text_ids = text_ids[:-excess] if len(text_ids) > excess else []
            else:
                text_ids = text_ids[excess:] if len(text_ids) > excess else []

            total_len = len(prefix_ids) + len(text_ids) + len(suffix_ids)
            # Still too long? Now start cutting suffix, then prefix
            if total_len > available:
                excess = total_len - available
                if len(suffix_ids) > excess:
                    suffix_ids = suffix_ids[:-excess]
                else:
                    excess -= len(suffix_ids)
                    suffix_ids = []
                    prefix_ids = prefix_ids[:-excess] if len(prefix_ids) > excess else []

        # Assemble input sequence
        input_ids = [self.tokenizer.cls_token_id] + prefix_ids + text_ids + suffix_ids + [self.tokenizer.sep_token_id]
        attention_mask = [1] * len(input_ids)

        # Pad to max_length
        pad_len = self.max_length - len(input_ids)
        if pad_len > 0:
            input_ids += [self.tokenizer.pad_token_id] * pad_len
            attention_mask += [0] * pad_len

        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)

        # Find [MASK] positions per task
        task_mask_indices = {}
        for task_name, keyword in self.task_words.items():
            keyword_ids = self.tokenizer.encode(keyword, add_special_tokens=False, max_length=20, truncation=True)
            keyword_len = len(keyword_ids)
            found = False

            for i in range(len(input_ids) - keyword_len):
                if input_ids[i:i + keyword_len].tolist() == keyword_ids:
                    # Find the nearest [MASK] after keyword
                    for j in range(i + keyword_len, len(input_ids)):
                        if input_ids[j] == self.mask_token_id:
                            task_mask_indices[task_name] = torch.tensor(j)
                            found = True
                            break
                if found:
                    break

            if not found:
                raise ValueError(f"[MASK] not found for task '{task_name}' using keyword '{keyword}'.")

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label,
            "mask_token_ids": task_mask_indices,
        }