import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoConfig
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import os
import pickle


# -------------- 🔄 Shared Preprocessing Function --------------
def preprocess_single_task_sample(args):
    text, label, model_name, template, max_length, truncation = args
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    mask_token = tokenizer.mask_token
    mask_token_id = tokenizer.mask_token_id

    prefix_str, suffix_str = template.split("{text}")
    prefix_ids = tokenizer(prefix_str, add_special_tokens=True)["input_ids"]
    suffix_ids = tokenizer(suffix_str, add_special_tokens=False)["input_ids"]
    text_ids = tokenizer(text, add_special_tokens=False)["input_ids"]

    total_len = len(prefix_ids) + len(text_ids) + len(suffix_ids)
    if total_len > max_length:
        max_text_len = max_length - len(prefix_ids) - len(suffix_ids)
        if truncation == "tail":
            text_ids = text_ids[:max_text_len]
        else:
            text_ids = text_ids[-max_text_len:]

    input_ids = prefix_ids + text_ids + suffix_ids
    attention_mask = [1] * len(input_ids)
    padding_length = max_length - len(input_ids)
    input_ids += [tokenizer.pad_token_id] * padding_length
    attention_mask += [0] * padding_length

    input_ids_tensor = torch.tensor(input_ids)
    attention_mask_tensor = torch.tensor(attention_mask)
    mask_token_indices = (input_ids_tensor == mask_token_id).nonzero(as_tuple=True)[0]

    return {
        "input_ids": input_ids_tensor,
        "attention_mask": attention_mask_tensor,
        "labels": torch.tensor(label),
        "mask_token_id": mask_token_indices,
    }


def preprocess_multi_task_sample(args):
    text, label, model_name, template, task_words, max_length, truncation = args
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    mask_token_id = tokenizer.mask_token_id

    prefix_str, suffix_str = template.split("{text}")
    prefix_ids = tokenizer(prefix_str, add_special_tokens=False)["input_ids"]
    suffix_ids = tokenizer(suffix_str, add_special_tokens=False)["input_ids"]
    text_ids = tokenizer(text, add_special_tokens=False)["input_ids"]

    available = max_length - 2
    total_len = len(prefix_ids) + len(text_ids) + len(suffix_ids)

    if total_len > available:
        excess = total_len - available
        if truncation == "tail":
            text_ids = text_ids[:-excess] if len(text_ids) > excess else []
        else:
            text_ids = text_ids[excess:] if len(text_ids) > excess else []
        total_len = len(prefix_ids) + len(text_ids) + len(suffix_ids)
        if total_len > available:
            excess = total_len - available
            if len(suffix_ids) > excess:
                suffix_ids = suffix_ids[:-excess]
            else:
                excess -= len(suffix_ids)
                suffix_ids = []
                prefix_ids = prefix_ids[:-excess] if len(prefix_ids) > excess else []

    input_ids = [tokenizer.cls_token_id] + prefix_ids + text_ids + suffix_ids + [tokenizer.sep_token_id]
    attention_mask = [1] * len(input_ids)
    pad_len = max_length - len(input_ids)
    input_ids += [tokenizer.pad_token_id] * pad_len
    attention_mask += [0] * pad_len

    input_ids_tensor = torch.tensor(input_ids)
    attention_mask_tensor = torch.tensor(attention_mask)

    task_mask_indices = {}
    for task_name, keyword in task_words.items():
        keyword_ids = tokenizer.encode(keyword, add_special_tokens=False)
        keyword_len = len(keyword_ids)
        found = False
        for i in range(len(input_ids_tensor) - keyword_len):
            if input_ids_tensor[i:i + keyword_len].tolist() == keyword_ids:
                for j in range(i + keyword_len, len(input_ids_tensor)):
                    if input_ids_tensor[j] == mask_token_id:
                        task_mask_indices[task_name] = torch.tensor(j)
                        found = True
                        break
            if found:
                break
        if not found:
            raise ValueError(f"[MASK] not found for task '{task_name}' using keyword '{keyword}'")

    return {
        "input_ids": input_ids_tensor,
        "attention_mask": attention_mask_tensor,
        "labels": {k: torch.tensor(label[k]) for k in label},
        "mask_token_ids": task_mask_indices,
    }


# -------------- 📦 Cached Dataset Wrappers --------------
class MaskitDataset(Dataset):
    def __init__(self, texts, labels, model_name, template, max_length,
                 cache_path=None, truncation="tail", use_parallel=True):
        self.cache_path = cache_path

        if cache_path and os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                self.data = pickle.load(f)
        else:
            args = [(texts[i], labels[i], model_name, template, max_length, truncation)
                    for i in range(len(texts))]
            if use_parallel:
                with Pool(cpu_count()) as pool:
                    self.data = list(tqdm(pool.imap(preprocess_single_task_sample, args), total=len(texts)))
            else:
                self.data = [preprocess_single_task_sample(arg) for arg in tqdm(args)]

            if cache_path:
                with open(cache_path, "wb") as f:
                    pickle.dump(self.data, f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class MultiMaskitDataset(Dataset):
    def __init__(self, texts, labels, model_name, template, task_words, max_length,
                 cache_path=None, truncation="tail", use_parallel=True):
        self.cache_path = cache_path

        if cache_path and os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                self.data = pickle.load(f)
        else:
            args = [
                (texts[i], {k: labels[k][i] for k in labels}, model_name,
                 template, task_words, max_length, truncation)
                for i in range(len(texts))
            ]
            if use_parallel:
                with Pool(cpu_count()) as pool:
                    self.data = list(tqdm(pool.imap(preprocess_multi_task_sample, args), total=len(texts)))
            else:
                self.data = [preprocess_multi_task_sample(arg) for arg in tqdm(args)]

            if cache_path:
                with open(cache_path, "wb") as f:
                    pickle.dump(self.data, f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
