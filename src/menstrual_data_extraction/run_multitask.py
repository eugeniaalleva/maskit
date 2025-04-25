import os
import argparse
import torch
import json
from typing import *
from transformers import AutoConfig
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from maskit.dataset import MultiMaskitDataset
from maskit.model import MultiMaskitModel
from maskit.loss import ManualWeightedLoss
from maskit.utils import move_to_device
from utils.data_processor import MenstrualDataProcessor
from sklearn.metrics import classification_report
from yacs.config import CfgNode

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--config_file', 
        '-c',
        type=str, 
        default='config.yaml', 
        help='(str) Path to the config file',
        required=True
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = get_args()
    config = CfgNode(new_allowed=True)
    config.merge_from_file(args.config_file)

    model_name = config.plm.model
    max_length = AutoConfig.from_pretrained(model_name).max_position_embeddings

    processor = MenstrualDataProcessor()
    train_data_path = f'{config.dataset.path}{config.dataset.train_file}'
    dev_data_path = f'{config.dataset.path}{config.dataset.dev_file}'

    tasks = config.tasks
    verbalizer_map = {}
    template = '{text}. '
    task_words = {}
    for task in tasks.keys():
        task_config = tasks[task]
        with open(task_config.prompt_verbalizer.file_path, 'r') as file:
            task_verbalizer = json.load(file)
        verbalizer_map[task] = task_verbalizer
        template = template + f'{task_config.prompt_template} [MASK]. '
        task_words[task] = f'{task_config.prompt_template}'
    template = template.strip()

    train_texts, train_labels = processor.get_multimask_examples(train_data_path, tasks, config.dataset.note_text_column)
    dev_texts, dev_lables = processor.get_multimask_examples(dev_data_path, tasks, config.dataset.note_text_column)

    train_dataset = MultiMaskitDataset(texts=train_texts, 
                            labels=train_labels, 
                            model_name=model_name, 
                            template=template, 
                            task_words=task_words,
                            max_length=max_length)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=config.train.batch_size)

    dev_dataset = MultiMaskitDataset(texts=dev_texts, 
                            labels=dev_lables, 
                            model_name=model_name, 
                            template=template, 
                            task_words=task_words,
                            max_length=max_length)
    dev_dataloader = DataLoader(dataset=dev_dataset, batch_size=config.train.batch_size)


    model = MultiMaskitModel(model_name=model_name, verbalizer_map=verbalizer_map)
    model.to(DEVICE)
    print(f'Model on: {DEVICE}')
    # task weights
    weights = [0.5 for i in range(len(tasks))]
    loss_wrapper = ManualWeightedLoss(weights=weights)
    loss_wrapper.to(DEVICE)

    # Train
    loss_fun = CrossEntropyLoss()
    optimizer = AdamW(list(model.parameters())+list(loss_wrapper.parameters()), lr=config.plm.optimize.lr)

    model.train()
    print(f"Fixed task weights: {weights}")
    epochs = config.train.num_epochs
    for epoch in range(epochs):
        epoch_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            batch = {key:move_to_device(value,DEVICE) for key, value in batch.items()}
            optimizer.zero_grad()
            logits = model(**batch)
            labels = batch['labels']
            task_losses = [loss_fun(logits[task], labels[task]) for task in verbalizer_map.keys()]
            total_loss = loss_wrapper(task_losses)
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()
        print(f"Epoch {epoch + 1}: loss = {epoch_loss:.4f}")

    # Inference
    model.eval()
    all_preds = {key: [] for key in verbalizer_map.keys()}
    all_labels = {key: [] for key in verbalizer_map.keys()}

    for batch in dev_dataloader:
        batch = {key: move_to_device(value, DEVICE) for key, value in batch.items()}
        logits = model(**batch)
        for task in verbalizer_map.keys():
            all_preds[task].extend(logits[task].argmax(dim=1).tolist())
            all_labels[task].extend(batch['labels'][task].tolist())

    # print classification reports once per task
    for task in verbalizer_map.keys():
        label_names = list(verbalizer_map[task].keys())
        label_ids = list(range(len(label_names)))
        
        print(f"\nClassification Report for task: {task}")
        print(classification_report(
            y_true=all_labels[task],
            y_pred=all_preds[task],
            labels=label_ids,
            target_names=label_names,
            digits=3,
            zero_division=0
        ))