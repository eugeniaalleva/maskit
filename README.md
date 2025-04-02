# [MASK]It -Efficient pre-trained encoder adaption leveraging the [MASK]

`[MASK]It` library allows to adapt pre-trained encoder transformer models for text classification 
leveraging the pre-training fill-mask objective. 
It supports multi-tasking via  `Multi[MASK]It` extension.

## Install
requires python 3.10 or above
```
pip install maskit-learn
```
Disable tokenizer parallelism to allow maskit parallelized dataset preprocessing.
```python
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
```
Set device
```python
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```
## Single Task Usage
### Define task
```python
# task and model
from transformers import AutoConfig
classes = ['happy', 'sad']
verbalizer_map = {'happy':['happy', 'fun'],
                  'sad':['sad', 'cry']}
model_name = 'google-bert/bert-base-uncased'
max_length = AutoConfig.from_pretrained(model_name).max_position_embeddings
```
### Load dataset
```python
from torch.utils.data import DataLoader
from maskit.dataset import MaskitDataset
texts = ['I am so happy today that I cannot stay still','I am very very sad unfortunately']
labels = [0,1]
template = '{text}. This sentence is: [MASK]'
dataset = MaskitDataset(texts=texts, 
                        labels=labels, 
                        model_name=model_name, 
                        template=template, 
                        max_length=max_length)
dataloader = DataLoader(dataset=dataset, batch_size=2)
```

### Load pre-trained model
```python
from maskit.model import MaskitModel
model = MaskitModel(model_name=model_name,
                    verbalizer_map=verbalizer_map)
model.to(DEVICE)
print(f'Model on: {DEVICE}')
```

### Train
Loss function and optimizer definition
```python
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW

loss_fun = CrossEntropyLoss()
optimizer = AdamW(model.named_parameters(), 1e-5)
```
Training loop
```python
model.train()
epochs = 2 
for epoch in range(epochs):
    epoch_loss = 0
    for batch in dataloader:
        # Prepare batch
        batch = {key: val.to(DEVICE) for key, val in batch.items()}
        # Forward pass
        logits = model(**batch)
        labels = batch['labels']
        loss = loss_fun(logits, labels)
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # Loss tracking
        epoch_loss += loss.item()
    print(f'epoch loss: {epoch_loss}')
```

### Inference
```python
model.eval()
predictions = []
labels = []
for batch in dataloader:
    # Prepare batch
    batch = {key: val.to(DEVICE) for key, val in batch.items()}
    # Forward pass
    logits = model(**batch)
    predictions.extend(logits.argmax(dim=1).tolist())
    labels.extend(batch['labels'].tolist())

for idx, text in enumerate(texts):
    print(f'Current sentence: {text}')
    print(f'Ground Truth: {classes[labels[idx]]}')
    print(f'Prediction: {classes[predictions[idx]]}')
    print('-'*50)
```

## Multi-Task Usage
### Define task
```python
# task and model
from transformers import AutoConfig
verbalizer_map = {'sentiment': {'happy':['happy', 'fun'],'sad':['sad', 'cry']},
                    'type': {'news':['news', 'journal'], 'fiction':['fiction', 'novel']}
                    }
model_name = 'google-bert/bert-base-uncased'
max_length = AutoConfig.from_pretrained(model_name).max_position_embeddings
```
### Load dataset
```python
from torch.utils.data import DataLoader
from maskit.dataset import MultiMaskitDataset
texts = [
    'Our colleagues from the war zone report intensification of fights',
    'As I looked in his eyes, I fell in love',
    "Today's weather is going to be sunny and warm",
    'His stomach hurt so much that he had to leave her alone'
    ]
labels = {
    'sentiment': [1,0,0,1],
    'type': [0,1,0,1]
}
template = '{text}. This sentence is: [MASK]. The text type is: [MASK]'
task_words = {
    'sentiment': 'This sentence is:',
    'type': 'The text type is:'
}
dataset = MultiMaskitDataset(texts=texts, 
                        labels=labels, 
                        model_name=model_name, 
                        template=template, 
                        task_words=task_words,
                        max_length=max_length)
dataloader = DataLoader(dataset=dataset, batch_size=2)
```

### Load pre-trained model
```python
from maskit.model import MultiMaskitModel
from maskit.loss import ManualWeightedLoss
model = MultiMaskitModel(model_name=model_name,
                        verbalizer_map=verbalizer_map)
model.to(DEVICE)
print(f'Model on: {DEVICE}')
# task weights
weights = [0.5, 0.5]
loss_wrapper = loss_wrapper = ManualWeightedLoss(weights=weights)
loss_wrapper.to(DEVICE)
```

### Train
Loss function and optimizer definition
```aiignore
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW

loss_fun = CrossEntropyLoss()
optimizer = AdamW(list(model.named_parameters())+list(loss_wrapper.parameters()), 1e-5)
```
Training loop
```aiignore
from maskit.utils import move_to_device
model.train()
print(f"Fixed task weights: {weights}")
epochs = 2
for epoch in range(epochs):
    epoch_loss = 0.0
    for step, batch in enumerate(dataloader):
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
```

### Inference
```aiignore
model.eval()
all_labels = all_preds = {key:[] for key in verbalizer_map.keys()}
for batch in dataloader:
    # Prepare batch
    batch = {key:move_to_device(value,DEVICE) for key, value in batch.items()}
    # Forward pass
    logits = model(**batch)
    for task in verbalizer_map.keys():
        all_preds[task].extend(logits[task].argmax(dim=1).tolist())
        all_labels[task].extend(batch['labels'][task].tolist())

for idx, text in enumerate(texts):
    print(f'Current sentence: {text}')
    for task in verbalizer_map.keys():
        print(f'Task {task}')
        print(f'Ground Truth: {list(verbalizer_map[task].keys())[all_labels[task][idx]]}')
        print(f'Prediction: {list(verbalizer_map[task].keys())[all_preds[task][idx]]}')
    print('-'*50)
```

