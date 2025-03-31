# [MASK]-It - Lightweight framework for efficient encoder model fine-tuning

## Install
requires python 3.10 or above
```
pip install maskit
```

## Usage
### Define task
```aiignore
# task and model
classes = ['happy', 'sad']
verbalizer_map = {'happy':['happy', 'fun'],
                  'sad':['sad', 'cry']
model_name = 'google-bert/bert-base-uncased'
```
### Select pre-trained model
```aiignore
from maskit.model import maskitModel
model = maskitModel(model_name=model_name,
                    verbalizer_map=verbalizer_map)
```
### Load dataset
```aiignore
text = ['I am so happy today that I cannot stay still',
        'I am very very sad unfortunately']

labels = [1,0]
template = '{text}. This sentence is: [MASK]'
dataset = maskitDataset(text, labels, model_name, template)
dataloader = DataLoader(dataset=dataset, batch_size=2)
```

### Inference
```aiignore
batch = next(iter(dataloader))
output = model(**batch)
```

