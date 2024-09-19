from datasets import load_dataset
import json

dataset = load_dataset("datapaf/UltimateQAFiltered2")

train_data = []

for item in dataset['train']:
    ex = {
        'instruction': f"QUESTION:\n{item['question']}\n\nCODE:\n```python\n{item['code']}\n```\n\nAnswer to QUESTION considering CODE.",
        'output': item['answer']
    }
    train_data.append(ex)

with open('DeepSeekCoder/train_default.json', 'w') as f:
    json.dump(train_data, f)
