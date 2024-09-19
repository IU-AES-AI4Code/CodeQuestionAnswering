from datasets import load_dataset
import json

dataset = load_dataset(
    'datapaf/UltimateQASummaries',
    split="train"
)

train = []

for item in dataset:
    train.append({
        'instruction': f"QUESTION:\n{item['question']}\n\nCODE SUMMARY:\n{item['summary']}\n\nCODE:\n{item['code']}\n\nAnswer to QUESTION considering CODE and CODE SUMMARY.",
        'output': item['answer']
    })

with open('train_summaries.json', "w") as f:
    json.dump(train, f)