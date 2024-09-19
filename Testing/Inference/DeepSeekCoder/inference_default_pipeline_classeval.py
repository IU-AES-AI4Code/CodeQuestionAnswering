import argparse
import torch
from transformers import pipeline
from datasets import load_dataset
from tqdm import tqdm
import json
import pandas as pd
import re

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default='DeepSeekCoderDefaultClassEval')
    parser.add_argument("--checkpoint", type=str, default='./DeepSeekDefaultNew/checkpoint-60000')
    parser.add_argument("--dataset", type=str, default='datapaf/ClassEvalQABenchmark')
    parser.add_argument("--data_dir", type=str, default='main')
    parser.add_argument("--split", type=str, default='test')
    parser.add_argument("--device", type=str, default='cuda:2')

    return parser.parse_args()

if __name__ == "__main__" :

    # parse args
    args = get_args()

    dataset = load_dataset(
        args.dataset,
        split=args.split,
        token=True
    )
    
    # load the model
    pipe = pipeline(
        "text-generation",
        model=args.checkpoint,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )
    # pipe.tokenizer.pad_token_id = 0
    pipe.tokenizer.padding_side = 'left'
    
    preds = {}
    for i in tqdm(range(len(dataset))):

        ex = dataset[i]
        question = ex['question']
        code = ex['method_code']
        true_answer = ex['answer']

        prompt_template = "<｜begin▁of▁sentence｜>You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.\n### Instruction:\nQUESTION:\n{question}\n\nCODE:\n```python\n{code}\n```\n\nAnswer to QUESTION considering CODE.\n### Response:\n"
        
        prompt = prompt_template.format(code=code, question=question)
        outputs = pipe(prompt, return_full_text=False, max_new_tokens=128)
        text = outputs[0]['generated_text']
        text = text.strip()
        preds[i] = {
            'question': question,
            'code': code,
            'answer': text, 
            'true_answer': true_answer
        }

        if i % 10 == 0:
            with open(f"{args.name}.json", "w") as f:
                f.write(json.dumps(preds, indent=4))

    with open(f"{args.name}.json", "w") as f:
        f.write(json.dumps(preds, indent=4))

    # save answers in a csv file
    preds_df = pd.DataFrame({"text": [item['answer'] for item in preds.values()]})
    preds_df.to_csv(f'{args.name}.csv', sep="\t", header=None)
