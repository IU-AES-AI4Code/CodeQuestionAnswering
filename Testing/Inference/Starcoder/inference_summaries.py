import argparse
import torch
from transformers import pipeline
from datasets import load_dataset
from tqdm import tqdm
import json
import pandas as pd
import re

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from transformers import set_seed
set_seed(42)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default='StarcoderWithSummariesUCQAQuality')
    parser.add_argument("--checkpoint", type=str, default='bigcode/Starcoder1024Tokens32LoraRankSummaries/starcoder-merged')
    parser.add_argument("--dataset", type=str, default='datapaf/UltimateQASummaries')
    parser.add_argument("--data_dir", type=str, default='main')
    parser.add_argument("--split", type=str, default='test')
    parser.add_argument("--device", type=str, default='cuda:0')

    return parser.parse_args()


if __name__ == "__main__" :

    # parse args
    args = get_args()

    # load the data to make inference on
    dataset = load_dataset(
        args.dataset,
        # data_dir=args.data_dir,
        split=args.split,
        token=True
    )
    prompt_template = "Question: {question}\n\nSummary: {summary}\n\nCode: {code}\n\nAnswer:"
    
    pipe = pipeline(
        "text-generation",
        model=args.checkpoint,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )
    pipe.tokenizer.pad_token_id = 0
    
    # get model predictions
    # periodically saves predictions in a json file
    preds = {} 
    for i in tqdm(range(len(dataset))):
        ex = dataset[i]

        # generate answer
        prompt = prompt_template.format(
            question=ex['question'],
            summary=ex['summary'],
            code=ex['code']
        )
        outputs = pipe(
            prompt,
            return_full_text=False,
            max_new_tokens=128,
            pad_token_id=0,
            eos_token_id=0
        )
        text = outputs[0]['generated_text'].strip()
        preds[i] = {
            'question': ex['question'],
            'summary': summary,
            'code': ex['code'],
            'answer': text,
            'true_answer': ex['answer']
        }
        if i % 10 == 0 or i == len(dataset)-1:
            with open(f"{args.name}.json", "w") as f:
                f.write(json.dumps(preds, indent=4))

    # with open(f"{args.name}.json") as f:
    #     preds = json.load(f)
    
    # save answers in a csv file
    preds_df = pd.DataFrame({'text': [pred['answer'] for pred in preds.values()]})
    preds_df.to_csv(f'{args.name}.csv', sep="\t", header=None)
