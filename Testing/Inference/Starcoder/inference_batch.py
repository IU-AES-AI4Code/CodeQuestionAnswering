import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import json
import pandas as pd
import re
import torch

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default='BatchingTest')
    parser.add_argument("--checkpoint", type=str, default='bigcode/Starcoder1024Tokens32LoraRankGrammarCorrected/starcoder-merged')
    parser.add_argument("--dataset", type=str, default='datapaf/UltimateQAFiltered2')
    parser.add_argument("--data_dir", type=str, default='.')
    parser.add_argument("--split", type=str, default='test')
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--batch_size", type=int, default=4)

    return parser.parse_args()



def compose_batch(samples):
    return [
        f"Question: {samples['question'][i]}\n\nCode: {samples['code'][i]}\n\nAnswer:"
        for i in range(len(samples))
    ]


if __name__ == "__main__" :

    # parse args
    args = get_args()

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'
    
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        device_map=args.device,
        torch_dtype=torch.bfloat16,
    ) 
    
    # load the data to make inference on
    dataset = load_dataset(
        args.dataset,
        split=args.split,
        token=True,
    )

    # get model predictions
    # periodically saves predictions in a json file
    preds = []
    for i in tqdm(range(0, len(dataset) // args.batch_size)):
        batch = compose_batch(dataset[i*args.batch_size:(i+1)*args.batch_size])
        print(len(batch))
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True
        ).to(args.device)
        outputs = model.generate(
            **inputs, max_new_tokens=128, pad_token_id=tokenizer.eos_token_id
        )
        preds += tokenizer.batch_decode(
            outputs[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True
        )
        if True:
        # if i % 10 == 0:
            with open(f"{args.name}.json", "w") as f:
                f.write(json.dumps(preds, indent=4))
        quit()

    with open(f"{args.name}.json", "w") as f:
        f.write(json.dumps(preds, indent=4))

    # with open(f"{args.name}.json", 'r') as f:
    #     preds_json = json.load(f)

    # extract answers from the predictions
    answers = []
    for i, text in enumerate(preds.values()):
        start = text.find("Answer:")
        answers.append(text[start+7:-13].strip())

    # save answers in a csv file
    preds_df = pd.DataFrame({"text": answers})
    preds_df.to_csv(f"{args.name}.csv", sep="\t", header=None)
