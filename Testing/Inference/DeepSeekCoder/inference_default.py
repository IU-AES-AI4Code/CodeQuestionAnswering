import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import json
import pandas as pd
import re

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default='DeepSeekCoderDefault')
    parser.add_argument("--checkpoint", type=str, default='./DeepSeekDefault/checkpoint-30000')
    parser.add_argument("--dataset", type=str, default='datapaf/UltimateQAFiltered2')
    parser.add_argument("--data_dir", type=str, default='.')
    parser.add_argument("--split", type=str, default='test')
    parser.add_argument("--device", type=str, default='cuda:2')

    return parser.parse_args()
    

if __name__ == "__main__" :

    # parse args
    args = get_args()

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        device_map=args.device
    ) 
    
    # load the data to make inference on
    dataset = load_dataset(
        args.dataset,
        # data_dir=args.data_dir,
        split=args.split,
        use_auth_token=True,
    )

    # get model predictions
    # periodically saves predictions in a json file
    preds = {} 
    for i in tqdm(range(len(dataset))):
        try:
            ex = dataset[i]
            q = f"{ex['question']}\n{ex['code']}"
            inputs = tokenizer.encode(q, return_tensors="pt").to(args.device)
            outputs = model.generate(inputs, max_new_tokens=512, pad_token_id=tokenizer.eos_token_id)
            text = tokenizer.decode(outputs[0])
            preds[i] = text
            print(text)
            quit()
        except Exception as e:
            print(e)
            preds[i] = None
        if i % 10 == 0:
            with open(f"{args.name}.json", "w") as f:
                f.write(json.dumps(preds, indent=4))

    with open(f"{args.name}.json", "w") as f:
        f.write(json.dumps(preds, indent=4))

    # with open(f"{args.name}.json", 'r') as f:
    #     preds_json = json.load(f)

    # # extract answers from the predictions
    # answers = []
    # for i, text in enumerate(preds.values()):
    #     start = text.find("Answer:")
    #     answers.append(text[start+7:-13].strip())

    # # save answers in a csv file
    # preds_df = pd.DataFrame({"text": answers})
    # preds_df.to_csv(f"{args.name}.csv", sep="\t", header=None)
