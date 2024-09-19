import streamlit as st
import pandas as pd
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
import json
import spacy

# import bert_score
import torch
from bert_score import BERTScorer
from evaluate import load
import gc
from numba import cuda 

# import spacy.cli
# spacy.сli.download("en_core_web_lg")

# nlp = spacy.load("en_core_web_lg", disable=["ner", "parser", "textcat"])

# bleurt = load('bleurt', 'BLEURT-20', module_type="metric")

# bert_scorer = BERTScorer(model_type="microsoft/deberta-xlarge-mnli")

if "metrics" not in st.session_state:
    st.session_state["metrics"] = None


def get_csv(file, name):
    df = pd.read_csv(file, sep="\t", header=None)
    df.drop(0, axis=1, inplace=True)
    df.rename({1: name}, axis=1, inplace=True)
    return df


def get_test_data_df(test_data_file):
    test_data = [json.loads(line) for line in test_data_file.readlines()]
    test_data_df = pd.DataFrame(
        {
            "QUESTION": [
                (
                    " ".join(item["question_tokens"])
                    if "question_tokens" in item
                    else item["question"]
                )
                for item in test_data
            ],
            "CODE": [
                " ".join(item["code_tokens"]) if "code_tokens" in item else item["code"]
                for item in test_data
            ],
            "ANSWER": [
                item["answer"] if "answer" in item else item["answers"]
                for item in test_data
            ],
        }
    )
    return test_data_df


def compose_df(pred_df, test_data_df):
    df = pd.concat([pred_df, test_data_df], axis=1)
    df["PREDICTION"] = df["PREDICTION"].astype(str)
    df["ANSWER"] = df["ANSWER"].astype(str)
    return df


def normalize_text(s):
    import string
    import re

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact_match(prediction, truth):
    # return int(normalize_text(prediction) == normalize_text(truth))
    return int(prediction == truth)


def compute_prec_rec_f1(prediction, truth):
    # pred_tokens = normalize_text(prediction).split()
    pred_tokens = prediction.split()
    # truth_tokens = normalize_text(truth).split()
    truth_tokens = truth.split()

    # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return [int(pred_tokens == truth_tokens)] * 3

    common_tokens = set(pred_tokens) & set(truth_tokens)

    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0, 0, 0

    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)
    f1 = 2 * (prec * rec) / (prec + rec)

    return prec, rec, f1


def compute_spacy_similarity(predictions, truths):
    # from tqdm import tqdm
    nlp = spacy.load("en_core_web_lg", disable=["ner", "parser", "textcat"])
    similarities = []
    for prediction, truth in zip(predictions, truths):
        doc1 = nlp(prediction)
        doc2 = nlp(truth)
        similarities.append(doc2.similarity(doc1))
    return np.array(similarities).mean()


def calculate_metrics(df, bleu_n_max=4):
    answers = df["ANSWER"].apply(normalize_text)
    predictions = df["PREDICTION"].apply(normalize_text)

    bleu = {
        i: corpus_bleu(
            list_of_references=answers.apply(lambda x: [x.split()]).to_list(),
            hypotheses=predictions.apply(lambda x: x.split()).to_list(),
            weights=(1 / i,) * i,
        )
        for i in range(1, bleu_n_max + 1)
    }

    exact_matches = np.where(
        [
            compute_exact_match(pred, answer)
            for pred, answer in zip(predictions, answers)
        ]
    )[0]

    bs_p, bs_r, bs_f1 = bert_score.score(
        df["PREDICTION"].to_list(),
        df["ANSWER"].to_list(),
        verbose=True,
        model_type="microsoft/deberta-xlarge-mnli",
    )
    
    prf1 = [
        compute_prec_rec_f1(pred, answer) for pred, answer in zip(predictions, answers)
    ]

    metrics = {f"BLEU": bleu}
    metrics["word2vec_sim"] = compute_spacy_similarity(predictions, answers)
    metrics["BERTScore"] = {
        "Precision": bs_p.mean().item(),
        "Recall": bs_r.mean().item(),
        "F1": bs_f1.mean().item(),
    }
    metrics["P"] = np.mean([item[0] for item in prf1])
    metrics["R"] = np.mean([item[1] for item in prf1])
    metrics["F1"] = np.mean([item[2] for item in prf1])
    metrics["EM"] = len(exact_matches) / len(df)

    return metrics


def calculate_metrics_with_std(df, bleu_n_max=4):
    answers = df["ANSWER"].apply(normalize_text)
    predictions = df["PREDICTION"].apply(normalize_text)

    # BLEU Scores
    bleu_answers = answers.apply(lambda x: [x.split()]).to_list()
    bleu_preds = predictions.apply(lambda x: x.split()).to_list()
    bleu_corpus = {
        i: corpus_bleu(bleu_answers, bleu_preds, weights=(1 / i,) * i)
        for i in range(1, bleu_n_max + 1)
    }

    bleu_sentence = [
        [
            sentence_bleu(a, p, weights=(1 / i,) * i)
            for a, p in zip(bleu_answers, bleu_preds)
        ]
        for i in range(1, bleu_n_max + 1)
    ]
    bleu_sentence = {
        str(i + 1): f"{np.mean(scores):.6f} ± {np.std(scores):.6f}"
        for i, scores in enumerate(bleu_sentence)
    }
    metrics = {f"BLEU (corpus)": bleu_corpus, f"BLEU (sentence)": bleu_sentence}

    # BERTScore
    bert_scorer = BERTScorer(model_type="microsoft/deberta-xlarge-mnli")    
    bs_p, bs_r, bs_f1 = bert_scorer.score(
        df["PREDICTION"].to_list(),
        df["ANSWER"].to_list(),
        verbose=True,
        # model_type="microsoft/deberta-xlarge-mnli",
    )
    
    metrics["BERTScore"] = {
        "Precision": f"{bs_p.mean().item():.6f} ± {bs_p.std().item():.6f}",
        "Recall": f"{bs_r.mean().item():.6f} ± {bs_r.std().item():.6f}",
        "F1": f"{bs_f1.mean().item():.6f} ± {bs_f1.std().item():.6f}",
    }

    # del bert_scorer
    # gc.collect()
    # torch.cuda.empty_cache()

    # BLEURT
    bleurt = load('bleurt', 'BLEURT-20', module_type="metric")
    bleurt_res = bleurt.compute(
        predictions=list(df["PREDICTION"]),
        references=list(df["ANSWER"]),
        # progress_bar=True
    )

    metrics["BLEURT"] = f"{np.mean(bleurt_res['scores']):.6f} ± {np.std(bleurt_res['scores']):.6f}"

    # del bleurt
    # gc.collect()
    # torch.cuda.empty_cache()
    
    prf1 = [
        compute_prec_rec_f1(pred, answer) for pred, answer in zip(predictions, answers)
    ]

    metrics["P"] = (
        f"{np.mean([item[0] for item in prf1]):.6f} ± {np.std([item[0] for item in prf1]):.6f}"
    )
    metrics["R"] = (
        f"{np.mean([item[1] for item in prf1]):.6f} ± {np.std([item[1] for item in prf1]):.6f}"
    )
    metrics["F1"] = (
        f"{np.mean([item[2] for item in prf1]):.6f} ± {np.std([item[2] for item in prf1]):.6f}"
    )
    metrics["word2vec_sim"] = compute_spacy_similarity(predictions, answers)

    exact_matches = np.where(
        [
            compute_exact_match(pred, answer)
            for pred, answer in zip(predictions, answers)
        ]
    )[0]
    metrics["EM"] = len(exact_matches) / len(df)

    return metrics


def download_json(metrics, file_name="metrics.json"):
    metrics_json = json.dumps(metrics, indent=4)
    st.download_button(
        label="Download metrics",
        data=metrics_json,
        file_name=file_name,
        mime="application/json",
    )


pred_file = st.file_uploader(label="Upload Predictions", type=["csv"])

test_data_file = st.file_uploader(label="Upload Testing Dataset", type=["jsonl"])

if (
    pred_file is not None
    and test_data_file is not None
    and st.session_state["metrics"] is None
):
    pred_df = get_csv(pred_file, "PREDICTION")
    test_data_df = get_test_data_df(test_data_file)

    df = compose_df(pred_df, test_data_df)

    # show targets and predictions
    st.write(df[["QUESTION", "PREDICTION", "ANSWER", "CODE"]])

    # show metrics
    processing_text = st.text("Calculating metrics...")
    st.session_state["metrics"] = calculate_metrics_with_std(
        df
    )  # Before was: calculate_metrics(df)
    download_json(
        st.session_state["metrics"], pred_file.name.split(".")[0] + "_metrics.json"
    )  # Button added
    processing_text.empty()
    st.write(st.session_state["metrics"])
else:
    st.write(st.session_state["metrics"])
