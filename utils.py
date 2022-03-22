import json
import gzip
from sklearn.metrics import f1_score, mean_squared_error
import pandas as pd
import datasets

import constants

### Dataset input/output helpers

def dump_jsonl_gz(obj, outpath):
    # obj is list of json
    with gzip.open(outpath, "wt") as fout:
        for o in obj:
            fout.write("%s\n" % json.dumps(o))

def load_jsonl_gz(inpath):
    ret = []
    with gzip.open(inpath, 'rt') as f:
        for line in f:
            ret.append(json.loads(line))
    return ret

def load_yelp_dataset(filepath):
    # collect the raw data
    ret = load_jsonl_gz(filepath)
    # put it into the raw text and score form needed for longformer
    df_contents = []
    for entry in ret:
        df_contents.append({
            "text": str(" ".join(entry["reviews"])),
            "label": float(entry["avg_score"]),
            # 'per_text_label': entry["scores"],
        })
    return datasets.Dataset.from_pandas(pd.DataFrame(df_contents))

### Model training and evaluation helpers

def tokenization(batched_text, tokenizer):
    result = tokenizer(batched_text["text"], padding="max_length", truncation=True, max_length=constants.MAX_SEQ_LENGTH)
    result["label"] = batched_text["label"]
    return result

def compute_mse_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions
    # preds = argmax(pred.predictions, axis=1)
    # preds = pred.predictions.argmax(-1)
    # precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    # acc = accuracy_score(labels, preds)
    return {
        'mse': mean_squared_error(labels, preds),
        # 'accuracy': acc,
        # 'f1': f1,
        # 'precision': precision,
        # 'recall': recall
    }

### Scoring helpers

# Calculate the MSE-full metric, given two matching-index sets of predictions
def compute_mse_full(pred1, pred2):
    diff = pred1 - pred2
    diff_sq = diff**2
    return sum(diff_sq)/len(diff_sq)