import json
import gzip
from sklearn.metrics import f1_score, mean_squared_error
from scipy.stats import wasserstein_distance
import numpy as np
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

# Calculate MSE, given two matching-index sets of predictions
def compute_mse(pred1, pred2):
    return mean_squared_error(pred1, pred2)
    # diff = pred1 - pred2
    # diff_sq = diff**2
    # return sum(diff_sq)/len(diff_sq)

# Calculate log(Wasserstein distance), given two lists of predictions (can be varying length)
def compute_wasserstein(pred1, pred2):
    return np.log(wasserstein_distance(pred1, pred2))

# Calculate mean error, given two matching-index sets of predictions
def compute_merror(predset, predbase):
    diff = [predset[i]-predbase[i] for i in range(len(predset))]
    return (sum(diff)/len(diff))[0]

# Calculate the percentage of predictions that are above the base value, given two matching-index sets of predictions
def compute_frachigher(predset, predbase):
    num_higher = [predset[i]-predbase[i] for i in range(len(predset))]
    num_higher = [(1 if e>0 else 0) for e in num_higher]
    return sum(num_higher)/len(num_higher)