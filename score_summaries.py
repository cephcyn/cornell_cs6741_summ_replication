import argparse
import os
import logging
import pickle

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import torch
import pandas as pd
import datasets

import utils
import constants

summary_paths = {
    "random": f"t{constants.SUMMARY_TOKEN_LIMIT}_random.jsonl.gz",
    "presumm": f"t{constants.SUMMARY_TOKEN_LIMIT}_presumm.jsonl.gz",
    "decsum011": f"t{constants.SUMMARY_TOKEN_LIMIT}_decsum011_beam{constants.DECSUM_BEAM_WIDTH}.jsonl.gz",
    "decsum010": f"t{constants.SUMMARY_TOKEN_LIMIT}_decsum010_beam{constants.DECSUM_BEAM_WIDTH}.jsonl.gz",
    "decsum001": f"t{constants.SUMMARY_TOKEN_LIMIT}_decsum001_beam{constants.DECSUM_BEAM_WIDTH}.jsonl.gz",
}

def pred_with_cache(src_path, out_path):
    preds = None
    if os.path.exists(out_path):
        logging.info(f"Predictions already exist in {out_path}")
        # load cache
        with open(out_path, "rb") as f:
            preds = pickle.load(f)
    else:
        logging.info(f"Predictions do not exist; using {src_path} to generate {out_path}")
        # create pred dataset, run predictions, export to cache
        dataset = utils.load_yelp_dataset(src_path)
        dataset = dataset.map(
            lambda bt: utils.tokenization(bt, tokenizer), batched=True, batch_size=len(dataset)
        )
        dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
        preds = trainer.predict(test_dataset=dataset)
        with open(out_path, "wb") as f:
            pickle.dump(preds, f)
    return preds

def pred_baseline(trainer, tokenizer):
    baseline_summ_path = os.path.join(
        constants.OUTPUT_DIR, 
        f"{constants.NUM_REVIEWS}reviews", 
        "test.jsonl.gz",
    )
    baseline_pred_path = os.path.join(
        constants.SCORING_DIR, 
        "preds_baseline.pkl",
    )
    return pred_with_cache(baseline_summ_path, baseline_pred_path)

def pred_summary(summ_type, trainer, tokenizer):
    if summ_type not in summary_paths:
        raise ValueError("invalid summary type?")
        
    summary_summ_path = os.path.join(
        constants.SUMMARY_DIR, 
        f"{constants.NUM_REVIEWS}reviews", 
        summary_paths[summ_type],
    )
    summary_pred_path = os.path.join(
        constants.SCORING_DIR, 
        f"preds_{summ_type}.pkl",
    )
    return pred_with_cache(summary_summ_path, summary_pred_path)

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    logging.info('Admin logged in')
    
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--summary_type",
        type=str,
        choices=['random', 'presumm', 'decsum011', 'decsum010', 'decsum001'],
        help="Summary type to create based on test dataset")
    
    args = parser.parse_args() 
    logging.info(f"args: {args}")
    
    logging.info(f"Importing finetuned models and relevant tokenizers")
    tokenizer = AutoTokenizer.from_pretrained(
        constants.LONGFORMER_MODEL_TYPE,
        max_length=constants.MAX_SEQ_LENGTH,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        os.path.join(
            constants.CHECKPOINT_DIR,
            constants.LONGFORMER_SAVE
        ), 
        num_labels=1,
    )
    training_args = TrainingArguments(
        output_dir = constants.LOGGING_DIR,
        dataloader_num_workers = constants.FINETUNE_NUM_WORKERS,
        per_device_eval_batch_size = constants.FINETUNE_EVAL_BATCH_SIZE,
    )
    # instantiate the trainer class and check for available devices
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=utils.compute_mse_metrics,
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"will be running model; using device={device}")
    # load in baseline predictions
    logging.info(f"generating/retrieving baseline predictions")
    baseline_preds = pred_baseline(trainer, tokenizer)
    # load in summary predictions
    logging.info(f"generating/retrieving summary={args.summary_type} predictions")
    summary_preds = pred_summary(args.summary_type, trainer, tokenizer)
    # calculate results
    logging.info(f"CALCULATING RESULTS")
    logging.info(f"{args.summary_type} :")
    logging.info(f"  MSE-Full: "+str(utils.compute_mse_full(baseline_preds[0], summary_preds[0])))
    logging.info(f"       MSE: "+str(summary_preds[2]["test_mse"]))
