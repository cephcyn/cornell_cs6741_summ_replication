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

summary_type_translations = {
    "random": f"t{constants.SUMMARY_TOKEN_LIMIT}_random",
    "presumm": f"t{constants.SUMMARY_TOKEN_LIMIT}_presumm",
    "decsum011": f"t{constants.SUMMARY_TOKEN_LIMIT}_decsum011_beam{constants.DECSUM_BEAM_WIDTH}",
    "decsum010": f"t{constants.SUMMARY_TOKEN_LIMIT}_decsum010_beam{constants.DECSUM_BEAM_WIDTH}",
    "decsum001": f"t{constants.SUMMARY_TOKEN_LIMIT}_decsum001_beam{constants.DECSUM_BEAM_WIDTH}",
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

def pred_summary(summ_name, trainer, tokenizer):        
    summary_summ_path = os.path.join(
        constants.SUMMARY_DIR, 
        f"{constants.NUM_REVIEWS}reviews", 
        f"{summ_name}.jsonl.gz",
    )
    summary_pred_path = os.path.join(
        constants.SCORING_DIR, 
        f"preds_{summ_name}.pkl",
    )
    return pred_with_cache(summary_summ_path, summary_pred_path)

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    logging.info('Admin logged in')
    
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--summary_name",
        type=str,
        help="Filename of generated summaries from the test dataset excluding filetype ending (for example, 't50_random')")
    parser.add_argument(
        "--summary_type",
        type=str,
        choices=['random', 'presumm', 'decsum011', 'decsum010', 'decsum001'],
        help="Type of generated summary from the test dataset")
    
    args = parser.parse_args() 
    logging.info(f"args: {args}")
    
    if not (args.summary_name or args.summary_type):
        parser.error('No clear summary source specified, pick exactly one of --summary_type or --summary_name')
    if (args.summary_name and args.summary_type):
        parser.error('No clear summary source specified, pick exactly one of --summary_type or --summary_name')
    
    summary_fname = None
    if args.summary_name:
        summary_fname = args.summary_name
    if args.summary_type:
        summary_fname = summary_type_translations[args.summary_type]
    
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
    logging.info(f"generating/retrieving summary={summary_fname} predictions")
    summary_preds = pred_summary(summary_fname, trainer, tokenizer)
    # calculate results
    logging.info(f"CALCULATING RESULTS")
    logging.info(f"{summary_fname} :")
    logging.info(f"  MSE-Full: "+str(utils.compute_mse(baseline_preds[0], summary_preds[0])))
    logging.info(f"       MSE: "+str(summary_preds[2]["test_mse"]))
