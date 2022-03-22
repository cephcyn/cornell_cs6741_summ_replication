import os
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import torch
import pandas as pd
import datasets

import utils
import constants

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    logging.info('Admin logged in')

    # load longformer tokenizer
    logging.info(f"loading tokenizer={constants.LONGFORMER_MODEL_TYPE}")
    tokenizer = AutoTokenizer.from_pretrained(
        constants.LONGFORMER_MODEL_TYPE, 
        max_length=constants.MAX_SEQ_LENGTH,
    )
    # load datasets
    yelp = {}
    for type_name in ["train", "dev"]: # test set is not relevant for finetuning here
        logging.info(f"loading {type_name} dataset")
        yelp[type_name] = utils.load_yelp_dataset(os.path.join(
            constants.OUTPUT_DIR, 
            f"{constants.NUM_REVIEWS}reviews", 
            f"{type_name}.jsonl.gz"
        ))
    # Reformat datasets
    for dkey in yelp.keys():
        yelp[dkey] = yelp[dkey].map(lambda bt: utils.tokenization(bt, tokenizer), batched=True, batch_size=len(yelp[dkey]))
        yelp[dkey].set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    logging.info(f"Datasets padded, example has {len(yelp["train"]["input_ids"][0])} tokens and label={yelp["train"]["label"][0]}")
    # Load pretrained, non-finetuned model
    logging.info(f"loading model={constants.LONGFORMER_MODEL_TYPE}")
    model = AutoModelForSequenceClassification.from_pretrained(
        constants.LONGFORMER_MODEL_TYPE,
        num_labels=1,
        gradient_checkpointing=True,
    )
    # define the training arguments
    training_args = TrainingArguments(
        output_dir = constants.CHECKPOINT_DIR,
        dataloader_num_workers = constants.FINETUNE_NUM_WORKERS,
        learning_rate = constants.FINETUNE_LEARNING_RATE,
        per_device_train_batch_size = constants.FINETUNE_TRAIN_BATCH_SIZE,
        # gradient_accumulation_steps = 32,    
        per_device_eval_batch_size = constants.FINETUNE_EVAL_BATCH_SIZE,
        num_train_epochs = constants.FINETUNE_NUM_EPOCHS,
        evaluation_strategy = "epoch",
        logging_first_step = True,
        save_strategy = "epoch",
        disable_tqdm = False, 
        load_best_model_at_end = True,
        warmup_steps = constants.FINETUNE_WARMUP_STEPS,
        fp16 = constants.FINETUNE_USE_FP16,
        logging_dir = constants.LOGGING_DIR,
        run_name = 'longformer-yelp-finetuning',
    )
    # instantiate the trainer class and check for available devices
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=utils.compute_mse_metrics,
        train_dataset=yelp["train"],
        eval_dataset=yelp["dev"],
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"starting training; using device={device}")
    # train the model
    trainer.train(resume_from_checkpoint=True)
    # save the best model
    model_save_location = os.path.join(
        constants.CHECKPOINT_DIR,
        constants.LONGFORMER_SAVE
    )
    logging.info(f"training done, saving to {model_save_location}")
    trainer.save_model(model_save_location)
    # evaluation
    eval_preds = trainer.evaluate()
    logging.info(f"=== EVALUATION ===")
    logging.info(eval_preds)
