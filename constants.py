# Base directory where all data relevant to this codebase are stored
BASE_DIR = "/data/jz549_data/6741_summfinal"

# Filepaths for where data is stored
YELP_DATA_DIR = f"{BASE_DIR}/dataset_raw"
OUTPUT_DIR = f"{BASE_DIR}/dataset"
LOGGING_DIR = f"{BASE_DIR}/logging"
CHECKPOINT_DIR = f"{BASE_DIR}/checkpoints"
SUMMARY_DIR = f"{BASE_DIR}/summaries"
SCORING_DIR = f"{BASE_DIR}/scoring"
BACKUPS_DIR = f"{BASE_DIR}/backups"

# Raw dataset information
DATASET_FNAME_BUSINESS = "yelp_academic_dataset_business.json"
DATASET_FNAME_REVIEW = "yelp_academic_dataset_review.json"

# Dataset preprocessing hyperparameters
NUM_REVIEWS = 50

# Longformer finetuning hyperparameters
LONGFORMER_MODEL_TYPE = "allenai/longformer-base-4096"
DATASET_PERC_TRAIN = 0.64
DATASET_PERC_DEV = 0.16
MAX_SEQ_LENGTH = 3000
FINETUNE_NUM_WORKERS = 32
FINETUNE_LEARNING_RATE = 5e-5
FINETUNE_TRAIN_BATCH_SIZE = 16
FINETUNE_EVAL_BATCH_SIZE = 32
FINETUNE_NUM_EPOCHS = 3
FINETUNE_WARMUP_STEPS = 500
FINETUNE_USE_FP16 = True
LONGFORMER_SAVE = "longformer_best"

# Summary generation hyperparameters
SUMMARY_TOKEN_LIMIT = 50
SUMMARY_SENTENCE_LIMIT = 15
DECSUM_BEAM_WIDTH = 4
MULTPOOL_PROCESSES = 16
PRESUMM_MODEL_FNAME = f"{BASE_DIR}/backups/presumm_model.ckpt"