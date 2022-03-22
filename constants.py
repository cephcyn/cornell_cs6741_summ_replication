# Filepaths for where data is stored
YELP_DATA_DIR = "/data/jz549_data/6741_summrepl/dataset_raw"
OUTPUT_DIR = "/data/jz549_data/6741_summrepl/dataset"
LOGGING_DIR = "/data/jz549_data/6741_summrepl/logging"
CHECKPOINT_DIR = "/data/jz549_data/6741_summrepl/checkpoints"
SUMMARY_DIR = "/data/jz549_data/6741_summrepl/summaries"
SCORING_DIR = "/data/jz549_data/6741_summrepl/scoring"
BACKUPS_DIR = "/data/jz549_data/6741_summrepl/backups"

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
