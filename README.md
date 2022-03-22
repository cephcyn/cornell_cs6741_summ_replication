# cornell_cs6741_summ_replication

A replication of part of Table 1 from "Decision-Focused Summarization" ([link](https://api.semanticscholar.org/CorpusID:237513705)).

Borrows some structure from the paper's released codebase [here](https://github.com/ChicagoHAI/decsum).

## Writeup

TODO misc notes for now...

Finetuning Longformer:
- Based on pretrained `longformer-base`, as that is what the decsum codebase uses.
- Finetuning `longformer-base` took approximately 4 hours (including restarting when out-of-memory) on a GeForce RTS 3090

Summarization:
- All summaries are cut off after N tokens
- (Random) Prioritizes sentences in a random order, terminates adding sentences upon encountering one that will exceed the N-token limit, and then re-orders all sentences so that they are sorted the same way as they were in the original text
- (DecSum) Excluding the "faithfulness" component of the DecSum summary quality metric, as that was reported to take over 10 hours per run on the test set.
- (DecSum) My implementation of DecSum doesn't follow their pseudocode / algorithm that they describe in "Algorithm 1" of the paper perfectly, as there seemed to be a typo in that algorithm
  - The "X <- X - x" line causes X to decrease in size, which doesn't match up with how the overall loss functions are defined.
  - Therefore, I would add an additional "Xoriginal <- X" before the while loop and replace all instances of "X" in the loss function calls with "Xoriginal"
- (DecSum) Token count is based on SpaCy model token counting, *not* Longformer model token counting.
  - The original paper is not clear about how token counts were done. The implementation uses SpaCy model token counting.
  - The difference between these: SpaCy tends to use clearer rules for tokenization, and generally counts distinct words. Longformer tokenizer seems to pick up on word prefixes, suffixes, or other modifiers more often.

Demo training scripts referenced:
- https://github.com/jlealtru/website_tutorials/blob/main/notebooks/Longformer%20with%20IMDB.ipynb (main reference)
- https://pytorch.org/tutorials/beginner/data_loading_tutorial.html (Dataset class implementation)
- https://github.com/huggingface/transformers/issues/7198 (checkpoint resuming)
- https://huggingface.co/docs/transformers/tasks/sequence_classification
- https://github.com/huggingface/notebooks/blob/master/transformers_doc/training.ipynb

## Run instructions

### 1. Environment setup

Use Python 3.8

```
pip install -r requirements.txt
pip3 install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio===0.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
python -m spacy download en_core_web_sm
```

Edit the `constants.py` file to alter experiment hyperparameters.
Arguments given in individual commands will override these defaults.

### 2. Dataset
Download the Yelp JSON dataset in `.tgz` format: https://www.yelp.com/dataset/download

Unzip the dataset to `YELP_DATA_DIR`:
```
tar -xvzf YELP_DATA_TGZ -C YELP_DATA_DIR
```

Preprocess the dataset into the task format (this uses a lightly modified version of the preprocessing script from the original paper codebase):
```
python -m yelp_preprocess [--yelp_data_dir YELP_DATA_DIR] [--output_dir OUTPUT_DIR] [--num_review NUM_REVIEWS]
```

### 3. Longformer finetuning

Run to finetune the Longformer evaluation model:
```
python -m longformer_finetune
```

The following summarization generation and evaluation code will not work without this model existing.

### 4. Run summarization generators

Run once each for each type of summary (`SUMMARY_TYPE`) being evaluated:
```
python -m generate_summaries --summary_type SUMMARY_TYPE
```

TODO text-only summarization, model-only summarization

### 5. Evaluate

TODO