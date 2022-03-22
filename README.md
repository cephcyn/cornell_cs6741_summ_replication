# cornell_cs6741_summ_replication

A replication of part of Table 1 from "Decision-Focused Summarization" ([link](https://api.semanticscholar.org/CorpusID:237513705)).

Borrows some structure from the paper's released codebase [here](https://github.com/ChicagoHAI/decsum).

## Writeup

### Introduction

The paper introduces the new task of "decision-focused summarization": generating summaries of a text to present relevant information for a decision being made that takes that text into account.
This is in contrast to most previous work on text summarization, which has sought to create general summaries of texts.
An example that the authors give to contrast these two scenarios is with medical diagnoses based on text medical notes: while medical text may contain information about some patient's foot injuries, if the text is being used for pancreatic cancer risk assessment, a summary of the text that contains information about foot injuries is very unlikely to be helpful.

The authors make several clear contributions:
- They build a dataset for this task (predicting Yelp score ratings based on the text of a review)
- They propose three goals of a decision-focused summarization (faithfulness, representativeness, and sentence diversity)
- They describe a summarization algorithm `DecSum` that selects sentences from a larger set of reviews based on these goals as a way of generating a summary
- They evaluate `DecSum` against three text-based summarization methods (random text selection, `BART`, and `PreSumm`) and two decision-model-based summarization methods (`IG`, `Attention`) using mean squared error between predictions made by a decision model on the original texts and the summaries as the reference metric.
- They evaluate these same summarization methods against each other with crowdsourced human decisions, judging on classification accuracy.

I focus on replicating parts of the first four points: building a dataset, implementing `DecSum`, implementing a subset of the other summarization methods they used, and comparing their mean squared error performance.

### Method

Dataset collection:
- The authors don't include their full dataset in the repository, so I re-downloaded the most recent version of the Yelp dataset and re-did preprocessing on that dataset with a new randomly selected train/dev/test dataset split in the same ratio that they used.
- The dataset that I had is 15% larger than theirs. (Their test set had size=3623, my test set had size=4172.)

Finetuning Longformer:
- I finetuned on pretrained `longformer-base-4096` (NOT `longformer-large-4096`), as that is what the decsum codebase uses.
  Oddly enough, their appendix description seems to contradict their public codebase.
  They describe using a model with 102M parameters, which matches the size of `longformer-large-4096`.
  The Longformer paper reports that `longformer-base-4096` only has 41M parameters.
  However, if I estimate model size based on training time, their reported training time (over 3 hours) matches up with how much time I'd guess training `longformer-base-4096` takes, since I spent only slightly less time than that training a `longformer-base-4096` model with a higher-spec GPU (I used a RTX 3090, they reported using a RTX Titan).
- I use the exact same training hyperparameters as the authors describe in their appendix, other than changing the number of workers and the batch size.

Summarization:
- Their appendix mentions that all sentence-based summaries are limited to N=50 tokens (words).
- (Random) Prioritizes sentences in a random order, terminates adding sentences upon encountering one that will exceed the N-token limit, and then re-orders all sentences so that they are sorted the same way as they were in the original text.
- (DecSum) I exclude the "faithfulness" component of the DecSum summary quality metric, as that was reported to take over 10 hours per run on the test set.
- (DecSum) My implementation of DecSum doesn't follow their pseudocode / algorithm that they describe in "Algorithm 1" of the paper perfectly, as there seemed to be a typo in that algorithm.
  - The "X <- X - x" line causes X to decrease in size, which doesn't match up with how the overall loss functions are defined.
  - I would add an additional "Xoriginal <- X" before the while loop and replace all instances of "X" in the loss function calls with "Xoriginal".
- (DecSum) Token count is based on SpaCy model token counting, *not* Longformer model token counting.
  - The original paper is not clear about how token counts were done. The implementation uses SpaCy model token counting.
  - The difference between these: SpaCy tends to use clearer rules for tokenization, and generally counts distinct words. Longformer tokenizer seems to pick up on word prefixes, suffixes, or other modifiers more often.

### Results

TODO

Runtimes:
- Collecting the dataset and preprocessing took approximately 5 minutes?
- Finetuning `longformer-base-4096` took slightly under 3 hours (excluding restarting when out-of-memory?) on a GeForce RTX 3090.
  - However, when I was initially trying to run this finetuning on Google Colab, the estimated finetuning time was over 36 hours (and it kicked me off of the server after about 4 hours of finetuning time).
- Generating random summaries for the test dataset took approximately 5 minutes?
- Generating DecSum(0,0,1) summaries for the test dataset took approximately 15 minutes?
- Generating DecSum(0,1,0) summaries for the test dataset took TODO minutes
- Generating DecSum(0,1,1) summaries for the test dataset took TODO minutes

### Appendix: Tutorials referenced when implementing all of this...

Demo training scripts referenced:
- https://github.com/jlealtru/website_tutorials/blob/main/notebooks/Longformer%20with%20IMDB.ipynb (main reference)
- https://pytorch.org/tutorials/beginner/data_loading_tutorial.html (Dataset class implementation)
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

TODO impl text-only summarization, model-only summarization

### 5. Evaluate

TODO cleanup this code...