# cornell_cs6741_summ_replication

A replication of part of Table 1 from "Decision-Focused Summarization" ([link](https://api.semanticscholar.org/CorpusID:237513705)).

Borrows some structure from the paper's released codebase [here](https://github.com/ChicagoHAI/decsum).

## Writeup

TODO

## Run instructions

### 1. Environment setup
```
pip install -r requirements.txt
pip install -Ue .
```

### 2. Dataset
Download the Yelp JSON dataset in `.tgz` format: https://www.yelp.com/dataset/download

Unzip the dataset to `YELP_DATA_DIR`:
```
tar -xvzf YELP_DATA_TGZ -C YELP_DATA_DIR
```

Preprocess the dataset into the task format (this uses a lightly modified version of the preprocessing script from the original paper codebase):
```
python -m yelp_preprocess [--yelp_data_dir YELP_DATA_DIR] [--output_dir OUTPUT_DIR]
```

### 3. Longformer finetuning

TODO

### 4. TODO...