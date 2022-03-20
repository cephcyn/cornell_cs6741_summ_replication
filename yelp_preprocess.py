"""
This file is a modification of https://github.com/ChicagoHAI/decsum/blob/main/preprocess/yelp_preprocess.py
Data can be downloaded from https://www.yelp.com/dataset/download
"""

import pickle
import gzip
import json
import pprint
import argparse
import os
from datetime import datetime
import logging
import random
import pandas as pd

# Default values for the cmd arguments
YELP_DATA_DIR = "/data/jz549_data/data_6741_repl"
OUTPUT_DIR = "/data/jz549_data/data_6741_repl_output"
NUM_REVIEWS = 50

# Filenames for different components of the Yelp dataset
DATASET_FNAME_BUSINESS = "yelp_academic_dataset_business.json"
DATASET_FNAME_REVIEW = "yelp_academic_dataset_review.json"
DATASET_PERC_TRAIN = 0.64
DATASET_PERC_DEV = 0.16

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logging.info('Admin logged in')

pp = pprint.PrettyPrinter(indent=4)
myprint = pp.pprint
DATE_PATTERN = "%Y-%m-%d %H:%M:%S"


def get_created_time(text):
    return int(datetime.strptime(text, DATE_PATTERN).strftime("%s"))


def convert_data(input_file, output_file):
    with open(input_file) as fin, gzip.open(output_file, "wt") as fout:
        business_dict = {}
        count = 0
        for line in fin:
            data = json.loads(line)
            business_id = data["business_id"]
            if business_id not in business_dict:
                business_dict[business_id] = []
            count += 1
            business_dict[business_id].append((
                get_created_time(data["date"]),
                data["user_id"],
                data["text"],
                data["stars"]))
            if count % 100000 == 0:
                logging.info(count)
        for b in business_dict:
            business_dict[b].sort()
            fout.write("%s\n" % json.dumps({"business": b, "reviews": business_dict[b]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--yelp_data_dir",
                        default=YELP_DATA_DIR,
                        type=str,
                        help="")
    parser.add_argument("--output_dir",
                        default=OUTPUT_DIR,
                        type=str,
                        help="")
    parser.add_argument("--data_split",
                        action='store_true', 
                        default=True,
                        help="")
    parser.add_argument("--num_review",
                        type=int,
                        default=NUM_REVIEWS,
                        help="Number of reviews for computing average rating")    
    
    args = parser.parse_args() 
    logging.info(f"args: {args}")
    
    if not os.path.isdir(args.output_dir):
        logging.info(f'Output directory does not exist. Creating at {args.output_dir}')
        os.makedirs(args.output_dir)
        logging.info(f'Output directory created.')

    restaurant_business_ids = set()
    with open(os.path.join(args.yelp_data_dir, DATASET_FNAME_BUSINESS), "r") as f:    
        for line in f:  
            if line:    
                json_content = json.loads(line)
                if json_content["categories"] != None:                        
                    categories = [val.lower().strip() for val in json_content["categories"].split(",")]
                    if "restaurants" in categories:
                        restaurant_business_ids.add(json_content["business_id"])
    
    logging.info(f"Finished reading the business JSON file. There are {len(restaurant_business_ids)} unique restaurants in the dataset.")

    grouped_reviews_filepath = os.path.join(args.output_dir, "grouped_reviews.jsonlist.gz")
    logging.info(f'Creating grouped reviews...')
    if not os.path.exists(grouped_reviews_filepath):
        logging.info(f'Grouped reviews not exist. Building grouped reviews at {grouped_reviews_filepath}')
        convert_data(os.path.join(args.yelp_data_dir, DATASET_FNAME_REVIEW), grouped_reviews_filepath)
        logging.info(f'Grouped reviews file built')

    logging.info(f"converting grouped reviews into resutaurant only reviews and compute average of first {args.num_review} reviews")
    out_file = os.path.join(args.output_dir, f"yelp_10reviews_{args.num_review}avg.jsonl.gz")
    if not os.path.exists(out_file):
        with gzip.open(grouped_reviews_filepath, 'rt') as f_in,  gzip.open(out_file, "wt") as fout:
            for l in f_in:
                r = json.loads(l)
                if r['business'] not in restaurant_business_ids:
                    continue
                if len(r['reviews']) >=args.num_review:
                    tmp = {"reviews":[], "scores":[]}
                    tmp['business'] = r['business']
                    for i in range(10):
                        tmp["reviews"].append(r["reviews"][i][2])
                        tmp["scores"].append(r["reviews"][i][3])
                    tmp["avg_score"] = sum([r["reviews"][j][3] for j in range(args.num_review)])/args.num_review
                    fout.write("%s\n" % json.dumps(tmp))
    else:
        logging.info(f'Conversion has already been done before')

    if args.data_split:

        split_datapath = os.path.join(args.output_dir, f"{args.num_review}reviews")
        logging.info(f"creating split dataset now in {split_datapath}")
        if not os.path.exists(split_datapath):
            logging.info(f'Split datapath not exist. Create split data dir at {split_datapath}')
            os.makedirs(split_datapath, exist_ok=True)
        
        def dump_jsonl_gz(obj, outpath):
            # obj is list of json
            with gzip.open(outpath, "wt") as fout:
                for o in obj:
                    fout.write("%s\n" % json.dumps(o))
        
        splits = {
            "train": DATASET_PERC_TRAIN, 
            "dev": DATASET_PERC_DEV, 
            "test": 1-DATASET_PERC_TRAIN-DATASET_PERC_DEV
        }
        split_ids_datapath = os.path.join(split_datapath, f"preprocess_{list(splits.keys())[-1]}_business_ids.csv")
        if not os.path.exists(split_ids_datapath):
            logging.info(f'Split IDs not exist. Creating ID lists in {split_ids_datapath} and comparable paths')
            logging.info(f'Creating train/dev/test split: {splits}')
            reordered_ids = list(restaurant_business_ids)
            random.shuffle(reordered_ids)
            last_idx = 0
            for s in splits:
                s_size = int(splits[s] * len(reordered_ids))
                s_fname = os.path.join(split_datapath, f"preprocess_{s}_business_ids.csv")
                pd.DataFrame({
                    'business': reordered_ids[last_idx:last_idx+s_size]
                }).to_csv(s_fname)
                last_idx += s_size
        split_ids = {}
        for s in splits:
            split_ids[s] = set(pd.read_csv(os.path.join(split_datapath, f"preprocess_{s}_business_ids.csv")).business.values)
        
        reviews = []
        with gzip.open(out_file, 'rt') as f:
            for line in f:
                reviews.append(json.loads(line))
            
        for split, ids in split_ids.items():
            split_reviews = []
            for review in reviews:
                if review['business'] in ids:
                    split_reviews.append(review)

            storepath = os.path.join(split_datapath, f"{split}.jsonl.gz")
            dump_jsonl_gz(split_reviews, storepath)
            logging.info(f"Data split length: {split} ({len(split_reviews)}), stored at: {storepath}")

    logging.info("Finished!")