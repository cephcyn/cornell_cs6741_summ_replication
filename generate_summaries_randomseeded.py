import argparse
import logging
import os
import itertools
import pickle
import transformers
from transformers import AutoTokenizer
import spacy

import utils
import constants
import generate_summaries

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    logging.info('Admin logged in')
    
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--seed",
        type=int,
        help="Seed to use when summarizing")
    
    args = parser.parse_args() 
    logging.info(f"args: {args}")
    
    # Load dataset that we're making summaries of
    testset = utils.load_jsonl_gz(os.path.join(
        constants.OUTPUT_DIR, 
        f"{constants.NUM_REVIEWS}reviews", 
        f"test.jsonl.gz"
    ))
    logging.info(f"Loaded dataset being summarized, length={len(testset)}")
    # Load SpaCy
    logging.info(f"now loading SpaCy model")
    nlp = spacy.load('en_core_web_sm', disable=['tokenizer', 'ner', 'tagger','textcat'])
    # Load longformer tokenizer
    logging.info(f"now loading tokenizer={constants.LONGFORMER_MODEL_TYPE}")
    tokenizer = AutoTokenizer.from_pretrained(
        constants.LONGFORMER_MODEL_TYPE,
        max_length=constants.MAX_SEQ_LENGTH,
    )
    # initialize sentence splits
    logging.info(f"pre-parsing individual sentences")
    testset_sentences_cache_fname = os.path.join(
        constants.BACKUPS_DIR, 
        f"{constants.NUM_REVIEWS}reviews", 
        f"sentencesplit_test.jsonl.gz"
    )
    # check if we can load from cache
    if os.path.exists(testset_sentences_cache_fname):
        logging.info(f"pre-parse backup exists, loading from {testset_sentences_cache_fname}")
        with open(testset_sentences_cache_fname, "rb") as f:
            testset_sentences = pickle.load(f)
    else:
        logging.info(f"pre-parse backup does not exist, will export to {testset_sentences_cache_fname}")
        testset_sentences = [split_sentences(" ".join(e["reviews"]), nlp, tokenizer) for e in testset]
        with open(testset_sentences_cache_fname, "wb") as f:
            pickle.dump(testset_sentences, f)
    
    # Generate summary now
    logging.info(f"GENERATING RANDOM SUMMARIES")
    logging.info(f"using seed={args.seed} ...")
    logging.info(f"mapping individual sentences into results")
    summaries_random = list(itertools.starmap(
        generate_summaries.generate_summary_random, 
        zip(
            testset,
            testset_sentences, 
            itertools.repeat(args.seed),
        ),
    ))
    random_outfile = os.path.join(
        constants.SUMMARY_DIR,
        f"{constants.NUM_REVIEWS}reviews",
        f"t{constants.SUMMARY_TOKEN_LIMIT}_random_{args.seed}.jsonl.gz",
    )
    logging.info(f"DONE GENERATING RANDOM SUMMARIES, NOW SAVING TO {random_outfile}")
    utils.dump_jsonl_gz(summaries_random, random_outfile)
    logging.info(f"Done!")
