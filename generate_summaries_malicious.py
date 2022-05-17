import argparse
import random
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

# Generate a strictly-ordered malicious summary: 
#   1. cut up reviews into sentences
#   2. order them by increasing or decreasing individual prediction
#   3. Add all of them (in the same order as original data) until summary
#      would exceed [constants.SUMMARY_TOKEN_LIMIT] tokens (incl begin and end)
# Takes in a dataset entry and its sentence-split line (position, sentence, tokencount, input_ids, attention_mask)
# Returns a full dataset line
def generate_summary_malranked(
    dataset_entry, entry_sentences, helper_sentpreds,
    maximize_pred=False,
    token_limit=constants.SUMMARY_TOKEN_LIMIT,
):
    # Do argument checks
    if (len(entry_sentences)!=len(helper_sentpreds)):
        raise ValueError(f"len(entry_sentences)={len(entry_sentences)} != len(r_helper_sentpreds)={len(helper_sentpreds)}")
    
    # Zip predicted scores together with source information
    # creates entries of [(prediction, sentence-split tuple) ...]
    selected_sentences = [
        (
            helper_sentpreds[i],
            entry_sentences[i],
        ) 
        for i in range(len(entry_sentences))
    ]
    
    # create re-ordered sentences
    # by default, prioritize sentences that have a minimized individual prediction
    # if we are trying to maximize, then this is descending order.
    # otherwise, ascending order.
    selected_sentences.sort(key=lambda x: x[0], reverse=maximize_pred)
    
    # Convert back to standard sentence-split format
    selected_sentences = [e[1] for e in selected_sentences]
    
    # cleanup sentences
    reordered_sentences, reordered_sentence_ixs = generate_summaries.cleanup_sentences(selected_sentences, token_limit)

    return {
        "reviews": reordered_sentences,
        "scores": dataset_entry["scores"],
        "business": dataset_entry["business"],
        "avg_score": dataset_entry["avg_score"],
        "sentence_ix": reordered_sentence_ixs,
    }

# Generate a sampled malicious summary: 
#   1. cut up reviews into sentences
#   2. order them using a random sampling based on increasing or decreasing individual prediction
#   3. Add all of them (in the same order as original data) until summary
#      would exceed [constants.SUMMARY_TOKEN_LIMIT] tokens (incl begin and end)
# Takes in a dataset entry and its sentence-split line (position, sentence, tokencount, input_ids, attention_mask)
# Returns a full dataset line
def generate_summary_malsampled(
    dataset_entry, entry_sentences, helper_sentpreds,
    maximize_pred=False, random_seed=None,
    token_limit=constants.SUMMARY_TOKEN_LIMIT,
):
    # Do argument checks
    if (len(entry_sentences)!=len(helper_sentpreds)):
        raise ValueError(f"len(entry_sentences)={len(entry_sentences)} != len(r_helper_sentpreds)={len(helper_sentpreds)}")
        
    # Initialize the random selector
    random_selector = None
    if random_seed is None:
        random_selector = random.Random(dataset_entry["business"])
    else:
        random_selector = random.Random(str(dataset_entry["business"])+str(random_seed))
        
    # Get prediction weights
    # by default, prioritize sentences that have a minimized individual prediction
    # if we are trying to maximize, then this is descending order.
    # otherwise, ascending order.
    if maximize_pred:
        # higher predictions have higher weight
        sampling_weights = [(i, helper_sentpreds[i]) for i in range(len(entry_sentences))]
    else:
        # lower predictions have higher weight
        sampling_weights = [(i, constants.MAX_PREDICTION-helper_sentpreds[i]) for i in range(len(entry_sentences))]
    
    # Build a sampling without replacement
    selected_sentences = []
    while len(selected_sentences)<len(entry_sentences):
        # select which to add
        selected_i = random_selector.choices(
            sampling_weights, 
            weights=[e[1] for e in sampling_weights],
        )[0]
        # add it
        selected_sentences = selected_sentences + [entry_sentences[selected_i[0]]]
        # remove it from the remaining available options
        sampling_weights = [e for e in sampling_weights if e[0]!=selected_i[0]]
    
    # cleanup sentences
    reordered_sentences, reordered_sentence_ixs = generate_summaries.cleanup_sentences(selected_sentences, token_limit)

    return {
        "reviews": reordered_sentences,
        "scores": dataset_entry["scores"],
        "business": dataset_entry["business"],
        "avg_score": dataset_entry["avg_score"],
        "sentence_ix": reordered_sentence_ixs,
    }

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    logging.info('Admin logged in')
    
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--malicioussummary_type",
        type=str,
        choices=['strictranking', 'sampled'],
        help="Summary type to create based on test dataset")
    parser.add_argument(
        "--seed",
        type=int,
        help="Seed to use when summarizing (relevant to sampled-malicious)")
    
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
        f"sentencesplit_test.pkl"
    )
    # check if we can load from cache
    if os.path.exists(testset_sentences_cache_fname):
        logging.info(f"pre-parse backup exists, loading from {testset_sentences_cache_fname}")
        with open(testset_sentences_cache_fname, "rb") as f:
            testset_sentences = pickle.load(f)
    else:
        raise NotImplementedError('generate_summaries has the original impl, but this script relies on this cache already existing')
        
    # initialize final predictions
    logging.info(f"making full-review predictions")
    baselinepred_cache_fname = os.path.join(
        constants.SCORING_DIR, 
        "preds_baseline.pkl",
    )
    # check if we can load from cache
    if os.path.exists(baselinepred_cache_fname):
        logging.info(f"baseline prediction backup exists, loading from {baselinepred_cache_fname}")
        with open(baselinepred_cache_fname, "rb") as f:
            baselinepred = pickle.load(f)
    else:
        raise NotImplementedError('score_summaries has the original impl, but this script relies on this cache already existing')
    
    # Retrieve helper data
    logging.info(f"generating helper per-sentence predictions")
    helper_sentpredictions_cache_fname = os.path.join(
        constants.BACKUPS_DIR, 
        f"{constants.NUM_REVIEWS}reviews", 
        f"decsumhelperr_test.pkl"
    )
    # check if we can load from cache
    if os.path.exists(helper_sentpredictions_cache_fname):
        logging.info(f"R-helper backup exists, loading from {helper_sentpredictions_cache_fname}")
        with open(helper_sentpredictions_cache_fname, "rb") as f:
            helper_sentpredictions = pickle.load(f)
    else:
        raise NotImplementedError('generate_summaries has the original impl, but this script relies on this cache already existing')
    
    # Generate summary now
    if args.malicioussummary_type == "strictranking":
        # Generate minimum-pred summary now
        logging.info(f"GENERATING STRICT-MALICIOUS-NEGATIVE SUMMARIES")
        logging.info(f"mapping individual sentences into results")
        summaries_malrankedmin = list(itertools.starmap(
            generate_summary_malranked, 
            zip(
                testset,
                testset_sentences, 
                helper_sentpredictions,
                itertools.repeat(False),
            ),
        ))
        malrankedmin_outfile = os.path.join(
            constants.SUMMARY_DIR,
            f"{constants.NUM_REVIEWS}reviews",
            f"t{constants.SUMMARY_TOKEN_LIMIT}_malrankedmin.jsonl.gz",
        )
        logging.info(f"DONE GENERATING STRICT-MALICIOUS-NEGATIVE SUMMARIES, NOW SAVING TO {malrankedmin_outfile}")
        utils.dump_jsonl_gz(summaries_malrankedmin, malrankedmin_outfile)

        # Generate maximum-pred summary now
        logging.info(f"GENERATING STRICT-MALICIOUS-POSITIVE SUMMARIES")
        logging.info(f"mapping individual sentences into results")
        summaries_malrankedmax = list(itertools.starmap(
            generate_summary_malranked, 
            zip(
                testset,
                testset_sentences, 
                helper_sentpredictions,
                itertools.repeat(True),
            ),
        ))
        malrankedmax_outfile = os.path.join(
            constants.SUMMARY_DIR,
            f"{constants.NUM_REVIEWS}reviews",
            f"t{constants.SUMMARY_TOKEN_LIMIT}_malrankedmax.jsonl.gz",
        )
        logging.info(f"DONE GENERATING STRICT-MALICIOUS-POSITIVE SUMMARIES, NOW SAVING TO {malrankedmax_outfile}")
        utils.dump_jsonl_gz(summaries_malrankedmax, malrankedmax_outfile)
    elif args.malicioussummary_type == "sampled":
        # Generate low-pred summary now
        logging.info(f"GENERATING SAMPLED-MALICIOUS-NEGATIVE SUMMARIES")
        logging.info(f"using seed={args.seed} ...")
        logging.info(f"mapping individual sentences into results")
        summaries_malsampledmin = list(itertools.starmap(
            generate_summary_malsampled, 
            zip(
                testset,
                testset_sentences, 
                helper_sentpredictions,
                itertools.repeat(False),
                itertools.repeat(args.seed),
            ),
        ))
        malsampledmin_outfile = os.path.join(
            constants.SUMMARY_DIR,
            f"{constants.NUM_REVIEWS}reviews",
            f"t{constants.SUMMARY_TOKEN_LIMIT}_malsampledmin_{args.seed}.jsonl.gz",
        )
        logging.info(f"DONE GENERATING SAMPLED-MALICIOUS-NEGATIVE SUMMARIES, NOW SAVING TO {malsampledmin_outfile}")
        utils.dump_jsonl_gz(summaries_malsampledmin, malsampledmin_outfile)

        # Generate high-pred summary now
        logging.info(f"GENERATING SAMPLED-MALICIOUS-POSITIVE SUMMARIES")
        logging.info(f"using seed={args.seed} ...")
        logging.info(f"mapping individual sentences into results")
        summaries_malsampledmax = list(itertools.starmap(
            generate_summary_malsampled, 
            zip(
                testset,
                testset_sentences, 
                helper_sentpredictions,
                itertools.repeat(True),
                itertools.repeat(args.seed),
            ),
        ))
        malsampledmax_outfile = os.path.join(
            constants.SUMMARY_DIR,
            f"{constants.NUM_REVIEWS}reviews",
            f"t{constants.SUMMARY_TOKEN_LIMIT}_malsampledmax_{args.seed}.jsonl.gz",
        )
        logging.info(f"DONE GENERATING SAMPLED-MALICIOUS-POSITIVE SUMMARIES, NOW SAVING TO {malsampledmax_outfile}")
        utils.dump_jsonl_gz(summaries_malsampledmax, malsampledmax_outfile)
    
    logging.info(f"Done!")
