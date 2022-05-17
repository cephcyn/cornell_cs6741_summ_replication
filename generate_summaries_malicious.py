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

# Generate a malicious summary: 
#   1. cut up reviews into sentences
#   2. order them by increasing or decreasing individual prediction
#   3. Add all of them (in the same order as original data) until summary
#      would exceed [constants.SUMMARY_TOKEN_LIMIT] tokens (incl begin and end)
# Takes in a dataset entry and its sentence-split line (position, sentence, tokencount, input_ids, attention_mask)
# Returns a full dataset line
def generate_summary_malicioussent(
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

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    logging.info('Admin logged in')
    
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
    
    # Retrieve helper data
    logging.info(f"generating helper per-sentence predictions")
    helper_sentpredictions_cache_fname = os.path.join(
        constants.BACKUPS_DIR, 
        f"{constants.NUM_REVIEWS}reviews", 
        f"decsumhelperr_test.jsonl.gz"
    )
    # check if we can load from cache
    if os.path.exists(helper_sentpredictions_cache_fname):
        logging.info(f"R-helper backup exists, loading from {helper_sentpredictions_cache_fname}")
        with open(helper_sentpredictions_cache_fname, "rb") as f:
            helper_sentpredictions = pickle.load(f)
    else:
        raise NotImplementedError('generate_summaries has the original impl, but this script relies on this cache already existing')
        
    # Generate minimum-pred summary now
    logging.info(f"GENERATING MALICIOUS-NEGATIVE SUMMARIES")
    logging.info(f"mapping individual sentences into results")
    summaries_malicioussentmin = list(itertools.starmap(
        generate_summary_malicioussent, 
        zip(
            testset,
            testset_sentences, 
            helper_sentpredictions,
            itertools.repeat(False),
        ),
    ))
    malicioussentmin_outfile = os.path.join(
        constants.SUMMARY_DIR,
        f"{constants.NUM_REVIEWS}reviews",
        f"t{constants.SUMMARY_TOKEN_LIMIT}_malicioussentmin.jsonl.gz",
    )
    logging.info(f"DONE GENERATING MALICIOUS-NEGATIVE SUMMARIES, NOW SAVING TO {malicioussentmin_outfile}")
    utils.dump_jsonl_gz(summaries_malicioussentmin, malicioussentmin_outfile)
        
    # Generate maximum-pred summary now
    logging.info(f"GENERATING MALICIOUS-POSITIVE SUMMARIES")
    logging.info(f"mapping individual sentences into results")
    summaries_malicioussentmax = list(itertools.starmap(
        generate_summary_malicioussent, 
        zip(
            testset,
            testset_sentences, 
            helper_sentpredictions,
            itertools.repeat(True),
        ),
    ))
    malicioussentmax_outfile = os.path.join(
        constants.SUMMARY_DIR,
        f"{constants.NUM_REVIEWS}reviews",
        f"t{constants.SUMMARY_TOKEN_LIMIT}_malicioussentmax.jsonl.gz",
    )
    logging.info(f"DONE GENERATING MALICIOUS-POSITIVE SUMMARIES, NOW SAVING TO {malicioussentmax_outfile}")
    utils.dump_jsonl_gz(summaries_malicioussentmax, malicioussentmax_outfile)
    
    logging.info(f"Done!")
