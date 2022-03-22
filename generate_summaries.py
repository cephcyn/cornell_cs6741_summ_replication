import argparse
import random
import logging
import os
import itertools
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer
import datasets
import spacy
import torch
from sentence_transformers import SentenceTransformer
from scipy.stats import wasserstein_distance
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import psutil 

import utils
import constants

# Split merged text blob into distinct sentences, with their ordering IDs and distinct token counts
# Takes mergedtext, returns [(position, sentence, tokencount, input_ids, attention_mask), ...]
def split_sentences(mergedtext, spacy_nlp, tokenizer):
    spacy_parse = spacy_nlp(mergedtext)
    # split merged text blob into sentences
    sentences = [s.text for s in spacy_parse.sents]

    # save the original sentence ordering, as well as number of tokens (excluding begin and end)
    tokenizer_calculated = tokenizer(sentences)
    sentence_ordering = range(len(sentences))
    sentence_tokencount = [len(s) for s in spacy_parse.sents]
    sentence_inputids = tokenizer_calculated["input_ids"]
    sentence_attnmask = tokenizer_calculated["attention_mask"]
    sentences = list(zip(sentence_ordering, sentences, sentence_tokencount, sentence_inputids, sentence_attnmask))
    
    return sentences

# Include only the top N sentences in the ordering, where N is the maximum you can get until token count starts exceeding limit
# Takes [(position, sentence, tokencount, input_ids, attention_mask), ...], returns sorted [sentence, sentence, ...]
def cleanup_sentences(ordered_sentences, token_limit):
    # pick out the top N sentences to hit the token count limit
    num_tokens_already = 0
    can_add = num_tokens_already+ordered_sentences[0][2] <= token_limit
    collected_sentences = []
    while can_add:
        to_add = ordered_sentences[0]
        collected_sentences.append(to_add)
        num_tokens_already += to_add[2]
        ordered_sentences = ordered_sentences[1:]
        if len(ordered_sentences)==0:
            break
        can_add = num_tokens_already+ordered_sentences[0][2] <= token_limit
    
    # sort sentences by original ordering
    collected_sentences.sort(key=lambda x: x[2])
    
    # flatten out the sentences back to original contents only
    return [e[1] for e in collected_sentences]

# Generate a random summary: 
#   1. cut up reviews into sentences
#   2. randomly order them
#   3. Add all of them (in the same order as original data) until summary
#      would exceed [constants.SUMMARY_TOKEN_LIMIT] tokens (incl begin and end)
# Takes in a dataset entry and its sentence-split line (position, sentence, tokencount, input_ids, attention_mask)
# Returns a full dataset line
def generate_summary_random(
    dataset_entry, entry_sentences, token_limit=constants.SUMMARY_TOKEN_LIMIT
):
    # create shuffling of sentences
    shuffled_sentences = entry_sentences.copy()
    # use the business ID string as a fixed seed to make things replicatable
    random.Random(dataset_entry["business"]).shuffle(shuffled_sentences)

    return {
        "reviews": cleanup_sentences(shuffled_sentences, token_limit),
        "scores": dataset_entry["scores"],
        "business": dataset_entry["business"],
        "avg_score": dataset_entry["avg_score"],
    }

# Generate a DecSum summary: 
#   1. cut up reviews into sentences
#   2. select the {sentence_count_limit} number of sentences that minimizes 
#      objective function via greedy beamsearch algorithm
#   3. Add all of them (in the same order as original data) until summary
#      would exceed [constants.SUMMARY_TOKEN_LIMIT] tokens
# Takes in a dataset entry and its sentence-split line (position, sentence, tokencount, input_ids, attention_mask)
# Returns a full dataset line
# Note: if gamma_distinct!=0, then sentbert is required
def generate_summary_decsum(
    dataset_entry, entry_sentences, model_trainer, tokenizer, sentbert,
    alpha_faithful, beta_represent, gamma_distinct,
    beam_width=constants.DECSUM_BEAM_WIDTH,
    token_limit=constants.SUMMARY_TOKEN_LIMIT, 
    sentence_count_limit=constants.SUMMARY_SENTENCE_LIMIT, 
):
    # Do argument checks
    if (gamma_distinct!=0) and (sentbert is None):
        raise ValueError("sentbert is None, should be passed when gamma_distinct!=0")

    # define faithfulness metric
    def objective_f(selected, all_sentences, model_trainer):
        if alpha_faithful==0:
            return 0
        raise NotImplementedError()
    # define representativeness metric
    def objective_r(selected, full_sentence_preds):
        if beta_represent==0:
            return 0
        selected_preds = [full_sentence_preds[e[0]] for e in selected]
        return np.log(wasserstein_distance(selected_preds, full_sentence_preds))
    # define textual non-redundancy metric
    def objective_d(selected, cos_similarity):
        if gamma_distinct==0:
            return 0
        selected_ids = [e[0] for e in selected]
        selected_cos_similarity = cos_similarity[selected_ids][:,selected_ids]
        # Pick out the maximally different sentence per sentence, then sum
        # Note that the diagonals were set to 0, so it'll avoid calculating self-similarity
        return sum(np.apply_along_axis(max, 1, selected_cos_similarity))
    
    # calculate prediction distributions per-sentence if we need to
    full_sentence_preds = None
    if beta_represent!=0:
        # setup the dataset format
        config_dataset = datasets.Dataset.from_dict({
            "text": [e[1] for e in entry_sentences],
            "label": [e[0] for e in entry_sentences],
        }).map(lambda x: tokenizer(x["text"], padding="max_length", truncation=True, max_length=constants.MAX_SEQ_LENGTH))
        config_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
        # run predictions per-sentence
        full_sentence_preds = model_trainer.predict(test_dataset=config_dataset)[0]
        full_sentence_preds = np.reshape(full_sentence_preds, (len(full_sentence_preds),))
        # clear out the dataset in memory
        config_dataset = None

    # Apply initial SentBERT pass if we need to
    full_cos_similarity = None
    if gamma_distinct!=0:
        full_cos_similarity = cosine_similarity(sentbert.encode(entry_sentences, convert_to_numpy=True))
        # We're always going to be ignoring sentence self-similarity
        np.fill_diagonal(full_cos_similarity, 0)

    # select the sentences that will be included with beam search!
    # each beam candidate is in format [objective_score, selected_sentences, available_sentences]
    beam_candidates = [[0, [], entry_sentences]]
    # Should we activate "objective_r only in first step"?
    is_first_step = beta_represent!=0
    while (
        (len(beam_candidates[0][1])<sentence_count_limit) and    # we haven't hit the sentence count threshold yet
        (len(beam_candidates[0][2])!=0) and                      # there's still sentences available to add
        (sum([e[2] for e in beam_candidates[0][1]])<token_limit) # we haven't hit the token count threshold yet
    ): 
        # take out the next best option
        current_candidate = beam_candidates[0]
        beam_candidates = beam_candidates[1:]
        # add all potential options
        new_options = []
        for s_i in range(len(current_candidate[2])):
            considering_sentence = current_candidate[2][s_i]
            considering_selected = current_candidate[1]+[considering_sentence]
            considering_remnants = current_candidate[2][:s_i]+current_candidate[2][s_i+1:]
            # in keeping with pseudocode, ONLY consider the representativeness metric on first step
            new_options.append([
                (
                    (not is_first_step)*alpha_faithful*objective_f(considering_selected, sentences, model_trainer)
                    + beta_represent*objective_r(considering_selected, full_sentence_preds)
                    + (not is_first_step)*gamma_distinct*objective_d(considering_selected, full_cos_similarity)
                ),
                considering_selected,
                considering_remnants
            ])
            is_first_step = False
        beam_candidates = beam_candidates + new_options
        # resort the beam candidates
        beam_candidates.sort(key=lambda x:x[0])
        # trim down the beam candidates
        beam_candidates = beam_candidates[:min(4, len(beam_candidates))]
    # we've outrunned the maximum sentence count / ran out of sentences we can add
    selected_sentences = beam_candidates[0][1]

    return {
        "reviews": cleanup_sentences(selected_sentences, token_limit),
        "scores": dataset_entry["scores"],
        "business": dataset_entry["business"],
        "avg_score": dataset_entry["avg_score"],
    }

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    logging.info('Admin logged in')
    
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--summary_type",
        type=str,
        choices=['random', 'presumm', 'decsum011', 'decsum010', 'decsum001'],
        help="Summary type to create based on test dataset")
    
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
    if args.summary_type == "random":
        logging.info(f"GENERATING RANDOM SUMMARIES")
        logging.info(f"mapping individual sentences into results")
        summaries_random = itertools.starmap(
            generate_summary_random, 
            zip(
                testset,
                testset_sentences, 
            ),
        )
        random_outfile = os.path.join(
            constants.SUMMARY_DIR,
            f"{constants.NUM_REVIEWS}reviews",
            f"t{constants.SUMMARY_TOKEN_LIMIT}_random.jsonl.gz",
        )
        logging.info(f"DONE GENERATING RANDOM SUMMARIES, NOW SAVING TO {random_outfile}")
        utils.dump_jsonl_gz(summaries_random, random_outfile)
    elif args.summary_type == "presumm":
        logging.info(f"GENERATING PRESUMM SUMMARIES")
        # TODO
        raise NotImplementedError()
        presumm_outfile = os.path.join(
            constants.SUMMARY_DIR,
            f"{constants.NUM_REVIEWS}reviews",
            f"t{constants.SUMMARY_TOKEN_LIMIT}_presumm.jsonl.gz",
        )
        logging.info(f"DONE GENERATING PRESUMM SUMMARIES, NOW SAVING TO {presumm_outfile}")
    elif args.summary_type[:6]=="decsum":
        logging.info(f"GENERATING DECSUM SUMMARIES")
        logging.info(f"need to initialize model for running...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logging.info(f"will be running model; using device={device}")
        sentbert = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens', device=device)
        logging.info(f"loaded SentenceTransformer")
        model_save_location = os.path.join(
            constants.CHECKPOINT_DIR,
            constants.LONGFORMER_SAVE
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_save_location, 
            num_labels=1,
        )
        # setup the trainer...
        model_trainer = Trainer(
            model=model,
        )
        logging.info(f"loaded trainer of finetuned longformer from {model_save_location}")
        opt_f = int(args.summary_type[6])
        opt_r = int(args.summary_type[7])
        opt_d = int(args.summary_type[8])
        logging.info(f"mapping individual sentences into results")
        summaries_decsum = itertools.starmap(
            generate_summary_decsum,
            zip(
                testset,
                testset_sentences,
                itertools.repeat(model_trainer),
                itertools.repeat(tokenizer),
                itertools.repeat(sentbert),
                itertools.repeat(opt_f), 
                itertools.repeat(opt_r), 
                itertools.repeat(opt_d),
            ),
        )
        decsum_outfile = os.path.join(
            constants.SUMMARY_DIR,
            f"{constants.NUM_REVIEWS}reviews",
            f"t{constants.SUMMARY_TOKEN_LIMIT}_decsum{opt_f}{opt_r}{opt_d}_beam{constants.DECSUM_BEAM_WIDTH}.jsonl.gz",
        )
        logging.info(f"DONE GENERATING DECSUM SUMMARIES, NOW SAVING TO {decsum_outfile}")
        utils.dump_jsonl_gz(summaries_decsum, decsum_outfile)
    logging.info(f"Done!")
