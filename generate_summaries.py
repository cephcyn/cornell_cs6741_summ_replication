import argparse
import random
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import spacy

import utils
import constants

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logging.info('Admin logged in')

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
def truncate_sentence_ordering(ordered_sentences, token_limit):
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
# Takes in an entire dataset, returns an entire dataset
def generate_summary_random(
    dataset, spacy_nlp, tokenizer, token_limit=constants.SUMMARY_TOKEN_LIMIT
):
    result_dataset = []
    for entry_i in range(len(dataset)):
        # grab sorting and token count information
        sentences = utils.split_sentences(" ".join(dataset[entry_i]["reviews"]), spacy_nlp, tokenizer)

        # create shuffling of sentences
        shuffled_sentences = sentences
        # use the business ID string as a fixed seed to make things replicatable
        random.Random(testset[entry_i]["business"]).shuffle(shuffled_sentences)

        # reduce limit by 2 when truncating...
        # assume "begin" and "end" tokens (from longformer) are always going to be added to any individual text
        result_dataset.append({
            "reviews": utils.truncate_sentence_ordering(shuffled_sentences, token_limit-2),
            "scores": dataset[entry_i]["scores"],
            "business": dataset[entry_i]["business"],
            "avg_score": dataset[entry_i]["avg_score"],
        })
    return result_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--summary_type",
        type=str,
        choices=['decsum001', 'decsum010', 'decsum011', 'random'],
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
    
    # Generate summary now
    if args.summary_type == "random":
        logging.info(f"GENERATING RANDOM SUMMARIES")
        summaries_random = generate_summary_random(testset, nlp, tokenizer, token_limit=constants.SUMMARY_TOKEN_LIMIT)
        random_outfile = os.path.join(
            constants.SUMMARY_DIR,
            f"{constants.NUM_REVIEWS}reviews",
            f"t{constants.SUMMARY_TOKEN_LIMIT}_random.jsonl.gz",
        )
        logging.info(f"DONE GENERATING RANDOM SUMMARIES, NOW SAVING TO {random_outfile}")
        utils.dump_jsonl_gz(summaries_random, random_outfile)
    elif args.summary_type[:6]=="decsum":
        logging.info(f"GENERATING DECSUM SUMMARIES")
        pass
        logging.info(f"DONE GENERATING DECSUM SUMMARIES, NOW SAVING TO {decsum_outfile}")
        pass
    logging.info(f"Done!")
