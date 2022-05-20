# Sentence-based extractive summarization

An exploration of how much sentence-based extractive summarization can be manipulated to support any particular final prediction, and what metrics may be effective if trying to uncover this manipulation.

This work is built on top of a replication project of "Decision-Focused Summarization" ([link](https://api.semanticscholar.org/CorpusID:237513705)).
The README document and relevant writeup for the replication checkpoint is accessible [here](readme_replication.md).

## Writeup

## Run instructions

### 1. Checkpoint environment/dataset/cachefile setup

Fork this repository, and then follow all run instructions in the replication checkpoint readme [here](readme_replication.md).

### 2. Generate and cache predictions of more random summaries

Run the random-seeded summarization generator multiple times with new seeds:

```
python -m generate_summaries_randomseeded --seed 0
python -m generate_summaries_randomseeded --seed 1
python -m generate_summaries_randomseeded --seed 2
python -m generate_summaries_randomseeded --seed 3
python -m generate_summaries_randomseeded --seed 4
python -m generate_summaries_randomseeded --seed 5
python -m generate_summaries_randomseeded --seed 6
python -m generate_summaries_randomseeded --seed 7
python -m generate_summaries_randomseeded --seed 8
python -m generate_summaries_randomseeded --seed 9
```

Score them (conveniently saving a cache of per-summary predictions along the way):

```
python -m score_summaries --summary_name t50_random_0
python -m score_summaries --summary_name t50_random_1
python -m score_summaries --summary_name t50_random_2
python -m score_summaries --summary_name t50_random_3
python -m score_summaries --summary_name t50_random_4
python -m score_summaries --summary_name t50_random_5
python -m score_summaries --summary_name t50_random_6
python -m score_summaries --summary_name t50_random_7
python -m score_summaries --summary_name t50_random_8
python -m score_summaries --summary_name t50_random_9
```

### 3. Generate and cache predictions of malicious high/low summaries

Run the malicious sentence generator:

```
# Generate the strictly-ranked summaries
python -m generate_summaries_malicious --malicioussummary_type strictranking
# Generate the sampled summaries
python -m generate_summaries_malicious --malicioussummary_type sampled --seed 0
python -m generate_summaries_malicious --malicioussummary_type sampled --seed 1
python -m generate_summaries_malicious --malicioussummary_type sampled --seed 2
python -m generate_summaries_malicious --malicioussummary_type sampled --seed 3
python -m generate_summaries_malicious --malicioussummary_type sampled --seed 4
python -m generate_summaries_malicious --malicioussummary_type sampled --seed 5
python -m generate_summaries_malicious --malicioussummary_type sampled --seed 6
python -m generate_summaries_malicious --malicioussummary_type sampled --seed 7
python -m generate_summaries_malicious --malicioussummary_type sampled --seed 8
python -m generate_summaries_malicious --malicioussummary_type sampled --seed 9
# Generate the baseline-meeting summaries
python -m generate_summaries_malicious --malicioussummary_type meetbaseline
```

Score them:

```
# Score the strictly-ranked summaries
python -m score_summaries --summary_name t50_malrankedmin
python -m score_summaries --summary_name t50_malrankedmax
# Score the sampled summaries
python -m score_summaries --summary_name t50_malsampledmin_0
python -m score_summaries --summary_name t50_malsampledmax_0
python -m score_summaries --summary_name t50_malsampledmin_1
python -m score_summaries --summary_name t50_malsampledmax_1
python -m score_summaries --summary_name t50_malsampledmin_2
python -m score_summaries --summary_name t50_malsampledmax_2
python -m score_summaries --summary_name t50_malsampledmin_3
python -m score_summaries --summary_name t50_malsampledmax_3
python -m score_summaries --summary_name t50_malsampledmin_4
python -m score_summaries --summary_name t50_malsampledmax_4
python -m score_summaries --summary_name t50_malsampledmin_5
python -m score_summaries --summary_name t50_malsampledmax_5
python -m score_summaries --summary_name t50_malsampledmin_6
python -m score_summaries --summary_name t50_malsampledmax_6
python -m score_summaries --summary_name t50_malsampledmin_7
python -m score_summaries --summary_name t50_malsampledmax_7
python -m score_summaries --summary_name t50_malsampledmin_8
python -m score_summaries --summary_name t50_malsampledmax_8
python -m score_summaries --summary_name t50_malsampledmin_9
python -m score_summaries --summary_name t50_malsampledmax_9
# Score the baseline-meeting summaries
python -m score_summaries --summary_name t50_meetbaseline
```

### 4. Do some analysis!

Open `final_analysis.ipynb` in Jupyter.
