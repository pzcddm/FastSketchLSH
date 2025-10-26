# Deduplication Comparison Experiments

This module compares different deduplication engines (FastSketchLSH and Rensa) on HuggingFace datasets. The `run.py` entry point handles dataset preparation, sketch generation, LSH queries, and prints timing breakdowns so you can record end-to-end performance.

## Prerequisites
- Install the native FastSketch extension: `cd fastsketchlsh_ext && pip install .`
- Install Python dependencies once: `pip install -r requirements.txt`
- (Optional) Choose a writable cache directory if you want to reuse HuggingFace downloads (`data/huggingface_cache` is used by default).

## Running an Experiment
1. From the project root run the driver script, selecting an engine and dataset:
   ```bash
   python -m comparison.run --engine fastsketch --dataset PINECONE --split train
   ```
   Valid `--engine` values are `fastsketch` and `rensa`. You can reference datasets either by the short enum (`PINECONE`, `HARIOM`, `SHUYUEJ`) or the full HuggingFace ID.
2. The script shuffles the dataset, tokenises each record into 3-gram shingles, sketches the token sets, builds the LSH index, and queries for duplicates.
3. Outputs include duplicate counts per query mode (e.g. `batch_duplicates=...`) and per-stage timings.
4. If direct access to HuggingFace is slow or blocked, append `--use-hf-mirror` or supply a custom endpoint with `--hf-endpoint https://hf-mirror.com`.

## Recording Deduplication Times
- The final line prints a comma-separated summary such as `FastSketchDeduplicator: dataset=0.842, sketch=0.215, build=0.051, query_batch=0.041`.
- Capture these values together with the command line you ran for a reproducible timing log.
- To avoid resketching when changing query parameters, use `--save-artifacts /path/to/sketch.npy` and later `--load-artifacts ...` (FastSketch engine only).

## Working With Cached Data
- Preprocessed token sets are stored under `comparison/processed_ds/` once generated, so repeated runs skip HuggingFace downloads and tokenisation.
- Clear files in that directory if you need to regenerate preprocessing with different options (e.g. a new `--split` or n-gram size).
