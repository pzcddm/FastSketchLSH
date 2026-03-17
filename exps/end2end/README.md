# End-to-End Deduplication Experiments

Fair end-to-end benchmarks comparing FastSketchLSH and Rensa using standard MinHash. Both engines start from the same tokenized documents; dataset loading and preprocessing are excluded from engine time. Measurements include sketching, LSH build, and querying so readers can inspect true wall-clock performance.

> **Why not Rensa's rho mode?** Rensa also exposes a `rho` sketch path (`digest_matrix_from_token_sets_rho`) that aggressively samples only a small subset of tokens per document instead of hashing them all. If a reader is comfortable with that trade-off, we would generally expect rho-mode Rensa to be faster than the standard-MinHash path compared here. FastSketchLSH does not currently implement an equally aggressive token-sampling scheme, so comparing rho-mode Rensa against standard FastSketch would not be an apples-to-apples benchmark. For readers who are happy to use token sampling and accept the associated accuracy trade-off, Rensa's rho path is a sensible option to evaluate directly; we simply leave it out of the tables below because it targets a different point on the speed/accuracy curve.

## Speed Highlights

Fixed parameters: `bands=8, rows=16, num_perm=128, threshold=0.8`.

All datasets measured on AMD EPYC 7352 (x86_64, 200 GB) with Rensa 0.4.0. See `docs/books3-server-experiment-guide.md` for server setup.

Note on the current code path:

- The benchmark tables below were collected before FastSketch's true LSH one-shot path landed.
- Current FastSketch code uses `insert_and_query_duplicates(...)` when `threads=1`, which fuses LSH build and duplicate-flagging into one pass.
- `threads>1` still uses the older `insert(...)` + `duplicates(...)` path because the current one-shot implementation is row-serial.
- Re-run medians on the target machine before treating the single-thread build/query rows below as the new baseline.

### Single-Thread (`threads=1`)

| Dataset | Engine | Sketch (s) | Build (s) | Query (s) | Total (s) | Sketch Speedup | Total Speedup |
|---------|--------|------------|-----------|-----------|-----------|----------------|---------------|
| PINECONE (100K docs) | Rensa | 1.593 | 0.064 | 0.000 | 1.657 | — | — |
| PINECONE | FastSketchLSH | 1.423 | 0.360 | 0.172 | 1.954 | **1.12x** | 0.85x |
| SHUYUEJ (37.8K docs) | Rensa | 1.252 | 0.011 | 0.000 | 1.263 | — | — |
| SHUYUEJ | FastSketchLSH | 0.799 | 0.114 | 0.067 | 0.980 | **1.57x** | **1.29x** |
| BOOKS3 (5.4K docs) | Rensa | 36.190 | 0.002 | 0.000 | 36.193 | — | — |
| BOOKS3 | FastSketchLSH | 14.843 | 0.014 | 0.006 | 14.863 | **2.44x** | **2.44x** |

### Multi-Thread (8 threads)

| Dataset | Engine | Sketch (s) | Build (s) | Query (s) | Total (s) | Sketch Speedup | Total Speedup |
|---------|--------|------------|-----------|-----------|-----------|----------------|---------------|
| PINECONE (100K docs) | Rensa | 1.085 | 0.040 | 0.000 | 1.125 | — | — |
| PINECONE | FastSketchLSH | 0.786 | 0.097 | 0.026 | 0.910 | **1.38x** | **1.24x** |
| SHUYUEJ (37.8K docs) | Rensa | 0.842 | 0.012 | 0.000 | 0.854 | — | — |
| SHUYUEJ | FastSketchLSH | 0.658 | 0.038 | 0.017 | 0.713 | **1.28x** | **1.20x** |
| BOOKS3 (5.4K docs) | Rensa | 24.605 | 0.002 | 0.000 | 24.607 | — | — |
| BOOKS3 | FastSketchLSH | 14.191 | 0.006 | 0.003 | 14.199 | **1.73x** | **1.73x** |

### Summary

| Dataset | threads=1 | | threads=8 | |
|---------|-----------|-------|-----------|-------|
| | Sketch | Total | Sketch | Total |
| PINECONE | 1.12x | 0.85x | 1.38x | 1.24x |
| SHUYUEJ | 1.57x | 1.29x | 1.28x | 1.20x |
| BOOKS3 | 2.44x | 2.44x | 1.73x | 1.73x |

In the AMD server snapshot above, FastSketchLSH sketch is **1.1–2.4x faster** than Rensa across all configurations. End-to-end total speedup ranges from **1.2–2.4x** on datasets where sketching dominates (SHUYUEJ, BOOKS3). On PINECONE at single-thread, the pre-one-shot FastSketch build+query overhead (0.53s vs 0.06s) offsets the sketch advantage; the current code is expected to improve this case because `threads=1` now uses true LSH one-shot duplicate flagging.

### Latest Local Validation After True One-Shot

This is a local Apple Silicon rerun used to validate the new code path, not a replacement for the AMD EPYC tables above.

| Dataset | threads | FastSketch Sketch (s) | FastSketch Build (s) | FastSketch Query (s) | FastSketch Total (s) | Rensa Total (s) | Total Speedup |
|---------|---------|-----------------------|----------------------|----------------------|----------------------|-----------------|---------------|
| PINECONE | 1 | 0.500 | 0.082 | 0.000 | 0.582 | 0.891 | **1.53x** |
| PINECONE | 8 | 0.162 | 0.021 | 0.007 | 0.190 | 0.319 | **1.68x** |
| SHUYUEJ | 1 | 0.364 | 0.021 | 0.000 | 0.385 | 0.910 | **2.37x** |
| SHUYUEJ | 8 | 0.121 | 0.007 | 0.002 | 0.130 | 0.325 | **2.49x** |

Targeted LSH-only check on the same single-thread sketch matrices:

| Dataset | Old Build+Query (s) | New One-Shot (s) | Speedup | Flags Equal |
|---------|----------------------|------------------|---------|-------------|
| PINECONE | 0.155 | 0.106 | **1.47x** | yes |
| SHUYUEJ | 0.045 | 0.019 | **2.32x** | yes |

### FastSketch Thread Scaling (BOOKS3)
- `threads=1` uses a fused chunked hash+sketch fast path (`init.cpp:466`) that bypasses OpenMP entirely and reuses parent buffers. `threads>1` falls through to `sketch_batch_flat_bytes`, which spawns per-thread worker copies and re-partitions work via `omp for schedule(static)`. For BOOKS3's 5.4K very-long documents, the worker-clone overhead and memory-bandwidth contention outweigh parallel gains, so single-thread is fastest.
- Build and query stages remain sub-millisecond relative to sketching, so the curve follows sketch throughput.

![FastSketch thread scaling on BOOKS3](results/fastsketch_thread_scaling_BOOKS3.png)

## Reproducing the Experiments

### Environment Setup
- Install the native FastSketch extension: `cd fastsketchlsh_ext && pip install .`
- Install Python dependencies: `pip install -r requirements.txt`

### Fair Benchmark (recommended)

`bench_fair.py` runs both engines in subprocess isolation. This is required because Rensa's Rayon thread pool initializes once per process and cannot be reconfigured — so each `(dataset, threads)` combination spawns a fresh subprocess with `RAYON_NUM_THREADS` pinned.

Current FastSketch adapter behavior:

- `threads=1`: use `insert_and_query_duplicates(...)` for true LSH one-shot duplicate flagging
- `threads>1`: use `insert(...)` then `duplicates(...)`

```bash
# Full benchmark: PINECONE + SHUYUEJ at threads=1 and threads=8
python -m exps.end2end.bench_fair --datasets PINECONE SHUYUEJ --threads 1 8

# Single dataset, single thread count
python -m exps.end2end.bench_fair --datasets PINECONE --threads 1

# Only run one engine
python -m exps.end2end.bench_fair --engine fastsketch
```

Results are printed as plain-text tables and saved as JSON to `exps/end2end/results/fair_benchmark_YYYY-MM-DD.json`.

### Manual Runs
1. Launch the driver with your chosen engine and dataset:
   ```bash
   python -m exps.end2end.run --engine fastsketch --dataset PINECONE
   ```
   Valid `--engine` values are `fastsketch` and `rensa`, and datasets accept either enum (`PINECONE`, `SHUYUEJ`, etc.) or full HuggingFace IDs.
2. The script shuffles the dataset, tokenises documents into 3-gram shingles, sketches them, builds the LSH index, and reports duplicate counts plus per-stage timings.
3. If direct HuggingFace access is slow, add `--use-hf-mirror` or `--hf-endpoint https://hf-mirror.com`.

### Automation Scripts
- `exps/end2end/run_all_comparisons.sh` runs both engines on the configured dataset list.
- `exps/end2end/fastsketch_thread_sweep.sh DATASET` benchmarks FastSketch with thread counts `{1, 2, 4, 8}`, logging JSONL timings and saving the scaling plot in `exps/end2end/results/`.

### Working With Cached Data
- Tokenised datasets live under `exps/end2end/processed_ds/` once generated; subsequent runs reuse them to skip downloads.
- Delete the cached files if you need to regenerate preprocessing (e.g. for a new split or n-gram size).
