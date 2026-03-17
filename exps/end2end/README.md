# End-to-End Deduplication Experiments

Fair end-to-end benchmarks comparing FastSketchLSH and Rensa using standard MinHash. Both engines start from the same tokenized documents; dataset loading and preprocessing are excluded from engine time. Measurements include sketching, LSH build, and querying so readers can inspect true wall-clock performance.

> **Why not Rensa's rho mode?** Rensa also exposes a "rho" sketch path (`digest_matrix_from_token_sets_rho`) that aggressively sub-samples a small number of tokens per document instead of hashing all of them. This dramatically cuts sketch time but sacrifices duplicate-detection accuracy. Comparing rho-mode Rensa against standard-MinHash FastSketch would not be an apples-to-apples comparison, so it is excluded from all benchmarks here.

## Speed Highlights

Fixed parameters: `bands=8, rows=16, num_perm=128, threshold=0.8`. All numbers are 3–5 run medians.

PINECONE and SHUYUEJ were measured on Apple Silicon (arm64, 16 GB). BOOKS3 documents are full-length books whose tokenized pickle alone exceeds 3 GB; the working set requires 64+ GB RAM, so it was measured on an AMD EPYC 7352 server (x86_64, 200 GB). See `docs/books3-server-experiment-guide.md` for server setup.

### Single-Thread (`threads=1`)

| Dataset | Engine | Sketch (s) | Build (s) | Query (s) | Total (s) | Sketch Speedup | Total Speedup |
|---------|--------|------------|-----------|-----------|-----------|----------------|---------------|
| PINECONE (100K docs) | Rensa | 0.879 | 0.016 | 0.000 | 0.895 | — | — |
| PINECONE | FastSketchLSH | 0.515 | 0.093 | 0.036 | 0.644 | **1.71x** | **1.39x** |
| SHUYUEJ (37.8K docs) | Rensa | 0.906 | 0.005 | 0.000 | 0.911 | — | — |
| SHUYUEJ | FastSketchLSH | 0.363 | 0.032 | 0.012 | 0.407 | **2.50x** | **2.24x** |
| BOOKS3 (15.4K docs) | Rensa | 95.915 | 0.005 | 0.003 | 95.923 | — | — |
| BOOKS3 | FastSketchLSH | 28.440 | 0.008 | 0.007 | 28.455 | **3.37x** | **3.37x** |

### Multi-Thread (8 threads)

| Dataset | Engine | Sketch (s) | Build (s) | Query (s) | Total (s) | Sketch Speedup | Total Speedup |
|---------|--------|------------|-----------|-----------|-----------|----------------|---------------|
| PINECONE (100K docs) | Rensa | 0.337 | 0.015 | 0.000 | 0.352 | — | — |
| PINECONE | FastSketchLSH | 0.155 | 0.023 | 0.007 | 0.185 | **2.17x** | **1.90x** |
| SHUYUEJ (37.8K docs) | Rensa | 0.293 | 0.005 | 0.000 | 0.298 | — | — |
| SHUYUEJ | FastSketchLSH | 0.115 | 0.007 | 0.002 | 0.124 | **2.55x** | **2.40x** |

### Summary

| Dataset | threads=1 | | threads=8 | |
|---------|-----------|-------|-----------|-------|
| | Sketch | Total | Sketch | Total |
| PINECONE | 1.71x | 1.39x | 2.17x | 1.90x |
| SHUYUEJ | 2.50x | 2.24x | 2.55x | 2.40x |
| BOOKS3 | 3.37x | 3.37x | — | — |

FastSketchLSH is **1.4–3.4x faster** than Rensa across all configurations when both use standard MinHash. The speedup is largest on BOOKS3 where documents are long and sketching dominates total time.

> **TODO — BOOKS3 re-benchmark:** The BOOKS3 numbers above were collected with an older build before the thread-aware `sketch_batch` dispatch landed. The current code parallelises both hashing and sketching via OpenMP, which produced significant gains on PINECONE and SHUYUEJ. We expect BOOKS3 speedups to improve as well, especially at 8 threads where long documents benefit most from parallel sketching. Plan: re-run `bench_fair.py --datasets BOOKS3 --threads 1 8` on the AMD EPYC server with the latest build and update these tables.

### FastSketch Thread Scaling (BOOKS3)
- Total time drops from `25.442s` at one thread to `12.921s` at eight threads, with sketching dominating the gains.
- Build and query stages remain sub-millisecond relative to sketching, so the curve follows sketch throughput.

![FastSketch thread scaling on BOOKS3](results/fastsketch_thread_scaling_BOOKS3.png)

## Reproducing the Experiments

### Environment Setup
- Install the native FastSketch extension: `cd fastsketchlsh_ext && pip install .`
- Install Python dependencies: `pip install -r requirements.txt`

### Fair Benchmark (recommended)

`bench_fair.py` runs both engines in subprocess isolation. This is required because Rensa's Rayon thread pool initializes once per process and cannot be reconfigured — so each `(dataset, threads)` combination spawns a fresh subprocess with `RAYON_NUM_THREADS` pinned.

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
