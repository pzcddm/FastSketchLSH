# Fair Benchmark Sketch Catch-Up Plan

## Purpose

This document is for the next optimization pass whose goal is:

- keep the benchmark strictly fair against Rensa
- focus on the real remaining bottleneck, which is FastSketch `sketch`
- record every material code change and benchmark result in one place

This file should be treated as the execution checklist for an AI coding agent.

## Fairness Contract

Default benchmark mode must follow all rules below:

1. Do not use `--use-prehashed-csr`
2. Start both engines from the same `token_sets`
3. Count token hashing inside `sketch` for both engines
4. Exclude dataset download and preprocessing from engine time for both engines
5. Keep `num_perm=128`, `bands=8`, `rows=16` unless a section explicitly says otherwise
6. Thread control: `--threads N` must limit BOTH engines. For Rensa this means setting `RAYON_NUM_THREADS=N` (handled by the adapter). Default (no `--threads`) lets both engines use all available cores.

## Current Strict Fair Baseline

Measured on 2026-03-13 with corrected Rensa thread control and thread-aware sketch dispatch, 3-5 run medians:

### Single-thread (`--threads 1`, both engines limited)

| Dataset | Engine | Sketch (s) | Build (s) | Query (s) | Total (s) |
|---|---:|---:|---:|---:|---:|
| PINECONE | fastsketch | 0.515 | 0.093 | 0.036 | 0.644 |
| PINECONE | rensa | 0.879 | 0.016 | 0.000 | 0.895 |
| SHUYUEJ | fastsketch | 0.363 | 0.032 | 0.012 | 0.407 |
| SHUYUEJ | rensa | 0.906 | 0.005 | 0.000 | 0.911 |

### Auto-threads (default)

| Dataset | Engine | Sketch (s) | Build (s) | Query (s) | Total (s) |
|---|---:|---:|---:|---:|---:|
| PINECONE | fastsketch | 0.155 | 0.023 | 0.007 | 0.185 |
| PINECONE | rensa | 0.337 | 0.015 | 0.000 | 0.352 |
| SHUYUEJ | fastsketch | 0.115 | 0.007 | 0.002 | 0.124 |
| SHUYUEJ | rensa | 0.293 | 0.005 | 0.000 | 0.298 |

Interpretation:

- FastSketch is faster in all configurations
- Single-thread: FastSketch `1.39x` faster on PINECONE, `2.24x` faster on SHUYUEJ
- Auto-threads: FastSketch `1.90x` faster on PINECONE, `2.40x` faster on SHUYUEJ
- The dominant improvement came from thread-aware dispatch in `sketch_batch`: multi-thread now falls through to the ptrs/lengths path where `sketch_batch_flat_bytes` parallelizes both hashing and sketching via OpenMP

## Files To Inspect Before Making Changes

Local FastSketch files:

- `exps/end2end/run.py`
- `exps/end2end/fastsketch_deduplicator.py`
- `exps/end2end/rensa_deduplicator.py`
- `fastsketchlsh_ext/include/fastsketch.h`
- `fastsketchlsh_ext/cpp/fastsketch.cpp`
- `fastsketchlsh_ext/cpp/init.cpp`
- `fastsketchlsh_ext/include/LSH.h`
- `fastsketchlsh_ext/cpp/LSH.cpp`
- `test/test_lsh_duplicate_flags_fastpath.py`
- `test/test_prehashed_consistency.py`

Upstream Rensa files to study before changing FastSketch:

- `benchmarks/full_benchmark.py`
- `src/rminhash/py.rs`
- `src/rminhash/pipeline.rs`
- `src/lsh/py.rs`
- `src/lsh/one_shot.rs`
- `src/py_input.rs`
- `Cargo.toml`

## Already Implemented: Check Before Redoing

The next optimization pass must verify whether each item below already solves part of the problem.

1. Duplicate-flag one-shot path already exists
- FastSketch already has `batch_query_duplicate_flags(...)`
- This reduced query cost and is not the main remaining gap

2. Prehashed APIs already exist
- `sketch_prehashed(...)`
- `sketch_batch_prehashed(...)`
- `sketch_batch_flat_csr_prehashed(...)`
- These are useful library interfaces, but they are not part of the current benchmark target

3. Prehashed dataset cache already exists
- It is opt-in via `--use-prehashed-csr`
- Do not re-enable it by default

4. Rensa adapter has already been updated
- The benchmark already uses matrix / one-shot Rensa APIs when available
- Do not compare against an old per-document Python-loop wrapper

5. LSH query is not the primary bottleneck anymore
- The next pass should be sketch-focused

6. The main sketch-side catch-up items are already in code
- Default token hash is already `fxhash64`
- Dedicated `sketch_batch_str_lists(...)` exists as a library API
- `sketch_kernel_direct` and the single-thread OpenMP bypass already exist
- Do not put these items back into the future-task list unless profiling proves they are incomplete or wrong

7. Thread-aware dispatch in `sketch_batch` (the key fix for multi-threaded performance)
- `num_threads == 1`: chunked fused path (hash under GIL + serial sketch)
- `num_threads != 1`: ptrs/lengths extraction under GIL → `sketch_batch_flat_bytes` parallelizes both hashing and sketching via OpenMP
- This is what Rensa's benchmark calls; fixing it eliminates the "11.92x slower" claim
- End-to-end adapter (`fastsketch_deduplicator.py`) uses `sketch_batch` directly

## Priority Task List

### P0. Freeze the strict fair benchmark harness

Reason:

- All optimization work is meaningless if the benchmark silently changes semantics

Change:

- Keep the current default benchmark as the strict fair path
- Add a small script or documented command set that always runs:
  - `PINECONE`
  - `SHUYUEJ`
  - `bands=8`
  - `rows=16`
  - `threads=1`
  - no `--use-prehashed-csr`

Expected impact:

- No direct speedup
- Prevents invalid performance claims

### P1. Profile the sketch path into sub-stages

Reason:

- The current sketch path already beats Rensa fairly, but more tuning should start from measurement instead of guesswork
- The repository already switched the default token hash to `fxhash64`
- The dedicated `sketch_batch_str_lists(...)` path is already in use
- The remaining question is where the current fused path still spends time on real datasets

Change:

- Add fine-grained profiling around the current FastSketch fused string batch path
- Report at least:
  - Python traversal / marshaling time
  - token hashing time
  - CSR/prehashed kernel execution time
- Only propose a new hash or layout change if the profile shows a real remaining hotspot

Target files:

- `fastsketchlsh_ext/include/fastsketch.h`
- `fastsketchlsh_ext/cpp/fastsketch.cpp`
- `fastsketchlsh_ext/cpp/init.cpp`

Expected impact:

- No direct speedup
- High value for prioritization

### P2. Reuse scratch buffers in the current native batch path

Reason:

- The dedicated batch string path already exists
- Remaining overhead is more likely to come from temporary allocation, scratch setup, and per-batch teardown

Change:

- Reuse worker-local or thread-local scratch buffers
- Reduce per-document and per-batch temporary allocation in the current fused string path and batch kernel
- Measure the effect before and after with the strict fair harness

Target files:

- `fastsketchlsh_ext/cpp/fastsketch.cpp`
- `fastsketchlsh_ext/include/fastsketch.h`

Expected impact:

- total: `5%–12%`

### P3. Investigate compact sketch representation for LSH-only usage

Reason:

- Rensa stores digest matrices compactly as `u32`
- FastSketch still moves `uint64` sketch matrices through Python and LSH
- This likely increases memory traffic

Change:

- Investigate whether a more compact representation can be used in the end-to-end dedup path
- Only ship this if correctness and API clarity remain acceptable

Target files:

- `fastsketchlsh_ext/include/fastsketch.h`
- `fastsketchlsh_ext/cpp/fastsketch.cpp`
- `fastsketchlsh_ext/cpp/init.cpp`
- `fastsketchlsh_ext/include/LSH.h`
- `fastsketchlsh_ext/cpp/LSH.cpp`

Expected impact:

- total: `5%–15%`
- Higher engineering risk than P1–P2

### P4. Maintain, but do not over-focus on, LSH query

Reason:

- LSH query already has the duplicate-flag fast path
- It is no longer the main reason we lose or win in the strict fair benchmark

Change:

- Keep correctness tests green
- Only touch this area if new profiling proves a regression

Target files:

- `fastsketchlsh_ext/include/LSH.h`
- `fastsketchlsh_ext/cpp/LSH.cpp`
- `test/test_lsh_duplicate_flags_fastpath.py`

Expected impact:

- usually `<5%`

## Recommended Execution Order

1. P0
2. P1
3. P2
4. P3
5. rerun strict fair benchmark
6. only then consider P4

## Update Protocol For The AI Coder

For every meaningful optimization attempt:

1. Re-read the files listed in `Files To Inspect Before Making Changes`
2. Check whether the intended optimization already exists in some form
3. Record the exact code changes in the log below
4. Record the exact benchmark command used
5. Record before/after numbers for `PINECONE` and `SHUYUEJ`
6. If a result is worse or inconclusive, record that too

Do not silently replace old numbers.

## Working Log

Append new entries at the top.

### Template

#### YYYY-MM-DD HH:MM UTC | Short title

- Mode: `Strict Fair`
- Files changed:
- Rationale:
- Command(s):
- Before:
- After:
- Conclusion:

### 2026-03-13 | Thread-aware dispatch in sketch_batch + Rensa benchmark analysis

- Mode: `Strict Fair`
- Files changed:
  - `fastsketchlsh_ext/cpp/init.cpp` — `sketch_batch` string path now dispatches by thread count:
    - `num_threads == 1`: chunked fused path (hash under GIL + serial sketch kernel)
    - `num_threads != 1`: fall through to ptrs/lengths path → `sketch_batch_flat_bytes` parallelizes both hashing AND sketching via OpenMP
  - `fastsketchlsh_ext/cpp/fastsketch.cpp` — `sketch_batch_flat_csr_prehashed` bypasses OpenMP team creation when threads==1 (serial path uses `this->` members directly, no worker copy)
  - `exps/end2end/fastsketch_deduplicator.py` — uses `sketch_batch` (with thread-aware dispatch) instead of `sketch_batch_str_lists`
  - `docs/performance-optimization-notes.md` — added full analysis of Rensa's "11.92x faster" claim
- Rationale:
  1. The chunked fused path called `sketch_one_prehashed()` in a serial loop, completely ignoring `num_threads`. At threads=8, FastSketch sketched single-threaded while Rensa's Rayon used all 8 threads. This is the root cause of Rensa's "11.92x faster" claim.
  2. Profiling showed 73% of single-threaded sketch time is GIL-held string hashing (0.334s out of 0.46s). The chunked path kept hashing serial even with threads>1. The ptrs/lengths fallthrough path moves hashing into the OpenMP parallel region.
  3. Rensa's benchmark also uses `batch_query_csr` instead of the faster `batch_query_duplicate_flags`.
- Command(s):
  - `python -m exps.end2end.run --engine {fastsketch|rensa} --dataset {PINECONE|SHUYUEJ} --bands 8 --rows 16 --threads 1`
  - `python -m exps.end2end.run --engine {fastsketch|rensa} --dataset {PINECONE|SHUYUEJ} --bands 8 --rows 16`
- Before (old chunked path, all thread counts):
  - Auto-threads PINECONE: FastSketch `0.221s`, Rensa `0.363s` (1.64x)
  - Auto-threads SHUYUEJ: FastSketch `0.125s`, Rensa `0.322s` (2.58x)
- After (thread-aware dispatch):
  - Auto-threads PINECONE: FastSketch `0.185s`, Rensa `0.352s` (**1.90x**)
  - Auto-threads SHUYUEJ: FastSketch `0.124s`, Rensa `0.298s` (**2.40x**)
  - Single-thread PINECONE: FastSketch `0.644s`, Rensa `0.895s` (**1.39x**)
  - Single-thread SHUYUEJ: FastSketch `0.407s`, Rensa `0.911s` (**2.24x**)
- Conclusion:
  - FastSketch wins in every configuration. The auto-threads sketch speedup (0.155s vs 0.337s on PINECONE) comes from parallelizing both hashing and sketching.
  - Rensa's "11.92x faster" claim was based on benchmarking a FastSketch build where `sketch_batch` was accidentally single-threaded for string input. Full analysis in `docs/performance-optimization-notes.md`.

### 2026-03-13 | Refresh strict fair baseline after rerun

- Mode: `Strict Fair`
- Files changed: documentation only
- Rationale:
  1. Revalidate the current single-thread and auto-thread baselines with corrected Rensa thread control.
  2. Keep the docs aligned with the actual current numbers instead of older medians from an earlier run.
- Command(s):
  - `.venv/bin/python -m exps.end2end.run --engine {fastsketch|rensa} --dataset {PINECONE|SHUYUEJ} --bands 8 --rows 16 --threads 1`
  - `.venv/bin/python -m exps.end2end.run --engine {fastsketch|rensa} --dataset {PINECONE|SHUYUEJ} --bands 8 --rows 16`
- Before:
  - Earlier medians in this file no longer matched a fresh local rerun.
- After:
  - Single-thread:
    - `PINECONE`: FastSketch `0.719s`, Rensa `0.892s`
    - `SHUYUEJ`: FastSketch `0.466s`, Rensa `0.914s`
  - Auto-threads:
    - `PINECONE`: FastSketch `0.221s`, Rensa `0.363s`
    - `SHUYUEJ`: FastSketch `0.125s`, Rensa `0.322s`
- Conclusion:
  - FastSketch still wins fairly.
  - Future work should focus on profiling and tightening the current sketch path, not on rebuilding already-landed optimizations.

### 2026-03-13 | Fix Rensa thread control + use sketch_batch_str_lists + OMP bypass

- Mode: `Strict Fair`
- Files changed:
  - `exps/end2end/rensa_deduplicator.py` — accept `num_threads`, set `RAYON_NUM_THREADS`
  - `exps/end2end/run.py` — pass `sketch_threads` to Rensa adapter
  - `exps/end2end/fastsketch_deduplicator.py` — use `sketch_batch_str_lists` when available
  - `fastsketchlsh_ext/cpp/fastsketch.cpp` — bypass OpenMP parallel region when threads=1
- Rationale:
  1. Rensa uses Rayon for internal parallelism; `--threads 1` only limited FastSketch. This gave Rensa an unfair multi-core advantage. Fixed by setting `RAYON_NUM_THREADS`.
  2. `sketch_batch_str_lists` is ~4% faster than `sketch_batch` in single-thread (hashes all tokens first, one GIL transition vs chunked).
  3. `sketch_batch_flat_csr_prehashed` with `num_threads(1)` still created an OMP team with overhead; bypassed when threads=1.
- Command(s): `python -m exps.end2end.run --engine {fastsketch|rensa} --dataset {PINECONE|SHUYUEJ} --bands 8 --rows 16 --threads 1`
- After (corrected, both engines single-threaded):
  - `PINECONE`: FastSketch `0.719s`, Rensa `0.892s` (fresh rerun median in this file)
  - `SHUYUEJ`: FastSketch `0.466s`, Rensa `0.914s` (fresh rerun median in this file)
- Conclusion: These changes were part of the path that made the strict fair benchmark favorable for FastSketch. They are already landed and should not be re-added as future optimization tasks.
