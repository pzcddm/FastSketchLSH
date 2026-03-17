# FastSketchLSH

## Introduction
FastSketchLSH delivers a Python-first package that wraps a high-performance C++/SIMD implementation of Fast Similarity Sketch (see Dahlgaard et al., FOCS'17 [arXiv:1704.04370](https://arxiv.org/abs/1704.04370) for the underlying algorithm). The goal is to make Jaccard estimation and locality-sensitive hashing (LSH) practical for large dataset deduplication.

![FastSimilaritySketch throughput advantage](https://raw.githubusercontent.com/pzcddm/FastSketchLSH/main/exps/sketch/records/minhash_QPS_vs_k_n1600.png)

### Headline Results
- `FastSimilaritySketch` maintains **sub-millisecond** sketch times even when each set holds **1 600 tokens**, keeping the absolute Jaccard error around **0.03–0.06**.
- At the sketch level, FastSimilaritySketch stays **200×–990× faster** than `datasketch` MinHash while matching its accuracy on the included microbenchmarks.
- Ground-truth comparisons confirm FastSketchLSH remains competitive on deduplication quality.

Current package version: `1.0.1`.
This patch release keeps the `v1.0.0` API surface unchanged and adds pytest collection isolation so default test runs stay on the local extension under active development.

## From v0.2.0 to v1.0.0

v1.0.0 keeps the same native kernels but simplifies the public Python API and adds a true LSH one-shot duplicate path. The main user-visible changes are:

- `FastSimilaritySketch(sketch_size=...)` -> `FastSimilaritySketch(size=...)`
- `sketch(...)` -> `__call__(...)`
- `sketch_batch(...)` -> `batch(...)`
- `sketch_batch_flat_csr_prehashed(...)` -> `batch_csr(..., prehashed=True)`
- `build_from_batch(...)` -> `insert(...)`
- `query_candidates(...)` / `batch_query_csr(...)` -> `query(...)`
- New: `insert_and_query_duplicates(...)` for fused LSH build + duplicate flagging

The full upgrade record lives in [`docs/release-notes-v0.2.0-to-v1.0.0.md`](docs/release-notes-v0.2.0-to-v1.0.0.md).
For method-by-method usage examples and input/return-shape details, see [`docs/api-usage.md`](docs/api-usage.md).

Pre-hashed inputs are still supported; they are now exposed through the unified `prehashed=True` flag:

```python
import numpy as np
from FastSketchLSH import FastSimilaritySketch

sketcher = FastSimilaritySketch(size=256, seed=42)

# Single sketch from pre-hashed values (zero-copy from NumPy)
hashes = np.array([0xDEAD, 0xBEEF, 0xCAFE, ...], dtype=np.uint64)
digest = sketcher(hashes, prehashed=True)

# Batch of pre-hashed arrays
batch = [np.array([...], dtype=np.uint64) for _ in range(1000)]
digests = sketcher.batch(batch, prehashed=True, num_threads=8)

# CSR layout for maximum throughput
data = np.array([...], dtype=np.uint64)
indptr = np.array([0, 120, 250, 500], dtype=np.uint64)
digests = sketcher.batch_csr(data, indptr, prehashed=True, num_threads=8)
```

All prehashed paths share the same SIMD-accelerated Round 1 / Round 2 bucket-fill logic and OpenMP batch parallelism as the token-based paths.

## How It Works
- **Fast Similarity Sketching**: SIMD-accelerated permutations compress a set into a fixed-length signature, expected time `O(n + k log k)` with `O(k)` space.
- **Banded LSH**: Signature rows are grouped into bands; items colliding in any band become candidates for deduplication.
- **Python ergonomics**: Thin wrappers expose the C++ core, plus reference implementations of competing sketches for fair comparisons.

## Installation
> **Prerequisite:** Python 3.11 or newer. Support for Python 3.8 and older
> interpreters is on the roadmap.

### PyPI (recommended)
```bash
pip install fastsketchlsh
```

### Build from source
1. Build the native extension:
   ```bash
   cd fastsketchlsh_ext
   pip install .
   ```
   This installs the `FastSketchLSH` Python module with SIMD kernels.
2. Install benchmark utilities (optional for reproducing experiments):
   ```bash
   pip install -r requirements.txt
   ```
3. Activate your environment (e.g. `source .venv/bin/activate`) before running scripts.

## Quick Start
### Sketch two sets and estimate their Jaccard similarity
```python
from FastSketchLSH import FastSimilaritySketch, estimate_jaccard

# Build list_a with 16,000 tokens labeled "a-0" to "a-15999"
# Build list_b with 8,000 overlapping + 8,000 new tokens (true Jaccard = 1/3)
list_a = [f"a-{i}" for i in range(16_000)]
list_b = [f"a-{i}" for i in range(8_000)] + [f"b-{i}" for i in range(8_000)]

sketcher = FastSimilaritySketch(size=256)
sig_a = sketcher(list_a)
sig_b = sketcher(list_b)

estimated = estimate_jaccard(sig_a, sig_b)
print(f"Estimated Jaccard similarity: {estimated:.4f}")
```


### Deduplication with LSH
This end-to-end sample downloads a small slice of Hugging Face’s `lucadiliello/bookcorpusopen` corpus, sketches every document with `k=128`, and groups the signatures into `16` bands. Sketching each document costs `O(n + k log k)` time with `O(k)` space, while an LSH probe runs in `O(k + c)` where `c` is the number of retrieved candidates. For pure duplicate-flagging, `insert_and_query_duplicates(...)` fuses LSH build and duplicate detection into one pass.

```python
from __future__ import annotations

from datasets import load_dataset
from FastSketchLSH import FastSimilaritySketch, LSH


def tokenize(text: str) -> list[str]:
    return sorted({token for token in text.lower().split() if token})


# Here, 'train[:2048]' tells Hugging Face Datasets to select only the first 2048 rows from the 'train' split.
dataset = load_dataset(
    "lucadiliello/bookcorpusopen",
    split="train[:2048]")

texts = [row["text"] for row in dataset if row.get("text")]
token_sets = [tokenize(text) for text in texts]

sketcher = FastSimilaritySketch(size=128, seed=42)
sketch_matrix = sketcher.batch(token_sets)

lsh = LSH(num_perm=128, num_bands=16)
# One-shot path: insert the batch and return duplicate flags immediately.
dup_flags = lsh.insert_and_query_duplicates(sketch_matrix).tolist()

doc_idx = 0
candidates = lsh.query(sketch_matrix[doc_idx])
print(f"Candidates for {doc_idx}:", candidates)

print("Duplicate flags:", dup_flags)
print("Total duplicates detected:", sum(dup_flags))
```

## Experiment Summaries
- **Sketch microbenchmarks (`exps/sketch/`)**: Full write-up, CSVs, and plotting helpers demonstrating latency and accuracy versus standard MinHash baselines. Reproduction steps live in `exps/sketch/README.md`.
- **Ground-truth accuracy (`exps/accuracy/`)**: Jaccard estimation and dedup quality measured against labelled datasets. See `exps/accuracy/README.md` for reproduction commands.
- **End-to-end pipelines (`exps/end2end/`)**: Thread-scaled deduplication sweeps on large corpora, plus scripts for batch comparisons. The current single-thread FastSketch path uses true LSH one-shot duplicate flagging. Details in `exps/end2end/README.md`.

Each experiment directory includes figures, CSV outputs, and exact command lines so you can replicate every result.

## Key Points
- FastSketchLSH packages a SIMD-backed sketch with Python convenience wrappers.
- Headline benchmarks show up to **990×** throughput gains over classic MinHash at comparable accuracy.
- Ready-to-run examples cover sketching, LSH-based deduplication, and full dataset experiments.
- For deeper reproduction details, consult the README in each experiment subdirectory.

## Future Work
- A MapReduce/Spark demo to deduplicate large datasets in distributed systems.
- A friendlier Python interface aligned with `datasketch` ergonomics.

## License
MIT. Research and educational use welcome.
