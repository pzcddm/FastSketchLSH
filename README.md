# FastSketchLSH

## Introduction
FastSketchLSH delivers a Python-first package that wraps a high-performance C++/SIMD implementation of Fast Similarity Sketch (see Dahlgaard et al., FOCS'17 [arXiv:1704.04370](https://arxiv.org/abs/1704.04370) for the underlying algorithm). The goal is to make Jaccard estimation and locality-sensitive hashing (LSH) practical for large dataset deduplication.

![FastSimilaritySketch throughput advantage](exps/sketch/records/minhash_QPS_vs_k_n1600.png)

| Dataset | Engine | Sketch (s) | Build (s) | Query (s) | Total (s) | FastSketchLSH Sketch Speedup | FastSketchLSH Total Speedup |
|---------|--------|------------|-----------|-----------|-----------|--------------------------|------------------------|
| BOOKCORPUSOPEN | rensa | 198.545 | 0.026 | 0.018 | 198.589 | - | - |
| BOOKCORPUSOPEN | fastsketchlsh | 55.280 | 0.039 | 0.031 | 55.350 | 3.59× | 3.59× |
| BOOKS3 | rensa | 95.915 | 0.005 | 0.003 | 95.923 | - | - |
| BOOKS3 | fastsketchlsh | 28.440 | 0.008 | 0.007 | 28.455 | 3.37× | 3.37× |
| PINECONE | rensa | 3.929 | 0.141 | 0.153 | 4.223 | - | - |
| PINECONE | fastsketchlsh | 1.521 | 0.249 | 0.396 | 2.166 | 2.58× | 1.95× |
| SHUYUEJ | rensa | 3.749 | 0.037 | 0.044 | 3.830 | - | - |
| SHUYUEJ | fastsketchlsh | 1.132 | 0.093 | 0.121 | 1.346 | 3.31× | 2.85× |

### Headline Results
- `FastSimilaritySketch` maintains **sub-millisecond** sketch times even when each set holds **1 600 tokens**, keeping the absolute Jaccard error around **0.03–0.06**.
- At the sketch level, FastSimilaritySketch stays **200×–990× faster** than `datasketch` MinHash and still **8×–23×** faster than Rensa’s `CMinHash`/`RMinHash`, while matching their accuracy—these gains matter most for large documents.
- End-to-end deduplication experiments show FastSketchLSH is typically **~2×–3.6× faster** than Rensa in single-thread runs.
- Ground-truth comparisons confirm FastSketchLSH matches or slightly exceeds the deduplication accuracy of both Rensa and datasketch.

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

sketcher = FastSimilaritySketch(sketch_size=256)
sig_a = sketcher.sketch(list_a)
sig_b = sketcher.sketch(list_b)

estimated = estimate_jaccard(sig_a, sig_b)
print(f"Estimated Jaccard similarity: {estimated:.4f}")
```


### Deduplication with LSH
This end-to-end sample downloads a small slice of Hugging Face’s `lucadiliello/bookcorpusopen` corpus, sketches every document with `k=128`, and groups the signatures into `16` bands. Sketching each document costs `O(n + k log k)` time with `O(k)` space, while an LSH probe runs in `O(k + c)` where `c` is the number of retrieved candidates.

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

sketcher = FastSimilaritySketch(sketch_size=128, seed=42)
# Use batch mode for faster sketching (much faster than one-by-one)
sketch_matrix = sketcher.sketch_batch(token_sets)

lsh = LSH(num_perm=128, num_bands=16)
lsh.build_from_batch(sketch_matrix)

doc_idx = 0
candidates = lsh.query_candidates(sketch_matrix[doc_idx])
print(f"Candidates for {doc_idx}:", candidates)

dup_flags = [1 if len(lsh.query_candidates(row)) > 1 else 0 for row in sketch_matrix]
print("Duplicate flags:", dup_flags)
print("Total duplicates detected:", sum(dup_flags))
```

## Experiment Summaries
- **Sketch microbenchmarks (`exps/sketch/`)**: Full write-up, CSVs, and plotting helpers demonstrating latency and accuracy versus `datasketch` and Rensa baselines. Reproduction steps live in `exps/sketch/README.md`.
- **Ground-truth accuracy (`exps/accuracy/`)**: Jaccard estimation and dedup quality measured against labelled datasets. See `exps/accuracy/README.md` for reproduction commands.
- **End-to-end pipelines (`exps/end2end/`)**: Thread-scaled deduplication sweeps on large corpora, plus scripts for batch comparisons. Details in `exps/end2end/README.md`.

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
