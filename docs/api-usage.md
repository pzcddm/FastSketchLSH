# FastSketchLSH API Usage

This document covers the current public Python API exposed by `FastSketchLSH` `v1.0.1`.
It focuses on how to call the library in real workloads: sketching token sets, sketching pre-hashed inputs, building LSH indexes, and retrieving duplicate flags.

## Overview

FastSketchLSH exposes two main classes and one helper function:

- `FastSimilaritySketch`
- `LSH`
- `estimate_jaccard`

The common workflow is:

1. Create a `FastSimilaritySketch`.
2. Sketch one row or a batch of rows.
3. Insert the sketch matrix into `LSH`.
4. Query candidates or duplicate flags.

## Data Model

- A single sketch is a 1-D signature of length `size` / `num_perm`.
- Batch sketching returns a `np.ndarray` of shape `(B, size)` with dtype `uint64`.
- `LSH` expects sketch rows whose width equals `num_perm`.
- Row ids in `LSH` are assigned in insertion order, starting from `0`.

## FastSimilaritySketch

### Constructor

```python
from FastSketchLSH import FastSimilaritySketch

sketcher = FastSimilaritySketch(size=128, seed=42)
```

Parameters:

- `size`: sketch width.
- `seed`: random seed used to initialize the sketch permutations.

### Sketch One Row With `__call__`

`FastSimilaritySketch` is callable.
Use it for a single token set.

```python
tokens = ["cat", "dog", "mouse"]
digest = sketcher(tokens)
```

Supported token inputs when `prehashed=False`:

- `list[str]`
- `list[bytes]`
- `list[int]`
- `tuple[int]`
- `np.ndarray[np.uint32]`
- `np.ndarray[np.int32]`

Return value:

- `list[int]` of length `size`

### Sketch Pre-Hashed One-Row Input

Use `prehashed=True` when the row already contains 64-bit token hashes.

```python
import numpy as np

token_hashes = np.array([0xDEAD, 0xBEEF, 0xCAFE], dtype=np.uint64)
digest = sketcher(token_hashes, prehashed=True)
```

Supported inputs when `prehashed=True`:

- `np.ndarray[np.uint64]`
- `np.ndarray[np.int64]`
- `list[int]`

Use this path when:

- token hashing is already done upstream
- the same token hashes are reused across multiple sketch sizes or experiments
- you want to sketch CSR-form batches of hashed tokens

### Batch Sketch With `batch`

Use `batch` when inputs are already grouped row-by-row.

```python
rows = [
    ["cat", "dog", "mouse"],
    ["cat", "dog", "mouse"],
    ["tree", "river", "mountain"],
]

sketches = sketcher.batch(rows, num_threads=8)
```

Return value:

- `np.ndarray` of shape `(B, size)`, dtype `uint64`

Supported batch inputs when `prehashed=False`:

- `list[list[str]]`
- `list[list[bytes]]`
- `list[list[int]]`
- `list[np.ndarray[np.uint32]]`
- `list[np.ndarray[np.int32]]`

Supported batch inputs when `prehashed=True`:

- `list[np.ndarray[np.uint64]]`

Example with pre-hashed rows:

```python
hashed_rows = [
    np.array([1, 2, 3, 4], dtype=np.uint64),
    np.array([5, 6, 7], dtype=np.uint64),
]

sketches = sketcher.batch(hashed_rows, prehashed=True, num_threads=8)
```

Thread control:

- `num_threads=0`: let OpenMP pick the default worker count
- `num_threads=1`: force the single-thread path
- `num_threads>1`: request that many OpenMP workers

### Batch Sketch With `batch_csr`

Use `batch_csr` when rows are already packed into CSR form.

```python
import numpy as np

data = np.array([10, 11, 12, 21, 22, 30], dtype=np.uint32)
indptr = np.array([0, 3, 5, 6], dtype=np.uint64)

sketches = sketcher.batch_csr(data, indptr, num_threads=8)
```

Return value:

- `np.ndarray` of shape `(B, size)`, dtype `uint64`

Parameters:

- `data`: flat token buffer
- `indptr`: row pointer array of length `B + 1`
- `prehashed=False`: `data` must be `np.uint32`
- `prehashed=True`: `data` must be `np.uint64`

Example with pre-hashed CSR input:

```python
hashed_data = np.array([101, 102, 201, 202, 203], dtype=np.uint64)
indptr = np.array([0, 2, 5], dtype=np.uint64)

sketches = sketcher.batch_csr(
    hashed_data,
    indptr,
    prehashed=True,
    num_threads=8,
)
```

## estimate_jaccard

Use `estimate_jaccard` on two 1-D sketches with the same width.

```python
from FastSketchLSH import estimate_jaccard

score = estimate_jaccard(sketch_a, sketch_b)
```

Return value:

- `float`

## LSH

### Constructor

```python
from FastSketchLSH import LSH

lsh = LSH(num_perm=128, num_bands=16, num_threads=0)
```

Parameters:

- `num_perm`: sketch width expected by the index
- `num_bands`: number of LSH bands
- `seed`: optional internal band-hash seed
- `num_threads`: OpenMP worker count for LSH build/query paths

Requirement:

- `num_perm` must match the sketch width produced by `FastSimilaritySketch`

### Insert Sketches With `insert`

Insert appends rows to the current index.

```python
lsh.insert(sketches)
```

Supported inputs:

- `np.ndarray` with shape `(B, num_perm)` and dtype `uint64`
- `list[np.ndarray[np.uint64]]`
- `list[list[int]]`

### Query Candidates With `query`

#### Query One Sketch

```python
candidates = lsh.query(sketches[0])
```

Return value:

- `list[int]`

#### Query a Batch

```python
batch_candidates = lsh.query(sketches)
```

Return value:

- `list[list[int]]`

#### Query a Batch as CSR

```python
flat, indptr = lsh.query(sketches, format="csr")
```

Return values:

- `flat`: 1-D `np.ndarray[np.uint64]`
- `indptr`: 1-D `np.ndarray[np.uint64]`

Use CSR output when:

- you want compact batch output
- you will post-process candidate counts yourself
- you want to avoid large Python nested lists

Note:

- when querying rows that are already in the index, each row typically matches its own row id as well

### Get Duplicate Flags With `duplicates`

Use `duplicates` when the index already contains the batch you want to self-query, or when you have inserted earlier rows and now want flags for a later block.

```python
flags = lsh.duplicates(sketches, self_start=0)
```

Return value:

- `np.ndarray` of shape `(B,)`, dtype `uint8`

Interpretation:

- `0`: no matching row besides the row's own id
- `1`: at least one matching row id differs from the row's own id

`self_start` tells FastSketchLSH what row id the first row in `sketches` should be treated as.

Example with a pre-filled index:

```python
warmup = sketcher.batch([["a"], ["b"]])
target = sketcher.batch([["cat", "dog"], ["cat", "dog"]])

lsh.insert(warmup)
lsh.insert(target)

flags = lsh.duplicates(target, self_start=len(warmup))
```

### Insert and Flag in One Step With `insert_and_query_duplicates`

Use this method when you want one-shot duplicate flags for the rows being inserted right now.

```python
flags = lsh.insert_and_query_duplicates(sketches)
```

Return value:

- `np.ndarray` of shape `(B,)`, dtype `uint8`

Supported inputs:

- `np.ndarray` with shape `(B, num_perm)` and dtype `uint64`
- `list[np.ndarray[np.uint64]]`
- `list[list[int]]`

This is the simplest path for one-shot deduplication jobs:

```python
sketches = sketcher.batch(token_sets, num_threads=8)
lsh = LSH(num_perm=128, num_bands=16)
dup_flags = lsh.insert_and_query_duplicates(sketches)
```

### Capacity and Thread Utilities

```python
lsh.reserve(expected_num_items=100_000)
lsh.set_num_threads(8)
lsh.clear()
```

Useful read-only properties:

- `lsh.num_perm`
- `lsh.num_bands`
- `lsh.band_size`
- `lsh.num_threads`

Module-level helper:

```python
from FastSketchLSH import omp_max_threads

print(omp_max_threads())
```

## End-to-End Examples

### Example 1: Sketch and Estimate Jaccard

```python
from FastSketchLSH import FastSimilaritySketch, estimate_jaccard

sketcher = FastSimilaritySketch(size=256, seed=42)

left = [f"a-{i}" for i in range(16_000)]
right = [f"a-{i}" for i in range(8_000)] + [f"b-{i}" for i in range(8_000)]

sig_left = sketcher(left)
sig_right = sketcher(right)

print(estimate_jaccard(sig_left, sig_right))
```

### Example 2: One-Shot Batch Deduplication

```python
from FastSketchLSH import FastSimilaritySketch, LSH

token_sets = [
    ["cat", "dog", "mouse"],
    ["cat", "dog", "mouse"],
    ["tree", "river", "mountain"],
]

sketcher = FastSimilaritySketch(size=128, seed=42)
sketches = sketcher.batch(token_sets, num_threads=1)

lsh = LSH(num_perm=128, num_bands=8, num_threads=1)
dup_flags = lsh.insert_and_query_duplicates(sketches)

print(dup_flags.tolist())
```

### Example 3: Query an Existing Index

```python
from FastSketchLSH import FastSimilaritySketch, LSH

base_rows = [
    ["cat", "dog", "mouse"],
    ["tree", "river", "mountain"],
]
query_rows = [
    ["cat", "dog", "mouse"],
    ["forest", "lake"],
]

sketcher = FastSimilaritySketch(size=128, seed=42)
base_sketches = sketcher.batch(base_rows)
query_sketches = sketcher.batch(query_rows)

lsh = LSH(num_perm=128, num_bands=8)
lsh.insert(base_sketches)

print(lsh.query(query_sketches[0]))
print(lsh.query(query_sketches, format="csr"))
```

## Practical Notes

- Prefer `np.uint32` for dense numeric token ids.
- Prefer `prehashed=True` when token hashes are already available.
- Use `batch_csr` when your upstream pipeline already stores rows as CSR.
- Use `insert_and_query_duplicates` for one-shot batch deduplication.
- Use `duplicates` when you need flags against an index that has already been populated.
