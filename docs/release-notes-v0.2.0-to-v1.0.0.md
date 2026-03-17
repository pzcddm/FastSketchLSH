# FastSketchLSH: v0.2.0 to v1.0.0

This file is the single retained record of the changes made between `v0.2.0` and `v1.0.0`.
It replaces the older optimization worklog documents.

## Patch release `v1.0.1`

`v1.0.1` keeps the `v1.0.0` public API intact.
This patch release bumps the package version and adds pytest collection isolation so default `pytest` runs do not import script-style files under `test/` that can preload an older installed extension module.

## Why `v1.0.0`

`v1.0.0` keeps the native FastSketch kernels and LSH implementation direction from `v0.2.0`,
but cleans up the public Python API so the common paths are smaller, more consistent, and easier
to document.

## Public API Changes

### FastSimilaritySketch

Old -> New:

- `FastSimilaritySketch(sketch_size=256)` -> `FastSimilaritySketch(size=256)`
- `sketcher.sketch(items)` -> `sketcher(items)`
- `sketcher.sketch_prehashed(items)` -> `sketcher(items, prehashed=True)`
- `sketcher.sketch_batch(rows, num_threads=8)` -> `sketcher.batch(rows, num_threads=8)`
- `sketcher.sketch_batch_prehashed(rows, num_threads=8)` -> `sketcher.batch(rows, prehashed=True, num_threads=8)`
- `sketcher.sketch_batch_flat_csr_prehashed(data, indptr, num_threads=8)` -> `sketcher.batch_csr(data, indptr, prehashed=True, num_threads=8)`

### LSH

Old -> New:

- `lsh.build_from_batch(sketches)` -> `lsh.insert(sketches)`
- `lsh.query_candidates(row)` -> `lsh.query(row)`
- `lsh.batch_query_csr(sketches)` -> `lsh.query(sketches, format="csr")`
- `lsh.batch_query_duplicate_flags(sketches, self_id_start=0)` -> `lsh.duplicates(sketches, self_start=0)`

New in `v1.0.0`:

- `lsh.insert_and_query_duplicates(sketches)`
  - inserts a batch and returns duplicate flags for the inserted rows
  - used by the current single-thread end-to-end FastSketch adapter

## Functional Changes

- Unified pre-hashed APIs under `prehashed=True`
- Consolidated the public API from many specialized methods into a small stable surface
- Added thread-aware dispatch in batch sketching so multi-thread runs parallelize both hashing and sketching
- Added duplicate-flag fast path for post-build LSH queries
- Added true LSH one-shot insert + duplicate flagging for single-thread end-to-end runs
- Updated the fair benchmark harness to use current Rensa matrix / one-shot APIs when available

## Benchmark-Relevant Changes

- End-to-end FastSketch now uses true LSH one-shot duplicate flagging when `threads=1`
- Multi-threaded end-to-end FastSketch still uses `insert(...)` then `duplicates(...)`
  because the current one-shot path is row-serial
- Benchmark docs and README examples now use the `v1.0.0` API names

## Migration Example

Before:

```python
from FastSketchLSH import FastSimilaritySketch, LSH

sketcher = FastSimilaritySketch(sketch_size=128, seed=42)
sketch_matrix = sketcher.sketch_batch(token_sets)

lsh = LSH(num_perm=128, num_bands=16)
lsh.build_from_batch(sketch_matrix)
flags = [1 if len(lsh.query_candidates(row)) > 1 else 0 for row in sketch_matrix]
```

After:

```python
from FastSketchLSH import FastSimilaritySketch, LSH

sketcher = FastSimilaritySketch(size=128, seed=42)
sketch_matrix = sketcher.batch(token_sets)

lsh = LSH(num_perm=128, num_bands=16)
flags = lsh.insert_and_query_duplicates(sketch_matrix).tolist()
```

## Scope

This file is intentionally a concise version-history summary, not a day-by-day optimization log.
