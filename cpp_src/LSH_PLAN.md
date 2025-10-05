## High‑throughput band‑parallel LSH (fused compute+insert) — Implementation Plan

### Objectives

- **Throughput**: Per‑band parallel build in steady state (lock‑free across bands).
- **Locality**: Minimize memory movement/allocations; keep tables hot in cache; avoid intermediate buffers.
- **Interoperability**: Accept precomputed sketches from multiple Python layouts with zero/low‑copy ingestion.
- **Determinism & robustness**: Portable band hashing; handle skew; scale beyond band count.

### Public API (C++ `LSH`)

- **Constructor**
  - `LSH(size_t num_perm, size_t num_bands,
          BandHashKind hash_kind = BandHashKind::splitmix64,
          uint64_t seed = 0x9e3779b97f4a7c15ULL)`
  - Preconditions: `num_perm > 0`, `num_bands > 0`, `num_perm % num_bands == 0`.

- **Batch build (preferred)**
  - `build_from_batch(const uint64_t* base, size_t batch, size_t t)`
    - 2D contiguous layout; row stride equals `t`.
  - `build_from_batch(const uint64_t* const* rows, size_t batch, size_t t)`
    - Array of row pointers; no copies.

- **Incremental** (Currently Do not implement it, leave it in the future work)

- **Query**
  - `query_candidates(const uint64_t* digest, size_t t) -> std::vector<size_t>`

- **Maintenance**
  - `clear()`, `reserve(size_t expected_num_items)`
  - Getters: `num_perm()`, `num_bands()`, `band_size()`

### File layout

- `cpp_src/include/LSH.h` — public interface and lightweight inline helpers
- `cpp_src/cpp/LSH.cpp` — implementation (hashing, build/query, threading)
- Bindings: `cpp_src/cpp/init.cpp` exposes batch APIs for Python (2D ndarray fast paths, list of ndarrays, list of lists); all heavy paths release the GIL.

### Python bindings (pybind11) — high-level API

- `LSH(num_perm: int, num_bands: int, hash_kind=BandHashKind.splitmix64, seed: int=...)`
- Build:
  - `build_from_batch(ndarray: np.ndarray[uint64, (B, t)])` — zero/low-copy, requires C-contiguous and `t == num_perm`.
  - `build_from_batch(rows: list[np.ndarray[uint64, (t,)]])` — zero/low-copy via row pointers.
  - `build_from_batch(rows: list[list[int]])` — single temporary `(B, t)` copy.
- Query (single):
  - `query_candidates(digest: np.ndarray[uint64, (t,)]) -> list[int]` — zero-copy read of digest.
  - `query_candidates(digest: Iterable[int]) -> list[int]` — converts to a temporary `uint64_t` buffer.
- Query (batch):
  - `batch_query_csr(arr: np.ndarray[uint64, (B, t)]) -> tuple[np.ndarray[uint64], np.ndarray[uint64]]`
    - Returns CSR-style `(flat, indptr)` with zero-copy wrapping of newly allocated output; preferred for throughput and minimal Python overhead.
  - `batch_query(arr: np.ndarray[uint64, (B, t)]) -> list[list[int]]`
    - Convenience list-of-lists; constructs Python lists with preallocation to minimize overhead.

Notes (performance/zero-copy):
- NumPy arguments use c_style and noconvert policies in bindings. Callers must pass `dtype=np.uint64`, C-contiguous arrays; otherwise a `TypeError` is raised (no implicit casts), ensuring zero-copy on inputs.
- All batch compute runs under GIL release. `batch_query_csr` minimizes Python allocations and is recommended for large batches.

### Internal data structures

- Per‑band tables (initial implementation):
  - `tables[band] : HashMap<uint64_t /*band_hash*/, Bucket>`
  - `Bucket = std::vector<size_t>` (stores integer key IDs)
  - Future work: add per‑band shards `tables[band][shard]` to scale beyond band count.

- **Hash map choice** (cache‑friendly, header‑only):
  - Default: `ankerl::unordered_dense::map` (open addressing, robin‑hood, excellent locality)

### Band hash function (SIMD‑friendly, deterministic)

- Avoid per‑byte FNV‑1a dependency; use a strong 64‑bit word‑wise mixer:
  - Default (initial): splitmix64 mix applied to `uint64_t` words
  - Future: add wyhash finalizer option

- Pseudocode (word‑wise):

```cpp
inline uint64_t mix64(uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33; return x; // splitmix64-style
}

uint64_t hash_band(const uint64_t* words, size_t band_size, uint64_t seed) {
    uint64_t acc = seed ^ (band_size * 0x9e3779b97f4a7c15ULL);
    for (size_t i = 0; i < band_size; ++i) {
        acc = mix64(acc ^ words[i]);
    }
    return acc;
}
```

- Complexity: time O(band_size), space O(1). Portable (no endianness issues; operates on digest words).

### Build algorithm (fused compute+insert, per‑band parallel)

- **Threading model**
  - Initial version: use at most `num_bands` threads (one thread per band). No shards in the initial version. No locks on the hot path by construction.
  - Future work: enable additional shards per band and MPSC routing to owner threads to scale beyond `num_bands`.

- **Per key digest**
  - For band `b`: slice `[b*band_size : (b+1)*band_size)` from the digest (no copies)
  - Compute `band_hash = hash_band(slice, band_size, band_salt(b))`
  - Insert into the band’s table: `tables[b][band_hash].push_back(key_id)`

- **Preallocation**
  - Performed inside `build_from_batch` (assume a single batch build in the initial version).
  - Total inserts `M = batch * num_bands`.
  - For each band, call `tables[b].reserve(ceil(batch / target_load_factor))` with `target_load_factor ≈ 0.6` before inserts.
  - Buckets: append‑only vectors; optionally pre‑reserve an average bucket size if an estimate is available; otherwise rely on amortized growth.
  - If multiple `build_from_batch` calls occur, capacity may need to grow and rehash; the initial version does not recompute global reserves across batches.

### Optional big‑batch mode (sort‑and‑build) — Future work

- Not in the initial version. Baseline uses fused per‑band insert without staging/sorting.
- Future: for very large static builds, per band, collect `(band_hash, key_id)` pairs, sort/group by `band_hash`, then linear build of buckets. This can improve locality and reduce allocations at extreme scale.

### Input ingestion (zero/low‑copy)

- 2D NumPy ndarray `shape=(B, t), dtype=uint64`:
  - Contiguous (C‑order): pass `base` (row stride equals `t`)
- List of 1D ndarrays `dtype=uint64, len==t`:
  - Build an array of row pointers; no data copies
- List of Python lists (last resort):
  - Validate length==t across rows; copy once into a temporary `(B, t)` buffer; fast‑fail on mismatch
- Always validate `t == num_perm` and `B > 0` (or allow empty batch as no‑op)

### Query path (fused compute+probe)

- For each band `b`:
  - Compute `band_hash` on the digest slice
  - Probe the band’s table and append bucket contents into a thread‑local vector
-- Optional candidate dedup:
  - Initial version (small and large): use a reserved flat hash set `ankerl::unordered_dense::set<size_t>` to deduplicate candidates.

### Concurrency & scheduling

- **Band‑level concurrency**: one thread per band (current implementation)
- **Scaling beyond bands**: future work via per‑band sharding and MPSC routing or concurrent hash tables
- **Skew handling**: can be addressed by sharding in future work; current version is per‑band only

### Memory & locality

- Use `ankerl::unordered_dense` maps (contiguous control/data) for fewer cache misses than node‑based tables
- Reserve map capacity up‑front to avoid rehashing; choose a generous load factor target (e.g., 0.5–0.7)
- Keep buckets as `std::vector<size_t>`; defer dedup/compaction to post‑build if needed
- Optional: `std::pmr::monotonic_buffer_resource` per band to reduce dynamic allocations (advanced)

### Corner cases & robustness

- Validate constructor arguments; throw `std::invalid_argument` on bad configs
- Empty batch → no‑op; `insert` validates `t == num_perm`
- Allocation failures: check `reserve` sizes (`batch * num_bands` fit in `size_t`)
- Determinism: band hashing uses digest words and fixed salts; results portable across platforms

### Complexity & performance

- Build time: O(batch * num_perm); parallel speedup up to `num_bands` (bounded by cores/memory BW)
- Query time: O(num_perm) to form candidates (plus output size)
- Space: O(unique band hashes + total stored IDs); low constant factors with flat hash maps

### Milestones / checklist

- [ ] Add `include/LSH.h`, `cpp/LSH.cpp` skeleton with API above
- [ ] Integrate `ankerl::unordered_dense` (or `phmap`) as the backing table
 - [ ] Implement band hashing (splitmix64) with per‑band salts
 - [ ] Implement fused per‑band builder with preallocation
- [ ] Implement query path with candidate dedup (initial: hash set; future: sort+unique/radix+unique)
- [ ] Extend bindings in `init.cpp` to accept 2D ndarray, list of ndarrays, list of lists (zero/low‑copy); release GIL
- [ ] Benchmarks: throughput, memory, skew stress, correctness

### Key Points

- **Fused per‑band compute+insert** avoids extra passes and maximizes locality.
- **Flat/open‑addressing maps** (`ankerl::unordered_dense`) beat `std::unordered_*` in speed/memory.
- **Word‑wise band hashing** (splitmix64) is portable and CPU‑friendly.
- **Zero/low‑copy ingestion** for ndarray layouts; single copy only for Python lists.
- **Shards per band** (future) scale parallelism; big‑batch sort‑and‑build (future) can help at extreme scale.




### Future Plan

- Add a returned [1,0] array when build_batch the LSH, which indicate whether each minhash(document) is firstly inserted into bands or not. Give us a rough deduplication result.
- Add per‑band sharding (`tables[band][shard]`) to scale beyond `num_bands` threads.
- Introduce MPSC owner routing for lock‑free multi‑producer → single‑writer shard inserts.
- Implement big‑batch sort‑and‑build pipeline for extreme‑scale static builds.
- Optimize large candidate dedup via `sort+unique` (and radix+unique) on contiguous vectors.
- Add `wyhash` finalizer as an alternative band mixer; configurable salts.
- Provide SIMD‑specialized splitmix64 paths (AVX‑512/NEON) and inter‑band SIMD for AVX2.
- Explore `std::pmr` arenas per band to reduce allocation overhead; compact bucket storage formats.
