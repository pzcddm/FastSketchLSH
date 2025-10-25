//
// High-throughput band-parallel LSH (fused compute+insert)
//
// Implementation details
// - One flat hash table per band: tables_[band] : map<uint64_t, vector<size_t>>
// - Band hash: splitmix64-style 64-bit mixer on word slices with per-band salts
// - Build path: fused compute+insert with at most num_bands threads (one per band)
// - Query path: fused compute+probe; candidate dedup via flat hash set
//
// Complexity
// - Build:  O(batch * num_perm)
// - Query:  O(num_perm + output_size)
// - Memory: O(unique band hashes + total stored IDs)
//
// This file implements the LSH class for high-throughput band-parallel locality-sensitive hashing.
// Each function is documented with a brief description of its purpose and usage.

#include <algorithm>
#include <cmath>
#include <atomic>
#include <memory>

#include "../include/LSH.h"

namespace {
    // Generate a unique salt for each band using the global seed and band index.
    static inline std::uint64_t make_salt(std::uint64_t seed, std::size_t band_index) {
        // Derive a stable salt per band from the global seed
        std::uint64_t x = seed ^ (0x9e3779b97f4a7c15ULL * static_cast<std::uint64_t>(band_index + 1));
        // Reuse the same mix function as LSH::mix64 without depending on private scope
        x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
        x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
        x ^= x >> 33; return x;
    }
}

// Constructor for LSH.
// Initializes the LSH object with the given number of permutations, bands, hash kind, and seed.
// Throws std::invalid_argument if parameters are invalid.
LSH::LSH(std::size_t num_perm,
         std::size_t num_bands,
         BandHashKind hash_kind,
         std::uint64_t seed,
         int num_threads)
    : num_perm_(num_perm),
      num_bands_(num_bands),
      band_size_(0),
      hash_kind_(hash_kind),
      seed_(seed),
      band_salts_(),
      tables_(),
      next_id_(0),
      last_candidates_(),
      num_threads_(0) {
    if (num_perm_ == 0 || num_bands_ == 0) {
        throw std::invalid_argument("num_perm and num_bands must be > 0");
    }
    if (num_perm_ % num_bands_ != 0) {
        throw std::invalid_argument("num_perm must be divisible by num_bands");
    }
    set_num_threads(num_threads);
    band_size_ = num_perm_ / num_bands_;
    band_salts_.resize(num_bands_);
    for (std::size_t b = 0; b < num_bands_; ++b) {
        band_salts_[b] = make_salt(seed_, b);
    }
    tables_.resize(num_bands_);
}

void LSH::set_num_threads(int num_threads) {
#ifndef _OPENMP
    if (num_threads > 1) {
        throw std::invalid_argument("OpenMP support is disabled; num_threads must be 0 or 1");
    }
#endif
    if (num_threads <= 0) {
        num_threads_ = 0; // auto (use all available threads)
    } else {
        num_threads_ = num_threads;
    }
}

// Reserve space in each band's hash table for the expected number of items.
// This helps minimize rehashing during insertion.
void LSH::reserve(std::size_t expected_num_items) {
    if (expected_num_items == 0) return;
    // Target load factor ~0.6; conservatively reserve to minimize rehash
    const double target_load = 0.6;
    const std::size_t map_capacity = static_cast<std::size_t>(std::ceil(expected_num_items / target_load));
    for (auto& table : tables_) {
        table.reserve(map_capacity);
    }
}

// Clear all hash tables and reset the next_id_ counter.
void LSH::clear() {
    for (auto& table : tables_) {
        table.clear();
    }
    next_id_ = 0;
}

// Build the LSH tables from a batch of contiguous rows (row-major).
// Each row is a sketch of length t (must equal num_perm_).
// The batch is inserted in parallel across bands.
void LSH::build_from_batch(const std::uint64_t* base,
                           std::size_t batch,
                           std::size_t t) {
    if (t != num_perm_) {
        throw std::invalid_argument("t must equal num_perm");
    }
    if (batch == 0) return;
    const std::size_t global_start_id = next_id_;
    next_id_ += batch;

    const std::size_t stride_elems = t;

    // Pre-reserve map capacity per band
    reserve(batch);

    // Parallelize across bands with OpenMP. Each band owns its table exclusively.
    const int threads = resolved_num_threads();
#ifndef _OPENMP
    (void)threads;
#endif
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(threads)
#else
    #pragma omp parallel for schedule(static)
#endif
    for (std::size_t b = 0; b < num_bands_; ++b) {
        BandTable& table = tables_[b];
        const std::uint64_t salt = band_salts_[b];
        const std::size_t offset = b * band_size_;
        for (std::size_t i = 0; i < batch; ++i) {
            const std::uint64_t* row = base + i * stride_elems;
            const std::uint64_t* slice = row + offset;
            const std::uint64_t h = band_hash(slice, band_size_, salt);
            const std::size_t id = global_start_id + i;
            table[h].push_back(id);
        }
    }
}

// Build the LSH tables from a batch of pointer-to-row arrays.
// Each row is a pointer to a sketch of length t (must equal num_perm_).
// The batch is inserted in parallel across bands.
void LSH::build_from_batch(const std::uint64_t* const* rows,
                           std::size_t batch,
                           std::size_t t) {
    if (t != num_perm_) {
        throw std::invalid_argument("t must equal num_perm");
    }
    if (batch == 0) return;
    const std::size_t global_start_id = next_id_;
    next_id_ += batch;

    // Pre-reserve map capacity per band
    reserve(batch);

    // Parallelize across bands with OpenMP. Each band owns its table exclusively.
    const int threads = resolved_num_threads();
#ifndef _OPENMP
    (void)threads;
#endif
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(threads)
#else
    #pragma omp parallel for schedule(static)
#endif
    for (std::size_t b = 0; b < num_bands_; ++b) {
        BandTable& table = tables_[b];
        const std::uint64_t salt = band_salts_[b];
        const std::size_t offset = b * band_size_;
        for (std::size_t i = 0; i < batch; ++i) {
            const std::uint64_t* row = rows[i];
            const std::uint64_t* slice = row + offset;
            const std::uint64_t h = band_hash(slice, band_size_, salt);
            const std::size_t id = global_start_id + i;
            table[h].push_back(id);
        }
    }
}

// Query the LSH tables for candidates similar to the given digest (sketch).
// Returns a deduplicated vector of candidate IDs.
// Time: O(num_bands_ * band_size_ + output_size)
const std::vector<std::size_t>& LSH::query_candidates(const std::uint64_t* digest,
                                                      std::size_t t) const {

    // Collect then sort+unique to deduplicate candidates across bands.
    // Reuse internal buffer to avoid per-call allocation.
    last_candidates_.clear();
    for (std::size_t b = 0; b < num_bands_; ++b) {
        const std::size_t offset = b * band_size_;
        const std::uint64_t salt = band_salts_[b];
        const std::uint64_t h = band_hash(digest + offset, band_size_, salt);
        const auto it = tables_[b].find(h);
        if (it == tables_[b].end()) continue;
        const Bucket& bucket = it->second;
        last_candidates_.insert(last_candidates_.end(), bucket.begin(), bucket.end());
    }
    std::sort(last_candidates_.begin(), last_candidates_.end());
    last_candidates_.erase(std::unique(last_candidates_.begin(), last_candidates_.end()), last_candidates_.end());
    return last_candidates_;
}

// Query the LSH tables for a batch of contiguous digests (row-major).
// Fills flat_out with all candidate IDs and indptr_out with row boundaries.
// Output is in CSR-like format: flat_out[indptr_out[i]:indptr_out[i+1]] are candidates for row i.
void LSH::query_candidates_batch(const std::uint64_t* base,
                                 std::size_t batch,
                                 std::size_t t,
                                 std::vector<std::size_t>& flat_out,
                                 std::vector<std::uint64_t>& indptr_out) const {
    if (t != num_perm_) {
        throw std::invalid_argument("t must equal num_perm");
    }
    indptr_out.clear(); indptr_out.resize(batch + 1);
    flat_out.clear();
    const std::size_t stride_elems = t;

    // Pass 1: compute per-row counts in parallel
    std::vector<std::uint64_t> counts(batch, 0);
    const int threads = resolved_num_threads();
#ifndef _OPENMP
    (void)threads;
#endif
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(threads)
#else
    #pragma omp parallel for schedule(static)
#endif
    for (std::size_t i = 0; i < batch; ++i) {
        const std::uint64_t* digest = base + i * stride_elems;
        std::vector<std::size_t> tmp;
        tmp.reserve(num_bands_ * std::max<std::size_t>(8, band_size_));
        for (std::size_t b = 0; b < num_bands_; ++b) {
            const std::size_t offset = b * band_size_;
            const std::uint64_t h = band_hash(digest + offset, band_size_, band_salts_[b]);
            const auto it = tables_[b].find(h);
            if (it == tables_[b].end()) continue;
            const Bucket& bucket = it->second;
            tmp.insert(tmp.end(), bucket.begin(), bucket.end());
        }
        std::sort(tmp.begin(), tmp.end());
        tmp.erase(std::unique(tmp.begin(), tmp.end()), tmp.end());
        counts[i] = static_cast<std::uint64_t>(tmp.size());
    }

    // Prefix-sum to build indptr and total size
    indptr_out[0] = 0;
    for (std::size_t i = 0; i < batch; ++i) {
        indptr_out[i + 1] = indptr_out[i] + counts[i];
    }
    const std::size_t total = static_cast<std::size_t>(indptr_out[batch]);
    flat_out.resize(total);

    // Pass 2: fill flat_out in parallel
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(threads)
#else
    #pragma omp parallel for schedule(static)
#endif
    for (std::size_t i = 0; i < batch; ++i) {
        const std::uint64_t* digest = base + i * stride_elems;
        std::vector<std::size_t> tmp;
        tmp.reserve(static_cast<std::size_t>(counts[i] + 8));
        for (std::size_t b = 0; b < num_bands_; ++b) {
            const std::size_t offset = b * band_size_;
            const std::uint64_t h = band_hash(digest + offset, band_size_, band_salts_[b]);
            const auto it = tables_[b].find(h);
            if (it == tables_[b].end()) continue;
            const Bucket& bucket = it->second;
            tmp.insert(tmp.end(), bucket.begin(), bucket.end());
        }
        std::sort(tmp.begin(), tmp.end());
        tmp.erase(std::unique(tmp.begin(), tmp.end()), tmp.end());
        const std::size_t start = static_cast<std::size_t>(indptr_out[i]);
        std::copy(tmp.begin(), tmp.end(), flat_out.begin() + static_cast<std::ptrdiff_t>(start));
    }
}

// Query the LSH tables for a batch of pointer-to-row digests.
// Fills flat_out with all candidate IDs and indptr_out with row boundaries.
// Output is in CSR-like format: flat_out[indptr_out[i]:indptr_out[i+1]] are candidates for row i.
void LSH::query_candidates_batch(const std::uint64_t* const* rows,
                                 std::size_t batch,
                                 std::size_t t,
                                 std::vector<std::size_t>& flat_out,
                                 std::vector<std::uint64_t>& indptr_out) const {
    if (t != num_perm_) {
        throw std::invalid_argument("t must equal num_perm");
    }
    indptr_out.clear(); indptr_out.resize(batch + 1);
    flat_out.clear();
    const int threads = resolved_num_threads();
#ifndef _OPENMP
    (void)threads;
#endif

    // Pass 1: compute per-row counts in parallel
    std::vector<std::uint64_t> counts(batch, 0);
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(threads)
#else
    #pragma omp parallel for schedule(static)
#endif
    for (std::size_t i = 0; i < batch; ++i) {
        const std::uint64_t* digest = rows[i];
        std::vector<std::size_t> tmp;
        tmp.reserve(num_bands_ * std::max<std::size_t>(8, band_size_));
        for (std::size_t b = 0; b < num_bands_; ++b) {
            const std::size_t offset = b * band_size_;
            const std::uint64_t h = band_hash(digest + offset, band_size_, band_salts_[b]);
            const auto it = tables_[b].find(h);
            if (it == tables_[b].end()) continue;
            const Bucket& bucket = it->second;
            tmp.insert(tmp.end(), bucket.begin(), bucket.end());
        }
        std::sort(tmp.begin(), tmp.end());
        tmp.erase(std::unique(tmp.begin(), tmp.end()), tmp.end());
        counts[i] = static_cast<std::uint64_t>(tmp.size());
    }

    // Prefix-sum to build indptr and total size
    indptr_out[0] = 0;
    for (std::size_t i = 0; i < batch; ++i) {
        indptr_out[i + 1] = indptr_out[i] + counts[i];
    }
    const std::size_t total = static_cast<std::size_t>(indptr_out[batch]);
    flat_out.resize(total);

    // Pass 2: fill flat_out in parallel
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(threads)
#else
    #pragma omp parallel for schedule(static)
#endif
    for (std::size_t i = 0; i < batch; ++i) {
        const std::uint64_t* digest = rows[i];
        std::vector<std::size_t> tmp;
        tmp.reserve(static_cast<std::size_t>(counts[i] + 8));
        for (std::size_t b = 0; b < num_bands_; ++b) {
            const std::size_t offset = b * band_size_;
            const std::uint64_t h = band_hash(digest + offset, band_size_, band_salts_[b]);
            const auto it = tables_[b].find(h);
            if (it == tables_[b].end()) continue;
            const Bucket& bucket = it->second;
            for (std::size_t id : bucket) tmp.push_back(id);
        }
        std::sort(tmp.begin(), tmp.end());
        tmp.erase(std::unique(tmp.begin(), tmp.end()), tmp.end());
        const std::size_t start = static_cast<std::size_t>(indptr_out[i]);
        std::copy(tmp.begin(), tmp.end(), flat_out.begin() + static_cast<std::ptrdiff_t>(start));
    }
}
