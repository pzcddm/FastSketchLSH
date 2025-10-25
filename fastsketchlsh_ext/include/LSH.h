//
// High-throughput band-parallel LSH (fused compute+insert)
//
// This header declares the C++ LSH class described in fastsketchlsh_ext/LSH_PLAN.md.
//
// Design goals:
//  - Lock-free across bands in steady state (one thread per band)
//  - Cache-friendly per-band hash tables
//  - Zero/low-copy ingestion from Python (handled in bindings)
//  - Deterministic, portable band hashing (splitmix64 style mixer)
//
// Complexity:
//  - Build time: O(batch * num_perm)
//  - Query time: O(num_perm + output_size)
//  - Space: O(unique band hashes + stored IDs)
//
#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

// Require ankerl::unordered_dense
#include <ankerl/unordered_dense.h>

class LSH {
public:
    enum class BandHashKind {
        splitmix64 = 0,
        wyhash_final = 1 // reserved for future use
    };

    // Construct LSH with t=num_perm, b=num_bands
    // Preconditions: num_perm>0, num_bands>0, num_perm % num_bands == 0
    explicit LSH(std::size_t num_perm,
                 std::size_t num_bands,
                 BandHashKind hash_kind = BandHashKind::splitmix64,
                 std::uint64_t seed = 0x9e3779b97f4a7c15ULL,
                 int num_threads = 0);

    // Reserve internal capacities for an expected number of items (rows).
    void reserve(std::size_t expected_num_items);

    // Clear all tables and reset counters.
    void clear();

    // Batch build from a contiguous 2D buffer with row stride equal to t.
    void build_from_batch(const std::uint64_t* base,
                          std::size_t batch,
                          std::size_t t);

    // Batch build from an array of row pointers (each row length is t words).
    void build_from_batch(const std::uint64_t* const* rows,
                          std::size_t batch,
                          std::size_t t);

    // Query candidate IDs for a single digest of length t words.
    // Fills and returns a reference to an internal buffer. Not thread-safe.
    // The returned reference remains valid until the next call on this instance.
    const std::vector<std::size_t>& query_candidates(const std::uint64_t* digest,
                                                     std::size_t t) const;

    // Batch query: build a flattened list of candidates and CSR-style indptr from contiguous 2D buffer.
    void query_candidates_batch(const std::uint64_t* base,
                                std::size_t batch,
                                std::size_t t,
                                std::vector<std::size_t>& flat_out,
                                std::vector<std::uint64_t>& indptr_out) const;

    void query_candidates_batch(const std::uint64_t* const* rows,
                                std::size_t batch,
                                std::size_t t,
                                std::vector<std::size_t>& flat_out,
                                std::vector<std::uint64_t>& indptr_out) const;

    // Getters
    inline std::size_t num_perm() const noexcept { return num_perm_; }
    inline std::size_t num_bands() const noexcept { return num_bands_; }
    inline std::size_t band_size() const noexcept { return band_size_; }
    inline int num_threads() const noexcept { return num_threads_; }
    void set_num_threads(int num_threads); // <=0 means auto (use OpenMP default/all threads)
    // threshold removed; kept in constructor signature for API compatibility

private:
    template <class K, class V>
        using FlatMap = ankerl::unordered_dense::map<K, V>;

    using Bucket = std::vector<std::size_t>;
    using BandTable = FlatMap<std::uint64_t, Bucket>;

    std::size_t num_perm_;
    std::size_t num_bands_;
    std::size_t band_size_;
    BandHashKind hash_kind_;
    std::uint64_t seed_;
    // Per-band random salt values used to randomize the hash function for each band.
    // Each band uses its own salt to ensure independent hash partitions, improving LSH effectiveness.
    std::vector<std::uint64_t> band_salts_;
    std::vector<BandTable> tables_; // one table per band
    std::size_t next_id_; // monotonically increasing row id assigned on build
    // Scratch buffer reused across queries to avoid per-call allocations.
    // Mutable to allow filling in const query methods. Single-thread only.
    mutable std::vector<std::size_t> last_candidates_;
    int num_threads_;

    static inline std::uint64_t mix64(std::uint64_t x) {
        x ^= x >> 33;
        x *= 0xff51afd7ed558ccdULL;
        x ^= x >> 33;
        x *= 0xc4ceb9fe1a85ec53ULL;
        x ^= x >> 33;
        return x;
    }

    static inline std::uint64_t wyhash_mix64(std::uint64_t x) {
        // Placeholder for future wyhash final mixer; keep behavior deterministic for now
        return mix64(x);
    }

    inline std::uint64_t band_hash(const std::uint64_t* words,
                                   std::size_t n,
                                   std::uint64_t salt) const {
        std::uint64_t acc = salt ^ (n * 0x9e3779b97f4a7c15ULL);
        for (std::size_t i = 0; i < n; ++i) {
            acc = mix64(acc ^ words[i]);
        }
        return acc;
    }

    inline int resolved_num_threads() const noexcept {
#ifdef _OPENMP
        if (num_threads_ > 0) {
            return num_threads_;
        }
        return omp_get_max_threads();
#else
        return 1;
#endif
    }
};
