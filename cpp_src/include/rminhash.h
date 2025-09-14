#ifndef RMINHASH_SIMD_H
#define RMINHASH_SIMD_H

#pragma once
#include <vector>
#include <string>
#include <cstdint>
#include <limits>
#include <random>
#include <algorithm>
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif
#include <cstring>
#include "murmurhash.h"

// ===================== Constants and Configuration =====================
static constexpr size_t BATCH_SIZE = 32;           // Items batch processing size (from Rust code)
static constexpr size_t PERM_CHUNK_SIZE = 16;      // Permutation chunk size for vectorization (from Rust code)

// Backend selection: AVX-512 if available, else scalar fallback
#if defined(__AVX512F__) && !defined(FORCE_SCALAR)
#define RMINHASH_SIMD_AVX512 1
#else
#define RMINHASH_SIMD_SCALAR 1
#endif

// ===================== Utility Functions =====================
inline uint32_t permute_hash(uint64_t h, uint64_t a, uint64_t b) {
    return static_cast<uint32_t>((a * h + b) >> 32);
}

// Fast hash function for uint32_t (similar to calculate_hash_fast in Rust)
inline uint64_t calculate_hash_fast(uint32_t x) {
    uint64_t z = (uint64_t)x + 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z = z ^ (z >> 31);
    return z;
}

// Hash function for byte strings using FNV1a-64
inline uint64_t calculate_hash_fast_bytes(const std::string& str) {
    const uint64_t OFF = 1469598103934665603ull;
    const uint64_t PRM = 1099511628211ull;
    uint64_t h = OFF;
    const uint8_t* data = reinterpret_cast<const uint8_t*>(str.data());
    for (size_t i = 0; i < str.size(); ++i) {
        h ^= (uint64_t)data[i];
        h *= PRM;
    }
    return h;
}

// SIMD permute_hash for 8 values (AVX-512)
#if defined(RMINHASH_SIMD_AVX512)
inline __m512i permute_hash_vec(__m512i h, uint64_t a, uint64_t b) {
    const __m512i va = _mm512_set1_epi64((long long)a);
    const __m512i vb = _mm512_set1_epi64((long long)b);
    __m512i result = _mm512_add_epi64(_mm512_mullo_epi64(va, h), vb);
    return _mm512_srli_epi64(result, 32);
}
#endif

// Convert 8 uint32 to 8 uint64 and hash them
#if defined(RMINHASH_SIMD_AVX512)
inline void hash_uint32x8_to_u64_avx512(const uint32_t* src, uint64_t* dst) {
    __m256i v32 = _mm256_loadu_si256((const __m256i*)src);
    __m512i x64 = _mm512_cvtepu32_epi64(v32);
    
    // Apply calculate_hash_fast vectorized
    const __m512i C1 = _mm512_set1_epi64(0x9E3779B97F4A7C15ull);
    const __m512i M1 = _mm512_set1_epi64(0xBF58476D1CE4E5B9ull);
    const __m512i M2 = _mm512_set1_epi64(0x94D049BB133111EBull);
    
    __m512i z = _mm512_add_epi64(x64, C1);
    __m512i t = _mm512_xor_si512(z, _mm512_srli_epi64(z, 30));
    t = _mm512_mullo_epi64(t, M1);
    t = _mm512_xor_si512(t, _mm512_srli_epi64(t, 27));
    t = _mm512_mullo_epi64(t, M2);
    t = _mm512_xor_si512(t, _mm512_srli_epi64(t, 31));
    
    _mm512_storeu_si512((void*)dst, t);
}
#endif

// ===================== Legacy Interface (for compatibility) =====================
struct PermPair {
    uint64_t a;
    uint64_t b;
};

class RMinSketch {
private:
    size_t num_perm;
    uint32_t random_seed;
    std::vector<PermPair> perm_pairs;
    std::vector<uint32_t> hash_values;
    uint32_t permute_hash(uint64_t h, uint64_t a, uint64_t b) const;
    
public:
    RMinSketch(size_t num_perm = 128, uint32_t seed = 42);
    std::vector<uint32_t> sketch(const std::vector<int>& items);
};

// ===================== Optimized SIMD Implementation =====================
class RMinHashSIMD {
private:
    size_t num_perm;
    uint64_t seed;
    std::vector<std::pair<uint64_t, uint64_t>> permutations; // (a, b) pairs
    std::vector<uint32_t> hash_values;
    
    // Persistent buffers for batch processing
    std::vector<uint64_t> hash_batch;
    alignas(64) uint32_t current_chunk[PERM_CHUNK_SIZE];
    
public:
    explicit RMinHashSIMD(size_t num_perm, uint64_t random_seed = 42);
    
    // Main interface functions (following fasthash_simd pattern)
    std::vector<uint32_t> sketch(const std::vector<uint32_t>& items);
    std::vector<uint32_t> sketch(const std::vector<uint32_t>& items,
                                double* prehash_ms,
                                double* update_ms,
                                double* total_ms);
    
    std::vector<uint32_t> sketch(const std::vector<std::string>& items);
    std::vector<uint32_t> sketch(const std::vector<std::string>& items,
                                double* prehash_ms,
                                double* update_ms,
                                double* total_ms);
    
    // Reset hash values for new computation
    void reset();
    
    // Get current digest
    std::vector<uint32_t> digest() const;
    
    // Compute Jaccard similarity with another RMinHashSIMD
    double jaccard(const RMinHashSIMD& other) const;
    
private:
    // Internal update functions
    void update_uint32(const std::vector<uint32_t>& items, 
                      double* prehash_ms = nullptr, 
                      double* update_ms = nullptr);
    void update_strings(const std::vector<std::string>& items,
                       double* prehash_ms = nullptr, 
                       double* update_ms = nullptr);
    
    // Process batch of pre-computed hashes
    void process_hash_batch_chunked(const std::vector<uint64_t>& batch_hashes);
    
    // SIMD optimized chunk processing
    void process_perm_chunk_simd(const std::vector<uint64_t>& batch_hashes,
                                size_t perm_start, size_t perm_end);
    
    // Scalar fallback for chunk processing
    void process_perm_chunk_scalar(const std::vector<uint64_t>& batch_hashes,
                                  size_t perm_start, size_t perm_end);
};

#endif // RMINHASH_SIMD_H
