#ifndef FAST_SIMILARITY_SKETCH_AVX512_PACKED_H
#define FAST_SIMILARITY_SKETCH_AVX512_PACKED_H

#include <immintrin.h>
#include <cstdint>
#include <vector>
#include <string>
#include <stdexcept>

// TODO: Can we pick a SIMD friendly Hash method? Not fnv1a64?
// TODO (Maybe): Test scatter and gather in line 100 - 111 (逐lane 更新) 但是得处理冲突情况

// ===================== Public constants and utilities =====================
static constexpr int I_BITS = 12;                      // Number of bits used for round index i (supports up to 4095)
static constexpr int I_SHIFT = 64 - I_BITS;            // i placed at the high bits
static constexpr uint64_t H52_MASK = (1ull << 52) - 1; // Low 52 bits mask for hash value

static inline uint64_t INF_KEY() { return ~0ull; }     // Infinity marker for empty bucket

// Utility function declarations
uint64_t pack_key(uint64_t i, uint64_t h52);
__m512i pack_key_vec(uint64_t i, __m512i h52);
uint64_t fnv1a64(const uint8_t* p, size_t n);
uint64_t hash_int32(uint32_t x);
uint64_t splitmix64(uint64_t x);
__m512i splitmix64_vec(__m512i x);
  // Hash 8 uint32 values into 8 uint64 using AVX-512 (declared for testing)
  void hash_int32x8_to_u64_avx512(const uint32_t* src, uint64_t* dst);
bool all_filled_avx512(const uint64_t* S, int t);
void warm_cache(uint64_t* S, int t);
void round1_block_avx512_no_reduce(
    const uint64_t* base_block, int nlanes,
    uint64_t round_i, uint64_t seed_i,
    uint64_t* S,
    uint64_t t_mask,
    uint64_t* h_lane,
    uint64_t* b_lane,
    uint64_t* key_lane,
    const __m512i& seedv,
    const __m512i& maskv,
    const __m512i& hiv,
    const __m512i& h52maskv);

// ===================== Inline definitions for benchmarking =====================
// These are provided inline so that standalone benchmarks can use the same
// implementations without requiring symbol export from the core library.

inline uint64_t fnv1a64(const uint8_t* p, size_t n) {
    const uint64_t OFF = 1469598103934665603ull;
    const uint64_t PRM = 1099511628211ull;
    uint64_t h = OFF;
    for (size_t i = 0; i < n; ++i) { h ^= (uint64_t)p[i]; h *= PRM; }
    return h;
}

inline uint64_t hash_int32(uint32_t x) {
    uint64_t z = (uint64_t)x + 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z = z ^ (z >> 31);
    return z;
}

inline uint64_t splitmix64(uint64_t x){
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

#ifdef FASTHASH_SIMD_ENABLE_AVX512_INLINE
// Use constants hoisted in the TU implementation when inlined is not needed.
inline __m512i splitmix64_vec(__m512i x){
    const __m512i C1 = _mm512_set1_epi64(0x9E3779B97F4A7C15ull);
    const __m512i M1 = _mm512_set1_epi64(0xBF58476D1CE4E5B9ull);
    const __m512i M2 = _mm512_set1_epi64(0x94D049BB133111EBull);
    x = _mm512_add_epi64(x, C1);
    __m512i t = _mm512_xor_si512(x, _mm512_srli_epi64(x, 30));
    t = _mm512_mullo_epi64(t, M1);
    t = _mm512_xor_si512(t, _mm512_srli_epi64(t, 27));
    t = _mm512_mullo_epi64(t, M2);
    t = _mm512_xor_si512(t, _mm512_srli_epi64(t, 31));
    return t;
}

inline void hash_int32x8_to_u64_avx512(const uint32_t* src, uint64_t* dst) {
    __m256i v32 = _mm256_loadu_si256((const __m256i*)src);
    __m512i x64 = _mm512_cvtepu32_epi64(v32);                 // zero-extends
    __m512i h = splitmix64_vec(x64);
    _mm512_storeu_si512((void*)dst, h);
}
#endif

// ===================== Main class declaration =====================
struct FastSimilaritySketchAVX512Packed {
    int t;
    uint64_t t_mask;                // t-1 (t must be a power of two)
    std::vector<uint64_t> seeds;    // 2*t seeds
    // Persistent buffer to avoid reallocating prehash storage every call
    std::vector<uint64_t> base_buffer;
    // Persistent buckets buffer to avoid per-call allocation; capacity up to 4096
    std::vector<uint64_t> buckets_S;
    // Preallocated lane buffers to avoid per-call stack arrays
    alignas(64) uint64_t h_lane_buf[8];
    alignas(64) uint64_t b_lane_buf[8];
    alignas(64) uint64_t key_lane_buf[8];

    explicit FastSimilaritySketchAVX512Packed(int sketch_size, uint64_t random_seed=42);
    // Input changed to vector<uint32_t> for SIMD-friendly zero-extension
    std::vector<uint64_t> sketch(const std::vector<uint32_t>& A);
    // Instrumented overload: returns per-phase timings (ms) if pointers are non-null
    std::vector<uint64_t> sketch(const std::vector<uint32_t>& A,
                                 double* prehash_ms,
                                 double* phase1_ms,
                                 double* phase2_ms);
};

#endif // FAST_SIMILARITY_SKETCH_AVX512_PACKED_H