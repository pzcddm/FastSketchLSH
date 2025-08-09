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
bool all_filled_avx512(const uint64_t* S, int t);
void warm_cache(uint64_t* S, int t);
void round1_block_avx512_no_reduce(
    const uint64_t* base_block, int nlanes,
    uint64_t round_i, uint64_t seed_i,
    uint64_t* S,
    uint64_t t_mask);

// ===================== Main class declaration =====================
struct FastSimilaritySketchAVX512Packed {
    int t;
    uint64_t t_mask;                // t-1 (t must be a power of two)
    std::vector<uint64_t> seeds;    // 2*t seeds

    explicit FastSimilaritySketchAVX512Packed(int sketch_size, uint64_t random_seed=42);
    // Input changed to vector<int> for better SIMD-friendly preprocessing
    std::vector<uint64_t> sketch(const std::vector<int>& A);
};

#endif // FAST_SIMILARITY_SKETCH_AVX512_PACKED_H