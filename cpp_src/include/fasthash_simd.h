#ifndef FASTHASH_SIMD_H
#define FASTHASH_SIMD_H

#include <vector>
#include <limits>
#include <random>
#include <utility>
#include <stdexcept>

// SIMD指令集检测宏
#if defined(__AVX2__)
#include <immintrin.h>
#define USE_AVX2
#elif defined(__SSE4_2__)
#include <nmmintrin.h>
#include <emmintrin.h>
#define USE_SSE
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#define USE_NEON
#endif

class FastSimilaritySketch {
public:
    FastSimilaritySketch(size_t sketch_size, uint32_t random_seed);
    
    std::vector<uint64_t> sketch(const std::vector<int>& items);

private:
    size_t sketch_size;
    std::vector<uint64_t> hash_seeds;

    // SIMD哈希批量计算
    void murmur_hash_batch(const int* items, size_t count, uint64_t seed, uint64_t* results);
    
    // 平台特定的SIMD实现
#ifdef USE_AVX2
    void murmur_hash_avx2(const int* items, size_t count, uint64_t seed, uint64_t* results);
#elif defined(USE_SSE)
    void murmur_hash_sse(const int* items, size_t count, uint64_t seed, uint64_t* results);
#elif defined(USE_NEON)
    void murmur_hash_neon(const int* items, size_t count, uint64_t seed, uint64_t* results);
#endif

    // MurmurHash3核心操作
    __m256i avx2_murmur3_mix(__m256i h, __m256i k);
    __m128i sse_murmur3_mix(__m128i h, __m128i k);
    uint64x2_t neon_murmur3_mix(uint64x2_t h, uint64x2_t k);
};

#endif // FASTHASH_SIMD_H
