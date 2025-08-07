#ifndef FAST_SIMILARITY_SKETCH_AVX512_PACKED_H
#define FAST_SIMILARITY_SKETCH_AVX512_PACKED_H

#include <immintrin.h>
#include <cstdint>
#include <vector>
#include <string>
#include <stdexcept>

// TODO: Can we pick a SIMD friendly Hash method? Not fnv1a64?
// TODO (Maybe): Test scatter and gather in line 100 - 111 (逐lane 更新) 但是得处理冲突情况

// ===================== 公共常量与工具 =====================
static constexpr int I_BITS = 12;                      // i 用 12 bits（支持到 4095）
static constexpr int I_SHIFT = 64 - I_BITS;            // 放在高位
static constexpr uint64_t H48_MASK = (1ull << 48) - 1; // 低 48 位

static inline uint64_t INF_KEY() { return ~0ull; }     // 空/极大值

// 工具函数声明
uint64_t pack_key(uint64_t i, uint64_t h48);
uint64_t fnv1a64(const uint8_t* p, size_t n);
uint64_t splitmix64(uint64_t x);
__m512i splitmix64_vec(__m512i x);
int count_filled_simd(const uint8_t* filled, int t);
void warm_cache(uint64_t* S, int t);
void round1_block_avx512_no_reduce(
    const uint64_t* base_block, int nlanes,
    uint64_t round_i, uint64_t seed_i,
    uint64_t* S, uint8_t* filled,
    uint64_t t_mask);

// ===================== 主类声明 =====================
struct FastSimilaritySketchAVX512Packed {
    int t;
    uint64_t t_mask;        // t-1（t 为 2 的幂）
    std::vector<uint64_t> seeds; // 2*t seeds

    explicit FastSimilaritySketchAVX512Packed(int sketch_size, uint64_t random_seed=42);
    std::vector<uint64_t> sketch(const std::vector<std::string>& A);
};

#endif // FAST_SIMILARITY_SKETCH_AVX512_PACKED_H