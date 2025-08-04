#include "fasthash_simd.h"
#include "../include/murmurhash.h"
#include <cstring>

// MurmurHash3常量
constexpr uint64_t C1 = 0x87c37b91114253d5ULL;
constexpr uint64_t C2 = 0x4cf5ad432745937fULL;
constexpr uint64_t C3 = 0x52dce729ULL;
constexpr uint64_t C4 = 0x38495ab5ULL;

FastSimilaritySketch::FastSimilaritySketch(size_t sketch_size, uint32_t random_seed) {
    if (sketch_size == 0) {
        throw std::invalid_argument("Sketch size (t) must be positive");
    }
    this->sketch_size = sketch_size;
    
    std::mt19937_64 gen(random_seed);
    std::uniform_int_distribution<uint64_t> dist(0, UINT64_MAX);
    
    hash_seeds.resize(2 * sketch_size);
    for (auto& seed : hash_seeds) {
        seed = dist(gen);
    }
}

// AVX2 MurmurHash3混合函数
__m256i FastSimilaritySketch::avx2_murmur3_mix(__m256i h, __m256i k) {
    __m256i k_mul = _mm256_mullo_epi64(k, _mm256_set1_epi64x(C1));
    k_mul = _mm256_rolv_epi64(k_mul, _mm256_set_epi64x(31, 31, 31, 31));
    k_mul = _mm256_mullo_epi64(k_mul, _mm256_set1_epi64x(C2));
    h = _mm256_xor_si256(h, k_mul);
    h = _mm256_rolv_epi64(h, _mm256_set_epi64x(27, 27, 27, 27));
    h = _mm256_add_epi64(h, _mm256_set1_epi64x(C3));
    h = _mm256_mullo_epi64(h, _mm256_set1_epi64x(5));
    return h;
}

// SSE MurmurHash3混合函数
__m128i FastSimilaritySketch::sse_murmur3_mix(__m128i h, __m128i k) {
    __m128i k_mul = _mm_mul_epu32(k, _mm_set1_epi64x(C1));
    k_mul = _mm_rol_epi64(k_mul, 31);
    k_mul = _mm_mul_epu32(k_mul, _mm_set1_epi64x(C2));
    h = _mm_xor_si128(h, k_mul);
    h = _mm_rol_epi64(h, 27);
    h = _mm_add_epi64(h, _mm_set1_epi64x(C3));
    h = _mm_mul_epu32(h, _mm_set1_epi64x(5));
    return h;
}

// NEON MurmurHash3混合函数
uint64x2_t FastSimilaritySketch::neon_murmur3_mix(uint64x2_t h, uint64x2_t k) {
    uint64x2_t k_mul = vmulq_u64(k, vdupq_n_u64(C1));
    k_mul = vorrq_u64(vshlq_n_u64(k_mul, 31), vshrq_n_u64(k_mul, 33));
    k_mul = vmulq_u64(k_mul, vdupq_n_u64(C2));
    h = veorq_u64(h, k_mul);
    h = vorrq_u64(vshlq_n_u64(h, 27), vshrq_n_u64(h, 37));
    h = vaddq_u64(h, vdupq_n_u64(C3));
    h = vmulq_u64(h, vdupq_n_u64(5));
    return h;
}

#ifdef USE_AVX2
void FastSimilaritySketch::murmur_hash_avx2(const int* items, size_t count, 
                                          uint64_t seed, uint64_t* results) {
    constexpr size_t SIMD_WIDTH = 4; // 每次处理4个int(128位哈希)
    size_t i = 0;
    
    __m256i h1 = _mm256_set1_epi64x(seed);
    __m256i h2 = _mm256_set1_epi64x(seed);
    
    for (; i + SIMD_WIDTH <= count; i += SIMD_WIDTH) {
        // 加载4个int数据
        __m128i data = _mm_loadu_si128(reinterpret_cast<const __m128i*>(items + i));
        __m256i k = _mm256_cvtepu32_epi64(data);
        
        // MurmurHash3核心步骤
        h1 = avx2_murmur3_mix(h1, k);
        h2 = _mm256_rolv_epi64(h2, _mm256_set_epi64x(31, 31, 31, 31));
        h2 = _mm256_add_epi64(h2, h1);
        h2 = _mm256_mullo_epi64(h2, _mm256_set1_epi64x(5));
        h2 = _mm256_add_epi64(h2, _mm256_set1_epi64x(C4));
    }
    
    // 最终混合
    h1 = _mm256_xor_si256(h1, _mm256_set1_epi64x(count));
    h2 = _mm256_xor_si256(h2, _mm256_set1_epi64x(count));
    
    h1 = _mm256_add_epi64(h1, h2);
    h2 = _mm256_add_epi64(h2, h1);
    
    // 存储结果
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(results + i - SIMD_WIDTH), h1);
    
    // 处理剩余元素
    for (; i < count; ++i) {
        MurmurHash3_x64_128(&items[i], sizeof(int), seed, &results[i]);
    }
}
#elif defined(USE_SSE)
void FastSimilaritySketch::murmur_hash_sse(const int* items, size_t count, 
                                         uint64_t seed, uint64_t* results) {
    constexpr size_t SIMD_WIDTH = 2; // 每次处理2个int
    size_t i = 0;
    
    __m128i h1 = _mm_set1_epi64x(seed);
    __m128i h2 = _mm_set1_epi64x(seed);
    
    for (; i + SIMD_WIDTH <= count; i += SIMD_WIDTH) {
        // 加载2个int数据
        __m128i data = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(items + i));
        __m128i k = _mm_cvtepu32_epi64(data);
        
        // MurmurHash3核心步骤
        h1 = sse_murmur3_mix(h1, k);
        h2 = _mm_rol_epi64(h2, 31);
        h2 = _mm_add_epi64(h2, h1);
        h2 = _mm_mul_epu32(h2, _mm_set1_epi64x(5));
        h2 = _mm_add_epi64(h2, _mm_set1_epi64x(C4));
    }
    
    // 最终混合
    h1 = _mm_xor_si128(h1, _mm_set1_epi64x(count));
    h2 = _mm_xor_si128(h2, _mm_set1_epi64x(count));
    
    h1 = _mm_add_epi64(h1, h2);
    h2 = _mm_add_epi64(h2, h1);
    
    // 存储结果
    _mm_storeu_si128(reinterpret_cast<__m128i*>(results + i - SIMD_WIDTH), h1);
    
    // 处理剩余元素
    for (; i < count; ++i) {
        MurmurHash3_x64_128(&items[i], sizeof(int), seed, &results[i]);
    }
}
#elif defined(USE_NEON)
void FastSimilaritySketch::murmur_hash_neon(const int* items, size_t count, 
                                          uint64_t seed, uint64_t* results) {
    constexpr size_t SIMD_WIDTH = 2; // 每次处理2个int
    size_t i = 0;
    
    uint64x2_t h1 = vdupq_n_u64(seed);
    uint64x2_t h2 = vdupq_n_u64(seed);
    
    for (; i + SIMD_WIDTH <= count; i += SIMD_WIDTH) {
        // 加载2个int数据
        uint32x2_t data = vld1_u32(items + i);
        uint64x2_t k = vmovl_u32(data);
        
        // MurmurHash3核心步骤
        h1 = neon_murmur3_mix(h1, k);
        h2 = vorrq_u64(vshlq_n_u64(h2, 31), vshrq_n_u64(h2, 33));
        h2 = vaddq_u64(h2, h1);
        h2 = vmulq_u64(h2, vdupq_n_u64(5));
        h2 = vaddq_u64(h2, vdupq_n_u64(C4));
    }
    
    // 最终混合
    h1 = veorq_u64(h1, vdupq_n_u64(count));
    h2 = veorq_u64(h2, vdupq_n_u64(count));
    
    h1 = vaddq_u64(h1, h2);
    h2 = vaddq_u64(h2, h1);
    
    // 存储结果
    vst1q_u64(results + i - SIMD_WIDTH, h1);
    
    // 处理剩余元素
    for (; i < count; ++i) {
        MurmurHash3_x64_128(&items[i], sizeof(int), seed, &results[i]);
    }
}
#endif

void FastSimilaritySketch::murmur_hash_batch(const int* items, size_t count, 
                                           uint64_t seed, uint64_t* results) {
#ifdef USE_AVX2
    murmur_hash_avx2(items, count, seed, results);
#elif defined(USE_SSE)
    murmur_hash_sse(items, count, seed, results);
#elif defined(USE_NEON)
    murmur_hash_neon(items, count, seed, results);
#else
    // 标量回退
    for (size_t i = 0; i < count; ++i) {
        MurmurHash3_x64_128(&items[i], sizeof(int), seed, &results[i]);
    }
#endif
}

std::vector<uint64_t> FastSimilaritySketch::sketch(const std::vector<int>& items) {
    using SketchPair = std::pair<size_t, uint64_t>;
    std::vector<SketchPair> S(sketch_size, 
        {std::numeric_limits<size_t>::max(), std::numeric_limits<uint64_t>::max()});
    
    // 预分配批量哈希结果缓冲区
    constexpr size_t BATCH_SIZE = 64;
    uint64_t batch_hashes[BATCH_SIZE];
    
    for (size_t i = 0; i < hash_seeds.size(); ++i) {
        size_t processed = 0;
        
        // 批量处理
        while (processed < items.size()) {
            size_t remain = items.size() - processed;
            size_t curr_batch = std::min(BATCH_SIZE, remain);
            
            // SIMD批量哈希计算
            murmur_hash_batch(items.data() + processed, curr_batch, 
                            hash_seeds[i], batch_hashes);
            
            // 手动展开的比较循环
            for (size_t j = 0; j < curr_batch; ++j) {
                size_t b = (i < sketch_size) ? 
                          (batch_hashes[j] % sketch_size) : 
                          (i - sketch_size);
                auto v = std::make_pair(i, batch_hashes[j]);
                
                if (v < S[b]) {
                    S[b] = v;
                }
            }
            processed += curr_batch;
        }
    }
    
    // 提取结果
    std::vector<uint64_t> final_sketch;
    final_sketch.reserve(sketch_size);
    for (const auto& pair : S) {
        final_sketch.push_back(pair.second);
    }
    return final_sketch;
}
