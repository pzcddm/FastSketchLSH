// fast_similarity_sketch_avx512_packed.cpp
// 内存访问模式优化版本
// 
// 主要优化点：
// 1. 向量化桶更新：使用 AVX-512 gather/scatter 指令减少内存访问
// 2. 寄存器内操作：避免临时数组的内存读写
// 3. 条件编译：可选择使用优化版本或原始版本
// 4. 预期性能提升：15-25% (内存访问优化) + 20-30% (SIMD优化) = 总计 30-50%
//
#include "../include/fasthash_simd.h"
#include <immintrin.h>
#include <cstdint>
#include <vector>
#include <string>
#include <random>
#include <chrono>
#include <cstring>
#ifdef DEMO_MAIN
#include <iostream>
#include <cmath>
#include <limits>
#endif
using namespace std;

// Backend selection: AVX-512 if available, else scalar fallback; define FORCE_SCALAR to disable SIMD
#if defined(__AVX512F__) && !defined(FORCE_SCALAR)
#define FASTHASH_SIMD_AVX512 1
#else
#define FASTHASH_SIMD_SCALAR 1
#endif

// Hoisted SplitMix64 constants (one-time broadcast at TU scope)
#if defined(FASTHASH_SIMD_AVX512)
static const __m512i SPLITMIX_C1 = _mm512_set1_epi64(0x9E3779B97F4A7C15ull);
static const __m512i SPLITMIX_M1 = _mm512_set1_epi64(0xBF58476D1CE4E5B9ull);
static const __m512i SPLITMIX_M2 = _mm512_set1_epi64(0x94D049BB133111EBull);
#endif

inline uint64_t pack_key(uint64_t i, uint64_t h52) {
    return (i << I_SHIFT) | (h52 & H52_MASK);
}

#if defined(FASTHASH_SIMD_AVX512)
inline __m512i pack_key_vec(uint64_t i, __m512i h52) {
    // Broadcast i into a vector of 64-bit lanes and shift into top bits
    const __m512i vi = _mm512_set1_epi64((long long)i);
    const __m512i hi = _mm512_slli_epi64(vi, I_SHIFT);
    const __m512i mask = _mm512_set1_epi64((long long)H52_MASK);
    const __m512i lo = _mm512_and_si512(h52, mask);
    return _mm512_or_si512(hi, lo);
}
#endif

// Note: scalar hashing utilities are defined inline in the public header
// (fnv1a64, hash_int32, splitmix64). We only keep the vectorized helpers here.

// splitmix64（AVX-512, 8-lane for uint64）
#if defined(FASTHASH_SIMD_AVX512)
inline __m512i splitmix64_vec(__m512i x){
    x = _mm512_add_epi64(x, SPLITMIX_C1);
    __m512i t = _mm512_xor_si512(x, _mm512_srli_epi64(x, 30));
    t = _mm512_mullo_epi64(t, SPLITMIX_M1);
    t = _mm512_xor_si512(t, _mm512_srli_epi64(t, 27));
    t = _mm512_mullo_epi64(t, SPLITMIX_M2);
    t = _mm512_xor_si512(t, _mm512_srli_epi64(t, 31));
    return t;
}
#endif

// Horizontal min for 8 lanes of unsigned 64-bit using two shuffles + min
#if defined(FASTHASH_SIMD_AVX512)
inline uint64_t hmin_epu64_8(__m512i v) {
    __m512i t = _mm512_shuffle_i64x2(v, v, 0x4E);
    v = _mm512_min_epu64(v, t);
    t = _mm512_shuffle_i64x2(v, v, 0xB1);
    v = _mm512_min_epu64(v, t);
    return (uint64_t)_mm_cvtsi128_si64(_mm512_castsi512_si128(v));
}
#endif

// Hash 8 int32 values into 8 uint64 using SplitMix64-style mixing
inline void hash_int32x8_to_u64_avx512(const uint32_t* src, uint64_t* dst) {
#if defined(FASTHASH_SIMD_AVX512)
    __m256i v32 = _mm256_loadu_si256((const __m256i*)src);
    __m512i x64 = _mm512_cvtepu32_epi64(v32);                 // zero-extends
    __m512i h = splitmix64_vec(x64);
    _mm512_storeu_si512((void*)dst, h);
#else
    for (int i = 0; i < 8; ++i) dst[i] = splitmix64((uint64_t)src[i]);
#endif
}

// SIMD check: whether all buckets in S are filled (i.e., not INF_KEY)
inline bool all_filled_avx512(const uint64_t* S, int t) {
#if defined(FASTHASH_SIMD_AVX512)
    int i = 0;
    const __m512i inf = _mm512_set1_epi64((long long)INF_KEY());
    for (; i + 8 <= t; i += 8) {
        __m512i v = _mm512_loadu_si512((const void*)(S + i));
        // Compare equal to INF_KEY; if any bit set, there exists an empty bucket
        __mmask8 meq = _mm512_cmpeq_epu64_mask(v, inf);
        if (meq) return false;
    }
    for (; i < t; ++i) {
        if (S[i] == INF_KEY()) return false;
    }
    return true;
#else
    for (int i = 0; i < t; ++i) if (S[i] == INF_KEY()) return false;
    return true;
#endif
}

// ===================== 向量化桶更新函数 =====================
// 使用 AVX-512 gather/scatter 指令，减少内存访问
#if defined(FASTHASH_SIMD_AVX512)
inline void update_buckets_vectorized(__m512i keys, __m512i buckets, uint64_t* S) {
    // 使用 gather 指令加载当前S值，避免多次内存访问
    __m512i current_S = _mm512_i64gather_epi64(buckets, S, 8);
    
    // 比较并选择最小值
    __mmask8 mask = _mm512_cmplt_epu64_mask(keys, current_S);
    
    // 条件更新：只在需要时写入，使用 scatter 指令
    if (mask) {
        _mm512_mask_i64scatter_epi64(S, mask, buckets, keys, 8);
    }
}
#endif

// ===================== 第 1 轮：AVX-512 批算 + 向量化更新（优化版本） =====================
// 优化说明：减少内存搬移，使用寄存器操作，向量化桶更新
#if defined(FASTHASH_SIMD_AVX512)
inline void round1_block_avx512_optimized(
    const uint64_t* base_block, int nlanes,
    uint64_t round_i, uint64_t seed_i,
    uint64_t* S,
    uint64_t t_mask)
{
    if (nlanes == 8) {
        // 直接在寄存器中处理，避免内存搬移
        __m512i x = _mm512_loadu_si512((const void*)base_block);
        __m512i seed_vec = _mm512_set1_epi64((long long)seed_i);
        __m512i round_vec = _mm512_set1_epi64((long long)round_i);
        __m512i t_mask_vec = _mm512_set1_epi64((long long)t_mask);
        
        // 计算哈希值
        __m512i h = splitmix64_vec(_mm512_xor_si512(x, seed_vec));
        
        // 计算桶索引
        __m512i b = _mm512_and_si512(h, t_mask_vec);
        
        // 打包键值
        __m512i keys = pack_key_vec(round_i, h);
        
        // 向量化更新S数组（关键优化）
        update_buckets_vectorized(keys, b, S);
    } else {
        // 标量处理剩余元素
        for (int k = 0; k < nlanes; k++) {
            uint64_t h = splitmix64(base_block[k] ^ seed_i);
            uint64_t b = h & t_mask;
            uint64_t key = pack_key(round_i, h);
            
            if (key < S[b]) {
                S[b] = key;
            }
        }
    }
}
#endif

// Scalar fallback for round1 block processing
inline void round1_block_fallback(
    const uint64_t* base_block, int nlanes,
    uint64_t round_i, uint64_t seed_i,
    uint64_t* S,
    uint64_t t_mask)
{
    for (int k = 0; k < nlanes; ++k) {
        uint64_t h = splitmix64(base_block[k] ^ seed_i);
        uint64_t b = h & t_mask;
        uint64_t key = pack_key(round_i, h);
        if (key < S[b]) S[b] = key;
    }
}

// ===================== 第 1 轮：AVX-512 批算 + 逐 lane 更新 =====================
// 说明：不做批内 reduce-by-bucket；对每个 lane 顺序：读 S[b] → 比较 → 写 S[b]。
#if defined(FASTHASH_SIMD_AVX512)
inline void round1_block_avx512_no_reduce(
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
    const __m512i& h52maskv)
{

    if (nlanes == 8) {
        __m512i x = _mm512_loadu_si512((const void*)base_block);
        x = _mm512_xor_si512(x, seedv);
        __m512i h = splitmix64_vec(x);
        __m512i b = _mm512_and_si512(h, maskv);
        _mm512_store_si512((void*)b_lane, b);
        __m512i kv = _mm512_or_si512(hiv, _mm512_and_si512(h, h52maskv));
        _mm512_store_si512((void*)key_lane, kv);
    } else {
        for (int k=0;k<nlanes;k++){
            uint64_t h = splitmix64(base_block[k] ^ seed_i);
            h_lane[k] = h;
            b_lane[k] = h & t_mask;
        }
    }
    if (nlanes != 8) {
        for (int k=0;k<nlanes;k++) {
            key_lane[k] = pack_key(round_i, h_lane[k]);
        }
    }

    // 逐 lane 更新
    for (int k=0;k<nlanes;k++){
        const uint64_t b = b_lane[k];
        const uint64_t cand = key_lane[k];
        const uint64_t old  = S[b];
        if (cand < old) {
            S[b] = cand; 
        }
    }
}
#endif

// ===================== 主类：2t 轮（packed key 版本） =====================

FastSimilaritySketchAVX512Packed::FastSimilaritySketchAVX512Packed(int sketch_size, uint64_t random_seed)
    : t(sketch_size), t_mask(sketch_size-1), seeds(2*sketch_size)
{
    if (t<=0 || (t & (t-1))!=0) throw runtime_error("t must be a power of two.");
    std::mt19937_64 rng(random_seed);
    for (int i=0;i<2*t;i++) seeds[i] = rng();
    if ((uint64_t)t > (1ull<<I_BITS))
        throw runtime_error("t can not be larger than 4096.");
    base_buffer.reserve(16384); // initial enough capacity; grows as needed
}

vector<uint64_t> FastSimilaritySketchAVX512Packed::sketch(const vector<uint32_t>& A) {
    return sketch(A, nullptr, nullptr, nullptr);
}

vector<uint64_t> FastSimilaritySketchAVX512Packed::sketch(const vector<uint32_t>& A,
                                                          double* prehash_ms,
                                                          double* phase1_ms,
                                                          double* phase2_ms) {
    const int n = (int)A.size();

    // 0) 预哈希（一次），避免 2t 次扫描长串
    auto t0 = std::chrono::high_resolution_clock::now();
    if (base_buffer.size() < static_cast<size_t>(n)) base_buffer.resize(n);
    uint64_t* base_ptr = base_buffer.data();
    int j0 = 0;
    for (; j0 + 8 <= n; j0 += 8) {
        hash_int32x8_to_u64_avx512(&A[j0], &base_ptr[j0]);
    }
    for (; j0 < n; ++j0) {
        base_ptr[j0] = hash_int32(A[j0]);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    if (prehash_ms) *prehash_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // 1) Buckets: one group (packed key)
    if (buckets_S.size() < static_cast<size_t>(t)) buckets_S.resize(t);
    std::memset(buckets_S.data(), 0xFF, static_cast<size_t>(t) * sizeof(uint64_t));
    // ==================== 第 0 ~ t -1 轮：i=0..t-1 ====================
    // when A.size() is larger than 600 (if t is almost 128), usually i = 0 the buckets are all filled.
    auto p1_start = std::chrono::high_resolution_clock::now();
    
#if defined(FASTHASH_SIMD_AVX512)
    const __m512i maskv = _mm512_set1_epi64((long long)t_mask);
    const __m512i h52maskv = _mm512_set1_epi64((long long)H52_MASK);
#endif
    for (int i=0; i<t; ++i) {
        const uint64_t seed_i = seeds[i];
#if defined(FASTHASH_SIMD_AVX512)
        const __m512i seedv = _mm512_set1_epi64((long long)seed_i);
        const __m512i hiv = _mm512_set1_epi64((long long)((uint64_t)i << I_SHIFT));
#endif

        int j = 0;
        // Process blocks with SIMD if available, otherwise scalar
#if defined(FASTHASH_SIMD_AVX512)
        for (; j+16<=n; j+=16) {
            round1_block_avx512_optimized(&base_ptr[j], 8, (uint64_t)i, seed_i,
                                          buckets_S.data(), t_mask);
            round1_block_avx512_optimized(&base_ptr[j+8], 8, (uint64_t)i, seed_i,
                                          buckets_S.data(), t_mask);
        }
        for (; j+8<=n; j+=8) {
            round1_block_avx512_optimized(&base_ptr[j], 8, (uint64_t)i, seed_i,
                                          buckets_S.data(), t_mask);
        }
        if (j < n) {
            round1_block_avx512_optimized(&base_ptr[j], n-j, (uint64_t)i, seed_i,
                                          buckets_S.data(), t_mask);
        }
#else
        for (; j+16<=n; j+=16) {
            round1_block_fallback(&base_ptr[j], 8, (uint64_t)i, seed_i, buckets_S.data(), t_mask);
            round1_block_fallback(&base_ptr[j+8], 8, (uint64_t)i, seed_i, buckets_S.data(), t_mask);
        }
        for (; j+8<=n; j+=8) {
            round1_block_fallback(&base_ptr[j], 8, (uint64_t)i, seed_i, buckets_S.data(), t_mask);
        }
        if (j < n) {
            round1_block_fallback(&base_ptr[j], n-j, (uint64_t)i, seed_i, buckets_S.data(), t_mask);
        }
#endif

        // End of round: check whether all buckets are filled
        if (all_filled_avx512(buckets_S.data(), t)) break;
    }
    auto p1_end = std::chrono::high_resolution_clock::now();
    if (phase1_ms) *phase1_ms = std::chrono::duration<double, std::milli>(p1_end - p1_start).count();

    // ==================== 第 t ~ 2t -1 轮：i=t..2t-1，只补空桶 ====================
    // 因为 key 的高位是 i，i>=t 的 key 一定大于第 1 轮写入的 key，
    // 所以这里只会写原先空桶（S[b]==INF_KEY），不会覆盖已有桶。
    auto p2_start = std::chrono::high_resolution_clock::now();
    if (!all_filled_avx512(buckets_S.data(), t)) {
        alignas(64) uint64_t h_lane[8];

        for (int i=t; i<2*t; ++i) {
            const int b = i - t;
            if (buckets_S[b] != INF_KEY()) continue; // Already filled bucket
            const uint64_t seed_i = seeds[i];

            // 在所有元素上找 min_h
            uint64_t min_h = ~0ull;
            int j = 0;
#if defined(FASTHASH_SIMD_AVX512)
            for (; j+8<=n; j+=8) {
                __m512i x = _mm512_loadu_si512((const void*)&base_ptr[j]);
                x = _mm512_xor_si512(x, _mm512_set1_epi64((long long)seed_i));
                __m512i h = splitmix64_vec(x);
                uint64_t block_min = hmin_epu64_8(h);
                if (block_min < min_h) min_h = block_min;
            }
#endif
            for (; j<n; ++j) {
                uint64_t h = splitmix64(base_ptr[j] ^ seed_i);
                if (h < min_h) min_h = h;
            }

#ifdef PREFETCH_BUCKET
            _mm_prefetch((const char*)&buckets_S[b], _MM_HINT_T0);
#endif
            const uint64_t key = pack_key((uint64_t)i, min_h);
            if (key < buckets_S[b]) {
                buckets_S[b] = key;
            }
        }

        // All filled check (optional)
        (void)all_filled_avx512;
    }
    auto p2_end = std::chrono::high_resolution_clock::now();
    if (phase2_ms) *phase2_ms = std::chrono::duration<double, std::milli>(p2_end - p2_start).count();
    return std::vector<uint64_t>(buckets_S.begin(), buckets_S.begin() + t);
}

// ===================== String/bytes overloads =====================
vector<uint64_t> FastSimilaritySketchAVX512Packed::sketch(const vector<string>& bytes) {
    return sketch(bytes, nullptr, nullptr, nullptr);
}

vector<uint64_t> FastSimilaritySketchAVX512Packed::sketch(const vector<string>& bytes,
                                                          double* prehash_ms,
                                                          double* phase1_ms,
                                                          double* phase2_ms) {
    const int n = (int)bytes.size();

    // 0) Prehash each string once using FNV1a-64 to a 64-bit base
    auto t0 = std::chrono::high_resolution_clock::now();
    if (base_buffer.size() < static_cast<size_t>(n)) base_buffer.resize(n);
    uint64_t* base_ptr = base_buffer.data();
    for (int j = 0; j < n; ++j) {
        const string& s = bytes[j];
        base_ptr[j] = fnv1a64(reinterpret_cast<const uint8_t*>(s.data()), s.size());
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    if (prehash_ms) *prehash_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // 1) Buckets: one group (packed key)
    if (buckets_S.size() < static_cast<size_t>(t)) buckets_S.resize(t);
    std::memset(buckets_S.data(), 0xFF, static_cast<size_t>(t) * sizeof(uint64_t));

    auto p1_start = std::chrono::high_resolution_clock::now();
#if defined(FASTHASH_SIMD_AVX512)
    const __m512i maskv = _mm512_set1_epi64((long long)t_mask);
    const __m512i h52maskv = _mm512_set1_epi64((long long)H52_MASK);
#endif
    for (int i=0; i<t; ++i) {
        const uint64_t seed_i = seeds[i];
#if defined(FASTHASH_SIMD_AVX512)
        const __m512i seedv = _mm512_set1_epi64((long long)seed_i);
        const __m512i hiv = _mm512_set1_epi64((long long)((uint64_t)i << I_SHIFT));
#endif

        int j = 0;
#if defined(FASTHASH_SIMD_AVX512)
        for (; j+16<=n; j+=16) {
            round1_block_avx512_optimized(&base_ptr[j], 8, (uint64_t)i, seed_i,
                                          buckets_S.data(), t_mask);
            round1_block_avx512_optimized(&base_ptr[j+8], 8, (uint64_t)i, seed_i,
                                          buckets_S.data(), t_mask);
        }
        for (; j+8<=n; j+=8) {
            round1_block_avx512_optimized(&base_ptr[j], 8, (uint64_t)i, seed_i,
                                          buckets_S.data(), t_mask);
        }
        if (j < n) {
            round1_block_avx512_optimized(&base_ptr[j], n-j, (uint64_t)i, seed_i,
                                          buckets_S.data(), t_mask);
        }
#else
        for (; j+16<=n; j+=16) {
            round1_block_fallback(&base_ptr[j], 8, (uint64_t)i, seed_i, buckets_S.data(), t_mask);
            round1_block_fallback(&base_ptr[j+8], 8, (uint64_t)i, seed_i, buckets_S.data(), t_mask);
        }
        for (; j+8<=n; j+=8) {
            round1_block_fallback(&base_ptr[j], 8, (uint64_t)i, seed_i, buckets_S.data(), t_mask);
        }
        if (j < n) {
            round1_block_fallback(&base_ptr[j], n-j, (uint64_t)i, seed_i, buckets_S.data(), t_mask);
        }
#endif
        if (all_filled_avx512(buckets_S.data(), t)) break;
    }
    auto p1_end = std::chrono::high_resolution_clock::now();
    if (phase1_ms) *phase1_ms = std::chrono::duration<double, std::milli>(p1_end - p1_start).count();

    // 2) Second phase for empty buckets only
    auto p2_start = std::chrono::high_resolution_clock::now();
    if (!all_filled_avx512(buckets_S.data(), t)) {
        for (int i=t; i<2*t; ++i) {
            const int b = i - t;
            if (buckets_S[b] != INF_KEY()) continue;
            const uint64_t seed_i = seeds[i];

            uint64_t min_h = ~0ull;
            int j = 0;
#if defined(FASTHASH_SIMD_AVX512)
            for (; j+8<=n; j+=8) {
                __m512i x = _mm512_loadu_si512((const void*)&base_ptr[j]);
                x = _mm512_xor_si512(x, _mm512_set1_epi64((long long)seed_i));
                __m512i h = splitmix64_vec(x);
                uint64_t block_min = hmin_epu64_8(h);
                if (block_min < min_h) min_h = block_min;
            }
#endif
            for (; j<n; ++j) {
                uint64_t h = splitmix64(base_ptr[j] ^ seed_i);
                if (h < min_h) min_h = h;
            }

            const uint64_t key = pack_key((uint64_t)i, min_h);
            if (key < buckets_S[b]) {
                buckets_S[b] = key;
            }
        }
    }
    auto p2_end = std::chrono::high_resolution_clock::now();
    if (phase2_ms) *phase2_ms = std::chrono::duration<double, std::milli>(p2_end - p2_start).count();
    return std::vector<uint64_t>(buckets_S.begin(), buckets_S.begin() + t);
}

// ===================== Demo =====================
#ifdef DEMO_MAIN

// To compile this file you can use this command to test it:
// g++ -O3 -std=c++17 -mavx512f -mavx512dq -mavx512vl -DDEMO_MAIN cpp_src/cpp/fasthash_simd.cpp -Icpp_src/include -o demo_fasthash_simd.exe
int main(){
    // Generate two integer sets:
    // A = {0, 1, ..., 7499}
    // B = {2500, 2501, ..., 9999}
    vector<uint32_t> A; A.reserve(7500);
    for (uint32_t i = 0; i < 7500u; ++i) A.push_back(i);
    vector<uint32_t> B; B.reserve(7500);
    for (uint32_t i = 2500u; i < 10000u; ++i) B.push_back(i);

    // Compute true Jaccard via two-pointer merge (both vectors are sorted)
    int inter = 0;
    size_t ia = 0, ib = 0;
    while (ia < A.size() && ib < B.size()) {
        if (A[ia] == B[ib]) { ++inter; ++ia; ++ib; }
        else if (A[ia] < B[ib]) { ++ia; }
        else { ++ib; }
    }
    const int uni = static_cast<int>(A.size() + B.size() - inter);
    const double j_true = uni > 0 ? static_cast<double>(inter) / static_cast<double>(uni) : 0.0;

    const int t = 256;      // sketch size (power of two, <= 4096)
    const int trials = 50;  // number of repetitions with different random seeds

    std::random_device rd;
    std::mt19937_64 seed_rng(rd());
    std::uniform_int_distribution<uint64_t> dist(0ull, std::numeric_limits<uint64_t>::max());

    double total_error = 0.0;
    double total_ms_A = 0.0;
    double total_ms_B = 0.0;
    double total_prehash_A = 0.0, total_phase1_A = 0.0, total_phase2_A = 0.0;
    double total_prehash_B = 0.0, total_phase1_B = 0.0, total_phase2_B = 0.0;

    for (int trial = 0; trial < trials; ++trial) {
        const uint64_t seed = dist(seed_rng);
        FastSimilaritySketchAVX512Packed sketcher(t, seed);

        // Time sketch generation for A and collect per-phase timings
        double prehashA=0.0, phase1A=0.0, phase2A=0.0;
        auto t0 = std::chrono::high_resolution_clock::now();
        auto S_A = sketcher.sketch(A, &prehashA, &phase1A, &phase2A);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms_A = std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_prehash_A += prehashA; total_phase1_A += phase1A; total_phase2_A += phase2A;

        // Time sketch generation for B and collect per-phase timings
        double prehashB=0.0, phase1B=0.0, phase2B=0.0;
        auto t2 = std::chrono::high_resolution_clock::now();
        auto S_B = sketcher.sketch(B, &prehashB, &phase1B, &phase2B);
        auto t3 = std::chrono::high_resolution_clock::now();
        double ms_B = std::chrono::duration<double, std::milli>(t3 - t2).count();
        total_prehash_B += prehashB; total_phase1_B += phase1B; total_phase2_B += phase2B;

        // Estimate Jaccard from sketches: fraction of equal entries
        int matches = 0;
        for (int i = 0; i < t; ++i) if (S_A[i] == S_B[i]) ++matches;
        double j_est = static_cast<double>(matches) / static_cast<double>(t);

        std::cout << "trial " << (trial+1) << ": "
                  << "A[prehash=" << prehashA << " ms, phase1=" << phase1A << " ms, phase2=" << phase2A << " ms], "
                  << "B[prehash=" << prehashB << " ms, phase1=" << phase1B << " ms, phase2=" << phase2B << " ms], "
                  << "estJ=" << j_est << "\n";

        total_error += std::abs(j_est - j_true);
        total_ms_A += ms_A;
        total_ms_B += ms_B;
    }

    const double mean_error = total_error / static_cast<double>(trials);
    const double mean_ms_A = total_ms_A / static_cast<double>(trials);
    const double mean_ms_B = total_ms_B / static_cast<double>(trials);
    const double mean_prehash_A = total_prehash_A / static_cast<double>(trials);
    const double mean_phase1_A  = total_phase1_A  / static_cast<double>(trials);
    const double mean_phase2_A  = total_phase2_A  / static_cast<double>(trials);
    const double mean_prehash_B = total_prehash_B / static_cast<double>(trials);
    const double mean_phase1_B  = total_phase1_B  / static_cast<double>(trials);
    const double mean_phase2_B  = total_phase2_B  / static_cast<double>(trials);

    std::cout << "|A| = " << A.size() << ", |B| = " << B.size() << "\n";
    std::cout << "Sketch size t = " << t << ", trials = " << trials << "\n";
    std::cout << "Mean time: A = " << mean_ms_A << " ms, B = " << mean_ms_B << " ms, total = " << (mean_ms_A + mean_ms_B) << " ms\n";
    std::cout << "Mean phases A: prehash = " << mean_prehash_A << " ms, phase1 = " << mean_phase1_A << " ms, phase2 = " << mean_phase2_A << " ms\n";
    std::cout << "Mean phases B: prehash = " << mean_prehash_B << " ms, phase1 = " << mean_phase1_B << " ms, phase2 = " << mean_phase2_B << " ms\n";
    std::cout << "True Jaccard: " << j_true << "\n";
    std::cout << "Mean absolute error over trials: " << mean_error << "\n";
    return 0;
}
#endif
