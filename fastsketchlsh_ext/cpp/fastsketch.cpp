// fastsketch.cpp (formerly fasthash_simd.cpp)
// Unified SIMD implementation backing FastSimilaritySketch

#include "../include/fastsketch.h"
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif
#include <cstdint>
#include <vector>
#include <string>
#include <random>
#include <chrono>
#include <cstring>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef DEMO_MAIN
#include <iostream>
#include <cmath>
#include <limits>
#endif
using namespace std;

#if defined(__AVX512F__) && !defined(FORCE_SCALAR)
#define FASTHASH_SIMD_AVX512 1
#else
#define FASTHASH_SIMD_SCALAR 1
#endif

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
    const __m512i vi = _mm512_set1_epi64((long long)i);
    const __m512i hi = _mm512_slli_epi64(vi, I_SHIFT);
    const __m512i mask = _mm512_set1_epi64((long long)H52_MASK);
    const __m512i lo = _mm512_and_si512(h52, mask);
    return _mm512_or_si512(hi, lo);
}
#endif

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

#if defined(FASTHASH_SIMD_AVX512)
inline uint64_t hmin_epu64_8(__m512i v) {
    __m512i t = _mm512_shuffle_i64x2(v, v, 0x4E);
    v = _mm512_min_epu64(v, t);
    t = _mm512_shuffle_i64x2(v, v, 0xB1);
    v = _mm512_min_epu64(v, t);
    return (uint64_t)_mm_cvtsi128_si64(_mm512_castsi512_si128(v));
}
#endif

inline void hash_int32x8_to_u64_avx512(const uint32_t* src, uint64_t* dst) {
#if defined(FASTHASH_SIMD_AVX512)
    __m256i v32 = _mm256_loadu_si256((const __m256i*)src);
    __m512i x64 = _mm512_cvtepu32_epi64(v32);
    __m512i h = splitmix64_vec(x64);
    _mm512_storeu_si512((void*)dst, h);
#else
    for (int i = 0; i < 8; ++i) dst[i] = splitmix64((uint64_t)src[i]);
#endif
}

inline bool all_filled_avx512(const uint64_t* S, int t) {
#if defined(FASTHASH_SIMD_AVX512)
    int i = 0;
    const __m512i inf = _mm512_set1_epi64((long long)INF_KEY());
    for (; i + 8 <= t; i += 8) {
        __m512i v = _mm512_loadu_si512((const void*)(S + i));
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

#if defined(FASTHASH_SIMD_AVX512)
inline void update_buckets_vectorized(__m512i keys, __m512i buckets, uint64_t* S) {
    __m512i current_S = _mm512_i64gather_epi64(buckets, S, 8);
    __mmask8 mask = _mm512_cmplt_epu64_mask(keys, current_S);
    if (mask) {
        _mm512_mask_i64scatter_epi64(S, mask, buckets, keys, 8);
    }
}
#endif

#if defined(FASTHASH_SIMD_AVX512)
inline void round1_block_avx512_optimized(
    const uint64_t* base_block, int nlanes,
    uint64_t round_i, uint64_t seed_i,
    uint64_t* S,
    uint64_t t_mask)
{
    if (nlanes == 8) {
        __m512i x = _mm512_loadu_si512((const void*)base_block);
        __m512i seed_vec = _mm512_set1_epi64((long long)seed_i);
        __m512i t_mask_vec = _mm512_set1_epi64((long long)t_mask);
        __m512i h = splitmix64_vec(_mm512_xor_si512(x, seed_vec));
        __m512i b = _mm512_and_si512(h, t_mask_vec);
        __m512i keys = pack_key_vec(round_i, h);
        update_buckets_vectorized(keys, b, S);
    } else {
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
        for (int k=0;k<nlanes;k++) {
            key_lane[k] = pack_key(round_i, h_lane[k]);
        }
    }
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

FastSimilaritySketch::FastSimilaritySketch(int sketch_size, uint64_t random_seed)
    : t(sketch_size), t_mask(sketch_size-1), seeds(2*sketch_size)
{
    if (t<=0 || (t & (t-1))!=0) throw runtime_error("t must be a power of two.");
    std::mt19937_64 rng(random_seed);
    for (int i=0;i<2*t;i++) seeds[i] = rng();
    if ((uint64_t)t > (1ull<<I_BITS))
        throw runtime_error("t can not be larger than 4096.");
    base_buffer.reserve(16384);
    buckets_S.resize(static_cast<size_t>(t));
}

FastSimilaritySketch::FastSimilaritySketch(int sketch_size, const std::vector<uint64_t>& precomputed_seeds)
    : t(sketch_size), t_mask(sketch_size-1), seeds(precomputed_seeds)
{
    base_buffer.reserve(16384);
    buckets_S.resize(static_cast<size_t>(t));
}

vector<uint64_t> FastSimilaritySketch::sketch(const vector<uint32_t>& A) {
    return sketch(A, nullptr, nullptr, nullptr);
}
vector<uint64_t> FastSimilaritySketch::sketch(const uint32_t* data, size_t n) {
    const int nn = static_cast<int>(n);
    auto t0 = std::chrono::high_resolution_clock::now();
    if (base_buffer.size() < n) base_buffer.resize(n);
    uint64_t* base_ptr = base_buffer.data();
    int j0 = 0;
    for (; j0 + 8 <= nn; j0 += 8) {
        hash_int32x8_to_u64_avx512(reinterpret_cast<const uint32_t*>(data + j0), &base_ptr[j0]);
    }
    for (; j0 < nn; ++j0) {
        base_ptr[j0] = hash_int32(data[j0]);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    (void)t0; (void)t1;

    std::memset(buckets_S.data(), 0xFF, static_cast<size_t>(t) * sizeof(uint64_t));

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
        for (; j+16<=nn; j+=16) {
            round1_block_avx512_optimized(&base_ptr[j], 8, (uint64_t)i, seed_i, buckets_S.data(), t_mask);
            round1_block_avx512_optimized(&base_ptr[j+8], 8, (uint64_t)i, seed_i, buckets_S.data(), t_mask);
        }
        for (; j+8<=nn; j+=8) {
            round1_block_avx512_optimized(&base_ptr[j], 8, (uint64_t)i, seed_i, buckets_S.data(), t_mask);
        }
        if (j < nn) {
            round1_block_avx512_optimized(&base_ptr[j], nn-j, (uint64_t)i, seed_i, buckets_S.data(), t_mask);
        }
#else
        for (; j+16<=nn; j+=16) {
            round1_block_fallback(&base_ptr[j], 8, (uint64_t)i, seed_i, buckets_S.data(), t_mask);
            round1_block_fallback(&base_ptr[j+8], 8, (uint64_t)i, seed_i, buckets_S.data(), t_mask);
        }
        for (; j+8<=nn; j+=8) {
            round1_block_fallback(&base_ptr[j], 8, (uint64_t)i, seed_i, buckets_S.data(), t_mask);
        }
        if (j < nn) {
            round1_block_fallback(&base_ptr[j], nn-j, (uint64_t)i, seed_i, buckets_S.data(), t_mask);
        }
#endif
        if (all_filled_avx512(buckets_S.data(), t)) break;
    }

    if (!all_filled_avx512(buckets_S.data(), t)) {
        for (int i=t; i<2*t; ++i) {
            const int b = i - t;
            if (buckets_S[b] != INF_KEY()) continue;
            const uint64_t seed_i = seeds[i];
            uint64_t min_h = ~0ull;
            int j = 0;
#if defined(FASTHASH_SIMD_AVX512)
            for (; j+8<=nn; j+=8) {
                __m512i x = _mm512_loadu_si512((const void*)&base_ptr[j]);
                x = _mm512_xor_si512(x, _mm512_set1_epi64((long long)seed_i));
                __m512i h = splitmix64_vec(x);
                uint64_t block_min = hmin_epu64_8(h);
                if (block_min < min_h) min_h = block_min;
            }
#endif
            for (; j<nn; ++j) {
                uint64_t h = splitmix64(base_ptr[j] ^ seed_i);
                if (h < min_h) min_h = h;
            }
            const uint64_t key = pack_key((uint64_t)i, min_h);
            if (key < buckets_S[b]) {
                buckets_S[b] = key;
            }
        }
    }
    last_digest.assign(buckets_S.begin(), buckets_S.begin() + t);
    return last_digest;
}

vector<uint64_t> FastSimilaritySketch::sketch(const vector<int>& A) {
    vector<uint32_t> conv;
    conv.reserve(A.size());
    for (int v : A) {
        if (v < 0) throw runtime_error("FastSimilaritySketch requires non-negative integers");
        conv.push_back(static_cast<uint32_t>(v));
    }
    return sketch(conv);
}

vector<uint64_t> FastSimilaritySketch::sketch(const vector<uint32_t>& A,
                                              double* prehash_ms,
                                              double* phase1_ms,
                                              double* phase2_ms) {
    const int n = (int)A.size();
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
            round1_block_avx512_optimized(&base_ptr[j], 8, (uint64_t)i, seed_i, buckets_S.data(), t_mask);
            round1_block_avx512_optimized(&base_ptr[j+8], 8, (uint64_t)i, seed_i, buckets_S.data(), t_mask);
        }
        for (; j+8<=n; j+=8) {
            round1_block_avx512_optimized(&base_ptr[j], 8, (uint64_t)i, seed_i, buckets_S.data(), t_mask);
        }
        if (j < n) {
            round1_block_avx512_optimized(&base_ptr[j], n-j, (uint64_t)i, seed_i, buckets_S.data(), t_mask);
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
    last_digest.assign(buckets_S.begin(), buckets_S.begin() + t);
    return last_digest;
}

vector<uint64_t> FastSimilaritySketch::sketch(const vector<string>& bytes) {
    return sketch(bytes, nullptr, nullptr, nullptr);
}

vector<uint64_t> FastSimilaritySketch::sketch(const vector<string>& bytes,
                                              double* prehash_ms,
                                              double* phase1_ms,
                                              double* phase2_ms) {
    const int n = (int)bytes.size();
    auto t0 = std::chrono::high_resolution_clock::now();
    if (base_buffer.size() < static_cast<size_t>(n)) base_buffer.resize(n);
    uint64_t* base_ptr = base_buffer.data();
    for (int j = 0; j < n; ++j) {
        const string& s = bytes[j];
#if defined(FASTSKETCH_USE_FXHASH)
        base_ptr[j] = fxhash64(reinterpret_cast<const uint8_t*>(s.data()), s.size());
#else
        base_ptr[j] = fnv1a64(reinterpret_cast<const uint8_t*>(s.data()), s.size());
#endif
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    if (prehash_ms) *prehash_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

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
            round1_block_avx512_optimized(&base_ptr[j], 8, (uint64_t)i, seed_i, buckets_S.data(), t_mask);
            round1_block_avx512_optimized(&base_ptr[j+8], 8, (uint64_t)i, seed_i, buckets_S.data(), t_mask);
        }
        for (; j+8<=n; j+=8) {
            round1_block_avx512_optimized(&base_ptr[j], 8, (uint64_t)i, seed_i, buckets_S.data(), t_mask);
        }
        if (j < n) {
            round1_block_avx512_optimized(&base_ptr[j], n-j, (uint64_t)i, seed_i, buckets_S.data(), t_mask);
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
    last_digest.assign(buckets_S.begin(), buckets_S.begin() + t);
    return last_digest;
}

vector<uint64_t> FastSimilaritySketch::sketch_utf8_views(const uint8_t* const* ptrs,
                                                         const size_t* lengths,
                                                         size_t n) {
    const int nn = static_cast<int>(n);
    if (base_buffer.size() < n) base_buffer.resize(n);
    uint64_t* base_ptr = base_buffer.data();
    for (size_t j = 0; j < n; ++j) {
#if defined(FASTSKETCH_USE_FXHASH)
        base_ptr[j] = fxhash64(ptrs[j], lengths[j]);
#else
        base_ptr[j] = fnv1a64(ptrs[j], lengths[j]);
#endif
    }

    std::memset(buckets_S.data(), 0xFF, static_cast<size_t>(t) * sizeof(uint64_t));

#if defined(FASTHASH_SIMD_AVX512)
    const __m512i maskv = _mm512_set1_epi64((long long)t_mask);
    const __m512i h52maskv = _mm512_set1_epi64((long long)H52_MASK);
#endif
    for (int i = 0; i < t; ++i) {
        const uint64_t seed_i = seeds[i];
#if defined(FASTHASH_SIMD_AVX512)
        const __m512i seedv = _mm512_set1_epi64((long long)seed_i);
        const __m512i hiv = _mm512_set1_epi64((long long)((uint64_t)i << I_SHIFT));
#endif
        int j = 0;
#if defined(FASTHASH_SIMD_AVX512)
        for (; j + 16 <= nn; j += 16) {
            round1_block_avx512_optimized(&base_ptr[j], 8, (uint64_t)i, seed_i, buckets_S.data(), t_mask);
            round1_block_avx512_optimized(&base_ptr[j + 8], 8, (uint64_t)i, seed_i, buckets_S.data(), t_mask);
        }
        for (; j + 8 <= nn; j += 8) {
            round1_block_avx512_optimized(&base_ptr[j], 8, (uint64_t)i, seed_i, buckets_S.data(), t_mask);
        }
        if (j < nn) {
            round1_block_avx512_optimized(&base_ptr[j], nn - j, (uint64_t)i, seed_i, buckets_S.data(), t_mask);
        }
#else
        for (; j + 16 <= nn; j += 16) {
            round1_block_fallback(&base_ptr[j], 8, (uint64_t)i, seed_i, buckets_S.data(), t_mask);
            round1_block_fallback(&base_ptr[j + 8], 8, (uint64_t)i, seed_i, buckets_S.data(), t_mask);
        }
        for (; j + 8 <= nn; j += 8) {
            round1_block_fallback(&base_ptr[j], 8, (uint64_t)i, seed_i, buckets_S.data(), t_mask);
        }
        if (j < nn) {
            round1_block_fallback(&base_ptr[j], nn - j, (uint64_t)i, seed_i, buckets_S.data(), t_mask);
        }
#endif
        if (all_filled_avx512(buckets_S.data(), t)) break;
    }

    if (!all_filled_avx512(buckets_S.data(), t)) {
        for (int i = t; i < 2 * t; ++i) {
            const int b = i - t;
            if (buckets_S[b] != INF_KEY()) continue;
            const uint64_t seed_i = seeds[i];
            uint64_t min_h = ~0ull;
            int j = 0;
#if defined(FASTHASH_SIMD_AVX512)
            for (; j + 8 <= nn; j += 8) {
                __m512i x = _mm512_loadu_si512((const void*)&base_ptr[j]);
                x = _mm512_xor_si512(x, _mm512_set1_epi64((long long)seed_i));
                __m512i h = splitmix64_vec(x);
                uint64_t block_min = hmin_epu64_8(h);
                if (block_min < min_h) min_h = block_min;
            }
#endif
            for (; j < nn; ++j) {
                uint64_t h = splitmix64(base_ptr[j] ^ seed_i);
                if (h < min_h) min_h = h;
            }
            const uint64_t key = pack_key((uint64_t)i, min_h);
            if (key < buckets_S[b]) {
                buckets_S[b] = key;
            }
        }
    }

    last_digest.assign(buckets_S.begin(), buckets_S.begin() + t);
    return last_digest;
}

// ===================== Batch Implementations =====================
vector<vector<uint64_t>> FastSimilaritySketch::sketch_batch(const vector<vector<uint32_t>>& batch,
                                                            int num_threads) {
    const size_t B = batch.size();
    vector<vector<uint64_t>> results(B);
    if (B == 0) return results;

#if defined(_OPENMP)
    const int threads = (num_threads > 0) ? num_threads : omp_get_max_threads();
    #pragma omp parallel num_threads(threads)
    {
        FastSimilaritySketch worker(this->t, this->seeds);
        #pragma omp for schedule(static)
        for (long i = 0; i < static_cast<long>(B); ++i) {
            results[i] = worker.sketch(batch[i]);
        }
    }
#else
    (void)num_threads;
    FastSimilaritySketch worker(this->t, this->seeds);
    for (size_t i = 0; i < B; ++i) {
        results[i] = worker.sketch(batch[i]);
    }
#endif
    return results;
}

vector<vector<uint64_t>> FastSimilaritySketch::sketch_batch(const vector<vector<int>>& batch,
                                                            int num_threads) {
    const size_t B = batch.size();
    vector<vector<uint64_t>> results(B);
    if (B == 0) return results;

#if defined(_OPENMP)
    const int threads = (num_threads > 0) ? num_threads : omp_get_max_threads();
    #pragma omp parallel num_threads(threads)
    {
        FastSimilaritySketch worker(this->t, this->seeds);
        #pragma omp for schedule(static)
        for (long i = 0; i < static_cast<long>(B); ++i) {
            results[i] = worker.sketch(batch[i]);
        }
    }
#else
    (void)num_threads;
    FastSimilaritySketch worker(this->t, this->seeds);
    for (size_t i = 0; i < B; ++i) {
        results[i] = worker.sketch(batch[i]);
    }
#endif
    return results;
}

vector<vector<uint64_t>> FastSimilaritySketch::sketch_batch(const vector<vector<string>>& batch,
                                                            int num_threads) {
    const size_t B = batch.size();
    vector<vector<uint64_t>> results(B);
    if (B == 0) return results;

#if defined(_OPENMP)
    const int threads = (num_threads > 0) ? num_threads : omp_get_max_threads();
    #pragma omp parallel num_threads(threads)
    {
        FastSimilaritySketch worker(this->t, this->seeds);
        #pragma omp for schedule(static)
        for (long i = 0; i < static_cast<long>(B); ++i) {
            results[i] = worker.sketch(batch[i]);
        }
    }
#else
    (void)num_threads;
    FastSimilaritySketch worker(this->t, this->seeds);
    for (size_t i = 0; i < B; ++i) {
        results[i] = worker.sketch(batch[i]);
    }
#endif
    return results;
}

void FastSimilaritySketch::sketch_batch_flat_csr(const uint32_t* data,
                                                 const uint64_t* indptr,
                                                 size_t B,
                                                 uint64_t* out_ptr,
                                                 int num_threads) {
#if defined(_OPENMP)
    const int threads = (num_threads > 0) ? num_threads : omp_get_max_threads();
    #pragma omp parallel num_threads(threads)
    {
        FastSimilaritySketch worker(this->t, this->seeds);
        #pragma omp for schedule(static)
        for (long i = 0; i < static_cast<long>(B); ++i) {
            const uint64_t start = indptr[i];
            const uint64_t end   = indptr[i+1];
            const size_t n = static_cast<size_t>(end - start);
            (void)worker.sketch(reinterpret_cast<const uint32_t*>(data + start), n);
            const vector<uint64_t>& S = worker.digest();
            std::memcpy(out_ptr + static_cast<size_t>(i) * static_cast<size_t>(t),
                        S.data(),
                        static_cast<size_t>(t) * sizeof(uint64_t));
        }
    }
#else
    FastSimilaritySketch worker(this->t, this->seeds);
    for (size_t i = 0; i < B; ++i) {
        const uint64_t start = indptr[i];
        const uint64_t end   = indptr[i+1];
        const size_t n = static_cast<size_t>(end - start);
        (void)worker.sketch(reinterpret_cast<const uint32_t*>(data + start), n);
        const vector<uint64_t>& S = worker.digest();
        std::memcpy(out_ptr + i * static_cast<size_t>(t), S.data(), static_cast<size_t>(t) * sizeof(uint64_t));
    }
#endif
}

void FastSimilaritySketch::sketch_batch_flat_ptrs(const uint32_t* const* data_ptrs,
                                                 const size_t* lengths,
                                                 size_t B,
                                                 uint64_t* out_ptr,
                                                 int num_threads) {
#if defined(_OPENMP)
    const int threads = (num_threads > 0) ? num_threads : omp_get_max_threads();
    #pragma omp parallel num_threads(threads)
    {
        FastSimilaritySketch worker(this->t, this->seeds);
        #pragma omp for schedule(static)
        for (long i = 0; i < static_cast<long>(B); ++i) {
            const uint32_t* ptr = data_ptrs[i];
            const size_t n = lengths[i];
            (void)worker.sketch(ptr, n);
            const vector<uint64_t>& S = worker.digest();
            std::memcpy(out_ptr + static_cast<size_t>(i) * static_cast<size_t>(t),
                        S.data(),
                        static_cast<size_t>(t) * sizeof(uint64_t));
        }
    }
#else
    FastSimilaritySketch worker(this->t, this->seeds);
    for (size_t i = 0; i < B; ++i) {
        const uint32_t* ptr = data_ptrs[i];
        const size_t n = lengths[i];
        (void)worker.sketch(ptr, n);
        const vector<uint64_t>& S = worker.digest();
        std::memcpy(out_ptr + i * static_cast<size_t>(t), S.data(), static_cast<size_t>(t) * sizeof(uint64_t));
    }
#endif
}

void FastSimilaritySketch::sketch_batch_flat_bytes(const uint8_t* const* ptrs,
                                                  const size_t* lengths,
                                                  const uint64_t* indptr,
                                                  size_t B,
                                                  uint64_t* out_ptr,
                                                  int num_threads) {
#if defined(_OPENMP)
    const int threads = (num_threads > 0) ? num_threads : omp_get_max_threads();
    #pragma omp parallel num_threads(threads)
    {
        FastSimilaritySketch worker(this->t, this->seeds);
        #pragma omp for schedule(static)
        for (long i = 0; i < static_cast<long>(B); ++i) {
            const uint64_t start = indptr[i];
            const uint64_t end   = indptr[i+1];
            const size_t n = static_cast<size_t>(end - start);
            // prehash bytes for this set into worker.base_buffer
            if (worker.base_buffer.size() < n) worker.base_buffer.resize(n);
            uint64_t* base_ptr = worker.base_buffer.data();
            size_t pos = 0;
            for (uint64_t j = start; j < end; ++j, ++pos) {
#if defined(FASTSKETCH_USE_FXHASH)
                base_ptr[pos] = fxhash64(ptrs[j], lengths[j]);
#else
                base_ptr[pos] = fnv1a64(ptrs[j], lengths[j]);
#endif
            }
            std::memset(worker.buckets_S.data(), 0xFF, static_cast<size_t>(t) * sizeof(uint64_t));
            for (int r = 0; r < t; ++r) {
                const uint64_t seed_i = seeds[r];
                int k = 0;
#if defined(FASTHASH_SIMD_AVX512)
                for (; k + 16 <= static_cast<int>(n); k += 16) {
                    round1_block_avx512_optimized(&base_ptr[k], 8, (uint64_t)r, seed_i, worker.buckets_S.data(), t_mask);
                    round1_block_avx512_optimized(&base_ptr[k+8], 8, (uint64_t)r, seed_i, worker.buckets_S.data(), t_mask);
                }
                for (; k + 8 <= static_cast<int>(n); k += 8) {
                    round1_block_avx512_optimized(&base_ptr[k], 8, (uint64_t)r, seed_i, worker.buckets_S.data(), t_mask);
                }
                if (k < static_cast<int>(n)) {
                    round1_block_avx512_optimized(&base_ptr[k], static_cast<int>(n) - k, (uint64_t)r, seed_i, worker.buckets_S.data(), t_mask);
                }
#else
                for (; k + 16 <= static_cast<int>(n); k += 16) {
                    round1_block_fallback(&base_ptr[k], 8, (uint64_t)r, seed_i, worker.buckets_S.data(), t_mask);
                    round1_block_fallback(&base_ptr[k+8], 8, (uint64_t)r, seed_i, worker.buckets_S.data(), t_mask);
                }
                for (; k + 8 <= static_cast<int>(n); k += 8) {
                    round1_block_fallback(&base_ptr[k], 8, (uint64_t)r, seed_i, worker.buckets_S.data(), t_mask);
                }
                if (k < static_cast<int>(n)) {
                    round1_block_fallback(&base_ptr[k], static_cast<int>(n) - k, (uint64_t)r, seed_i, worker.buckets_S.data(), t_mask);
                }
#endif
                if (all_filled_avx512(worker.buckets_S.data(), t)) break;
            }
            if (!all_filled_avx512(worker.buckets_S.data(), t)) {
                for (int r = t; r < 2 * t; ++r) {
                    const int b = r - t;
                    if (worker.buckets_S[b] != INF_KEY()) continue;
                    const uint64_t seed_i = seeds[r];
                    uint64_t min_h = ~0ull;
                    size_t j = 0;
#if defined(FASTHASH_SIMD_AVX512)
                    for (; j + 8 <= n; j += 8) {
                        __m512i x = _mm512_loadu_si512((const void*)&base_ptr[j]);
                        x = _mm512_xor_si512(x, _mm512_set1_epi64((long long)seed_i));
                        __m512i h = splitmix64_vec(x);
                        uint64_t block_min = hmin_epu64_8(h);
                        if (block_min < min_h) min_h = block_min;
                    }
#endif
                    for (; j < n; ++j) {
                        uint64_t h = splitmix64(base_ptr[j] ^ seed_i);
                        if (h < min_h) min_h = h;
                    }
                    const uint64_t key = pack_key((uint64_t)r, min_h);
                    if (key < worker.buckets_S[b]) worker.buckets_S[b] = key;
                }
            }
            std::memcpy(out_ptr + static_cast<size_t>(i) * static_cast<size_t>(t),
                        worker.buckets_S.data(),
                        static_cast<size_t>(t) * sizeof(uint64_t));
        }
    }
#else
    FastSimilaritySketch worker(this->t, this->seeds);
    for (size_t i = 0; i < B; ++i) {
        const uint64_t start = indptr[i];
        const uint64_t end   = indptr[i+1];
        const size_t n = static_cast<size_t>(end - start);
        if (worker.base_buffer.size() < n) worker.base_buffer.resize(n);
        uint64_t* base_ptr = worker.base_buffer.data();
        size_t pos = 0;
        for (uint64_t j = start; j < end; ++j, ++pos) {
#if defined(FASTSKETCH_USE_FXHASH)
            base_ptr[pos] = fxhash64(ptrs[j], lengths[j]);
#else
            base_ptr[pos] = fnv1a64(ptrs[j], lengths[j]);
#endif
        }
        std::memset(worker.buckets_S.data(), 0xFF, static_cast<size_t>(t) * sizeof(uint64_t));
        for (int r = 0; r < t; ++r) {
            const uint64_t seed_i = seeds[r];
            int k = 0;
            for (; k + 16 <= static_cast<int>(n); k += 16) {
                round1_block_fallback(&base_ptr[k], 8, (uint64_t)r, seed_i, worker.buckets_S.data(), t_mask);
                round1_block_fallback(&base_ptr[k+8], 8, (uint64_t)r, seed_i, worker.buckets_S.data(), t_mask);
            }
            for (; k + 8 <= static_cast<int>(n); k += 8) {
                round1_block_fallback(&base_ptr[k], 8, (uint64_t)r, seed_i, worker.buckets_S.data(), t_mask);
            }
            if (k < static_cast<int>(n)) {
                round1_block_fallback(&base_ptr[k], static_cast<int>(n) - k, (uint64_t)r, seed_i, worker.buckets_S.data(), t_mask);
            }
            if (all_filled_avx512(worker.buckets_S.data(), t)) break;
        }
        if (!all_filled_avx512(worker.buckets_S.data(), t)) {
            for (int r = t; r < 2 * t; ++r) {
                const int b = r - t;
                if (worker.buckets_S[b] != INF_KEY()) continue;
                const uint64_t seed_i = seeds[r];
                uint64_t min_h = ~0ull;
                for (size_t j = 0; j < n; ++j) {
                    uint64_t h = splitmix64(base_ptr[j] ^ seed_i);
                    if (h < min_h) min_h = h;
                }
                const uint64_t key = pack_key((uint64_t)r, min_h);
                if (key < worker.buckets_S[b]) worker.buckets_S[b] = key;
            }
        }
        std::memcpy(out_ptr + i * static_cast<size_t>(t), worker.buckets_S.data(), static_cast<size_t>(t) * sizeof(uint64_t));
    }
#endif
}


#ifdef DEMO_MAIN
int main(){
    vector<uint32_t> A; A.reserve(1001);
    for (uint32_t i = 0; i <= 1000u; ++i) A.push_back(i);
    int t = 128;
    FastSimilaritySketch sketcher(t, 42);
    auto S_A = sketcher.sketch(A);
    std::cout << "t=" << t << ", S[0]=" << S_A[0] << "\n";
    return 0;
}
#endif


