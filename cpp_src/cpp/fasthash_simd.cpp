// fast_similarity_sketch_avx512_packed.cpp
#include "../include/fasthash_simd.h"
#include <immintrin.h>
#include <cstdint>
#include <vector>
#include <string>
#include <random>
#ifdef DEMO_MAIN
#include <iostream>
#endif
using namespace std;


inline uint64_t pack_key(uint64_t i, uint64_t h52) {
    return (i << I_SHIFT) | (h52 & H52_MASK);
}

inline __m512i pack_key_vec(uint64_t i, __m512i h52) {
    // Broadcast i into a vector of 64-bit lanes and shift into top bits
    const __m512i vi = _mm512_set1_epi64((long long)i);
    const __m512i hi = _mm512_slli_epi64(vi, I_SHIFT);
    const __m512i mask = _mm512_set1_epi64((long long)H52_MASK);
    const __m512i lo = _mm512_and_si512(h52, mask);
    return _mm512_or_si512(hi, lo);
}

// Note: scalar hashing utilities are defined inline in the public header
// (fnv1a64, hash_int32, splitmix64). We only keep the vectorized helpers here.

// splitmix64（AVX-512, 8-lane for uint64）
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

// Hash 8 int32 values into 8 uint64 using SplitMix64-style mixing
inline void hash_int32x8_to_u64_avx512(const int* src, uint64_t* dst) {
    __m256i v32 = _mm256_loadu_si256((const __m256i*)src);
    __m512i x64 = _mm512_cvtepi32_epi64(v32);                 // sign-extends
    const __m512i mask32 = _mm512_set1_epi64(0xFFFFFFFFull);  // clear sign-extended high bits
    x64 = _mm512_and_si512(x64, mask32);
    __m512i h = splitmix64_vec(x64);
    _mm512_storeu_si512((void*)dst, h);
}

// Horizontal min across 8 lanes of unsigned 64-bit using AVX-512 + VL
// TODO: I am not sure if this is fast, maybe it is even slower than the scalar version. Please check and do some experiments.
inline uint64_t horizontal_min_epu64(__m512i v) {
    __m256i m256  = _mm256_min_epu64(_mm512_castsi512_si256(v),
                                     _mm512_extracti64x4_epi64(v, 1));
    __m128i m128  = _mm_min_epu64(_mm256_castsi256_si128(m256),
                                  _mm256_extracti128_si256(m256, 1));
    uint64_t a = (uint64_t)_mm_cvtsi128_si64(m128);
    uint64_t b = (uint64_t)_mm_extract_epi64(m128, 1);
    return a < b ? a : b;
}

// SIMD check: whether all buckets in S are filled (i.e., not INF_KEY)
inline bool all_filled_avx512(const uint64_t* S, int t) {
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
}

// ===================== 第 1 轮：AVX-512 批算 + 逐 lane 更新 =====================
// 说明：不做批内 reduce-by-bucket；对每个 lane 顺序：读 S[b] → 比较 → 写 S[b]。
inline void round1_block_avx512_no_reduce(
    const uint64_t* base_block, int nlanes,
    uint64_t round_i, uint64_t seed_i,
    uint64_t* S,
    uint64_t t_mask)
{
    alignas(64) uint64_t h_lane[8], b_lane[8], key_lane[8];

    if (nlanes == 8) {
        __m512i x = _mm512_loadu_si512((const void*)base_block);
        x = _mm512_xor_si512(x, _mm512_set1_epi64((long long)seed_i));
        __m512i h = splitmix64_vec(x);
        __m512i b = _mm512_and_si512(h, _mm512_set1_epi64((long long)t_mask));
        _mm512_store_si512((void*)h_lane, h);
        _mm512_store_si512((void*)b_lane, b);
    } else {
        for (int k=0;k<nlanes;k++){
            uint64_t h = splitmix64(base_block[k] ^ seed_i);
            h_lane[k] = h;
            b_lane[k] = h & t_mask;
        }
    }
    if (nlanes == 8) {
        __m512i hv = _mm512_loadu_si512((const void*)h_lane);
        __m512i kv = pack_key_vec(round_i, hv);
        _mm512_store_si512((void*)key_lane, kv);
    } else {
        for (int k=0;k<nlanes;k++) {
            key_lane[k] = pack_key(round_i, h_lane[k]);
        }
    }

    // 逐 lane 更新（可选预取：让 S[b] 尽量命中 L1）
    for (int k=0;k<nlanes;k++){
        const uint64_t b = b_lane[k];
#ifdef PREFETCH_BUCKET
        _mm_prefetch((const char*)&S[b], _MM_HINT_T0);
#endif
        const uint64_t cand = key_lane[k];
        const uint64_t old  = S[b];
        if (cand < old) {
            S[b] = cand;
        }
    }
}

// ===================== 主类：2t 轮（packed key 版本） =====================

FastSimilaritySketchAVX512Packed::FastSimilaritySketchAVX512Packed(int sketch_size, uint64_t random_seed)
    : t(sketch_size), t_mask(sketch_size-1), seeds(2*sketch_size)
{
    if (t<=0 || (t & (t-1))!=0) throw runtime_error("t must be a power of two.");
    std::mt19937_64 rng(random_seed);
    for (int i=0;i<2*t;i++) seeds[i] = rng();
    if ((uint64_t)t > (1ull<<I_BITS))
        throw runtime_error("t can not be larger than 4096.");
}

vector<uint64_t> FastSimilaritySketchAVX512Packed::sketch(const vector<int>& A) {
    const int n = (int)A.size();

    // 0) 预哈希（一次），避免 2t 次扫描长串
    vector<uint64_t> base(n);
    int j0 = 0;
    for (; j0 + 8 <= n; j0 += 8) {
        hash_int32x8_to_u64_avx512(&A[j0], &base[j0]);
    }
    for (; j0 < n; ++j0) {
        base[j0] = hash_int32(static_cast<uint32_t>(A[j0]));
    }

    // 1) Buckets: one group (packed key)
    vector<uint64_t> S(t, INF_KEY());
    // ==================== 第 0 ~ t -1 轮：i=0..t-1 ====================
    for (int i=0; i<t; ++i) {
        const uint64_t seed_i = seeds[i];

        int j = 0;
        // AVX-512 8 lanes for uint64
        for (; j+8<=n; j+=8) {
            round1_block_avx512_no_reduce(&base[j], 8, (uint64_t)i, seed_i,
                                          S.data(), t_mask);
        }
        if (j < n) {
            round1_block_avx512_no_reduce(&base[j], n-j, (uint64_t)i, seed_i,
                                          S.data(), t_mask);
        }

        // End of round: check whether all buckets are filled
        if (all_filled_avx512(S.data(), t)) break;
    }

    // ==================== 第 t ~ 2t -1 轮：i=t..2t-1，只补空桶 ====================
    // 因为 key 的高位是 i，i>=t 的 key 一定大于第 1 轮写入的 key，
    // 所以这里只会写原先空桶（S[b]==INF_KEY），不会覆盖已有桶。
    if (!all_filled_avx512(S.data(), t)) {
        alignas(64) uint64_t h_lane[8];

        for (int i=t; i<2*t; ++i) {
            const int b = i - t;
            if (S[b] != INF_KEY()) continue; // Already filled bucket
            const uint64_t seed_i = seeds[i];

            // 在所有元素上找 min_h（AVX-512 批处理 + 批内水平最小）
            uint64_t min_h = ~0ull;
            int j = 0;
            for (; j+8<=n; j+=8) {
                __m512i x = _mm512_loadu_si512((const void*)&base[j]);
                x = _mm512_xor_si512(x, _mm512_set1_epi64((long long)seed_i));
                __m512i h = splitmix64_vec(x);
                uint64_t block_min = horizontal_min_epu64(h);
                if (block_min < min_h) min_h = block_min;
            }
            for (; j<n; ++j) {
                uint64_t h = splitmix64(base[j] ^ seed_i);
                if (h < min_h) min_h = h;
            }

#ifdef PREFETCH_BUCKET
            _mm_prefetch((const char*)&S[b], _MM_HINT_T0);
#endif
            const uint64_t key = pack_key((uint64_t)i, min_h);
            if (key < S[b]) {
                S[b] = key;
            }
        }

        // All filled check (optional)
        (void)all_filled_avx512;
    }
    return S;
}

// ===================== Demo =====================
#ifdef DEMO_MAIN
int main(){
    vector<int> A;
    A.reserve(5000);
    for (int i=0;i<5000;i++){
        A.push_back(i);
    }

    int t = 128; // 2 的幂：64/128/512 都 OK
    FastSimilaritySketchAVX512Packed sk(t, 42);
    auto v = sk.sketch(A);

    std::cout << "sketch size = " << v.size() << "\nfirst 8 hash values:\n";
    for (int i=0;i<min(8,(int)v.size());++i) std::cout << v[i] << "\n";
    return 0;
}
#endif
