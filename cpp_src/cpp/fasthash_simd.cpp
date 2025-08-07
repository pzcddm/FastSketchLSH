// fast_similarity_sketch_avx512_packed.cpp
#include "../include/fasthash_simd.h"
#include <immintrin.h>
#include <cstdint>
#include <vector>
#include <string>
#include <random>
using namespace std;




static inline uint64_t pack_key(uint64_t i, uint64_t h48) {
    return (i << I_SHIFT) | (h48 & H48_MASK);
}

// 一次性预哈希：FNV-1a(64)（长串只扫一次）
static inline uint64_t fnv1a64(const uint8_t* p, size_t n) {
    const uint64_t OFF = 1469598103934665603ull;
    const uint64_t PRM = 1099511628211ull;
    uint64_t h = OFF;
    for (size_t i = 0; i < n; ++i) { h ^= (uint64_t)p[i]; h *= PRM; }
    return h;
}

// splitmix64（scalar）
static inline uint64_t splitmix64(uint64_t x){
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

// splitmix64（AVX-512, 16-lane）
static inline __m512i splitmix64_vec(__m512i x){
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

// SIMD 统计 filled 数组之和（uint8_t ∈ {0,1}）
static inline int count_filled_simd(const uint8_t* filled, int t){
    int cnt = 0, i = 0;
    alignas(64) uint64_t tmp[8];
    const __m512i z = _mm512_setzero_si512();
    for (; i + 64 <= t; i += 64) {
        __m512i v = _mm512_loadu_si512((const void*)(filled + i)); // 64 字节
        __m512i sad = _mm512_sad_epu8(v, z); // 每 8B 求和 → 8 个 u64
        _mm512_store_si512((void*)tmp, sad);
        cnt += (int)(tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7]);
    }
    for (; i < t; ++i) cnt += filled[i];
    return cnt;
}

// 可选：每轮开始时“暖缓存”（把 S 的若干行预取到 L1）
static inline void warm_cache(uint64_t* S, int t){
#ifdef WARM_CACHE
    for (int i = 0; i < t; i += 8)
        _mm_prefetch((const char*)&S[i], _MM_HINT_T0);
#endif
}

// ===================== 第 1 轮：AVX-512 批算 + 逐 lane 更新 =====================
// 说明：不做批内 reduce-by-bucket；对每个 lane 顺序：读 S[b] → 比较 → 写 S[b]。
static inline void round1_block_avx512_no_reduce(
    const uint64_t* base_block, int nlanes,
    uint64_t round_i, uint64_t seed_i,
    uint64_t* S, uint8_t* filled,
    uint64_t t_mask)
{
    alignas(64) uint64_t h_lane[16], b_lane[16], key_lane[16];

    if (nlanes == 16) {
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
    for (int k=0;k<nlanes;k++) {
        key_lane[k] = pack_key(round_i, h_lane[k]);
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
            S[b]      = cand;
            filled[b] = 1; // 仅打标；计数在轮末 SIMD 汇总
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
    // 断言：i 至多到 2t-1（≤ 1023 when t≤512），安全落在 12 bit。
    if (2ULL * (uint64_t)t - 1 >= (1ull<<I_BITS))
        throw runtime_error("I_BITS too small for 2*t rounds.");
}

vector<uint64_t> FastSimilaritySketchAVX512Packed::sketch(const vector<string>& A) {
    const int n = (int)A.size();

    // 0) 预哈希（一次），避免 2t 次扫描长串
    vector<uint64_t> base(n);
    for (int j=0;j<n;j++)
        base[j] = fnv1a64((const uint8_t*)A[j].data(), A[j].size());

    // 1) 桶：仅 1 组（packed key），filled 标记
    vector<uint64_t> S(t, INF_KEY());
    vector<uint8_t>  filled(t, 0);

    // ==================== 第 1 轮：i=0..t-1 ====================
    for (int i=0; i<t; ++i) {
        const uint64_t seed_i = seeds[i];
        warm_cache(S.data(), t); // 可选

        int j = 0;
        for (; j+16<=n; j+=16) {
            round1_block_avx512_no_reduce(&base[j], 16, (uint64_t)i, seed_i,
                                          S.data(), filled.data(), t_mask);
        }
        if (j < n) {
            round1_block_avx512_no_reduce(&base[j], n-j, (uint64_t)i, seed_i,
                                          S.data(), filled.data(), t_mask);
        }

        // 轮末：一次性 SIMD 统计 filled_cnt（不在轮内维护）
        int filled_cnt = count_filled_simd(filled.data(), t);
        if (filled_cnt == t) break;
    }

    // ==================== 第 2 轮：i=t..2t-1，只补空桶 ====================
    // 因为 key 的高位是 i，i>=t 的 key 一定大于第 1 轮写入的 key，
    // 所以这里只会写原先空桶（S[b]==INF_KEY），不会覆盖已有桶。
    int filled_cnt = count_filled_simd(filled.data(), t);
    if (filled_cnt < t) {
        alignas(64) uint64_t h_lane[16];

        for (int i=t; i<2*t; ++i) {
            const int b = i - t;
            if (filled[b]) continue; // 已填的桶跳过
            const uint64_t seed_i = seeds[i];

            // 在所有元素上找 min_h（AVX-512 批处理 + 批内水平最小）
            uint64_t min_h = ~0ull;
            int j = 0;
            for (; j+16<=n; j+=16) {
                __m512i x = _mm512_loadu_si512((const void*)&base[j]);
                x = _mm512_xor_si512(x, _mm512_set1_epi64((long long)seed_i));
                __m512i h = splitmix64_vec(x);
                _mm512_store_si512((void*)h_lane, h);
                for (int k=0;k<16;k++) min_h = std::min(min_h, h_lane[k]);
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
                filled[b] = 1; // 只打标，计数在轮末/末尾做
            }
        }

        // 第二轮结束后再做一次 filled 计数（如需返回/断言）
        filled_cnt = count_filled_simd(filled.data(), t);
        (void)filled_cnt;
    }

    // 2) 输出：每桶取低 48 位哈希
    vector<uint64_t> out(t);
    for (int b=0;b<t;b++) {
        const uint64_t key = S[b];
        out[b] = (key == INF_KEY()) ? 0ull : (key & H48_MASK);
    }
    return out;
}

// ===================== Demo =====================
#ifdef DEMO_MAIN
int main(){
    vector<string> A;
    A.reserve(5000);
    for (int i=0;i<5000;i++){
        string s = "item_" + to_string(i);
        if (i%13==0) s += string(2000, 'x'); // 模拟较长字符串（实际可到 1e4~3e4）
        A.push_back(std::move(s));
    }

    int t = 128; // 2 的幂：64/128/512 都 OK
    FastSimilaritySketchAVX512Packed sk(t, 42);
    auto v = sk.sketch(A);

    cout << "sketch size = " << v.size() << "\nfirst 8 hash48 values:\n";
    for (int i=0;i<min(8,(int)v.size());++i) cout << v[i] << "\n";
    return 0;
}
#endif
