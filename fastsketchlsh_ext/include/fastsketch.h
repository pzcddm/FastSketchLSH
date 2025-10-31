#ifndef FAST_SIMILARITY_SKETCH_H
#define FAST_SIMILARITY_SKETCH_H

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif
#include <cstdint>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstring>

// ===================== Public constants and utilities =====================
static constexpr int I_BITS = 12;                      // Number of bits used for round index i (supports up to 4095)
static constexpr int I_SHIFT = 64 - I_BITS;            // i placed at the high bits
static constexpr uint64_t H52_MASK = (1ull << 52) - 1; // Low 52 bits mask for hash value

static inline uint64_t INF_KEY() { return ~0ull; }     // Infinity marker for empty bucket

// Utility function declarations
uint64_t pack_key(uint64_t i, uint64_t h52);
uint64_t fnv1a64(const uint8_t* p, size_t n);
uint64_t fxhash64(const uint8_t* p, size_t n);
uint64_t hash_int32(uint32_t x);
uint64_t splitmix64(uint64_t x);
bool all_filled_avx512(const uint64_t* S, int t);
void warm_cache(uint64_t* S, int t);

#if defined(__AVX512F__)
__m512i pack_key_vec(uint64_t i, __m512i h52);
__m512i splitmix64_vec(__m512i x);
// Hash 8 uint32 values into 8 uint64 using AVX-512 (declared for testing)
void hash_int32x8_to_u64_avx512(const uint32_t* src, uint64_t* dst);
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
#endif

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

// FxHasher (rustc_hash) style fast 64-bit hasher for byte strings
// Processes 8 bytes per step; not cryptographic. Suitable for prehashing.
inline uint64_t rotl64(uint64_t x, unsigned r) {
    return (x << r) | (x >> (64u - r));
}

inline uint64_t fxhash64(const uint8_t* p, size_t n) {
    const uint64_t K = 0x517cc1b727220a95ull;
    uint64_t h = 0;
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        uint64_t w;
        std::memcpy(&w, p + i, 8);
        h = (rotl64(h, 5) ^ w) * K;
    }
    if (i < n) {
        uint64_t tail = 0;
        std::memcpy(&tail, p + i, n - i);
        h = (rotl64(h, 5) ^ tail) * K;
    }
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

#if defined(FASTHASH_SIMD_ENABLE_AVX512_INLINE) && defined(__AVX512F__)
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
class FastSimilaritySketch {
public:
    const int t;
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
    // Last computed digest for compatibility with LSH Rensa
    std::vector<uint64_t> last_digest;

    explicit FastSimilaritySketch(int sketch_size, uint64_t random_seed=42);
    // Construct with precomputed seeds (avoids RNG in constructor)
    FastSimilaritySketch(int sketch_size, const std::vector<uint64_t>& precomputed_seeds);
    // Input changed to vector<uint32_t> for SIMD-friendly zero-extension
    std::vector<uint64_t> sketch(const std::vector<uint32_t>& A);
    // Pointer-based overload to enable zero-copy batch processing
    std::vector<uint64_t> sketch(const uint32_t* data, size_t n);
    // Compatibility overload for legacy int vectors
    std::vector<uint64_t> sketch(const std::vector<int>& A);
    // Instrumented overload: returns per-phase timings (ms) if pointers are non-null
    std::vector<uint64_t> sketch(const std::vector<uint32_t>& A,
                                 double* prehash_ms,
                                 double* phase1_ms,
                                 double* phase2_ms);
    // Overload: support hashing arbitrary byte strings (e.g., Python bytes/utf-8 encoded str)
    std::vector<uint64_t> sketch(const std::vector<std::string>& bytes);
    // Instrumented overload for strings
    std::vector<uint64_t> sketch(const std::vector<std::string>& bytes,
                                 double* prehash_ms,
                                 double* phase1_ms,
                                 double* phase2_ms);
    // Zero-copy UTF-8 view sketcher (expects pointers that remain valid for the call duration)
    std::vector<uint64_t> sketch_utf8_views(const uint8_t* const* ptrs,
                                            const size_t* lengths,
                                            size_t n);

    // Batch APIs: compute sketches for a batch of inputs.
    // Each batch element corresponds to one set. num_threads=0 uses all available threads when OpenMP is enabled.
    std::vector<std::vector<uint64_t>> sketch_batch(const std::vector<std::vector<uint32_t>>& batch,
                                                    int num_threads = 0);
    std::vector<std::vector<uint64_t>> sketch_batch(const std::vector<std::vector<int>>& batch,
                                                    int num_threads = 0);
    std::vector<std::vector<uint64_t>> sketch_batch(const std::vector<std::vector<std::string>>& batch,
                                                    int num_threads = 0);

    // Batch flat-output APIs: removed list-based flat variants (use CSR version)

    // Zero-copy CSR-style batch: data is concatenated items, indptr has length B+1
    void sketch_batch_flat_csr(const uint32_t* data,
                               const uint64_t* indptr,
                               size_t B,
                               uint64_t* out_ptr,
                               int num_threads = 0);

    // Pointer-array batch: avoids concatenation copy by taking pointers to each set
    void sketch_batch_flat_ptrs(const uint32_t* const* data_ptrs,
                                const size_t* lengths,
                                size_t B,
                                uint64_t* out_ptr,
                                int num_threads = 0);

    // Batch over byte-strings using raw pointers and lengths (flattened with indptr)
    void sketch_batch_flat_bytes(const uint8_t* const* ptrs,
                                 const size_t* lengths,
                                 const uint64_t* indptr,
                                 size_t B,
                                 uint64_t* out_ptr,
                                 int num_threads = 0);

    // Access last computed digest (for LSH Rensa compatibility)
    const std::vector<uint64_t>& digest() const noexcept { return last_digest; }
};

#endif // FAST_SIMILARITY_SKETCH_H

