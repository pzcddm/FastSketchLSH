//! Optimized R-MinHash implementation with SIMD acceleration
//! 
//! This implementation mimics the Rust RMinHash algorithm with the following optimizations:
//! - Batch processing of items for better cache utilization  
//! - Chunked permutation processing for SIMD vectorization
//! - AVX-512 acceleration where available
//! - Support for both uint32_t and string inputs
//! - Timing instrumentation for performance analysis
//!
//! The algorithm follows the design from the Rust implementation while leveraging
//! C++ SIMD intrinsics for maximum performance. We want to compare it with our fastsketch so that we can find the best implementation.
//! The efficiency of this implementation is not as good as rensa
//! TODO: Need to be investigated.
#include "../include/rminhash.h"
#include <stdexcept>
#include <chrono>
#include <algorithm>
#include <cstring>
#ifdef DEMO_MAIN
#include <iostream>
#include <cmath>
#include <limits>
#endif

using namespace std;

// ===================== Legacy Implementation (for compatibility) =====================

uint32_t RMinSketch::permute_hash(uint64_t h, uint64_t a, uint64_t b) const {
    return static_cast<uint32_t>((a * h + b) >> 32);
}

RMinSketch::RMinSketch(size_t num_perm, uint32_t random_seed) 
    : num_perm(num_perm), random_seed(random_seed) {
    // Initialize random number generator
    std::mt19937_64 gen(random_seed);
    std::uniform_int_distribution<uint64_t> dist_a(1, UINT64_MAX);
    std::uniform_int_distribution<uint64_t> dist_b(0, UINT64_MAX);

    // Generate permutation pairs (ensure 'a' is odd)
    perm_pairs.reserve(num_perm);
    for (size_t i = 0; i < num_perm; ++i) {
        perm_pairs.push_back({dist_a(gen) | 1ULL, dist_b(gen)});
    }
    // Initialize hash values to maximum
    hash_values.resize(num_perm, std::numeric_limits<uint32_t>::max());
}

std::vector<uint32_t> RMinSketch::sketch(const std::vector<int>& items) {
    // Reset hash values
    std::fill(hash_values.begin(), hash_values.end(), std::numeric_limits<uint32_t>::max());

    for (const auto& item : items) {
        uint64_t hash_val[2];
        MurmurHash3_x64_128(&item, sizeof(int), random_seed, hash_val);
        
        // Compute min-hash for each permutation
        for (size_t j = 0; j < num_perm; ++j) {
            uint32_t ph = permute_hash(hash_val[0], perm_pairs[j].a, perm_pairs[j].b);
            hash_values[j] = std::min(hash_values[j], ph);
        }
    }
    return hash_values;
}

// ===================== Optimized SIMD Implementation =====================

RMinHashSIMD::RMinHashSIMD(size_t num_perm, uint64_t random_seed) 
    : num_perm(num_perm), seed(random_seed) {
    
    // Initialize random number generator (same as Rust: Xoshiro256PlusPlus equivalent)
    std::mt19937_64 rng(random_seed);
    
    // Generate permutation pairs (ensure 'a' is odd for better distribution)
    permutations.reserve(num_perm);
    for (size_t i = 0; i < num_perm; ++i) {
        uint64_t a = rng() | 1ULL;  // Ensure odd multiplier
        uint64_t b = rng();
        permutations.emplace_back(a, b);
    }
    
    // Initialize hash values to maximum (like Rust: u32::MAX)
    hash_values.resize(num_perm, std::numeric_limits<uint32_t>::max());
    
    // Pre-allocate batch buffer
    hash_batch.reserve(BATCH_SIZE);
}

void RMinHashSIMD::reset() {
    std::fill(hash_values.begin(), hash_values.end(), std::numeric_limits<uint32_t>::max());
}

std::vector<uint32_t> RMinHashSIMD::digest() const {
    return hash_values;
}

// ===================== Main Interface Functions =====================

std::vector<uint32_t> RMinHashSIMD::sketch(const std::vector<uint32_t>& items) {
    reset();
    update_uint32(items);
    return digest();
}

std::vector<uint32_t> RMinHashSIMD::sketch(const std::vector<uint32_t>& items,
                                          double* prehash_ms,
                                          double* update_ms,
                                          double* total_ms) {
    auto start_time = std::chrono::high_resolution_clock::now();
    reset();
    update_uint32(items, prehash_ms, update_ms);
    auto end_time = std::chrono::high_resolution_clock::now();
    
    if (total_ms) {
        *total_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    }
    return digest();
}

std::vector<uint32_t> RMinHashSIMD::sketch(const std::vector<std::string>& items) {
    reset();
    update_strings(items);
    return digest();
}

std::vector<uint32_t> RMinHashSIMD::sketch(const std::vector<std::string>& items,
                                          double* prehash_ms,
                                          double* update_ms,
                                          double* total_ms) {
    auto start_time = std::chrono::high_resolution_clock::now();
    reset();
    update_strings(items, prehash_ms, update_ms);
    auto end_time = std::chrono::high_resolution_clock::now();
    
    if (total_ms) {
        *total_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    }
    return digest();
}

// ===================== Internal Update Functions =====================

void RMinHashSIMD::update_uint32(const std::vector<uint32_t>& items, 
                                 double* prehash_ms, 
                                 double* update_ms) {
    const size_t n = items.size();
    if (n == 0) return;
    
    auto prehash_start = std::chrono::high_resolution_clock::now();
    
    // Process items in batches for better cache utilization (mimic Rust BATCH_SIZE=32)
    for (size_t start = 0; start < n; start += BATCH_SIZE) {
        size_t end = std::min(start + BATCH_SIZE, n);
        size_t batch_size = end - start;
        
        // First pass: compute all hashes in batch
        hash_batch.clear();
        hash_batch.reserve(batch_size);
        
        size_t j = start;
        // SIMD hash computation for blocks of 8
#if defined(RMINHASH_SIMD_AVX512)
        alignas(64) uint64_t hash_block[8];
        for (; j + 8 <= end; j += 8) {
            hash_uint32x8_to_u64_avx512(&items[j], hash_block);
            for (int k = 0; k < 8; ++k) {
                hash_batch.push_back(hash_block[k]);
            }
        }
#endif
        // Process remaining items with scalar hash
        for (; j < end; ++j) {
            hash_batch.push_back(calculate_hash_fast(items[j]));
        }
        
        auto prehash_end = std::chrono::high_resolution_clock::now();
        auto update_start = prehash_end;
        
        // Second pass: update hash values using chunked processing
        process_hash_batch_chunked(hash_batch);
        
        auto update_end = std::chrono::high_resolution_clock::now();
        
        if (prehash_ms) {
            *prehash_ms += std::chrono::duration<double, std::milli>(prehash_end - prehash_start).count();
        }
        if (update_ms) {
            *update_ms += std::chrono::duration<double, std::milli>(update_end - update_start).count();
        }
        
        prehash_start = std::chrono::high_resolution_clock::now();
    }
}

void RMinHashSIMD::update_strings(const std::vector<std::string>& items,
                                 double* prehash_ms, 
                                 double* update_ms) {
    const size_t n = items.size();
    if (n == 0) return;
    
    auto prehash_start = std::chrono::high_resolution_clock::now();
    
    // Process items in batches for better cache utilization
    for (size_t start = 0; start < n; start += BATCH_SIZE) {
        size_t end = std::min(start + BATCH_SIZE, n);
        
        // First pass: compute all hashes in batch
        hash_batch.clear();
        for (size_t j = start; j < end; ++j) {
            hash_batch.push_back(calculate_hash_fast_bytes(items[j]));
        }
        
        auto prehash_end = std::chrono::high_resolution_clock::now();
        auto update_start = prehash_end;
        
        // Second pass: update hash values using chunked processing
        process_hash_batch_chunked(hash_batch);
        
        auto update_end = std::chrono::high_resolution_clock::now();
        
        if (prehash_ms) {
            *prehash_ms += std::chrono::duration<double, std::milli>(prehash_end - prehash_start).count();
        }
        if (update_ms) {
            *update_ms += std::chrono::duration<double, std::milli>(update_end - update_start).count();
        }
        
        prehash_start = std::chrono::high_resolution_clock::now();
    }
}

// ===================== Chunked Processing (mimic Rust algorithm) =====================

void RMinHashSIMD::process_hash_batch_chunked(const std::vector<uint64_t>& batch_hashes) {
    // Process permutations in chunks for better vectorization (mimic Rust PERM_CHUNK_SIZE=16)
    const size_t num_complete_chunks = num_perm / PERM_CHUNK_SIZE;
    
    // Process complete chunks
    for (size_t chunk = 0; chunk < num_complete_chunks; ++chunk) {
        size_t perm_start = chunk * PERM_CHUNK_SIZE;
        size_t perm_end = perm_start + PERM_CHUNK_SIZE;
        
#if defined(RMINHASH_SIMD_AVX512)
        process_perm_chunk_simd(batch_hashes, perm_start, perm_end);
#else
        process_perm_chunk_scalar(batch_hashes, perm_start, perm_end);
#endif
    }
    
    // Handle remainder
    size_t remainder_start = num_complete_chunks * PERM_CHUNK_SIZE;
    if (remainder_start < num_perm) {
        process_perm_chunk_scalar(batch_hashes, remainder_start, num_perm);
    }
}

#if defined(RMINHASH_SIMD_AVX512)
void RMinHashSIMD::process_perm_chunk_simd(const std::vector<uint64_t>& batch_hashes,
                                          size_t perm_start, size_t perm_end) {
    // Load current hash values for this chunk
    std::memcpy(current_chunk, &hash_values[perm_start], 
                (perm_end - perm_start) * sizeof(uint32_t));
    
    // Process each item hash against all permutations in this chunk
    for (uint64_t item_hash : batch_hashes) {
        // Process permutations in groups of 8 for SIMD (when chunk size allows)
        size_t perm_idx = perm_start;
        
        // SIMD processing for groups of 8 permutations
        for (; perm_idx + 8 <= perm_end; perm_idx += 8) {
            // Load 8 hash values from current chunk position
            size_t chunk_offset = perm_idx - perm_start;
            __m256i current_8 = _mm256_loadu_si256((const __m256i*)&current_chunk[chunk_offset]);
            __m512i current_64 = _mm512_cvtepu32_epi64(current_8);
            
            // Apply 8 different permutations to the same item_hash
            alignas(64) uint64_t perm_results[8];
            const __m512i item_vec = _mm512_set1_epi64((long long)item_hash);
            
            // Note: For true vectorization, we'd need to process 8 different (a,b) pairs
            // For now, we'll do scalar permutations but vectorized comparisons
            for (int i = 0; i < 8; ++i) {
                auto [a, b] = permutations[perm_idx + i];
                perm_results[i] = (uint64_t)permute_hash(item_hash, a, b);
            }
            
            __m512i new_hashes = _mm512_load_si512((const void*)perm_results);
            __m512i min_hashes = _mm512_min_epu64(current_64, new_hashes);
            
            // Convert back to 32-bit and store
            __m256i min_32 = _mm512_cvtepi64_epi32(min_hashes);
            _mm256_storeu_si256((__m256i*)&current_chunk[chunk_offset], min_32);
        }
        
        // Handle remainder with scalar operations
        for (; perm_idx < perm_end; ++perm_idx) {
            size_t chunk_offset = perm_idx - perm_start;
            auto [a, b] = permutations[perm_idx];
            uint32_t hash = permute_hash(item_hash, a, b);
            current_chunk[chunk_offset] = std::min(current_chunk[chunk_offset], hash);
        }
    }
    
    // Store results back to main hash_values array
    std::memcpy(&hash_values[perm_start], current_chunk,
                (perm_end - perm_start) * sizeof(uint32_t));
}
#endif

void RMinHashSIMD::process_perm_chunk_scalar(const std::vector<uint64_t>& batch_hashes,
                                            size_t perm_start, size_t perm_end) {
    // Load current hash values for this chunk
    size_t chunk_size = perm_end - perm_start;
    std::memcpy(current_chunk, &hash_values[perm_start], chunk_size * sizeof(uint32_t));
    
    // Process each item hash against all permutations in this chunk
    for (uint64_t item_hash : batch_hashes) {
        for (size_t perm_idx = perm_start; perm_idx < perm_end; ++perm_idx) {
            size_t chunk_offset = perm_idx - perm_start;
            auto [a, b] = permutations[perm_idx];
            uint32_t hash = permute_hash(item_hash, a, b);
            current_chunk[chunk_offset] = std::min(current_chunk[chunk_offset], hash);
        }
    }
    
    // Store results back to main hash_values array
    std::memcpy(&hash_values[perm_start], current_chunk, chunk_size * sizeof(uint32_t));
}

// ===================== Jaccard Similarity (mimic Rust implementation) =====================

double RMinHashSIMD::jaccard(const RMinHashSIMD& other) const {
    if (num_perm != other.num_perm) {
        throw std::invalid_argument("Cannot compare RMinHashSIMD with different num_perm");
    }
    
    size_t equal_count = 0;
    
    // Process in chunks of 8 for CPU-friendly operations (mimic Rust manual unrolling)
    const size_t num_complete_8_chunks = num_perm / 8;
    
    for (size_t chunk = 0; chunk < num_complete_8_chunks; ++chunk) {
        size_t base = chunk * 8;
        // Manual unrolling for better performance (mimic Rust code)
        equal_count += (hash_values[base + 0] == other.hash_values[base + 0]) ? 1 : 0;
        equal_count += (hash_values[base + 1] == other.hash_values[base + 1]) ? 1 : 0;
        equal_count += (hash_values[base + 2] == other.hash_values[base + 2]) ? 1 : 0;
        equal_count += (hash_values[base + 3] == other.hash_values[base + 3]) ? 1 : 0;
        equal_count += (hash_values[base + 4] == other.hash_values[base + 4]) ? 1 : 0;
        equal_count += (hash_values[base + 5] == other.hash_values[base + 5]) ? 1 : 0;
        equal_count += (hash_values[base + 6] == other.hash_values[base + 6]) ? 1 : 0;
        equal_count += (hash_values[base + 7] == other.hash_values[base + 7]) ? 1 : 0;
    }
    
    // Handle remainder
    size_t remainder_start = num_complete_8_chunks * 8;
    for (size_t i = remainder_start; i < num_perm; ++i) {
        if (hash_values[i] == other.hash_values[i]) {
            ++equal_count;
        }
    }
    
    return static_cast<double>(equal_count) / static_cast<double>(num_perm);
}

// ===================== Demo =====================
#ifdef DEMO_MAIN

// g++ -std=c++17 -O3 -Wall -Wextra -DRMINHASH_SIMD_AVX512 -march=skylake-avx512 -I "D:\Codes\InvestigateDocDuplicate\fastsketchlsh_ext\include" -c "D:\Codes\InvestigateDocDuplicate\fastsketchlsh_ext\cpp\rminhash.cpp" -o "D:\Codes\InvestigateDocDuplicate\fastsketchlsh_ext\build\rminhash_avx512.o"
// g++ -std=c++17 -O3 -Wall -Wextra -DRMINHASH_SIMD_AVX512 -DDEMO_MAIN -march=skylake-avx512 -I "D:\Codes\InvestigateDocDuplicate\fastsketchlsh_ext\include" "D:\Codes\InvestigateDocDuplicate\fastsketchlsh_ext\cpp\rminhash.cpp" "D:\Codes\InvestigateDocDuplicate\fastsketchlsh_ext\cpp\murmurhash3.cpp" -o "D:\Codes\InvestigateDocDuplicate\fastsketchlsh_ext\build\rminhash_demo_avx512.exe"
int main() {
    // Generate two integer sets for testing
    // A = {0, 1, ..., 1000}  
    // B = {500, 501, ..., 1500}
    vector<uint32_t> A; A.reserve(1001);
    for (uint32_t i = 0; i <= 1000u; ++i) A.push_back(i);
    vector<uint32_t> B; B.reserve(1001);
    for (uint32_t i = 500u; i <= 1500u; ++i) B.push_back(i);


    const double j_true = 0.3333333;

    const size_t num_perm = 128;
    const int trials = 100;

    std::random_device rd;
    std::mt19937_64 seed_rng(rd());

    double total_error = 0.0;
    double total_ms_A = 0.0;
    double total_ms_B = 0.0;
    double total_prehash_A = 0.0, total_update_A = 0.0;
    double total_prehash_B = 0.0, total_update_B = 0.0;

    for (int trial = 0; trial < trials; ++trial) {
        const uint64_t seed = seed_rng();
        
        // Create separate sketchers for A and B with same seed
        RMinHashSIMD sketcher_A(num_perm, seed);
        RMinHashSIMD sketcher_B(num_perm, seed);

        // Time sketch generation for A
        double prehashA = 0.0, updateA = 0.0, totalA = 0.0;
        auto S_A = sketcher_A.sketch(A, &prehashA, &updateA, &totalA);
        total_prehash_A += prehashA;
        total_update_A += updateA;
        total_ms_A += totalA;

        // Time sketch generation for B
        double prehashB = 0.0, updateB = 0.0, totalB = 0.0;
        auto S_B = sketcher_B.sketch(B, &prehashB, &updateB, &totalB);
        total_prehash_B += prehashB;
        total_update_B += updateB;
        total_ms_B += totalB;

        // Compute jaccard between sketches of A and B
        double j_est = sketcher_A.jaccard(sketcher_B);

        std::cout << "trial " << (trial + 1) << ": "
                  << "A[prehash=" << prehashA << " ms, update=" << updateA << " ms], "
                  << "B[prehash=" << prehashB << " ms, update=" << updateB << " ms], "
                  << "estJ=" << j_est << "\n";

        total_error += std::abs(j_est - j_true);
    }

    const double mean_error = total_error / static_cast<double>(trials);
    const double mean_ms_A = total_ms_A / static_cast<double>(trials);
    const double mean_ms_B = total_ms_B / static_cast<double>(trials);

    std::cout << "|A| = " << A.size() << ", |B| = " << B.size() << "\n";
    std::cout << "Sketch size = " << num_perm << ", trials = " << trials << "\n";
    std::cout << "Mean time: A = " << mean_ms_A << " ms, B = " << mean_ms_B << " ms\n";
    std::cout << "True Jaccard: " << j_true << "\n";
    std::cout << "Mean absolute error: " << mean_error << "\n";

    return 0;
}
#endif
