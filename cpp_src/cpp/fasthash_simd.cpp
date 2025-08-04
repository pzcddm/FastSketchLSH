#include "../include/fasthash_simd.h"
#include "../include/murmurhash.h"
#include <algorithm>
#include <memory>
#include <immintrin.h>
#include <thread>
#include <vector>

FastSimilaritySketchSIMD::FastSimilaritySketchSIMD(size_t sketch_size, uint32_t random_seed) {
    if (sketch_size == 0) {
        throw std::invalid_argument("Sketch size (t) must be positive");
    }
    this->sketch_size = sketch_size;
    
    // Initialize random generator with fixed seed
    std::mt19937_64 gen(random_seed);
    std::uniform_int_distribution<uint64_t> dist(0, UINT64_MAX);
    
    // Generate 2t hash seeds
    hash_seeds.resize(2 * sketch_size);
    for (auto& seed : hash_seeds) {
        seed = dist(gen);
    }
}

void FastSimilaritySketchSIMD::compute_hashes_simd(const int* data, size_t count, uint64_t seed, uint64_t* hash_results) const {
    // Process 4 items at a time using AVX2 (256-bit registers)
    constexpr size_t simd_width = 4;
    size_t i = 0;
    
    // Convert seed to __m256i for SIMD operations
    __m256i seed_vec = _mm256_set1_epi64x(seed);
    
    for (; i + simd_width <= count; i += simd_width) {
        // Load 4 integers
        __m256i items = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i));
        
        // Compute hashes for each item (this is a placeholder - actual MurmurHash3 SIMD implementation would go here)
        // In a real implementation, we would have a SIMD-optimized version of MurmurHash3
        for (size_t j = 0; j < simd_width; j++) {
            MurmurHash3_x64_128(data + i + j, sizeof(int), seed, hash_results + 2*(i + j));
        }
    }
    
    // Process remaining items
    for (; i < count; i++) {
        MurmurHash3_x64_128(data + i, sizeof(int), seed, hash_results + 2*i);
    }
}

void FastSimilaritySketchSIMD::update_sketch_threadsafe(std::vector<std::pair<size_t, uint64_t>>& S,
                                                  std::vector<bool>& filled_bins,
                                                  size_t& filled_count,
                                                  size_t b,
                                                  const std::pair<size_t, uint64_t>& v,
                                                  std::mutex& mutex) const {
    std::lock_guard<std::mutex> lock(mutex);
    if (v < S[b]) {
        S[b] = v;
        if (!filled_bins[b]) {
            filled_bins[b] = true;
            filled_count++;
        }
    }
}

void FastSimilaritySketchSIMD::process_hash_seed(size_t i, const std::vector<int>& items,
                                           std::vector<std::pair<size_t, uint64_t>>& S,
                                           std::vector<bool>& filled_bins,
                                           size_t& filled_count,
                                           std::atomic<bool>& early_exit) const {
    if (early_exit.load()) return;
    
    uint64_t current_seed = hash_seeds[i];
    std::vector<uint64_t> hash_results(items.size() * 2); // Each hash produces 2 uint64_t
    
    // Use SIMD to compute hashes in bulk
    compute_hashes_simd(items.data(), items.size(), current_seed, hash_results.data());
    
    std::mutex mutex;
    
    for (size_t item_idx = 0; item_idx < items.size(); item_idx++) {
        if (early_exit.load()) break;
        
        uint64_t* hash_val = hash_results.data() + 2 * item_idx;
        size_t b = (i < sketch_size) ? (hash_val[0] % sketch_size) : (i - sketch_size);
        auto v = std::make_pair(i, hash_val[0]);
        
        update_sketch_threadsafe(S, filled_bins, filled_count, b, v, mutex);
    }
    
    if (filled_count >= sketch_size) {
        early_exit.store(true);
    }
}

std::vector<uint64_t> FastSimilaritySketchSIMD::sketch(const std::vector<int>& items) {
    using SketchPair = std::pair<size_t, uint64_t>;
    std::vector<SketchPair> S(sketch_size, {std::numeric_limits<size_t>::max(), 
                                  std::numeric_limits<uint64_t>::max()});
    std::vector<bool> filled_bins(sketch_size, false);
    size_t filled_count = 0;
    std::atomic<bool> early_exit(false);
    
    // Determine number of threads to use
    const size_t num_threads = std::min(hash_seeds.size(), static_cast<size_t>(std::thread::hardware_concurrency()));
    
    // Process hash seeds in parallel
    std::vector<std::thread> threads;
    for (size_t i = 0; i < hash_seeds.size(); i += num_threads) {
        // Launch a batch of threads
        for (size_t t = 0; t < num_threads && (i + t) < hash_seeds.size(); t++) {
            threads.emplace_back([&, idx = i + t] {
                process_hash_seed(idx, items, S, filled_bins, filled_count, early_exit);
            });
        }
        
        // Wait for this batch to complete
        for (auto& thread : threads) {
            thread.join();
        }
        threads.clear();
        
        if (early_exit.load()) break;
    }
    
    // Extract final sketch values
    std::vector<uint64_t> final_sketch;
    final_sketch.reserve(sketch_size);
    for (const auto& pair_item : S) {
        final_sketch.push_back(pair_item.second);
    }
    
    return final_sketch;
}