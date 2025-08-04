#ifndef FASTHASH_SIMD_H
#define FASTHASH_SIMD_H

#include <vector>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <random>
#include <immintrin.h>  // For SIMD intrinsics
#include <thread>
#include <mutex>
#include <atomic>

class FastSimilaritySketchSIMD {
public:
    FastSimilaritySketchSIMD(size_t sketch_size, uint32_t random_seed);
    
    std::vector<uint64_t> sketch(const std::vector<int>& items);
    
private:
    size_t sketch_size;
    std::vector<uint64_t> hash_seeds;
    
    // SIMD-optimized hash computation
    void compute_hashes_simd(const int* data, size_t count, uint64_t seed, uint64_t* hash_results) const;
    
    // Thread-safe sketch update
    void update_sketch_threadsafe(std::vector<std::pair<size_t, uint64_t>>& S,
                                 std::vector<bool>& filled_bins,
                                 size_t& filled_count,
                                 size_t b,
                                 const std::pair<size_t, uint64_t>& v,
                                 std::mutex& mutex) const;
    
    // Parallel processing helper
    void process_hash_seed(size_t i, const std::vector<int>& items,
                          std::vector<std::pair<size_t, uint64_t>>& S,
                          std::vector<bool>& filled_bins,
                          size_t& filled_count,
                          std::atomic<bool>& early_exit) const;
};

#endif // FASTHASH_SIMD_H