#include "../include/kminhash.h"
#include <algorithm>

KMinHashSketch::KMinHashSketch(size_t k, uint32_t random_seed) : k_(k) {
    if (k <= 0) {
        throw std::invalid_argument("Sketch size k must be positive");
    }
    init_hash_seeds(random_seed);
    min_hashes_.resize(k_, std::numeric_limits<uint64_t>::max());
}

std::vector<uint64_t> KMinHashSketch::sketch(const std::vector<int>& items) {
    std::fill(min_hashes_.begin(), min_hashes_.end(), std::numeric_limits<uint64_t>::max()); 
    for (size_t i = 0; i < k_; ++i) {
        for (const auto& item : items) {
            uint64_t h[2];
            MurmurHash3_x64_128(&item, sizeof(int), hash_seeds_[i], h);
            if (h[0] < min_hashes_[i]) {
                min_hashes_[i] = h[0];
            }
        }
    }
    
    return min_hashes_;
}