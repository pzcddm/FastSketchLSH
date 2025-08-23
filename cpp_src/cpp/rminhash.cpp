#include "../include/rminhash.h"
#include <stdexcept>

uint32_t RMinSketch::permute_hash (uint64_t h, uint64_t a, uint64_t b) const {
    return static_cast<uint32_t>((a * h + b) >> 32);
}

RMinSketch::RMinSketch(size_t num_perm, uint32_t random_seed) {
    // Initialize random number generator
    std::mt19937_64 gen(random_seed);
    std::uniform_int_distribution<uint64_t> dist_a(1, UINT64_MAX);
    std::uniform_int_distribution<uint64_t> dist_b(0, UINT64_MAX);

    // Generate permutation pairs (ensure 'a' is odd)
    perm_pairs.reserve(num_perm);
    for (uint32_t i = 0; i < num_perm; ++i) {
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
        for (uint32_t j = 0; j < num_perm; ++j) {
            uint32_t ph = permute_hash(hash_val[0], perm_pairs[j].a, perm_pairs[j].b);
            hash_values[j] = std::min(hash_values[j], ph);
        }
    }
    return hash_values;
}
