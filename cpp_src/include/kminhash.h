#pragma once
#include <vector>
#include <cstdint>
#include <random>
#include <limits>
#include <string>
#include "murmurhash.h"
class KMinHashSketch {
private:
    size_t k_;
    std::vector<uint32_t> hash_seeds_;
    std::vector<uint64_t> min_hashes_;

    void init_hash_seeds(uint32_t random_seed) {
        std::mt19937 rng(random_seed);
        std::uniform_int_distribution<uint32_t> dist(0, 0x7FFFFFFF);
        hash_seeds_.reserve(k_);
        for (size_t i = 0; i < k_; ++i) {
            hash_seeds_.push_back(dist(rng));
        }
    }

public:
    KMinHashSketch(size_t k, uint32_t random_seed = 42);
    std::vector<uint64_t> sketch(const std::vector<int>& items);
};