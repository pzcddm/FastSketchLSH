#include "../include/rminhash.h"
#include <stdexcept>

uint32_t RMinSketch::permute_hash (uint64_t h, uint64_t a, uint64_t b) const {
    return static_cast<uint32_t>((a * h + b) >> 32);
}

RMinSketch::RMinSketch(size_t num_perm, uint32_t random_seed) {
    // 初始化随机数生成器
    std::mt19937_64 gen(random_seed);
    std::uniform_int_distribution<uint64_t> dist_a(1, UINT64_MAX);
    std::uniform_int_distribution<uint64_t> dist_b(0, UINT64_MAX);

    // 生成排列对 (确保a是奇数)
    perm_pairs.reserve(num_perm);
    for (uint32_t i = 0; i < num_perm; ++i) {
      perm_pairs.push_back({dist_a(gen) | 1ULL, dist_b(gen)});
    }
    // 初始化哈希值为最大值
    hash_values.resize(num_perm, std::numeric_limits<uint32_t>::max());
}

std::vector<uint32_t> RMinSketch::sketch(const std::vector<int>& items) {
    // 重置哈希值
    std::fill(hash_values.begin(), hash_values.end(), std::numeric_limits<uint32_t>::max());

    for (const auto& item : items) {
        uint64_t hash_val[2];
        MurmurHash3_x64_128(&item, sizeof(int), random_seed, hash_val);
        
        // 对每个排列计算最小哈希
        for (uint32_t j = 0; j < num_perm; ++j) {
            uint32_t ph = permute_hash(hash_val[0], perm_pairs[j].a, perm_pairs[j].b);
            hash_values[j] = std::min(hash_values[j], ph);
        }
    }
    return hash_values;
}
