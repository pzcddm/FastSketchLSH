#include "../include/fasthash.h"
#include "../include/murmurhash.h"

FastSimilaritySketch::FastSimilaritySketch(size_t sketch_size, uint32_t random_seed) {
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


std::vector<uint64_t> FastSimilaritySketch::sketch(const std::vector<int>& items) {
    using SketchPair = std::pair<size_t, uint64_t>;
    std::vector<SketchPair> S(sketch_size, {std::numeric_limits<size_t>::max(), 
                                  std::numeric_limits<uint64_t>::max()});
    size_t filled_count = 0;
    
    for (size_t i = 0; i < hash_seeds.size(); ++i) {
        uint64_t current_seed = hash_seeds[i];
        
        for (const auto& item : items) {
            uint64_t hash_val[2];
            MurmurHash3_x64_128(&item, sizeof(int), hash_seeds[i], hash_val);
            
            size_t b = (i < sketch_size) ? (hash_val[0] % sketch_size) : (i - sketch_size);
            auto v = std::make_pair(i, hash_val[0]);
            
            if (v < S[b]) {
                S[b] = v;
                if (filled_count < sketch_size && S[b].first != std::numeric_limits<size_t>::max()) {
                    filled_count++;
                }
            }
        }
        
        if (filled_count == sketch_size) break;
    }
    
    // Extract final sketch values
    std::vector<uint64_t> final_sketch;
    final_sketch.reserve(sketch_size);
    for (const std::pair<size_t, uint64_t>& pair_item : S) {
        // final_sketch.push_back(static_cast<uint32_t>(pair_item.second >> 32));
        final_sketch.push_back(pair_item.second);
    }
    
    return final_sketch;
}