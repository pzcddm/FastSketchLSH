#include <vector>
#include <string>
#include <limits>
#include <random>
#include <cstdint>
#include <algorithm>
#include <functional>

class FastSimilaritySketch {
private:
    size_t sketch_size;  // Sketch size
    std::vector<uint64_t> hash_seeds;  // 2t hash seeds

public:
    FastSimilaritySketch(size_t sketch_size, uint32_t random_seed = 42);

    std::vector<uint64_t> sketch(const std::vector<int>& items);
};