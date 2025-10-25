#ifndef FAST_SIMILARITY_SKETCH_DEPRECATED_H
#define FAST_SIMILARITY_SKETCH_DEPRECATED_H

#include <vector>
#include <string>
#include <limits>
#include <random>
#include <cstdint>
#include <algorithm>
#include <functional>

class [[deprecated("Scalar implementation is deprecated; use FastSimilaritySketch from fastsketch.h instead")]] FastSimilaritySketchDeprecated {
private:
    size_t sketch_size;  // Sketch size
    std::vector<uint64_t> hash_seeds;  // 2t hash seeds
    std::vector<uint64_t> S;

public:
    FastSimilaritySketchDeprecated(size_t sketch_size, uint32_t random_seed = 42);

    std::vector<uint64_t> sketch(const std::vector<int>& items);

    // New overload: support hashing arbitrary byte strings (e.g., Python bytes/utf-8 encoded str)
    std::vector<uint64_t> sketch(const std::vector<std::string>& items);

    const std::vector<uint64_t>& digest() const noexcept {return S;}
};

#endif // FAST_SIMILARITY_SKETCH_DEPRECATED_H