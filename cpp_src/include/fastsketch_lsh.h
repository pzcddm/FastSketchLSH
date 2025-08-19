#ifndef FAST_SKETCH_LSH_H
#define FAST_SKETCH_LSH_H

#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <cstdint>
#include <algorithm>
#include <stdexcept>
#include "fasthash.h"

class FastSketchLSH {
private:
    float threshold;
    size_t sketch_size;
    size_t bands;
    size_t rows_per_band;
    uint32_t random_seed;
    FastSimilaritySketch sketcher;

    // Buckets: vector of unordered_maps, one per band
    std::vector<std::unordered_map<size_t, std::unordered_set<std::string>>> buckets;

    // Store keys for removal
    std::unordered_set<std::string> keys;

    // Hash function for a band
    size_t band_hash(const std::vector<uint64_t>& band) {
        size_t seed = 0;
        for (auto val : band) {
            seed ^= std::hash<uint64_t>{}(val) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }

public:
    FastSketchLSH(float threshold, size_t sketch_size, size_t bands, uint32_t random_seed = 42);

    void insert(const std::string& key, const std::vector<int>& set);
    // Overload for string items
    void insert(const std::string& key, const std::vector<std::string>& set);

    std::vector<std::string> query(const std::vector<int>& set);
    // Overload for string items
    std::vector<std::string> query(const std::vector<std::string>& set);

    void remove(const std::string& key);

    void clear();
};

#endif // FAST_SKETCH_LSH_H