#include "../include/fastsketch_lsh.h"

FastSketchLSH::FastSketchLSH(float threshold, size_t sketch_size, size_t bands, uint32_t random_seed)
    : threshold(threshold), sketch_size(sketch_size), bands(bands),
      random_seed(random_seed), sketcher(sketch_size, random_seed) {
    if (threshold <= 0 || threshold >= 1) {
        throw std::invalid_argument("Threshold must be in (0, 1).");
    }
    if (sketch_size % bands != 0) {
        throw std::invalid_argument("sketch_size must be divisible by bands.");
    }
    rows_per_band = sketch_size / bands;
    buckets.resize(bands);
}


void FastSketchLSH::insert(const std::string& key, const std::vector<int>& set) {
    //split and insert sketches into buckets
    auto sketch = sketcher.sketch(set);
    for (size_t b = 0; b < bands; ++b) {
        size_t start = b * rows_per_band;
        size_t end = start + rows_per_band;
        std::vector<uint64_t> band(sketch.begin() + start, sketch.begin() + end);
        size_t h = band_hash(band);
        buckets[b][h].insert(key);
    }
    keys.insert(key);
}

void FastSketchLSH::insert(const std::string& key, const std::vector<std::string>& set) {
    auto sketch = sketcher.sketch(set);
    for (size_t b = 0; b < bands; ++b) {
        size_t start = b * rows_per_band;
        size_t end = start + rows_per_band;
        std::vector<uint64_t> band(sketch.begin() + start, sketch.begin() + end);
        size_t h = band_hash(band);
        buckets[b][h].insert(key);
    }
    keys.insert(key);
}

std::vector<std::string> FastSketchLSH::query(const std::vector<int>& set) {
    //query for similar set in buckets
    auto sketch = sketcher.sketch(set);
    std::unordered_set<std::string> candidates;
    for (size_t b = 0; b < bands; ++b) {
        size_t start = b * rows_per_band;
        size_t end = start + rows_per_band;
        std::vector<uint64_t> band(sketch.begin() + start, sketch.begin() + end);
        size_t h = band_hash(band);
        if (buckets[b].count(h)) {
            candidates.insert(buckets[b][h].begin(), buckets[b][h].end());
        }
    }
    return std::vector<std::string>(candidates.begin(), candidates.end());
}

std::vector<std::string> FastSketchLSH::query(const std::vector<std::string>& set) {
    auto sketch = sketcher.sketch(set);
    std::unordered_set<std::string> candidates;
    for (size_t b = 0; b < bands; ++b) {
        size_t start = b * rows_per_band;
        size_t end = start + rows_per_band;
        std::vector<uint64_t> band(sketch.begin() + start, sketch.begin() + end);
        size_t h = band_hash(band);
        if (buckets[b].count(h)) {
            candidates.insert(buckets[b][h].begin(), buckets[b][h].end());
        }
    }
    return std::vector<std::string>(candidates.begin(), candidates.end());
}

void FastSketchLSH::remove(const std::string& key) {
    if (keys.count(key)) {
        for (auto& band_map : buckets) {
            for (auto& bucket : band_map) {
                bucket.second.erase(key);
            }
        }
        keys.erase(key);
    }
}

void FastSketchLSH::clear() {
    for (auto& band_map : buckets) {
        band_map.clear();
    }
    keys.clear();
}

