// Deprecated: This file contains the legacy scalar FastSimilaritySketch implementation.
// It remains for archival/review purposes and internal C++ use, but is no longer
// exposed through the Python bindings.
//
// Note: The class is marked [[deprecated]] in the public header to surface
// compile-time warnings for new C++ uses.

#include "../include/fasthash.h"
#include "../include/murmurhash.h"
#include <stdexcept>
#ifdef DEMO_MAIN
#include <iostream>
#endif
using namespace std;


FastSimilaritySketchDeprecated::FastSimilaritySketchDeprecated(size_t sketch_size, uint32_t random_seed) {
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

    S.reserve(sketch_size);
}


std::vector<uint64_t> FastSimilaritySketchDeprecated::sketch(const std::vector<int>& items) {
    constexpr uint64_t SHIFT = 52;
    constexpr uint64_t MASK  = (uint64_t(1) << SHIFT) - 1;

    S.assign(sketch_size, std::numeric_limits<uint64_t>::max());
    std::vector<bool> filled_bins(sketch_size, false);
    size_t filled_cnt = 0;
    
    for (size_t i = 0; i < hash_seeds.size(); ++i) {
        const uint64_t seed = hash_seeds[i];
        for (const auto& item : items) {
            //Generate 128-bit hash value using mmh3
            uint64_t hash_val[2];
            MurmurHash3_x64_128(&item, sizeof(int), seed, hash_val);
            //First t hashes select buckets by modulo, last t are directly mapped into buckets
            size_t b = (i < sketch_size) ? (hash_val[0] % sketch_size) : (i - sketch_size);
            uint64_t mixed = ((hash_val[0] >> 12) ^ hash_val[0]) & MASK;
            uint64_t v = (static_cast<uint64_t>(i) << SHIFT) | mixed;
            //Each bucket only stores a 64-bit integer
            if (v < S[b]) {
                S[b] = v;
                if (!filled_bins[b]) {
                    filled_bins[b] = true;
                    filled_cnt++;
                }
            }
        }
        if (filled_cnt == sketch_size) break;
    }
    return S;
}

// Overload for byte-string inputs
std::vector<uint64_t> FastSimilaritySketchDeprecated::sketch(const std::vector<std::string>& items) {
    constexpr uint64_t SHIFT = 52;
    constexpr uint64_t MASK  = (uint64_t(1) << SHIFT) - 1;

    S.assign(sketch_size, std::numeric_limits<uint64_t>::max());
    std::vector<bool> filled_bins(sketch_size, false);
    size_t filled_cnt = 0;

    for (size_t i = 0; i < hash_seeds.size(); ++i) {
        const uint64_t seed = hash_seeds[i];
        for (const auto& bytes : items) {
            uint64_t hash_val[2];
            MurmurHash3_x64_128(bytes.data(), static_cast<int>(bytes.size()), static_cast<uint32_t>(seed), hash_val);
            size_t b = (i < sketch_size) ? (hash_val[0] % sketch_size) : (i - sketch_size);
            uint64_t mixed = ((hash_val[0] >> 12) ^ hash_val[0]) & MASK;
            uint64_t v = (static_cast<uint64_t>(i) << SHIFT) | mixed;
            if (v < S[b]) {
                S[b] = v;
                if (!filled_bins[b]) {
                    filled_bins[b] = true;
                    filled_cnt++;
                }
            }
        }
        //early-stop
        if (filled_cnt == sketch_size) break;
    }
    return S;
}

// ===================== Demo =====================
#ifdef DEMO_MAIN
int main(){
    vector<int> A;
    A.reserve(5000);
    for (int i=0;i<5000;i++){
        A.push_back(i);
    }

    int t = 128; // 2^k: 64/128/512 are OK
    FastSimilaritySketchDeprecated sk(t, 42);
    auto v = sk.sketch(A);

    std::cout << "sketch size = " << v.size() << "\nfirst 8 hash values:\n";
    for (int i=0;i<min(8,(int)v.size());++i) std::cout << v[i] << "\n";
    return 0;
}
#endif


