#include "../include/cminhash.h"

CMinHashSketch::CMinHashSketch (size_t num_perm, uint32_t seed) 
    : num_perm(num_perm), seed(seed), 
      hash_values(num_perm, std::numeric_limits<uint64_t>::max()){
  std::mt19937_64 rng (seed);
  sigma_a = (rng() | 1);  // Ensure it is an odd number
  sigma_b = rng();
  pi_c = (rng() | 1);
  pi_d = rng();

  pi_precomputed.resize(num_perm);
  for (size_t k = 0; k < num_perm; ++k) {
    pi_precomputed[k] = pi_c * k + pi_d;
  }
}

std::vector<uint32_t> CMinHashSketch::sketch(const std::vector<int>& items) {
    std::fill(hash_values.begin(), hash_values.end(), std::numeric_limits<uint64_t>::max());
    for (const auto item : items) {
        // use MurmurHash3_x64_128 to compute hash value
        uint64_t h[2];
        MurmurHash3_x64_128(&item, sizeof(int), seed, h);
        
        // compute σ(h) = a*h + b
        uint64_t sigma_h = sigma_a * h[0] + sigma_b;
        
        // Calculate all π values and update the minimum value
        for (size_t k = 0; k < num_perm; ++k) {
            uint64_t pi_val = pi_c * sigma_h + pi_precomputed[k];
            if (pi_val < hash_values[k]) {
                hash_values[k] = pi_val;
            }
        }
    }
    // Take the upper 32 bits as the signature
    std::vector<uint32_t> result;
    result.reserve(num_perm);
    for (auto val : hash_values) {
        result.push_back(static_cast<uint32_t>(val >> 32));
    }
    return result;
}