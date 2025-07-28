#pragma once
#include <vector>
#include <string>
#include <cstdint>

class CMinHashSketch
{
private:
  size_t num_perm;
  uint32_t seed;
  uint64_t sigma_a, sigma_b;
  uint64_t pi_c, pi_d;
  std::vector<uint64_t> pi_precomputed;
  std::vector<uint64_t> hash_values;
public:
  CMinHashSketch(size_t num_perm = 128, uint32_t seed = 42);
  std::vector<uint32_t> sketch(const std::vector<std::string>& items);
};
