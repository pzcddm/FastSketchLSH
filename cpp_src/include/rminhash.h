#pragma once
#include <vector>
#include <string>
#include <cstdint>
#include <limits>
#include <random>
#include <algorithm>
#include "murmurhash.h"

struct PermPair
{
  uint64_t a;
  uint64_t b;
};


class RMinSketch
{
private:
  size_t num_perm;
  uint32_t random_seed;
  std::vector<PermPair> perm_pairs;
  std::vector<uint32_t> hash_values;
  uint32_t permute_hash (uint64_t h, uint64_t a, uint64_t b) const;
public:
  RMinSketch(size_t num_perm = 128, uint32_t seed = 42);
  std::vector<uint32_t> sketch(const std::vector<int>& items);
};
