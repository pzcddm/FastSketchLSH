#ifndef FAST_SKETCH_RENSA_LSH_H
#define FAST_SKETCH_RENSA_LSH_H

#include <vector>
#include <unordered_map>
#include "fasthash.h"

class FastSketchLSHRensa {
public:
  FastSketchLSHRensa(double threshold, std::size_t num_perm, std::size_t num_bands);

  void insert(std::size_t key, const FastSimilaritySketch& minhash);

  std::vector<std::size_t> query(const FastSimilaritySketch& minhash) const;

  std::size_t num_perm()  const noexcept { return num_perm_;  }
  std::size_t num_bands() const noexcept { return num_bands_; }
  std::size_t band_size() const noexcept { return band_size_; }
  double      threshold() const noexcept { return threshold_; }

private:
  double threshold_;
  std::size_t num_perm_;
  std::size_t num_bands_;
  std::size_t band_size_;
  // 每个 band 一张哈希表：band_hash -> [keys...]
  std::vector<std::unordered_map<std::uint64_t, std::vector<std::size_t>>> hash_tables_;

  static std::uint64_t calculate_band_hash(const std::vector<std::uint32_t>& band);

};


#endif  // FAST_SKETCH_RENSA_LSH_H
