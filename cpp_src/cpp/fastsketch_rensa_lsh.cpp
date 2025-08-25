#include "../include/fastsketch_rensa_lsh.h"
#include <cstdint>
#include <stdexcept>
#include <algorithm>
#include <string>

FastSketchLSHRensa::FastSketchLSHRensa(double threshold, std::size_t num_perm, std::size_t num_bands)
: threshold_(threshold),
  num_perm_(num_perm),
  num_bands_(num_bands),
  band_size_(0),
  hash_tables_()
{
  if (num_perm_ == 0 || num_bands_ == 0) {
    throw std::invalid_argument("num_perm and num_bands must be > 0");
  }
  if (num_perm_ % num_bands_ != 0) {
    throw std::invalid_argument("num_perm must be divisible by num_bands");
  }
  band_size_ = num_perm_ / num_bands_;
  hash_tables_.resize(num_bands_);
}

void FastSketchLSHRensa::insert(std::size_t key, const FastSimilaritySketch& minhash) {
  const auto& d = minhash.digest();
  if (d.size() != num_perm_) {
    throw std::invalid_argument("MinHash permutation count mismatch: got " +
                                std::to_string(d.size()) + ", expect " +
                                std::to_string(num_perm_));
  }

  for (std::size_t i = 0; i < num_bands_; ++i) {
    const std::size_t start = i * band_size_;
    const std::size_t len   = band_size_;
    std::vector<uint32_t> slice(d.begin() + start, d.begin() + start + len);
    const std::uint64_t h = calculate_band_hash(slice);
    auto& bucket = hash_tables_[i][h];
    bucket.push_back(key);
  }
}

std::vector<std::size_t> FastSketchLSHRensa::query(const FastSimilaritySketch& minhash) const {
  const auto& d = minhash.digest();
  if (d.size() != num_perm_) {
    throw std::invalid_argument("MinHash permutation count mismatch: got " +
                                std::to_string(d.size()) + ", expect " +
                                std::to_string(num_perm_));
  }

  std::vector<std::size_t> candidates;
  for (std::size_t i = 0; i < num_bands_; ++i) {
    const std::size_t start = i * band_size_;
    const std::size_t len   = band_size_;
    std::vector<uint32_t> slice(d.begin() + start, d.begin() + start + len);
    const std::uint64_t h = calculate_band_hash(slice);
    const auto tbl_it = hash_tables_[i].find(h);
    if (tbl_it != hash_tables_[i].end()) {
      const auto& keys = tbl_it->second;
      candidates.insert(candidates.end(), keys.begin(), keys.end());
    }
  }
  std::sort(candidates.begin(), candidates.end());
  candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
  return candidates;
}

uint64_t FastSketchLSHRensa::calculate_band_hash(const std::vector<std::uint32_t>& band) {
    // 使用 FNV-1a 64-bit 哈希
    static constexpr std::uint64_t kOffset = 14695981039346656037ull;
    static constexpr std::uint64_t kPrime  = 1099511628211ull;

    std::uint64_t hash = kOffset;
    std::size_t chunks = band.size() / 4;
    std::size_t remainder = band.size() % 4;

    for (std::size_t i = 0; i < chunks; ++i) {
        std::uint64_t val1 = static_cast<std::uint64_t>(band[i * 4]) |
                            (static_cast<std::uint64_t>(band[i * 4 + 1]) << 32);
        std::uint64_t val2 = static_cast<std::uint64_t>(band[i * 4 + 2]) |
                            (static_cast<std::uint64_t>(band[i * 4 + 3]) << 32);
        const unsigned char* p1 = reinterpret_cast<const unsigned char*>(&val1);
        for (std::size_t j = 0; j < sizeof(val1); ++j) {
            hash ^= static_cast<std::uint64_t>(p1[j]);
            hash *= kPrime;
        }
        const unsigned char* p2 = reinterpret_cast<const unsigned char*>(&val2);
        for (std::size_t j = 0; j < sizeof(val2); ++j) {
            hash ^= static_cast<std::uint64_t>(p2[j]);
            hash *= kPrime;
        }
    }

    for (std::size_t i = band.size() - remainder; i < band.size(); ++i) {
        std::uint32_t val = band[i];
        const unsigned char* p = reinterpret_cast<const unsigned char*>(&val);
        for (std::size_t j = 0; j < sizeof(val); ++j) {
            hash ^= static_cast<std::uint64_t>(p[j]);
            hash *= kPrime;
        }
    }
    return hash;
}