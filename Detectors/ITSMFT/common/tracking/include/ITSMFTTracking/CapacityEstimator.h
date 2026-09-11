// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
///
/// \file CapacityEstimator.h
/// \brief Cross-timeframe output-size prediction.
///

#ifndef ALICEO2_ITSMFT_TRACKING_CAPACITYESTIMATOR_H_
#define ALICEO2_ITSMFT_TRACKING_CAPACITYESTIMATOR_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>

namespace o2::itsmft::tracking
{

enum SlabSite : uint8_t {
  Tracklets = 0,
  Cells,
  Neighbours,
  RoadCandidates,
  Roads,
  TrackSeeds,
  TracksExtended,
  Tracks,
  NSlabSite,
};
constexpr const char* const SlabSiteNames[SlabSite::NSlabSite]{"Tracklets", "Cells", "Neighbours", "RoadCandidates", "Roads", "TrackSeeds", "TracksExtended", "Tracks"};

class CapacityEstimator
{
 public:
  struct Config {
    float alpha{0.2f};
    float marginInit{1.30f};
    float marginMin{1.10f};
    float marginMax{4.00f};
    float marginUp{1.50f};
    float marginOverflowSlack{1.05f};
    float marginDown{0.98f};
    float lowWatermark{0.60f};
    uint32_t decayAfter{2};
    size_t floorSlots{1024};
  };

  using KeyType = uint64_t;

  struct Decoded {
    SlabSite site;
    int iteration;
    int variant;
    int slot;
  };

  struct Statistics {
    size_t requested{0};
    size_t granted{0};
    size_t emitted{0};
    size_t spilled{0};
    size_t maxEmitted{0};
    uint32_t samples{0};
    uint32_t overflowEvents{0};
    uint32_t nLowStreak{0};
  };

  static constexpr KeyType makeKey(SlabSite site, int iteration, int variant, int slot) noexcept
  {
    return (static_cast<KeyType>(site) << 56) |
           (static_cast<KeyType>(iteration & 0xFF) << 48) |
           (static_cast<KeyType>(variant & 0xFFFF) << 32) |
           static_cast<KeyType>(static_cast<uint32_t>(slot));
  }

  static constexpr Decoded decodeKey(KeyType key) noexcept
  {
    return {
      .site = static_cast<SlabSite>((key >> 56) & 0xFF),
      .iteration = static_cast<int>((key >> 48) & 0xFF),
      .variant = static_cast<int>((key >> 32) & 0xFFFF),
      .slot = static_cast<int>(static_cast<uint32_t>(key & 0xFFFFFFFF))};
  }

  static constexpr int makeVariant(int high, int low) noexcept
  {
    return ((high & 0xFF) << 8) | (low & 0xFF);
  }

  static constexpr int getVariantHigh(int variant) noexcept
  {
    return (variant >> 8) & 0xFF;
  }

  static constexpr int getVariantLow(int variant) noexcept
  {
    return variant & 0xFF;
  }

  CapacityEstimator();
  explicit CapacityEstimator(Config cfg);
  ~CapacityEstimator();
  CapacityEstimator(const CapacityEstimator&) = delete;
  CapacityEstimator& operator=(const CapacityEstimator&) = delete;

  void reset();
  void beginTransaction();
  void commitTransaction() noexcept;
  void rollbackTransaction() noexcept;
  size_t capacity(uint64_t key, double scale) const;
  size_t peakCapacity(uint64_t key) const;
  double expected(uint64_t key, double scale) const;
  Statistics statistics(uint64_t key) const;
  void update(uint64_t key, double scale, size_t emitted, size_t capacityUsed, bool overflowed, bool memoryLimited);
  void update(uint64_t key, double scale, size_t requested, size_t granted, size_t emitted,
              size_t spilled, bool overflowed, bool memoryLimited);
  void print() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> mImpl;
};

template <typename Emit>
int runOnSlab(CapacityEstimator& estimator, const CapacityEstimator::KeyType key, const double scale, Emit&& emit, const size_t floorCapacity = 0)
{
  const auto toInt = [](const size_t v) { return static_cast<int>(std::min(v, static_cast<size_t>(std::numeric_limits<int>::max()))); };
  const int initialCapacity = toInt(estimator.capacity(key, scale));
  int capacity = std::max(initialCapacity, toInt(floorCapacity));
  int emitted = 0;
  bool overflowed = false;
  bool needsRetry = false;
  do {
    const int attemptCapacity = capacity;
    emitted = emit(attemptCapacity);
    needsRetry = emitted > attemptCapacity;
    overflowed |= needsRetry;
    capacity = emitted;
  } while (needsRetry);
  estimator.update(key, scale, emitted, initialCapacity, overflowed, false);
  return emitted;
}

} // namespace o2::itsmft::tracking

#endif /* ALICEO2_ITSMFT_TRACKING_CAPACITYESTIMATOR_H_ */
