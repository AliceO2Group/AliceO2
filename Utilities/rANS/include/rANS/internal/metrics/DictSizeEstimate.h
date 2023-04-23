// Copyright 2019-2023 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   DictSizeEstimate.h
/// @author Michael Lettrich
/// @brief  Low overhead dictionary size estimate that can be computed alogiside other metrics.

#ifndef RANS_INTERNAL_METRICS_DICTSIZEESTIMATE_H_
#define RANS_INTERNAL_METRICS_DICTSIZEESTIMATE_H_

#include <cstddef>
#include <cmath>

#include <fairlogger/Logger.h>

#include "rANS/internal/common/defaults.h"
#include "rANS/internal/common/utils.h"

namespace o2::rans::internal
{
class DictSizeEstimate
{
 public:
  DictSizeEstimate() = default;
  inline DictSizeEstimate(size_t numSamples)
  {
    if (numSamples > 0) {
      mScalingFactor = utils::pow2(DefaultScalingBits) / static_cast<double_t>(numSamples);
    }
  };

  [[nodiscard]] inline size_t getIndexSize() const noexcept { return mIndexSizeBits; };
  [[nodiscard]] inline size_t getFreqSize() const noexcept { return mFreqSizeBits; };
  [[nodiscard]] inline size_t getIndexSizeB() const noexcept { return utils::toBytes(mIndexSizeBits); };
  [[nodiscard]] inline size_t getFreqSizeB() const noexcept { return utils::toBytes(mFreqSizeBits); };

  [[nodiscard]] inline size_t getSizeB(size_t nNonzero, size_t renormingBits) const
  {
    using namespace utils;

    assert(isValidRenormingPrecision(renormingBits));
    const float_t rescalingFactor = static_cast<float_t>(pow2(renormingBits)) / pow2(DefaultScalingBits);
    const int64_t freqRescaled = getFreqSize() + static_cast<float_t>(nNonzero) * internal::fastlog2(rescalingFactor);
    return toBytes(getIndexSize() + std::max(static_cast<int64_t>(nNonzero), freqRescaled));
  };

  inline void updateIndexSize(uint32_t delta)
  {
    mIndexSizeBits += computeEliasDeltaLength(delta);
  };
  inline void updateFreqSize(uint32_t frequency)
  {
    assert(frequency > 0);
    const uint32_t scaledFrequency = std::max(1u, roundSymbolFrequency(frequency * mScalingFactor));
    mFreqSizeBits += std::max(1u, computeEliasDeltaLength(scaledFrequency));
  };

 private:
  [[nodiscard]] inline uint32_t computeEliasDeltaLength(uint32_t x) const noexcept
  {
    using namespace utils;
    assert(x > 0);
    return symbolLengthBits(x) + 2u * symbolLengthBits(symbolLengthBits(x) + 1u) + 1u;
  };

  static constexpr size_t DefaultScalingBits = defaults::MaxRenormPrecisionBits;

  double_t mScalingFactor{1.0};
  size_t mIndexSizeBits{};
  size_t mFreqSizeBits{};
};

class DictSizeEstimateCounter
{
 public:
  inline DictSizeEstimateCounter(DictSizeEstimate* estimate) : mEstimate{estimate} {};

  inline void update() noexcept { ++mDelta; };
  inline void update(uint32_t frequency)
  {
    assert(frequency > 0);
    assert(mDelta > 0);
    mEstimate->updateIndexSize(mDelta);
    mEstimate->updateFreqSize(frequency);
    mDelta = 0u;
  };

 private:
  uint32_t mDelta{};
  DictSizeEstimate* mEstimate{};
};
} // namespace o2::rans::internal

#endif /* RANS_INTERNAL_METRICS_DICTSIZEESTIMATE_H_ */
