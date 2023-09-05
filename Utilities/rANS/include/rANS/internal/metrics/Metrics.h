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

/// @file   Metrics.h
/// @author Michael Lettrich
/// @brief  Computes and provides essential metrics on the dataset used for parameter and size estimates by other algorithms.

#ifndef RANS_INTERNAL_METRICS_METRICS_H_
#define RANS_INTERNAL_METRICS_METRICS_H_

#include <cstdint>

#include <fairlogger/Logger.h>
#include <gsl/span>

#include "rANS/internal/common/utils.h"
#include "rANS/internal/containers/DenseHistogram.h"
#include "rANS/internal/containers/AdaptiveHistogram.h"
#include "rANS/internal/metrics/properties.h"
#include "rANS/internal/metrics/utils.h"
#include "rANS/internal/metrics/SizeEstimate.h"
#include "rANS/internal/transform/algorithm.h"

namespace o2::rans
{

template <typename source_T>
class Metrics
{
  inline static constexpr float_t defaultCutoffPrecision = 0.999;

 public:
  using source_type = source_T;

  Metrics() = default;
  inline Metrics(const DenseHistogram<source_type>& histogram, float_t cutoffPrecision = defaultCutoffPrecision) { init(histogram, {}, {}, cutoffPrecision); };
  inline Metrics(const AdaptiveHistogram<source_type>& histogram, float_t cutoffPrecision = defaultCutoffPrecision) { init(histogram, {}, {}, cutoffPrecision); };
  inline Metrics(const SparseHistogram<source_type>& histogram, float_t cutoffPrecision = defaultCutoffPrecision) { init(histogram, {}, {}, cutoffPrecision); };

  inline Metrics(const DenseHistogram<source_type>& histogram, source_type min, source_type max, float_t cutoffPrecision = defaultCutoffPrecision) { init(histogram, {min}, {max}, cutoffPrecision); };
  inline Metrics(const AdaptiveHistogram<source_type>& histogram, source_type min, source_type max, float_t cutoffPrecision = defaultCutoffPrecision) { init(histogram, {min}, {max}, cutoffPrecision); };
  inline Metrics(const SparseHistogram<source_type>& histogram, source_type min, source_type max, float_t cutoffPrecision = defaultCutoffPrecision) { init(histogram, {min}, {max}, cutoffPrecision); };

  [[nodiscard]] inline const DatasetProperties<source_type>& getDatasetProperties() const noexcept { return mDatasetProperties; };
  [[nodiscard]] inline const CoderProperties<source_type>& getCoderProperties() const noexcept { return mCoderProperties; };

  [[nodiscard]] inline DatasetProperties<source_type>& getDatasetProperties() noexcept { return mDatasetProperties; };
  [[nodiscard]] inline CoderProperties<source_type>& getCoderProperties() noexcept { return mCoderProperties; };
  [[nodiscard]] inline SizeEstimate getSizeEstimate() const noexcept { return SizeEstimate(*this); };

 protected:
  template <typename histogram_T>
  void init(const histogram_T& histogram, std::optional<source_type> min, std::optional<source_type> max, float_t cutoffPrecision);

  template <typename histogram_T>
  void computeMetrics(const histogram_T& histogram, std::optional<source_type> min, std::optional<source_type> max);
  size_t computeRenormingPrecision(float_t cutoffPrecision) noexcept;
  size_t computeIncompressibleCount(gsl::span<uint32_t> distribution, uint32_t renormingPrecision) noexcept;

  DatasetProperties<source_type> mDatasetProperties{};
  CoderProperties<source_type> mCoderProperties{};
};

template <typename source_T>
template <typename histogram_T>
inline void Metrics<source_T>::init(const histogram_T& histogram, std::optional<source_type> min, std::optional<source_type> max, float_t cutoffPrecision)
{
  computeMetrics(histogram, min, max);
  mCoderProperties.renormingPrecisionBits = computeRenormingPrecision(cutoffPrecision);
  mCoderProperties.nIncompressibleSymbols = computeIncompressibleCount(mDatasetProperties.symbolLengthDistribution, *mCoderProperties.renormingPrecisionBits);
  mCoderProperties.nIncompressibleSamples = computeIncompressibleCount(mDatasetProperties.weightedSymbolLengthDistribution, *mCoderProperties.renormingPrecisionBits);
}

template <typename source_T>
template <typename histogram_T>
void Metrics<source_T>::computeMetrics(const histogram_T& histogram, std::optional<source_type> min, std::optional<source_type> max)
{
  using namespace internal;
  using namespace utils;
  using source_type = typename histogram_T::source_type;
  using value_type = typename histogram_T::value_type;
  static_assert(std::is_same_v<source_type, source_T>);

  mCoderProperties.dictSizeEstimate = DictSizeEstimate{histogram.getNumSamples()};
  if (histogram.getNumSamples() > 0) {
    const auto [trimmedBegin, trimmedEnd] = trim(histogram);
    if (min.has_value()) {
      mDatasetProperties.min = *min;
      mDatasetProperties.max = *max;
    } else {
      std::tie(mDatasetProperties.min, mDatasetProperties.max) = getMinMax(histogram, trimmedBegin, trimmedEnd);
    }
    assert(mDatasetProperties.max >= mDatasetProperties.min);
    mDatasetProperties.numSamples = histogram.getNumSamples();
    mDatasetProperties.alphabetRangeBits = getRangeBits(mDatasetProperties.min, mDatasetProperties.max);

    const double_t reciprocalNumSamples = 1.0 / static_cast<double_t>(histogram.getNumSamples());

    source_type lastIndex = mDatasetProperties.min;

    forEachIndexValue(histogram, trimmedBegin, trimmedEnd, [&, this](const source_type& index, const uint32_t& frequency) {
      if (frequency) {
        assert(lastIndex <= index);
        source_type delta = index - lastIndex;
        mCoderProperties.dictSizeEstimate.updateIndexSize(delta + (delta == 0));
        lastIndex = index;
        mCoderProperties.dictSizeEstimate.updateFreqSize(frequency);
        ++mDatasetProperties.nUsedAlphabetSymbols;

        const double_t probability = static_cast<double_t>(frequency) * reciprocalNumSamples;
        const float_t fractionalBitLength = -fastlog2(probability);
        const uint32_t bitLength = std::ceil(fractionalBitLength);

        assert(bitLength > 0);
        const uint32_t symbolDistributionBucket = bitLength - 1;
        mDatasetProperties.entropy += probability * fractionalBitLength;
        ++mDatasetProperties.symbolLengthDistribution[symbolDistributionBucket];
        mDatasetProperties.weightedSymbolLengthDistribution[symbolDistributionBucket] += frequency;
      }
    });
  }
};

template <typename source_T>
inline size_t Metrics<source_T>::computeIncompressibleCount(gsl::span<uint32_t> distribution, uint32_t renormingPrecision) noexcept
{
  assert(utils::isValidRenormingPrecision(renormingPrecision));
  size_t incompressibleCount = 0;
  if (renormingPrecision > 0) {
    incompressibleCount = std::accumulate(utils::advanceIter(distribution.data(), renormingPrecision), distribution.data() + distribution.size(), incompressibleCount);
  } else {
    // In case of an empty source message we allocate a precision of 0 Bits => 2**0 = 1
    // This 1 entry is marked as the incompressible symbol, to ensure we somewhat can handle nasty surprises.
    incompressibleCount = 1;
  };
  return incompressibleCount;
};

template <typename source_T>
inline size_t Metrics<source_T>::computeRenormingPrecision(float_t cutoffPrecision) noexcept
{

  const auto& dp = this->mDatasetProperties;

  constexpr size_t SafetyMargin = 1;
  const size_t cutoffSamples = std::ceil(static_cast<double_t>(cutoffPrecision) *
                                         static_cast<double_t>(dp.numSamples));
  size_t cumulatedSamples = 0;

  size_t renormingBits = std::count_if(dp.weightedSymbolLengthDistribution.begin(),
                                       dp.weightedSymbolLengthDistribution.end(),
                                       [&cumulatedSamples, cutoffSamples](const uint32_t& frequency) {
                                         if (cumulatedSamples < cutoffSamples) {
                                           cumulatedSamples += frequency;
                                           return true;
                                         } else {
                                           return false;
                                         }
                                       });

  if (cumulatedSamples == 0) {
    // if the message is empty, cumulated precision will be 0. The algorithm will be unable to meet the cutoff precision.
    // We therefore set renorming Bits to 0, which will result in 2**0 = 1 entry, which will be assigned to the incompressible symbol.
    renormingBits = 0;
  } else {
    // ensure renorming is in interval [MinThreshold, MaxThreshold]
    renormingBits = utils::sanitizeRenormingBitRange(renormingBits + SafetyMargin);
  }
  assert(utils::isValidRenormingPrecision(renormingBits));
  return renormingBits;
};

} // namespace o2::rans

#endif /* RANS_INTERNAL_METRICS_METRICS_H_ */
