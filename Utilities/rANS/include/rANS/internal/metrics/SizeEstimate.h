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

/// @file   SizeEstimate.h
/// @author Michael Lettrich
/// @brief  Estimate sizes of different rANS Buffers and decide if packing should be prefered over compression.

#ifndef RANS_INTERNAL_METRICS_SIZEESTIMATE_H_
#define RANS_INTERNAL_METRICS_SIZEESTIMATE_H_

#include <cstdint>
#include <cmath>

#include <fairlogger/Logger.h>

#include "rANS/internal/common/defaults.h"
#include "rANS/internal/common/typetraits.h"
#include "rANS/internal/common/codertraits.h"
#include "rANS/internal/common/utils.h"
#include "rANS/internal/metrics/Metrics.h"

namespace o2::rans
{

template <CoderTag tag_V = defaults::DefaultTag, size_t lowerBound_V = defaults::CoderPreset<tag_V>::renormingLowerBound>
inline constexpr size_t addEncoderOverheadEstimateB(size_t sizeB) noexcept
{
  constexpr size_t nStreams = defaults::CoderPreset<tag_V>::nStreams;
  using state_type = typename internal::CoderTraits_t<tag_V>::state_type;
  constexpr size_t minSize = nStreams * sizeof(state_type); // mandatory size of flushing
  constexpr size_t overhead = utils::toBytes(lowerBound_V * nStreams);

  return std::max(minSize, sizeB + overhead);
}

template <typename source_T>
class Metrics;

class SizeEstimate
{
 public:
  inline SizeEstimate() = default;

  template <typename source_T>
  inline explicit SizeEstimate(const Metrics<source_T>& metrics) noexcept;

  [[nodiscard]] size_t getEntropySizeB() const;
  template <typename T = uint8_t>
  [[nodiscard]] size_t getCompressedDatasetSize(double_t safetyFactor = 1.2) const;
  template <typename T = uint8_t>
  [[nodiscard]] size_t getCompressedDictionarySize(double_t safetyFactor = 2) const;
  template <typename T = uint8_t>
  [[nodiscard]] size_t getIncompressibleSize(double_t safetyFactor = 1.2) const;
  template <typename T = uint8_t>
  [[nodiscard]] size_t getPackedDatasetSize(double_t safetyFactor = 1) const;

  [[nodiscard]] inline bool preferPacking(double_t weight = 1) const;

 private:
  size_t mEntropySizeB{};
  size_t mCompressedDatasetSizeB{};
  size_t mCompressedDictionarySizeB{};
  size_t mIncompressibleSizeB{};
  size_t mPackedDatasetSizeB{};
};

template <typename source_T>
inline SizeEstimate::SizeEstimate(const Metrics<source_T>& metrics) noexcept
{
  const auto& datasetProperties = metrics.getDatasetProperties();
  const auto& coderProperties = metrics.getCoderProperties();
  const auto& nSamples = datasetProperties.numSamples;

  if (nSamples > 0) {
    mEntropySizeB = utils::toBytes(datasetProperties.entropy * nSamples);
    mCompressedDatasetSizeB = addEncoderOverheadEstimateB<>(mEntropySizeB);
    mCompressedDictionarySizeB = coderProperties.dictSizeEstimate.getSizeB(datasetProperties.nUsedAlphabetSymbols,
                                                                           *coderProperties.renormingPrecisionBits);
    mIncompressibleSizeB = utils::toBytes(datasetProperties.alphabetRangeBits * (*coderProperties.nIncompressibleSamples));
    mPackedDatasetSizeB = utils::toBytes(datasetProperties.alphabetRangeBits * nSamples);
  } else {
    // special case: store no data for empty dataset
    mEntropySizeB = 0;
    mCompressedDatasetSizeB = 0;
    mCompressedDictionarySizeB = 0;
    mIncompressibleSizeB = 0;
    mPackedDatasetSizeB = 0;
  }
};

[[nodiscard]] inline size_t SizeEstimate::getEntropySizeB() const
{
  return mEntropySizeB;
};

template <typename T>
[[nodiscard]] inline size_t SizeEstimate::getCompressedDatasetSize(double_t safetyFactor) const
{
  return utils::nBytesTo<T>(std::ceil(mCompressedDatasetSizeB * safetyFactor));
};

template <typename T>
[[nodiscard]] inline size_t SizeEstimate::getCompressedDictionarySize(double_t safetyFactor) const
{
  constexpr size_t MaxOverhead = 8; // maximal absolute overhead
  return utils::nBytesTo<T>(std::ceil(mCompressedDictionarySizeB * safetyFactor) + MaxOverhead);
};

template <typename T>
[[nodiscard]] inline size_t SizeEstimate::getIncompressibleSize(double_t safetyFactor) const
{
  return utils::nBytesTo<T>(std::ceil(mIncompressibleSizeB * safetyFactor));
};

template <typename T>
[[nodiscard]] inline size_t SizeEstimate::getPackedDatasetSize(double_t safetyFactor) const
{
  return utils::nBytesTo<T>(std::ceil(mPackedDatasetSizeB * safetyFactor));
};

[[nodiscard]] inline bool SizeEstimate::preferPacking(double_t weight) const
{
  // convention: always pack empty dataset.
  return (mPackedDatasetSizeB * weight) <= (mCompressedDatasetSizeB +
                                            mCompressedDictionarySizeB +
                                            mIncompressibleSizeB);
};

} // namespace o2::rans

#endif /* RANS_INTERNAL_METRICS_SIZEESTIMATE_H_ */
