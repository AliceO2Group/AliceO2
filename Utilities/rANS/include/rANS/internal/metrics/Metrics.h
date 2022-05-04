// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
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

#include "rANS/internal/common/utils.h"
#include "rANS/internal/containers/Histogram.h"
#include "rANS/internal/containers/RenormedHistogram.h"
#include "rANS/internal/containers/SymbolTable.h"
#include "rANS/internal/metrics/properties.h"
#include "rANS/internal/metrics/utils.h"

namespace o2::rans
{

template <typename source_T>
class Metrics
{
 public:
  using source_type = source_T;

  Metrics() = default;
  explicit Metrics(const Histogram<source_type>& histogram);
  explicit Metrics(const RenormedHistogram<source_type>& histogram);
  explicit Metrics(const Histogram<source_type>& histogram, size_t renormingPrecision);

  [[nodiscard]] const DatasetProperties<source_type>& getDatasetProperties() const noexcept { return mDatasetProperties; };
  [[nodiscard]] const CoderProperties<source_type>& getCoderProperties() const noexcept { return mCoderProperties; };

  void updateCoderProperties(const RenormedHistogram<source_type>& histogram);
  template <typename symbol_T>
  void updateCoderProperties(const SymbolTable<source_type, symbol_T>& symbolTable);

 private:
  template <class Histogram_T>
  void computeMetrics(const Histogram_T& frequencyTable);
  void updateCoderProperties(size_t newRenormingPrecisionBits, source_type min, source_type max, bool computeIncompressible = true);

  DatasetProperties<source_type> mDatasetProperties{};
  CoderProperties<source_type> mCoderProperties{};
};

template <typename source_T>
inline Metrics<source_T>::Metrics(const Histogram<source_T>& histogram)
{
  computeMetrics(histogram);
  const size_t renormingBits = internal::computeRenormingPrecision<>(mDatasetProperties.weightedSymbolLengthDistribution.begin(),
                                                                     mDatasetProperties.weightedSymbolLengthDistribution.end());
  const auto [min, max] = getMinMax(histogram);
  updateCoderProperties(renormingBits, min, max);
}

template <typename source_T>
inline Metrics<source_T>::Metrics(const Histogram<source_T>& histogram, size_t renormingPrecision)
{
  using namespace internal;

  computeMetrics(histogram);
  const size_t renormingBits = sanitizeRenormingBitRange(renormingPrecision);
  const auto [min, max] = getMinMax(histogram);
  updateCoderProperties(renormingPrecision, min, max);
}

template <typename source_T>
inline Metrics<source_T>::Metrics(const RenormedHistogram<source_T>& histogram)
{
  computeMetrics(histogram);
  updateCoderProperties(histogram);
};

template <typename source_T>
inline void Metrics<source_T>::updateCoderProperties(const RenormedHistogram<source_T>& histogram)
{
  const auto [min, max] = getMinMax(histogram);
  updateCoderProperties(histogram.getRenormingBits(), min, max, histogram.hasIncompressibleSymbol());
};

template <typename source_T>
template <typename symbol_T>
inline void Metrics<source_T>::updateCoderProperties(const SymbolTable<source_type, symbol_T>& symbolTable)
{
  const auto [min, max] = getMinMax(symbolTable);
  updateCoderProperties(symbolTable.getPrecision(), min, max, symbolTable.hasEscapeSymbol());
};

template <typename source_T>
template <class Histogram_T>
void Metrics<source_T>::computeMetrics(const Histogram_T& histogram)
{
  using namespace internal;
  static_assert(std::is_same_v<typename Histogram_T::source_type, source_type>);

  mCoderProperties.dictSizeEstimate = DictSizeEstimate{histogram.getNumSamples()};
  DictSizeEstimateCounter dictSizeCounter{&(mCoderProperties.dictSizeEstimate)};

  const auto trimmedFrequencyView = trim(makeHistogramView(histogram));
  mDatasetProperties.min = trimmedFrequencyView.getMin();
  mDatasetProperties.max = trimmedFrequencyView.getMax();
  assert(mDatasetProperties.max >= mDatasetProperties.min);
  mDatasetProperties.numSamples = histogram.getNumSamples();
  mDatasetProperties.alphabetRangeBits = getRangeBits(mDatasetProperties.min, mDatasetProperties.max);

  const double_t reciprocalNumSamples = 1.0 / static_cast<double_t>(histogram.getNumSamples());

  for (size_t i = 0; i < trimmedFrequencyView.size(); ++i) {
    const uint32_t frequency = trimmedFrequencyView.data()[i];
    dictSizeCounter.update();

    if (frequency) {
      dictSizeCounter.update(frequency);
      ++mDatasetProperties.nUsedAlphabetSymbols;

      const double_t probability = static_cast<double_t>(frequency) * reciprocalNumSamples;
      const float_t length = fastlog2(probability);

      mDatasetProperties.entropy -= probability * length;
      mDatasetProperties.symbolLengthDistribution[static_cast<uint32_t>(-length)] += frequency;
      mDatasetProperties.weightedSymbolLengthDistribution[static_cast<uint32_t>(-length)] += probability;
    }
  }
};

template <typename source_T>
inline void Metrics<source_T>::updateCoderProperties(size_t newRenormingPrecisionBits, source_type min, source_type max, bool computeIncompressible)
{

  mCoderProperties.renormingPrecisionBits = internal::sanitizeRenormingBitRange(newRenormingPrecisionBits);
  mCoderProperties.min = min;
  mCoderProperties.max = max;
  if (computeIncompressible) {
    mCoderProperties.nIncompressibleSymbols = internal::computeNIncompressibleSymbols<>(mDatasetProperties.symbolLengthDistribution.begin(),
                                                                                        mDatasetProperties.symbolLengthDistribution.end(),
                                                                                        mCoderProperties.renormingPrecisionBits);
  } else {
    mCoderProperties.nIncompressibleSymbols = 0;
  }
}

} // namespace o2::rans

#endif /* RANS_INTERNAL_METRICS_METRICS_H_ */
