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

/// @file   utils.h
/// @author Michael Lettrich
/// @brief  utility functions related to calculating different dataset metrics

#ifndef RANS_INTERNAL_METRICS_UTILS_H_
#define RANS_INTERNAL_METRICS_UTILS_H_

#include <cstddef>
#include <cmath>
#include <cstdint>

#include "rANS/internal/common/utils.h"
#include "rANS/internal/containers/Histogram.h"
#include "rANS/internal/containers/RenormedHistogram.h"
#include "rANS/internal/containers/HistogramView.h"

namespace o2::rans
{
namespace internal
{
template <typename IT>
inline constexpr size_t computeNIncompressibleSymbols(IT begin, IT end, uint32_t renormingPrecision) noexcept
{
  assert(isValidRenormingPrecision(renormingPrecision));
  size_t nIncompressibleSymbols = 0;

  if (renormingPrecision > 0) {
    for (auto frequencyIter = internal::advanceIter(begin, renormingPrecision); frequencyIter != end; ++frequencyIter) {
      nIncompressibleSymbols += *frequencyIter;
    }
  } else {
    // In case of an empty source message we allocate a precision of 0 Bits => 2**0 = 1
    // This 1 entry is marked as the incompressible symbol, to ensure we somewhat can handle nasty surprises.
    nIncompressibleSymbols = 1;
  };
  return nIncompressibleSymbols;
};

template <typename IT>
inline constexpr size_t computeRenormingPrecision(IT begin, IT end, float_t cutoffPrecision = 0.999) noexcept
{
  constexpr size_t SafetyMargin = 1;
  float_t cumulatedPrecision = 0;
  size_t renormingBits = 0;

  for (auto iter = begin; iter != end && cumulatedPrecision < cutoffPrecision; ++iter) {
    cumulatedPrecision += *iter;
    ++renormingBits;
  }

  if (cumulatedPrecision == 0) {
    // if the message is empty, cumulated precision will be 0. The algorithm will be unable to meet the cutoff precision.
    // We therefore set renorming Bits to 0, which will result in 2**0 = 1 entry, which will be assigned to the incompressible symbol.
    renormingBits = 0;
  } else {
    // ensure renorming is in interval [MinThreshold, MaxThreshold]
    renormingBits = sanitizeRenormingBitRange(renormingBits + SafetyMargin);
  }
  assert(isValidRenormingPrecision(renormingBits));
  return renormingBits;
};
} // namespace internal

template <typename source_T>
double_t computeExpectedCodewordLength(const Histogram<source_T>& histogram, const RenormedHistogram<source_T>& rescaledHistogram)
{
  assert(histogram.getNumSamples() > 0);
  assert(rescaledHistogram.getNumSamples() > 0);

  using namespace internal;
  using value_type = typename Histogram<source_T>::value_type;

  const auto histogramView = makeHistogramView(histogram);
  const auto renormedView = makeHistogramView(rescaledHistogram);

  auto getRescaledFrequency = [&renormedView](source_T sourceSymbol) -> value_type {
    if (sourceSymbol >= renormedView.getMin() && sourceSymbol <= renormedView.getMax()) {
      return renormedView[sourceSymbol];
    } else {
      return static_cast<value_type>(0);
    }
  };

  double_t expectedCodewordLength = 0;
  value_type trueIncompressibleFrequency = 0;

  assert(histogram.countNUsedAlphabetSymbols() >= rescaledHistogram.countNUsedAlphabetSymbols());

  double_t reciprocalNumSamples = 1.0 / histogram.getNumSamples();
  double_t reciprocalNumSamplesRescaled = 1.0 / rescaledHistogram.getNumSamples();

  // all "normal symbols"
  for (value_type sourceSymbol = histogramView.getMin(); sourceSymbol <= histogramView.getMax(); ++sourceSymbol) {

    const value_type frequency = histogramView[sourceSymbol];
    if (frequency) {
      const value_type rescaledFrequency = getRescaledFrequency(sourceSymbol);
      const double_t trueProbability = static_cast<double_t>(frequency) * reciprocalNumSamples;

      if (rescaledFrequency) {
        const double_t rescaledProbability = static_cast<double_t>(rescaledFrequency) * reciprocalNumSamplesRescaled;
        expectedCodewordLength -= trueProbability * fastlog2(rescaledProbability);
      } else {
        trueIncompressibleFrequency += frequency;
      }
    }
  }

  // incompressibleSymbol:
  const double_t trueIncompressibleProbability = static_cast<double_t>(trueIncompressibleFrequency) * reciprocalNumSamples;
  if (trueIncompressibleProbability) {
    const double_t rescaledProbability = static_cast<double_t>(rescaledHistogram.getIncompressibleSymbolFrequency()) * reciprocalNumSamplesRescaled;
    expectedCodewordLength -= trueIncompressibleProbability * fastlog2(rescaledProbability);
    expectedCodewordLength += trueIncompressibleProbability * fastlog2(numBitsForNSymbols(renormedView.size()));
  }

  return expectedCodewordLength;
};

} // namespace o2::rans

#endif /* RANS_INTERNAL_METRICS_UTILS_H_ */