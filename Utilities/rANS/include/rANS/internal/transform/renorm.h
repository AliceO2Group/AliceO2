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

/// @file   renorm.h
/// @author Michael Lettrich
/// @brief  Renorm histogram to sum of frequencies = 2^P for use in fast rans coding. Includes estimation of P.

#ifndef RANS_INTERNAL_TRANSFORM_RENORM_H_
#define RANS_INTERNAL_TRANSFORM_RENORM_H_

#include <fairlogger/Logger.h>

#include "rANS/internal/containers/RenormedHistogram.h"
#include "rANS/internal/containers/Histogram.h"
#include "rANS/internal/metrics/Metrics.h"
#include "rANS/internal/common/utils.h"

namespace o2::rans
{

namespace renormImpl
{

template <typename source_T>
inline size_t getNUsedAlphabetSymbols(const Histogram<source_T>& f)
{
  if constexpr (sizeof(source_T) <= 2) {
    const size_t nUsedAlphabetSymbols = f.empty() ? 0 : f.size();
    return nUsedAlphabetSymbols;
  } else {
    return f.countNUsedAlphabetSymbols();
  }
}

template <typename source_T>
RenormedHistogram<source_T> renorm(Histogram<source_T> histogram, Metrics<source_T>& metrics, bool forceIncompressible, size_t lowProbabilityCutoffBits = 0)
{
  using namespace o2::rans;
  using namespace o2::rans::internal;

  if (histogram.empty()) {
    LOG(warning) << "rescaling Frequency Table for empty message";
  }

  using source_type = source_T;
  using count_type = typename Histogram<source_T>::value_type;
  using difference_type = typename Histogram<source_T>::difference_type;
  using container_type = typename Histogram<source_T>::container_type;
  using iterator_type = typename container_type::iterator;

  const source_type offset = histogram.getOffset();
  const double_t nSamples = histogram.getNumSamples();
  const size_t renormingPrecisionBits = *metrics.getCoderProperties().renormingPrecisionBits;
  const size_t nUsedAlphabetSymbols = metrics.getDatasetProperties().nUsedAlphabetSymbols;

  const count_type nSamplesRescaled = pow2(renormingPrecisionBits);
  const double_t probabilityCutOffThreshold = 1.0 / static_cast<double_t>(pow2(renormingPrecisionBits + lowProbabilityCutoffBits));

  // scaling
  double_t incompressibleSymbolProbability = 0;
  count_type nIncompressibleSamples = 0;
  count_type nIncompressibleSymbols = 0;
  count_type nSamplesRescaledUncorrected = 0;
  std::vector<iterator_type> correctableIndices;
  correctableIndices.reserve(nUsedAlphabetSymbols);

  auto scaleFrequency = [nSamplesRescaled](double_t symbolProbability) -> double_t { return symbolProbability * nSamplesRescaled; };

  container_type rescaledHistogram = std::move(histogram).release();

  for (auto frequencyIter = rescaledHistogram.begin(); frequencyIter != rescaledHistogram.end(); ++frequencyIter) {
    const count_type frequency = *frequencyIter;
    if (frequency > 0) {
      const double_t symbolProbability = static_cast<double_t>(frequency) / nSamples;
      if (symbolProbability < probabilityCutOffThreshold) {
        nIncompressibleSamples += frequency;
        ++nIncompressibleSymbols;
        incompressibleSymbolProbability += symbolProbability;
        *frequencyIter = 0;
      } else {
        const double_t scaledFrequencyD = scaleFrequency(symbolProbability);
        count_type rescaledFrequency = internal::roundSymbolFrequency(scaledFrequencyD);
        assert(rescaledFrequency > 0);
        *frequencyIter = rescaledFrequency;
        nSamplesRescaledUncorrected += rescaledFrequency;
        if (rescaledFrequency > 1) {
          correctableIndices.push_back(frequencyIter);
        }
      }
    }
  }

  // treat incompressible symbol:
  const count_type incompressibleSymbolFrequency = [&]() -> count_type {
    // The Escape symbol for incompressible data is required
    const bool requireIncompressible = incompressibleSymbolProbability > 0. // if the algorithm eliminates infrequent symbols
                                       || nSamples == 0                     // if the message we built the histogram from was empty
                                       || forceIncompressible;              // or we want to reuse the symbol table later with different data

    // if requireIncompressible == false it casts into 0, else it casts into 1 which is exactly our lower bound for each case, and we avoid branching.
    return std::max(static_cast<count_type>(requireIncompressible), static_cast<count_type>(incompressibleSymbolProbability * nSamplesRescaled));
  }();

  nSamplesRescaledUncorrected += incompressibleSymbolFrequency;

  // correction
  std::stable_sort(correctableIndices.begin(), correctableIndices.end(), [&rescaledHistogram](const iterator_type& a, const iterator_type& b) { return *a < *b; });

  difference_type nCorrections = static_cast<difference_type>(nSamplesRescaled) - static_cast<difference_type>(nSamplesRescaledUncorrected);
  const double_t rescalingFactor = static_cast<double_t>(nSamplesRescaled) / static_cast<double_t>(nSamplesRescaledUncorrected);

  for (auto iter : correctableIndices) {
    if (std::abs(nCorrections) > 0) {
      const difference_type uncorrectedFrequency = *iter;
      difference_type correction = uncorrectedFrequency - roundSymbolFrequency(uncorrectedFrequency * rescalingFactor);

      if (nCorrections < 0) {
        // overshoot - correct downwards by subtracting correction in [1,|nCorrections|]
        correction = std::max(1l, std::min(correction, std::abs(nCorrections)));
      } else {
        // correct upwards by subtracting correction in [-1, -nCorrections]
        correction = std::min(-1l, std::max(correction, -nCorrections));
      }

      // the corrected frequency must be at least 1 though
      const count_type correctedFrequency = std::max(1l, uncorrectedFrequency - correction);
      nCorrections += uncorrectedFrequency - correctedFrequency;
      *iter = correctedFrequency;
    } else {
      break;
    }
  }

  if (std::abs(nCorrections) > 0) {
    throw HistogramError(fmt::format("rANS rescaling incomplete: {} corrections Remaining", nCorrections));
  }

  RenormedHistogram<source_type> ret{std::move(rescaledHistogram), renormingPrecisionBits, incompressibleSymbolFrequency};

  auto& coderProperties = metrics.getCoderProperties();
  *coderProperties.renormingPrecisionBits = renormingPrecisionBits;
  *coderProperties.nIncompressibleSymbols = nIncompressibleSymbols;
  *coderProperties.nIncompressibleSamples = nIncompressibleSamples;
  std::tie(*coderProperties.min, *coderProperties.max) = getMinMax(ret);

  return ret;
};

} // namespace renormImpl

template <typename source_T>
RenormedHistogram<source_T> renorm(Histogram<source_T> histogram, size_t newPrecision, bool forceIncompressible = false, size_t lowProbabilityCutoffBits = 0)
{
  const size_t nUsedAlphabetSymbols = renormImpl::getNUsedAlphabetSymbols(histogram);
  Metrics<source_T> metrics{};
  *metrics.getCoderProperties().renormingPrecisionBits = newPrecision;
  metrics.getDatasetProperties().nUsedAlphabetSymbols = nUsedAlphabetSymbols;
  return renormImpl::renorm(std::move(histogram), metrics, forceIncompressible, lowProbabilityCutoffBits);
};

template <typename source_T>
RenormedHistogram<source_T> renorm(Histogram<source_T> histogram, Metrics<source_T>& metrics, bool forceIncompressible = false, size_t lowProbabilityCutoffBits = 0)
{
  return renormImpl::renorm(std::move(histogram), metrics, forceIncompressible, lowProbabilityCutoffBits);
};

template <typename source_T>
RenormedHistogram<source_T> renorm(Histogram<source_T> histogram, bool forceIncompressible = false)
{
  Metrics<source_T> metrics{histogram};
  return renorm(std::move(histogram), metrics, forceIncompressible);
};

} // namespace o2::rans

#endif /* RANS_INTERNAL_TRANSFORM_RENORM_H_ */
