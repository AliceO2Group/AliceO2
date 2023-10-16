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

/// @file   renorm.h
/// @author Michael Lettrich
/// @brief  Renorm histogram to sum of frequencies = 2^P for use in fast rans coding. Includes estimation of P.

#ifndef RANS_INTERNAL_TRANSFORM_RENORM_H_
#define RANS_INTERNAL_TRANSFORM_RENORM_H_

#include <fairlogger/Logger.h>

#include "rANS/internal/containers/RenormedHistogram.h"
#include "rANS/internal/containers/DenseHistogram.h"
#include "rANS/internal/containers/AdaptiveHistogram.h"
#include "rANS/internal/containers/SparseHistogram.h"
#include "rANS/internal/metrics/Metrics.h"
#include "rANS/internal/common/utils.h"
#include "rANS/internal/transform/algorithm.h"

namespace o2::rans
{

enum class RenormingPolicy { Auto,                  // make datadriven decision if a symbol will be marked as incompressible
                             ForceIncompressible }; // add a default incompressible symbol even if data does not require it

namespace renormImpl
{

template <typename source_T>
inline size_t getNUsedAlphabetSymbols(const DenseHistogram<source_T>& f)
{
  if constexpr (sizeof(source_T) <= 2) {
    const size_t nUsedAlphabetSymbols = f.empty() ? 0 : f.size();
    return nUsedAlphabetSymbols;
  } else {
    return countNUsedAlphabetSymbols(f);
  }
}

template <typename source_T>
inline size_t getNUsedAlphabetSymbols(const AdaptiveHistogram<source_T>& f)
{
  return countNUsedAlphabetSymbols(f);
}

template <typename source_T>
inline size_t getNUsedAlphabetSymbols(const SparseHistogram<source_T>& f)
{
  return f.size();
}

template <typename histogram_T>
decltype(auto) renorm(histogram_T histogram, Metrics<typename histogram_T::source_type>& metrics, RenormingPolicy renormingPolicy, size_t lowProbabilityCutoffBits = 0)
{
  using namespace o2::rans;
  using namespace o2::rans::internal;

  if (histogram.empty()) {
    LOG(warning) << "rescaling Frequency Table for empty message";
  }
  using histogram_type = histogram_T;
  using source_type = typename histogram_type::source_type;
  using count_type = typename histogram_type::value_type;
  using difference_type = typename histogram_type::difference_type;
  using container_type = typename histogram_type::container_type;
  using iterator_type = typename container_type::iterator;

  const double_t nSamples = histogram.getNumSamples();
  const size_t renormingPrecisionBits = *metrics.getCoderProperties().renormingPrecisionBits;
  const size_t nUsedAlphabetSymbols = metrics.getDatasetProperties().nUsedAlphabetSymbols;

  const count_type nSamplesRescaled = utils::pow2(renormingPrecisionBits);
  const double_t probabilityCutOffThreshold = 1.0 / static_cast<double_t>(utils::pow2(renormingPrecisionBits + lowProbabilityCutoffBits));

  // scaling
  double_t incompressibleSymbolProbability = 0;
  count_type nIncompressibleSamples = 0;
  count_type nIncompressibleSymbols = 0;
  count_type nSamplesRescaledUncorrected = 0;
  std::vector<std::pair<source_type, std::reference_wrapper<count_type>>> correctableIndices;
  correctableIndices.reserve(nUsedAlphabetSymbols);

  auto scaleFrequency = [nSamplesRescaled](double_t symbolProbability) -> double_t { return symbolProbability * nSamplesRescaled; };

  container_type rescaledHistogram = std::move(histogram).release();

  forEachIndexValue(rescaledHistogram, [&](const source_type& index, count_t& frequency) {
    if (frequency > 0) {
      const double_t symbolProbability = static_cast<double_t>(frequency) / nSamples;
      if (symbolProbability < probabilityCutOffThreshold) {
        nIncompressibleSamples += frequency;
        ++nIncompressibleSymbols;
        incompressibleSymbolProbability += symbolProbability;
        frequency = 0;
      } else {
        const double_t scaledFrequencyD = scaleFrequency(symbolProbability);
        count_type rescaledFrequency = internal::roundSymbolFrequency(scaledFrequencyD);
        assert(rescaledFrequency > 0);
        frequency = rescaledFrequency;
        nSamplesRescaledUncorrected += rescaledFrequency;
        if (rescaledFrequency > 1) {
          correctableIndices.emplace_back(std::make_pair(index, std::ref(frequency)));
        }
      }
    }
  });

  // treat incompressible symbol:
  const count_type incompressibleSymbolFrequency = [&]() -> count_type {
    // The Escape symbol for incompressible data is required
    const bool requireIncompressible = incompressibleSymbolProbability > 0.                          // if the algorithm eliminates infrequent symbols
                                       || nSamples == 0                                              // if the message we built the histogram from was empty
                                       || (renormingPolicy == RenormingPolicy::ForceIncompressible); // or we want to reuse the symbol table later with different data

    // if requireIncompressible == false it casts into 0, else it casts into 1 which is exactly our lower bound for each case, and we avoid branching.
    return std::max(static_cast<count_type>(requireIncompressible), static_cast<count_type>(incompressibleSymbolProbability * nSamplesRescaled));
  }();

  nSamplesRescaledUncorrected += incompressibleSymbolFrequency;

  // correction
  const auto nSorted = [&]() {
    const auto& datasetProperties = metrics.getDatasetProperties();
    float_t cumulProbability{};
    size_t nSymbols{};
    for (size_t i = 0; i < datasetProperties.weightedSymbolLengthDistribution.size(); ++i) {
      cumulProbability += datasetProperties.weightedSymbolLengthDistribution[i];
      nSymbols += datasetProperties.symbolLengthDistribution[i];
      if (cumulProbability > 0.99) {
        break;
      }
    }
    return nSymbols;
  }();

  if ((nSorted < correctableIndices.size()) && (renormingPolicy != RenormingPolicy::ForceIncompressible)) {
    std::partial_sort(correctableIndices.begin(), correctableIndices.begin() + nSorted, correctableIndices.end(), [](const auto& a, const auto& b) { return a.second < b.second; });
  } else {
    std::stable_sort(correctableIndices.begin(), correctableIndices.end(), [](const auto& a, const auto& b) { return a.second < b.second; });
  }

  difference_type nCorrections = static_cast<difference_type>(nSamplesRescaled) - static_cast<difference_type>(nSamplesRescaledUncorrected);
  const double_t rescalingFactor = static_cast<double_t>(nSamplesRescaled) / static_cast<double_t>(nSamplesRescaledUncorrected);

  for (auto& [index, value] : correctableIndices) {
    if (std::abs(nCorrections) > 0) {
      const difference_type uncorrectedFrequency = value;
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
      static_cast<count_type&>(value) = correctedFrequency;
    } else {
      break;
    }
  }

  if (std::abs(nCorrections) > 0) {
    throw HistogramError(fmt::format("rANS rescaling incomplete: {} corrections Remaining", nCorrections));
  }

  auto& coderProperties = metrics.getCoderProperties();
  *coderProperties.renormingPrecisionBits = renormingPrecisionBits;
  *coderProperties.nIncompressibleSymbols = nIncompressibleSymbols;
  *coderProperties.nIncompressibleSamples = nIncompressibleSamples;

  if constexpr (isDenseContainer_v<histogram_type>) {
    RenormedDenseHistogram<source_type> ret{std::move(rescaledHistogram), renormingPrecisionBits, incompressibleSymbolFrequency};
    std::tie(*coderProperties.min, *coderProperties.max) = getMinMax(ret);
    return ret;
  } else if constexpr (isAdaptiveContainer_v<histogram_type>) {
    RenormedAdaptiveHistogram<source_type> ret{std::move(rescaledHistogram), renormingPrecisionBits, incompressibleSymbolFrequency};
    std::tie(*coderProperties.min, *coderProperties.max) = getMinMax(ret);
    return ret;
  } else {
    static_assert(isSetContainer_v<histogram_type>);
    RenormedSparseHistogram<source_type> ret{std::move(rescaledHistogram), renormingPrecisionBits, incompressibleSymbolFrequency};
    std::tie(*coderProperties.min, *coderProperties.max) = getMinMax(ret);
    return ret;
  }
};
} // namespace renormImpl

template <typename histogram_T>
decltype(auto) renorm(histogram_T histogram, size_t newPrecision, RenormingPolicy renormingPolicy = RenormingPolicy::Auto, size_t lowProbabilityCutoffBits = 0)
{
  using source_type = typename histogram_T::source_type;
  const size_t nUsedAlphabetSymbols = renormImpl::getNUsedAlphabetSymbols(histogram);
  Metrics<source_type> metrics{};
  *metrics.getCoderProperties().renormingPrecisionBits = newPrecision;
  metrics.getDatasetProperties().nUsedAlphabetSymbols = nUsedAlphabetSymbols;
  return renormImpl::renorm(std::move(histogram), metrics, renormingPolicy, lowProbabilityCutoffBits);
};

template <typename histogram_T>
decltype(auto) renorm(histogram_T histogram, Metrics<typename histogram_T::source_type>& metrics, RenormingPolicy renormingPolicy = RenormingPolicy::Auto, size_t lowProbabilityCutoffBits = 0)
{
  return renormImpl::renorm(std::move(histogram), metrics, renormingPolicy, lowProbabilityCutoffBits);
};

template <typename histogram_T>
decltype(auto) renorm(histogram_T histogram, RenormingPolicy renormingPolicy = RenormingPolicy::Auto)
{
  using source_type = typename histogram_T::source_type;
  Metrics<source_type> metrics{histogram};
  return renorm(std::move(histogram), metrics, renormingPolicy);
};

} // namespace o2::rans

#endif /* RANS_INTERNAL_TRANSFORM_RENORM_H_ */
