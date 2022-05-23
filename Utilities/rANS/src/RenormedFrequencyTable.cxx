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

/// @file   RenormedFrequencyTable.cxx
/// @author Michael Lettrich
/// @since  Aug 1, 2020
/// @brief Implementation of a frequency table for rANS symbole (i.e. a histogram)

#include "rANS/RenormedFrequencyTable.h"

namespace o2
{
namespace rans
{

RenormedFrequencyTable renorm(FrequencyTable frequencyTable, size_t newPrecision)
{

  using namespace internal;
  LOG(trace) << "start rescaling frequency table";
  RANSTimer t;
  t.start();

  if (frequencyTable.empty()) {
    LOG(warning) << "rescaling Frequency Table for empty message";
  }

  if (newPrecision == 0) {
    newPrecision = computeRenormingPrecision(frequencyTable);
  }

  size_t nSamples = frequencyTable.getNumSamples();
  count_t nIncompressible = frequencyTable.getIncompressibleSymbolFrequency();
  count_t nUsedAlphabetSymbols = frequencyTable.getNUsedAlphabetSymbols();
  const symbol_t offset = frequencyTable.getMinSymbol();

  // add an incompressible symbol if not present
  if (!frequencyTable.hasIncompressibleSymbols()) {
    nIncompressible = 1;
    nSamples += nIncompressible;
    nUsedAlphabetSymbols += 1;
  }

  histogram_t frequencies = std::move(frequencyTable).release();
  frequencies.push_back(nIncompressible);

  std::vector<uint64_t> cumulativeFrequencies(frequencies.size() + 1);
  cumulativeFrequencies[0] = 0;
  std::inclusive_scan(frequencies.begin(), frequencies.end(), ++cumulativeFrequencies.begin(), std::plus<>(), 0ull);

  auto getFrequency = [&cumulativeFrequencies](count_t i) {
    assert(cumulativeFrequencies[i + 1] >= cumulativeFrequencies[i]);
    return cumulativeFrequencies[i + 1] - cumulativeFrequencies[i];
  };

  const auto sortIdx = [&]() {
    std::vector<size_t> indices;
    indices.reserve(nUsedAlphabetSymbols);

    // we will sort only those memorize only those entries which can be used
    for (size_t i = 0; i < frequencies.size(); i++) {
      if (frequencies[i] != 0) {
        indices.push_back(i);
      }
    }
    std::sort(indices.begin(), indices.end(), [&](count_t i, count_t j) { return getFrequency(i) < getFrequency(j); });

    return indices;
  }();

  // resample distribution based on cumulative frequencies
  const count_t newCumulatedFrequency = pow2(newPrecision);
  assert(newCumulatedFrequency >= nUsedAlphabetSymbols);
  size_t needsShift = 0;
  for (size_t i = 0; i < sortIdx.size(); i++) {
    if (static_cast<count_t>(getFrequency(sortIdx[i])) * (newCumulatedFrequency - needsShift) / nSamples >= 1) {
      break;
    }
    needsShift++;
  }

  size_t shift = 0;
  auto beforeUpdate = cumulativeFrequencies[0];
  for (size_t i = 0; i < frequencies.size(); i++) {
    if (frequencies[i] && static_cast<uint64_t>(cumulativeFrequencies[i + 1] - beforeUpdate) * (newCumulatedFrequency - needsShift) / nSamples < 1) {
      shift++;
    }
    beforeUpdate = cumulativeFrequencies[i + 1];
    cumulativeFrequencies[i + 1] = (static_cast<uint64_t>(newCumulatedFrequency - needsShift) * cumulativeFrequencies[i + 1]) / nSamples + shift;
  }
  assert(shift == needsShift);

  // verify
#if !defined(NDEBUG)
  assert(cumulativeFrequencies.front() == 0 &&
         cumulativeFrequencies.back() == newCumulatedFrequency);
  for (size_t i = 0; i < frequencies.size(); i++) {
    if (frequencies[i] == 0) {
      assert(cumulativeFrequencies[i + 1] == cumulativeFrequencies[i]);
    } else {
      assert(cumulativeFrequencies[i + 1] > cumulativeFrequencies[i]);
    }
  }
#endif

  // calculate updated frequencies
  for (size_t i = 0; i < frequencies.size(); i++) {
    frequencies[i] = getFrequency(i);
  }

  assert(frequencies.size() >= 1);
  FrequencyTable rescaledFrequencies{frequencies.begin(), --frequencies.end(), offset, frequencies.back()};

  t.stop();
  LOG(debug1) << __func__ << " inclusive time (ms): " << t.getDurationMS();

  LOG(trace) << "done rescaling frequency table";
  return RenormedFrequencyTable{std::move(rescaledFrequencies), newPrecision};
};

RenormedFrequencyTable renormCutoffIncompressible(FrequencyTable frequencyTable, uint8_t newPrecision, uint8_t lowProbabilityCutoffBits)
{
  using namespace internal;
  LOG(trace) << "start rescaling frequency table";
  RANSTimer t;
  t.start();

  if (frequencyTable.empty()) {
    LOG(warning) << "rescaling Frequency Table for empty message";
  }

  if (newPrecision == 0) {
    newPrecision = computeRenormingPrecision(frequencyTable);
  }

  const count_t nSamplesRescaled = 1 << newPrecision;
  const double_t probabilityCutOffThreshold = 1 / static_cast<double_t>(1 << (newPrecision + lowProbabilityCutoffBits));

  // scaling
  double_t incompressibleSymbolProbability = static_cast<double_t>(frequencyTable.getIncompressibleSymbolFrequency()) / nSamplesRescaled;
  count_t nSamplesRescaledUncorrected = 0;
  std::vector<size_t> correctableIndices;
  correctableIndices.reserve(frequencyTable.getNUsedAlphabetSymbols());

  auto scaleFrequency = [nSamplesRescaled](double_t symbolProbability) { return symbolProbability * nSamplesRescaled; };
  auto roundDownFrequency = [](double_t i) { return static_cast<count_t>(i); };
  auto roundUpFrequency = [roundDownFrequency](count_t i) { return roundDownFrequency(i) + 1; };
  auto roundFrequency = [roundDownFrequency, roundUpFrequency](double_t rescaledFrequency) {
    if (rescaledFrequency * rescaledFrequency <= (roundDownFrequency(rescaledFrequency) * roundUpFrequency(rescaledFrequency))) {
      return roundDownFrequency(rescaledFrequency);
    } else {
      return roundUpFrequency(rescaledFrequency);
    }
  };

  histogram_t rescaledFrequencies(frequencyTable.size());

  for (size_t i = 0; i < frequencyTable.size(); ++i) {
    const count_t frequency = frequencyTable.at(i);
    const double_t symbolProbability = static_cast<double_t>(frequency) / static_cast<double_t>(frequencyTable.getNumSamples());
    if (symbolProbability < probabilityCutOffThreshold) {
      incompressibleSymbolProbability += symbolProbability;
      rescaledFrequencies[i] = 0;
    } else {
      const double_t scaledFrequencyD = scaleFrequency(symbolProbability);
      count_t rescaledFrequency = roundFrequency(scaledFrequencyD);
      assert(rescaledFrequency > 0);
      rescaledFrequencies[i] = rescaledFrequency;
      nSamplesRescaledUncorrected += rescaledFrequency;
      if (rescaledFrequency > 1) {
        correctableIndices.push_back(i);
      }
    }
  }

  // treat incompressible symbol
  const count_t incompressibleSymbolFrequency = std::max(static_cast<count_t>(1), static_cast<count_t>(incompressibleSymbolProbability * nSamplesRescaled));
  nSamplesRescaledUncorrected += incompressibleSymbolFrequency;

  // correction
  std::stable_sort(correctableIndices.begin(), correctableIndices.end(), [&rescaledFrequencies](const count_t& a, const count_t& b) { return rescaledFrequencies[a] < rescaledFrequencies[b]; });

  int32_t nCorrections = nSamplesRescaled - nSamplesRescaledUncorrected;
  const double_t rescalingFactor = static_cast<double_t>(nSamplesRescaled) / static_cast<double_t>(nSamplesRescaledUncorrected);

  for (auto index : correctableIndices) {
    if (std::abs(nCorrections) > 0) {
      const count_t uncorrectedFrequency = rescaledFrequencies[index];
      int32_t correction = uncorrectedFrequency - roundFrequency(uncorrectedFrequency * rescalingFactor);

      if (nCorrections < 0) {
        // overshoot - correct downwards by subtracting correction in [1,|nCorrections|]
        correction = std::max(1, std::min(correction, std::abs(nCorrections)));
      } else {
        // correct upwards by subtracting correction in [-1, -nCorrections]
        correction = std::min(-1, std::max(correction, -nCorrections));
      }

      // the corrected frequency must be at least 1 though
      const count_t correctedFrequency = std::max(1u, uncorrectedFrequency - correction);
      nCorrections += uncorrectedFrequency - correctedFrequency;
      rescaledFrequencies[index] = correctedFrequency;
    } else {
      break;
    }
  }

  if (std::abs(nCorrections) > 0) {
    throw std::runtime_error(fmt::format("rANS rescaling incomplete: {} corrections Remaining", nCorrections));
  }

  FrequencyTable newFrequencyTable{rescaledFrequencies.begin(), rescaledFrequencies.end(), frequencyTable.getMinSymbol(), incompressibleSymbolFrequency};

  t.stop();
  LOG(debug1) << __func__ << " inclusive time (ms): " << t.getDurationMS();

  LOG(trace) << "done rescaling frequency table";
  return RenormedFrequencyTable{std::move(newFrequencyTable), newPrecision};
}

} // namespace rans
} // namespace o2