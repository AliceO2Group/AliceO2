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

/// @file   FrequencyTable.cxx
/// @author Michael Lettrich
/// @since  Aug 1, 2020
/// @brief Implementation of a frequency table for rANS symbole (i.e. a histogram)

#include "rANS/FrequencyTable.h"

namespace o2
{
namespace rans
{

FrequencyTable& FrequencyTable::trim()
{
  auto trimmedHistogram = utils::trim(utils::HistogramView(mFrequencyTable.begin(), mFrequencyTable.end(), this->mOffset));

  histogram_t newFrequencyTable{trimmedHistogram.begin(), trimmedHistogram.end()};
  mFrequencyTable = std::move(newFrequencyTable);
  mOffset = trimmedHistogram.getOffset();

  return *this;
}

FrequencyTable& FrequencyTable::resize(symbol_t min, symbol_t max, bool truncate)
{
  assert(max >= min);

  const size_t newSize = max - min + 1;
  const symbol_t oldOffset = mOffset;
  mOffset = min;
  mNumSamples = 0;

  if (this->empty()) {
    mFrequencyTable.resize(newSize, 0);
    return *this;
  } else {
    histogram_t oldFrequencyTable = std::move(mFrequencyTable);
    auto oldHistogram = utils::HistogramView{oldFrequencyTable.begin(), oldFrequencyTable.end(), oldOffset};
    mFrequencyTable = histogram_t(newSize, 0);

    const bool extendTable = !truncate;
    return this->addFrequencies(oldHistogram.begin(), oldHistogram.end(), oldHistogram.getMin(), extendTable);
  }
}

inline double_t computeEntropy(const FrequencyTable& table)
{
  double_t entropy = std::accumulate(table.begin(), table.end(), 0, [&table](double_t entropy, count_t frequency) {
    const double_t p = static_cast<double_t>(frequency) / static_cast<double_t>(table.getNumSamples());
    const double_t length = p == 0 ? 0 : std::log2(p);
    return entropy -= p * length;
  });
  entropy += [&table]() {
    const double_t p = static_cast<double_t>(table.getIncompressibleSymbolFrequency()) / static_cast<double_t>(table.getNumSamples());
    return p * (-std::log2(p) + table.getAlphabetRangeBits());
  }();

  return entropy;
};

count_t computeRenormingPrecision(const FrequencyTable& frequencyTable)
{
  const auto nused = frequencyTable.getNUsedAlphabetSymbols();
  const uint8_t minBits = nused > 0 ? std::ceil(std::log2(nused)) : 0;
  const uint8_t estimate = minBits * 3u / 2u;
  const uint8_t maxThreshold = std::max(minBits, MaxRenormThreshold);
  const uint8_t minThreshold = std::max(estimate, MinRenormThreshold);

  return std::min(minThreshold, maxThreshold);
};

std::ostream& operator<<(std::ostream& out, const FrequencyTable& fTable)
{
  out << "FrequencyTable: {"
      << "numSymbols: " << fTable.getNumSamples() << ", "
      << "alphabetRange: " << fTable.getAlphabetRangeBits() << ", "
      << "alphabetSize: " << fTable.getNUsedAlphabetSymbols() << ", "
      << "minSymbol: " << fTable.getMinSymbol() << ", "
      << "maxSymbol: " << fTable.getMaxSymbol() << ", "
      << "incompressibleSymbolFrequency: " << fTable.getIncompressibleSymbolFrequency() << ", "
      << "sizeFrequencyTable: " << fTable.size() << ", "
      << "sizeFrequencyTableB: " << fTable.size() * sizeof(typename o2::rans::symbol_t) << ", "
      << "entropy: " << computeEntropy(fTable) << "}";

  return out;
}

} // namespace rans
} // namespace o2
