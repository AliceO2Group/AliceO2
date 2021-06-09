// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

void FrequencyTable::resizeFrequencyTable(symbol_t min, symbol_t max)
{
  LOG(trace) << "start resizing frequency table";
  internal::RANSTimer t;
  t.start();
  // calculate new dimensions
  const symbol_t newMin = std::min(mMin, min);
  const symbol_t newMax = std::max(mMax, max);
  const size_t newSize = newMax - newMin + 1;

  // empty - init and prevent special treatment of corner cases when resizing.
  if (mFrequencyTable.empty()) {
    mFrequencyTable.resize(newSize, 0);
  }

  // if the new size is bigger than the old one we need to resize the frequency table
  if (newSize > mFrequencyTable.size()) {
    const size_t offset = newMin < mMin ? std::abs(min - mMin) : 0;
    std::vector<count_t> tmpFrequencyTable;
    tmpFrequencyTable.reserve(newSize);
    // insert initial offset if applicable
    tmpFrequencyTable.insert(std::begin(tmpFrequencyTable), offset, 0);
    // append current frequency table
    tmpFrequencyTable.insert(std::end(tmpFrequencyTable), mFrequencyTable.begin(), mFrequencyTable.end());
    //fill tail with zeroes if applicable
    const size_t tail = newMax > mMax ? std::abs(max - mMax) : 0;
    tmpFrequencyTable.insert(std::end(tmpFrequencyTable), tail, 0);

    mFrequencyTable = std::move(tmpFrequencyTable);
  }

  assert(mFrequencyTable.size() == newSize);
  mMin = newMin;
  mMax = newMax;

  t.stop();
  LOG(debug1) << __func__ << " inclusive time (ms): " << t.getDurationMS();
  LOG(trace) << "done resizing frequency table";
}

std::ostream& operator<<(std::ostream& out, const FrequencyTable& fTable)
{
  double entropy = 0;
  for (auto frequency : fTable) {
    if (frequency > 0) {
      const double p = (frequency * 1.0) / fTable.getNumSamples();
      entropy -= p * std::log2(p);
    }
  }

  out << "FrequencyTable: {"
      << "numSymbols: " << fTable.getNumSamples() << ", "
      << "alphabetRange: " << fTable.getAlphabetRangeBits() << ", "
      << "alphabetSize: " << fTable.getNUsedAlphabetSymbols() << ", "
      << "minSymbol: " << fTable.getMinSymbol() << ", "
      << "maxSymbol: " << fTable.getMaxSymbol() << ", "
      << "sizeFrequencyTableB: " << fTable.size() << ", "
      << "sizeFrequencyTableB: " << fTable.size() * sizeof(typename o2::rans::FrequencyTable::symbol_t) << ", "
      << "entropy: " << entropy << "}";

  return out;
}

} //namespace rans
} // namespace o2
