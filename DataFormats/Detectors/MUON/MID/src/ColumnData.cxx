// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/src/ColumnData.cxx
/// \brief  Strip pattern (aka digits)
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   19 February 2018

#include "DataFormatsMID/ColumnData.h"

#include <bitset>

namespace o2
{
namespace mid
{

void ColumnData::setPattern(uint16_t pattern, int cathode, int line)
{
  /// Sets the pattern
  if (cathode == 0)
    setBendPattern(pattern, line);
  else
    setNonBendPattern(pattern);
}

uint16_t ColumnData::getPattern(int cathode, int line) const
{
  /// Gets the pattern
  return (cathode == 0) ? getBendPattern(line) : getNonBendPattern();
}

void ColumnData::addStrip(int strip, int cathode, int line)
{
  /// Adds a strip to the pattern
  int ipat = (cathode == 1) ? 4 : line;
  patterns[ipat] |= (1 << strip);
}

bool ColumnData::isStripFired(int istrip, int cathode, int line) const
{
  /// Checks if the strip is fired
  return (cathode == 0) ? isBPStripFired(istrip, line) : isNBPStripFired(istrip);
}

ColumnData& operator|=(ColumnData& col1, const ColumnData& col2)
{
  /// Merge operator for ColumnData
  if (col1.deId != col2.deId || col1.columnId != col2.columnId) {
    throw std::runtime_error("Cannot merge ColumnData");
  }
  for (size_t ipat = 0; ipat < col1.patterns.size(); ++ipat) {
    col1.patterns[ipat] |= col2.patterns[ipat];
  }
  return col1;
}

ColumnData operator|(const ColumnData& col1, const ColumnData& col2)
{
  /// Merge operator for ColumnData
  ColumnData out = col1;
  out |= col2;
  return out;
}

std::ostream& operator<<(std::ostream& os, const ColumnData& col)
{
  /// Output streamer for ColumnData
  os << "deId: " << static_cast<int>(col.deId) << "  col: " << static_cast<int>(col.columnId);
  os << "  NBP: " << std::bitset<16>(col.getNonBendPattern());
  os << "  BP: ";
  for (int iline = 0; iline < 4; ++iline) {
    os << " " << std::bitset<16>(col.getBendPattern(iline));
  }
  return os;
}

} // namespace mid
} // namespace o2
