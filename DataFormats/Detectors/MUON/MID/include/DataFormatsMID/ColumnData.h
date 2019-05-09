// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   DataFormatsMID/ColumnData.h
/// \brief  Strip pattern (aka digits)
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   19 February 2018

#ifndef O2_MID_COLUMNDATA_H
#define O2_MID_COLUMNDATA_H

#include <cstdint>
#include <iostream>
#include <array>

namespace o2
{
namespace mid
{
/// Column data structure for MID
struct ColumnData {
  uint8_t deId = 0;                 ///< Index of the detection element
  uint8_t columnId = 0;             ///< Column in DE
  std::array<uint16_t, 5> patterns; ///< Strip patterns

  /// Sets the bending plane pattern
  void setBendPattern(uint16_t pattern, int line) { patterns[line] = pattern; }
  /// Gets the bending plane pattern
  uint16_t getBendPattern(int line) const { return patterns[line]; }

  /// Sets the non-bending plane pattern
  void setNonBendPattern(uint16_t pattern) { patterns[4] = pattern; }
  /// Gets the non-bending plane pattern
  uint16_t getNonBendPattern() const { return patterns[4]; }

  void setPattern(uint16_t pattern, int cathode, int line);
  uint16_t getPattern(int cathode, int line) const;

  void addStrip(int strip, int cathode, int line);

  /// Checks if strip is fired in the non-bending plane
  bool isNBPStripFired(int istrip) const { return patterns[4] & (1 << istrip); }
  /// Checks if strip is fired in the bending plane
  bool isBPStripFired(int istrip, int line) const { return patterns[line] & (1 << istrip); }

  bool isStripFired(int istrip, int cathode, int line) const;
};

ColumnData operator|(const ColumnData& col1, const ColumnData& col2);
ColumnData& operator|=(ColumnData& col1, const ColumnData& col2);
std::ostream& operator<<(std::ostream& os, const ColumnData& col);

} // namespace mid
} // namespace o2

#endif /* O2_MID_COLUMNDATA_H */
