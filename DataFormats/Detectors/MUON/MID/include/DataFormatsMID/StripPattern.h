// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   DataFormatsMID/StripPattern.h
/// \brief  Strip pattern (aka digits)
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   19 February 2018

#ifndef O2_MID_STRIPPATTERN_H
#define O2_MID_STRIPPATTERN_H

#include <boost/serialization/access.hpp>
#include <cstdint>
#include <array>

namespace o2
{
namespace mid
{
/// Strip pattern structure fot MID

struct StripPattern {
  /// Set non-bending plane pattern
  void setNonBendPattern(uint16_t pattern) { patterns[4] = pattern; }
  /// Get non-bending plane pattern
  uint16_t getNonBendPattern() { return patterns[4]; }
  /// Set bending plane pattern
  void setBendPattern(uint16_t pattern, uint16_t line) { patterns[line] = pattern; }
  /// Get bending plane pattern
  uint16_t getBendPattern(uint16_t line) { return patterns[line]; }

  /// Check if strip is fired
  bool isStripFired(uint16_t istrip, uint16_t line) { return patterns[line] & (1 << istrip); }
  /// Check if strip is fired in the non-bending plane
  bool isNBPStripFired(uint16_t istrip) { return isStripFired(istrip, 4); }
  /// Check if strip is fired in the bending plane
  bool isBPStripFired(uint16_t istrip, uint16_t line) { return isStripFired(istrip, line); }

  std::array<uint16_t, 5> patterns; ///< patterns
};

/// Column data structure for MID
struct ColumnData {
  uint8_t deId;          ///< Index of the detection element
  uint8_t columnId;      ///< Column in DE
  StripPattern patterns; ///< Strip patterns

  friend class boost::serialization::access;

  /// Serializes the struct
  template <class Archive>
  void serialize(Archive& ar, const unsigned int version)
  {
    ar& deId;
    ar& columnId;
    ar& patterns.patterns;
  }
};
} // namespace mid
} // namespace o2

#endif /* O2_MID_STRIPPATTERN_H */
