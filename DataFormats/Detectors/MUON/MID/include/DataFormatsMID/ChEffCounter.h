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

/// \file   DataFormatsMID/ChEffCounter.h
/// \brief  Chamber efficiency counters
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   12 September 2022

#ifndef O2_MID_CHEFFCOUNTER_H
#define O2_MID_CHEFFCOUNTER_H

#include <cstdint>
#include <array>
#include <cstddef>

namespace o2
{
namespace mid
{
enum class EffCountType {
  BendPlane,    ///< Bending plane counters
  NonBendPlane, ///< Non-bending plane counters
  BothPlanes,   ///< Both plane counters
  AllTracks     ///< All tracks counters
};

/// Column data structure for MID
struct ChEffCounter {
  uint8_t deId = 0;               ///< Index of the detection element
  uint8_t columnId = 0;           ///< Column in DE
  uint8_t lineId = 0;             ///< Line in column
  std::array<uint32_t, 4> counts; ///< Counts

  /// @brief Returns the efficiency counter
  /// @param type Efficiency counter type
  /// @return Efficiency counter
  uint32_t getCounts(EffCountType type) const { return counts[static_cast<size_t>(type)]; }
};

} // namespace mid
} // namespace o2

#endif /* O2_MID_CHEFFCOUNTER_H */
