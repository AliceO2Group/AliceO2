// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MIDRaw/LocalBoardRO.h
/// \brief  Structure to store the FEE local board information
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   19 November 2019
#ifndef O2_MID_LocalBoardRO_H
#define O2_MID_LocalBoardRO_H

#include <cstdint>
#include <array>

namespace o2
{
namespace mid
{
struct LocalBoardRO {
  uint8_t statusWord{0};                 /// Status word
  uint8_t eventWord{0};                  /// Event word
  uint8_t boardId{0};                    /// Board ID in crate
  uint8_t firedChambers{0};              /// Fired chambers
  std::array<uint16_t, 4> patternsBP{};  /// Bending plane pattern
  std::array<uint16_t, 4> patternsNBP{}; /// Non-bending plane pattern
};

std::ostream& operator<<(std::ostream& os, const LocalBoardRO& loc);
} // namespace mid
} // namespace o2

#endif /* O2_MID_LocalBoardRO_H */
