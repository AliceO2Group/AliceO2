// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/Raw/src/LocalBoardRO.cxx
/// \brief  Structure to store the FEE local board information
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   19 November 2019

#include "MIDRaw/LocalBoardRO.h"

#include <iostream>
#include "fmt/format.h"
#include "MIDRaw/CrateParameters.h"

namespace o2
{
namespace mid
{
std::ostream& operator<<(std::ostream& os, const LocalBoardRO& loc)
{
  /// Stream operator for LocalBoardRO
  os << fmt::format("Crate ID: {:2d}  Loc ID: {:2d}  status: 0x{:2x}  event: 0x{:2x}  firedChambers: 0x{:1x}", static_cast<int>(crateparams::getCrateId(loc.boardId)), static_cast<int>(crateparams::getLocId(loc.boardId)), static_cast<int>(loc.statusWord), static_cast<int>(loc.eventWord), static_cast<int>(loc.firedChambers));
  for (int ich = 0; ich < 4; ++ich) {
    os << fmt::format("  X: 0x{:4x} Y: 0x{:4x}", loc.patternsBP[ich], loc.patternsNBP[ich]);
  }
  return os;
}

} // namespace mid
} // namespace o2