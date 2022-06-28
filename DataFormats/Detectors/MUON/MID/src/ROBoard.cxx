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

/// \file   MID/src/ROBoard.cxx
/// \brief  Structure to store the readout board information
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   19 November 2019

#include "DataFormatsMID/ROBoard.h"

#include <iostream>
#include "fmt/format.h"

namespace o2
{
namespace mid
{
std::ostream& operator<<(std::ostream& os, const ROBoard& board)
{
  os << fmt::format("Crate ID: {:2d}  {} ID: {:2d}  status: 0x{:2x}  trig: 0x{:2x}  fired: 0x{:1x}", static_cast<int>(raw::getCrateId(board.boardId)), (raw::isLoc(board.statusWord) ? "Loc" : "Reg"), static_cast<int>(raw::getLocId(board.boardId)), static_cast<int>(board.statusWord), static_cast<int>(board.triggerWord), static_cast<int>(board.firedChambers));
  for (int ich = 0; ich < 4; ++ich) {
    os << fmt::format("  X: 0x{:4x} Y: 0x{:4x}", board.patternsBP[ich], board.patternsNBP[ich]);
  }
  return os;
}

} // namespace mid
} // namespace o2