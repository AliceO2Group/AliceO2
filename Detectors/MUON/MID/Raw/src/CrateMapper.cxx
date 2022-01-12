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

/// \file   MID/Raw/src/CrateMapper.cxx
/// \brief  FEE ID to board ID converter
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   15 November 2019

#include "MIDRaw/CrateMapper.h"

#include <algorithm>
#include <array>
#include <exception>
#include "fmt/format.h"
#include "DataFormatsMID/ROBoard.h"
#include "MIDBase/DetectorParameters.h"
#include "MIDRaw/CrateParameters.h"
#include "MIDRaw/GBTMapper.h"

namespace o2
{
namespace mid
{
CrateMapper::CrateMapper()
{
  /// Ctor
  init();
}

void CrateMapper::init()
{
  /// Initalizes the inner mapping
  // Crate 0
  // link 0
  mROToDEMap.emplace(raw::makeUniqueLocID(0, 0), deBoardId(0, 0, 0));
  mROToDEMap.emplace(raw::makeUniqueLocID(0, 1), deBoardId(1, 0, 0));
  mROToDEMap.emplace(raw::makeUniqueLocID(0, 2), deBoardId(1, 0, 1));
  mROToDEMap.emplace(raw::makeUniqueLocID(0, 3), deBoardId(2, 0, 0));
  mROToDEMap.emplace(raw::makeUniqueLocID(0, 4), deBoardId(2, 0, 1));
  mROToDEMap.emplace(raw::makeUniqueLocID(0, 5), deBoardId(3, 0, 0));
  mROToDEMap.emplace(raw::makeUniqueLocID(0, 6), deBoardId(3, 0, 1));
  mROToDEMap.emplace(raw::makeUniqueLocID(0, 7), deBoardId(3, 0, 2));
  // link 1
  mROToDEMap.emplace(raw::makeUniqueLocID(0, 8), deBoardId(5, 0, 1));
  mROToDEMap.emplace(raw::makeUniqueLocID(0, 9), deBoardId(5, 0, 2));
  mROToDEMap.emplace(raw::makeUniqueLocID(0, 10), deBoardId(5, 0, 3));
  mROToDEMap.emplace(raw::makeUniqueLocID(0, 11), deBoardId(6, 0, 0));
  mROToDEMap.emplace(raw::makeUniqueLocID(0, 12), deBoardId(6, 0, 1));
  mROToDEMap.emplace(raw::makeUniqueLocID(0, 13), deBoardId(7, 0, 0));
  mROToDEMap.emplace(raw::makeUniqueLocID(0, 14), deBoardId(7, 0, 1));
  mROToDEMap.emplace(raw::makeUniqueLocID(0, 15), deBoardId(8, 0, 0));

  // Crate 1, 3
  for (int icrate = 0; icrate < 2; ++icrate) {
    uint8_t crateId = (icrate == 0) ? 1 : 3;
    uint8_t columnId = (icrate == 0) ? 1 : 2;
    // link 0
    mROToDEMap.emplace(raw::makeUniqueLocID(crateId, 0), deBoardId(0, columnId, 0));
    mROToDEMap.emplace(raw::makeUniqueLocID(crateId, 1), deBoardId(1, columnId, 0));
    mROToDEMap.emplace(raw::makeUniqueLocID(crateId, 2), deBoardId(1, columnId, 1));
    mROToDEMap.emplace(raw::makeUniqueLocID(crateId, 3), deBoardId(2, columnId, 0));
    mROToDEMap.emplace(raw::makeUniqueLocID(crateId, 4), deBoardId(2, columnId, 1));
    mROToDEMap.emplace(raw::makeUniqueLocID(crateId, 5), deBoardId(3, columnId, 0));
    mROToDEMap.emplace(raw::makeUniqueLocID(crateId, 6), deBoardId(3, columnId, 1));
    mROToDEMap.emplace(raw::makeUniqueLocID(crateId, 7), deBoardId(3, columnId, 2));

    // link 1
    mROToDEMap.emplace(raw::makeUniqueLocID(crateId, 8), deBoardId(3, columnId, 3));
    mROToDEMap.emplace(raw::makeUniqueLocID(crateId, 9), deBoardId(4, columnId, 0));
    mROToDEMap.emplace(raw::makeUniqueLocID(crateId, 10), deBoardId(4, columnId, 1));
    mROToDEMap.emplace(raw::makeUniqueLocID(crateId, 11), deBoardId(4, columnId, 2));
    mROToDEMap.emplace(raw::makeUniqueLocID(crateId, 12), deBoardId(4, columnId, 3));
    mROToDEMap.emplace(raw::makeUniqueLocID(crateId, 13), deBoardId(5, columnId, 0));
    mROToDEMap.emplace(raw::makeUniqueLocID(crateId, 14), deBoardId(5, columnId, 1));
  }

  // Crate 2
  // link 0
  mROToDEMap.emplace(raw::makeUniqueLocID(2, 0), deBoardId(5, 1, 2));
  mROToDEMap.emplace(raw::makeUniqueLocID(2, 1), deBoardId(5, 1, 3));
  mROToDEMap.emplace(raw::makeUniqueLocID(2, 2), deBoardId(6, 1, 0));
  mROToDEMap.emplace(raw::makeUniqueLocID(2, 3), deBoardId(6, 1, 1));
  mROToDEMap.emplace(raw::makeUniqueLocID(2, 4), deBoardId(7, 1, 0));
  mROToDEMap.emplace(raw::makeUniqueLocID(2, 5), deBoardId(7, 1, 1));
  mROToDEMap.emplace(raw::makeUniqueLocID(2, 6), deBoardId(8, 1, 0));
  // link 1
  mROToDEMap.emplace(raw::makeUniqueLocID(2, 8), deBoardId(5, 2, 2));
  mROToDEMap.emplace(raw::makeUniqueLocID(2, 9), deBoardId(5, 2, 3));
  mROToDEMap.emplace(raw::makeUniqueLocID(2, 10), deBoardId(6, 2, 0));
  mROToDEMap.emplace(raw::makeUniqueLocID(2, 11), deBoardId(6, 2, 1));
  mROToDEMap.emplace(raw::makeUniqueLocID(2, 12), deBoardId(7, 2, 0));
  mROToDEMap.emplace(raw::makeUniqueLocID(2, 13), deBoardId(7, 2, 1));
  mROToDEMap.emplace(raw::makeUniqueLocID(2, 14), deBoardId(8, 2, 0));

  // Crate 4, 5, 6
  for (int icrate = 0; icrate < 3; ++icrate) {
    uint16_t crateId = icrate + 4;
    uint8_t columnId = icrate + 3;
    // link 0
    mROToDEMap.emplace(raw::makeUniqueLocID(crateId, 0), deBoardId(0, columnId, 0));
    mROToDEMap.emplace(raw::makeUniqueLocID(crateId, 1), deBoardId(1, columnId, 0));
    mROToDEMap.emplace(raw::makeUniqueLocID(crateId, 2), deBoardId(1, columnId, 1));
    mROToDEMap.emplace(raw::makeUniqueLocID(crateId, 3), deBoardId(2, columnId, 0));
    mROToDEMap.emplace(raw::makeUniqueLocID(crateId, 4), deBoardId(2, columnId, 1));
    mROToDEMap.emplace(raw::makeUniqueLocID(crateId, 5), deBoardId(3, columnId, 0));
    mROToDEMap.emplace(raw::makeUniqueLocID(crateId, 6), deBoardId(3, columnId, 1));
    mROToDEMap.emplace(raw::makeUniqueLocID(crateId, 7), deBoardId(4, columnId, 0));
    // link 1
    mROToDEMap.emplace(raw::makeUniqueLocID(crateId, 8), deBoardId(4, columnId, 1));
    mROToDEMap.emplace(raw::makeUniqueLocID(crateId, 9), deBoardId(5, columnId, 0));
    mROToDEMap.emplace(raw::makeUniqueLocID(crateId, 10), deBoardId(5, columnId, 1));
    mROToDEMap.emplace(raw::makeUniqueLocID(crateId, 11), deBoardId(6, columnId, 0));
    mROToDEMap.emplace(raw::makeUniqueLocID(crateId, 12), deBoardId(6, columnId, 1));
    mROToDEMap.emplace(raw::makeUniqueLocID(crateId, 13), deBoardId(7, columnId, 0));
    mROToDEMap.emplace(raw::makeUniqueLocID(crateId, 14), deBoardId(7, columnId, 1));
    mROToDEMap.emplace(raw::makeUniqueLocID(crateId, 15), deBoardId(8, columnId, 0));
  }

  // Crate 7
  uint8_t rpcLineId = 0;
  for (uint8_t iboard = 0; iboard < 9; ++iboard) {
    mROToDEMap.emplace(raw::makeUniqueLocID(7, iboard), deBoardId(rpcLineId++, 6, 0));
  }

  /// Build the inverse map
  for (auto& item : mROToDEMap) {
    mDEToROMap.emplace(item.second, item.first);
  }

  /// Build the map of the local boards with direct Y input from FEE
  for (auto& item : mROToDEMap) {
    bool hasDirectInputY = false;
    auto lineId = getLineId(item.second);
    if (lineId == 0) { // First loc in the RPC
      hasDirectInputY = true;
    } else {
      auto crateId = raw::getCrateId(item.first);
      auto locId = raw::getLocId(item.first);
      if ((crateId == 0 && locId == 8) ||
          (crateId == 2 && (locId == 0 || locId == 8))) {
        hasDirectInputY = true;
      }
    }
    if (hasDirectInputY) {
      mLocIdsWithDirectInputY.emplace(item.first);
    }
  }
}

uint8_t CrateMapper::deLocalBoardToRO(uint8_t deId, uint8_t columnId, uint8_t lineId) const
{
  /// Converts the local board ID in  in MT11 right to the local board ID in FEE
  auto item = mDEToROMap.find(deBoardId(detparams::getRPCLine(deId), columnId, lineId));
  if (item == mDEToROMap.end()) {
    throw std::runtime_error(fmt::format("Non-existent deId: {:d}  columnId: {:d}  lineId: {:d}", deId, columnId, lineId));
  }
  return detparams::isRightSide(deId) ? item->second : item->second + (crateparams::sNCratesPerSide << 4);
}

uint16_t CrateMapper::roLocalBoardToDE(uint8_t uniqueLocId) const
{
  /// Converts the local board ID in FEE to the local board ID in MT11 right
  auto item = mROToDEMap.find(getROBoardIdRight(uniqueLocId));
  if (item == mROToDEMap.end()) {
    throw std::runtime_error(fmt::format("Non-existent crateId: {:d}  boardId: {:d}", raw::getCrateId(uniqueLocId), raw::getLocId(uniqueLocId)));
  }
  return item->second;
}

std::vector<uint8_t> CrateMapper::getROBoardIds(uint16_t gbtUniqueId) const
{
  std::vector<uint8_t> roBoardIds;
  std::array<uint8_t, 2> offsets{0, 0x80};
  for (auto& item : mROToDEMap) {
    // For simplicity, the map only contains one side
    // So we loop on the two sides
    for (auto& off : offsets) {
      auto locId = item.first + off;
      if (gbtUniqueId != 0xFFFF) {
        if (!gbtmapper::isBoardInGBT(locId, gbtUniqueId)) {
          continue;
        }
      }
      roBoardIds.emplace_back(locId);
    }
  }
  std::sort(roBoardIds.begin(), roBoardIds.end());
  return roBoardIds;
}

} // namespace mid
} // namespace o2
