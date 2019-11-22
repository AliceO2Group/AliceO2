// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/Raw/src/CrateMapper.cxx
/// \brief  FEE ID to board ID converter
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   15 November 2019

#include "MIDRaw/CrateMapper.h"

#include <sstream>
#include <exception>
#include "MIDBase/DetectorParameters.h"
#include "MIDRaw/CrateParameters.h"

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
  mROToDEMap.emplace(crateparams::makeUniqueLocID(0, 0), deBoardId(0, 0, 0));
  mROToDEMap.emplace(crateparams::makeUniqueLocID(0, 1), deBoardId(1, 0, 0));
  mROToDEMap.emplace(crateparams::makeUniqueLocID(0, 2), deBoardId(1, 0, 1));
  mROToDEMap.emplace(crateparams::makeUniqueLocID(0, 3), deBoardId(2, 0, 0));
  mROToDEMap.emplace(crateparams::makeUniqueLocID(0, 4), deBoardId(2, 0, 1));
  mROToDEMap.emplace(crateparams::makeUniqueLocID(0, 5), deBoardId(3, 0, 0));
  mROToDEMap.emplace(crateparams::makeUniqueLocID(0, 6), deBoardId(3, 0, 1));
  mROToDEMap.emplace(crateparams::makeUniqueLocID(0, 7), deBoardId(3, 0, 2));
  // link 1
  mROToDEMap.emplace(crateparams::makeUniqueLocID(0, 8), deBoardId(5, 0, 1));
  mROToDEMap.emplace(crateparams::makeUniqueLocID(0, 9), deBoardId(5, 0, 2));
  mROToDEMap.emplace(crateparams::makeUniqueLocID(0, 10), deBoardId(5, 0, 3));
  mROToDEMap.emplace(crateparams::makeUniqueLocID(0, 11), deBoardId(6, 0, 0));
  mROToDEMap.emplace(crateparams::makeUniqueLocID(0, 12), deBoardId(6, 0, 1));
  mROToDEMap.emplace(crateparams::makeUniqueLocID(0, 13), deBoardId(7, 0, 0));
  mROToDEMap.emplace(crateparams::makeUniqueLocID(0, 14), deBoardId(7, 0, 1));
  mROToDEMap.emplace(crateparams::makeUniqueLocID(0, 15), deBoardId(8, 0, 0));

  // Crate 1, 3
  for (int icrate = 0; icrate < 2; ++icrate) {
    uint8_t crateId = (icrate == 0) ? 1 : 3;
    uint8_t columnId = (icrate == 0) ? 1 : 2;
    // link 0
    mROToDEMap.emplace(crateparams::makeUniqueLocID(crateId, 0), deBoardId(0, columnId, 0));
    mROToDEMap.emplace(crateparams::makeUniqueLocID(crateId, 1), deBoardId(1, columnId, 0));
    mROToDEMap.emplace(crateparams::makeUniqueLocID(crateId, 2), deBoardId(1, columnId, 1));
    mROToDEMap.emplace(crateparams::makeUniqueLocID(crateId, 3), deBoardId(2, columnId, 0));
    mROToDEMap.emplace(crateparams::makeUniqueLocID(crateId, 4), deBoardId(2, columnId, 1));
    mROToDEMap.emplace(crateparams::makeUniqueLocID(crateId, 5), deBoardId(3, columnId, 0));
    mROToDEMap.emplace(crateparams::makeUniqueLocID(crateId, 6), deBoardId(3, columnId, 1));
    mROToDEMap.emplace(crateparams::makeUniqueLocID(crateId, 7), deBoardId(3, columnId, 2));
    // link 1
    mROToDEMap.emplace(crateparams::makeUniqueLocID(crateId, 8), deBoardId(3, columnId, 3));
    mROToDEMap.emplace(crateparams::makeUniqueLocID(crateId, 9), deBoardId(4, columnId, 0));
    mROToDEMap.emplace(crateparams::makeUniqueLocID(crateId, 10), deBoardId(4, columnId, 1));
    mROToDEMap.emplace(crateparams::makeUniqueLocID(crateId, 11), deBoardId(4, columnId, 2));
    mROToDEMap.emplace(crateparams::makeUniqueLocID(crateId, 12), deBoardId(4, columnId, 3));
    mROToDEMap.emplace(crateparams::makeUniqueLocID(crateId, 13), deBoardId(5, columnId, 0));
    mROToDEMap.emplace(crateparams::makeUniqueLocID(crateId, 14), deBoardId(5, columnId, 1));
  }

  // Crate 2
  // link 0
  mROToDEMap.emplace(crateparams::makeUniqueLocID(2, 0), deBoardId(5, 1, 2));
  mROToDEMap.emplace(crateparams::makeUniqueLocID(2, 1), deBoardId(5, 1, 3));
  mROToDEMap.emplace(crateparams::makeUniqueLocID(2, 2), deBoardId(6, 1, 0));
  mROToDEMap.emplace(crateparams::makeUniqueLocID(2, 3), deBoardId(6, 1, 1));
  mROToDEMap.emplace(crateparams::makeUniqueLocID(2, 4), deBoardId(7, 1, 0));
  mROToDEMap.emplace(crateparams::makeUniqueLocID(2, 5), deBoardId(7, 1, 1));
  mROToDEMap.emplace(crateparams::makeUniqueLocID(2, 6), deBoardId(8, 1, 0));
  // link 1
  mROToDEMap.emplace(crateparams::makeUniqueLocID(2, 8), deBoardId(5, 2, 2));
  mROToDEMap.emplace(crateparams::makeUniqueLocID(2, 9), deBoardId(5, 2, 3));
  mROToDEMap.emplace(crateparams::makeUniqueLocID(2, 10), deBoardId(6, 2, 0));
  mROToDEMap.emplace(crateparams::makeUniqueLocID(2, 11), deBoardId(6, 2, 1));
  mROToDEMap.emplace(crateparams::makeUniqueLocID(2, 12), deBoardId(7, 2, 0));
  mROToDEMap.emplace(crateparams::makeUniqueLocID(2, 13), deBoardId(7, 2, 1));
  mROToDEMap.emplace(crateparams::makeUniqueLocID(2, 14), deBoardId(8, 2, 0));

  // Crate 4, 5, 6
  for (int icrate = 0; icrate < 3; ++icrate) {
    uint16_t crateId = icrate + 4;
    uint8_t columnId = icrate + 3;
    // link 0
    mROToDEMap.emplace(crateparams::makeUniqueLocID(crateId, 0), deBoardId(0, columnId, 0));
    mROToDEMap.emplace(crateparams::makeUniqueLocID(crateId, 1), deBoardId(1, columnId, 0));
    mROToDEMap.emplace(crateparams::makeUniqueLocID(crateId, 2), deBoardId(1, columnId, 1));
    mROToDEMap.emplace(crateparams::makeUniqueLocID(crateId, 3), deBoardId(2, columnId, 0));
    mROToDEMap.emplace(crateparams::makeUniqueLocID(crateId, 4), deBoardId(2, columnId, 1));
    mROToDEMap.emplace(crateparams::makeUniqueLocID(crateId, 5), deBoardId(3, columnId, 0));
    mROToDEMap.emplace(crateparams::makeUniqueLocID(crateId, 6), deBoardId(3, columnId, 1));
    mROToDEMap.emplace(crateparams::makeUniqueLocID(crateId, 7), deBoardId(4, columnId, 0));
    // link 1
    mROToDEMap.emplace(crateparams::makeUniqueLocID(crateId, 8), deBoardId(4, columnId, 1));
    mROToDEMap.emplace(crateparams::makeUniqueLocID(crateId, 9), deBoardId(5, columnId, 0));
    mROToDEMap.emplace(crateparams::makeUniqueLocID(crateId, 10), deBoardId(5, columnId, 1));
    mROToDEMap.emplace(crateparams::makeUniqueLocID(crateId, 11), deBoardId(6, columnId, 0));
    mROToDEMap.emplace(crateparams::makeUniqueLocID(crateId, 12), deBoardId(6, columnId, 1));
    mROToDEMap.emplace(crateparams::makeUniqueLocID(crateId, 13), deBoardId(7, columnId, 0));
    mROToDEMap.emplace(crateparams::makeUniqueLocID(crateId, 14), deBoardId(7, columnId, 1));
    mROToDEMap.emplace(crateparams::makeUniqueLocID(crateId, 15), deBoardId(8, columnId, 0));
  }

  // Crate 7
  uint8_t rpcLineId = 0;
  for (uint8_t iboard = 0; iboard < 9; ++iboard) {
    mROToDEMap.emplace(crateparams::makeUniqueLocID(7, iboard), deBoardId(rpcLineId++, 6, 0));
  }

  /// Build the inverse map
  for (auto& item : mROToDEMap) {
    mDEToROMap.emplace(item.second, item.first);
  }
}

uint16_t CrateMapper::deLocalBoardToRO(uint8_t deId, uint8_t columnId, uint8_t lineId) const
{
  /// Converts the local board ID in  in MT11 right to the local board ID in FEE
  auto item = mDEToROMap.find(deBoardId(detparams::getRPCLine(deId), columnId, lineId));
  if (item == mDEToROMap.end()) {
    std::stringstream ss;
    ss << "Non-existant deId: " << static_cast<int>(deId) << "  columnId: " << static_cast<int>(columnId) << "  lineId: " << static_cast<int>(lineId);
    throw std::runtime_error(ss.str());
  }
  return detparams::isRightSide(deId) ? item->second : item->second + (crateparams::sNCratesPerSide << 4);
}

uint16_t CrateMapper::roLocalBoardToDE(uint8_t crateId, uint8_t boardId) const
{
  /// Converts the local board ID in FEE to the local board ID in MT11 right
  auto item = mROToDEMap.find(crateparams::makeUniqueLocID(crateId % crateparams::sNCratesPerSide, boardId));
  if (item == mROToDEMap.end()) {
    std::stringstream ss;
    ss << "Non-existant crateId: " << static_cast<int>(crateId) << "  boardId: " << static_cast<int>(boardId);
    throw std::runtime_error(ss.str());
  }
  return item->second;
}

} // namespace mid
} // namespace o2
