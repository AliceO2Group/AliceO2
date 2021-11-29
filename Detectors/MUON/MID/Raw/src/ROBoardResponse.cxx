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

/// \file   MID/Raw/src/ROBoardResponse.cxx
/// \brief  Local board response
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   12 November 2021

#include "MIDRaw/ROBoardResponse.h"

#include <unordered_map>
#include "MIDRaw/CrateParameters.h"

namespace o2
{
namespace mid
{

ROBoardResponse::ROBoardResponse(const std::vector<ROBoardConfig>& configurations) : mConfigHandler(configurations)
{
}

bool ROBoardResponse::isZeroSuppressed(const ROBoard& loc) const
{
  auto cfg = mConfigHandler.getConfig(loc.boardId);
  if (cfg.configWord & crateconfig::sXorY) {
    return false;
  }
  for (int ich = 0; ich < 4; ++ich) {
    if (loc.patternsBP[ich] && loc.patternsNBP[ich]) {
      return false;
    }
  }
  return true;
}

bool ROBoardResponse::applyZeroSuppression(std::vector<ROBoard>& locs) const
{
  std::vector<ROBoard> zsLocs;
  bool isSuppressed = false;
  for (auto& loc : locs) {
    if (!isZeroSuppressed(loc)) {
      zsLocs.emplace_back(loc);
      isSuppressed = true;
    }
  }
  locs.swap(zsLocs);
  return isSuppressed;
}

std::vector<ROBoard> ROBoardResponse::getTriggerResponse(uint8_t triggerWord) const
{
  std::vector<ROBoard> locBoards;
  auto& cfgMap = mConfigHandler.getConfigMap();
  for (auto& item : cfgMap) {
    locBoards.push_back({raw::sSTARTBIT | raw::sCARDTYPE, triggerWord, item.first, 0});
    if (triggerWord & (raw::sSOX | raw::sEOX)) {
      /// Write masks
      if (item.second.configWord & crateconfig::sMonmoff) {
        locBoards.back().statusWord |= raw::sMASKED;
        for (int ich = 0; ich < 4; ++ich) {
          locBoards.back().patternsBP[ich] = item.second.masksBP[ich];
          locBoards.back().patternsNBP[ich] = item.second.masksNBP[ich];
        }
      }
    }
  }
  auto regBoards = getRegionalResponse(locBoards);
  locBoards.insert(locBoards.begin(), regBoards.begin(), regBoards.end());
  return locBoards;
}

std::vector<ROBoard> ROBoardResponse::getRegionalResponse(const std::vector<ROBoard>& locs) const
{
  std::unordered_map<uint8_t, uint8_t> firedLocs;

  uint8_t triggerWord = 0;
  for (auto& loc : locs) {
    auto locId = raw::getLocId(loc.boardId);
    auto regId = 8 * crateparams::getGBTIdFromBoardInCrate(locId) + (locId % 8) / 4;
    auto uniqueRegId = raw::makeUniqueLocID(raw::getCrateId(loc.boardId), regId);
    int locPos = locId % 4;
    firedLocs[uniqueRegId] |= (1 << locPos);
    triggerWord = loc.triggerWord;
  }

  std::vector<ROBoard> regBoards;
  for (auto& item : firedLocs) {
    regBoards.push_back({raw::sSTARTBIT, triggerWord, item.first, item.second});
  }

  return regBoards;
}

} // namespace mid
} // namespace o2
