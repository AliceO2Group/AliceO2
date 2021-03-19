// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/Raw/src/ColumnDataToLocalBoard.cxx
/// \brief  Converter from ColumnData to raw local boards
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   20 April 2020

#include "MIDRaw/ColumnDataToLocalBoard.h"

#include "MIDBase/DetectorParameters.h"
#include "MIDRaw/CrateParameters.h"

namespace o2
{
namespace mid
{

bool ColumnDataToLocalBoard::keepBoard(const ROBoard& loc) const
{
  for (int ich = 0; ich < 4; ++ich) {
    if (loc.patternsBP[ich] && loc.patternsNBP[ich]) {
      return true;
    }
  }
  return false;
}

void ColumnDataToLocalBoard::process(gsl::span<const ColumnData> data)
{
  /// Converts incoming data to FEE format
  mLocalBoardsMap.clear();
  mGBTMap.clear();

  // First fill the map with the active local boards.
  // Each local board gets a unique id.
  for (auto& col : data) {
    for (int iline = mMapping.getFirstBoardBP(col.columnId, col.deId); iline <= mMapping.getLastBoardBP(col.columnId, col.deId); ++iline) {
      if (col.getBendPattern(iline) || col.getNonBendPattern()) {
        auto uniqueLocId = mCrateMapper.deLocalBoardToRO(col.deId, col.columnId, iline);
        auto& roData = mLocalBoardsMap[uniqueLocId];
        roData.statusWord = raw::sSTARTBIT | raw::sCARDTYPE;
        roData.boardId = uniqueLocId;
        int ich = detparams::getChamber(col.deId);
        roData.firedChambers |= (1 << ich);
        roData.patternsBP[ich] = col.getBendPattern(iline);
        roData.patternsNBP[ich] = col.getNonBendPattern();
      }
    }
  }

  // Then group the boards belonging to the same GBT link
  for (auto& item : mLocalBoardsMap) {
    if (mDebugMode || keepBoard(item.second)) {
      auto crateId = raw::getCrateId(item.first);
      auto feeId = crateparams::makeGBTUniqueId(crateId, crateparams::getGBTIdFromBoardInCrate(raw::getLocId(item.second.boardId)));
      mGBTMap[feeId].emplace_back(item.second);
    }
  }
}

} // namespace mid
} // namespace o2
