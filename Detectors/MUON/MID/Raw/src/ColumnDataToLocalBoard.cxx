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

void ColumnDataToLocalBoard::process(gsl::span<const ColumnData> data, bool allowEmpty)
{
  /// Converts incoming data to FEE format
  mLocalBoardsMap.clear();

  // First fill the map with the active local boards.
  // Each local board gets a unique id.
  for (auto& col : data) {
    for (int iline = mMapping.getFirstBoardBP(col.columnId, col.deId), lastLine = mMapping.getLastBoardBP(col.columnId, col.deId); iline <= lastLine; ++iline) {
      auto bp = col.getBendPattern(iline);
      auto nbp = col.getNonBendPattern();
      // Finding the loc ID is time consuming.
      // So let us first check if we need to fill this board.
      if (allowEmpty || bp || nbp) {
        auto uniqueLocId = mCrateMapper.deLocalBoardToRO(col.deId, col.columnId, iline);
        if (nbp && !mCrateMapper.hasDirectInputY(uniqueLocId)) {
          // If this local board has no non-bending input attached, we set it to 0
          // But if the bending-plane was 0 and we do not allow empty boards, we can stop here.
          if (bp == 0 && !allowEmpty) {
            continue;
          }
          nbp = 0;
        }
        auto& roData = mLocalBoardsMap[uniqueLocId];
        roData.statusWord = raw::sSTARTBIT | raw::sCARDTYPE;
        roData.boardId = uniqueLocId;
        int ich = detparams::getChamber(col.deId);
        roData.firedChambers |= (1 << ich);
        roData.patternsBP[ich] = bp;
        roData.patternsNBP[ich] = nbp;
      }
    }
  }
}

std::vector<ROBoard> ColumnDataToLocalBoard::getData() const
{
  std::vector<ROBoard> roBoards;
  for (auto& item : mLocalBoardsMap) {
    roBoards.emplace_back(item.second);
  }
  return roBoards;
}

} // namespace mid
} // namespace o2
