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

/// \file   MID/Filtering/src/FetToDead.cxx
/// \brief  Class to convert the FEE test event into dead channels
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   10 May 2021

#include "MIDFiltering/FetToDead.h"

#include "MIDFiltering/MaskMaker.h"

namespace o2
{
namespace mid
{
FetToDead::FetToDead()
{
  /// Default ctr.
  setMasks(makeDefaultMasks());
}

void FetToDead::setMasks(std::vector<ColumnData> masks)
{
  /// Sets the default masks reflecting the active channels from the mapping
  mMasksHandler.setFromChannelMasks(masks);
  mInvertedActive.clear();
  std::vector<ColumnData> inverted;
  for (auto& col : masks) {
    invertPattern(col, inverted);
  }
  for (auto& col : inverted) {
    mInvertedActive[getColumnDataUniqueId(col.deId, col.columnId)] = col;
  }
}

std::vector<ColumnData> FetToDead::process(gsl::span<const ColumnData> fetData)
{
  /// Converts the FET result to a list of dead channels
  std::vector<ColumnData> deadChannels;
  std::unordered_map<uint16_t, bool> dataIds;
  // First we loop on FET output and returns the empty strips
  for (auto& col : fetData) {
    invertPattern(col, deadChannels);
    dataIds[getColumnDataUniqueId(col.deId, col.columnId)] = true;
  }

  // Then we loop on the active channels and check if they where among the FET data.
  // If they are not there, it means that non of the channels answered.
  // So the full board was dead.
  for (auto& item : mInvertedActive) {
    auto found = dataIds.find(item.first);
    if (found == dataIds.end()) {
      deadChannels.emplace_back(item.second);
    }
  }
  return deadChannels;
}

bool FetToDead::invertPattern(const ColumnData& col, std::vector<ColumnData>& invertedData)
{
  /// Inverts the pattern and add it to the output data
  ColumnData invertedCol{col.deId, col.columnId};
  invertedCol.setNonBendPattern(~col.getNonBendPattern());
  for (int iline = 0; iline < 4; ++iline) {
    invertedCol.setBendPattern(~col.getBendPattern(iline), iline);
  }
  mMasksHandler.applyMask(invertedCol);
  if (invertedCol.isEmpty()) {
    return false;
  }
  invertedData.emplace_back(invertedCol);
  return true;
}

} // namespace mid
} // namespace o2
