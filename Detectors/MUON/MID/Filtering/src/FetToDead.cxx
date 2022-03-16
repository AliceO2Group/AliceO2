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
  mRefMasks = makeDefaultMasks();
}

void FetToDead::checkChannels(const ColumnData& mask, ColumnData fet, std::vector<ColumnData>& badChannels) const
{
  bool isBad = false;
  auto pattern = ((fet.getNonBendPattern() ^ mask.getNonBendPattern()) & mask.getNonBendPattern());
  fet.setNonBendPattern(pattern);
  if (pattern != 0) {
    isBad = true;
  }

  for (int iline = 0; iline < 4; ++iline) {
    pattern = ((fet.getBendPattern(iline) ^ mask.getBendPattern(iline)) & mask.getBendPattern(iline));
    fet.setBendPattern(pattern, iline);
    if (pattern != 0) {
      isBad = true;
    }
  }
  if (isBad) {
    badChannels.emplace_back(fet);
  }
}

std::vector<ColumnData> FetToDead::process(gsl::span<const ColumnData> fetData)
{
  /// Converts the FET result to a list of dead channels
  mFetData.clear();
  for (auto& col : fetData) {
    mFetData.emplace(getColumnDataUniqueId(col.deId, col.columnId), col);
  }
  std::vector<ColumnData> deadChannels;
  for (auto& mask : mRefMasks) {
    auto fetIt = mFetData.find(getColumnDataUniqueId(mask.deId, mask.columnId));
    ColumnData fet;
    if (fetIt == mFetData.end()) {
      fet.deId = mask.deId;
      fet.columnId = mask.columnId;
    } else {
      fet = fetIt->second;
    }
    checkChannels(mask, fet, deadChannels);
  }
  return deadChannels;
}

} // namespace mid
} // namespace o2
