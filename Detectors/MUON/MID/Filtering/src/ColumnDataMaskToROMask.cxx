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

/// \file   MID/Filtering/src/ColumnDataMaskToROMask.cxx
/// \brief  Converts ColumnData masks into local board masks
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   12 October 2021

#include "MIDFiltering/ColumnDataMaskToROMask.h"

#include <fstream>
#include "fmt/format.h"
#include "MIDFiltering/MaskMaker.h"
#include "MIDFiltering/ChannelMasksHandler.h"

namespace o2
{
namespace mid
{
ColumnDataMaskToROMask::ColumnDataMaskToROMask()
{
  // Default ctr
  mDefaultMasks = makeDefaultMasks();
  mColToBoard.setDebugMode(true);
}

bool ColumnDataMaskToROMask::needsMask(const ROBoard& mask, bool hasDirectInputY) const
{
  /// Returns true if the local board is masked
  if (!hasDirectInputY) {
    return true;
  }

  for (int ich = 0; ich < 4; ++ich) {
    if (mask.patternsBP[ich] != 0xFFFF || mask.patternsNBP[ich] != 0xFFFF) {
      return true;
    }
  }
  return false;
}

uint32_t ColumnDataMaskToROMask::makeConfigWord(const ROBoard& mask) const
{
  /// Computes the configuration word for the local board
  auto cfgWord = sTxDataMask;
  bool hasDirectInputY = mCrateMapper.hasDirectInputY(mask.boardId);
  bool isMasked = needsMask(mask, hasDirectInputY);
  if (isMasked) {
    cfgWord |= sMonmoff;
  }
  if (!hasDirectInputY) {
    cfgWord |= sXorY;
  }
  return cfgWord;
}

std::vector<ROBoard> ColumnDataMaskToROMask::convert(gsl::span<const ColumnData> colDataMasks)
{
  /// Converts ColumnData masks into local board masks
  mColToBoard.process(colDataMasks);

  std::vector<ROBoard> roMasks;
  for (auto& mapIt : mColToBoard.getData()) {
    for (auto& board : mapIt.second) {
      roMasks.emplace_back(board);
    }
  }
  return roMasks;
}

void ColumnDataMaskToROMask::write(gsl::span<const ColumnData> colDataMasks, const char* outFilename)
{
  /// Writes the mask to file

  // FIXME: currently, the masks are written to a file that is then read by WinCC
  // In the future, this should be moved to ccdb

  auto roMasks = convert(colDataMasks);

  // We order the masks for easier reading from humans
  std::sort(roMasks.begin(), roMasks.end(), [](const ROBoard& loc1, const ROBoard& loc2) { return loc1.boardId < loc2.boardId; });

  std::ofstream outFile(outFilename);
  for (auto& mask : roMasks) {
    auto cfgWord = makeConfigWord(mask);
    outFile << fmt::format("{:02x} {:08x}", mask.boardId, cfgWord);

    bool isMasked = cfgWord & sMonmoff;

    for (int ich = 0; ich < 4; ++ich) {
      uint16_t valBP = isMasked ? mask.patternsBP[ich] : 0;
      uint16_t valNBP = (isMasked && (cfgWord & sXorY) == 0) ? mask.patternsNBP[ich] : 0;
      outFile << fmt::format(" {:04x}{:04x}", valBP, valNBP);
    }
    outFile << std::endl;
  }
}
} // namespace mid
} // namespace o2
