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

/// \file   MID/Filtering/src/MaskMaker.cxx
/// \brief  Function to produce the MID masks
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   05 March 2021

#include "MIDFiltering/MaskMaker.h"
#include "MIDFiltering/ChannelMasksHandler.h"
#include "MIDBase/ColumnDataHandler.h"
#include "MIDBase/Mapping.h"
#include "MIDBase/DetectorParameters.h"
#include "DataFormatsMID/ROFRecord.h"
#include "DataFormatsMID/ROBoard.h"
#include "MIDRaw/DecodedDataAggregator.h"

namespace o2
{
namespace mid
{
std::vector<ColumnData> makeBadChannels(const ChannelScalers& scalers, double timeOrTriggers, double threshold)
{
  /// Makes the mask from the scalers
  uint32_t nThresholdEvents = static_cast<uint32_t>(threshold * timeOrTriggers);
  ColumnDataHandler handler;
  for (const auto scaler : scalers.getScalers()) {
    if (scaler.second >= nThresholdEvents) {
      handler.add(scalers.getDeId(scaler.first), scalers.getColumnId(scaler.first), scalers.getLineId(scaler.first), scalers.getStrip(scaler.first), scalers.getCathode(scaler.first));
    }
  }
  return handler.getMerged();
}

std::vector<ColumnData> makeMasks(const ChannelScalers& scalers, double timeOrTriggers, double threshold, const std::vector<ColumnData>& refMasks)
{
  auto badChannels = makeBadChannels(scalers, timeOrTriggers, threshold);
  ChannelMasksHandler maskHandler;
  maskHandler.switchOffChannels(badChannels);
  std::vector<ColumnData> masks(maskHandler.getMasks());

  if (!refMasks.empty()) {
    ChannelMasksHandler defaultMaskHandler;
    defaultMaskHandler.setFromChannelMasks(refMasks);

    for (auto& mask : masks) {
      defaultMaskHandler.applyMask(mask);
    }
  }

  return masks;
}

std::vector<ColumnData> makeDefaultMasks()
{
  /// Makes the default mask from the mapping
  Mapping mapping;
  ChannelMasksHandler masksHandler;
  uint16_t fullPattern = 0xFFFF;
  for (int ide = 0; ide < detparams::NDetectionElements; ++ide) {
    for (int icol = mapping.getFirstColumn(ide); icol < 7; ++icol) {
      ColumnData mask;
      mask.deId = static_cast<uint8_t>(ide);
      mask.columnId = static_cast<uint8_t>(icol);
      // int nFullPatterns = 0;
      for (int iline = mapping.getFirstBoardBP(icol, ide), lastLine = mapping.getLastBoardBP(icol, ide); iline <= lastLine; ++iline) {
        mask.setBendPattern(fullPattern, iline);
        // ++nFullPatterns;
      }
      for (int istrip = 0; istrip < mapping.getNStripsNBP(icol, ide); ++istrip) {
        mask.addStrip(istrip, 1, 0);
      }
      masksHandler.setFromChannelMask(mask);
    }
  }

  return masksHandler.getMasks();
}

std::vector<ROBoard> getActiveBoards(const FEEIdConfig& feeIdConfig, const CrateMasks& masks)
{
  /// Gets the list of the active boards from the crate configuration
  std::vector<ROBoard> activeBoards;
  auto gbtUniqueIds = feeIdConfig.getConfiguredGBTUniqueIDs();
  ROBoard board;
  board.statusWord = raw::sCARDTYPE;
  board.patternsBP.fill(0xFFFF);
  board.patternsNBP.fill(0xFFFF);
  board.firedChambers = 0xF;
  for (auto& gbtUniqueId : gbtUniqueIds) {
    auto mask = masks.getMask(gbtUniqueId);
    auto crateId = crateparams::getCrateIdFromGBTUniqueId(gbtUniqueId);
    for (int iloc = 0; iloc < 8; ++iloc) {
      if ((mask >> iloc) & 0x1) {
        board.boardId = raw::makeUniqueLocID(crateId, crateparams::getLocIdInCrate(gbtUniqueId, iloc));
        activeBoards.emplace_back(board);
      }
    }
  }
  return activeBoards;
}

std::vector<ColumnData> makeDefaultMasksFromCrateConfig(const FEEIdConfig& feeIdConfig, const CrateMasks& crateMasks)
{
  /// Makes the default masks from the crate configuration
  auto activeBoards = getActiveBoards(feeIdConfig, crateMasks);
  std::vector<ROFRecord> rofs;
  rofs.push_back({InteractionRecord(), EventType::Standard, 0, activeBoards.size()});
  DecodedDataAggregator aggregator;
  aggregator.process(activeBoards, rofs);
  std::vector<ColumnData> maskedBoards(aggregator.getData());
  ChannelMasksHandler masksHandler;
  masksHandler.setFromChannelMasks(makeDefaultMasks());
  for (auto& col : maskedBoards) {
    masksHandler.applyMask(col);
  }

  return maskedBoards;
}

} // namespace mid
} // namespace o2
