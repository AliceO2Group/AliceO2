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

/// \file   MID/Raw/src/DecodedDataAggregator.cxx
/// \brief  MID decoded raw data aggregator
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   26 February 2020

#include "MIDRaw/DecodedDataAggregator.h"

#include "Framework/Logger.h"
#include "MIDBase/DetectorParameters.h"
#include "MIDRaw/CrateParameters.h"

namespace o2
{
namespace mid
{

ColumnData& DecodedDataAggregator::FindColumnData(uint8_t deId, uint8_t columnId, size_t firstEntry, size_t evtTypeIdx)
{
  /// Gets the matching column data
  /// Adds one if not found
  for (auto colIt = mData[evtTypeIdx].begin() + firstEntry, end = mData[evtTypeIdx].end(); colIt != end; ++colIt) {
    if (colIt->deId == deId && colIt->columnId == columnId) {
      return *colIt;
    }
  }
  mData[evtTypeIdx].push_back({deId, columnId});
  return mData[evtTypeIdx].back();
}

void DecodedDataAggregator::addData(const ROBoard& loc, size_t firstEntry, size_t evtTypeIdx)
{
  /// Converts the local board data to ColumnData
  uint8_t uniqueLocId = loc.boardId;
  uint8_t crateId = raw::getCrateId(uniqueLocId);
  bool isRightSide = crateparams::isRightSide(crateId);
  try {
    uint16_t deBoardId = mCrateMapper.roLocalBoardToDE(uniqueLocId);
    auto rpcLineId = detparams::getRPCLine(detparams::getDEIdFromFEEId(deBoardId));
    auto columnId = detparams::getColumnIdFromFEEId(deBoardId);
    auto lineId = detparams::getLineIdFromFEEId(deBoardId);
    for (int ich = 0; ich < 4; ++ich) {
      if (((loc.firedChambers >> ich) & 0x1) == 0) {
        continue;
      }
      uint8_t deId = detparams::getDEId(isRightSide, ich, rpcLineId);
      auto& col = FindColumnData(deId, columnId, firstEntry, evtTypeIdx);
      col.setBendPattern(loc.patternsBP[ich], lineId);
      col.setNonBendPattern(col.getNonBendPattern() | loc.patternsNBP[ich]);
    }
  } catch (const std::exception& except) {
    LOG(alarm) << except.what();
  }
}

void DecodedDataAggregator::process(gsl::span<const ROBoard> localBoards, gsl::span<const ROFRecord> rofRecords)
{
  /// Aggregates the decoded raw data

  // First clear the output
  for (auto& data : mData) {
    data.clear();
  }
  for (auto& rof : mROFRecords) {
    rof.clear();
  }

  // Fill the map with ordered events
  for (auto rofIt = rofRecords.begin(); rofIt != rofRecords.end(); ++rofIt) {
    mEventIndexes[static_cast<int>(rofIt->eventType)][rofIt->interactionRecord.toLong()].emplace_back(rofIt - rofRecords.begin());
  }

  const ROFRecord* rof = nullptr;
  for (size_t ievtType = 0; ievtType < mEventIndexes.size(); ++ievtType) {
    for (auto& item : mEventIndexes[ievtType]) {
      size_t firstEntry = mData[ievtType].size();
      for (auto& idx : item.second) {
        // In principle all of these ROF records have the same timestamp
        rof = &rofRecords[idx];
        for (size_t iloc = rof->firstEntry; iloc < rof->firstEntry + rof->nEntries; ++iloc) {
          addData(localBoards[iloc], firstEntry, ievtType);
        }
      }
      auto nEntries = mData[ievtType].size() - firstEntry;
      if (nEntries > 0) {
        mROFRecords[ievtType].emplace_back(rof->interactionRecord, rof->eventType, firstEntry, nEntries);
      }
    }
    // Clear the inner objects when the computation is done
    mEventIndexes[ievtType].clear();
  } // loop on event types
}

} // namespace mid
} // namespace o2
