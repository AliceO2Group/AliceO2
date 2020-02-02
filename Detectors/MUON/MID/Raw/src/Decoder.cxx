// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/Raw/src/Decoder.cxx
/// \brief  MID raw data decoder
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   30 September 2019

#include "MIDRaw/Decoder.h"

#include "MIDBase/DetectorParameters.h"

#include "MIDRaw/CrateParameters.h"

namespace o2
{
namespace mid
{

ColumnData& Decoder::FindColumnData(uint8_t deId, uint8_t columnId, size_t firstEntry)
{
  /// Gets the matching column data
  /// Adds one if not found
  for (auto colIt = mData.begin() + firstEntry; colIt != mData.end(); ++colIt) {
    if (colIt->deId == deId && colIt->columnId == columnId) {
      return *colIt;
    }
  }
  mData.push_back({deId, columnId});
  return mData.back();
}

void Decoder::addData(const LocalBoardRO& loc, size_t firstEntry)
{
  /// Convert the loc data to ColumnData
  uint8_t uniqueLocId = loc.boardId;
  uint8_t crateId = crateparams::getCrateId(uniqueLocId);
  bool isRightSide = crateparams::isRightSide(crateId);
  uint16_t deBoardId = mCrateMapper.roLocalBoardToDE(crateId, crateparams::getLocId(loc.boardId));
  auto rpcLineId = mCrateMapper.getRPCLine(deBoardId);
  auto columnId = mCrateMapper.getColumnId(deBoardId);
  auto lineId = mCrateMapper.getLineId(deBoardId);
  for (int ich = 0; ich < 4; ++ich) {
    if (((loc.firedChambers >> ich) & 0x1) == 0) {
      continue;
    }
    uint8_t deId = detparams::getDEId(isRightSide, ich, rpcLineId);
    auto& col = FindColumnData(deId, columnId, firstEntry);
    col.setBendPattern(loc.patternsBP[ich], lineId);
    col.setNonBendPattern(loc.patternsNBP[ich]);
  }
}

void Decoder::process(gsl::span<const raw::RawUnit> bytes)
{
  /// Decode the raw data

  // First clear the output
  mData.clear();
  mROFRecords.clear();

  // Process the input data
  mCRUUserLogicDecoder.process(bytes);

  if (mCRUUserLogicDecoder.getROFRecords().empty()) {
    return;
  }

  // Fill the map with ordered events
  for (auto rofIt = mCRUUserLogicDecoder.getROFRecords().begin(); rofIt != mCRUUserLogicDecoder.getROFRecords().end(); ++rofIt) {
    mOrderIndexes[rofIt->interactionRecord.toLong()].emplace_back(rofIt - mCRUUserLogicDecoder.getROFRecords().begin());
  }

  const ROFRecord* rof = nullptr;
  for (auto& item : mOrderIndexes) {
    size_t firstEntry = mData.size();
    for (auto& idx : item.second) {
      // In principle all of these ROF records have the same timestamp
      rof = &mCRUUserLogicDecoder.getROFRecords()[idx];
      for (size_t iloc = rof->firstEntry; iloc < rof->firstEntry + rof->nEntries; ++iloc) {
        addData(mCRUUserLogicDecoder.getData()[iloc], firstEntry);
      }
    }
    mROFRecords.emplace_back(rof->interactionRecord, rof->eventType, firstEntry, mData.size() - firstEntry);
  }

  // Clear the inner objects when the computation is done
  mOrderIndexes.clear();
}

} // namespace mid
} // namespace o2
