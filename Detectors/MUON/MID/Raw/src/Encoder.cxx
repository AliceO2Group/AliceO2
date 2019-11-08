// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/Raw/src/Encoder.cxx
/// \brief  MID raw data encoder
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   30 September 2019

#include "MIDRaw/Encoder.h"

#include "MIDBase/DetectorParameters.h"

namespace o2
{
namespace mid
{

void Encoder::clear()
{
  /// Reset bytes
  for (auto& linkEnc : mCRUUserLogicEncoders) {
    linkEnc.clear();
  }
}

void Encoder::newHeader(const InteractionRecord& ir)
{
  /// Add new RDH
  std::vector<InteractionRecord> HBIRVec;
  InteractionRecord irFrom = mLastIR.isDummy() ? mHBFUtils.getFirstIR() : mLastIR + 1;
  mHBFUtils.fillHBIRvector(HBIRVec, irFrom, ir);
  mLastIR = ir;
  for (auto& hbIr : HBIRVec) {
    auto rdh = mHBFUtils.createRDH<header::RAWDataHeader>(hbIr);
    for (auto linkEncIt = mCRUUserLogicEncoders.begin(); linkEncIt != mCRUUserLogicEncoders.end(); ++linkEncIt) {
      // The link ID is sequential
      uint16_t feeId = linkEncIt - mCRUUserLogicEncoders.begin();
      linkEncIt->newHeader(feeId, rdh);
    }
  }
}

void Encoder::setHeaderOffset(bool headerOffset)
{
  for (auto& linkEnc : mCRUUserLogicEncoders) {
    linkEnc.setHeaderOffset(headerOffset);
  }
}

bool Encoder::convertData(gsl::span<const ColumnData> data)
{
  /// Convert incoming data to FEE format
  mROData.clear();
  for (auto& col : data) {
    for (int iline = 0; iline < 4; ++iline) {
      if (col.getBendPattern(iline) == 0) {
        continue;
      }
      auto uniqueLocId = mCrateMapper.deLocalBoardToRO(col.deId, col.columnId, iline);
      auto& roData = mROData[uniqueLocId];
      roData.boardId = crateparams::getLocId(uniqueLocId);
      int ich = detparams::getChamber(col.deId);
      roData.firedChambers |= (1 << ich);
      roData.patternsBP[ich] = col.getBendPattern(iline);
      roData.patternsNBP[ich] = col.getNonBendPattern();
    }
  }
  return mROData.empty();
}

void Encoder::process(gsl::span<const ColumnData> data, const InteractionRecord& ir, EventType eventType)
{
  /// Encode data
  newHeader(ir);

  if (convertData(data)) {
    return;
  }

  std::vector<LocalBoardRO> localBoardROs;
  localBoardROs.reserve(8);
  uint16_t lastROId = crateparams::sNGBTs + 1;
  // The local board ID is defined as:
  // board ID in crate [0-3]
  // crate ID [4-7]
  // With this definition, the local boards within the same crate have a sequential ID
  // This is important since it means that the boards within the same crate are ordered in the mROData map.
  // So, we can cumulate the strip pattern of one link, and process them when we switch to another link
  for (auto& item : mROData) {
    auto crateId = crateparams::getCrateId(item.first);
    auto roId = crateparams::makeROId(crateId, crateparams::getGBTIdFromBoardInCrate(item.second.boardId));
    if (roId != lastROId) {
      // We are in a new link
      // Let us check that this is not just the first board
      if (lastROId < crateparams::sNGBTs) {
        // We process the collected boards since they belong to the same link
        mCRUUserLogicEncoders[lastROId].process(localBoardROs, ir.bc, eventType);
        // And we clear the vector so that we can start collecting the board of a new link
        localBoardROs.clear();
      }
      lastROId = roId;
    }
    localBoardROs.emplace_back(item.second);
  }

  // With the current logic, the last read link is not processed
  // Let us process it here
  if (!localBoardROs.empty()) {
    mCRUUserLogicEncoders[lastROId].process(localBoardROs, ir.bc, eventType);
  }
}

const std::vector<raw::RawUnit>& Encoder::getBuffer()
{
  /// Gets the buffer
  mBytes.clear();
  for (uint16_t roId = 0; roId < crateparams::sNGBTs; ++roId) {
    auto& buffer = mCRUUserLogicEncoders[roId].getBuffer();
    std::copy(buffer.begin(), buffer.end(), std::back_inserter(mBytes));
  }
  return mBytes;
}

} // namespace mid
} // namespace o2
