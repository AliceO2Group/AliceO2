// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/Raw/src/CRUUserLogicEncoder.cxx
/// \brief  Raw data encoder for MID CRU user logic
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   18 November 2019

#include "MIDRaw/CRUUserLogicEncoder.h"

#include "MIDRaw/CrateParameters.h"
#include "RawInfo.h"

namespace o2
{
namespace mid
{
void CRUUserLogicEncoder::add(int value, unsigned int nBits)
{
  /// Add information
  for (int ibit = 0; ibit < nBits; ++ibit) {
    if (mBitIndex == raw::sElementSizeInBits) {
      mBitIndex = 0;
      if (mBytes.size() == mHeaderIndex + sMaxPageSize) {
        completePage(false);
        // Copy previous header
        std::vector<raw::RawUnit> oldHeader(mBytes.begin() + mHeaderIndex, mBytes.begin() + mHeaderIndex + raw::sHeaderSizeInElements);
        mHeaderIndex = mBytes.size();
        std::copy(oldHeader.begin(), oldHeader.end(), std::back_inserter(mBytes));
        // And increase the page counter
        ++getRDH()->pageCnt;
      }
      mBytes.push_back(0);
    }
    bool isOn = (value >> ibit) & 0x1;
    if (isOn) {
      mBytes.back() |= (1 << mBitIndex);
    }
    ++mBitIndex;
  }
}

void CRUUserLogicEncoder::clear()
{
  /// Reset bytes
  mBytes.clear();
  mBitIndex = 0;
  mHeaderIndex = 0;
}

void CRUUserLogicEncoder::newHeader(uint16_t feeId, const header::RAWDataHeader& baseRDH)
{
  /// Add new RDH
  completePage(true);
  mHeaderIndex = mBytes.size();
  for (size_t iel = 0; iel < raw::sHeaderSizeInElements; ++iel) {
    mBytes.emplace_back(0);
  }
  auto rdh = getRDH();
  *rdh = baseRDH;
  rdh->feeId = feeId;
  mBitIndex = raw::sElementSizeInBits;
}

void CRUUserLogicEncoder::completePage(bool stop)
{
  /// Complete the information on the page
  if (mBytes.empty()) {
    return;
  }
  auto pageSizeInElements = mBytes.size() - mHeaderIndex;
  if (mHeaderOffset && pageSizeInElements < sMaxPageSize) {
    // Write zeros up to the end of the page
    mBytes.resize(mHeaderIndex + sMaxPageSize, 0);
  }

  // This has to be done after the resize
  auto rdh = getRDH();
  rdh->memorySize = pageSizeInElements * raw::sElementSizeInBytes;
  rdh->offsetToNext = (mBytes.size() - mHeaderIndex) * raw::sElementSizeInBytes;
  rdh->stop = stop;
}

const std::vector<raw::RawUnit>& CRUUserLogicEncoder::getBuffer()
{
  /// Gets the buffer
  if (!mBytes.empty() && !getRDH()->stop) {
    // Close page before getting buffer
    completePage(true);
  }
  return mBytes;
}

void CRUUserLogicEncoder::process(gsl::span<const LocalBoardRO> sortedData, const uint16_t bc, EventType eventType)
{
  /// Encode data

  // FIXME: Finalize the final format of the event word together with the CRU + RO team
  uint16_t eventWord = 0;
  if (eventType == EventType::Noise) {
    eventWord = 0x8;
  } else if (eventType == EventType::Dead) {
    eventWord = 0xc;
  }
  add(eventWord, sNBitsEventWord);
  add(bc, sNBitsLocalClock);
  add(crateparams::getCrateIdFromROId(getRDH()->feeId), sNBitsCrateId);
  add(sortedData.size(), sNBitsNFiredBoards);
  for (auto& loc : sortedData) {
    add(loc.boardId, sNBitsLocId);
    add(loc.firedChambers, sNBitsFiredChambers);
    for (int ich = 0; ich < 4; ++ich) {
      if ((loc.firedChambers >> ich) & 0x1) {
        add(loc.patternsBP[ich], sNBitsFiredStrips);
        add(loc.patternsNBP[ich], sNBitsFiredStrips);
      }
    }
  }
}

} // namespace mid
} // namespace o2
