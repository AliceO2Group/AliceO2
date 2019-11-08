// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/Raw/src/CRUUserLogicDecoder.cxx
/// \brief  MID CRU user logic decoder
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   30 September 2019

#include "MIDRaw/CRUUserLogicDecoder.h"

#include "RawInfo.h"

#include "MIDRaw/CrateParameters.h"
#include "CommonConstants/Triggers.h"

namespace o2
{
namespace mid
{

void CRUUserLogicDecoder::reset()
{
  /// Rewind bytes
  mData.clear();
  mROFRecords.clear();
}

void CRUUserLogicDecoder::process(gsl::span<const raw::RawUnit> bytes)
{
  /// Sets the buffer and reset the internal indexes
  reset();
  mBuffer.setBuffer(bytes);

  // Each data block consists of:
  // 28 bits of event type
  // up to 8 local boards info consisting of at most 136 bits
  // So, if we want to have a full block,
  // the buffer size must be at least 28 + 8 * 136 = 1116 bits
  // i.e. 35 uint32_t
  while (mBuffer.hasNext(35)) {
    processBlock();
  }
}

bool CRUUserLogicDecoder::processBlock()
{
  /// Processes the next block of data
  size_t firstEntry = mData.size();

  uint8_t eventWord = mBuffer.next(sNBitsEventWord);
  uint16_t localClock = mBuffer.next(sNBitsLocalClock);
  uint8_t crateId = mBuffer.next(sNBitsCrateId);
  int nFiredBoards = mBuffer.next(sNBitsNFiredBoards);

  EventType eventType = EventType::Standard;
  if (crateparams::isCalibration(eventWord)) {
    eventType = EventType::Noise;
  } else if (crateparams::isFET(eventWord)) {
    eventType = EventType::Dead;
  }
  InteractionRecord intRec(localClock, mBuffer.getRDH()->triggerOrbit);

  for (int iboard = 0; iboard < nFiredBoards; ++iboard) {
    uint8_t locId = mBuffer.next(sNBitsLocId);
    uint8_t firedChambers = mBuffer.next(sNBitsFiredChambers);
    mData.push_back({0, 0, crateparams::makeUniqueLocID(crateId, locId), firedChambers});
    for (int ich = 0; ich < 4; ++ich) {
      if ((firedChambers >> ich) & 0x1) {
        mData.back().patternsBP[ich] = mBuffer.next(sNBitsFiredStrips);
        mData.back().patternsNBP[ich] = mBuffer.next(sNBitsFiredStrips);
      }
    }
  }

  mROFRecords.emplace_back(intRec, eventType, firstEntry, mData.size() - firstEntry);

  mBuffer.skipOverhead();

  return true;
}

} // namespace mid
} // namespace o2
