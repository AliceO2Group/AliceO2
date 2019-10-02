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
#include "RawInfo.h"

#include <map>

namespace o2
{
namespace mid
{

void Decoder::init()
{
  mData.reserve(8000);
  mROFRecords.reserve(2000);
}

int Decoder::next(unsigned int nBits)
{
  /// Get value
  int value = 0;
  for (int ibit = 0; ibit < nBits; ++ibit) {
    if (mBitIndex == raw::sElementSizeInBits) {
      mBitIndex = 0;
      ++mVectorIndex;
      nextPayload();
    }
    bool isOn = (mBytes[mVectorIndex] >> mBitIndex) & 0x1;
    if (isOn) {
      value |= (1 << ibit);
    }
    ++mBitIndex;
  }
  return value;
}

void Decoder::reset()
{
  /// Rewind bytes
  mVectorIndex = 0;
  mHeaderIndex = 0;
  mNextHeaderIndex = 0;
  mBitIndex = 0;
  mRDH = nullptr;
  mData.clear();
  mROFRecords.clear();
}

void Decoder::process(gsl::span<const raw::RawUnit> bytes)
{
  /// Sets the buffer and reset the internal indexes
  reset();
  if (bytes.empty()) {
    return;
  }
  mBytes = bytes;
  while (nextColumnData())
    ;
}

bool Decoder::isEndOfBuffer() const
{
  /// We are at the end of the buffer
  return mVectorIndex == mBytes.size();
}

bool Decoder::nextPayload()
{
  /// Go to next payload
  while (mVectorIndex == mNextHeaderIndex) {
    mRDH = reinterpret_cast<const header::RAWDataHeader*>(&mBytes[mVectorIndex]);
    mHeaderIndex = mNextHeaderIndex;
    mNextHeaderIndex += mRDH->offsetToNext / raw::sElementSizeInBytes;
    mBitIndex = 0;
    if (mRDH->memorySize > raw::sHeaderSizeInBytes) {
      // Payload is not empty: go to end of header
      mVectorIndex += raw::sHeaderSizeInElements;
    } else {
      // Payload is empty: go directly to next header and check it
      mVectorIndex = mNextHeaderIndex;
      if (isEndOfBuffer()) {
        return false;
      }
    }
  }
  return true;
}

bool Decoder::nextColumnData()
{
  /// Gets the array of column data
  if (isEndOfBuffer() || !nextPayload()) {
    return false;
  }

  EventType eventType = static_cast<EventType>(next(sNBitsEventType));
  uint16_t localClock = next(sNBitsLocalClock);
  InteractionRecord intRec(localClock, mRDH->triggerOrbit);
  auto firstEntry = mData.size();
  int nFiredRPCs = next(sNBitsNFiredRPCs);
  ColumnData colData;
  for (int irpc = 0; irpc < nFiredRPCs; ++irpc) {
    int rpcId = next(sNBitsRPCId);
    int nFiredColumns = next(sNBitsNFiredColumns);
    colData.deId = static_cast<uint8_t>(rpcId);
    for (int icol = 0; icol < nFiredColumns; ++icol) {
      colData.columnId = static_cast<uint8_t>(next(sNBitsColumnId));
      for (int iline = 0; iline < 4; ++iline) {
        // Reset pattern
        colData.setBendPattern(0, iline);
      }
      int nFiredBoards = next(sNBitsNFiredBoards);
      for (int iboard = 0; iboard < nFiredBoards; ++iboard) {
        int boardId = next(sNBitsBoardId);
        colData.setNonBendPattern(next(sNBitsFiredStrips));
        colData.setBendPattern(next(sNBitsFiredStrips), boardId);
      }
      mData.emplace_back(colData);
    }
  }

  mROFRecords.emplace_back(intRec, eventType, firstEntry, mData.size() - firstEntry);

  // The payload bits are contiguous.
  // However there is an overhead between the last payload byte and the next header.
  // If we read the full memory size, we
  // Which means that the size is too small to have another block of data.
  // When we are close to the new header, we jump directly to the next header
  if ((mVectorIndex - mHeaderIndex + 1) * raw::sElementSizeInBytes == mRDH->memorySize) {
    mVectorIndex = mNextHeaderIndex;
    mBitIndex = 0;
  }

  return true;
}

} // namespace mid
} // namespace o2
