// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/Raw/src/RawBuffer.cxx
/// \brief  MID CRU user logic decoder
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   30 September 2019

#include "MIDRaw/RawBuffer.h"

#include "MIDRaw/RawUnit.h"
#include "RawInfo.h"

namespace o2
{
namespace mid
{

template <typename T>
unsigned int RawBuffer<T>::next(unsigned int nBits)
{
  /// Reads the next nBits
  if (mNextHeaderIndex == 0) {
    // This is the first entry
    nextPayload();
  }
  unsigned int value = 0;
  for (int ibit = 0; ibit < nBits; ++ibit) {
    if (mBitIndex == mElementSizeInBits) {
      mBitIndex = 0;
      ++mElementIndex;
      nextPayload();
    }
    bool isOn = (mBytes[mElementIndex] >> mBitIndex) & 0x1;
    if (isOn) {
      value |= (1 << ibit);
    }
    ++mBitIndex;
  }
  return value;
}

template <typename T>
T RawBuffer<T>::next()
{
  /// Reads the next word
  nextPayload();
  return mBytes[mElementIndex++];
}

template <typename T>
bool RawBuffer<T>::nextHeader()
{
  /// Goes to next RDH
  if (mNextHeaderIndex >= mBytes.size()) {
    // This is the end of the buffer
    if (mUnconsumed.empty()) {
      mElementIndex = mNextHeaderIndex;
      mEndOfPayloadIndex = mNextHeaderIndex;
      return false;
    }
    // We were reading the unconsumed part: switch to new buffer
    mUnconsumed.clear();
    reset();
    mBytes = mCurrentBuffer;
  }
  mHeaderIndex = mNextHeaderIndex;
  mRDH = reinterpret_cast<const header::RAWDataHeader*>(&mBytes[mHeaderIndex]);
  mEndOfPayloadIndex = mHeaderIndex + (mRDH->memorySize / mElementSizeInBytes);
  // Go to end of header, i.e. beginning of payload
  mElementIndex = mHeaderIndex + (mRDH->headerSize / mElementSizeInBytes);
  mNextHeaderIndex = mHeaderIndex + mRDH->offsetToNext / mElementSizeInBytes;
  mBitIndex = 0;

  return true;
}

template <typename T>
bool RawBuffer<T>::nextPayload()
{
  /// Goes to next payload
  while (mElementIndex == mEndOfPayloadIndex) {
    if (!nextHeader()) {
      return false;
    }
  }
  return true;
}

template <typename T>
void RawBuffer<T>::reset()
{
  /// Rewind bytes
  mElementIndex = 0;
  mHeaderIndex = 0;
  mNextHeaderIndex = 0;
  mEndOfPayloadIndex = 0;
  mBitIndex = 0;
  mRDH = nullptr;
  mUnconsumed.clear();
}

template <typename T>
void RawBuffer<T>::setBuffer(gsl::span<const T> bytes, ResetMode resetMode)
{
  /// Sets the buffer and reset the internal indexes
  if (resetMode == ResetMode::keepUnconsumed && !mUnconsumed.empty()) {
    // There are some unconsumed bytes from the previous buffer
    mNextHeaderIndex -= mHeaderIndex;
    mEndOfPayloadIndex -= mHeaderIndex;
    mElementIndex -= mHeaderIndex;
    mHeaderIndex = 0;
    mBytes = gsl::span<const T>(mUnconsumed);
  } else {
    mBytes = bytes;
    if (resetMode != ResetMode::bufferOnly) {
      reset();
    }
  }
  mCurrentBuffer = bytes;
}

template <typename T>
bool RawBuffer<T>::isHBClosed()
{
  /// Tests if the HB is closed
  if (!mRDH) {
    return false;
  }
  return mRDH->stop;
}

template <typename T>
void RawBuffer<T>::skipOverhead()
{
  /// This function must be called after reading a block of data
  /// if the payload is not provided and/or readout in bytes.
  ///
  /// In this case, indeed, there can  be an overhead between the last useful bit and the declared memory size,
  /// which is in bytes.
  /// Calling this function allows to jump directly to the next header when needed
  if (mElementIndex == mEndOfPayloadIndex - 1) {
    mElementIndex = mEndOfPayloadIndex;
    mBitIndex = 0;
  }
}

template <typename T>
bool RawBuffer<T>::hasNext(unsigned int nBytes)
{
  /// Tests if the buffer has nBytes left

  // We first need to go to the next payload
  // If we do not, we could have a set of empty HBs in front of us
  // With lot of memory left in the buffer but no payload
  nextPayload();
  bool isOk = mCurrentBuffer.size() + mUnconsumed.size() - mNextHeaderIndex + mEndOfPayloadIndex - mElementIndex >= nBytes / mElementSizeInBytes;
  if (!isOk && mElementIndex != mCurrentBuffer.size()) {
    // Store the remaining bits for further use
    // We need to do it here because the vector of which the mBytes is just a span might not be valid afterwards
    // (e.g. when we do the next setBuffer)
    // If we do not want to invalidate the mRDH pointer, we need to copy bytes from the last header
    mUnconsumed.insert(mUnconsumed.end(), mBytes.begin() + mHeaderIndex, mBytes.end());
  }
  return isOk;
}

template class RawBuffer<raw::RawUnit>;
template class RawBuffer<uint8_t>;

} // namespace mid
} // namespace o2
