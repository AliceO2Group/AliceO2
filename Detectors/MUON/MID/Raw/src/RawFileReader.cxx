// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/Raw/src/RawFileReader.cxx
/// \brief  MID raw file reader
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   25 November 2019

#include "MIDRaw/RawFileReader.h"

#include <iostream>
#include "Headers/RAWDataHeader.h"

namespace o2
{
namespace mid
{
template <typename T>
bool RawFileReader<T>::init(const char* inFilename, bool readContinuous)
{
  /// Initializes the raw file reader
  mFile.open(inFilename, std::ios::binary);
  if (!mFile.is_open()) {
    std::cerr << "Cannot open the " << inFilename << " file !";
    mState = 2;
    return false;
  }
  mBytes.reserve(2 * raw::sMaxBufferSize);
  mReadContinuous = readContinuous;
  mHBCounters.fill(0);

  return true;
}

template <typename T>
void RawFileReader<T>::read(size_t nBytes)
{
  /// Reads nBytes of the file
  size_t currentIndex = mBytes.size();
  mBytes.resize(currentIndex + nBytes / sizeof(T));
  mFile.read(reinterpret_cast<char*>(&(mBytes[currentIndex])), nBytes);
}

template <typename T>
void RawFileReader<T>::clear()
{
  /// Clears the bytes and counters
  mBytes.clear();
  mBuffer.setBuffer(mBytes, RawBuffer<T>::ResetMode::all);
  mHBCounters.fill(0);
}

template <typename T>
bool RawFileReader<T>::hasFullInfo()
{
  /// Tests if we have read the same number of HBs for all GBT links

  // We assume here that we read the data of one HV for all of the GBT links
  // before moving to the next HB
  for (uint16_t feeId = 1; feeId < crateparams::sNGBTs; ++feeId) {
    if (mHBCounters[feeId] != mHBCounters[0]) {
      return false;
    }
  }
  return mHBCounters[0] != 0;
}

template <typename T>
bool RawFileReader<T>::readAllGBTs(bool reset)
{
  /// Keeps reading the file until it reads the information of all GBT links
  /// It returns the number of HBs read

  if (reset) {
    clear();
  }

  while (!hasFullInfo()) {
    if (!readHB()) {
      return false;
    }
  }
  return true;
}

template <typename T>
bool RawFileReader<T>::replaceRDH(size_t headerIndex)
{
  /// Replaces the current RDH with a custom one if needed.
  /// This is done to be able to correctly read test data
  /// that have a wrong RDH
  if (mCustomRDH.offsetToNext > 0) {
    header::RAWDataHeader* rdh = reinterpret_cast<header::RAWDataHeader*>(&mBytes[headerIndex]);
    *rdh = mCustomRDH;
    return true;
  }
  return false;
}

template <typename T>
bool RawFileReader<T>::readHB(bool sendCompleteHBs)
{
  /// Reads one HB
  if (mState != 0) {
    return false;
  }
  auto gbtId = 0;
  bool isHBClosed = false;
  while (!isHBClosed) {
    // Read header
    size_t headerIndex = mBytes.size();
    read(raw::sHeaderSizeInBytes);

    // The check on the eof needs to be placed here and not at the beginning of the function.
    // The reason is that the eof flag is set if we try to read after the eof
    // But, since we know the size, we read up to the last character.
    // So we turn on the eof flag only if we try to read past the last data.
    // Of course, we resized the mBytes before trying to read.
    // Since we read 0, we need to remove the last bytes
    if (mFile.eof()) {
      mBytes.resize(headerIndex);
      if (mReadContinuous) {
        mFile.clear();
        mFile.seekg(0, std::ios::beg);
        read(raw::sHeaderSizeInBytes);
      } else {
        mState = 1;
        return false;
      }
    }
    replaceRDH(headerIndex);
    // We use the buffer only to correctly initialize the RDH
    mBuffer.setBuffer(mBytes, RawBuffer<T>::ResetMode::bufferOnly);
    mBuffer.nextHeader();
    isHBClosed = mBuffer.isHBClosed();
    gbtId = mBuffer.getRDH()->feeId;
    if (gbtId >= crateparams::sNGBTs) {
      // FIXME: this is a problem of the header of some test files
      gbtId = 0;
    }
    if (mBuffer.getRDH()->offsetToNext > raw::sHeaderSizeInBytes) {
      read(mBuffer.getRDH()->offsetToNext - raw::sHeaderSizeInBytes);
      // CAVEAT: to save memory / CPU time, the RawBuffer does not hold a copy of the buffer,
      // but just a span of it.
      // If we add bytes to mBytes, the vector can go beyond the capacity
      // and the memory is re-allocated.
      // If this happens, the span is no longer valid, and we can no longer
      // call mBuffer.getRDH() until we pass it mBytes again
      // To do so, you need to call:
      // mBuffer.setBuffer(mBytes, RawBuffer<T>::ResetMode::bufferOnly);
      // mBuffer.nextHeader();
    }
    if (!sendCompleteHBs) {
      break;
    }
  }

  ++mHBCounters[gbtId];

  return true;
}

template class RawFileReader<raw::RawUnit>;
template class RawFileReader<uint8_t>;

} // namespace mid
} // namespace o2