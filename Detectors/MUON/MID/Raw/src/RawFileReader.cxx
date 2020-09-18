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
#include "Headers/RDHAny.h"
#include "DPLUtils/RawParser.h"
#include "DetectorsRaw/RDHUtils.h"

namespace o2
{
namespace mid
{
bool RawFileReader::init(const char* inFilename, bool readContinuous)
{
  /// Initializes the raw file reader
  mFile.open(inFilename, std::ios::binary);
  if (!mFile.is_open()) {
    std::cerr << "Cannot open the " << inFilename << " file !";
    mState = 2;
    return false;
  }
  mReadContinuous = readContinuous;
  return true;
}

void RawFileReader::read(size_t nBytes)
{
  /// Reads nBytes of the file
  size_t currentIndex = mBytes.size();
  mBytes.resize(currentIndex + nBytes);
  mFile.read(reinterpret_cast<char*>(&(mBytes[currentIndex])), nBytes);
}

void RawFileReader::clear()
{
  /// Clears the bytes and counters
  mBytes.clear();
}

void RawFileReader::setCustomPayloadSize(uint16_t memorySize, uint16_t offsetToNext)
{
  /// Sets a custom memory and payload size
  /// This is done to be able to correctly read test data
  /// that have a wrong RDH
  o2::header::RAWDataHeader rdh;
  rdh.word1 |= offsetToNext;
  rdh.word1 |= (memorySize << 16);
  setCustomRDH(rdh);
}

bool RawFileReader::replaceRDH(size_t headerIndex)
{
  /// Replaces the current RDH with a custom one if needed.
  /// This is done to be able to correctly read test data
  /// that have a wrong RDH
  if (o2::raw::RDHUtils::getOffsetToNext(mCustomRDH) > 0) {
    header::RAWDataHeader* rdh = reinterpret_cast<header::RAWDataHeader*>(&mBytes[headerIndex]);
    *rdh = mCustomRDH;
    return true;
  }
  return false;
}

bool RawFileReader::readHB(bool sendCompleteHBs)
{
  /// Reads one HB
  if (mState != 0) {
    return false;
  }
  bool isHBClosed = false;
  while (!isHBClosed) {
    // Read header
    size_t headerIndex = mBytes.size();
    read(sHeaderSize);

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
        read(sHeaderSize);
      } else {
        mState = 1;
        return false;
      }
    }
    replaceRDH(headerIndex);
    // We use the buffer only to correctly initialize the RDH
    o2::framework::RawParser parser(mBytes.data(), mBytes.size());
    auto lastIt = parser.begin();
    for (auto it = parser.begin(), end = parser.end(); it != end; ++it) {
      lastIt = it;
    }
    auto const* rdhPtr = reinterpret_cast<const o2::header::RDHAny*>(lastIt.raw());
    isHBClosed = o2::raw::RDHUtils::getStop(rdhPtr);
    auto offsetNext = o2::raw::RDHUtils::getOffsetToNext(rdhPtr);
    if (offsetNext > sHeaderSize) {
      read(offsetNext - sHeaderSize);
    }
    if (!sendCompleteHBs) {
      break;
    }
  }
  return true;
}

} // namespace mid
} // namespace o2
