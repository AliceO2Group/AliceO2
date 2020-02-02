// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/Raw/src/CRUBareDecoder.cxx
/// \brief  MID CRU core decoder
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   30 September 2019

#include "MIDRaw/CRUBareDecoder.h"

#include "RawInfo.h"

namespace o2
{
namespace mid
{

void CRUBareDecoder::reset()
{
  /// Rewind bytes
  mData.clear();
  mROFRecords.clear();
}

void CRUBareDecoder::init(bool debugMode)
{
  /// Initializes the task
  if (debugMode == true) {
    mAddReg = std::bind(&CRUBareDecoder::addBoard, this, std::placeholders::_1);
    mCheckBoard = [](size_t) { return true; };
  }
}

void CRUBareDecoder::process(gsl::span<const uint8_t> bytes)
{
  /// Decodes the buffer
  reset();

  mBuffer.setBuffer(bytes);

  // Each CRU word consists of 128 bits, i.e 16 bytes
  while (mBuffer.hasNext(16)) {
    processGBT(0);
    processGBT(5);
    // The GBT word consists of 10 bytes
    // but the CRU word is 16 bytes
    // So we have 6 empty bytes
    for (int iword = 0; iword < 6; ++iword) {
      mBuffer.next();
    }
  }
}

void CRUBareDecoder::processGBT(size_t offset)
{
  /// Processes the GBT

  // Regional link
  // loc#0 loc#1 loc#2 loc#3 reg
  std::array<uint8_t, 5> bytes{mBuffer.next(), mBuffer.next(), mBuffer.next(), mBuffer.next(), mBuffer.next()};

  // Byte corresponding to regional
  size_t ibyte = 4;
  size_t ilink = offset + ibyte;

  if (mELinkDecoders[ilink].add(bytes[ibyte], 0x80) && mELinkDecoders[ilink].isComplete()) {
    // In principle, in the same HB we should have the info of only 1 crate
    // So it should be safe to store this information
    // and apply it to all subsequent e-links, without checking the clock
    mCrateId = mELinkDecoders[ilink].getId();
    mAddReg(ilink);
    mELinkDecoders[ilink].reset();
  }

  // local links
  for (ibyte = 0; ibyte < 4; ++ibyte) {
    ilink = offset + ibyte;
    if (mELinkDecoders[ilink].add(bytes[ibyte], 0xc0) && mELinkDecoders[ilink].isComplete()) {
      if (mCheckBoard(ilink)) {
        addLoc(ilink);
      }
      mELinkDecoders[ilink].reset();
    }
  }
}

bool CRUBareDecoder::checkBoard(size_t ilink)
{
  /// Performs checks on the board
  uint8_t expectedId = ilink - (ilink / 5);
  return (expectedId == mELinkDecoders[ilink].getId() % 8);
}

void CRUBareDecoder::addBoard(size_t ilink)
{
  /// Adds the local or regional board to the output data vector
  uint16_t localClock = mELinkDecoders[ilink].getCounter();
  EventType eventType = EventType::Standard;
  if (mELinkDecoders[ilink].getEventWord() & (1 << 4)) {
    mCalibClocks[ilink] = localClock;
    eventType = EventType::Noise;
  } else if (localClock == mCalibClocks[ilink] + sDelayCalibToFET) {
    eventType = EventType::Dead;
  }
  auto firstEntry = mData.size();
  InteractionRecord intRec(localClock - sDelayBCToLocal, mBuffer.getRDH()->triggerOrbit);
  mData.push_back({mELinkDecoders[ilink].getStatusWord(), mELinkDecoders[ilink].getEventWord(), crateparams::makeUniqueLocID(mCrateId, mELinkDecoders[ilink].getId()), mELinkDecoders[ilink].getInputs()});
  mROFRecords.emplace_back(intRec, eventType, firstEntry, 1);
}

void CRUBareDecoder::addLoc(size_t ilink)
{
  /// Adds the local board to the output data vector
  addBoard(ilink);
  bool invert = (mROFRecords.back().eventType == EventType::Dead);
  for (int ich = 0; ich < 4; ++ich) {
    if ((mData.back().firedChambers & (1 << ich))) {
      mData.back().patternsBP[ich] = getPattern(mELinkDecoders[ilink].getPattern(0, ich), invert);
      mData.back().patternsNBP[ich] = getPattern(mELinkDecoders[ilink].getPattern(1, ich), invert);
    }
  }
  mELinkDecoders[ilink].reset();
}

uint16_t CRUBareDecoder::getPattern(uint16_t pattern, bool invert) const
{
  /// Gets the proper pattern
  return (invert) ? ~pattern : pattern;
}

bool CRUBareDecoder::isComplete() const
{
  /// Checks that all links have finished reading
  for (auto& elink : mELinkDecoders) {
    if (elink.getNBytes() > 0) {
      return false;
    }
  }
  return true;
}

} // namespace mid
} // namespace o2
