// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/Raw/src/GBTBareDecoder.cxx
/// \brief  MID GBT decoder without user logic
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   12 March 2020

#include "MIDRaw/GBTBareDecoder.h"

namespace o2
{
namespace mid
{

void GBTBareDecoder::init(uint16_t feeId, uint8_t mask, bool isDebugMode)
{
  /// Initializes the task
  mOutputHandler.setFeeId(feeId);
  mMask = mask;
  if (isDebugMode) {
    mOnDoneLoc = &GBTOutputHandler::onDoneLocDebug;
    mProcessReg = &GBTBareDecoder::processRegDebug;
  }
}

void GBTBareDecoder::process(gsl::span<const uint8_t> bytes, uint16_t bc, uint32_t orbit, uint16_t pageCnt)
{
  /// Decodes the buffer
  mOutputHandler.setIR(bc, orbit, pageCnt);

  uint8_t byte = 0;
  size_t ilink = 0, linkMask = 0, byteOffset = 0;

  for (int ireg = 0; ireg < 2; ++ireg) {
    byteOffset = 5 * ireg;
    ilink = 8 + ireg;
    linkMask = (1 << ilink);
    for (size_t idx = byteOffset + 4; idx < bytes.size(); idx += 16) {
      byte = bytes[idx];
      if ((mIsFeeding & linkMask) || byte) {
        std::invoke(mProcessReg, this, ilink, byte);
      }
    }
    for (int ib = 0; ib < 4; ++ib) {
      ilink = ib + 4 * ireg;
      linkMask = (1 << ilink);
      if (mMask & linkMask) {
        for (size_t idx = byteOffset + ib; idx < bytes.size(); idx += 16) {
          byte = bytes[idx];
          if ((mIsFeeding & linkMask) || byte) {
            processLoc(ilink, byte);
          }
        }
      }
    }
  }
}

void GBTBareDecoder::processLoc(size_t ilink, uint8_t byte)
{
  /// Processes the local board information
  if (mELinkDecoders[ilink].getNBytes() > 0) {
    mELinkDecoders[ilink].add(byte);
    if (mELinkDecoders[ilink].isComplete()) {
      std::invoke(mOnDoneLoc, mOutputHandler, ilink, mELinkDecoders[ilink]);
      mELinkDecoders[ilink].reset();
      mIsFeeding &= (~(1 << ilink));
    }
  } else if ((byte & (raw::sSTARTBIT | raw::sCARDTYPE)) == (raw::sSTARTBIT | raw::sCARDTYPE)) {
    mELinkDecoders[ilink].add(byte);
    mIsFeeding |= (1 << ilink);
  }
}

void GBTBareDecoder::processRegDebug(size_t ilink, uint8_t byte)
{
  /// Processes the regional board information in debug mode
  if (mELinkDecoders[ilink].getNBytes() > 0) {
    mELinkDecoders[ilink].add(byte);
    if (mELinkDecoders[ilink].isComplete()) {
      mOutputHandler.onDoneRegDebug(ilink, mELinkDecoders[ilink]);
      mELinkDecoders[ilink].reset();
      mIsFeeding &= (~(1 << ilink));
    }
  } else if (byte & raw::sSTARTBIT) {
    mELinkDecoders[ilink].add(byte);
    mIsFeeding |= (1 << ilink);
  }
}

bool GBTBareDecoder::isComplete() const
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
