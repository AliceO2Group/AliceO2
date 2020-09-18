// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/Raw/src/GBTUserLogicDecoder.cxx
/// \brief  MID GBT decoder with user logic zero suppression
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   15 April 2020

#include "MIDRaw/GBTUserLogicDecoder.h"

namespace o2
{
namespace mid
{

void GBTUserLogicDecoder::init(uint16_t feeId, bool isDebugMode)
{
  /// Initializes the task
  mOutputHandler.setFeeId(feeId);
  if (isDebugMode) {
    mOnDoneLoc = &GBTOutputHandler::onDoneLocDebug;
    mOnDoneReg = &GBTOutputHandler::onDoneRegDebug;
  }
}

void GBTUserLogicDecoder::process(gsl::span<const uint8_t> bytes, uint16_t bc, uint32_t orbit, uint16_t pageCnt)
{
  /// Decodes the buffer
  mOutputHandler.setIR(bc, orbit, pageCnt);

  bool isFeeding = false;
  for (auto& byte : bytes) {
    if (mELinkDecoder.getNBytes() == 0 && (byte & raw::sSTARTBIT) == 0) {
      // The e-link decoder is empty, meaning that we expect a new board.
      // The first byte of the board should have the STARTBIT on.
      // If this is not the case, it means that:
      // a) there was a problem in the decoding
      // b) we reached the end of the payload (and we have zeros until the end of the 256 bits word)
      // In both cases, we need to stop
      break;
    }
    mELinkDecoder.add(byte);
    if (mELinkDecoder.isComplete()) {
      if (raw::isLoc(mELinkDecoder.getStatusWord())) {
        std::invoke(mOnDoneLoc, mOutputHandler, mELinkDecoder.getId() % 8, mELinkDecoder);
      } else {
        size_t ilink = 8 + mELinkDecoder.getId() % 8;
        if (ilink > 9) {
          continue;
        }
        std::invoke(mOnDoneReg, mOutputHandler, 8 + mELinkDecoder.getId() % 8, mELinkDecoder);
      }
      mELinkDecoder.reset();
    }
  }
}
} // namespace mid
} // namespace o2
