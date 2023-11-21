// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file PixelData.cxx
/// \brief Implementation for transient data of single pixel and set of pixels from current chip

#include "ITSMFTReconstruction/PixelData.h"
#include "ITSMFTBase/SegmentationAlpide.h"
#include "Framework/Logger.h"
#include <cassert>
#include <bitset>

using namespace o2::itsmft;

void PixelData::sanityCheck() const
{
  // make sure the mask used in this class are compatible with Alpide segmenations
  static_assert(RowMask + 1 >= o2::itsmft::SegmentationAlpide::NRows,
                "incompatible mask, does not match Alpide segmentations");
}

void ChipPixelData::print() const
{
  // print chip data
  std::bitset<4> flg(mROFlags);
  printf("Chip %d in Orbit %6d BC:%4d (ROFrame %d) ROFlags: 4b'%4s | %4lu hits\n", mChipID,
         mInteractionRecord.orbit, mInteractionRecord.bc, mROFrame, flg.to_string().c_str(), mPixels.size());
  for (int i = 0; i < mPixels.size(); i++) {
    printf("#%4d C:%4d R: %3d %s\n", i, mPixels[i].getCol(), mPixels[i].getRow(), mPixels[i].isMasked() ? "*" : "");
  }
}

std::string ChipPixelData::getErrorDetails(int pos) const
{
  // if possible, extract more detailed info about the error
  if (pos == int(ChipStat::RepeatingPixel)) {
    return fmt::format(": row{}/col{}", mErrorInfo & 0xffff, (mErrorInfo >> 16) & 0xffff);
  }
  if (pos == int(ChipStat::UnknownWord)) {
    std::string rbuf = ": 0x<";
    int nc = getNBytesInRawBuff();
    for (int i = 0; i < nc; i++) {
      rbuf += fmt::format(fmt::runtime(i ? " {:02x}" : "{:02x}"), (int)getRawErrBuff()[i]);
    }
    rbuf += '>';
    return rbuf;
  }
  return {};
}
