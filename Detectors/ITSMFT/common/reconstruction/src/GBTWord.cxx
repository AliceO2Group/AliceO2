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

// \file GBTWord.cxx
// \brief Classes for creation/interpretation of ITS/MFT GBT data

#include "ITSMFTReconstruction/GBTWord.h"
#include "Framework/Logger.h"
#include <sstream>

using namespace o2::itsmft;

void GBTWord::printX(bool padded) const
{
  /// print in right aligned hex format, optionally padding to 128 bits
  if (padded) {
    LOGF(info, "0x: %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x",
         data8[15], data8[14], data8[13], data8[12], data8[11], data8[10],
         data8[9], data8[8], data8[7], data8[6], data8[5], data8[4], data8[3], data8[2], data8[1], data8[0]);
  } else {
    LOGF(info, "0x: %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x",
         data8[9], data8[8], data8[7], data8[6], data8[5], data8[4], data8[3], data8[2], data8[1], data8[0]);
  }
}

void GBTWord::printB(bool padded) const
{
  /// print in bitset format, optionally padding to 128 bits
  int nw = padded ? GBTPaddedWordLength : GBTWordLength;
  std::stringstream ss;
  for (int i = nw; i--;) {
    uint8_t v = data8[i];
    ss << ' ';
    for (int j = 8; j--;) {
      ss << ((v & (0x1 << j)) ? '1' : '0');
    }
  }
  LOGF(info, "0b: %s", ss.str());
}
