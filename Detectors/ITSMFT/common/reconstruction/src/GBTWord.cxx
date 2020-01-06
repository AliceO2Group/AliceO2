// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

// \file GBTWord.cxx
// \brief Classes for creation/interpretation of ITS/MFT GBT data

#include "ITSMFTReconstruction/GBTWord.h"

using namespace o2::itsmft;

void GBTWord::printX(bool padded) const
{
  /// print in right aligned hex format, optionally padding to 128 bits
  int nw = padded ? 16 : 10;
  printf("0x:");
  for (int i = nw; i--;) {
    printf(" %02x", mData8[i]);
  }
  printf("\n");
}

void GBTWord::printB(bool padded) const
{
  /// print in bitset format, optionally padding to 128 bits
  int nw = padded ? 16 : 8;
  printf("0b:");
  for (int i = nw; i--;) {
    uint8_t v = mData8[i];
    printf(" ");
    for (int j = 8; j--;) {
      printf("%d", (v & (0x1 << j)) ? 1 : 0);
    }
  }
  printf("\n");
}
