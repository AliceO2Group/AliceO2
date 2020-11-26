// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Digit.cxx
/// \brief Implementation of the ITSMFT digit

#include "DataFormatsITSMFT/Digit.h"
#include <iostream>

ClassImp(o2::itsmft::Digit);
ClassImp(o2::itsmft::DigitHW);

using namespace o2::itsmft;

Digit::Digit(UShort_t chipindex, UShort_t row, UShort_t col, Int_t charge)
  : mChipIndex(chipindex), mRow(row), mCol(col)
{
  setCharge(charge);
}

std::ostream& Digit::print(std::ostream& output) const
{
  output << "ITSMFTDigit chip [" << mChipIndex << "] R:" << mRow << " C:" << mCol << " Q: " << mCharge;
  return output;
}

DigitHW::DigitHW(UShort_t half, UShort_t disk, UShort_t plane, UShort_t zone, UShort_t cableHW, UShort_t chipindex, UShort_t row, UShort_t col, Int_t charge)
  : mHalf(half), mDisk(disk), mPlane(plane), mZone(zone), mCableHW(cableHW), Digit(chipindex,row,col,charge)
{

}

