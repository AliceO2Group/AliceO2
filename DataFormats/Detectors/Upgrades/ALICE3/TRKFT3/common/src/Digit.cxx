// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "DataFormatsTRKFT3/Digit.h"
#include <iostream>

ClassImp(o2::trkft3::Digit);

using namespace o2::trkft3;

Digit::Digit(UShort_t chipindex, UShort_t row, UShort_t col, Int_t charge)
  : mChipIndex(chipindex), mRow(row), mCol(col)
{
  setCharge(charge);
}

std::ostream& Digit::print(std::ostream& output) const
{
  output << "TRKFT3Digit chip [" << mChipIndex << "] R:" << mRow << " C:" << mCol << " Q: " << mCharge;
  return output;
}
