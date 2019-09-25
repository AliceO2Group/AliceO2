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

#include "ITSMFTBase/Digit.h"
#include <iostream>

ClassImp(o2::itsmft::Digit);

using namespace o2::itsmft;

Digit::Digit(UShort_t chipindex, UInt_t frame, UShort_t row, UShort_t col, Int_t charge)
  : mChipIndex(chipindex), mRow(row), mCol(col), mROFrame(0)
{
  setROFrame(frame);
  setCharge(charge);
}

std::ostream& Digit::print(std::ostream& output) const
{
  output << "ITSMFTDigit chip [" << mChipIndex << "] R:" << mRow << " C:" << mCol << " Q: " << mCharge << "ROFrame "
         << getROFrame();
  return output;
}
