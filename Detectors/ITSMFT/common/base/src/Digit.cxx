// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Digit.cxx
/// \brief Implementation of the ITSMFT digit

#include "ITSMFTBase/Digit.h"

ClassImp(o2::ITSMFT::Digit)

  using namespace o2::ITSMFT;

Digit::Digit() : FairTimeStamp(), mChipIndex(0), mRow(0), mCol(0), mCharge(0.), mLabels{ -1, -1, -1 } {}
Digit::Digit(UShort_t chipindex, UShort_t row, UShort_t col, Double_t charge, Double_t time)
  : FairTimeStamp(time), mChipIndex(chipindex), mRow(row), mCol(col), mCharge(charge), mLabels{ -1, -1, -1 }
{
}

Digit::~Digit() = default;

Digit& Digit::operator+=(const Digit& other)
{
  mCharge += other.mCharge;
  return *this;
}

const Digit Digit::operator+(const Digit& other)
{
  Digit result(*this);
  result += other;
  return result;
}

std::ostream& Digit::print(std::ostream& output) const
{
  output << "ITSMFT Digit of chip index [" << mChipIndex << "] and pixel [" << mRow << ',' << mCol << "] with charge "
         << mCharge << " at time stamp" << fTimeStamp;
  return output;
}
