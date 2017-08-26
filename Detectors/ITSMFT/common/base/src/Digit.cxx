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

ClassImp(o2::ITSMFT::Digit)

  using namespace o2::ITSMFT;

Digit::Digit() : FairTimeStamp(), mChipIndex(0), mRow(0), mCol(0), mCharge(0.f), mROFrame(0), mLabels{ -1, -1, -1 } {}

Digit::Digit(UShort_t chipindex, UInt_t frame, UShort_t row, UShort_t col, Float_t charge, Double_t time)
  : FairTimeStamp(time), mChipIndex(chipindex), mRow(row), mCol(col), mCharge(charge), mROFrame(0),mLabels{ -1, -1, -1 }
{
  setROFrame(frame);
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
         << mCharge << "ROFrame " << getROFrame() << "(" << getNOverflowFrames() << ") at time stamp" << fTimeStamp;
  return output;
}
