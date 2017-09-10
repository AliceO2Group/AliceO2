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

Digit::Digit(UShort_t chipindex, UInt_t frame, UShort_t row, UShort_t col, Float_t charge, Double_t time)
  : FairTimeStamp(time), mChipIndex(chipindex), mRow(row), mCol(col), mCharge(charge), mROFrame(0)
{
  setROFrame(frame);
}

Digit::~Digit() = default;

Digit& Digit::operator+=(const Digit& other)
{
  mCharge += other.mCharge;
  // transfer labels
  int lbid=0;
  for (;lbid<maxLabels;lbid++) {
    if ( mLabels[lbid].isEmpty() ) break;
  }
  if (lbid<maxLabels) {
    for (int i=0;i<maxLabels;i++) {
      if ( other.mLabels[i].isEmpty() ) break; // all labels transferred
      mLabels[lbid++] = other.mLabels[i];
      if (lbid>=maxLabels) break; // no more room
    }
  }
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
  output << "ITSMFTDigit chip [" << mChipIndex << "] R:" << mRow << " C:" << mCol << " Q: "
         << mCharge << "ROFrame " << getROFrame() << "(" << getNOverflowFrames() << ") time " << fTimeStamp;
  for (int i=0;i<maxLabels;i++) {
    if ( mLabels[i].isEmpty() ) break;
    output << " Lb" << i << ' ' << mLabels[i];
  }
  
  return output;
}
