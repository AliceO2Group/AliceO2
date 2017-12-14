// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <iostream>
#include <TMath.h>
#include "EMCALBase/Digit.h"

using namespace o2::EMCAL;

ClassImp(Digit)

Digit::Digit(Int_t tower, Double_t amplitude, Double_t time):
FairTimeStamp(time),
mTower(tower),
mAmplitude(amplitude)
{
}

bool Digit::operator<(const Digit &other) const {
  return GetTimeStamp() < other.GetTimeStamp();
}

const Digit Digit::operator+(const Digit& other) {
  Digit result(*this);
  result += other;
  return result;
}

Digit& Digit::operator+=(const Digit& other) {
  if(mTower==other.GetTower() && TMath::Abs(GetTimeStamp()-other.GetTimeStamp())<=100) mAmplitude += other.GetAmplitude();
  // Does nothing if the digits are in different towers.
  return *this;
}

void Digit::PrintStream(std::ostream &stream) const {
  stream << "EMCAL Digit: Tower " << mTower << ", Time " << GetTimeStamp() << " with amplitude " << mAmplitude;
}

std::ostream& operator<<(std::ostream &stream, const Digit &digi){
  digi.PrintStream(stream);
  return stream;
}
