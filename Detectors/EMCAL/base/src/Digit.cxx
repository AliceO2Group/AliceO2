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
#include "EMCALBase/Digit.h"

using namespace o2::EMCAL;

ClassImp(Digit)

Digit::Digit(Int_t module, Int_t tower, Double_t amplitude, Double_t time):
FairTimeStamp(time),
mModule(module),
mTower(tower),
mAmplitude(amplitude)
{
}

bool Digit::operator<(const Digit &other) const {
  return GetTimeStamp() < other.GetTimeStamp();
}

void Digit::PrintStream(std::ostream &stream) const {
  stream << "EMCAL Digit: Module " << mModule <<", Tower " << mTower << ", Time " << GetTimeStamp() << " wiht amplitude " << mAmplitude;
}

std::ostream& operator<<(std::ostream &stream, const Digit &digi){
  digi.PrintStream(stream);
  return stream;
}
