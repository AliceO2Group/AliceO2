// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "DataFormatsEMCAL/Digit.h"
#include <iostream>

using namespace o2::emcal;

Digit::Digit(Short_t tower, Double_t energy, Double_t time)
  : DigitBase(time), mTower(tower), mEnergy(energy)
{
}

Digit& Digit::operator+=(const Digit& other)
{
  if (canAdd(other))
    mEnergy += other.getEnergy();
  // Does nothing if the digits are in different towers.
  return *this;
}

void Digit::PrintStream(std::ostream& stream) const
{
  stream << "EMCAL Digit: Tower " << mTower << ", Time " << getTimeStamp() << ", Energy " << mEnergy;
}

std::ostream& operator<<(std::ostream& stream, const Digit& digi)
{
  digi.PrintStream(stream);
  return stream;
}
