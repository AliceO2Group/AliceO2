// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "FairLogger.h"

#include "DataFormatsCPV/Digit.h"
#include "CPVBase/Hit.h"
#include <iostream>

using namespace o2::cpv;

ClassImp(Digit);

Digit::Digit(unsigned short absId, float amplitude, int label)
  : DigitBase(0), mAmplitude(amplitude), mAbsId(absId), mLabel(label)
{
}
Digit::Digit(const Hit& hit, int label) : mAbsId(hit.GetDetectorID()), mAmplitude(hit.GetEnergyLoss()), mLabel(label)
{
}
void Digit::fillFromHit(const Hit& hit)
{
  mAbsId = hit.GetDetectorID();
  mAmplitude = hit.GetEnergyLoss();
}

bool Digit::canAdd(const Digit other) const
{
  return (mAbsId == other.getAbsId() && fabs(getTimeStamp() - other.getTimeStamp()) <= kTimeGate);
}

Digit& Digit::operator+=(const Digit& other)
{

  // Adds the amplitude of digits
  if (mLabel == -1) {
    mLabel = other.mLabel;
  } else {
    if (mLabel != other.mLabel && other.mLabel != -1) {
      //if Label indexes are different, something wrong
      LOG(ERROR) << "Adding digits with different references to Labels:" << mLabel << " and " << other.mLabel;
    }
  }

  mAmplitude += other.mAmplitude;

  return *this;
}

void Digit::PrintStream(std::ostream& stream) const
{
  stream << "CPV Digit: cell absId " << mAbsId << ", Time " << getTimeStamp() << " with amplitude " << mAmplitude;
}

std::ostream& operator<<(std::ostream& stream, const Digit& digi)
{
  digi.PrintStream(stream);
  return stream;
}
