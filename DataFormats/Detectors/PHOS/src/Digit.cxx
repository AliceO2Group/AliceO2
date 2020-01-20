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

#include "DataFormatsPHOS/Digit.h"
#include "PHOSBase/Hit.h"
#include <iostream>

using namespace o2::phos;

ClassImp(Digit);

Digit::Digit(short absId, float amplitude, float time, int label)
  : DigitBase(time), mAmplitude(amplitude), mTime(time), mAbsId(absId), mLabel(label)
{
}
Digit::Digit(const Hit& hit, int label) : mAbsId(hit.GetDetectorID()), mAmplitude(hit.GetEnergyLoss()), mTime(hit.GetTime()), mLabel(label)
{
}
void Digit::fillFromHit(const Hit& hit)
{
  mAbsId = hit.GetDetectorID();
  mAmplitude = hit.GetEnergyLoss();
  mTime = hit.GetTime();
}

bool Digit::canAdd(const Digit other) const
{
  return (mAbsId == other.getAbsId() && fabs(getTimeStamp() - other.getTimeStamp()) <= kTimeGate);
}

Digit& Digit::operator+=(const Digit& other)
{

  // Adds the amplitude of digits
  // TODO: What about time? Should we assign time of more energetic digit? More complicated treatment?
  if (mAmplitude < other.mAmplitude) {
    mTime = other.mTime;
  }

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
  stream << "PHOS Digit: cell absId " << mAbsId << ", Time " << getTimeStamp() << " with amplitude " << mAmplitude;
}

std::ostream& operator<<(std::ostream& stream, const Digit& digi)
{
  digi.PrintStream(stream);
  return stream;
}
