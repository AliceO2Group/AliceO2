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

#include "CPVBase/Digit.h"
#include <iostream>

using namespace o2::cpv;

ClassImp(Digit);

Digit::Digit(int absId, float amplitude, float time, int label)
  : DigitBase(time), mAbsId(absId), mAmplitude(amplitude), mLabel(label)
{
}
Digit::Digit(Hit hit, int label) : mAbsId(hit.GetDetectorID()), mAmplitude(hit.GetEnergyLoss()), mLabel(label)
{
}
void Digit::FillFromHit(Hit hit)
{
  mAbsId = hit.GetDetectorID();
  mAmplitude = hit.GetEnergyLoss();
}

bool Digit::operator<(const Digit& other) const
{
  if (fabs(getTimeStamp() - other.getTimeStamp()) <= kTimeGate) {
    return getAbsId() < other.getAbsId();
  } else {
    return getTimeStamp() < other.getTimeStamp();
  }
}

bool Digit::operator>(const Digit& other) const
{
  if (fabs(getTimeStamp() - other.getTimeStamp()) <= kTimeGate) {
    return getAbsId() > other.getAbsId();
  } else {
    return getTimeStamp() > other.getTimeStamp();
  }
}

bool Digit::canAdd(const Digit other) const
{
  return (mAbsId == other.getAbsId() && fabs(getTimeStamp() - other.getTimeStamp()) <= kTimeGate);
}

Digit& Digit::operator+=(const Digit& other)
{
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
  stream << "CPV Digit: pad absId " << mAbsId << ",  with amplitude " << mAmplitude;
}

std::ostream& operator<<(std::ostream& stream, const Digit& digi)
{
  digi.PrintStream(stream);
  return stream;
}
