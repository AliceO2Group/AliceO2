// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include <fairlogger/Logger.h>

#include "DataFormatsPHOS/Digit.h"
#include "PHOSBase/Hit.h"
#include <iostream>

using namespace o2::phos;

ClassImp(Digit);

Digit::Digit(short absId, float amplitude, float time, int label)
  : DigitBase(time), mIsHighGain(true), mAbsId(absId), mLabel(label), mAmplitude(amplitude), mTime(time)
{
}
Digit::Digit(short truId, float amplitude, float time, bool isTrigger2x2, int /*dummy*/)
  : DigitBase(time), mIsHighGain(true), mAbsId(truId), mLabel(-1), mAmplitude(amplitude), mTime(time)
{
  setHighGain(isTrigger2x2);
}
Digit::Digit(const Hit& hit, int label) : mIsHighGain(true), mAbsId(hit.GetDetectorID()), mLabel(label), mAmplitude(hit.GetEnergyLoss()), mTime(hit.GetTime())
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
      // if Label indexes are different, something wrong
      LOG(error) << "Adding digits with different references to Labels:" << mLabel << " and " << other.mLabel;
    }
  }

  mAmplitude += other.mAmplitude;

  return *this;
}
void Digit::addEnergyTime(float energy, float time)
{
  // Adds the amplitude of digits
  // TODO: What about time? Should we assign time of more energetic digit? More complicated treatment?
  if (mAmplitude < energy) {
    mTime = time;
  }
  mAmplitude += energy;
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
