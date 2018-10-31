// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "PHOSBase/Digit.h"
#include <iostream>

using namespace o2::phos;

ClassImp(Digit);

constexpr int Digit::kMaxLabels;

Digit::Digit(Int_t absId, Double_t amplitude, Double_t time, Int_t label)
  : DigitBase(time), mAbsId(absId), mAmplitude(amplitude), mTime(time), mNlabels(0)
{
  if (label >= 0) {
    mLabels[0] = label; // sofar there is no lables, no need to to sort
    mEProp[0] = 1.;
    mNlabels = 1;
  }
}
Digit::Digit(Hit hit) : mAbsId(hit.GetDetectorID()), mAmplitude(hit.GetEnergyLoss()), mTime(hit.GetTime()), mNlabels(0)
{
  mLabels[0] = hit.GetTrackID(); // so far there is no lables, no need to to sort
  mEProp[0] = 1.;
  for (Int_t i = 1; i < kMaxLabels; i++) {
    mLabels[i] = -1;
    mEProp[i] = 0.;
  }
}
void Digit::FillFromHit(Hit hit)
{
  mAbsId = hit.GetDetectorID();
  mAmplitude = hit.GetEnergyLoss();
  mTime = hit.GetTime();
  if (hit.GetTrackID() >= 0) {
    mLabels[0] = hit.GetTrackID(); // so far there is no lables, no need to to sort
    mEProp[0] = 1.;
    mNlabels = 1;
  } else {
    mNlabels = 0;
  }
  for (Int_t i = 1; i < kMaxLabels; i++) {
    mLabels[i] = 0;
    mEProp[i] = 0.;
  }
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

  // Adds the amplitude of digits and completes the list of primary particles
  double scaleThis = mAmplitude / (mAmplitude + other.mAmplitude);
  double scaleOther = other.mAmplitude / (mAmplitude + other.mAmplitude);
  for (Int_t i = 0; i < mNlabels; i++) {
    mEProp[i] *= scaleThis;
  }
  if (other.mNlabels > 0) {
    // copy and scale EProp of other digit
    double otherEProp[kMaxLabels];
    for (Int_t i = 0; i < other.mNlabels; i++) {
      otherEProp[i] = scaleOther * other.mEProp[i];
    }
    double tmpEProp[kMaxLabels];
    Label tmpLabels[kMaxLabels];

    // Now find largest Energy Proportion
    int i1 = 0, i2 = 0, i = 0;
    while (i < kMaxLabels) {
      if (i1 >= mNlabels) {
        while (i2 < other.mNlabels) {
          tmpEProp[i] = otherEProp[i2];
          tmpLabels[i] = other.mLabels[i2];
          i++;
          i2++;
        }
        break;
      }
      if (i2 >= other.mNlabels) {
        while (i1 < mNlabels) {
          tmpEProp[i] = mEProp[i1];
          tmpLabels[i] = mLabels[i1];
          i++;
          i1++;
        }
        break;
      }

      if (mEProp[i1] > otherEProp[i2]) {
        tmpEProp[i] = mEProp[i1];
        tmpLabels[i] = mLabels[i1];
        i++;
        i1++;
      } else {
        tmpEProp[i] = otherEProp[i2];
        tmpLabels[i] = other.mLabels[i2];
        i++;
        i2++;
      }
    }
    // Copy to current digit
    for (Int_t ii = 0; ii < i; ii++) {
      mEProp[ii] = tmpEProp[ii];
      mLabels[ii] = tmpLabels[ii];
    }
  }

  mNlabels = std::min(kMaxLabels, other.mNlabels + mNlabels);

  // TODO: What about time? Should we assign time of more energetic digit? More complicated treatment?
  if (mAmplitude < other.mAmplitude) {
    mTime = other.mTime;
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
