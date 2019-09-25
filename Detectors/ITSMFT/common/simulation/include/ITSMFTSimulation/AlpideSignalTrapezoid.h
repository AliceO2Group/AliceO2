// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file AlpideSignalTrapezoid.h
/// \brief Simple class describing ALPIDE signal time shape

#ifndef ALICEO2_ITSMFT_ALPIDESIGNALTRAPEZOID_H
#define ALICEO2_ITSMFT_ALPIDESIGNALTRAPEZOID_H

#include <Rtypes.h>

namespace o2
{
namespace itsmft
{

class AlpideSignalTrapezoid
{
 public:
  AlpideSignalTrapezoid(float duration = 7500., float rise = 1100., float qrise0 = 450.);
  AlpideSignalTrapezoid(const AlpideSignalTrapezoid&) = default;
  AlpideSignalTrapezoid& operator=(const AlpideSignalTrapezoid&) = default;
  ~AlpideSignalTrapezoid() = default;

  float getCollectedCharge(float totalNEle, float tMin, float tMax) const;

  float getDuration() const { return mDuration; }
  float getMaxRiseTime() const { return mMaxRiseTime; }
  float getChargeRise0() const { return mChargeRise0; }
  float getExtraDuration(float riseTime) const { return riseTime * 0.5; }

  // This method queried by digitizer to decided in home many ROFrames the hit can contribute
  // In case we describe extra duration at small charges, it should be accounted here
  float getMaxDuration() const { return getDuration(); }

  void setParameters(float dur, float rise, float qrise0)
  {
    init(dur, rise, qrise0);
  }
  void setDuration(float d) { init(d, mMaxRiseTime, mChargeRise0); }
  void setMaxRiseTime(float r) { init(mDuration, r, mChargeRise0); }
  void setChargeRise0(float q) { init(mDuration, mMaxRiseTime, q); }

  void print() const;

 private:
  void init(float dur, float rise, float qrise0);

  float mDuration = 7500.f;           ///< total duration in ns for signal above mChargeRise0
  float mMaxRiseTime = 1100.f;        ///< rise time in ns for smallest charge
  float mChargeRise0 = 450.f;         ///< charge at which rise time is ~0
  float mChargeRise0Inv = 1. / 450.f; ///< its inverse

  ClassDefNV(AlpideSignalTrapezoid, 1);
};

inline float AlpideSignalTrapezoid::getCollectedCharge(float totalNEle, float tMin, float tMax) const
{
  // calculate max number of electrons seen by the strobe from tMin to tMax (in nanosec),
  // provided that the total injected charge was totalNEle electrons

  // estimate rise time for given charge
  float riseTime = totalNEle > mChargeRise0 ? 0. : mMaxRiseTime * (1.f - totalNEle * mChargeRise0Inv);

  if (tMax >= riseTime && tMin <= mDuration) { // strobe overlaps flat top
    return totalNEle;
  }
  if (tMax > 0. && tMin < riseTime) { // strobe overlaps with rise
    return totalNEle * tMax / riseTime;
  }
  return 0;
}
} // namespace itsmft
} // namespace o2

#endif
