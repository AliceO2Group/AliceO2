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
namespace ITSMFT
{

class AlpideSignalTrapezoid
{
 public:
  AlpideSignalTrapezoid(float duration = 6000., float rise = 50., float decay = 30.);
  AlpideSignalTrapezoid(const AlpideSignalTrapezoid&) = default;
  AlpideSignalTrapezoid& operator=(const AlpideSignalTrapezoid&) = default;
  ~AlpideSignalTrapezoid() = default;

  float getIntegral(float totalNEle, float tMax, float tMin = 0) const;

  float getDuration() const { return mDuration; }
  float getRiseTime() const { return mRiseTime; }
  float getDecayTime() const { return mDecayTime; }
  float getFlatTopTime() const { return mFlatTopTime; }
  float getNormalization() const { return mNorm; }
  float getReducedTime(float t) const;

  void setParameters(float dur, float rise, float dec)
  {
    init(dur, rise, dec);
  }
  void setDuration(float d) { init(d, mRiseTime, mDecayTime); }
  void setRiseTime(float r) { init(mDuration, r, mDecayTime); }
  void setDecayTime(float d) { init(mDuration, mRiseTime, d); }

 private:
  void init(float dur, float rise, float decay);

  float mDuration = 0;
  float mRiseTime = 0;    ///< rise time in nanoseconts
  float mFlatTopTime = 0; ///< flat top duration
  float mDecayTime = 0;   ///< decay time in nanoseconts
  float mNorm = 1.f;      ///< normalization to 1
  // aux parameters
  float mRiseTimeH = 0.f;     ///< half rise time
  float mDecayTimeH = 0.f;    ///< half decay time
  float mRiseTimeInvH = 0.f;  ///< half inverse of the rise time
  float mDecayTimeInvH = 0.f; ///< half inverse of decay time

  ClassDefNV(AlpideSignalTrapezoid, 1);
};

inline float AlpideSignalTrapezoid::getReducedTime(float t) const
{
  // return time intervals needed for the effective area calculation
  float tef = 0.f;
  if (t < 0.)
    return 0.;
  if (t < mRiseTime) {
    return t * t * mRiseTimeInvH;
  }
  tef += mRiseTimeH;
  t -= mRiseTime;
  if (t < mFlatTopTime) {
    return tef + t;
  }
  t -= mFlatTopTime;
  tef += mFlatTopTime + mDecayTimeH;
  if (t < mDecayTime) {
    float resid = mDecayTime - t;
    tef -= resid * resid * mDecayTimeInvH;
  }
  return tef;
}

inline float AlpideSignalTrapezoid::getIntegral(float totalNEle, float tMax, float tMin) const
{
  // calculate number of electrons between time tMin and tMax (in nanosec)
  float g = totalNEle * (getReducedTime(tMax) - getReducedTime(tMin));
  return mNorm * g;
}
}
}

#endif
