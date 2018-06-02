// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file AlpideSignalTrapezoid.cxx
/// \brief Implementation of the ALPIDE signal time shape as trapezoid

/*

  This is a simple implementation of the ALPIDE signal time shape
  via trapezoid. Described by the amplitude, rise time, flat-top duration
  and decay time. The integral is equal to total input charge.
  The time is in nanoseconds

 */

#include "ITSMFTSimulation/AlpideSignalTrapezoid.h"
#include <cassert>

using namespace o2::ITSMFT;

AlpideSignalTrapezoid::AlpideSignalTrapezoid(float duration, float rise, float decay)
{
  init(duration, rise, decay);
}

void AlpideSignalTrapezoid::init(float dur, float rise, float decay)
{
  // init with new parameters
  mDuration = dur;
  mRiseTime = rise;
  mDecayTime = decay;
  mFlatTopTime = dur - rise - decay;
  assert(mRiseTime >= 0.f && mDecayTime >= 0. && mFlatTopTime > 0.);
  mNorm = 1.f / (mFlatTopTime + 0.5f * (mRiseTime + mDecayTime));
  mRiseTimeH = 0.5 * mRiseTime;
  mDecayTimeH = 0.5 * mDecayTime;
  mRiseTimeInvH = mRiseTime > 0.f ? 0.5f / mRiseTime : 0.f;
  mDecayTimeInvH = mDecayTime > 0.f ? 0.5f / mDecayTime : 0.f;
}
