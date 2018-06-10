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
  via trapezoid. Described by the rise time and the flat-top.
  Risetime depends on the injected charge linearly: from mMaxRiseTime for 
  charge=0 to 0. for charge>mChargeRise0.
  Since for the small signals the duration increases, a fraction of the risetime is
  added to the flat-top duration
  The time is in nanoseconds

 */

#include "ITSMFTSimulation/AlpideSignalTrapezoid.h"
#include <TClass.h>
#include <cassert>

using namespace o2::ITSMFT;

//_________________________________________________________________
AlpideSignalTrapezoid::AlpideSignalTrapezoid(float duration, float rise, float qRise0)
{
  init(duration, rise, qRise0);
}

//_________________________________________________________________
void AlpideSignalTrapezoid::init(float dur, float rise, float qRise0)
{
  // init with new parameters
  mDuration = dur;
  mMaxRiseTime = rise;
  mChargeRise0 = qRise0;
  assert(mMaxRiseTime > 0.f && qRise0 > 1.f && mDuration > mMaxRiseTime);
  mChargeRise0Inv = 1. / mChargeRise0;
}

//_________________________________________________________________
void AlpideSignalTrapezoid::print() const
{
  ///< print parameters
  printf("%s | Duration: %.1f MaxRiseTime: %.1f (RiseTime=0 at q=%.1f) (ns)\n",
         Class()->GetName(), mDuration, mMaxRiseTime, mChargeRise0);
}
