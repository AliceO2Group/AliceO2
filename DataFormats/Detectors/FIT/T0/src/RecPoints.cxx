// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "DataFormatsFITT0/RecPoints.h"
#include "T0Base/Geometry.h"
#include <cassert>
#include <iostream>
#include <CommonDataFormat/InteractionRecord.h>

using namespace o2::t0;

void RecPoints::fillFromDigits(const o2::t0::Digit& digit)
{
  mCollisionTime = {};

  Int_t ndigitsC = 0, ndigitsA = 0;
  constexpr Int_t nMCPsA = 4 * o2::t0::Geometry::NCellsA;
  constexpr Int_t nMCPsC = 4 * o2::t0::Geometry::NCellsC;
  constexpr Int_t nMCPs = nMCPsA + nMCPsC;
  Float_t sideAtime = 0, sideCtime = 0;

  mIntRecord = digit.getInteractionRecord();
  mEventTime = o2::InteractionRecord::bc2ns(mIntRecord.bc, mIntRecord.orbit);

  Float_t BCEventTime = 12.5;

  mTimeAmp = digit.getChDgData();
  for (auto& d : mTimeAmp) {
    d.CFDTime -= mEventTime /*- BCEventTime*/;
    if (abs(d.CFDTime - BCEventTime) < 2) {
      if (d.ChId < nMCPsA) {
        sideAtime += d.CFDTime;
        ndigitsA++;
      } else {
        sideCtime += d.CFDTime;
        ndigitsC++;
      }
    }
  }

  mCollisionTime[TimeA] = (ndigitsA > 0) ? sideAtime / Float_t(ndigitsA) : 2 * o2::InteractionRecord::DummyTime;
  mCollisionTime[TimeC] = (ndigitsC > 0) ? sideCtime / Float_t(ndigitsC) : 2 * o2::InteractionRecord::DummyTime;

  if (ndigitsA > 0 && ndigitsC > 0) {
    mVertex = (mCollisionTime[TimeA] - mCollisionTime[TimeC]) / 2.;
    mCollisionTime[TimeMean] = (mCollisionTime[TimeA] + mCollisionTime[TimeC]) / 2.;
  } else {
    mVertex = 0.;
    mCollisionTime[TimeMean] = std::min(mCollisionTime[TimeA], mCollisionTime[TimeC]);
  }
}
