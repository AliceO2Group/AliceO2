// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "DataFormatsFT0/RecPoints.h"
#include "FT0Base/Geometry.h"
#include "DataFormatsFT0/Digit.h"
#include <cmath>
#include <cassert>
#include <iostream>
#include <CommonDataFormat/InteractionRecord.h>
#include <Framework/Logger.h>

using namespace o2::ft0;

void RecPoints::fillFromDigits(const o2::ft0::Digit& digit)
{
  mCollisionTime = {};

  Int_t ndigitsC = 0, ndigitsA = 0;
  constexpr Int_t nMCPsA = 4 * Geometry::NCellsA;
  constexpr Int_t nMCPsC = 4 * Geometry::NCellsC;
  constexpr Int_t nMCPs = nMCPsA + nMCPsC;
  Float_t sideAtime = 0, sideCtime = 0;

  std::vector<o2::ft0::ChannelData> chDgDataArr;

  mIntRecord = digit.getInteractionRecord();
  mEventTime = o2::InteractionRecord::bc2ns(mIntRecord.bc, mIntRecord.orbit);
  LOG(INFO) << " event time " << mEventTime << " orbit " << mIntRecord.orbit << " bc " << mIntRecord.bc;
  mTimeAmp = digit.getChDgData();
  for (auto& d : mTimeAmp) {
    d.CFDTime *= 13;
    d.QTCAmpl /= Geometry::MV_2_Nchannels;
    LOG(DEBUG) << " mcp " << d.ChId << " cfd " << d.CFDTime << " amp " << d.QTCAmpl;
    chDgDataArr.emplace_back(o2::ft0::ChannelData{d.ChId, int(d.CFDTime), int(d.QTCAmpl), d.numberOfParticles});
    setChDgData(std::move(chDgDataArr));
    if (std::fabs(d.CFDTime) < 2000) {
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
  LOG(INFO) << " coll time " << mCollisionTime[TimeMean];
}
