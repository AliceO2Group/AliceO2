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
#include <DataFormatsFT0/ChannelData.h>
#include <DataFormatsFT0/Digit.h>
#include <cmath>
#include <cassert>
#include <iostream>
#include <CommonDataFormat/InteractionRecord.h>
#include <Framework/Logger.h>

using namespace o2::ft0;
/*
void RecPoints::fillFromDigits(std::vector<o2::ft0::BCData>& digitsBC,
                               std::vector<o2::ft0::ChannelData>& digitsCh)
{
  mCollisionTime = {};

  Int_t ndigitsC = 0, ndigitsA = 0;
  constexpr Int_t nMCPsA = 4 * Geometry::NCellsA;
  constexpr Int_t nMCPsC = 4 * Geometry::NCellsC;
  constexpr Int_t nMCPs = nMCPsA + nMCPsC;
  Float_t sideAtime = 0, sideCtime = 0;

  //  mIntRecord = digit.getInteractionRecord();
  mTimeStamp = o2::InteractionRecord::bc2ns(mIntRecord.bc, mIntRecord.orbit);

  LOG(INFO) << " event time " << mTimeStamp << " orbit " << mIntRecord.orbit << " bc " << mIntRecord.bc;

  int nbc = ft0BCData.size();
    LOG(INFO) << "Entry " << ient << " : " << nbc << " BCs stored";
    int itrig = 0;

    for (int ibc = 0; ibc < nbc; ibc++) {
      const auto& bcd = ft0BCData[ibc];
      if (bcd.triggers >0) {
        LOG(INFO) << "Triggered BC " << itrig++;
      }
      bcd.print();
      //
      auto channels = bcd.getBunchChannelData(ft0ChData);
      int nch = channels.size();
      for (int ich = 0; ich < nch; ich++) {
        channels[ich].CFDTime *= 13;
        channels[ich].QTCAmpl /= Geometry::MV_2_Nchannels;
        LOG(DEBUG) << " mcp " << channels[ich].ChId << " cfd " << channels[ich].CFDTime << " amp " << channels[ich].QTCAmpl;
        chDgDataArr.emplace_back(o2::ft0::ChannelData{channels[ich].ChId, int(channels[ich].CFDTime), int(channels[ich].QTCAmpl), channels[ich].numberOfParticles});

        if (std::fabs( channels[ich].CFDTime) < 2000) {
      if (channels[ich].ChId < nMCPsA) {
        sideAtime += channels[ich].CFDTime;
        ndigitsA++;
      } else {
        sideCtime += channels[ich].CFDTime;
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
*/
gsl::span<const ChannelData> RecPoints::getBunchChannelData(const gsl::span<const ChannelData> tfdata) const
{
  // extract the span of channel data for this bunch from the whole TF data
  return gsl::span<const ChannelData>(&tfdata[ref.getFirstEntry()], ref.getEntries());
}
