// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file  CollisionTimeRecoTask.cxx
/// \brief Implementation of the FIT reconstruction task

#include "FT0Reconstruction/CollisionTimeRecoTask.h"
#include "FairLogger.h" // for LOG
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
//_____________________________________________________________________
CollisionTimeRecoTask::CollisionTimeRecoTask()
{
  // at the moment nothing.
}

*/

//_____________________________________________________________________
void CollisionTimeRecoTask::Process(const std::vector<o2::ft0::Digit>& digitsBC,
                                    const std::vector<o2::ft0::ChannelData>& digitsCh,
                                    RecPoints& recPoints) const
{

  LOG(INFO) << "Running reconstruction on new event";

  std::vector<o2::ft0::ChannelData> recCh;
  int first = recCh.size(), nStored = 0;

  Int_t ndigitsC = 0, ndigitsA = 0;
  constexpr Int_t nMCPsA = 4 * Geometry::NCellsA;
  constexpr Int_t nMCPsC = 4 * Geometry::NCellsC;
  constexpr Int_t nMCPs = nMCPsA + nMCPsC;
  Float_t sideAtime = 0, sideCtime = 0;

  auto timeStamp = o2::InteractionRecord::bc2ns(mIntRecord.bc, mIntRecord.orbit);

  LOG(INFO) << " event time " << timeStamp << " orbit " << mIntRecord.orbit << " bc " << mIntRecord.bc;

  int nbc =digitsBC.size();
  int itrig = 0;

  for (int ibc = 0; ibc < nbc; ibc++) {
    const auto& bcd = digitsBC[ibc];
    /*    if (bcd.Triggers > 0) {
      LOG(INFO) << "Triggered BC " << itrig++;
    }
    */
    bcd.print();
    //
    auto& channels = bcd.getBunchChannelData(digitsCh);
    int nch = channels.size();
    for (int ich = 0; ich < nch; ich++) {
      channels[ich].CFDTime *= 13;
      channels[ich].QTCAmpl /= Geometry::MV_2_Nchannels;
      LOG(DEBUG) << " mcp " << channels[ich].ChId << " cfd " << channels[ich].CFDTime << " amp " << channels[ich].QTCAmpl;
      recCh.emplace_back(channels[ich].ChId, int(channels[ich].CFDTime), int(channels[ich].QTCAmpl), channels[ich].numberOfParticles);
      nStored++;

      if (std::fabs(channels[ich].CFDTime) < 2000) {
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
    recPoints.emplace_back(const mCollisiontime,
                           vertex, first, nStored, iRec, chTrig);
  }
}
//________________________________________________________
void CollisionTimeRecoTask::FinishTask()
{
  // finalize digitization, if needed, flash remaining digits
  // if (!mContinuous)   return;
}
