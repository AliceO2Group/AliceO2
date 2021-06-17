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
#include "FT0Simulation/DigitizationParameters.h"
#include <DataFormatsFT0/ChannelData.h>
#include <DataFormatsFT0/Digit.h>
#include <cmath>
#include <bitset>
#include <cassert>
#include <iostream>
#include <CommonDataFormat/InteractionRecord.h>
#include <Framework/Logger.h>

using namespace o2::ft0;

o2::ft0::RecPoints CollisionTimeRecoTask::process(o2::ft0::Digit const& bcd,
                                                  gsl::span<const o2::ft0::ChannelData> inChData,
                                                  gsl::span<o2::ft0::ChannelDataFloat> outChData)
{
  LOG(DEBUG) << "Running reconstruction on new event";

  Int_t ndigitsC = 0, ndigitsA = 0;

  constexpr Int_t nMCPsA = 4 * Geometry::NCellsA;
  //  constexpr Int_t nMCPsC = 4 * Geometry::NCellsC;
  //  constexpr Int_t nMCPs = nMCPsA + nMCPsC;

  Float_t sideAtime = 0, sideCtime = 0;

  // auto timeStamp = o2::InteractionRecord::bc2ns(bcd.mIntRecord.bc, bcd.mIntRecord.orbit);

  // LOG(DEBUG) << " event time " << timeStamp << " orbit " << bcd.mIntRecord.orbit << " bc " << bcd.mIntRecord.bc;

  int nch = inChData.size();
  const auto parInv = DigitizationParameters::Instance().mMV_2_NchannelsInverse;
  for (int ich = 0; ich < nch; ich++) {
    int offsetChannel = getOffset(ich, inChData[ich].QTCAmpl);

    outChData[ich] = o2::ft0::ChannelDataFloat{inChData[ich].ChId,
                                               (inChData[ich].CFDTime - offsetChannel) * Geometry::ChannelWidth,
                                               (double)inChData[ich].QTCAmpl * parInv,
                                               inChData[ich].ChainQTC};

    //  only signals with amplitude participate in collision time
    if (outChData[ich].QTCAmpl > 0) {
      if (outChData[ich].ChId < nMCPsA) {
        sideAtime += outChData[ich].CFDTime;
        ndigitsA++;
      } else {
        sideCtime += outChData[ich].CFDTime;
        ndigitsC++;
      }
    }
  }
  std::array<Float_t, 4> mCollisionTime = {2 * o2::InteractionRecord::DummyTime,
                                           2 * o2::InteractionRecord::DummyTime,
                                           2 * o2::InteractionRecord::DummyTime,
                                           2 * o2::InteractionRecord::DummyTime};
  // !!!! tobe done::should be fix with ITS vertex
  mCollisionTime[TimeA] = (ndigitsA > 0) ? sideAtime / Float_t(ndigitsA) : 2 * o2::InteractionRecord::DummyTime;
  mCollisionTime[TimeC] = (ndigitsC > 0) ? sideCtime / Float_t(ndigitsC) : 2 * o2::InteractionRecord::DummyTime;

  if (ndigitsA > 0 && ndigitsC > 0) {
    mCollisionTime[Vertex] = (mCollisionTime[TimeA] - mCollisionTime[TimeC]) / 2.;
    mCollisionTime[TimeMean] = (mCollisionTime[TimeA] + mCollisionTime[TimeC]) / 2.;
  } else {
    mCollisionTime[TimeMean] = std::min(mCollisionTime[TimeA], mCollisionTime[TimeC]);
  }
  LOG(DEBUG) << " Collision time " << mCollisionTime[TimeA] << " " << mCollisionTime[TimeC] << " " << mCollisionTime[TimeMean] << " " << mCollisionTime[Vertex];
  return RecPoints{
    mCollisionTime, bcd.ref.getFirstEntry(), bcd.ref.getEntries(), bcd.mIntRecord, bcd.mTriggers};
}
//______________________________________________________
void CollisionTimeRecoTask::FinishTask()
{
  // finalize digitization, if needed, flash remaining digits
  // if (!mContinuous)   return;
}
//______________________________________________________
int CollisionTimeRecoTask::getOffset(int channel, int amp)
{
  if (!mCalibSlew || !mCalibOffset) {
    return 0;
  }
  int offsetChannel = mCalibOffset->mTimeOffsets[channel];
  TGraph& gr = mCalibSlew->at(channel);
  double slewoffset = gr.Eval(amp);
  LOG(DEBUG) << "@@@CollisionTimeRecoTask::getOffset(int channel, int amp) " << channel << " " << amp << " " << offsetChannel << " " << slewoffset;
  return offsetChannel + int(slewoffset);
}
