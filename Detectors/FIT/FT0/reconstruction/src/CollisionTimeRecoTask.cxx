// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
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
#include "FT0Base/FT0DigParam.h"
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
  LOG(debug) << "Running reconstruction on new event";

  Int_t ndigitsC = 0, ndigitsA = 0;

  constexpr Int_t nMCPsA = 4 * Geometry::NCellsA;

  Float_t sideAtime = 0, sideCtime = 0;

  int nch = inChData.size();
  const auto parInv = FT0DigParam::Instance().mMV_2_NchannelsInverse;
  for (int ich = 0; ich < nch; ich++) {
    int offsetChannel = getOffset(int(inChData[ich].ChId), inChData[ich].QTCAmpl);
    outChData[ich] = o2::ft0::ChannelDataFloat{inChData[ich].ChId,
                                               (inChData[ich].CFDTime - offsetChannel) * Geometry::ChannelWidth,
                                               (float)inChData[ich].QTCAmpl,
                                               inChData[ich].ChainQTC};

    //  only signals with amplitude participate in collision time
    if (outChData[ich].QTCAmpl > FT0DigParam::Instance().mAmpThresholdForReco && std::abs(outChData[ich].CFDTime) < FT0DigParam::Instance().mTimeThresholdForReco) {
      if (outChData[ich].ChId < nMCPsA) {
        sideAtime += outChData[ich].CFDTime;
        ndigitsA++;
      } else {
        sideCtime += outChData[ich].CFDTime;
        LOG(debug) << "cfd " << outChData[ich].ChId << " dig " << 13.2 * inChData[ich].CFDTime << " rec " << outChData[ich].CFDTime << " amp " << (float)inChData[ich].QTCAmpl << " offset " << offsetChannel;
        ndigitsC++;
      }
    }
  }
  auto sDummyCollissionTime = o2::ft0::RecPoints::sDummyCollissionTime;
  std::array<short, 4> mCollisionTime = {sDummyCollissionTime, sDummyCollissionTime, sDummyCollissionTime, sDummyCollissionTime};
  // !!!! tobe done::should be fix with ITS vertex
  mCollisionTime[TimeA] = (ndigitsA > 0) ? sideAtime / ndigitsA : sDummyCollissionTime; // 2 * o2::InteractionRecord::DummyTime;
  mCollisionTime[TimeC] = (ndigitsC > 0) ? sideCtime / ndigitsC : sDummyCollissionTime; //2 * o2::InteractionRecord::DummyTime;

  if (ndigitsA > 0 && ndigitsC > 0) {
    mCollisionTime[Vertex] = (mCollisionTime[TimeA] - mCollisionTime[TimeC]) / 2.;
    mCollisionTime[TimeMean] = (mCollisionTime[TimeA] + mCollisionTime[TimeC]) / 2.;
  } else {
    mCollisionTime[TimeMean] = std::min(mCollisionTime[TimeA], mCollisionTime[TimeC]);
  }
  LOG(debug) << " Nch " << nch << " Collision time " << mCollisionTime[TimeA] << " " << mCollisionTime[TimeC] << " " << mCollisionTime[TimeMean] << " " << mCollisionTime[Vertex];
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
  if (!mCalibOffset) {
    return 0;
  }
  int offsetChannel = mCalibOffset->mTimeOffsets[channel];
  double slewoffset = 0;
  if (mCalibSlew) {
    TGraph& gr = mCalibSlew->at(channel);
    slewoffset = gr.Eval(amp);
  }
  LOG(debug) << "CollisionTimeRecoTask::getOffset(int channel, int amp) " << channel << " " << amp << " " << offsetChannel << " " << slewoffset;
  return offsetChannel + int(slewoffset);
}
