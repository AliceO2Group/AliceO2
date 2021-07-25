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

/// \file  BaseRecoTask.cxx
/// \brief Implementation of the FV0 reconstruction task

#include "FV0Reconstruction/BaseRecoTask.h"
#include "FairLogger.h" // for LOG
#include "DataFormatsFV0/RecPoints.h"
#include "FV0Base/Geometry.h"
#include "FV0Simulation/FV0DigParam.h"
#include "FV0Simulation/DigitizationConstant.h"
#include <DataFormatsFV0/ChannelData.h>
#include <DataFormatsFV0/BCData.h>
#include <cmath>
#include <bitset>
#include <cassert>
#include <iostream>
#include <algorithm>
#include <CommonDataFormat/InteractionRecord.h>
#include <Framework/Logger.h>

using namespace o2::fv0;
using RP = o2::fv0::RecPoints;

RP BaseRecoTask::process(o2::fv0::BCData const& bcd,
                         gsl::span<const o2::fv0::ChannelData> inChData,
                         gsl::span<o2::fv0::ChannelDataFloat> outChData)
{
  LOG(INFO) << "Running reconstruction on new event";

  Float_t sideAtimeFirst = 1e10;
  Int_t ndigitsA = 0;
  Float_t sideAtimeAvg = 0;
  Int_t ndigitsASelected = 0;
  Float_t sideAtimeAvgSelected = 0;

  auto timeStamp = o2::InteractionRecord::bc2ns(bcd.getIntRecord().bc, bcd.getIntRecord().orbit);

  LOG(INFO) << " event time " << timeStamp << " orbit " << bcd.getIntRecord().orbit << " bc " << bcd.getIntRecord().bc;

  int nch = inChData.size();
  const auto parInv = 1; // TODO: Check what value should be used. In FTO it was 16./7: DigitizationParameters::Instance().mMV_2_NchannelsInverse;
  for (int ich = 0; ich < nch; ich++) {
    LOG(INFO) << "  channel " << ich << " / " << nch;
    int offsetChannel = 0; // TODO: Not used until calibration is implemented. In FT0 it was: getOffset(ich, inChData[ich].QTCAmpl);

    outChData[ich] = o2::fv0::ChannelDataFloat{inChData[ich].pmtNumber,
                                               (inChData[ich].time - offsetChannel) * DigitizationConstant::TIME_PER_TDCCHANNEL,
                                               (double)inChData[ich].chargeAdc * parInv,
                                               0}; // Fill with ADC number once implemented

    //  only signals with amplitude participate in collision time
    if (outChData[ich].charge > 0) {
      sideAtimeFirst = std::min(static_cast<Double_t>(sideAtimeFirst), outChData[ich].time);
      sideAtimeAvg += outChData[ich].time;
      ndigitsA++;
    }
    const float chargeThreshold = 10; // TODO: move to digitization parameters or constants and adjust to reasonable value
    if (outChData[ich].charge > 0) {
      sideAtimeAvgSelected += outChData[ich].time;
      ndigitsASelected++;
    }
  }
  const int nsToPs = 1e3;
  std::array<short, 3> mCollisionTime = {RP::sDummyCollissionTime, RP::sDummyCollissionTime, RP::sDummyCollissionTime};
  mCollisionTime[RP::TimeFirst] = (ndigitsA > 0) ? round(sideAtimeFirst * nsToPs) : RP::sDummyCollissionTime;
  mCollisionTime[RP::TimeGlobalMean] = (ndigitsA > 0) ? round(sideAtimeAvg * nsToPs / Float_t(ndigitsA)) : RP::sDummyCollissionTime;
  mCollisionTime[RP::TimeSelectedMean] = (ndigitsASelected > 0) ? round(sideAtimeAvgSelected * nsToPs / Float_t(ndigitsASelected)) : RP::sDummyCollissionTime;

  return RecPoints{mCollisionTime, bcd.ref.getFirstEntry(), bcd.ref.getEntries(), bcd.getIntRecord(), bcd.mTriggers};
}
//______________________________________________________
void BaseRecoTask::FinishTask()
{
  // finalize digitization, if needed, flash remaining digits
  // if (!mContinuous)   return;
}
//______________________________________________________
/*int CollisionTimeRecoTask::getOffset(int channel, int amp)
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
*/
