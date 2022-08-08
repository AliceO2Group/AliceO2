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
#include "DataFormatsFV0/RecPoints.h"
#include "FV0Base/Geometry.h"
#include "FV0Simulation/FV0DigParam.h"
#include "FV0Simulation/DigitizationConstant.h"
#include <DataFormatsFV0/ChannelData.h>
#include <DataFormatsFV0/Digit.h>
#include <CommonDataFormat/InteractionRecord.h>
#include <Framework/Logger.h>

using namespace o2::fv0;
using RP = o2::fv0::RecPoints;

RP BaseRecoTask::process(o2::fv0::Digit const& bcd,
                         gsl::span<const o2::fv0::ChannelData> inChData,
                         gsl::span<o2::fv0::ChannelDataFloat> outChData)
{
  LOG(debug) << "Running reconstruction on new event";

  Int_t ndigitsA = 0;
  Int_t ndigitsASelected = 0;
  Float_t sideAtimeFirst = 1e10;
  Float_t sideAtimeAvg = 0;
  Float_t sideAtimeAvgSelected = 0;

  auto timeStamp = o2::InteractionRecord::bc2ns(bcd.getIntRecord().bc, bcd.getIntRecord().orbit);

  LOG(debug) << " event time " << timeStamp << " orbit " << bcd.getIntRecord().orbit << " bc " << bcd.getIntRecord().bc;

  int nch = inChData.size();
  for (int ich = 0; ich < nch; ich++) {
    LOG(debug) << "  channel " << ich << " / " << nch;
    int offsetChannel = getOffset(int(inChData[ich].ChId));
    outChData[ich] = o2::fv0::ChannelDataFloat{inChData[ich].ChId,
                                               (inChData[ich].CFDTime - offsetChannel) * DigitizationConstant::TIME_PER_TDCCHANNEL,
                                               (float)inChData[ich].QTCAmpl,
                                               inChData[ich].ChainQTC};

    // Conditions for reconstructing collision time (3 variants: first, average-relaxed and average-tight)
    if (outChData[ich].charge > FV0DigParam::Instance().chargeThrForMeanTime) {
      sideAtimeFirst = std::min(static_cast<Double_t>(sideAtimeFirst), outChData[ich].time);
      if (inChData[ich].areAllFlagsGood()) {
        if (std::abs(outChData[ich].time) < FV0DigParam::Instance().mTimeThresholdForReco) {
          sideAtimeAvg += outChData[ich].time;
          ndigitsA++;
        }
        if (outChData[ich].charge > FV0DigParam::Instance().mAmpThresholdForReco && std::abs(outChData[ich].time) < FV0DigParam::Instance().mTimeThresholdForReco) {
          sideAtimeAvgSelected += outChData[ich].time;
          ndigitsASelected++;
        }
      }
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
int BaseRecoTask::getOffset(int channel)
{
  if (!mCalibOffset) {
    return 0;
  }
  int offsetChannel = mCalibOffset->mTimeOffsets[channel];
  return offsetChannel;
}
