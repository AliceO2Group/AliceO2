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
/// \brief Implementation of the FT0 reconstruction task

#include "FT0Reconstruction/CollisionTimeRecoTask.h"
#include <fairlogger/Logger.h> // for LOG
#include "DataFormatsFT0/RecPoints.h"
#include "FT0Base/Geometry.h"
#include "FT0Base/FT0DigParam.h"
#include <DataFormatsFT0/ChannelData.h>
#include <DataFormatsFT0/Digit.h>
#include <DataFormatsFT0/DigitFilterParam.h>
#include <DataFormatsFT0/CalibParam.h>
#include <cmath>
#include <bitset>
#include <cassert>
#include <iostream>
#include <CommonDataFormat/InteractionRecord.h>
#include <Framework/Logger.h>

using namespace o2::ft0;
using RP = o2::ft0::RecPoints;

void CollisionTimeRecoTask::processTF(const gsl::span<const o2::ft0::Digit>& digits,
                                      const gsl::span<const o2::ft0::ChannelData>& channels,
                                      std::vector<o2::ft0::RecPoints>& vecRecPoints,
                                      std::vector<o2::ft0::ChannelDataFloat>& vecChData)
{
  //  vecRecPoints.reserve(digits.size());
  //  vecChData.reserve(channels.size());
  for (const auto& digit : digits) {
    if (!ChannelFilterParam::Instance().checkTCMbits(digit.getTriggers().getTriggersignals())) {
      continue;
    }
    const auto channelsPerDigit = digit.getBunchChannelData(channels);
    vecRecPoints.emplace_back(processDigit(digit, channelsPerDigit, vecChData));
  }
}
RP CollisionTimeRecoTask::processDigit(const o2::ft0::Digit& digit,
                                       const gsl::span<const o2::ft0::ChannelData> inChData,
                                       std::vector<o2::ft0::ChannelDataFloat>& outChData)
{
  LOG(debug) << "Running reconstruction on new event";
  const int firstEntry = outChData.size();
  unsigned int ndigitsA = 0;
  unsigned int ndigitsC = 0;
  float sideAtime = 0;
  float sideCtime = 0;

  constexpr int nMCPsA = 4 * Geometry::NCellsA;
  const auto parInv = FT0DigParam::Instance().mMV_2_NchannelsInverse;

  int nch{0};
  for (const auto& channelData : inChData) {
    const float timeInPS = getTimeInPS(channelData);
    if (ChannelFilterParam::Instance().checkAll(channelData)) {
      outChData.emplace_back(channelData.ChId, timeInPS, (float)channelData.QTCAmpl, channelData.ChainQTC);
      nch++;
    }
    //  only signals with amplitude participate in collision time
    if (TimeFilterParam::Instance().checkAll(channelData)) {
      if (channelData.ChId < nMCPsA) {
        sideAtime += timeInPS;
        ndigitsA++;
      } else if (channelData.ChId < NCHANNELS) {
        sideCtime += timeInPS;
        ndigitsC++;
      }
    }
  }
  std::array<short, 4> mCollisionTime = {RP::sDummyCollissionTime, RP::sDummyCollissionTime, RP::sDummyCollissionTime, RP::sDummyCollissionTime};
  // !!!! tobe done::should be fix with ITS vertex
  mCollisionTime[TimeA] = (ndigitsA > 0) ? sideAtime / ndigitsA : RP::sDummyCollissionTime; // 2 * o2::InteractionRecord::DummyTime;
  mCollisionTime[TimeC] = (ndigitsC > 0) ? sideCtime / ndigitsC : RP::sDummyCollissionTime; // 2 * o2::InteractionRecord::DummyTime;

  if (ndigitsA > 0 && ndigitsC > 0) {
    mCollisionTime[Vertex] = (mCollisionTime[TimeA] - mCollisionTime[TimeC]) / 2.;
    mCollisionTime[TimeMean] = (mCollisionTime[TimeA] + mCollisionTime[TimeC]) / 2.;
  } else {
    mCollisionTime[TimeMean] = std::min(mCollisionTime[TimeA], mCollisionTime[TimeC]);
  }
  return RecPoints{mCollisionTime, firstEntry, nch, digit.mIntRecord, digit.mTriggers};
}
//______________________________________________________
void CollisionTimeRecoTask::FinishTask()
{
  // finalize digitization, if needed, flash remaining digits
  // if (!mContinuous)   return;
}

float CollisionTimeRecoTask::getTimeInPS(const o2::ft0::ChannelData& channelData)
{
  float offsetChannel{0};
  float slewoffset{0};
  if (mTimeCalibObject && channelData.ChId < NCHANNELS) {
    // Temporary, will be changed to status bit checking
    // Check statistics
    const auto& stat = mTimeCalibObject->mTime[channelData.ChId].mStat;
    const bool isEnoughStat = stat > CalibParam::Instance().mMaxEntriesThreshold;
    const bool isNotGoogStat = stat > CalibParam::Instance().mMinEntriesThreshold && !isEnoughStat;
    // Check fit quality
    const auto& meanGaus = mTimeCalibObject->mTime[channelData.ChId].mGausMean;
    const auto& meanHist = mTimeCalibObject->mTime[channelData.ChId].mStatMean;
    const auto& sigmaGaus = mTimeCalibObject->mTime[channelData.ChId].mGausRMS;
    const auto& rmsHist = mTimeCalibObject->mTime[channelData.ChId].mStatRMS;
    const bool isGoodFitResult = (mTimeCalibObject->mTime[channelData.ChId].mStatusBits & 1) > 0;
    const bool isBadFit = std::abs(meanGaus - meanHist) > CalibParam::Instance().mMaxDiffMean || rmsHist < CalibParam::Instance().mMinRMS || sigmaGaus > CalibParam::Instance().mMaxSigma;

    if (isEnoughStat && isGoodFitResult && !isBadFit) {
      offsetChannel = meanGaus;
    } else if ((isNotGoogStat || isEnoughStat) && isBadFit) {
      offsetChannel = meanHist;
    }
  }
  /*
  if (mCalibSlew  && channelData.ChId < NCHANNELS) {
    TGraph& gr = mCalibSlew->at(channelData.ChId);
    slewoffset = gr.Eval(channelData.QTCAmpl);
  }
  */
  const float globalOffset = (offsetChannel + slewoffset) * Geometry::ChannelWidth;
  return float(channelData.CFDTime) * Geometry::ChannelWidth - globalOffset;
}
