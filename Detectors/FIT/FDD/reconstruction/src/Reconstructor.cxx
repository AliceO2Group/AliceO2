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

/// \file  Reconstructor.cxx
/// \brief Implementation of the FDD reconstruction

#include "CommonDataFormat/InteractionRecord.h"
#include "FDDReconstruction/Reconstructor.h"
#include "FDDBase/Constants.h"
#include <DataFormatsFDD/ChannelData.h>
#include <DataFormatsFDD/Digit.h>
#include <DataFormatsFDD/FDDRecoConfig.h>
#include <fairlogger/Logger.h>

using namespace o2::fdd;

//_____________________________________________________________________
void Reconstructor::process(o2::fdd::Digit const& digitBC, gsl::span<const o2::fdd::ChannelData> inChData,
                            std::vector<o2::fdd::RecPoint>& RecPoints, std::vector<o2::fdd::ChannelDataFloat>& outChData)
// gsl::span<o2::fdd::ChannelDataFloat> outChData)
{
  // Compute charge weighted average time
  Double_t timeFDA = 0, timeFDC = 0;
  Double_t weightFDA = 0.0, weightFDC = 0.0;
  Int_t nInTimeA = 0, nInTimeC = 0;
  int firstEntry = outChData.size();
  int nStored = 0;
  int nch = inChData.size();

  bool atLeastOneVertexChannel = false;
  bool atLeastOneChannelInSwAmplitudeRange = false;

  for (int ich = 0; ich < nch; ich++) {
    if (mDeadChannelMap && !mDeadChannelMap->isChannelAlive(inChData[ich].mPMNumber)) {
      LOG(debug) << "Channel " << ich << " is dead - discarding data";
      continue;
    }
    bool inTime = inChData[ich].getFlag(ChannelData::EEventDataBit::kIsEventInTVDC);
    bool inAdcGate = inChData[ich].getFlag(ChannelData::EEventDataBit::kIsCFDinADCgate);
    if (inAdcGate) {
      outChData.emplace_back((int)inChData[ich].mPMNumber, (inChData[ich].mTime) * timePerTDC,
                             (double)inChData[ich].mChargeADC, inChData[ich].mFEEBits);
      nStored++;
    }

    float adc = inChData[ich].mChargeADC;
    float time = (inChData[ich].mTime) * timePerTDC;

    float timeWeight = 1;
    if (adc > FDDRecoConfig::Instance().AmplitudeCutOnCollisionTimeWeights) {
      timeWeight = adc * adc;
    }
    if ((int)inChData[ich].mPMNumber < 8) {
      nInTimeC += inTime;
      timeFDC += time * timeWeight;
      weightFDC += timeWeight;
    } else {
      nInTimeA += inTime;
      timeFDA += time * timeWeight;
      weightFDA += timeWeight;
    }

    atLeastOneVertexChannel = FDDRecoConfig::areChannelDataFlagsGood(inChData[ich].mFEEBits);
    atLeastOneChannelInSwAmplitudeRange = (adc >= FDDRecoConfig::SwTriggerAmplitudeLowThreshold && adc <= FDDRecoConfig::SwTriggerAmplitudeHighThreshold);
  }
  const int nsToPs = 1e3;
  std::array<int, 2> mCollisionTime = {o2::fdd::RecPoint::sDummyCollissionTime, o2::fdd::RecPoint::sDummyCollissionTime};

  float meanTimeSideA = round(timeFDA / weightFDA * nsToPs);
  float meanTimeSideC = round(timeFDC / weightFDC * nsToPs);
  float sideTimeDifference = fabs(meanTimeSideC - meanTimeSideA);
  bool swTriggerBit = false;
  swTriggerBit = (
    FDDRecoConfig::SwTriggerEnabled && 
    (sideTimeDifference => FDDRecoConfig::SwTriggerTimeLowThreshold 
      && sideTimeDifference <= FDDRecoConfig::SwTriggerTimeHighThreshold
      && atLeastOneVertexChannel 
      && atLeastOneChannelInSwAmplitudeRange
    )
  );

  if (nInTimeA > 0 && weightFDA > FDDRecoConfig::Instance().AmplitudeCollisionTimeCutSideA) {
    mCollisionTime[o2::fdd::RecPoint::TimeA] = round(timeFDA / weightFDA * nsToPs);
  }
  if (nInTimeC > 0 && weightFDC > FDDRecoConfig::Instance().AmplitudeCollisionTimeCutSideC) {
    mCollisionTime[o2::fdd::RecPoint::TimeC] = round(timeFDC / weightFDC * nsToPs);
  }
  RecPoints.emplace_back(mCollisionTime, firstEntry, nStored, digitBC.getIntRecord(), digitBC.mTriggers);
}
//________________________________________________________
void Reconstructor::finish()
{
  // finalize reconstruction, if needed, flash remaining recpoints
  // if (!mContinuous)   return;
}
