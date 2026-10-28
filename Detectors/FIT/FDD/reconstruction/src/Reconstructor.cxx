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

  int64_t timeSideA = 0;
  int64_t timeSideC = 0;
  int64_t entriesNumSideA = 0;
  int64_t entriesNumSideC = 0;
  uint32_t storedEntries = 0;

  bool swOrGateSideA = false;
  bool swOrGateSideC = false;

  for (const auto& channelData: inChData) {
    if (mDeadChannelMap && !mDeadChannelMap->isChannelAlive(channelData.mPMNumber)) {
      LOG(debug) << "Channel " << channelData.mPMNumber << " is dead - discarding data";
      continue;
    }

    if (isInAmplitudeWindow(channelData.mChargeADC) 
      && isInTimeWindow(channelData.mTime) 
      && isChannelDataGood(channelData.mFEEBits))
    {
      outChData.emplace_back((int)inChData[ich].mPMNumber, (inChData[ich].mTime) * timePerTDC,
                             (double)inChData[ich].mChargeADC, inChData[ich].mFEEBits);
      storedEntries++;
    }
    
    if (FDDRecoConfig::Instance().SwTriggerEnabled) {
      if(channelData.mPMNumber <= sSideAMaxId) {
        swOrGateSideA |= checkSwOrGateConditions(channelData);
      } else {
        swOrGateSideC |= checkSwOrGateConditions(channelData);
      }
    }

    if (!isInAmplitudeWindow(channelData.mChargeADC) 
      || !isInTimeWindow(channelData.mTime) 
      || !isChannelDataGood(channelData.mFEEBits)) {
        continue;
      }
    
    if(channelData.mPMNumber <= sSideAMaxId) {
      timeSideA += channelData.mTime;
      entriesNumSideA++;
    } else {
      timeSideC += channelData.mTime;
      entriesNumSideC++;
    }
  }

  constexpr float nsToPs = 1e3;
  float meanTimeSideA = std::round(float(timeSideA) * timePerTDC / float(entriesNumSideA));
  float meanTimeSideC = std::round(float(timeSideC) * timePerTDC / float(entriesNumSideC));

  bool swVertexTrg = FDDRecoConfig::Instance().SwTriggerEnabled ? calculateSwTrigger(meanTimeSideA, meanTimeSideC, swOrGateSideA, swOrGateSideC) : false;

  std::array<int, 2> mCollisionTime = {o2::fdd::RecPoint::sDummyCollissionTime, o2::fdd::RecPoint::sDummyCollissionTime};
  if (entriesSideA > 0) {
    mCollisionTime[o2::fdd::RecPoint::TimeA] = meanTimeSideA * nsToPs;
  }
  if (entriesSideC > 0) {
    mCollisionTime[o2::fdd::RecPoint::TimeC] = meanTimeSideC * nsToPs;
  }
  RecPoints.emplace_back(mCollisionTime, firstEntry, nStored, digitBC.getIntRecord(), digitBC.mTriggers);
}
//________________________________________________________
void Reconstructor::finish()
{
  // finalize reconstruction, if needed, flash remaining recpoints
  // if (!mContinuous)   return;
}
