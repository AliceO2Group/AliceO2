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
  const int firstEntry = outChData.size();

  bool atLeastOneVertexChannel = false;
  bool atLeastOneChannelInSwAmplitudeRange = false;

  int64_t timeSideA = 0;
  int64_t timeSideC = 0;
  int64_t entriesNumSideA = 0;
  int64_t entriesNumSideC = 0;
  uint32_t storedEntries = 0;

  bool swOrGateSideA = false;
  bool swOrGateSideC = false;
  int64_t swTimeSideA = 0;
  int64_t swTimeSideC = 0;
  int64_t swEntriesNumSideA = 0;
  int64_t swEntriesNumSideC = 0;

  bool coincidencePairsOrTrg[Nchannels] = {false};
  int64_t coincidencePairsTime[Nchannels] = {0};

  for (const auto& channelData : inChData) {
    if (mDeadChannelMap && !mDeadChannelMap->isChannelAlive(channelData.mPMNumber)) {
      LOG(debug) << "Channel " << channelData.mPMNumber << " is dead - discarding data";
      continue;
    }
    if (!isInAmplitudeWindow(channelData.mChargeADC) || !isInTimeWindow(channelData.mTime) || !isChannelDataGood(channelData.mFEEBits)) {
      continue;
    }
    const auto& chNum = channelData.mPMNumber;

    outChData.emplace_back((int)chNum, (channelData.mTime) * timePerTDC,
                           (double)channelData.mChargeADC, channelData.mFEEBits);
    storedEntries++;

    if (FDDRecoConfig::Instance().SwTriggerEnabled) {
      coincidencePairsOrTrg[chNum] |= checkSwOrGateConditions(channelData);
      coincidencePairsTime[chNum] = channelData.mTime;
      if (FDDRecoConfig::Instance().SwTriggerCoincidenceMode && coincidencePairsOrTrg[chNum] && coincidencePairsOrTrg[coincidencePairs[chNum]]) {
        if (chNum >= sSideAStartId) {
          swOrGateSideA = true;
          swTimeSideA += channelData.mTime + coincidencePairsTime[coincidencePairs[chNum]];
          swEntriesNumSideA += 2;
        } else {
          swOrGateSideC = true;
          swTimeSideC += channelData.mTime + coincidencePairsTime[coincidencePairs[chNum]];
          swEntriesNumSideC += 2;
        }
      } else if (!FDDRecoConfig::Instance().SwTriggerCoincidenceMode && coincidencePairsOrTrg[chNum]) {
        if (chNum >= sSideAStartId) {
          swOrGateSideA = true;
          swTimeSideA += channelData.mTime;
          swEntriesNumSideA++;
        } else {
          swOrGateSideC = true;
          swTimeSideC += channelData.mTime;
          swEntriesNumSideC++;
        }
      }
    }

    if (!isChannelDataGoodForCollisionTime(channelData.mFEEBits) || !isInCollisionTimeAmplitudeWindow(channelData.mChargeADC) || !isInCollisionTimeTimeWindow(channelData.mTime)) {
      continue;
    }

    if (channelData.mPMNumber >= sSideAStartId) {
      timeSideA += channelData.mTime;
      entriesNumSideA++;
    } else {
      timeSideC += channelData.mTime;
      entriesNumSideC++;
    }
  }

  constexpr float nsToPs = 1e3;
  float meanTimeSideA = entriesNumSideA > 0 ? float(timeSideA) * timePerTDC / float(entriesNumSideA) : 0;
  float meanTimeSideC = entriesNumSideC > 0 ? float(timeSideC) * timePerTDC / float(entriesNumSideC) : 0;

  bool swVertexTrg = false;
  if (FDDRecoConfig::Instance().SwTriggerEnabled) {
    float swMeanTimeSideA = swEntriesNumSideA > 0 ? float(swTimeSideA) * timePerTDC / float(swEntriesNumSideA) : 0;
    float swMeanTimeSideC = swEntriesNumSideC > 0 ? float(swTimeSideC) * timePerTDC / float(swEntriesNumSideC) : 0;
    swVertexTrg = calculateSwTrigger(swMeanTimeSideA, swMeanTimeSideC, swOrGateSideA, swOrGateSideC);
  }

  std::array<int, 2> mCollisionTime = {o2::fdd::RecPoint::sDummyCollissionTime, o2::fdd::RecPoint::sDummyCollissionTime};
  if (entriesNumSideA > 0) {
    mCollisionTime[o2::fdd::RecPoint::TimeA] = std::round(meanTimeSideA * nsToPs);
  }
  if (entriesNumSideC > 0) {
    mCollisionTime[o2::fdd::RecPoint::TimeC] = std::round(meanTimeSideC * nsToPs);
  }
  RecPoints.emplace_back(mCollisionTime, firstEntry, storedEntries, digitBC.getIntRecord(), digitBC.mTriggers, swOrGateSideA, swOrGateSideC, swVertexTrg);
}
//________________________________________________________
void Reconstructor::finish()
{
  // finalize reconstruction, if needed, flash remaining recpoints
  // if (!mContinuous)   return;
}
