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
  for (int ich = 0; ich < nch; ich++) {
    bool inTime = inChData[ich].getFlag(ChannelData::EEventDataBit::kIsEventInTVDC);
    bool inAdcGate = inChData[ich].getFlag(ChannelData::EEventDataBit::kIsCFDinADCgate);
    if (inAdcGate) {
      outChData.emplace_back((int)inChData[ich].mPMNumber, (inChData[ich].mTime) * timePerTDC,
                             (double)inChData[ich].mChargeADC, inChData[ich].mFEEBits);
      nStored++;
    }

    Float_t adc = inChData[ich].mChargeADC;
    Float_t time = (inChData[ich].mTime) * timePerTDC;
    if (time == o2::InteractionRecord::DummyTime) {
      continue;
    }
    Float_t timeErr = 1;
    if (adc > 3) {
      timeErr = 1. / adc;
    }
    if ((int)inChData[ich].mPMNumber < 8) {
      nInTimeC += inTime;
      timeFDC += time / (timeErr * timeErr);
      weightFDC += 1. / (timeErr * timeErr);
    } else {
      nInTimeA += inTime;
      timeFDA += time / (timeErr * timeErr);
      weightFDA += 1. / (timeErr * timeErr);
    }
  }
  const int nsToPs = 1e3;
  std::array<int, 2> mCollisionTime = {o2::fdd::RecPoint::sDummyCollissionTime, o2::fdd::RecPoint::sDummyCollissionTime};
  /// Avg time for each side, only if one channel satisfy the TVDC condition (if not, also avg time is propagated for background study using AO2D)
  if (nInTimeA > 0) {
    mCollisionTime[o2::fdd::RecPoint::TimeA] = (weightFDA > 1) ? round(timeFDA / weightFDA * nsToPs)
                                                               : o2::fdd::RecPoint::sDummyCollissionTime;
  } else {
    if (weightFDA > 0) {
      mCollisionTime[o2::fdd::RecPoint::TimeA] = round(timeFDA / weightFDA * nsToPs);
    }
  }
  if (nInTimeC > 0) {
    mCollisionTime[o2::fdd::RecPoint::TimeC] = (weightFDC > 1) ? round(timeFDC / weightFDC * nsToPs)
                                                               : o2::fdd::RecPoint::sDummyCollissionTime;
  } else {
    if (weightFDC > 0) {
      mCollisionTime[o2::fdd::RecPoint::TimeC] = round(timeFDC / weightFDC * nsToPs);
    }
  }
  RecPoints.emplace_back(mCollisionTime, firstEntry, nStored, digitBC.getIntRecord(), digitBC.mTriggers);
}
//________________________________________________________
void Reconstructor::finish()
{
  // finalize reconstruction, if needed, flash remaining recpoints
  // if (!mContinuous)   return;
}
