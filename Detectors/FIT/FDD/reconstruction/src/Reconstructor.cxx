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
#include "FairLogger.h"

using namespace o2::fdd;

//_____________________________________________________________________
o2::fdd::RecPoint Reconstructor::process(o2::fdd::Digit const& digitBC,
                                         gsl::span<const o2::fdd::ChannelData> inChData,
                                         gsl::span<o2::fdd::ChannelDataFloat> outChData)
{
  // Compute charge weighted average time
  Double_t timeFDA = 0, timeFDC = 0;
  Double_t weightFDA = 0.0, weightFDC = 0.0;
  Int_t nInTimeA = 0, nInTimeC = 0;

  int nch = inChData.size();
  for (int ich = 0; ich < nch; ich++) {
    outChData[ich] = o2::fdd::ChannelDataFloat{inChData[ich].mPMNumber,
                                               (inChData[ich].mTime) * timePerTDC,
                                               (double)inChData[ich].mChargeADC,
                                               inChData[ich].mFEEBits};

    Float_t adc = outChData[ich].mChargeADC;
    Float_t time = outChData[ich].mTime;
    bool inTime = inChData[ich].getFlag(ChannelData::EEventDataBit::kIsEventInTVDC);
    if (time == o2::InteractionRecord::DummyTime) {
      continue;
    }
    Float_t timeErr = 1;
    if (adc > 3) {
      timeErr = 1 / adc;
    }
    if (outChData[ich].mPMNumber < 8) {
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
  /// Avg time for each side, only if one channel satisfy the TVDC condition
  if (nInTimeA > 0) {
    mCollisionTime[o2::fdd::RecPoint::TimeA] = (weightFDA > 1) ? round(timeFDA / weightFDA * nsToPs) : o2::fdd::RecPoint::sDummyCollissionTime;
  }
  if (nInTimeC > 0) {
    mCollisionTime[o2::fdd::RecPoint::TimeC] = (weightFDC > 1) ? round(timeFDC / weightFDC * nsToPs) : o2::fdd::RecPoint::sDummyCollissionTime;
  }
  return RecPoint{mCollisionTime, digitBC.ref.getFirstEntry(), digitBC.ref.getEntries(), digitBC.getIntRecord(), digitBC.mTriggers};
}
//________________________________________________________
void Reconstructor::finish()
{
  // finalize digitization, if needed, flash remaining digits
  // if (!mContinuous)   return;
}
