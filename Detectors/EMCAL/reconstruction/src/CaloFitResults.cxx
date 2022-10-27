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

/// \file CaloFitResults.cxx
/// \author Hadi Hassan (hadi.hassan@cern.ch)

#include <fairlogger/Logger.h>
#include "TMath.h"

#include "EMCALReconstruction/CaloFitResults.h"
#include "DataFormatsEMCAL/Constants.h"

using namespace o2::emcal;

CaloFitResults::CaloFitResults(unsigned short maxSig, float ped,
                               int fitstatus, float amp,
                               double time, int maxTimebin,
                               float chi, unsigned short ndf, unsigned short minSig) : mMaxSig(maxSig),
                                                                                       mPed(ped),
                                                                                       mStatus(fitstatus),
                                                                                       mAmpSig(amp),
                                                                                       mTime(time),
                                                                                       mMaxTimebin(maxTimebin),
                                                                                       mChi2Sig(chi),
                                                                                       mNdfSig(ndf),
                                                                                       mMinSig(minSig)
{
}

CaloFitResults::CaloFitResults(unsigned short maxSig, float ped,
                               int fitstatus, float amp,
                               int maxTimebin) : mMaxSig(maxSig),
                                                 mPed(ped),
                                                 mStatus(fitstatus),
                                                 mAmpSig(amp),
                                                 mTime(maxTimebin),
                                                 mMaxTimebin(maxTimebin)
{
}

CaloFitResults::CaloFitResults(int maxSig, int minSig) : mMaxSig(maxSig),
                                                         mMinSig(minSig)
{
}

CaloFitResults& CaloFitResults::operator=(const CaloFitResults& source)
{
  if (this != &source) {
    mMaxSig = source.mMaxSig;
    mPed = source.mPed;
    mStatus = source.mStatus;
    mAmpSig = source.mAmpSig;
    mTime = source.mTime;
    mMaxTimebin = source.mMaxTimebin;
    mChi2Sig = source.mChi2Sig;
    mNdfSig = source.mNdfSig;
    mMinSig = source.mMinSig;
  }
  return *this;
}

bool CaloFitResults::operator==(const CaloFitResults& other) const
{
  return (mMaxSig == other.mMaxSig) && (mPed == other.mPed) &&
         (mStatus == other.mStatus) && (mAmpSig == other.mAmpSig) && (mTime == other.mTime) &&
         (mMaxTimebin == other.mMaxTimebin) && (mChi2Sig == other.mChi2Sig) &&
         (mNdfSig == other.mNdfSig) && (mMinSig == other.mMinSig);
}
