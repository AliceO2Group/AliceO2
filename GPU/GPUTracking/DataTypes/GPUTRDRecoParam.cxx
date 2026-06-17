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

/// \file GPUTRDRecoParam.cxx
/// \brief Error parameterizations and helper functions for TRD reconstruction
/// \author Ole Schmidt

#include "GPUO2InterfaceConfiguration.inc"
#include "GPUSettings.h"
#include "GPUTRDRecoParam.h"
#include "GPUCommonLogger.h"
#include "GPUCommonMath.h"

using namespace o2::gpu;

// error parameterizations taken from http://cds.cern.ch/record/2724259 Appendix A
void GPUTRDRecoParam::init(float bz, const GPUSettingsRec* rec)
{
  float resRPhiIdeal = 0.04f;
  float resVsTanPhiMisalign = 0.f;
  if (rec) {
    resRPhiIdeal = rec->trd.trkltResRPhiIdeal;
    resVsTanPhiMisalign = rec->trd.trkltResVsTanPhiMisalign;
    mPileUpRangeBefore = -rec->trd.pileupBwdNBC;
    mPileUpRangeAfter = rec->trd.pileupFwdNBC;
  }
#ifndef GPUCA_STANDALONE
  else {
    const auto& rtrd = GPU_GET_CONFIG(GPUSettingsRecTRD);
    resRPhiIdeal = rtrd.trkltResRPhiIdeal;
    resVsTanPhiMisalign = rtrd.trkltResVsTanPhiMisalign;
    mPileUpRangeBefore = -rtrd.pileupBwdNBC;
    mPileUpRangeAfter = rtrd.pileupFwdNBC;
  }
#endif

  if (CAMath::Abs(CAMath::Abs(bz) - 2) < 0.1) {
    if (bz > 0) {
      // magnetic field +0.2 T
      mRPhiC2 = 4.55e-2f;
    } else {
      // magnetic field -0.2 T
      mRPhiC2 = 4.55e-2f;
    }
  } else if (CAMath::Abs(CAMath::Abs(bz) - 5) < 0.1) {
    if (bz > 0) {
      // magnetic field +0.5 T
      mRPhiC2 = 0.0961f;
    } else {
      // magnetic field -0.5 T
      mRPhiC2 = 0.1156f;
    }
  } else {
    LOGP(warning, "No error parameterization available for Bz= {}. Keeping default value (sigma_y = const. = 1cm)", bz);
  }

  mRPhiA = resRPhiIdeal;
  mRPhiATgp = resVsTanPhiMisalign;
  mLorentzAngle = -0.02f + 0.13f * bz / 5.f;

  mDyA2 = 6e-3f;
  mDyC2 = 0.3f;

  LOGP(info, "Loaded parameterizations for Bz={}: PhiRes:[{},{},{},{}] DyRes:[{},{},{}]",
       bz, mRPhiA, mRPhiATgp, mLorentzAngle, mRPhiC2, mDyA2, mLorentzAngle, mDyC2);
}

void GPUTRDRecoParam::recalcTrkltCov(const float tilt, const float snp, const float rowSize, float* cov, const float pull, const int occupancy) const
{
  float t2 = tilt * tilt;      // tan^2 (tilt)
  float c2 = 1.f / (1.f + t2); // cos^2 (tilt)
  float sy2 = getRPhiRes(snp, CAMath::Abs(pull), occupancy);
  float sz2 = rowSize * rowSize / 12.f;
  cov[0] = c2 * (sy2 + t2 * sz2);
  cov[1] = c2 * tilt * (sz2 - sy2);
  cov[2] = c2 * (t2 * sy2 + sz2);
}
