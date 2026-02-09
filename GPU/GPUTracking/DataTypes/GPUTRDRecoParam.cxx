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

#include "GPUSettings.h"
#include "GPUTRDRecoParam.h"
#include "GPUCommonLogger.h"
#include "GPUCommonMath.h"

using namespace o2::gpu;

// error parameterizations taken from http://cds.cern.ch/record/2724259 Appendix A
void GPUTRDRecoParam::init(float bz, const GPUSettingsRec* rec)
{
  float resRPhiIdeal2 = rec ? rec->trd.trkltResRPhiIdeal * rec->trd.trkltResRPhiIdeal : 1.6e-3f;

  if (CAMath::Abs(CAMath::Abs(bz) - 2) < 0.1) {
    if (bz > 0) {
      // magnetic field +0.2 T
      mRPhiA2 = resRPhiIdeal2;
      mRPhiB = -1.43e-2f;
      mRPhiC2 = 4.55e-2f;

      mDyB = 0.035f;
      mCorrYDyB = 0.035f;
    } else {
      // magnetic field -0.2 T
      mRPhiA2 = resRPhiIdeal2;
      mRPhiB = 1.43e-2f;
      mRPhiC2 = 4.55e-2f;

      mDyB = -0.065f;
      mCorrYDyB = -0.065f;
    }
  } else if (CAMath::Abs(CAMath::Abs(bz) - 5) < 0.1) {
    if (bz > 0) {
      // magnetic field +0.5 T
      mRPhiA2 = resRPhiIdeal2;
      mRPhiB = 0.125f;
      mRPhiC2 = 0.0961f;

      mDyB = 0.11f;
      mCorrYDyB = 0.11f;
    } else {
      // magnetic field -0.5 T
      mRPhiA2 = resRPhiIdeal2;
      mRPhiB = -0.14f;
      mRPhiC2 = 0.1156f;

      mDyB = -0.14f;
      mCorrYDyB = -0.14f;
    }
  } else {
    LOGP(warning, "No error parameterization available for Bz= {}. Keeping default value (sigma_y = const. = 1cm)", bz);
  }
  
  mDyA2 = 6e-3f;
  mDyC2 = 0.3f;
  mCorrYDyA = 0.27f;
  mCorrYDyC = -0.44f;

  LOGP(info, "Loaded parameterizations for Bz={}: PhiRes:[{},{},{}] DyRes:[{},{},{}] CorrYDy:[{},{},{}]",
       bz, mRPhiA2, mRPhiB, mRPhiC2, mDyA2, mDyB, mDyC2, mCorrYDyA, mCorrYDyB, mCorrYDyC);
}

void GPUTRDRecoParam::recalcTrkltCov(const float tilt, const float snp, const float rowSize, float* cov) const
{
  float t2 = tilt * tilt;      // tan^2 (tilt)
  float c2 = 1.f / (1.f + t2); // cos^2 (tilt)
  float sy2 = getRPhiRes(snp);
  float sz2 = rowSize * rowSize / 12.f;
  cov[0] = c2 * (sy2 + t2 * sz2);
  cov[1] = c2 * tilt * (sz2 - sy2);
  cov[2] = c2 * (t2 * sy2 + sz2);
}

void GPUTRDRecoParam::recalcTrkltCovDy(const float tilt, const float snp, float* cov) const
{
  float t2 = tilt * tilt;      // tan^2 (tilt)
  float c2 = 1.f / (1.f + t2); // cos^2 (tilt)
  float sy2 = getRPhiRes(snp);
  float sdy2 = getDyRes(snp);
  cov[3] = getCorrYDy(snp) * CAMath::Sqrt(sdy2 * c2 * sy2);
  cov[4] = -tilt * getCorrYDy(snp) * CAMath::Sqrt(sdy2 * c2 * sy2);
  cov[5] = sdy2;
}
