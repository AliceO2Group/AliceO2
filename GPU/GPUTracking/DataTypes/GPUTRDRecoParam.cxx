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

      mDyA2 = 1.225e-3f;
      mDyB = -9.8e-3f;
      mDyC2 = 3.88e-2f;

      mAngleToDyA = -0.1f;
      mAngleToDyB = 1.89f;
      mAngleToDyC = -0.4f;
    } else {
      // magnetic field -0.2 T
      mRPhiA2 = resRPhiIdeal2;
      mRPhiB = 1.43e-2f;
      mRPhiC2 = 4.55e-2f;

      mDyA2 = 1.225e-3f;
      mDyB = 9.8e-3f;
      mDyC2 = 3.88e-2f;

      mAngleToDyA = 0.1f;
      mAngleToDyB = 1.89f;
      mAngleToDyC = 0.4f;
    }
  } else if (CAMath::Abs(CAMath::Abs(bz) - 5) < 0.1) {
    if (bz > 0) {
      // magnetic field +0.5 T
      mRPhiA2 = resRPhiIdeal2;
      mRPhiB = 0.125f;
      mRPhiC2 = 0.0961f;

      mDyA2 = 1.681e-3f;
      mDyB = 0.15f;
      mDyC2 = 0.1849f;

      mAngleToDyA = 0.13f;
      mAngleToDyB = 2.43f;
      mAngleToDyC = -0.58f;
    } else {
      // magnetic field -0.5 T
      mRPhiA2 = resRPhiIdeal2;
      mRPhiB = -0.14f;
      mRPhiC2 = 0.1156f;

      mDyA2 = 2.209e-3f;
      mDyB = -0.15f;
      mDyC2 = 0.2025f;

      mAngleToDyA = -0.15f;
      mAngleToDyB = 2.34f;
      mAngleToDyC = 0.56f;
    }
  } else {
    LOGP(warning, "No error parameterization available for Bz= {}. Keeping default value (sigma_y = const. = 1cm)", bz);
  }
  LOGP(info, "Loaded parameterizations for Bz={}: PhiRes:[{},{},{}] DyRes:[{},{},{}] Angle2Dy:[{},{},{}]",
       bz, mRPhiA2, mRPhiB, mRPhiC2, mDyA2, mDyB, mDyC2, mAngleToDyA, mAngleToDyB, mAngleToDyC);
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
