// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUdEdx.h
/// \author David Rohr

#ifndef GPUDEDX_H
#define GPUDEDX_H

#include "GPUDef.h"
#include "GPUTPCGeometry.h"
#include "GPUCommonMath.h"
#include "GPUParam.h"
#include "GPUdEdxInfo.h"
#if defined(HAVE_O2HEADERS) && !defined(GPUCA_OPENCL1)
#include "TPCdEdxCalibrationSplines.h"
#endif

namespace GPUCA_NAMESPACE
{
namespace gpu
{
#if !defined(HAVE_O2HEADERS) || defined(GPUCA_OPENCL1)

class GPUdEdx
{
 public:
  GPUd() void clear() {}
  GPUd() void fillCluster(float qtot, float qmax, int padRow, float trackSnp, float trackTgl, const GPUParam& param, const TPCdEdxCalibrationSplines* splines, float z) {}
  GPUd() void fillSubThreshold(int padRow, const GPUParam& param) {}
  GPUd() void computedEdx(GPUdEdxInfo& output, const GPUParam& param) {}
};

#else

class GPUdEdx
{
 public:
  // The driver must call clear(), fill clusters row by row outside-in, then run computedEdx() to get the result
  GPUd() void clear();
  GPUd() void fillCluster(float qtot, float qmax, int padRow, float trackSnp, float trackTgl, const GPUParam& param, const TPCdEdxCalibrationSplines* splines, float z);
  GPUd() void fillSubThreshold(int padRow, const GPUParam& param);
  GPUd() void computedEdx(GPUdEdxInfo& output, const GPUParam& param);

 private:
  GPUd() float GetSortTruncMean(GPUCA_DEDX_STORAGE_TYPE* array, int count, int trunclow, int trunchigh);
  GPUd() void checkSubThresh(int roc);

  template <typename T, typename fake = void>
  struct scalingFactor;
  template <typename fake>
  struct scalingFactor<unsigned short, fake> {
    static constexpr float factor = 4.f;
    static constexpr float round = 0.5f;
  };
  template <typename fake>
  struct scalingFactor<float, fake> {
    static constexpr float factor = 1.f;
    static constexpr float round = 0.f;
  };
#if defined(__CUDACC__) || defined(__HIPCC__)
  template <typename fake>
  struct scalingFactor<half, fake> {
    static constexpr float factor = 1.f;
    static constexpr float round = 0.f;
  };
#endif

  static constexpr int MAX_NCL = GPUCA_ROW_COUNT; // Must fit in mNClsROC (unsigned char)!

  GPUCA_DEDX_STORAGE_TYPE mChargeTot[MAX_NCL]; // No need for default, just some memory
  GPUCA_DEDX_STORAGE_TYPE mChargeMax[MAX_NCL]; // No need for default, just some memory
  float mSubThreshMinTot = 0.f;
  float mSubThreshMinMax = 0.f;
  unsigned char mNClsROC[4] = {0};
  unsigned char mNClsROCSubThresh[4] = {0};
  unsigned char mCount = 0;
  unsigned char mLastROC = 255;
  char mNSubThresh = 0;
};

GPUdi() void GPUdEdx::checkSubThresh(int roc)
{
  if (roc != mLastROC) {
    if (mNSubThresh && mCount + mNSubThresh <= MAX_NCL) {
      for (int i = 0; i < mNSubThresh; i++) {
        mChargeTot[mCount] = (GPUCA_DEDX_STORAGE_TYPE)(mSubThreshMinTot * scalingFactor<GPUCA_DEDX_STORAGE_TYPE>::factor + scalingFactor<GPUCA_DEDX_STORAGE_TYPE>::round);
        mChargeMax[mCount++] = (GPUCA_DEDX_STORAGE_TYPE)(mSubThreshMinMax * scalingFactor<GPUCA_DEDX_STORAGE_TYPE>::factor + scalingFactor<GPUCA_DEDX_STORAGE_TYPE>::round);
      }
      mNClsROC[mLastROC] += mNSubThresh;
      mNClsROCSubThresh[mLastROC] += mNSubThresh;
    }
    mNSubThresh = 0;
    mSubThreshMinTot = 1e10f;
    mSubThreshMinMax = 1e10f;
  }

  mLastROC = roc;
}

GPUdnii() void GPUdEdx::fillCluster(float qtot, float qmax, int padRow, float trackSnp, float trackTgl, const GPUParam& GPUrestrict() param, const TPCdEdxCalibrationSplines* splines, float z)
{
  if (mCount >= MAX_NCL) {
    return;
  }
  const int roc = param.tpcGeometry.GetROC(padRow);
  checkSubThresh(roc);
  float snp2 = trackSnp * trackSnp;
  if (snp2 > GPUCA_MAX_SIN_PHI_LOW) {
    snp2 = GPUCA_MAX_SIN_PHI_LOW;
  }
  const float tgl2 = trackTgl * trackTgl;
  float factor = CAMath::Sqrt((1 - snp2) / (1 + tgl2));
  factor /= param.tpcGeometry.PadHeight(padRow);
  qtot *= factor;
  qmax *= factor;

  const float sec2 = 1.f / (1.f - snp2);
  // angleZ local dip angle: z angle - dz/dx (cm/cm)
  float angleZ = CAMath::Sqrt(tgl2 * sec2);
  if (angleZ > 3) {
    angleZ = 3;
  }

  const int region = param.tpcGeometry.GetRegion(padRow);
  z = CAMath::Abs(z);

  // get the correction for qMax and qTot from the splines for given angle and drift length
  const float qMaxCorr = splines->interpolateqMax(region, angleZ, z);
  const float qTotCorr = splines->interpolateqTot(region, angleZ, z);
  qmax /= qMaxCorr;
  qtot /= qTotCorr;

  mChargeTot[mCount] = (GPUCA_DEDX_STORAGE_TYPE)(qtot * scalingFactor<GPUCA_DEDX_STORAGE_TYPE>::factor + scalingFactor<GPUCA_DEDX_STORAGE_TYPE>::round);
  mChargeMax[mCount++] = (GPUCA_DEDX_STORAGE_TYPE)(qmax * scalingFactor<GPUCA_DEDX_STORAGE_TYPE>::factor + scalingFactor<GPUCA_DEDX_STORAGE_TYPE>::round);
  mNClsROC[roc]++;
  if (qtot < mSubThreshMinTot) {
    mSubThreshMinTot = qtot;
  }
  if (qmax < mSubThreshMinMax) {
    mSubThreshMinMax = qmax;
  }
}

GPUdi() void GPUdEdx::fillSubThreshold(int padRow, const GPUParam& GPUrestrict() param)
{
  const int roc = param.tpcGeometry.GetROC(padRow);
  checkSubThresh(roc);
  mNSubThresh++;
}

#endif // !HAVE_O2HEADERS || __OPENCL1__
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
