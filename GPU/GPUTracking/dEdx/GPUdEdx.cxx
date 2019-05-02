// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUdEdx.cxx
/// \author David Rohr

#include "GPUdEdx.h"
#include "GPUTPCGeometry.h"
#include "GPUdEdxInfo.h"
#include "GPUCommonAlgorithm.h"
#include "GPUParam.h"

using namespace GPUCA_NAMESPACE::gpu;

GPUd() void GPUdEdx::clear() { new (this) GPUdEdx; }

GPUd() void GPUdEdx::computedEdx(GPUdEdxInfo& output, const GPUParam& param)
{
  checkSubThresh(255);
  const int truncLow = param.rec.dEdxTruncLow;
  const int truncHigh = param.rec.dEdxTruncHigh;
  const int countIROC = mNClsROC[0];
  const int countOROC1 = mNClsROC[1];
  const int countOROC2 = mNClsROC[2];
  const int countOROC3 = mNClsROC[3];
  output.dEdxTotIROC = GetSortTruncMean(mChargeTot + countOROC3 + countOROC2 + countOROC1, countIROC, truncLow, truncHigh);
  output.dEdxTotOROC1 = GetSortTruncMean(mChargeTot + countOROC3 + countOROC2, countOROC1, truncLow, truncHigh);
  output.dEdxTotOROC2 = GetSortTruncMean(mChargeTot + countOROC3, countOROC2, truncLow, truncHigh);
  output.dEdxTotOROC3 = GetSortTruncMean(mChargeTot, countOROC3, truncLow, truncHigh);
  output.dEdxTotTPC = GetSortTruncMean(mChargeTot, mCount, truncLow, truncHigh);
  output.dEdxMaxIROC = GetSortTruncMean(mChargeMax + countOROC3 + countOROC2 + countOROC1, countIROC, truncLow, truncHigh);
  output.dEdxMaxOROC1 = GetSortTruncMean(mChargeMax + countOROC3 + countOROC2, countOROC1, truncLow, truncHigh);
  output.dEdxMaxOROC2 = GetSortTruncMean(mChargeMax + countOROC3, countOROC2, truncLow, truncHigh);
  output.dEdxMaxOROC3 = GetSortTruncMean(mChargeMax, countOROC3, truncLow, truncHigh);
  output.dEdxMaxTPC = GetSortTruncMean(mChargeMax, mCount, truncLow, truncHigh);
  output.NHitsIROC = countIROC - mNClsROCSubThresh[0];
  output.NHitsSubThresholdIROC = countIROC;
  output.NHitsOROC1 = countOROC1 - mNClsROCSubThresh[1];
  output.NHitsSubThresholdOROC1 = countOROC1;
  output.NHitsOROC2 = countOROC2 - mNClsROCSubThresh[2];
  output.NHitsSubThresholdOROC2 = countOROC2;
  output.NHitsOROC2 = countOROC3 - mNClsROCSubThresh[3];
  output.NHitsSubThresholdOROC2 = countOROC3;
}

GPUd() float GPUdEdx::GetSortTruncMean(float* array, int count, int trunclow, int trunchigh)
{
  trunclow = count * trunclow / 128;
  trunchigh = count * trunchigh / 128;
  if (count - trunclow - trunchigh <= 0) {
    return (0.);
  }
  CAAlgo::sort(array, array + count);
  float mean = 0;
  for (int i = trunclow; i < count - trunchigh; i++) {
    mean += array[i];
  }
  return (mean / (count - trunclow - trunchigh));
}
