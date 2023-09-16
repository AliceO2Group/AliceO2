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

/// \file GPUParam.inc
/// \author David Rohr, Sergey Gorbunov

#ifndef GPU_GPUPARAM_INC_H
#define GPU_GPUPARAM_INC_H

#include "GPUParam.h"
#include "GPUTPCGMMergedTrackHit.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{

MEM_CLASS_PRE()
GPUdi() void MEM_LG(GPUParam)::Slice2Global(int iSlice, float x, float y, float z, float* X, float* Y, float* Z) const
{
  // conversion of coordinates sector->global
  *X = x * SliceParam[iSlice].CosAlpha - y * SliceParam[iSlice].SinAlpha;
  *Y = y * SliceParam[iSlice].CosAlpha + x * SliceParam[iSlice].SinAlpha;
  *Z = z;
}

MEM_CLASS_PRE()
GPUdi() void MEM_LG(GPUParam)::Global2Slice(int iSlice, float X, float Y, float Z, float* x, float* y, float* z) const
{
  // conversion of coordinates global->sector
  *x = X * SliceParam[iSlice].CosAlpha + Y * SliceParam[iSlice].SinAlpha;
  *y = Y * SliceParam[iSlice].CosAlpha - X * SliceParam[iSlice].SinAlpha;
  *z = Z;
}

#ifdef GPUCA_TPC_GEOMETRY_O2

MEM_CLASS_PRE()
GPUdi() void MEM_LG(GPUParam)::GetClusterErrorsSeeding2(int iRow, float z, float sinPhi, float DzDs, float& ErrY2, float& ErrZ2) const
{
  GetClusterErrors2(iRow, z, sinPhi, DzDs, ErrY2, ErrZ2);
}

MEM_CLASS_PRE()
GPUdi() float MEM_LG(GPUParam)::GetClusterError2(int yz, int type, float z, float angle2) const
{
  MakeType(const float*) c = ParamErrors[yz][type];
  float v = c[0] + c[1] * angle2 + c[2] * z + c[3] * 0;
  v = CAMath::Abs(v);
  v *= yz ? rec.tpc.clusterError2CorrectionZ : rec.tpc.clusterError2CorrectionY;
  v += yz ? rec.tpc.clusterError2AdditionalZ : rec.tpc.clusterError2AdditionalY;
  return v;
}

MEM_CLASS_PRE()
GPUdi() float MEM_LG(GPUParam)::GetSystematicClusterErrorIFC2(float x, float z, bool sideC) const
{
  const float kEpsZBoundary = 1.e-6f; // to disentangle A,C and A+C-common regions
  const float kMaxExpArg = 9.f;       // limit r-dumped error to this exp. argument
  const float kMaxExpArgZ = CAMath::Sqrt(2.f * kMaxExpArg);

  double sysErr = 0;
  if (rec.tpc.sysClusErrorInner0 > 0.f) {
    double dr = CAMath::Abs(x - 85.f);
    float argExp = dr / rec.tpc.sysClusErrorInner1; // is the Z-range limited ?
    if (argExp < kMaxExpArg) {
      if (rec.tpc.sysClusErrorZRegion < -kEpsZBoundary && !sideC) {
        return 0;    // don't apply to A-side clusters if the Z-boundary is for C-region
      } else if (rec.tpc.sysClusErrorZRegion > kEpsZBoundary && sideC) {
        return 0; // don't apply to C-side clusters if the Z-boundary is for A-region
      }
      const float dz = CAMath::Abs((rec.tpc.sysClusErrorZRegion-z)*rec.tpc.sysClusErrorZRegionSigInv);
      if (dz < kMaxExpArgZ) { // is it small enough to call exp?
        argExp += 0.5f * dz*dz;
        if (argExp<kMaxExpArg) {
          sysErr = rec.tpc.sysClusErrorInner0 * CAMath::Exp(-argExp);
        }
      }
    }
  }
  return sysErr * sysErr;
}

#else // GPUCA_TPC_GEOMETRY_O2

MEM_CLASS_PRE()
GPUdi() float MEM_LG(GPUParam)::GetClusterErrorSeeding(int yz, int type, float z, float angle2) const
{
  MakeType(const float*) c = ParamErrorsSeeding0[yz][type];
  float v = c[0] + c[1] * z + c[2] * angle2;
  v = CAMath::Abs(v);
  return v;
}

MEM_CLASS_PRE()
GPUdi() void MEM_LG(GPUParam)::GetClusterErrorsSeeding2(int iRow, float z, float sinPhi, float DzDs, float& ErrY2, float& ErrZ2) const
{
  int rowType = tpcGeometry.GetROC(iRow);
  z = CAMath::Abs(tpcGeometry.TPCLength() - CAMath::Abs(z));
  float s2 = sinPhi * sinPhi;
  if (s2 > 0.95f * 0.95f) {
    s2 = 0.95f * 0.95f;
  }
  float sec2 = 1.f / (1.f - s2);
  float angleY2 = s2 * sec2;          // dy/dx
  float angleZ2 = DzDs * DzDs * sec2; // dz/dx

  ErrY2 = GetClusterErrorSeeding(0, rowType, z, angleY2);
  ErrZ2 = GetClusterErrorSeeding(1, rowType, z, angleZ2);
  ErrY2 = ErrY2 * ErrY2 * rec.tpc.clusterError2CorrectionY + rec.tpc.clusterError2AdditionalY;
  ErrZ2 = ErrZ2 * ErrZ2 * rec.tpc.clusterError2CorrectionZ + rec.tpc.clusterError2AdditionalZ;
}

MEM_CLASS_PRE()
GPUdi() float MEM_LG(GPUParam)::GetClusterError2(int yz, int type, float z, float angle2) const
{
  MakeType(const float*) c = ParamS0Par[yz][type];
  float v = c[0] + c[1] * z + c[2] * angle2 + c[3] * z * z + c[4] * angle2 * angle2 + c[5] * z * angle2;
  v = CAMath::Abs(v);
  if (v < 0.0001f) {
    v = 0.0001f;
  }
  v *= yz ? rec.tpc.clusterError2CorrectionZ : rec.tpc.clusterError2CorrectionY;
  v += yz ? rec.tpc.clusterError2AdditionalZ : rec.tpc.clusterError2AdditionalY;
  return v;
}

MEM_CLASS_PRE()
GPUdi() float MEM_LG(GPUParam)::GetSystematicClusterErrorIFC2(float x, float z, bool sideC) const
{
  return 0;
}

#endif // !GPUCA_TPC_GEOMETRY_O2

MEM_CLASS_PRE()
GPUdi() void MEM_LG(GPUParam)::GetClusterErrors2(int iRow, float z, float sinPhi, float DzDs, float& ErrY2, float& ErrZ2) const
{
  // Calibrated cluster error from OCDB for Y and Z
  int rowType = tpcGeometry.GetROC(iRow);
  z = CAMath::Abs(tpcGeometry.TPCLength() - CAMath::Abs(z));
  float s2 = sinPhi * sinPhi;
  if (s2 > 0.95f * 0.95f) {
    s2 = 0.95f * 0.95f;
  }
  float sec2 = 1.f / (1.f - s2);
  float angleY2 = s2 * sec2;          // dy/dx
  float angleZ2 = DzDs * DzDs * sec2; // dz/dx

  ErrY2 = GetClusterError2(0, rowType, z, angleY2);
  ErrZ2 = GetClusterError2(1, rowType, z, angleZ2);
}

MEM_CLASS_PRE()
GPUdi() void MEM_LG(GPUParam)::UpdateClusterError2ByState(short clusterState, float& ErrY2, float& ErrZ2) const
{
  if (clusterState & GPUTPCGMMergedTrackHit::flagEdge) {
    ErrY2 += rec.tpc.extraClusterErrorEdgeY2;
    ErrZ2 += rec.tpc.extraClusterErrorEdgeZ2;
  }
  if (clusterState & GPUTPCGMMergedTrackHit::flagSingle) {
    ErrY2 += rec.tpc.extraClusterErrorSingleY2;
    ErrZ2 += rec.tpc.extraClusterErrorSingleZ2;
  }
  if (clusterState & (GPUTPCGMMergedTrackHit::flagSplitPad | GPUTPCGMMergedTrackHit::flagShared | GPUTPCGMMergedTrackHit::flagSingle)) {
    ErrY2 += rec.tpc.extraClusterErrorSplitPadSharedSingleY2;
    ErrY2 *= rec.tpc.extraClusterErrorFactorSplitPadSharedSingleY2;
  }
  if (clusterState & (GPUTPCGMMergedTrackHit::flagSplitTime | GPUTPCGMMergedTrackHit::flagShared | GPUTPCGMMergedTrackHit::flagSingle)) {
    ErrZ2 += rec.tpc.extraClusterErrorSplitTimeSharedSingleZ2;
    ErrZ2 *= rec.tpc.extraClusterErrorFactorSplitTimeSharedSingleZ2;
  }
}

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
