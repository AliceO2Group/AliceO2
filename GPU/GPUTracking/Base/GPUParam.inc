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
#if !defined(__OPENCL__) || defined(__OPENCLCPP__)
#include "GPUTPCClusterOccupancyMap.h"
#endif

namespace GPUCA_NAMESPACE
{
namespace gpu
{

MEM_CLASS_PRE()
GPUdi() void MEM_LG(GPUParam)::Slice2Global(int32_t iSlice, float x, float y, float z, float* X, float* Y, float* Z) const
{
  // conversion of coordinates sector->global
  *X = x * SliceParam[iSlice].CosAlpha - y * SliceParam[iSlice].SinAlpha;
  *Y = y * SliceParam[iSlice].CosAlpha + x * SliceParam[iSlice].SinAlpha;
  *Z = z;
}

MEM_CLASS_PRE()
GPUdi() void MEM_LG(GPUParam)::Global2Slice(int32_t iSlice, float X, float Y, float Z, float* x, float* y, float* z) const
{
  // conversion of coordinates global->sector
  *x = X * SliceParam[iSlice].CosAlpha + Y * SliceParam[iSlice].SinAlpha;
  *y = Y * SliceParam[iSlice].CosAlpha - X * SliceParam[iSlice].SinAlpha;
  *z = Z;
}

#ifdef GPUCA_TPC_GEOMETRY_O2

MEM_CLASS_PRE()
GPUdi() void MEM_LG(GPUParam)::GetClusterErrorsSeeding2(uint8_t sector, int32_t iRow, float z, float sinPhi, float DzDs, float time, float& ErrY2, float& ErrZ2) const
{
  const int32_t rowType = tpcGeometry.GetROC(iRow);
  z = CAMath::Abs(tpcGeometry.TPCLength() - CAMath::Abs(z));
  const float s2 = CAMath::Min(sinPhi * sinPhi, 0.95f * 0.95f);
  const float sec2 = 1.f / (1.f - s2);
  const float angleY2 = s2 * sec2;          // dy/dx
  const float angleZ2 = DzDs * DzDs * sec2; // dz/dx

  const float unscaledMult = time >= 0.f ? GetUnscaledMult(time) / tpcGeometry.Row2X(iRow) : 0.f;

  ErrY2 = GetClusterErrorSeeding(0, rowType, z, angleY2, unscaledMult); // Returns Err2
  ErrZ2 = GetClusterErrorSeeding(1, rowType, z, angleZ2, unscaledMult); // Returns Err2
}

MEM_CLASS_PRE()
GPUdi() float MEM_LG(GPUParam)::GetClusterErrorSeeding(int32_t yz, int32_t type, float zDiff, float angle2, float unscaledMult) const // Note, returns Err2 despite the name not containing 2
{
  MakeType(const float*) c = ParamErrors[yz][type]; // Note: c[0] = p[0]^2, c[1] = p[1]^2 * padHeight, c[2] = p[2]^2 / tpcLength / padHeight, c[3] = p[3]^2 * clusterErrorOccupancyScaler^2
  float v = c[0] + c[1] * angle2 + c[2] * zDiff + c[3] * (unscaledMult * unscaledMult);
  v = CAMath::Abs(v);
  v *= yz ? rec.tpc.clusterError2CorrectionZ : rec.tpc.clusterError2CorrectionY;
  v += yz ? rec.tpc.clusterError2AdditionalZSeeding : rec.tpc.clusterError2AdditionalYSeeding;
  return v;
}

MEM_CLASS_PRE()
GPUdi() float MEM_LG(GPUParam)::GetClusterError2(int32_t yz, int32_t type, float zDiff, float angle2, float unscaledMult, float scaledInvAvgCharge, float scaledInvCharge) const
{
  MakeType(const float*) c = ParamErrors[yz][type]; // Note: c[0] = p[0]^2, c[1] = p[1]^2 * padHeight, c[2] = p[2]^2 / tpcLength / padHeight, c[3] = p[3]^2 * clusterErrorOccupancyScaler^2
  float v = c[0] + c[1] * angle2 * scaledInvAvgCharge + c[2] * zDiff * scaledInvCharge + c[3] * (unscaledMult * unscaledMult) * (scaledInvAvgCharge * scaledInvAvgCharge);
  v = CAMath::Abs(v);
  v *= yz ? rec.tpc.clusterError2CorrectionZ : rec.tpc.clusterError2CorrectionY;
  v += yz ? rec.tpc.clusterError2AdditionalZ : rec.tpc.clusterError2AdditionalY;
  return v;
}

MEM_CLASS_PRE()
GPUdi() float MEM_LG(GPUParam)::GetSystematicClusterErrorIFC2(float x, float y, float z, bool sideC) const
{
  float sysErr = 0.f;
  const float kMaxExpArg = 9.f; // limit r-dumped error to this exp. argument

  const float rIFC = 83.5;
  const float r = CAMath::Sqrt(x * x + y * y);
  if (r - rIFC < rec.tpc.sysClusErrorMinDist) {
    return rec.tpc.sysClusErrorMaskError;
  }

  if (sideC && rec.tpc.sysClusErrorNormIFCCE) {
    float dr = CAMath::Abs(x - 85.f);
    float argExpIFCCE = dr * rec.tpc.sysClusErrorSlopeIFCCE;
    float dz = CAMath::Abs((rec.tpc.sysClusErrorIFCCEZRegion - z) * rec.tpc.sysClusErrorslopeIFCCEZ);
    argExpIFCCE += 0.5f * dz * dz;
    if (argExpIFCCE < kMaxExpArg) {
      float tmp = rec.tpc.sysClusErrorNormIFCCE * CAMath::Exp(-argExpIFCCE);
      sysErr += tmp * tmp;
    }
  }

  if (rec.tpc.sysClusErrorNormIFC) {
    float argExpIFC = (r - rIFC) * rec.tpc.sysClusErrorSlopeIFC;
    if (argExpIFC < kMaxExpArg) {
      float tmp = rec.tpc.sysClusErrorNormIFC * CAMath::Exp(-argExpIFC);
      sysErr += tmp * tmp;
    }
  }

  return sysErr;
}

MEM_CLASS_PRE()
GPUdi() float MEM_LG(GPUParam)::GetSystematicClusterErrorC122(float x, float y, uint8_t sector) const
{
  const float dx = x - 83.f;
  if (dx > occupancyTotal * rec.tpc.sysClusErrorC12Box) {
    return 0.f;
  }
  CONSTEXPR float dEdgeInv = 18.f / CAMath::Pi();
  const float dy = (sector == (GPUCA_NSLICES / 2 + 1) ? 0.5f : -0.5f) * (y / x) * dEdgeInv + 0.5f;
  const float errC12 = rec.tpc.sysClusErrorC12Norm * occupancyTotal * dy;
  return errC12 * errC12;
}

#else // GPUCA_TPC_GEOMETRY_O2

MEM_CLASS_PRE()
GPUdi() float MEM_LG(GPUParam)::GetClusterErrorSeeding(int32_t yz, int32_t type, float zDiff, float angle2, float scaledMult) const
{
  MakeType(const float*) c = ParamErrorsSeeding0[yz][type];
  float v = c[0] + c[1] * zDiff + c[2] * angle2;
  v = CAMath::Abs(v);
  return v;
}

MEM_CLASS_PRE()
GPUdi() void MEM_LG(GPUParam)::GetClusterErrorsSeeding2(uint8_t sector, int32_t iRow, float z, float sinPhi, float DzDs, float time, float& ErrY2, float& ErrZ2) const
{
  int32_t rowType = tpcGeometry.GetROC(iRow);
  z = CAMath::Abs(tpcGeometry.TPCLength() - CAMath::Abs(z));
  const float s2 = CAMath::Min(sinPhi * sinPhi, 0.95f * 0.95f);
  float sec2 = 1.f / (1.f - s2);
  float angleY2 = s2 * sec2;          // dy/dx
  float angleZ2 = DzDs * DzDs * sec2; // dz/dx

  ErrY2 = GetClusterErrorSeeding(0, rowType, z, angleY2, 0.f);
  ErrZ2 = GetClusterErrorSeeding(1, rowType, z, angleZ2, 0.f);
  ErrY2 = ErrY2 * ErrY2 * rec.tpc.clusterError2CorrectionY + rec.tpc.clusterError2AdditionalY;
  ErrZ2 = ErrZ2 * ErrZ2 * rec.tpc.clusterError2CorrectionZ + rec.tpc.clusterError2AdditionalZ;
}

MEM_CLASS_PRE()
GPUdi() float MEM_LG(GPUParam)::GetClusterError2(int32_t yz, int32_t type, float zDiff, float angle2, float unscaledMult, float avgInvCharge, float invCharge) const
{
  MakeType(const float*) c = ParamS0Par[yz][type];
  float v = c[0] + c[1] * zDiff + c[2] * angle2 + c[3] * zDiff * zDiff + c[4] * angle2 * angle2 + c[5] * zDiff * angle2;
  v = CAMath::Abs(v);
  if (v < 0.0001f) {
    v = 0.0001f;
  }
  v *= yz ? rec.tpc.clusterError2CorrectionZ : rec.tpc.clusterError2CorrectionY;
  v += yz ? rec.tpc.clusterError2AdditionalZ : rec.tpc.clusterError2AdditionalY;
  return v;
}

MEM_CLASS_PRE()
GPUdi() float MEM_LG(GPUParam)::GetSystematicClusterErrorIFC2(float trackX, float trackY, float z, bool sideC) const
{
  return 0;
}

MEM_CLASS_PRE()
GPUdi() float MEM_LG(GPUParam)::GetSystematicClusterErrorC122(float trackX, float trackY, uint8_t sector) const
{
  return 0;
}

#endif // !GPUCA_TPC_GEOMETRY_O2

MEM_CLASS_PRE()
GPUdi() void MEM_LG(GPUParam)::GetClusterErrors2(uint8_t sector, int32_t iRow, float z, float sinPhi, float DzDs, float time, float avgInvCharge, float invCharge, float& ErrY2, float& ErrZ2) const
{
  const int32_t rowType = tpcGeometry.GetROC(iRow);
  z = CAMath::Abs(tpcGeometry.TPCLength() - CAMath::Abs(z));
  const float s2 = CAMath::Min(sinPhi * sinPhi, 0.95f * 0.95f);
  const float sec2 = 1.f / (1.f - s2);
  const float angleY2 = s2 * sec2;          // dy/dx
  const float angleZ2 = DzDs * DzDs * sec2; // dz/dx

  const float unscaledMult = time >= 0.f ? GetUnscaledMult(time) / tpcGeometry.Row2X(iRow) : 0.f;
  const float scaledInvAvgCharge = avgInvCharge * rec.tpc.clusterErrorChargeScaler > 0.f ? avgInvCharge * rec.tpc.clusterErrorChargeScaler : 1.f;
  const float scaledInvCharge = invCharge * rec.tpc.clusterErrorChargeScaler > 0.f ? invCharge * rec.tpc.clusterErrorChargeScaler : 1.f;

  ErrY2 = GetClusterError2(0, rowType, z, angleY2, unscaledMult, scaledInvAvgCharge, scaledInvCharge);
  ErrZ2 = GetClusterError2(1, rowType, z, angleZ2, unscaledMult, scaledInvAvgCharge, scaledInvCharge);
}

MEM_CLASS_PRE()
GPUdi() void MEM_LG(GPUParam)::UpdateClusterError2ByState(int16_t clusterState, float& ErrY2, float& ErrZ2) const
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

MEM_CLASS_PRE()
GPUdi() float MEM_LG(GPUParam)::GetUnscaledMult(float time) const
{
#if !defined(__OPENCL__) || defined(__OPENCLCPP__)
  if (!occupancyMap) {
    return 0.f;
  }
  const uint32_t bin = CAMath::Max(0.f, time / rec.tpc.occupancyMapTimeBins);
  return occupancyMap[bin];
#else
  return 0.f;
#endif
}

MEM_CLASS_PRE()
GPUdi() bool MEM_LG(GPUParam)::rejectEdgeClusterByY(float uncorrectedY, int32_t iRow, float trackSigmaY) const
{
  return CAMath::Abs(uncorrectedY) > (tpcGeometry.NPads(iRow) - 1) * 0.5f * tpcGeometry.PadWidth(iRow) + rec.tpc.rejectEdgeClustersMargin + trackSigmaY * rec.tpc.rejectEdgeClustersSigmaMargin;
}

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
