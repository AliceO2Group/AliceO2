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

/// \file GPUTPCGMTrackParam.h
/// \author David Rohr, Sergey Gorbunov

#ifndef GPUTPCGMTRACKPARAM_H
#define GPUTPCGMTRACKPARAM_H

#include "GPUTPCDef.h"
#include "GPUTPCGMMergedTrackHit.h"
#include "GPUTPCGMMergerTypes.h"
#include "GPUCommonMath.h"
#include "GPUdEdxInfo.h"

#ifndef GPUCA_GPUCODE_DEVICE
#include <cstddef>
#endif

class AliExternalTrackParam;

namespace o2::tpc
{
struct ClusterNative;
};

namespace o2::gpu
{
class GPUTPCTracker;
class GPUTPCGMMerger;
class GPUTPCGMBorderTrack;
struct GPUParam;
class GPUTPCGMPhysicalTrackModel;
class GPUTPCGMPolynomialField;
class GPUTPCGMMergedTrack;
class GPUTPCGMPropagator;
class GPUdEdx;
struct GPUTPCGMMergedTrackHit;

namespace gputpcgmmergertypes
{
struct InterpolationErrorHit;
} // namespace gputpcgmmergertypes

/**
 * @class GPUTPCGMTrackParam
 *
 * GPUTPCGMTrackParam class describes the track parametrisation
 * which is used by the GPUTPCGMTracker sector tracker.
 *
 */
class GPUTPCGMTrackParam
{
 public:
  GPUd() float& X()
  {
    return mX;
  }
  GPUd() float& Y()
  {
    return mP[0];
  }
  GPUd() float& Z()
  {
    return mP[1];
  }
  GPUd() float& SinPhi()
  {
    return mP[2];
  }
  GPUd() float& DzDs()
  {
    return mP[3];
  }
  GPUd() float& QPt()
  {
    return mP[4];
  }
  GPUd() float& TOffset()
  {
    return mTOffset;
  }

  GPUhd() float GetX() const { return mX; }
  GPUhd() float GetY() const { return mP[0]; }
  GPUhd() float GetZ() const { return mP[1]; }
  GPUd() float GetSinPhi() const { return mP[2]; }
  GPUd() float GetDzDs() const { return mP[3]; }
  GPUd() float GetQPt() const { return mP[4]; }
  GPUd() float GetTOffset() const { return mTOffset; }

  GPUd() float GetKappa(float Bz) const { return -mP[4] * Bz; }

  GPUd() void SetX(float v) { mX = v; }

  GPUd() float* Par()
  {
    return mP;
  }
  GPUd() const float* GetPar() const { return mP; }
  GPUd() float GetPar(int32_t i) const { return (mP[i]); }
  GPUd() void SetPar(int32_t i, float v) { mP[i] = v; }

  GPUd() float& Chi2()
  {
    return mChi2;
  }
  GPUd() int32_t& NDF()
  {
    return mNDF;
  }

  GPUd() float Err2Y() const { return mC[0]; }
  GPUd() float Err2Z() const { return mC[2]; }
  GPUd() float Err2SinPhi() const { return mC[5]; }
  GPUd() float Err2DzDs() const { return mC[9]; }
  GPUd() float Err2QPt() const { return mC[14]; }

  GPUd() float GetChi2() const { return mChi2; }
  GPUd() int32_t GetNDF() const { return mNDF; }

  GPUd() float GetCosPhi() const { return CAMath::Sqrt(float(1.f) - GetSinPhi() * GetSinPhi()); }

  GPUd() float GetErr2Y() const { return mC[0]; }
  GPUd() float GetErr2Z() const { return mC[2]; }
  GPUd() float GetErr2SinPhi() const { return mC[5]; }
  GPUd() float GetErr2DzDs() const { return mC[9]; }
  GPUd() float GetErr2QPt() const { return mC[14]; }

  GPUd() float* Cov()
  {
    return mC;
  }

  GPUd() const float* GetCov() const { return mC; }
  GPUd() float GetCov(int32_t i) const { return mC[i]; }

  GPUd() void SetCov(int32_t i, float v) { mC[i] = v; }
  GPUd() void SetChi2(float v) { mChi2 = v; }
  GPUd() void SetNDF(int32_t v) { mNDF = v; }

  GPUd() float GetMirroredY(float Bz) const;

  GPUd() void ResetCovariance();

  GPUd() bool CheckNumericalQuality(float overrideCovYY = -1.f) const;
  GPUd() bool CheckCov() const;

  GPUd() bool Fit(GPUTPCGMMerger& merger, int32_t iTrk, int32_t& N, int32_t& NTolerated, float& Alpha, GPUTPCGMMergedTrack& track, bool rebuilt);
  GPUd() void DodEdx(GPUdEdx& dEdx, GPUdEdx& dEdxAlt, GPUTPCGMMerger& merger, bool finalFit, int ihit, int ihitMergeFirst, int wayDirection, const GPUTPCGMMergedTrackHit* clusters, uint8_t clusterState, float zz, uint8_t dEdxSubThresholdRow);
  GPUd() int32_t FitHit(GPUTPCGMMerger& merger, const int32_t iTrk, const GPUTPCGMMergedTrack& track, const float xx, const float yy, const float zz, const uint8_t clusterState, const float clAlpha, const int32_t iWay, const bool inFlyDirection, float& deltaZ, float& lastUpdateX, GPUTPCGMMergedTrackHit* clusters, GPUTPCGMPropagator& prop, gputpcgmmergertypes::InterpolationErrorHit& inter, GPUdEdx& dEdx, GPUdEdx& dEdxAlt, float& sumInvSqrtCharge, int32_t& nAvgCharge, const int32_t ihit, const int32_t ihitMergeFirst, const bool allowChangeClusters, const bool refit, const bool finalFit, int32_t& nMissed, int32_t& nMissed2, int32_t& resetT0, float uncorrectedY);
  GPUd() void FitAddRow(const int32_t iRow, const uint8_t sector, const int32_t iTrk, const GPUTPCGMMergedTrack& track, GPUTPCGMPropagator& prop, const bool inFlyDirection, GPUTPCGMMerger& merger, uint8_t* dEdxSubThresholdRow, const bool dodEdx, const bool doAttach, const bool doInterpolate);
  GPUd() void HandleCrossCE(const GPUParam& param, const uint8_t sector, const uint8_t& lastSector);
  GPUd() static void RefitTrack(GPUTPCGMMergedTrack& track, int32_t iTrk, GPUTPCGMMerger& merger, bool rebuilt);
  GPUd() void MoveToReference(GPUTPCGMPropagator& prop, const GPUParam& param, float& alpha);
  GPUd() void MirrorTo(GPUTPCGMPropagator& prop, float toY, float toZ, bool inFlyDirection, const GPUParam& param, uint8_t row, uint8_t clusterState, bool mirrorParameters, int8_t sector);
  GPUd() int32_t MergeDoubleRowClusters(int32_t& ihit, int32_t wayDirection, GPUTPCGMMergedTrackHit* clusters, const GPUTPCGMMerger& merger, GPUTPCGMPropagator& prop, float& xx, float& yy, float& zz, int32_t maxN, float clAlpha, uint8_t& clusterState, const bool markReject);
  GPUd() float FindBestInterpolatedHit(GPUTPCGMMerger& merger, gputpcgmmergertypes::InterpolationErrorHit& inter, const uint8_t sector, const uint8_t row, const float deltaZ, const float sumInvSqrtCharge, const int nAvgCharge, const GPUTPCGMPropagator& prop, const int32_t iTrk, bool interOnly);
  GPUd() void InterpolateMissingRows(GPUTPCGMMerger& merger, gputpcgmmergertypes::InterpolationErrors& interpolation, GPUTPCGMMergedTrackHit* clusters, int32_t ihit, int32_t interpolationIndex, int32_t lastRow, const float deltaZ, const float sumInvSqrtCharge, const int32_t nAvgCharge, const GPUTPCGMPropagator& prop, const int32_t iTrk);

  GPUd() float AttachClusters(const GPUTPCGMMerger& merger, int32_t sector, int32_t iRow, int32_t iTrack, bool goodLeg, GPUTPCGMPropagator& prop); // Returns uncorrectedY for later use
  GPUd() float AttachClusters(const GPUTPCGMMerger& merger, int32_t sector, int32_t iRow, int32_t iTrack, bool goodLeg, float Y, float Z);
  GPUd() void AttachClustersLooper(const GPUTPCGMMerger& merger, int32_t sector, int32_t iRow, const int32_t iTrack, const bool up, const GPUTPCGMPropagator& prop);
  GPUd() void AttachClustersLooperFollow(const GPUTPCGMMerger& merger, GPUTPCGMPropagator& prop, int32_t sector, int32_t iRow, const int32_t iTrack, const bool up);
  GPUd() void StoreLoopPropagation(const GPUTPCGMMerger& merger, int32_t sector, int32_t iRow, int32_t iTrack, bool outwards, float alpha);
  GPUd() void StoreOuter(gputpcgmmergertypes::GPUTPCOuterParam* outerParam, float alpha);
  GPUd() static void PropagateLooper(const GPUTPCGMMerger& merger, int32_t loopIdx);

  GPUd() void AddCovDiagErrors(const float* errors2);
  GPUd() void AddCovDiagErrorsWithCorrelations(const float* errors2);

  GPUdi() void MarkClusters(GPUTPCGMMergedTrackHit* GPUrestrict() clusters, int32_t ihitFirst, int32_t ihitLast, int32_t wayDirection, uint8_t state)
  {
    clusters[ihitFirst].state |= state;
    while (ihitFirst != ihitLast) {
      clusters[ihitFirst += wayDirection].state |= state;
    }
  }
  GPUdi() void UnmarkClusters(GPUTPCGMMergedTrackHit* GPUrestrict() clusters, int32_t ihitFirst, int32_t ihitLast, int32_t wayDirection, uint8_t state)
  {
    clusters[ihitFirst].state &= ~state;
    while (ihitFirst != ihitLast) {
      clusters[ihitFirst += wayDirection].state &= ~state;
    }
  }
  GPUdi() static void NormalizeAlpha(float& alpha)
  {
    if (alpha > CAMath::Pi()) {
      alpha -= CAMath::TwoPi();
    } else if (alpha <= -CAMath::Pi()) {
      alpha += CAMath::TwoPi();
    }
  }

  GPUd() void Rotate(float alpha);
  GPUd() float ShiftZ(const GPUTPCGMMerger& merger, int32_t sector, float cltmax, float cltmin, float clx);
  GPUd() float ShiftZ(const GPUTPCGMMergedTrackHit* clusters, const GPUTPCGMMerger& merger, int32_t N);

  GPUd() static float Reciprocal(float x) { return 1.f / x; }
  GPUdi() static void Assign(float& x, bool mask, float v)
  {
    if (mask) {
      x = v;
    }
  }

  GPUdi() static void Assign(int32_t& x, bool mask, int32_t v)
  {
    if (mask) {
      x = v;
    }
  }

  GPUdi() void ConstrainSinPhi(float limit = GPUCA_MAX_SIN_PHI)
  {
    if (mP[2] > limit) {
      mP[2] = limit;
    } else if (mP[2] < -limit) {
      mP[2] = -limit;
    }
  }

 private:
  GPUd() int32_t initResetT0();

  float mX;        // x position
  float mTOffset;  // Z offset with early transform, T offset otherwise
  float mP[5];     // 'active' track parameters: Y, Z, SinPhi, DzDs, q/Pt
  float mC[15];    // the covariance matrix for Y,Z,SinPhi,..
  float mChi2;     // the chi^2 value
  int32_t mNDF;    // the Number of Degrees of Freedom
};

struct GPUTPCGMLoopData {
  GPUTPCGMTrackParam param;
  uint32_t track;
  float alpha;
  uint8_t sector;
  uint8_t row;
  uint8_t outwards;
};

GPUdi() int32_t GPUTPCGMTrackParam::initResetT0()
{
  const float absQPt = CAMath::Abs(mP[4]);
  if (absQPt < (150.f / 40.f)) {
    return 150.f / 40.f;
  }
  return CAMath::Max(10.f, 150.f / mP[4]);
}

GPUdi() void GPUTPCGMTrackParam::ResetCovariance()
{
  mC[0] = 100.f;
  mC[1] = 0.f;
  mC[2] = 100.f;
  mC[3] = 0.f;
  mC[4] = 0.f;
  mC[5] = 1.f;
  mC[6] = 0.f;
  mC[7] = 0.f;
  mC[8] = 0.f;
  mC[9] = 10.f;
  mC[10] = 0.f;
  mC[11] = 0.f;
  mC[12] = 0.f;
  mC[13] = 0.f;
  mC[14] = 10.f;
  mChi2 = 0;
  mNDF = -5;
}

GPUdi() float GPUTPCGMTrackParam::GetMirroredY(float Bz) const
{
  // get Y of the point which has the same X, but located on the other side of trajectory
  float qptBz = GetQPt() * Bz;
  float cosPhi2 = 1.f - GetSinPhi() * GetSinPhi();
  if (CAMath::Abs(qptBz) < 1.e-8f) {
    qptBz = 1.e-8f;
  }
  if (cosPhi2 < 0.f) {
    cosPhi2 = 0.f;
  }
  return GetY() - 2.f * CAMath::Sqrt(cosPhi2) / qptBz;
}
} // namespace o2::gpu

#endif
