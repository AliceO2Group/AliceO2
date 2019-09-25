// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCGMPropagator.h
/// \author Sergey Gorbunov, David Rohr

#ifndef GPUTPCGMPROPAGATOR_H
#define GPUTPCGMPROPAGATOR_H

#include "GPUCommonDef.h"
#include "GPUTPCGMOfflineStatisticalErrors.h"
#include "GPUTPCGMPhysicalTrackModel.h"
#include "GPUTPCGMPolynomialField.h"
#include "GPUCommonMath.h"

namespace o2
{
namespace base
{
struct MatBudget;
class MatLayerCylSet;
} // namespace base
} // namespace o2

namespace GPUCA_NAMESPACE
{
namespace gpu
{
class GPUTPCGMTrackParam;
struct GPUParam;

/**
 * @class GPUTPCGMPropagator
 *
 */

class GPUTPCGMPropagator
{
 public:
  /// Enumeration of field regions
  enum FieldRegion {
    TPC = 0, ///< TPC
    ITS = 1, ///< ITS
    TRD = 2  ///< outer TPC -> outer TRD
  };

  GPUdDefault() GPUTPCGMPropagator() CON_DEFAULT;

  struct MaterialCorrection {
    GPUd() MaterialCorrection() : radLen(29.532f), rho(1.025e-3f), rhoOverRadLen(rho / radLen), DLMax(0.f), EP2(0.f), sigmadE2(0.f), k22(0.f), k33(0.f), k43(0.f), k44(0.f) {}

    float radLen, rho, rhoOverRadLen, DLMax, EP2, sigmadE2, k22, k33, k43, k44; // precalculated values for MS and EnergyLoss correction
  };

  GPUd() void SetMaterial(float radLen, float rho);
  GPUd() o2::base::MatBudget getMatBudget(float* p1, float* p2);

  GPUd() void SetPolynomialField(const GPUTPCGMPolynomialField* field) { mField = field; }

  GPUd() void SelectFieldRegion(FieldRegion region) { mFieldRegion = region; }

  GPUd() void SetFitInProjections(bool Flag) { mFitInProjections = Flag; }
  GPUd() void SetToyMCEventsFlag(bool Flag) { mToyMCEvents = Flag; }
  GPUd() void SetSpecialErrors(bool Flag) { mSpecialErrors = Flag; }
  GPUd() void SetMatLUT(const o2::base::MatLayerCylSet* lut) { mMatLUT = lut; }

  GPUd() void SetMaxSinPhi(float maxSinPhi) { mMaxSinPhi = maxSinPhi; }

  GPUd() void SetTrack(GPUTPCGMTrackParam* track, float Alpha);
  GPUd() void ResetT0()
  {
    if (!mT) {
      return;
    }
    mT0.Set(*mT);
  }

  GPUd() int RotateToAlpha(float newAlpha);

  GPUd() int PropagateToXAlpha(float posX, float posAlpha, bool inFlyDirection);

  GPUd() int PropagateToXAlphaBz(float posX, float posAlpha, bool inFlyDirection);

  GPUd() int Update(float posY, float posZ, int iRow, const GPUParam& param, short clusterState, bool rejectChi2, bool refit);
  GPUd() int Update(float posY, float posZ, short clusterState, bool rejectChi2, float err2Y, float err2Z);
  GPUd() float PredictChi2(float posY, float posZ, int iRow, const GPUParam& param, short clusterState) const;
  GPUd() float PredictChi2(float posY, float posZ, float err2Y, float err2Z) const;
  GPUd() int RejectCluster(float chiY, float chiZ, unsigned char clusterState)
  {
    if (chiY > 9.f || chiZ > 9.f) {
      return 2;
    }
    if ((chiY > 6.25f || chiZ > 6.25f) && (clusterState & (GPUTPCGMMergedTrackHit::flagSplit | GPUTPCGMMergedTrackHit::flagShared))) {
      return 2;
    }
    if ((chiY > 1.f || chiZ > 6.25f) && (clusterState & (GPUTPCGMMergedTrackHit::flagEdge | GPUTPCGMMergedTrackHit::flagSingle))) {
      return 2;
    }
    return 0;
  }

  GPUd() float GetBz(float Alpha, float X, float Y, float Z) const;
  GPUd() void GetBxByBz(float Alpha, float X, float Y, float Z, float B[3]) const;

  GPUd() void GetErr2(float& err2Y, float& err2Z, const GPUParam& param, float posZ, int iRow, short clusterState) const;

  GPUd() float GetAlpha() const { return mAlpha; }
  GPUd() float GetQPt0() const { return mT0.GetQPt(); }
  GPUd() float GetSinPhi0() const { return mT0.GetSinPhi(); }
  GPUd() float GetCosPhi0() const { return mT0.GetCosPhi(); }
  GPUd() void Mirror(bool inFlyDirection);
  GPUd() void Rotate180();
  GPUd() void ChangeDirection();
  GPUd() float GetMirroredYModel() const;
  GPUd() float GetMirroredYTrack() const;
  GPUd() int GetPropagatedYZ(float x, float& projY, float& projZ);
  GPUd() bool GetFitInProjections() const { return mFitInProjections; }

  GPUd() GPUTPCGMPhysicalTrackModel& Model()
  {
    return mT0;
  }
  GPUd() void CalculateMaterialCorrection();
  GPUd() void SetStatErrorCurCluster(GPUTPCGMMergedTrackHit* c) { mStatErrors.SetCurCluster(c); }

 private:
  GPUd() static float ApproximateBetheBloch(float beta2);
  GPUd() int FollowLinearization(const GPUTPCGMPhysicalTrackModel& t0e, float Bz, float dLp, bool inFlyDirection);

  const GPUTPCGMPolynomialField* mField = nullptr;
  FieldRegion mFieldRegion = TPC;

  GPUTPCGMTrackParam* mT = nullptr;
  float mAlpha = 0; // rotation angle of the track coordinate system
  GPUTPCGMPhysicalTrackModel mT0;
  MaterialCorrection mMaterial;
  bool mSpecialErrors = 0;
  bool mFitInProjections = 1; // fit (Y,SinPhi,QPt) and (Z,DzDs) paramteres separatelly
  bool mToyMCEvents = 0;      // events are simulated with simple home-made simulation
  float mMaxSinPhi = GPUCA_MAX_SIN_PHI;

  GPUTPCGMOfflineStatisticalErrors mStatErrors;
  const o2::base::MatLayerCylSet* mMatLUT = nullptr;
};

GPUd() inline void GPUTPCGMPropagator::SetMaterial(float radLen, float rho)
{
  mMaterial.rho = rho;
  mMaterial.radLen = radLen;
  mMaterial.rhoOverRadLen = (radLen > 1.e-4f) ? rho / radLen : 0.f;
  CalculateMaterialCorrection();
}

GPUd() inline void GPUTPCGMPropagator::SetTrack(GPUTPCGMTrackParam* track, float Alpha)
{
  mT = track;
  if (!mT) {
    return;
  }
  mT0.Set(*mT);
  mAlpha = Alpha;
  CalculateMaterialCorrection();
}

GPUd() inline float GPUTPCGMPropagator::GetMirroredYModel() const
{
  float Bz = GetBz(mAlpha, mT0.GetX(), mT0.GetY(), mT0.GetZ());
  return mT0.GetMirroredY(Bz);
}

GPUd() inline float GPUTPCGMPropagator::GetMirroredYTrack() const
{
  if (!mT) {
    return -1.E10f;
  }
  float Bz = GetBz(mAlpha, mT->GetX(), mT->GetY(), mT->GetZ());
  return mT->GetMirroredY(Bz);
}
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
