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

/// \file GPUTRDInterfaces.h
/// \author David Rohr, Ole Schmidt

#ifndef GPUTRDINTERFACES_H
#define GPUTRDINTERFACES_H

// This is an interface header for making the TRD tracking portable between O2, and Ru2 format

#include "GPUCommonDef.h"
#include "GPUCommonMath.h"
#include "GPUTPCGMMergedTrack.h"
#include "GPUTPCGMTrackParam.h"
#include "GPUTRDDef.h"

namespace o2::gpu
{
template <typename T>
class trackInterface;
template <typename T>
class propagatorInterface;
} // namespace o2::gpu

#include "DetectorsBase/Propagator.h"
#include "GPUTRDInterfaceO2Track.h"

namespace o2::gpu
{

GPUdi() trackInterface<o2::track::TrackParCov>::trackInterface(const GPUTPCGMMergedTrack& trk) { set(trk.OuterParam().X, trk.OuterParam().alpha, trk.OuterParam().P, trk.OuterParam().C); }
GPUdi() trackInterface<o2::track::TrackParCov>::trackInterface(const gputpcgmmergertypes::GPUTPCOuterParam& param) { set(param.X, param.alpha, param.P, param.C); }

template <>
class propagatorInterface<o2::base::Propagator>
{
 public:
  typedef o2::base::Propagator propagatorParam;
  GPUd() propagatorInterface(const propagatorParam* prop) : mProp(prop) {};
  GPUd() propagatorInterface(const propagatorInterface<o2::base::Propagator>&) = delete;
  GPUd() propagatorInterface& operator=(const propagatorInterface<o2::base::Propagator>&) = delete;

  GPUdi() bool propagateToX(float x, float maxSnp, float maxStep) { return mProp->PropagateToXBxByBz(*mParam, x, maxSnp, maxStep); }
  GPUdi() int32_t getPropagatedYZ(float x, float& projY, float& projZ) { return static_cast<int32_t>(mParam->getYZAt(x, mProp->getNominalBz(), projY, projZ)); }

  GPUdi() void setTrack(trackInterface<o2::track::TrackParCov>* trk) { mParam = trk; }
  GPUdi() void setFitInProjections(bool flag) {}

  GPUdi() float getAlpha() { return (mParam) ? mParam->getAlpha() : 99999.f; }
  GPUdi() bool update(const float p[2], const float cov[3])
  {
    if (mParam) {
      std::array<float, 2> pTmp = {p[0], p[1]};
      std::array<float, 3> covTmp = {cov[0], cov[1], cov[2]};
      return mParam->update(pTmp, covTmp);
    } else {
      return false;
    }
  }
  GPUdi() float getPredictedChi2(const float p[2], const float cov[3])
  {
    if (mParam) {
      std::array<float, 2> pTmp = {p[0], p[1]};
      std::array<float, 3> covTmp = {cov[0], cov[1], cov[2]};
      return mParam->getPredictedChi2(pTmp, covTmp);
    } else {
      return 99999.f;
    }
  }
  GPUdi() bool rotate(float alpha) { return (mParam) ? mParam->rotate(alpha) : false; }

  trackInterface<o2::track::TrackParCov>* mParam{nullptr};
  const o2::base::Propagator* mProp;
};

} // namespace o2::gpu

#include "GPUTPCGMPropagator.h"
#include "GPUParam.h"
#include "GPUDef.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"

namespace o2::gpu
{

template <>
class trackInterface<GPUTPCGMTrackParam> : public GPUTPCGMTrackParam
{
 public:
  GPUdDefault() trackInterface() = default;
  GPUd() trackInterface(const GPUTPCGMTrackParam& param) = delete;
  GPUd() trackInterface(const GPUTPCGMMergedTrack& trk) : GPUTPCGMTrackParam(trk.GetParam()), mAlpha(trk.GetAlpha()) {}
  GPUd() trackInterface(const gputpcgmmergertypes::GPUTPCOuterParam& param) : GPUTPCGMTrackParam(), mAlpha(param.alpha)
  {
    SetX(param.X);
    for (int32_t i = 0; i < 5; i++) {
      SetPar(i, param.P[i]);
    }
    for (int32_t i = 0; i < 15; i++) {
      SetCov(i, param.C[i]);
    }
  };
  GPUdDefault() trackInterface(const trackInterface<GPUTPCGMTrackParam>& param) = default;
  GPUdDefault() trackInterface& operator=(const trackInterface<GPUTPCGMTrackParam>& param) = default;
  GPUd() trackInterface(const o2::dataformats::TrackTPCITS& param) : GPUTPCGMTrackParam(), mAlpha(param.getParamOut().getAlpha())
  {
    SetX(param.getParamOut().getX());
    SetPar(0, param.getParamOut().getY());
    SetPar(1, param.getParamOut().getZ());
    SetPar(2, param.getParamOut().getSnp());
    SetPar(3, param.getParamOut().getTgl());
    SetPar(4, param.getParamOut().getQ2Pt());
    for (int32_t i = 0; i < 15; i++) {
      SetCov(i, param.getParamOut().getCov()[i]);
    }
  }
  GPUd() trackInterface(const o2::tpc::TrackTPC& param) : GPUTPCGMTrackParam(), mAlpha(param.getParamOut().getAlpha())
  {
    SetX(param.getParamOut().getX());
    SetPar(0, param.getParamOut().getY());
    SetPar(1, param.getParamOut().getZ());
    SetPar(2, param.getParamOut().getSnp());
    SetPar(3, param.getParamOut().getTgl());
    SetPar(4, param.getParamOut().getQ2Pt());
    for (int32_t i = 0; i < 15; i++) {
      SetCov(i, param.getParamOut().getCov()[i]);
    }
  }

  GPUd() float getX() const
  {
    return GetX();
  }
  GPUd() float getAlpha() const { return mAlpha; }
  GPUd() float getY() const { return GetY(); }
  GPUd() float getZ() const { return GetZ(); }
  GPUd() float getSnp() const { return GetSinPhi(); }
  GPUd() float getTgl() const { return GetDzDs(); }
  GPUd() float getQ2Pt() const { return GetQPt(); }
  GPUd() float getEta() const { return -CAMath::Log(CAMath::Tan(0.5f * (0.5f * M_PI - CAMath::ATan(getTgl())))); }
  GPUd() float getPt() const { return CAMath::Abs(getQ2Pt()) > 0 ? CAMath::Abs(1.f / getQ2Pt()) : 99999.f; }
  GPUd() float getSigmaY2() const { return GetErr2Y(); }
  GPUd() float getSigmaZ2() const { return GetErr2Z(); }
  GPUd() float getSigmaZY() const { return GetCov(1); }

  GPUd() const float* getPar() const { return GetPar(); }
  GPUd() const float* getCov() const { return GetCov(); }
  GPUd() void resetCovariance(float s) { ResetCovariance(); }
  GPUd() void updateCovZ2(float addZerror) { SetCov(2, GetErr2Z() + addZerror); }
  GPUd() void setAlpha(float alpha) { mAlpha = alpha; }
  GPUd() void set(float x, float alpha, const float param[5], const float cov[15])
  {
    SetX(x);
    for (int32_t i = 0; i < 5; i++) {
      SetPar(i, param[i]);
    }
    for (int32_t j = 0; j < 15; j++) {
      SetCov(j, cov[j]);
    }
    setAlpha(alpha);
  }

  typedef GPUTPCGMTrackParam baseClass;

 private:
  float mAlpha = 0.f; // rotation along phi wrt global coordinate system

  ClassDefNV(trackInterface, 1);
};

template <>
class propagatorInterface<GPUTPCGMPropagator> : public GPUTPCGMPropagator
{
 public:
  typedef GPUTPCGMPolynomialField propagatorParam;
  GPUd() propagatorInterface(const propagatorParam* pField) : GPUTPCGMPropagator(), mTrack(nullptr)
  {
    this->SetMaterialTPC();
    this->SetPolynomialField(pField);
    this->SetMaxSinPhi(constants::MAX_SIN_PHI);
    this->SetFitInProjections(0);
    this->SelectFieldRegion(GPUTPCGMPropagator::TRD);
  };
  propagatorInterface(const propagatorInterface<GPUTPCGMPropagator>&) = delete;
  propagatorInterface& operator=(const propagatorInterface<GPUTPCGMPropagator>&) = delete;
  GPUd() void setTrack(trackInterface<GPUTPCGMTrackParam>* trk)
  {
    SetTrack(trk, trk->getAlpha());
    mTrack = trk;
  }
  GPUd() bool propagateToX(float x, float maxSnp, float maxStep)
  {
    // bool ok = PropagateToXAlpha(x, GetAlpha(), true) == 0 ? true : false;
    int32_t retVal = PropagateToXAlpha(x, GetAlpha(), true);
    bool ok = (retVal == 0) ? true : false;
    ok = mTrack->CheckNumericalQuality();
    return ok;
  }
  GPUd() int32_t getPropagatedYZ(float x, float& projY, float& projZ) { return GetPropagatedYZ(x, projY, projZ); }
  GPUd() void setFitInProjections(bool flag) { SetFitInProjections(flag); }
  GPUd() bool rotate(float alpha)
  {
    if (RotateToAlpha(alpha) == 0) {
      mTrack->setAlpha(alpha);
      return mTrack->CheckNumericalQuality();
    }
    return false;
  }
  GPUd() bool update(const float p[2], const float cov[3])
  {
    // TODO sigma_yz not taken into account yet, is not zero due to pad tilting!
    return Update(p[0], p[1], 0, false, cov[0], cov[2]) == 0 ? true : false;
  }
  GPUd() float getAlpha() { return GetAlpha(); }
  // TODO sigma_yz not taken into account yet, is not zero due to pad tilting!
  GPUd() float getPredictedChi2(const float p[2], const float cov[3]) const { return PredictChi2(p[0], p[1], cov[0], cov[2]); }

  trackInterface<GPUTPCGMTrackParam>* mTrack;
};
} // namespace o2::gpu

#endif // GPUTRDINTERFACES_H
