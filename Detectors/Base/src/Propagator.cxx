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

#include "DetectorsBase/Propagator.h"
#include "GPUCommonLogger.h"
#include "GPUCommonConstants.h"
#include "GPUCommonMath.h"
#include "GPUTPCGMPolynomialField.h"
#include "MathUtils/Utils.h"
#include "ReconstructionDataFormats/Vertex.h"

using namespace o2::base;
using namespace o2::gpu;

#if !defined(GPUCA_GPUCODE)
#include "Field/MagFieldFast.h" // Don't use this on the GPU
#endif

#if !defined(GPUCA_STANDALONE) && !defined(GPUCA_GPUCODE)
#include "Field/MagneticField.h"
#include "DataFormatsParameters/GRPObject.h"
#include "DataFormatsParameters/GRPMagField.h"
#include "DetectorsBase/GeometryManager.h"
#include <FairRunAna.h> // eventually will get rid of it
#include <TGeoGlobalMagField.h>

template <typename value_T>
PropagatorImpl<value_T>::PropagatorImpl(bool uninitialized)
{
  if (uninitialized) {
    return;
  }
  ///< construct checking if needed components were initialized
  updateField();
}

//____________________________________________________________
template <typename value_T>
void PropagatorImpl<value_T>::updateField()
{
  if (!mField) {
    mField = static_cast<o2::field::MagneticField*>(TGeoGlobalMagField::Instance()->GetField());
    if (!mField) {
      LOG(warning) << "No Magnetic Field in TGeoGlobalMagField, checking legacy FairRunAna";
      mField = dynamic_cast<o2::field::MagneticField*>(FairRunAna::Instance()->GetField());
    }
    if (!mField) {
      LOG(fatal) << "Magnetic field is not initialized!";
    }
    if (!mField->getFastField() && mField->fastFieldExists()) {
      mField->AllowFastField(true);
      mFieldFast = mField->getFastField();
    }
  }
  const value_type xyz[3] = {0.};
  if (mFieldFast) {
    mFieldFast->GetBz(xyz, mBz);
  } else {
    mBz = mField->GetBz(xyz[0], xyz[1], xyz[2]);
  }
}

//____________________________________________________________
template <typename value_T>
int PropagatorImpl<value_T>::initFieldFromGRP(const std::string grpFileName, bool verbose)
{
  /// load grp and init magnetic field
  if (verbose) {
    LOG(info) << "Loading field from GRP of " << grpFileName;
  }
  const auto grp = o2::parameters::GRPObject::loadFrom(grpFileName);
  if (!grp) {
    return -1;
  }
  if (verbose) {
    grp->print();
  }

  return initFieldFromGRP(grp);
}

//____________________________________________________________
template <typename value_T>
int PropagatorImpl<value_T>::initFieldFromGRP(const o2::parameters::GRPObject* grp, bool verbose)
{
  /// init mag field from GRP data and attach it to TGeoGlobalMagField

  if (TGeoGlobalMagField::Instance()->IsLocked()) {
    if (TGeoGlobalMagField::Instance()->GetField()->TestBit(o2::field::MagneticField::kOverrideGRP)) {
      LOG(warning) << "ExpertMode!!! GRP information will be ignored";
      LOG(warning) << "ExpertMode!!! Running with the externally locked B field";
      return 0;
    } else {
      LOG(info) << "Destroying existing B field instance";
      delete TGeoGlobalMagField::Instance();
    }
  }
  auto fld = o2::field::MagneticField::createFieldMap(grp->getL3Current(), grp->getDipoleCurrent(), o2::field::MagneticField::kConvLHC, grp->getFieldUniformity());
  TGeoGlobalMagField::Instance()->SetField(fld);
  TGeoGlobalMagField::Instance()->Lock();
  if (verbose) {
    LOG(info) << "Running with the B field constructed out of GRP";
    LOG(info) << "Access field via TGeoGlobalMagField::Instance()->Field(xyz,bxyz) or via";
    LOG(info) << "auto o2field = static_cast<o2::field::MagneticField*>( TGeoGlobalMagField::Instance()->GetField() )";
  }
  return 0;
}

//____________________________________________________________
template <typename value_T>
int PropagatorImpl<value_T>::initFieldFromGRP(const o2::parameters::GRPMagField* grp, bool verbose)
{
  /// init mag field from GRP data and attach it to TGeoGlobalMagField

  if (TGeoGlobalMagField::Instance()->IsLocked()) {
    if (TGeoGlobalMagField::Instance()->GetField()->TestBit(o2::field::MagneticField::kOverrideGRP)) {
      LOG(warning) << "ExpertMode!!! GRP information will be ignored";
      LOG(warning) << "ExpertMode!!! Running with the externally locked B field";
      return 0;
    } else {
      LOG(info) << "Destroying existing B field instance";
      delete TGeoGlobalMagField::Instance();
    }
  }
  auto fld = o2::field::MagneticField::createFieldMap(grp->getL3Current(), grp->getDipoleCurrent(), o2::field::MagneticField::kConvLHC, grp->getFieldUniformity());
  TGeoGlobalMagField::Instance()->SetField(fld);
  TGeoGlobalMagField::Instance()->Lock();
  if (verbose) {
    LOG(info) << "Running with the B field constructed out of GRP";
    LOG(info) << "Access field via TGeoGlobalMagField::Instance()->Field(xyz,bxyz) or via";
    LOG(info) << "auto o2field = static_cast<o2::field::MagneticField*>( TGeoGlobalMagField::Instance()->GetField() )";
  }
  return 0;
}

#elif !defined(GPUCA_GPUCODE)
template <typename value_T>
PropagatorImpl<value_T>::PropagatorImpl(bool uninitialized)
{
} // empty dummy constructor for standalone benchmark
#endif

//_______________________________________________________________________
template <typename value_T>
GPUd() bool PropagatorImpl<value_T>::PropagateToXBxByBz(TrackParCov_t& track, value_type xToGo, value_type maxSnp, value_type maxStep,
                                                        PropagatorImpl<value_T>::MatCorrType matCorr, track::TrackLTIntegral* tofInfo, int signCorr) const
{
  //----------------------------------------------------------------
  //
  // Propagates the track to the plane X=xk (cm)
  // taking into account all the three components of the magnetic field
  // and correcting for the crossed material.
  //
  // maxStep  - maximal step for propagation
  // tofInfo  - optional container for track length and PID-dependent TOF integration
  //
  // matCorr  - material correction type, it is up to the user to make sure the pointer is attached (if LUT is requested)
  //----------------------------------------------------------------
  auto dx = xToGo - track.getX();
  int dir = dx > 0.f ? 1 : -1;
  if (!signCorr) {
    signCorr = -dir; // sign of eloss correction is not imposed
  }

  gpu::gpustd::array<value_type, 3> b{};
  while (math_utils::detail::abs<value_type>(dx) > Epsilon) {
    auto step = math_utils::detail::min<value_type>(math_utils::detail::abs<value_type>(dx), maxStep);
    if (dir < 0) {
      step = -step;
    }
    auto x = track.getX() + step;
    auto xyz0 = track.getXYZGlo();
    getFieldXYZ(xyz0, &b[0]);

    if (!track.propagateTo(x, b)) {
      return false;
    }
    if (maxSnp > 0 && math_utils::detail::abs<value_type>(track.getSnp()) >= maxSnp) {
      return false;
    }
    if (matCorr != MatCorrType::USEMatCorrNONE) {
      auto xyz1 = track.getXYZGlo();
      auto mb = getMatBudget(matCorr, xyz0, xyz1);
      if (!track.correctForMaterial(mb.meanX2X0, mb.getXRho(signCorr))) {
        return false;
      }

      if (tofInfo) {
        tofInfo->addStep(mb.length, track.getP2Inv()); // fill L,ToF info using already calculated step length
        tofInfo->addX2X0(mb.meanX2X0);
        tofInfo->addXRho(mb.getXRho(signCorr));
      }
    } else if (tofInfo) { // if tofInfo filling was requested w/o material correction, we need to calculate the step lenght
      auto xyz1 = track.getXYZGlo();
      math_utils::Vector3D<value_type> stepV(xyz1.X() - xyz0.X(), xyz1.Y() - xyz0.Y(), xyz1.Z() - xyz0.Z());
      tofInfo->addStep(stepV.R(), track.getP2Inv());
    }
    dx = xToGo - track.getX();
  }
  track.setX(xToGo);
  return true;
}

//_______________________________________________________________________
template <typename value_T>
GPUd() bool PropagatorImpl<value_T>::PropagateToXBxByBz(TrackPar_t& track, value_type xToGo, value_type maxSnp, value_type maxStep,
                                                        PropagatorImpl<value_T>::MatCorrType matCorr, track::TrackLTIntegral* tofInfo, int signCorr) const
{
  //----------------------------------------------------------------
  //
  // Propagates the track params to the plane X=xk (cm), NO error evaluation
  // taking into account all the three components of the magnetic field
  // and optionally correcting for the e.loss crossed material.
  //
  // maxStep  - maximal step for propagation
  // tofInfo  - optional container for track length and PID-dependent TOF integration
  //
  // matCorr  - material correction type, it is up to the user to make sure the pointer is attached (if LUT is requested)
  //----------------------------------------------------------------
  auto dx = xToGo - track.getX();
  int dir = dx > 0.f ? 1 : -1;
  if (!signCorr) {
    signCorr = -dir; // sign of eloss correction is not imposed
  }

  gpu::gpustd::array<value_type, 3> b{};
  while (math_utils::detail::abs<value_type>(dx) > Epsilon) {
    auto step = math_utils::detail::min<value_type>(math_utils::detail::abs<value_type>(dx), maxStep);
    if (dir < 0) {
      step = -step;
    }
    auto x = track.getX() + step;
    auto xyz0 = track.getXYZGlo();
    getFieldXYZ(xyz0, &b[0]);

    if (!track.propagateParamTo(x, b)) {
      return false;
    }
    if (maxSnp > 0 && math_utils::detail::abs<value_type>(track.getSnp()) >= maxSnp) {
      return false;
    }
    if (matCorr != MatCorrType::USEMatCorrNONE) {
      auto xyz1 = track.getXYZGlo();
      auto mb = getMatBudget(matCorr, xyz0, xyz1);
      if (!track.correctForELoss(((signCorr < 0) ? -mb.length : mb.length) * mb.meanRho)) {
        return false;
      }
      if (tofInfo) {
        tofInfo->addStep(mb.length, track.getP2Inv()); // fill L,ToF info using already calculated step length
        tofInfo->addX2X0(mb.meanX2X0);
        tofInfo->addXRho(mb.getXRho(signCorr));
      }
    } else if (tofInfo) { // if tofInfo filling was requested w/o material correction, we need to calculate the step lenght
      auto xyz1 = track.getXYZGlo();
      math_utils::Vector3D<value_type> stepV(xyz1.X() - xyz0.X(), xyz1.Y() - xyz0.Y(), xyz1.Z() - xyz0.Z());
      tofInfo->addStep(stepV.R(), track.getP2Inv());
    }
    dx = xToGo - track.getX();
  }
  track.setX(xToGo);
  return true;
}

//_______________________________________________________________________
template <typename value_T>
GPUd() bool PropagatorImpl<value_T>::propagateToX(TrackParCov_t& track, value_type xToGo, value_type bZ, value_type maxSnp, value_type maxStep,
                                                  PropagatorImpl<value_T>::MatCorrType matCorr, track::TrackLTIntegral* tofInfo, int signCorr) const
{
  //----------------------------------------------------------------
  //
  // Propagates the track to the plane X=xk (cm)
  // taking into account all the three components of the magnetic field
  // and correcting for the crossed material.
  //
  // maxStep  - maximal step for propagation
  // tofInfo  - optional container for track length and PID-dependent TOF integration
  //
  // matCorr  - material correction type, it is up to the user to make sure the pointer is attached (if LUT is requested)
  //----------------------------------------------------------------
  auto dx = xToGo - track.getX();
  int dir = dx > 0.f ? 1 : -1;
  if (!signCorr) {
    signCorr = -dir; // sign of eloss correction is not imposed
  }

  while (math_utils::detail::abs<value_type>(dx) > Epsilon) {
    auto step = math_utils::detail::min<value_type>(math_utils::detail::abs<value_type>(dx), maxStep);
    if (dir < 0) {
      step = -step;
    }
    auto x = track.getX() + step;
    auto xyz0 = track.getXYZGlo();

    if (!track.propagateTo(x, bZ)) {
      return false;
    }
    if (maxSnp > 0 && math_utils::detail::abs<value_type>(track.getSnp()) >= maxSnp) {
      return false;
    }
    if (matCorr != MatCorrType::USEMatCorrNONE) {
      auto xyz1 = track.getXYZGlo();
      auto mb = getMatBudget(matCorr, xyz0, xyz1);
      //
      if (!track.correctForMaterial(mb.meanX2X0, mb.getXRho(signCorr))) {
        return false;
      }

      if (tofInfo) {
        tofInfo->addStep(mb.length, track.getP2Inv()); // fill L,ToF info using already calculated step length
        tofInfo->addX2X0(mb.meanX2X0);
        tofInfo->addXRho(mb.getXRho(signCorr));
      }
    } else if (tofInfo) { // if tofInfo filling was requested w/o material correction, we need to calculate the step lenght
      auto xyz1 = track.getXYZGlo();
      math_utils::Vector3D<value_type> stepV(xyz1.X() - xyz0.X(), xyz1.Y() - xyz0.Y(), xyz1.Z() - xyz0.Z());
      tofInfo->addStep(stepV.R(), track.getP2Inv());
    }
    dx = xToGo - track.getX();
  }
  track.setX(xToGo);
  return true;
}

//_______________________________________________________________________
template <typename value_T>
GPUd() bool PropagatorImpl<value_T>::propagateToX(TrackPar_t& track, value_type xToGo, value_type bZ, value_type maxSnp, value_type maxStep,
                                                  PropagatorImpl<value_T>::MatCorrType matCorr, track::TrackLTIntegral* tofInfo, int signCorr) const
{
  //----------------------------------------------------------------
  //
  // Propagates the track parameters only to the plane X=xk (cm)
  // taking into account all the three components of the magnetic field
  // and correcting for the crossed material.
  //
  // maxStep  - maximal step for propagation
  // tofInfo  - optional container for track length and PID-dependent TOF integration
  //
  // matCorr  - material correction type, it is up to the user to make sure the pointer is attached (if LUT is requested)
  //----------------------------------------------------------------
  auto dx = xToGo - track.getX();
  int dir = dx > 0.f ? 1 : -1;
  if (!signCorr) {
    signCorr = -dir; // sign of eloss correction is not imposed
  }

  while (math_utils::detail::abs<value_type>(dx) > Epsilon) {
    auto step = math_utils::detail::min<value_type>(math_utils::detail::abs<value_type>(dx), maxStep);
    if (dir < 0) {
      step = -step;
    }
    auto x = track.getX() + step;
    auto xyz0 = track.getXYZGlo();

    if (!track.propagateParamTo(x, bZ)) {
      return false;
    }
    if (maxSnp > 0 && math_utils::detail::abs<value_type>(track.getSnp()) >= maxSnp) {
      return false;
    }
    if (matCorr != MatCorrType::USEMatCorrNONE) {
      auto xyz1 = track.getXYZGlo();
      auto mb = getMatBudget(matCorr, xyz0, xyz1);
      //
      if (!track.correctForELoss(mb.getXRho(signCorr))) {
        return false;
      }

      if (tofInfo) {
        tofInfo->addStep(mb.length, track.getP2Inv()); // fill L,ToF info using already calculated step length
        tofInfo->addX2X0(mb.meanX2X0);
        tofInfo->addXRho(mb.getXRho(signCorr));
      }
    } else if (tofInfo) { // if tofInfo filling was requested w/o material correction, we need to calculate the step lenght
      auto xyz1 = track.getXYZGlo();
      math_utils::Vector3D<value_type> stepV(xyz1.X() - xyz0.X(), xyz1.Y() - xyz0.Y(), xyz1.Z() - xyz0.Z());
      tofInfo->addStep(stepV.R(), track.getP2Inv());
    }
    dx = xToGo - track.getX();
  }
  track.setX(xToGo);
  return true;
}

//_______________________________________________________________________
template <typename value_T>
template <typename track_T>
GPUd() bool PropagatorImpl<value_T>::propagateToAlphaX(track_T& track, value_type alpha, value_type x, bool bzOnly, value_type maxSnp, value_type maxStep, int minSteps,
                                                       MatCorrType matCorr, track::TrackLTIntegral* tofInfo, int signCorr) const
{
  // propagate to alpha,X, if needed in a few steps
  auto snp = track.getSnpAt(alpha, x, getNominalBz());
  // apply safety factor 0.9 for crude rotation estimate
  if (math_utils::detail::abs<value_type>(snp) < maxSnp * 0.9 && track.rotate(alpha)) {
    auto dx = math_utils::detail::abs<value_type>(x - track.getX());
    if (dx < Epsilon) {
      return true;
    }
    return propagateTo(track, x, bzOnly, maxSnp, math_utils::detail::min<value_type>(dx / minSteps, maxStep), matCorr, tofInfo, signCorr);
  }
  return false;
  /*
  // try to go in a few steps with intermediate rotations


  auto alphaTrg = alpha;
  math_utils::detail::bringToPMPi<value_type>(alphaTrg);
  auto alpCurr = track.getAlpha();
  math_utils::detail::bringToPMPi<value_type>(alpCurr);
  int nsteps = minSteps > 2 ? minSteps : 2;
  auto dalp = math_utils::detail::deltaPhiSmall<value_type>(alpCurr, alphaTrg) / nsteps; // effective  (alpha - alphaCurr)/nsteps
  auto xtmp = (track.getX() + x) / nsteps;
  return track.rotate(alpCurr + dalp) && propagateTo(track, xtmp, bzOnly, maxSnp, maxStep, matCorr, tofInfo, signCorr) &&
         track.rotate(alpha) && propagateTo(track, x, bzOnly, maxSnp, maxStep, matCorr, tofInfo, signCorr);
  */
}

//_______________________________________________________________________
template <typename value_T>
GPUd() bool PropagatorImpl<value_T>::propagateToDCA(const o2::dataformats::VertexBase& vtx, TrackParCov_t& track, value_type bZ,
                                                    value_type maxStep, PropagatorImpl<value_type>::MatCorrType matCorr,
                                                    o2::dataformats::DCA* dca, track::TrackLTIntegral* tofInfo,
                                                    int signCorr, value_type maxD) const
{
  // propagate track to DCA to the vertex
  value_type sn, cs, alp = track.getAlpha();
  math_utils::detail::sincos<value_type>(alp, sn, cs);
  value_type x = track.getX(), y = track.getY(), snp = track.getSnp(), csp = math_utils::detail::sqrt<value_type>((1.f - snp) * (1.f + snp));
  value_type xv = vtx.getX() * cs + vtx.getY() * sn, yv = -vtx.getX() * sn + vtx.getY() * cs, zv = vtx.getZ();
  x -= xv;
  y -= yv;
  // Estimate the impact parameter neglecting the track curvature
  value_type d = math_utils::detail::abs<value_type>(x * snp - y * csp);
  if (d > maxD) {
    return false;
  }
  value_type crv = track.getCurvature(bZ);
  value_type tgfv = -(crv * x - snp) / (crv * y + csp);
  sn = tgfv / math_utils::detail::sqrt<value_type>(1.f + tgfv * tgfv);
  cs = math_utils::detail::sqrt<value_type>((1. - sn) * (1. + sn));
  cs = (math_utils::detail::abs<value_type>(tgfv) > o2::constants::math::Almost0) ? sn / tgfv : o2::constants::math::Almost1;

  x = xv * cs + yv * sn;
  yv = -xv * sn + yv * cs;
  xv = x;

  auto tmpT(track); // operate on the copy to recover after the failure
  alp += math_utils::detail::asin<value_type>(sn);
  if (!tmpT.rotate(alp) || !propagateToX(tmpT, xv, bZ, 0.85, maxStep, matCorr, tofInfo, signCorr)) {
#ifndef GPUCA_ALIGPUCODE
    LOG(debug) << "failed to propagate to alpha=" << alp << " X=" << xv << vtx << " | Track is: " << tmpT.asString();
#elif !defined(GPUCA_NO_FMT)
    LOG(debug) << "failed to propagate to alpha=" << alp << " X=" << xv << vtx;
#endif
    return false;
  }
  track = tmpT;
  if (dca) {
    math_utils::detail::sincos<value_type>(alp, sn, cs);
    auto s2ylocvtx = vtx.getSigmaX2() * sn * sn + vtx.getSigmaY2() * cs * cs - 2. * vtx.getSigmaXY() * cs * sn;
    dca->set(track.getY() - yv, track.getZ() - zv,
             track.getSigmaY2() + s2ylocvtx, track.getSigmaZY(), track.getSigmaZ2() + vtx.getSigmaZ2());
  }
  return true;
}

//_______________________________________________________________________
template <typename value_T>
GPUd() bool PropagatorImpl<value_T>::propagateToDCABxByBz(const o2::dataformats::VertexBase& vtx, TrackParCov_t& track,
                                                          value_type maxStep, PropagatorImpl<value_type>::MatCorrType matCorr,
                                                          o2::dataformats::DCA* dca, track::TrackLTIntegral* tofInfo,
                                                          int signCorr, value_type maxD) const
{
  // propagate track to DCA to the vertex
  value_type sn, cs, alp = track.getAlpha();
  math_utils::detail::sincos<value_type>(alp, sn, cs);
  value_type x = track.getX(), y = track.getY(), snp = track.getSnp(), csp = math_utils::detail::sqrt<value_type>((1.f - snp) * (1.f + snp));
  value_type xv = vtx.getX() * cs + vtx.getY() * sn, yv = -vtx.getX() * sn + vtx.getY() * cs, zv = vtx.getZ();
  x -= xv;
  y -= yv;
  // Estimate the impact parameter neglecting the track curvature
  value_type d = math_utils::detail::abs<value_type>(x * snp - y * csp);
  if (d > maxD) {
    return false;
  }
  value_type crv = track.getCurvature(mBz);
  value_type tgfv = -(crv * x - snp) / (crv * y + csp);
  sn = tgfv / math_utils::detail::sqrt<value_type>(1.f + tgfv * tgfv);
  cs = math_utils::detail::sqrt<value_type>((1. - sn) * (1. + sn));
  cs = (math_utils::detail::abs<value_type>(tgfv) > o2::constants::math::Almost0) ? sn / tgfv : o2::constants::math::Almost1;

  x = xv * cs + yv * sn;
  yv = -xv * sn + yv * cs;
  xv = x;

  auto tmpT(track); // operate on the copy to recover after the failure
  alp += math_utils::detail::asin<value_type>(sn);
  if (!tmpT.rotate(alp) || !PropagateToXBxByBz(tmpT, xv, 0.85, maxStep, matCorr, tofInfo, signCorr)) {
#ifndef GPUCA_ALIGPUCODE
    LOG(debug) << "failed to propagate to alpha=" << alp << " X=" << xv << vtx << " | Track is: " << tmpT.asString();
#elif !defined(GPUCA_NO_FMT)
    LOG(debug) << "failed to propagate to alpha=" << alp << " X=" << xv << vtx;
#endif
    return false;
  }
  track = tmpT;
  if (dca) {
    math_utils::detail::sincos<value_type>(alp, sn, cs);
    auto s2ylocvtx = vtx.getSigmaX2() * sn * sn + vtx.getSigmaY2() * cs * cs - 2. * vtx.getSigmaXY() * cs * sn;
    dca->set(track.getY() - yv, track.getZ() - zv,
             track.getSigmaY2() + s2ylocvtx, track.getSigmaZY(), track.getSigmaZ2() + vtx.getSigmaZ2());
  }
  return true;
}

//_______________________________________________________________________
template <typename value_T>
GPUd() bool PropagatorImpl<value_T>::propagateToDCA(const math_utils::Point3D<value_type>& vtx, TrackPar_t& track, value_type bZ,
                                                    value_type maxStep, PropagatorImpl<value_T>::MatCorrType matCorr,
                                                    gpu::gpustd::array<value_type, 2>* dca, track::TrackLTIntegral* tofInfo,
                                                    int signCorr, value_type maxD) const
{
  // propagate track to DCA to the vertex
  value_type sn, cs, alp = track.getAlpha();
  math_utils::detail::sincos<value_type>(alp, sn, cs);
  value_type x = track.getX(), y = track.getY(), snp = track.getSnp(), csp = math_utils::detail::sqrt<value_type>((1.f - snp) * (1.f + snp));
  value_type xv = vtx.X() * cs + vtx.Y() * sn, yv = -vtx.X() * sn + vtx.Y() * cs, zv = vtx.Z();
  x -= xv;
  y -= yv;
  // Estimate the impact parameter neglecting the track curvature
  value_type d = math_utils::detail::abs<value_type>(x * snp - y * csp);
  if (d > maxD) {
    return false;
  }
  value_type crv = track.getCurvature(bZ);
  value_type tgfv = -(crv * x - snp) / (crv * y + csp);
  sn = tgfv / math_utils::detail::sqrt<value_type>(1.f + tgfv * tgfv);
  cs = math_utils::detail::sqrt<value_type>((1. - sn) * (1. + sn));
  cs = (math_utils::detail::abs<value_type>(tgfv) > o2::constants::math::Almost0) ? sn / tgfv : o2::constants::math::Almost1;

  x = xv * cs + yv * sn;
  yv = -xv * sn + yv * cs;
  xv = x;

  auto tmpT(track); // operate on the copy to recover after the failure
  alp += math_utils::detail::asin<value_type>(sn);
  if (!tmpT.rotateParam(alp) || !propagateToX(tmpT, xv, bZ, 0.85, maxStep, matCorr, tofInfo, signCorr)) {
#ifndef GPUCA_ALIGPUCODE
    LOG(debug) << "failed to propagate to alpha=" << alp << " X=" << xv << " for vertex "
               << vtx.X() << ' ' << vtx.Y() << ' ' << vtx.Z() << " | Track is: " << tmpT.asString();
#else
    LOG(debug) << "failed to propagate to alpha=" << alp << " X=" << xv << " for vertex " << vtx.X() << ' ' << vtx.Y() << ' ' << vtx.Z();
#endif
    return false;
  }
  track = tmpT;
  if (dca) {
    (*dca)[0] = track.getY() - yv;
    (*dca)[1] = track.getZ() - zv;
  }
  return true;
}

//_______________________________________________________________________
template <typename value_T>
GPUd() bool PropagatorImpl<value_T>::propagateToDCABxByBz(const math_utils::Point3D<value_type>& vtx, TrackPar_t& track,
                                                          value_type maxStep, PropagatorImpl<value_T>::MatCorrType matCorr,
                                                          gpu::gpustd::array<value_type, 2>* dca, track::TrackLTIntegral* tofInfo,
                                                          int signCorr, value_type maxD) const
{
  // propagate track to DCA to the vertex
  value_type sn, cs, alp = track.getAlpha();
  math_utils::detail::sincos<value_type>(alp, sn, cs);
  value_type x = track.getX(), y = track.getY(), snp = track.getSnp(), csp = math_utils::detail::sqrt<value_type>((1.f - snp) * (1.f + snp));
  value_type xv = vtx.X() * cs + vtx.Y() * sn, yv = -vtx.X() * sn + vtx.Y() * cs, zv = vtx.Z();
  x -= xv;
  y -= yv;
  // Estimate the impact parameter neglecting the track curvature
  value_type d = math_utils::detail::abs<value_type>(x * snp - y * csp);
  if (d > maxD) {
    return false;
  }
  value_type crv = track.getCurvature(mBz);
  value_type tgfv = -(crv * x - snp) / (crv * y + csp);
  sn = tgfv / math_utils::detail::sqrt<value_type>(1.f + tgfv * tgfv);
  cs = math_utils::detail::sqrt<value_type>((1. - sn) * (1. + sn));
  cs = (math_utils::detail::abs<value_type>(tgfv) > o2::constants::math::Almost0) ? sn / tgfv : o2::constants::math::Almost1;

  x = xv * cs + yv * sn;
  yv = -xv * sn + yv * cs;
  xv = x;

  auto tmpT(track); // operate on the copy to recover after the failure
  alp += math_utils::detail::asin<value_type>(sn);
  if (!tmpT.rotateParam(alp) || !PropagateToXBxByBz(tmpT, xv, 0.85, maxStep, matCorr, tofInfo, signCorr)) {
#ifndef GPUCA_ALIGPUCODE
    LOG(debug) << "failed to propagate to alpha=" << alp << " X=" << xv << " for vertex "
               << vtx.X() << ' ' << vtx.Y() << ' ' << vtx.Z() << " | Track is: " << tmpT.asString();
#else
    LOG(debug) << "failed to propagate to alpha=" << alp << " X=" << xv << " for vertex " << vtx.X() << ' ' << vtx.Y() << ' ' << vtx.Z();
#endif
    return false;
  }
  track = tmpT;
  if (dca) {
    (*dca)[0] = track.getY() - yv;
    (*dca)[1] = track.getZ() - zv;
  }
  return true;
}

//____________________________________________________________
template <typename value_T>
GPUd() float PropagatorImpl<value_T>::estimateLTIncrement(const o2::track::TrackParametrization<value_type>& trc,
                                                          const o2::math_utils::Point3D<value_type>& posStart,
                                                          const o2::math_utils::Point3D<value_type>& posEnd) const
{
  // estimate helical step increment between 2 point
  float dX = posEnd.X() - posStart.X(), dY = posEnd.Y() - posStart.Y(), dZ = posEnd.Z() - posStart.Z(), d2XY = dX * dX + dY * dY;
  if (getNominalBz() != 0) { // circular arc = 2*R*asin(dXY/2R)
    float b[3];
    o2::math_utils::Point3D<float> posAv(0.5 * (posEnd.X() + posStart.X()), 0.5 * (posEnd.Y() + posStart.Y()), 0.5 * (posEnd.Z() + posStart.Z()));
    getFieldXYZ(posAv, b);
    float curvH = math_utils::detail::abs<value_type>(0.5f * trc.getCurvature(b[2])), asArg = curvH * math_utils::detail::sqrt<value_type>(d2XY);
    if (curvH > 0.f) {
      d2XY = asArg < 1.f ? math_utils::detail::asin<value_type>(asArg) / curvH : o2::constants::math::PIHalf / curvH;
      d2XY *= d2XY;
    }
  }
  return math_utils::detail::sqrt<value_type>(d2XY + dZ * dZ);
}

//____________________________________________________________
template <typename value_T>
GPUd() void PropagatorImpl<value_T>::estimateLTFast(o2::track::TrackLTIntegral& lt, const o2::track::TrackParametrization<value_type>& trc) const
{
  value_T xdca = 0., ydca = 0., length = 0.; // , zdca = 0. // zdca might be used in future
  o2::math_utils::CircleXY<value_T> c;
  constexpr float TinyF = 1e-9;
  auto straigh_line_approx = [&]() {
    auto csp2 = (1.f - trc.getSnp()) * (1.f + trc.getSnp());
    if (csp2 > TinyF) {
      auto csp = math_utils::detail::sqrt<value_type>(csp2);
      auto tgp = trc.getSnp() / csp, f = trc.getX() * tgp - trc.getY();
      xdca = tgp * f * csp2;
      ydca = -f * csp2;
      auto dx = xdca - trc.getX(), dy = ydca - trc.getY(), dz = dx * trc.getTgl() / csp;
      return math_utils::detail::sqrt<value_type>(dx * dx + dy * dy + dz * dz);
    } else {                                                                                                                           //  track is parallel to Y axis
      xdca = trc.getX();                                                                                                               // ydca = 0
      return math_utils::detail::abs<value_type>(trc.getY() * math_utils::detail::sqrt<value_type>(1. + trc.getTgl() * trc.getTgl())); // distance from the current point to DCA
    }
  };
  trc.getCircleParamsLoc(mBz, c);
  if (c.rC != 0.) {                                                     // helix
    auto distC = math_utils::detail::sqrt<value_type>(c.getCenterD2()); // distance from the circle center to origin
    if (distC > 1.e-3) {
      auto nrm = (distC - c.rC) / distC;
      xdca = nrm * c.xC; // coordinates of the DCA to 0,0 in the local frame
      ydca = nrm * c.yC;
      auto v0x = trc.getX() - c.xC, v0y = trc.getY() - c.yC, v1x = xdca - c.xC, v1y = ydca - c.yC;
      auto angcos = (v0x * v1x + v0y * v1y) / (c.rC * c.rC);
      if (math_utils::detail::abs<value_type>(angcos) < 1.f) {
        auto ang = math_utils::detail::acos<value_type>(angcos);
        if ((trc.getSign() > 0.f) == (mBz > 0.f)) {
          ang = -ang;   // we need signeg angle
          c.rC = -c.rC; // we need signed curvature for zdca
        }
        // zdca = trc.getZ() + (trc.getSign() > 0. ? c.rC : -c.rC) * trc.getTgl() * ang;
        length = math_utils::detail::abs<value_type>(c.rC * ang * math_utils::detail::sqrt<value_type>(1. + trc.getTgl() * trc.getTgl()));
      } else { // calculation of the arc length between the position and DCA makes no sense
        length = straigh_line_approx();
      }
    } else { // track with circle center at the origin, and LT makes no sense, take direct distance
      xdca = trc.getX();
      ydca = trc.getY();
    }
  } else { // straight line
    length = straigh_line_approx();
  }
  // since we assume the track or its parent comes from the beam-line or decay, add XY(?) distance to it
  length += math_utils::detail::sqrt<value_type>(xdca * xdca + ydca * ydca);
  lt.addStep(length, trc.getP2Inv());
}

//____________________________________________________________
template <typename value_T>
GPUd() MatBudget PropagatorImpl<value_T>::getMatBudget(PropagatorImpl<value_type>::MatCorrType corrType, const math_utils::Point3D<value_type>& p0, const math_utils::Point3D<value_type>& p1) const
{
#if !defined(GPUCA_STANDALONE) && !defined(GPUCA_GPUCODE)
  if (corrType == MatCorrType::USEMatCorrTGeo) {
    return GeometryManager::meanMaterialBudget(p0, p1);
  }
  if (!mMatLUT) {
    if (mTGeoFallBackAllowed) {
      return GeometryManager::meanMaterialBudget(p0, p1);
    } else {
      throw std::runtime_error("requested MatLUT is absent and fall-back to TGeo is disabled");
    }
  }
#endif
  return mMatLUT->getMatBudget(p0.X(), p0.Y(), p0.Z(), p1.X(), p1.Y(), p1.Z());
}

template <typename value_T>
template <typename T>
GPUd() void PropagatorImpl<value_T>::getFieldXYZImpl(const math_utils::Point3D<T> xyz, T* bxyz) const
{
  if (mGPUField) {
#if defined(GPUCA_GPUCODE_DEVICE) && defined(GPUCA_HAS_GLOBAL_SYMBOL_CONSTANT_MEM)
    const auto* f = &GPUCA_CONSMEM.param.polynomialField; // Access directly from constant memory on GPU (copied here to avoid complicated header dependencies)
#else
    const auto* f = mGPUField;
#endif
    float bxyzF[3] = {};
    f->GetField(xyz.X(), xyz.Y(), xyz.Z(), bxyzF);
    // copy and convert
    constexpr value_type kCLight1 = 1. / o2::gpu::gpu_common_constants::kCLight;
    for (uint i = 0; i < 3; ++i) {
      bxyz[i] = static_cast<value_type>(bxyzF[i]) * kCLight1;
    }
  } else {
#ifndef GPUCA_GPUCODE
    if (mFieldFast) {
      mFieldFast->Field(xyz, bxyz); // Must not call the host-only function in GPU compilation
    } else {
#ifdef GPUCA_STANDALONE
      LOG(fatal) << "Normal field cannot be used in standalone benchmark";
#else
      mField->field(xyz, bxyz);
#endif
    }
#endif
  }
}

template <typename value_T>
GPUd() void PropagatorImpl<value_T>::getFieldXYZ(const math_utils::Point3D<float> xyz, float* bxyz) const
{
  getFieldXYZImpl<float>(xyz, bxyz);
}

template <typename value_T>
GPUd() void PropagatorImpl<value_T>::getFieldXYZ(const math_utils::Point3D<double> xyz, double* bxyz) const
{
  getFieldXYZImpl<double>(xyz, bxyz);
}

namespace o2::base
{
template class PropagatorImpl<float>;
#ifndef GPUCA_GPUCODE_DEVICE
template class PropagatorImpl<double>;
#endif
#ifndef __HIPCC__ // TODO: Fixme: must prevent HIP from compiling this, should file bug report
template bool GPUd() PropagatorImpl<float>::propagateToAlphaX<PropagatorImpl<float>::TrackPar_t>(PropagatorImpl<float>::TrackPar_t&, float, float, bool, float, float, int, PropagatorImpl<float>::MatCorrType matCorr, track::TrackLTIntegral*, int) const;
template bool GPUd() PropagatorImpl<float>::propagateToAlphaX<PropagatorImpl<float>::TrackParCov_t>(PropagatorImpl<float>::TrackParCov_t&, float, float, bool, float, float, int, PropagatorImpl<float>::MatCorrType matCorr, track::TrackLTIntegral*, int) const;
#ifndef GPUCA_GPUCODE_DEVICE
template bool PropagatorImpl<double>::propagateToAlphaX<PropagatorImpl<double>::TrackPar_t>(PropagatorImpl<double>::TrackPar_t&, double, double, bool, double, double, int, PropagatorImpl<double>::MatCorrType matCorr, track::TrackLTIntegral*, int) const;
template bool PropagatorImpl<double>::propagateToAlphaX<PropagatorImpl<double>::TrackParCov_t>(PropagatorImpl<double>::TrackParCov_t&, double, double, bool, double, double, int, PropagatorImpl<double>::MatCorrType matCorr, track::TrackLTIntegral*, int) const;
#endif
#endif
} // namespace o2::base
