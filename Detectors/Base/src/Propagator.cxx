// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "DetectorsBase/Propagator.h"
#include <FairLogger.h>
#include <FairRunAna.h> // eventually will get rid of it
#include <TGeoGlobalMagField.h>
#include "DataFormatsParameters/GRPObject.h"
#include "Field/MagFieldFast.h"
#include "Field/MagneticField.h"
#include "MathUtils/Utils.h"

using namespace o2::base;

constexpr int Propagator::USEMatCorrNONE;
constexpr int Propagator::USEMatCorrTGeo;
constexpr int Propagator::USEMatCorrLUT;

Propagator::Propagator()
{
  ///< construct checking if needed components were initialized

  // we need the geoemtry loaded
  if (!gGeoManager) {
    LOG(FATAL) << "No active geometry!";
  }

  o2::field::MagneticField* slowField = nullptr;
  slowField = static_cast<o2::field::MagneticField*>(TGeoGlobalMagField::Instance()->GetField());
  if (!slowField) {
    LOG(WARNING) << "No Magnetic Field in TGeoGlobalMagField, checking legacy FairRunAna";
    slowField = dynamic_cast<o2::field::MagneticField*>(FairRunAna::Instance()->GetField());
  }
  if (!slowField) {
    LOG(FATAL) << "Magnetic field is not initialized!";
  }
  if (!slowField->getFastField()) {
    slowField->AllowFastField(true);
  }
  mField = slowField->getFastField();
  const float xyz[3] = {0.};
  mField->GetBz(xyz, mBz);
}

//_______________________________________________________________________
bool Propagator::PropagateToXBxByBz(o2::track::TrackParCov& track, float xToGo, float mass, float maxSnp, float maxStep,
                                    int matCorr, o2::track::TrackLTIntegral* tofInfo, int signCorr) const
{
  //----------------------------------------------------------------
  //
  // Propagates the track to the plane X=xk (cm)
  // taking into account all the three components of the magnetic field
  // and correcting for the crossed material.
  //
  // mass     - mass used in propagation - used for energy loss correction (if <0 then q=2)
  // maxStep  - maximal step for propagation
  // tofInfo  - optional container for track length and PID-dependent TOF integration
  //
  // matCorr  - material correction type, it is up to the user to make sure the pointer is attached (if LUT is requested)
  //----------------------------------------------------------------
  const float Epsilon = 0.00001;
  auto dx = xToGo - track.getX();
  int dir = dx > 0.f ? 1 : -1;
  if (!signCorr) {
    signCorr = -dir; // sign of eloss correction is not imposed
  }

  std::array<float, 3> b;
  while (std::abs(dx) > Epsilon) {
    auto step = std::min(std::abs(dx), maxStep);
    if (dir < 0) {
      step = -step;
    }
    auto x = track.getX() + step;
    auto xyz0 = track.getXYZGlo();
    mField->Field(xyz0, b.data());

    if (!track.propagateTo(x, b)) {
      return false;
    }
    if (maxSnp > 0 && std::abs(track.getSnp()) >= maxSnp) {
      return false;
    }
    if (matCorr != USEMatCorrNONE) {
      auto xyz1 = track.getXYZGlo();
      auto mb = getMatBudget(matCorr, xyz0, xyz1);
      if (!track.correctForMaterial(mb.meanX2X0, ((signCorr < 0) ? -mb.length : mb.length) * mb.meanRho, mass)) {
        return false;
      }

      if (tofInfo) {
        tofInfo->addStep(mb.length, track); // fill L,ToF info using already calculated step length
        tofInfo->addX2X0(mb.meanX2X0);
      }
    } else if (tofInfo) { // if tofInfo filling was requested w/o material correction, we need to calculate the step lenght
      auto xyz1 = track.getXYZGlo();
      Vector3D<float> stepV(xyz1.X() - xyz0.X(), xyz1.Y() - xyz0.Y(), xyz1.Z() - xyz0.Z());
      tofInfo->addStep(stepV.R(), track);
    }
    dx = xToGo - track.getX();
  }
  return true;
}

//_______________________________________________________________________
bool Propagator::propagateToX(o2::track::TrackParCov& track, float xToGo, float bZ, float mass, float maxSnp, float maxStep,
                              int matCorr, o2::track::TrackLTIntegral* tofInfo, int signCorr) const
{
  //----------------------------------------------------------------
  //
  // Propagates the track to the plane X=xk (cm)
  // taking into account all the three components of the magnetic field
  // and correcting for the crossed material.
  //
  // mass     - mass used in propagation - used for energy loss correction (if <0 then q=2)
  // maxStep  - maximal step for propagation
  // tofInfo  - optional container for track length and PID-dependent TOF integration
  //
  // matCorr  - material correction type, it is up to the user to make sure the pointer is attached (if LUT is requested)
  //----------------------------------------------------------------
  const float Epsilon = 0.00001;
  auto dx = xToGo - track.getX();
  int dir = dx > 0.f ? 1 : -1;
  if (!signCorr) {
    signCorr = -dir; // sign of eloss correction is not imposed
  }

  while (std::abs(dx) > Epsilon) {
    auto step = std::min(std::abs(dx), maxStep);
    if (dir < 0) {
      step = -step;
    }
    auto x = track.getX() + step;
    auto xyz0 = track.getXYZGlo();

    if (!track.propagateTo(x, bZ)) {
      return false;
    }
    if (maxSnp > 0 && std::abs(track.getSnp()) >= maxSnp) {
      return false;
    }
    if (matCorr != USEMatCorrNONE) {
      auto xyz1 = track.getXYZGlo();
      auto mb = getMatBudget(matCorr, xyz0, xyz1);
      //
      if (!track.correctForMaterial(mb.meanX2X0, ((signCorr < 0) ? -mb.length : mb.length) * mb.meanRho, mass)) {
        return false;
      }

      if (tofInfo) {
        tofInfo->addStep(mb.length, track); // fill L,ToF info using already calculated step length
        tofInfo->addX2X0(mb.meanX2X0);
      }
    } else if (tofInfo) { // if tofInfo filling was requested w/o material correction, we need to calculate the step lenght
      auto xyz1 = track.getXYZGlo();
      Vector3D<float> stepV(xyz1.X() - xyz0.X(), xyz1.Y() - xyz0.Y(), xyz1.Z() - xyz0.Z());
      tofInfo->addStep(stepV.R(), track);
    }
    dx = xToGo - track.getX();
  }
  return true;
}

//_______________________________________________________________________
bool Propagator::propagateToDCA(const Point3D<float>& vtx, o2::track::TrackParCov& track, float bZ,
                                float mass, float maxStep, int matCorr,
                                o2::track::TrackLTIntegral* tofInfo, int signCorr, float maxD) const
{
  // propagate track to DCA to the vertex
  float sn, cs, alp = track.getAlpha();
  o2::utils::sincosf(alp, sn, cs);
  float x = track.getX(), y = track.getY(), snp = track.getSnp(), csp = std::sqrt((1.f - snp) * (1.f + snp));
  float xv = vtx.X() * cs + vtx.Y() * sn, yv = -vtx.X() * sn + vtx.Y() * cs, zv = vtx.Z();
  x -= xv;
  y -= yv;
  //Estimate the impact parameter neglecting the track curvature
  Double_t d = std::abs(x * snp - y * csp);
  if (d > maxD) {
    return false;
  }
  float crv = track.getCurvature(bZ);
  float tgfv = -(crv * x - snp) / (crv * y + csp);
  sn = tgfv / std::sqrt(1.f + tgfv * tgfv);
  cs = std::sqrt((1. - sn) * (1. + sn));
  cs = (std::abs(tgfv) > o2::constants::math::Almost0) ? sn / tgfv : o2::constants::math::Almost1;

  x = xv * cs + yv * sn;
  yv = -xv * sn + yv * cs;
  xv = x;

  o2::track::TrackParCov tmpT(track); // operate on the copy to recover after the failure
  alp += std::asin(sn);
  if (!tmpT.rotate(alp) || !propagateToX(tmpT, xv, bZ, mass, 0.85, maxStep, matCorr, tofInfo, signCorr)) {
    LOG(ERROR) << "failed to propagate to alpha=" << alp << " X=" << xv << " for vertex "
               << vtx.X() << ' ' << vtx.Y() << ' ' << vtx.Z() << " | Track is: ";
    tmpT.print();
    return false;
  }
  track = tmpT;
  return true;
}

//____________________________________________________________
int Propagator::initFieldFromGRP(const std::string grpFileName, std::string grpName)
{
  /// load grp and init magnetic field
  LOG(INFO) << "Loading field from GRP of " << grpFileName;
  const auto grp = o2::parameters::GRPObject::loadFrom(grpFileName, grpName);
  if (!grp) {
    return -1;
  }
  grp->print();

  return initFieldFromGRP(grp);
}

//____________________________________________________________
int Propagator::initFieldFromGRP(const o2::parameters::GRPObject* grp)
{
  /// init mag field from GRP data and attach it to TGeoGlobalMagField

  if (TGeoGlobalMagField::Instance()->IsLocked()) {
    if (TGeoGlobalMagField::Instance()->GetField()->TestBit(o2::field::MagneticField::kOverrideGRP)) {
      LOG(WARNING) << "ExpertMode!!! GRP information will be ignored";
      LOG(WARNING) << "ExpertMode!!! Running with the externally locked B field";
      return 0;
    } else {
      LOG(INFO) << "Destroying existing B field instance";
      delete TGeoGlobalMagField::Instance();
    }
  }
  auto fld = o2::field::MagneticField::createFieldMap(grp->getL3Current(), grp->getDipoleCurrent());
  TGeoGlobalMagField::Instance()->SetField(fld);
  TGeoGlobalMagField::Instance()->Lock();
  LOG(INFO) << "Running with the B field constructed out of GRP";
  LOG(INFO) << "Access field via TGeoGlobalMagField::Instance()->Field(xyz,bxyz) or via";
  LOG(INFO) << "auto o2field = static_cast<o2::field::MagneticField*>( TGeoGlobalMagField::Instance()->GetField() )";

  return 0;
}
