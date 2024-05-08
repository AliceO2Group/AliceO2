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

/// @file   AlignableSensorTRD.h
/// @author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  2021-02-01
/// @brief  TRD sensor

#include "Align/AlignableSensorTRD.h"
#include "Align/AlignableDetectorTRD.h"
#include "Align/utils.h"
#include "Framework/Logger.h"
#include "Align/AlignmentPoint.h"
#include "DetectorsBase/GeometryManager.h"

namespace o2
{
namespace align
{
using namespace o2::align::utils;
using namespace TMath;

//_________________________________________________________
AlignableSensorTRD::AlignableSensorTRD(const char* name, int vid, int iid, int isec, Controller* ctr) : AlignableSensor(name, vid, iid, ctr), mSector(isec)
{
  // def c-tor
}

//____________________________________________
void AlignableSensorTRD::prepareMatrixClAlg()
{
  // prepare alignment matrix in the pseudo-LOCAL frame of TRD (account that the chamber has extra X,Y rotations
  TGeoHMatrix ma = getMatrixL2GIdeal().Inverse();
  ma *= getMatrixL2G();
  setMatrixClAlg(ma);
  //
}

//____________________________________________
void AlignableSensorTRD::prepareMatrixClAlgReco()
{
  // prepare alignment matrix in the pseudo-LOCAL frame of TRD (account that the chamber has extra X,Y rotations
  TGeoHMatrix ma = getMatrixL2GIdeal().Inverse();
  ma *= getMatrixL2G();
  setMatrixClAlgReco(ma);
  //
}

//____________________________________________
void AlignableSensorTRD::prepareMatrixL2GIdeal()
{
  TGeoHMatrix Rxy;
  Rxy.RotateX(-90);
  Rxy.RotateY(-90);
  TGeoHMatrix mtmp;
  if (!base::GeometryManager::getOriginalMatrix(getSymName(), mtmp)) {
    LOG(fatal) << "Failed to find ideal L2G matrix for " << getSymName();
  }
  mtmp *= Rxy;
  setMatrixL2GIdeal(mtmp);
}

//____________________________________________
void AlignableSensorTRD::prepareMatrixL2G(bool reco)
{
  TGeoHMatrix Rxy;
  Rxy.RotateX(-90);
  Rxy.RotateY(-90);
  const char* path = getSymName();
  const TGeoHMatrix* l2g = nullptr;
  if (!gGeoManager->GetAlignableEntry(path) || !(l2g = base::GeometryManager::getMatrix(path))) {
    LOGP(fatal, "Failed to find L2G matrix for {}alignable {} -> {}", gGeoManager->GetAlignableEntry(path) ? "" : "non-", path, (void*)l2g);
  }
  TGeoHMatrix mtmp = *l2g;
  mtmp *= Rxy;
  reco ? setMatrixL2GReco(mtmp) : setMatrixL2G(mtmp);
}

//____________________________________________
void AlignableSensorTRD::prepareMatrixT2L()
{
  // extract from geometry T2L matrix
  double alp = math_utils::detail::sector2Angle<float>(mSector);
  mAlp = alp;
  TGeoHMatrix Rs;
  Rs.RotateZ(-alp * TMath::RadToDeg());
  TGeoHMatrix m0 = getMatrixL2GIdeal();
  m0.MultiplyLeft(Rs);
  TGeoHMatrix t2l = m0.Inverse();
  setMatrixT2L(t2l);
  double loc[3] = {0, 0, 0}, glo[3];
  t2l.MasterToLocal(loc, glo);
  mX = glo[0];
  //
}

//____________________________________________
void AlignableSensorTRD::dPosTraDParCalib(const AlignmentPoint* pnt, double* deriv, int calibID, const AlignableVolume* parent) const
{
  // calculate point position X,Y,Z derivatives wrt calibration parameter calibID of given parent
  // parent=0 means top detector object calibration
  //
  deriv[0] = deriv[1] = deriv[2] = 0;
  //
  if (!parent) { // TRD detector global calibration
    //
    switch (calibID) {
      case AlignableDetectorTRD::CalibNRCCorrDzDtgl: // correction for Non-Crossing tracklets Z,Y shift: Z -> Z + calib*tgl, Y -> Y + calib*tgl*tilt*sign(tilt);
      {
        double sgYZ = pnt->getYZErrTracking()[1]; // makes sense only for nonRC tracklets
        if (std::abs(sgYZ) > 0.01) {
          const double kTilt = 2. * TMath::DegToRad();
          deriv[2] = pnt->getTrParamWSA()[AlignmentPoint::kParTgl];
          deriv[1] = deriv[2] * Sign(kTilt, sgYZ);
        }
        break;
      }
      case AlignableDetectorTRD::CalibDVT: // correction for bias in VdriftT
      {
        // error in VdriftT equivalent to shift in X at which Y measurement is evaluated
        // Y -> Y + dVdriftT * tg_phi, where tg_phi is the slope of the track in YX plane
        double snp = pnt->getTrParamWSA(AlignmentPoint::kParSnp), slpY = snp / std::sqrt((1.f - snp) * (1.f + snp));
        deriv[1] = slpY;
        break;
      }
      default:
        break;
    }
  }
}

} // namespace align
} // namespace o2
