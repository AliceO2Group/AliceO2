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
void AlignableSensorTRD::prepareMatrixT2L()
{
  // extract from geometry T2L matrix
  double alp = math_utils::detail::sector2Angle<float>(mSector);
  mAlp = alp;
  double loc[3] = {0, 0, 0}, glo[3];
  getMatrixL2GIdeal().LocalToMaster(loc, glo);
  mX = Sqrt(glo[0] * glo[0] + glo[1] * glo[1]);
  TGeoHMatrix t2l;
  t2l.SetDx(mX); // to remove when T2L will be clarified
  t2l.RotateZ(alp * RadToDeg());
  const TGeoHMatrix l2gi = getMatrixL2GIdeal().Inverse();
  t2l.MultiplyLeft(&l2gi);

  setMatrixT2L(t2l);
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
