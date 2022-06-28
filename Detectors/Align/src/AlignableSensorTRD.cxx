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
//#include "AliTRDgeometry.h"
#include "Align/AlignableDetectorTRD.h"
#include "Align/utils.h"
#include "Framework/Logger.h"
#include "Align/AlignmentPoint.h"
//#include "AliTrackPointArray.h"
//#include "AliESDtrack.h"
//#include "AliTrackerBase.h"

ClassImp(o2::align::AlignableSensorTRD)

  using namespace o2::align::utils;
using namespace TMath;

namespace o2
{
namespace align
{

//_________________________________________________________
AlignableSensorTRD::AlignableSensorTRD(const char* name, int vid, int iid, int isec)
  : AlignableSensor(name, vid, iid), fSector(isec)
{
  // def c-tor
}

//_________________________________________________________
AlignableSensorTRD::~AlignableSensorTRD()
{
  // d-tor
}

//____________________________________________
void AlignableSensorTRD::prepareMatrixT2L()
{
  // extract from geometry T2L matrix
  double alp = sector2Alpha(fSector);
  double loc[3] = {0, 0, 0}, glo[3];
  getMatrixL2GIdeal().LocalToMaster(loc, glo);
  double x = Sqrt(glo[0] * glo[0] + glo[1] * glo[1]);
  TGeoHMatrix t2l;
  t2l.SetDx(x);
  t2l.RotateZ(alp * RadToDeg());
  const TGeoHMatrix& l2gi = getMatrixL2GIdeal().Inverse();
  t2l.MultiplyLeft(&l2gi);
  /*
  const TGeoHMatrix* t2l = AliGeomManager::GetTracking2LocalMatrix(getVolID());
  if (!t2l) {
    Print("long");
    AliFatalF("Failed to find T2L matrix for VID:%d %s",getVolID(),getSymName());
  }
  */
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
      case AlignableDetectorTRD::kCalibNRCCorrDzDtgl: { // correction for Non-Crossing tracklets Z,Y shift: Z -> Z + calib*tgl, Y -> Y + calib*tgl*tilt*sign(tilt);
        double sgYZ = pnt->getYZErrTracking()[1];       // makes sense only for nonRC tracklets
        if (Abs(sgYZ) > 0.01) {
          const double kTilt = 2. * TMath::DegToRad();
          deriv[2] = pnt->getTrParamWSA()[AlignmentPoint::kParTgl];
          deriv[1] = deriv[2] * Sign(kTilt, sgYZ);
        }
        break;
      }
        //
      case AlignableDetectorTRD::kCalibDVT: { // correction for bias in VdriftT
        // error in VdriftT equivalent to shift in X at which Y measurement is evaluated
        // Y -> Y + dVdriftT * tg_phi, where tg_phi is the slope of the track in YX plane
        double snp = pnt->getTrParamWSA(AlignmentPoint::kParSnp), slpY = snp / Sqrt((1 - snp) * (1 + snp));
        deriv[1] = slpY;
        break;
      }

      default:
        break;
    };
  }
  //
}

} // namespace align
} // namespace o2
