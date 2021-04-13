// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   AliAlgSensITS.h
/// @author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  2021-02-01
/// @brief  ITS sensor

#include "Align/AliAlgSensITS.h"
#include "Align/AliAlgAux.h"
#include "Framework/Logger.h"
//#include "AliTrackPointArray.h"
//#include "AliESDtrack.h"
#include "Align/AliAlgPoint.h"
#include "Align/AliAlgDet.h"

ClassImp(o2::align::AliAlgSensITS);

using namespace o2::align::AliAlgAux;
using namespace TMath;

namespace o2
{
namespace align
{

//_________________________________________________________
AliAlgSensITS::AliAlgSensITS(const char* name, int vid, int iid)
  : AliAlgSens(name, vid, iid)
{
  // def c-tor
}

//_________________________________________________________
AliAlgSensITS::~AliAlgSensITS()
{
  // d-tor
}
/*
//__________________________________________________________________
void AliAlgSensITS::setTrackingFrame()
{
  // define tracking frame of the sensor
  double tra[3]={0},loc[3],glo[3];
  // ITS defines tracking frame with origin in sensor, others at 0
  getMatrixT2L().LocalToMaster(tra,loc);
  getMatrixL2GIdeal().LocalToMaster(loc,glo);
  fX = Sqrt(glo[0]*glo[0]+glo[1]*glo[1]);
  fAlp = ATan2(glo[1],glo[0]);
  AliAlgAux::bringToPiPM(fAlp);
}
*/

//____________________________________________
AliAlgPoint* AliAlgSensITS::TrackPoint2AlgPoint(int pntId, const AliTrackPointArray* trpArr, const AliESDtrack*)
{
  // convert the pntId-th point to AliAlgPoint
  //
  AliAlgDet* det = getDetector();
  AliAlgPoint* pnt = det->getPointFromPool();
  pnt->setSensor(this);
  //
  double tra[3], locId[3], loc[3],
    glo[3] = {trpArr->GetX()[pntId], trpArr->GetY()[pntId], trpArr->GetZ()[pntId]};
  const TGeoHMatrix& matL2Grec = getMatrixL2GReco(); // local to global matrix used for reconstruction
  const TGeoHMatrix& matT2L = getMatrixT2L();        // matrix for tracking to local frame translation
  //
  // undo reco-time alignment
  matL2Grec.MasterToLocal(glo, locId); // go to local frame using reco-time matrix, here we recover ideal measurement
  //
  getMatrixClAlg().LocalToMaster(locId, loc); // apply alignment
  //
  matT2L.MasterToLocal(loc, tra); // go to tracking frame
  //
  /*
  double gloT[3];
  TGeoHMatrix t2g;
  getMatrixT2G(t2g); t2g.LocalToMaster(tra,gloT);
  printf("\n%5d %s\n",getVolID(), getSymName());
  printf("GloOR: %+.4e %+.4e %+.4e\n",glo[0],glo[1],glo[2]);
  printf("LocID: %+.4e %+.4e %+.4e\n",locId[0],locId[1],locId[2]);
  printf("LocML: %+.4e %+.4e %+.4e\n",loc[0],loc[1],loc[2]);
  printf("Tra  : %+.4e %+.4e %+.4e\n",tra[0],tra[1],tra[2]);
  printf("GloTR: %+.4e %+.4e %+.4e\n",gloT[0],gloT[1],gloT[2]);
  */
  //
  if (!det->getUseErrorParam()) {
    // convert error
    TGeoHMatrix hcov;
    double hcovel[9];
    const float* pntcov = trpArr->GetCov() + pntId * 6; // 6 elements per error matrix
    hcovel[0] = double(pntcov[0]);
    hcovel[1] = double(pntcov[1]);
    hcovel[2] = double(pntcov[2]);
    hcovel[3] = double(pntcov[1]);
    hcovel[4] = double(pntcov[3]);
    hcovel[5] = double(pntcov[4]);
    hcovel[6] = double(pntcov[2]);
    hcovel[7] = double(pntcov[4]);
    hcovel[8] = double(pntcov[5]);
    hcov.SetRotation(hcovel);
    hcov.Multiply(&matL2Grec);
    const TGeoHMatrix& l2gi = matL2Grec.Inverse();
    hcov.MultiplyLeft(&l2gi); // errors in local frame
    hcov.Multiply(&matT2L);
    const TGeoHMatrix& t2li = matT2L.Inverse();
    hcov.MultiplyLeft(&t2li); // errors in tracking frame
    //
    double* hcovscl = hcov.GetRotationMatrix();
    const double* sysE = getAddError(); // additional syst error
    pnt->setYZErrTracking(hcovscl[4] + sysE[0] * sysE[0], hcovscl[5], hcovscl[8] + sysE[1] * sysE[1]);
  } else { // errors will be calculated just before using the point in the fit, using track info
    pnt->setYZErrTracking(0, 0, 0);
    pnt->setNeedUpdateFromTrack();
  }
  pnt->setXYZTracking(tra[0], tra[1], tra[2]);
  pnt->setAlphaSens(getAlpTracking());
  pnt->setXSens(getXTracking());
  pnt->setDetID(det->getDetID());
  pnt->setSID(getSID());
  //
  pnt->setContainsMeasurement();
  //
  pnt->init();
  //
  return pnt;
  //
}

} // namespace align
} // namespace o2
