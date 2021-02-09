// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   AliAlgSensTPC.h
/// @author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  2021-02-01
/// @brief  TPC sensor (chamber)

#include "Align/AliAlgSensTPC.h"
#include "Align/AliAlgAux.h"
#include "Framework/Logger.h"
//#include "AliTrackPointArray.h"
//#include "AliESDtrack.h"
#include "Align/AliAlgPoint.h"
#include "Align/AliAlgDet.h"

ClassImp(o2::align::AliAlgSensTPC)

  using namespace o2::align::AliAlgAux;
using namespace TMath;

namespace o2
{
namespace align
{

//_________________________________________________________
AliAlgSensTPC::AliAlgSensTPC(const char* name, Int_t vid, Int_t iid, Int_t isec)
  : AliAlgSens(name, vid, iid), fSector(isec)
{
  // def c-tor
}

//_________________________________________________________
AliAlgSensTPC::~AliAlgSensTPC()
{
  // d-tor
}
/*
//__________________________________________________________________
void AliAlgSensTPC::SetTrackingFrame()
{
  // define tracking frame of the sensor: just rotation by sector angle
  fAlp = Sector2Alpha(fSector);
  fX = 0;
}
*/

//____________________________________________
void AliAlgSensTPC::PrepareMatrixT2L()
{
  // extract from geometry T2L matrix
  double alp = Sector2Alpha(fSector);
  double loc[3] = {0, 0, 0}, glo[3];
  GetMatrixL2GIdeal().LocalToMaster(loc, glo);
  double x = Sqrt(glo[0] * glo[0] + glo[1] * glo[1]);
  TGeoHMatrix t2l;
  t2l.SetDx(x);
  t2l.RotateZ(alp * RadToDeg());
  const TGeoHMatrix& l2gi = GetMatrixL2GIdeal().Inverse();
  t2l.MultiplyLeft(&l2gi);
  /*
  const TGeoHMatrix* t2l = AliGeomManager::GetTracking2LocalMatrix(GetVolID());
  if (!t2l) {
    Print("long");
    AliFatalF("Failed to find T2L matrix for VID:%d %s",GetVolID(),GetSymName());
  }
  */
  SetMatrixT2L(t2l);
  //
}

//____________________________________________
AliAlgPoint* AliAlgSensTPC::TrackPoint2AlgPoint(int pntId, const AliTrackPointArray* trpArr, const AliESDtrack*)
{
  // convert the pntId-th point to AliAlgPoint
  //
  AliAlgDet* det = GetDetector();
  AliAlgPoint* pnt = det->GetPointFromPool();
  pnt->SetSensor(this);
  //
  double tra[3], locId[3], loc[3],
    glo[3] = {trpArr->GetX()[pntId], trpArr->GetY()[pntId], trpArr->GetZ()[pntId]};
  const TGeoHMatrix& matL2Grec = GetMatrixL2GReco(); // local to global matrix used for reconstruction
  const TGeoHMatrix& matT2L = GetMatrixT2L();        // matrix for tracking to local frame translation
  //
  // undo reco-time alignment
  matL2Grec.MasterToLocal(glo, locId); // go to local frame using reco-time matrix, here we recover ideal measurement
  //
  GetMatrixClAlg().LocalToMaster(locId, loc); // apply alignment
  //
  matT2L.MasterToLocal(loc, tra); // go to tracking frame
  //
  /*
  double gloT[3];
  TGeoHMatrix t2g;
  GetMatrixT2G(t2g); t2g.LocalToMaster(tra,gloT);
  printf("\n%5d %s\n",GetVolID(), GetSymName());
  printf("GloOR: %+.4e %+.4e %+.4e\n",glo[0],glo[1],glo[2]);
  printf("LocID: %+.4e %+.4e %+.4e\n",locId[0],locId[1],locId[2]);
  printf("LocML: %+.4e %+.4e %+.4e\n",loc[0],loc[1],loc[2]);
  printf("Tra  : %+.4e %+.4e %+.4e\n",tra[0],tra[1],tra[2]);
  printf("GloTR: %+.4e %+.4e %+.4e\n",gloT[0],gloT[1],gloT[2]);
  */
  //
  if (!det->GetUseErrorParam()) {
    // convert error
    TGeoHMatrix hcov;
    Double_t hcovel[9];
    const Float_t* pntcov = trpArr->GetCov() + pntId * 6; // 6 elements per error matrix
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
    Double_t* hcovscl = hcov.GetRotationMatrix();
    const double* sysE = GetAddError(); // additional syst error
    pnt->SetYZErrTracking(hcovscl[4] + sysE[0] * sysE[0], hcovscl[5], hcovscl[8] + sysE[1] * sysE[1]);
  } else { // errors will be calculated just before using the point in the fit, using track info
    pnt->SetYZErrTracking(0, 0, 0);
    pnt->SetNeedUpdateFromTrack();
  }
  pnt->SetXYZTracking(tra[0], tra[1], tra[2]);
  pnt->SetAlphaSens(GetAlpTracking());
  pnt->SetXSens(GetXTracking());
  pnt->SetDetID(det->GetDetID());
  pnt->SetSID(GetSID());
  //
  pnt->SetContainsMeasurement();
  //
  pnt->Init();
  //
  return pnt;
  //
}

} // namespace align
} // namespace o2
