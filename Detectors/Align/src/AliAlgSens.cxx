// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   AliAlgSens.h
/// @author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  2021-02-01
/// @brief  End-chain alignment volume in detector branch, where the actual measurement is done.

#include <stdio.h>
#include <TClonesArray.h>
#include "Align/AliAlgSens.h"
#include "Align/AliAlgAux.h"
#include "Framework/Logger.h"
//#include "AliGeomManager.h"
//#include "AliExternalTrackParam.h"
#include "Align/AliAlgPoint.h"
#include "Align/AliAlgDet.h"
#include "Align/AliAlgDOFStat.h"

ClassImp(AliAlgSens);

using namespace o2::align::AliAlgAux;
using namespace TMath;

namespace o2
{
namespace align
{

//_________________________________________________________
AliAlgSens::AliAlgSens(const char* name, Int_t vid, Int_t iid)
  : AliAlgVol(name, iid), fSID(0), fDet(0), fMatClAlg(), fMatClAlgReco()
{
  // def c-tor
  SetVolID(vid);
  fAddError[0] = fAddError[1] = 0;
  fConstrChild = 0; // sensors don't have children
}

//_________________________________________________________
AliAlgSens::~AliAlgSens()
{
  // d-tor
}

//_________________________________________________________
void AliAlgSens::DPosTraDParGeomLOC(const AliAlgPoint* pnt, double* deriv) const
{
  // Jacobian of position in sensor tracking frame (tra) vs sensor LOCAL frame
  // parameters in TGeoHMatrix convention.
  // Result is stored in array deriv as linearized matrix 6x3
  const double kDelta[kNDOFGeom] = {0.1, 0.1, 0.1, 0.5, 0.5, 0.5};
  double delta[kNDOFGeom], pos0[3], pos1[3], pos2[3], pos3[3];
  TGeoHMatrix matMod;
  //
  memset(delta, 0, kNDOFGeom * sizeof(double));
  memset(deriv, 0, kNDOFGeom * 3 * sizeof(double));
  const double* tra = pnt->GetXYZTracking();
  //
  for (int ip = kNDOFGeom; ip--;) {
    //
    if (!IsFreeDOF(ip))
      continue;
    //
    double var = kDelta[ip];
    delta[ip] -= var;
    // variation matrix in tracking frame for variation in sensor LOCAL frame
    GetDeltaT2LmodLOC(matMod, delta);
    matMod.LocalToMaster(tra, pos0); // varied position in tracking frame
    //
    delta[ip] += 0.5 * var;
    GetDeltaT2LmodLOC(matMod, delta);
    matMod.LocalToMaster(tra, pos1); // varied position in tracking frame
    //
    delta[ip] += var;
    GetDeltaT2LmodLOC(matMod, delta);
    matMod.LocalToMaster(tra, pos2); // varied position in tracking frame
    //
    delta[ip] += 0.5 * var;
    GetDeltaT2LmodLOC(matMod, delta);
    matMod.LocalToMaster(tra, pos3); // varied position in tracking frame
    //
    delta[ip] = 0;
    double* curd = deriv + ip * 3;
    for (int i = 3; i--;)
      curd[i] = (8. * (pos2[i] - pos1[i]) - (pos3[i] - pos0[i])) / 6. / var;
  }
  //
}

//_________________________________________________________
void AliAlgSens::DPosTraDParGeomLOC(const AliAlgPoint* pnt, double* deriv, const AliAlgVol* parent) const
{
  // Jacobian of position in sensor tracking frame (tra) vs parent volume LOCAL frame parameters.
  // NO check of parentship is done!
  // Result is stored in array deriv as linearized matrix 6x3
  const double kDelta[kNDOFGeom] = {0.1, 0.1, 0.1, 0.5, 0.5, 0.5};
  double delta[kNDOFGeom], pos0[3], pos1[3], pos2[3], pos3[3];
  TGeoHMatrix matMod;
  // this is the matrix for transition from sensor to parent volume local frames: LOC=matRel*loc
  TGeoHMatrix matRel = parent->GetMatrixL2GIdeal().Inverse();
  matRel *= GetMatrixL2GIdeal();
  //
  memset(delta, 0, kNDOFGeom * sizeof(double));
  memset(deriv, 0, kNDOFGeom * 3 * sizeof(double));
  const double* tra = pnt->GetXYZTracking();
  //
  for (int ip = kNDOFGeom; ip--;) {
    //
    if (!IsFreeDOF(ip))
      continue;
    //
    double var = kDelta[ip];
    delta[ip] -= var;
    GetDeltaT2LmodLOC(matMod, delta, matRel);
    matMod.LocalToMaster(tra, pos0); // varied position in tracking frame
    //
    delta[ip] += 0.5 * var;
    GetDeltaT2LmodLOC(matMod, delta, matRel);
    matMod.LocalToMaster(tra, pos1); // varied position in tracking frame
    //
    delta[ip] += var;
    GetDeltaT2LmodLOC(matMod, delta, matRel);
    matMod.LocalToMaster(tra, pos2); // varied position in tracking frame
    //
    delta[ip] += 0.5 * var;
    GetDeltaT2LmodLOC(matMod, delta, matRel);
    matMod.LocalToMaster(tra, pos3); // varied position in tracking frame
    //
    delta[ip] = 0;
    double* curd = deriv + ip * 3;
    for (int i = 3; i--;)
      curd[i] = (8. * (pos2[i] - pos1[i]) - (pos3[i] - pos0[i])) / 6. / var;
  }
  //
}

//_________________________________________________________
void AliAlgSens::DPosTraDParGeomTRA(const AliAlgPoint* pnt, double* deriv) const
{
  // Jacobian of position in sensor tracking frame (tra) vs sensor TRACKING
  // frame parameters in TGeoHMatrix convention, i.e. the modified parameter is
  // tra' = tau*tra
  //
  // Result is stored in array deriv as linearized matrix 6x3
  const double kDelta[kNDOFGeom] = {0.1, 0.1, 0.1, 0.5, 0.5, 0.5};
  double delta[kNDOFGeom], pos0[3], pos1[3], pos2[3], pos3[3];
  TGeoHMatrix matMod;
  //
  memset(delta, 0, kNDOFGeom * sizeof(double));
  memset(deriv, 0, kNDOFGeom * 3 * sizeof(double));
  const double* tra = pnt->GetXYZTracking();
  //
  for (int ip = kNDOFGeom; ip--;) {
    //
    if (!IsFreeDOF(ip))
      continue;
    //
    double var = kDelta[ip];
    delta[ip] -= var;
    GetDeltaT2LmodTRA(matMod, delta);
    matMod.LocalToMaster(tra, pos0); // varied position in tracking frame
    //
    delta[ip] += 0.5 * var;
    GetDeltaT2LmodTRA(matMod, delta);
    matMod.LocalToMaster(tra, pos1); // varied position in tracking frame
    //
    delta[ip] += var;
    GetDeltaT2LmodTRA(matMod, delta);
    matMod.LocalToMaster(tra, pos2); // varied position in tracking frame
    //
    delta[ip] += 0.5 * var;
    GetDeltaT2LmodTRA(matMod, delta);
    matMod.LocalToMaster(tra, pos3); // varied position in tracking frame
    //
    delta[ip] = 0;
    double* curd = deriv + ip * 3;
    for (int i = 3; i--;)
      curd[i] = (8. * (pos2[i] - pos1[i]) - (pos3[i] - pos0[i])) / 6. / var;
  }
  //
}

//_________________________________________________________
void AliAlgSens::DPosTraDParGeomTRA(const AliAlgPoint* pnt, double* deriv, const AliAlgVol* parent) const
{
  // Jacobian of position in sensor tracking frame (tra) vs sensor TRACKING
  // frame parameters in TGeoHMatrix convention, i.e. the modified parameter is
  // tra' = tau*tra
  //
  // Result is stored in array deriv as linearized matrix 6x3
  const double kDelta[kNDOFGeom] = {0.1, 0.1, 0.1, 0.5, 0.5, 0.5};
  double delta[kNDOFGeom], pos0[3], pos1[3], pos2[3], pos3[3];
  TGeoHMatrix matMod;
  //
  // 1st we need a matrix for transition between child and parent TRACKING frames
  // Let TRA,LOC are positions in tracking and local frame of parent, linked as LOC=T2L*TRA
  // and tra,loc are positions in tracking and local frame of child,  linked as loc=t2l*tra
  // The loc and LOC are linked as LOC=R*loc, where R = L2G^-1*l2g, with L2G and l2g
  // local2global matrices for parent and child
  //
  // Then, TRA = T2L^-1*LOC = T2L^-1*R*loc = T2L^-1*R*t2l*tra
  // -> TRA = matRel*tra, with matRel = T2L^-1*L2G^-1 * l2g*t2l
  // Note that l2g*t2l are tracking to global matrices
  TGeoHMatrix matRel, t2gP;
  GetMatrixT2G(matRel);       // t2g matrix of child
  parent->GetMatrixT2G(t2gP); // t2g matrix of parent
  const TGeoHMatrix& t2gpi = t2gP.Inverse();
  matRel.MultiplyLeft(&t2gpi);
  //
  memset(delta, 0, kNDOFGeom * sizeof(double));
  memset(deriv, 0, kNDOFGeom * 3 * sizeof(double));
  const double* tra = pnt->GetXYZTracking();
  //
  for (int ip = kNDOFGeom; ip--;) {
    //
    if (!IsFreeDOF(ip))
      continue;
    //
    double var = kDelta[ip];
    delta[ip] -= var;
    GetDeltaT2LmodTRA(matMod, delta, matRel);
    matMod.LocalToMaster(tra, pos0); // varied position in tracking frame
    //
    delta[ip] += 0.5 * var;
    GetDeltaT2LmodTRA(matMod, delta, matRel);
    matMod.LocalToMaster(tra, pos1); // varied position in tracking frame
    //
    delta[ip] += var;
    GetDeltaT2LmodTRA(matMod, delta, matRel);
    matMod.LocalToMaster(tra, pos2); // varied position in tracking frame
    //
    delta[ip] += 0.5 * var;
    GetDeltaT2LmodTRA(matMod, delta, matRel);
    matMod.LocalToMaster(tra, pos3); // varied position in tracking frame
    //
    delta[ip] = 0;
    double* curd = deriv + ip * 3;
    for (int i = 3; i--;)
      curd[i] = (8. * (pos2[i] - pos1[i]) - (pos3[i] - pos0[i])) / 6. / var;
  }
  //
}

//_________________________________________________________
void AliAlgSens::DPosTraDParGeom(const AliAlgPoint* pnt, double* deriv, const AliAlgVol* parent) const
{
  // calculate point position derivatives in tracking frame of sensor
  // vs standard geometrical DOFs of its parent volume (if parent!=0) or sensor itself
  Frame_t frame = parent ? parent->GetVarFrame() : GetVarFrame();
  switch (frame) {
    case kLOC:
      parent ? DPosTraDParGeomLOC(pnt, deriv, parent) : DPosTraDParGeomLOC(pnt, deriv);
      break;
    case kTRA:
      parent ? DPosTraDParGeomTRA(pnt, deriv, parent) : DPosTraDParGeomTRA(pnt, deriv);
      break;
    default:
      AliErrorF("Alignment frame %d is not implemented", parent->GetVarFrame());
      break;
  }
}

//__________________________________________________________________
void AliAlgSens::GetModifiedMatrixT2LmodLOC(TGeoHMatrix& matMod, const Double_t* delta) const
{
  // prepare the sensitive module tracking2local matrix from its current T2L matrix
  // by applying local delta of modification of LOCAL frame:
  // loc' = delta*loc = T2L'*tra = T2L'*T2L^-1*loc   ->  T2L' = delta*T2L
  Delta2Matrix(matMod, delta);
  matMod.Multiply(&GetMatrixT2L());
}

//__________________________________________________________________
void AliAlgSens::GetModifiedMatrixT2LmodTRA(TGeoHMatrix& matMod, const Double_t* delta) const
{
  // prepare the sensitive module tracking2local matrix from its current T2L matrix
  // by applying local delta of modification of TRACKING frame:
  // loc' = T2L'*tra = T2L*delta*tra    ->  T2L' = T2L*delta
  Delta2Matrix(matMod, delta);
  matMod.MultiplyLeft(&GetMatrixT2L());
}

//__________________________________________________________________
void AliAlgSens::AddChild(AliAlgVol*)
{
  AliFatalF("Sensor volume cannot have childs: id=%d %s", GetVolID(), GetName());
}

//__________________________________________________________________
Int_t AliAlgSens::Compare(const TObject* b) const
{
  // compare VolIDs
  return GetUniqueID() < b->GetUniqueID() ? -1 : 1;
}

//__________________________________________________________________
void AliAlgSens::SetTrackingFrame()
{
  // define tracking frame of the sensor
  //  AliWarningF("Generic method called for %s",GetSymName());
  double tra[3] = {0}, glo[3];
  TGeoHMatrix t2g;
  GetMatrixT2G(t2g);
  t2g.LocalToMaster(tra, glo);
  fX = Sqrt(glo[0] * glo[0] + glo[1] * glo[1]);
  fAlp = ATan2(glo[1], glo[0]);
  AliAlgAux::BringToPiPM(fAlp);
  //
}

//____________________________________________
void AliAlgSens::Print(const Option_t* opt) const
{
  // print info
  TString opts = opt;
  opts.ToLower();
  printf("Lev:%2d IntID:%7d %s VId:%6d X:%8.4f Alp:%+.4f | Err: %.4e %.4e | Used Points: %d\n",
         CountParents(), GetInternalID(), GetSymName(), GetVolID(), fX, fAlp,
         fAddError[0], fAddError[1], fNProcPoints);
  printf("     DOFs: Tot: %d (offs: %5d) Free: %d  Geom: %d {", fNDOFs, fFirstParGloID, fNDOFFree, fNDOFGeomFree);
  for (int i = 0; i < kNDOFGeom; i++)
    printf("%d", IsFreeDOF(i) ? 1 : 0);
  printf("} in %s frame\n", fgkFrameName[fVarFrame]);
  //
  //
  //
  if (opts.Contains("par") && fParVals) {
    printf("     Lb: ");
    for (int i = 0; i < fNDOFs; i++)
      printf("%10d  ", GetParLab(i));
    printf("\n");
    printf("     Vl: ");
    for (int i = 0; i < fNDOFs; i++)
      printf("%+9.3e  ", GetParVal(i));
    printf("\n");
    printf("     Er: ");
    for (int i = 0; i < fNDOFs; i++)
      printf("%+9.3e  ", GetParErr(i));
    printf("\n");
  }
  //
  if (opts.Contains("mat")) { // print matrices
    printf("L2G ideal   : ");
    GetMatrixL2GIdeal().Print();
    printf("L2G misalign: ");
    GetMatrixL2G().Print();
    printf("L2G RecoTime: ");
    GetMatrixL2GReco().Print();
    printf("T2L         : ");
    GetMatrixT2L().Print();
    printf("ClAlg       : ");
    GetMatrixClAlg().Print();
    printf("ClAlgReco: ");
    GetMatrixClAlgReco().Print();
  }
  //
}

//____________________________________________
void AliAlgSens::PrepareMatrixT2L()
{
  // extract from geometry T2L matrix
  const TGeoHMatrix* t2l = AliGeomManager::GetTracking2LocalMatrix(GetVolID());
  if (!t2l) {
    Print("long");
    AliFatalF("Failed to find T2L matrix for VID:%d %s", GetVolID(), GetSymName());
  }
  SetMatrixT2L(*t2l);
  //
}

//____________________________________________
void AliAlgSens::PrepareMatrixClAlg()
{
  // prepare alignment matrix in the LOCAL frame: delta = Gideal^-1 * G
  TGeoHMatrix ma = GetMatrixL2GIdeal().Inverse();
  ma *= GetMatrixL2G();
  SetMatrixClAlg(ma);
  //
}

//____________________________________________
void AliAlgSens::PrepareMatrixClAlgReco()
{
  // prepare alignment matrix used at reco time
  TGeoHMatrix ma = GetMatrixL2GIdeal().Inverse();
  ma *= GetMatrixL2GReco();
  SetMatrixClAlgReco(ma);
  //
}

//____________________________________________
void AliAlgSens::UpdatePointByTrackInfo(AliAlgPoint* pnt, const AliExternalTrackParam* t) const
{
  // update
  fDet->UpdatePointByTrackInfo(pnt, t);
}

//____________________________________________
void AliAlgSens::DPosTraDParCalib(const AliAlgPoint* pnt, double* deriv, int calibID, const AliAlgVol* parent) const
{
  // calculate point position X,Y,Z derivatives wrt calibration parameter calibID of given parent
  // parent=0 means top detector object calibration
  //
  deriv[0] = deriv[1] = deriv[2] = 0;
}

//______________________________________________________
Int_t AliAlgSens::FinalizeStat(AliAlgDOFStat* st)
{
  // finalize statistics on processed points
  if (st)
    FillDOFStat(st);
  return fNProcPoints;
}

//_________________________________________________________________
void AliAlgSens::UpdateL2GRecoMatrices(const TClonesArray* algArr, const TGeoHMatrix* cumulDelta)
{
  // recreate fMatL2GReco matrices from ideal L2G matrix and alignment objects
  // used during data reconstruction.
  // On top of what each volume does, also update misalignment matrix inverse
  //
  AliAlgVol::UpdateL2GRecoMatrices(algArr, cumulDelta);
  PrepareMatrixClAlgReco();
  //
}

/*
//_________________________________________________________________
AliAlgPoint* AliAlgSens::TrackPoint2AlgPoint(int, const AliTrackPointArray*, const AliESDtrack*)
{
  // dummy converter
  AliError("Generic method, must be implemented in specific sensor");
  return 0;
}
*/

//_________________________________________________________________
void AliAlgSens::ApplyAlignmentFromMPSol()
{
  // apply to the tracking coordinates in the sensor frame the full chain
  // of alignments found by MP for this sensor and its parents
  //
  const AliAlgVol* vol = this;
  TGeoHMatrix deltaG;
  // create global combined delta:
  // DeltaG = deltaG_0*...*deltaG_j, where delta_i is global delta of each member of hierarchy
  while (vol) {
    TGeoHMatrix deltaGJ;
    vol->CreateAlignmenMatrix(deltaGJ);
    deltaG.MultiplyLeft(&deltaGJ);
    vol = vol->GetParent();
  }
  //
  // update misaligned L2G matrix
  deltaG *= GetMatrixL2GIdeal();
  SetMatrixL2G(deltaG);
  //
  // update local misalignment matrix
  PrepareMatrixClAlg();
  //
}

/*
//_________________________________________________________________
void AliAlgSens::ApplyAlignmentFromMPSol()
{
  // apply to the tracking coordinates in the sensor frame the full chain
  // of alignments found by MP for this sensor and its parents
  double delta[kNDOFGeom]={0};
  //
  TGeoHMatrix matMod;
  //
  // sensor proper variation
  GetParValGeom(delta);
  IsFrameTRA() ? GetDeltaT2LmodTRA(matMod,delta) : GetDeltaT2LmodLOC(matMod,delta);
  fMatClAlg.MultiplyLeft(&matMod);
  //
  AliAlgVol* parent = this;
  while ((parent==parent->GetParent())) {
    // this is the matrix for transition from sensor to parent volume frame
    parent->GetParValGeom(delta);
    TGeoHMatrix matRel,t2gP;
    if (parent->IsFrameTRA()) {
      GetMatrixT2G(matRel);           // t2g matrix of child
      parent->GetMatrixT2G(t2gP);     // t2g matrix of parent
      matRel.MultiplyLeft(&t2gP.Inverse());
      GetDeltaT2LmodTRA(matMod, delta, matRel);
    }
    else {
      matRel = parent->GetMatrixL2GIdeal().Inverse();
      matRel *= GetMatrixL2GIdeal();
      GetDeltaT2LmodLOC(matMod, delta, matRel);
    }
    fMatClAlg.MultiplyLeft(&matMod);
  }
  //
}

*/

} // namespace align
} // namespace o2
