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

/// @file   AlignableSensor.h
/// @author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  2021-02-01
/// @brief  End-chain alignment volume in detector branch, where the actual measurement is done.

#include "Align/AlignableSensor.h"
#include "Framework/Logger.h"
#include "Align/AlignmentPoint.h"
#include "Align/AlignableDetector.h"
#include "DetectorsBase/GeometryManager.h"
#include <vector>

ClassImp(o2::align::AlignableSensor);

using namespace o2::align::utils;
using namespace TMath;

namespace o2
{
namespace align
{

//_________________________________________________________
AlignableSensor::AlignableSensor(const char* name, int vid, int iid, Controller* ctr)
  : AlignableVolume(name, iid, ctr), mSID(0), mDet(nullptr), mMatClAlg(), mMatClAlgReco()
{
  // def c-tor
  setVolID(vid);
  mAddError[0] = mAddError[1] = 0;
  mConstrChild = 0; // sensors don't have children
}

//_________________________________________________________
void AlignableSensor::dPosTraDParGeomLOC(const AlignmentPoint* pnt, double* deriv) const
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
  const double* tra = pnt->getXYZTracking();
  //
  for (int ip = kNDOFGeom; ip--;) {
    //
    if (!isFreeDOF(ip)) {
      continue;
    }
    //
    double var = kDelta[ip];
    delta[ip] -= var;
    // variation matrix in tracking frame for variation in sensor LOCAL frame
    getDeltaT2LmodLOC(matMod, delta);
    matMod.LocalToMaster(tra, pos0); // varied position in tracking frame
    //
    delta[ip] += 0.5 * var;
    getDeltaT2LmodLOC(matMod, delta);
    matMod.LocalToMaster(tra, pos1); // varied position in tracking frame
    //
    delta[ip] += var;
    getDeltaT2LmodLOC(matMod, delta);
    matMod.LocalToMaster(tra, pos2); // varied position in tracking frame
    //
    delta[ip] += 0.5 * var;
    getDeltaT2LmodLOC(matMod, delta);
    matMod.LocalToMaster(tra, pos3); // varied position in tracking frame
    //
    delta[ip] = 0;
    double* curd = deriv + ip * 3;
    for (int i = 3; i--;) {
      curd[i] = (8. * (pos2[i] - pos1[i]) - (pos3[i] - pos0[i])) / 6. / var;
    }
  }
  //
}

//_________________________________________________________
void AlignableSensor::dPosTraDParGeomLOC(const AlignmentPoint* pnt, double* deriv, const AlignableVolume* parent) const
{
  // Jacobian of position in sensor tracking frame (tra) vs parent volume LOCAL frame parameters.
  // NO check of parentship is done!
  // Result is stored in array deriv as linearized matrix 6x3
  const double kDelta[kNDOFGeom] = {0.1, 0.1, 0.1, 0.5, 0.5, 0.5};
  double delta[kNDOFGeom], pos0[3], pos1[3], pos2[3], pos3[3];
  TGeoHMatrix matMod;
  // this is the matrix for transition from sensor to parent volume local frames: LOC=matRel*loc
  TGeoHMatrix matRel = parent->getMatrixL2GIdeal().Inverse();
  matRel *= getMatrixL2GIdeal();
  //
  memset(delta, 0, kNDOFGeom * sizeof(double));
  memset(deriv, 0, kNDOFGeom * 3 * sizeof(double));
  const double* tra = pnt->getXYZTracking();
  //
  for (int ip = kNDOFGeom; ip--;) {
    //
    if (!isFreeDOF(ip)) {
      continue;
    }
    //
    double var = kDelta[ip];
    delta[ip] -= var;
    getDeltaT2LmodLOC(matMod, delta, matRel);
    matMod.LocalToMaster(tra, pos0); // varied position in tracking frame
    //
    delta[ip] += 0.5 * var;
    getDeltaT2LmodLOC(matMod, delta, matRel);
    matMod.LocalToMaster(tra, pos1); // varied position in tracking frame
    //
    delta[ip] += var;
    getDeltaT2LmodLOC(matMod, delta, matRel);
    matMod.LocalToMaster(tra, pos2); // varied position in tracking frame
    //
    delta[ip] += 0.5 * var;
    getDeltaT2LmodLOC(matMod, delta, matRel);
    matMod.LocalToMaster(tra, pos3); // varied position in tracking frame
    //
    delta[ip] = 0;
    double* curd = deriv + ip * 3;
    for (int i = 3; i--;) {
      curd[i] = (8. * (pos2[i] - pos1[i]) - (pos3[i] - pos0[i])) / 6. / var;
    }
  }
  //
}

//_________________________________________________________
void AlignableSensor::dPosTraDParGeomTRA(const AlignmentPoint* pnt, double* deriv) const
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
  const double* tra = pnt->getXYZTracking();
  //
  for (int ip = kNDOFGeom; ip--;) {
    //
    if (!isFreeDOF(ip)) {
      continue;
    }
    //
    double var = kDelta[ip];
    delta[ip] -= var;
    getDeltaT2LmodTRA(matMod, delta);
    matMod.LocalToMaster(tra, pos0); // varied position in tracking frame
    //
    delta[ip] += 0.5 * var;
    getDeltaT2LmodTRA(matMod, delta);
    matMod.LocalToMaster(tra, pos1); // varied position in tracking frame
    //
    delta[ip] += var;
    getDeltaT2LmodTRA(matMod, delta);
    matMod.LocalToMaster(tra, pos2); // varied position in tracking frame
    //
    delta[ip] += 0.5 * var;
    getDeltaT2LmodTRA(matMod, delta);
    matMod.LocalToMaster(tra, pos3); // varied position in tracking frame
    //
    delta[ip] = 0;
    double* curd = deriv + ip * 3;
    for (int i = 3; i--;) {
      curd[i] = (8. * (pos2[i] - pos1[i]) - (pos3[i] - pos0[i])) / 6. / var;
    }
  }
  //
}

//_________________________________________________________
void AlignableSensor::dPosTraDParGeomTRA(const AlignmentPoint* pnt, double* deriv, const AlignableVolume* parent) const
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
  getMatrixT2G(matRel);       // t2g matrix of child
  parent->getMatrixT2G(t2gP); // t2g matrix of parent
  const TGeoHMatrix& t2gpi = t2gP.Inverse();
  matRel.MultiplyLeft(&t2gpi);
  //
  memset(delta, 0, kNDOFGeom * sizeof(double));
  memset(deriv, 0, kNDOFGeom * 3 * sizeof(double));
  const double* tra = pnt->getXYZTracking();
  //
  for (int ip = kNDOFGeom; ip--;) {
    //
    if (!isFreeDOF(ip)) {
      continue;
    }
    //
    double var = kDelta[ip];
    delta[ip] -= var;
    getDeltaT2LmodTRA(matMod, delta, matRel);
    matMod.LocalToMaster(tra, pos0); // varied position in tracking frame
    //
    delta[ip] += 0.5 * var;
    getDeltaT2LmodTRA(matMod, delta, matRel);
    matMod.LocalToMaster(tra, pos1); // varied position in tracking frame
    //
    delta[ip] += var;
    getDeltaT2LmodTRA(matMod, delta, matRel);
    matMod.LocalToMaster(tra, pos2); // varied position in tracking frame
    //
    delta[ip] += 0.5 * var;
    getDeltaT2LmodTRA(matMod, delta, matRel);
    matMod.LocalToMaster(tra, pos3); // varied position in tracking frame
    //
    delta[ip] = 0;
    double* curd = deriv + ip * 3;
    for (int i = 3; i--;) {
      curd[i] = (8. * (pos2[i] - pos1[i]) - (pos3[i] - pos0[i])) / 6. / var;
    }
  }
  //
}

//_________________________________________________________
void AlignableSensor::dPosTraDParGeom(const AlignmentPoint* pnt, double* deriv, const AlignableVolume* parent) const
{
  // calculate point position derivatives in tracking frame of sensor
  // vs standard geometrical DOFs of its parent volume (if parent!=0) or sensor itself
  Frame_t frame = parent ? parent->getVarFrame() : getVarFrame();
  switch (frame) {
    case kLOC:
      parent ? dPosTraDParGeomLOC(pnt, deriv, parent) : dPosTraDParGeomLOC(pnt, deriv);
      break;
    case kTRA:
      parent ? dPosTraDParGeomTRA(pnt, deriv, parent) : dPosTraDParGeomTRA(pnt, deriv);
      break;
    default:
      LOG(error) << "Alignment frame " << parent->getVarFrame() << " is not implemented";
      break;
  }
}

//__________________________________________________________________
void AlignableSensor::getModifiedMatrixT2LmodLOC(TGeoHMatrix& matMod, const double* delta) const
{
  // prepare the sensitive module tracking2local matrix from its current T2L matrix
  // by applying local delta of modification of LOCAL frame:
  // loc' = delta*loc = T2L'*tra = T2L'*T2L^-1*loc   ->  T2L' = delta*T2L
  delta2Matrix(matMod, delta);
  matMod.Multiply(&getMatrixT2L());
}

//__________________________________________________________________
void AlignableSensor::getModifiedMatrixT2LmodTRA(TGeoHMatrix& matMod, const double* delta) const
{
  // prepare the sensitive module tracking2local matrix from its current T2L matrix
  // by applying local delta of modification of TRACKING frame:
  // loc' = T2L'*tra = T2L*delta*tra    ->  T2L' = T2L*delta
  delta2Matrix(matMod, delta);
  matMod.MultiplyLeft(&getMatrixT2L());
}

//__________________________________________________________________
void AlignableSensor::addChild(AlignableVolume*)
{
  LOG(fatal) << "Sensor volume cannot have children: id=" << getVolID() << " " << GetName();
}

//__________________________________________________________________
int AlignableSensor::Compare(const TObject* b) const
{
  // compare VolIDs
  return GetUniqueID() < b->GetUniqueID() ? -1 : 1;
}

//__________________________________________________________________
/* // RS FIXME REM
void AlignableSensor::setTrackingFrame()
{
  // define tracking frame of the sensor
  //  AliWarningF("Generic method called for %s",getSymName());
  double tra[3] = {0}, glo[3];
  TGeoHMatrix t2g;
  getMatrixT2G(t2g);
  t2g.LocalToMaster(tra, glo);
  mX = Sqrt(glo[0] * glo[0] + glo[1] * glo[1]);
  mAlp = ATan2(glo[1], glo[0]);
  utils::bringToPiPM(mAlp);
  //
}
*/
//____________________________________________
void AlignableSensor::Print(const Option_t* opt) const
{
  // print info
  TString opts = opt;
  opts.ToLower();
  printf("Lev:%2d IntID:%7d %s VId:%6d X:%8.4f Alp:%+.4f | Err: %.4e %.4e | Used Points: %d\n",
         countParents(), getInternalID(), getSymName(), getVolID(), mX, mAlp,
         mAddError[0], mAddError[1], mNProcPoints);
  printf("     DOFs: Tot: %d (offs: %5d) Free: %d  Geom: %d {", mNDOFs, mFirstParGloID, mNDOFsFree, mNDOFGeomFree);
  for (int i = 0; i < kNDOFGeom; i++) {
    printf("%d", isFreeDOF(i) ? 1 : 0);
  }
  printf("} in %s frame\n", sFrameName[mVarFrame]);
  //
  //
  //
  if (opts.Contains("par") && mFirstParGloID >= 0) {
    printf("     Lb: ");
    for (int i = 0; i < mNDOFs; i++) {
      printf("%10d  ", getParLab(i));
    }
    printf("\n");
    printf("     Vl: ");
    for (int i = 0; i < mNDOFs; i++) {
      printf("%+9.3e  ", getParVal(i));
    }
    printf("\n");
    printf("     Er: ");
    for (int i = 0; i < mNDOFs; i++) {
      printf("%+9.3e  ", getParErr(i));
    }
    printf("\n");
  }
  //
  if (opts.Contains("mat")) { // print matrices
    printf("L2G ideal   : ");
    getMatrixL2GIdeal().Print();
    printf("L2G misalign: ");
    getMatrixL2G().Print();
    printf("L2G RecoTime: ");
    getMatrixL2GReco().Print();
    printf("T2L         : ");
    getMatrixT2L().Print();
    printf("ClAlg       : ");
    getMatrixClAlg().Print();
    printf("ClAlgReco: ");
    getMatrixClAlgReco().Print();
  }
  //
}

//____________________________________________
void AlignableSensor::prepareMatrixClAlg()
{
  // prepare alignment matrix in the LOCAL frame: delta = Gideal^-1 * G
  TGeoHMatrix ma = getMatrixL2GIdeal().Inverse();
  ma *= getMatrixL2G();
  setMatrixClAlg(ma);
  //
}

//____________________________________________
void AlignableSensor::prepareMatrixClAlgReco()
{
  // prepare alignment matrix used at reco time
  TGeoHMatrix ma = getMatrixL2GIdeal().Inverse();
  ma *= getMatrixL2GReco();
  setMatrixClAlgReco(ma);
  //
}

//____________________________________________
void AlignableSensor::updatePointByTrackInfo(AlignmentPoint* pnt, const trackParam_t* t) const
{
  // update
  mDet->updatePointByTrackInfo(pnt, t);
}

//____________________________________________
void AlignableSensor::dPosTraDParCalib(const AlignmentPoint* pnt, double* deriv, int calibID, const AlignableVolume* parent) const
{
  // calculate point position X,Y,Z derivatives wrt calibration parameter calibID of given parent
  // parent=0 means top detector object calibration
  //
  deriv[0] = deriv[1] = deriv[2] = 0;
}

//______________________________________________________
int AlignableSensor::finalizeStat(DOFStatistics& st)
{
  // finalize statistics on processed points
  fillDOFStat(st);
  return mNProcPoints;
}

//_________________________________________________________________
void AlignableSensor::updateL2GRecoMatrices(const std::vector<o2::detectors::AlignParam>& algArr, const TGeoHMatrix* cumulDelta)
{
  // recreate mMatL2GReco matrices from ideal L2G matrix and alignment objects
  // used during data reconstruction.
  // On top of what each volume does, also update misalignment matrix inverse
  //
  AlignableVolume::updateL2GRecoMatrices(algArr, cumulDelta);
  prepareMatrixClAlgReco();
  //
}

//_________________________________________________________________
void AlignableSensor::applyAlignmentFromMPSol()
{
  // apply to the tracking coordinates in the sensor frame the full chain
  // of alignments found by MP for this sensor and its parents
  //
  const AlignableVolume* vol = this;
  TGeoHMatrix deltaG;
  // create global combined delta:
  // DeltaG = deltaG_0*...*deltaG_j, where delta_i is global delta of each member of hierarchy
  while (vol) {
    TGeoHMatrix deltaGJ;
    vol->createAlignmenMatrix(deltaGJ);
    deltaG.MultiplyLeft(&deltaGJ);
    vol = vol->getParent();
  }
  //
  // update misaligned L2G matrix
  deltaG *= getMatrixL2GIdeal();
  setMatrixL2G(deltaG);
  //
  // update local misalignment matrix
  prepareMatrixClAlg();
  //
}
/*
//_________________________________________________________________
void AlignableSensor::applyAlignmentFromMPSol()
{
  // apply to the tracking coordinates in the sensor frame the full chain
  // of alignments found by MP for this sensor and its parents
  double delta[kNDOFGeom]={0};
  //
  TGeoHMatrix matMod;
  //
  // sensor proper variation
  getParValGeom(delta);
  isFrameTRA() ? getDeltaT2LmodTRA(matMod,delta) : getDeltaT2LmodLOC(matMod,delta);
  mMatClAlg.MultiplyLeft(&matMod);
  //
  AlignableVolume* parent = this;
  while ((parent==parent->getParent())) {
    // this is the matrix for transition from sensor to parent volume frame
    parent->getParValGeom(delta);
    TGeoHMatrix matRel,t2gP;
    if (parent->isFrameTRA()) {
      getMatrixT2G(matRel);           // t2g matrix of child
      parent->getMatrixT2G(t2gP);     // t2g matrix of parent
      matRel.MultiplyLeft(&t2gP.Inverse());
      getDeltaT2LmodTRA(matMod, delta, matRel);
    }
    else {
      matRel = parent->getMatrixL2GIdeal().Inverse();
      matRel *= getMatrixL2GIdeal();
      getDeltaT2LmodLOC(matMod, delta, matRel);
    }
    mMatClAlg.MultiplyLeft(&matMod);
  }
  //
}
*/

} // namespace align
} // namespace o2
