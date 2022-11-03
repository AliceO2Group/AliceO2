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

/// \file AlignParam.cxx
/// \brief Implementation of the base alignment parameters class

#include <fairlogger/Logger.h>
#include <TGeoManager.h>
#include <TGeoMatrix.h>
#include <TGeoOverlap.h>
#include <TGeoPhysicalNode.h>

#include "DetectorsCommonDataFormats/AlignParam.h"

using namespace o2::detectors;

//___________________________________________________
AlignParam::AlignParam(const char* symname, int algID,       // volume symbolic name and its alignable ID
                       double x, double y, double z,         // delta translation
                       double psi, double theta, double phi, // delta rotation
                       bool global)                          // global (preferable) or local delta definition
  : mSymName(symname), mAlignableID(algID)
{
  /// standard constructor with 3 translation + 3 rotation parameters
  /// If the user explicitly sets the global variable to false then the
  /// parameters are interpreted as giving the local transformation.
  /// This requires to have a gGeoMenager active instance, otherwise the
  /// constructor will fail (no object created)

  if (global) {
    setGlobalParams(x, y, z, psi, theta, phi);
  } else {
    setLocalParams(x, y, z, psi, theta, phi);
  }
}

//___________________________________________________
TGeoHMatrix AlignParam::createMatrix() const
{
  /// create a copy of alignment global delta matrix
  TGeoHMatrix mat;
  setMatrixTranslation(mX, mY, mZ, mat);
  setMatrixRotation(mPhi, mTheta, mPsi, mat);
  return mat;
}

//___________________________________________________
void AlignParam::setMatrixRotation(double psi, double theta, double phi, TGeoHMatrix& dest) const
{
  /// apply rotation to matrix
  double rot[9] = {};
  anglesToMatrix(psi, theta, phi, rot);
  dest.SetRotation(rot);
}

//_____________________________________________________________________________
void AlignParam::setTranslation(const TGeoMatrix& m)
{
  /// set the translation parameters extracting them from the matrix
  /// passed as argument

  if (m.IsTranslation()) {
    const double* tr = m.GetTranslation();
    mX = tr[0];
    mY = tr[1];
    mZ = tr[2];
  } else {
    mX = mY = mZ = 0.;
  }
}

//_____________________________________________________________________________
bool AlignParam::setRotation(const TGeoMatrix& m)
{
  /// set the rotation parameters extracting them from the matrix
  /// passed as argument

  if (m.IsRotation()) {
    const double* rot = m.GetRotationMatrix();
    double psi, theta, phi;
    if (!matrixToAngles(rot, psi, theta, phi)) {
      return false;
    }
    setRotation(psi, theta, phi);
  } else {
    mPsi = mTheta = mPhi = 0.;
  }
  return true;
}

//_____________________________________________________________________________
bool AlignParam::matrixToAngles(const double* rot, double& psi, double& theta, double& phi) const
{
  /// Calculates the Euler angles in "x y z" notation
  /// using the rotation matrix
  /// Returns false in case the rotation angles can not be
  /// extracted from the matrix
  //
  if (std::abs(rot[0]) < 1e-7 || std::abs(rot[8]) < 1e-7) {
    LOG(error) << "Failed to extract roll-pitch-yall angles!";
    return false;
  }
  psi = std::atan2(-rot[5], rot[8]);
  theta = std::asin(rot[2]);
  phi = std::atan2(-rot[1], rot[0]);
  return true;
}

//_____________________________________________________________________________
void AlignParam::anglesToMatrix(double psi, double theta, double phi, double* rot) const
{
  // Calculates the rotation matrix using the
  // Euler angles in "x y z" notation
  //
  double sinpsi = std::sin(psi);
  double cospsi = std::cos(psi);
  double sinthe = std::sin(theta);
  double costhe = std::cos(theta);
  double sinphi = std::sin(phi);
  double cosphi = std::cos(phi);

  rot[0] = costhe * cosphi;
  rot[1] = -costhe * sinphi;
  rot[2] = sinthe;
  rot[3] = sinpsi * sinthe * cosphi + cospsi * sinphi;
  rot[4] = -sinpsi * sinthe * sinphi + cospsi * cosphi;
  rot[5] = -costhe * sinpsi;
  rot[6] = -cospsi * sinthe * cosphi + sinpsi * sinphi;
  rot[7] = cospsi * sinthe * sinphi + sinpsi * cosphi;
  rot[8] = costhe * cospsi;
}

//_____________________________________________________________________________
bool AlignParam::setLocalParams(double x, double y, double z, double psi, double theta, double phi)
{
  /// Set the global delta transformation by passing the parameters
  /// for the local delta transformation (3 shifts and 3 angles).
  /// In case that the TGeo was not initialized or not closed,
  /// returns false and the object parameters are not set.

  TGeoHMatrix m;
  double tr[3] = {x, y, z};
  m.SetTranslation(tr);
  setMatrixRotation(psi, theta, phi, m);
  return setLocalParams(m);
}

//_____________________________________________________________________________
bool AlignParam::setLocalParams(const TGeoMatrix& m)
{
  // Set the global delta transformation by passing the TGeo matrix
  // for the local delta transformation.
  // In case that the TGeo was not initialized or not closed,
  // returns false and the object parameters are not set.
  //
  if (!gGeoManager || !gGeoManager->IsClosed()) {
    LOG(error) << "Can't set the local alignment object parameters! gGeoManager doesn't exist or it is still open!";
    return false;
  }

  const char* symname = getSymName().c_str();
  TGeoHMatrix gprime, gprimeinv;
  TGeoPhysicalNode* pn = nullptr;
  TGeoPNEntry* pne = gGeoManager->GetAlignableEntry(symname);
  if (pne) {
    pn = pne->GetPhysicalNode();
    if (pn) {
      if (pn->IsAligned()) {
        LOG(warning) << "Volume " << symname << " has been misaligned already!";
      }
      gprime = *pn->GetMatrix();
    } else {
      gprime = pne->GetGlobalOrig();
    }
  } else {
    LOG(warning) << "The symbolic volume name " << symname
                 << " does not correspond to a physical entry. Using it as volume path!";
    if (!gGeoManager->cd(symname)) {
      LOG(error) << "Volume name or path " << symname << " is not valid!";
      return false;
    }
    gprime = *gGeoManager->GetCurrentMatrix();
  }

  TGeoHMatrix m1; // the TGeoHMatrix copy of the local delta "m"
  m1.SetTranslation(m.GetTranslation());
  m1.SetRotation(m.GetRotationMatrix());

  gprimeinv = gprime.Inverse();
  m1.Multiply(&gprimeinv);
  m1.MultiplyLeft(&gprime);

  setGlobalParams(m1);
  return true;
}

//_____________________________________________________________________________
bool AlignParam::createLocalMatrix(TGeoHMatrix& m) const
{
  // Get the matrix for the local delta transformation.
  // In case that the TGeo was not initialized or not closed,
  // returns false and the object parameters are not set.
  //
  if (!gGeoManager || !gGeoManager->IsClosed()) {
    LOG(error) << "Can't get the local alignment object parameters! gGeoManager doesn't exist or it is still open!";
    return false;
  }

  const char* symname = getSymName().c_str();
  TGeoPhysicalNode* node;
  TGeoPNEntry* pne = gGeoManager->GetAlignableEntry(symname);
  if (pne) {
    if (!pne->GetPhysicalNode()) {
      node = gGeoManager->MakeAlignablePN(pne);
    } else {
      node = pne->GetPhysicalNode();
    }
  } else {
    LOG(warning) << "The symbolic volume name " << symname
                 << " does not correspond to a physical entry. Using it as volume path!";
    node = (TGeoPhysicalNode*)gGeoManager->MakePhysicalNode(symname);
  }

  if (!node) {
    LOG(error) << "Volume name or path " << symname << " is not valid!";
    return false;
  }
  m = createMatrix();
  TGeoHMatrix gprime, gprimeinv;
  gprime = *node->GetMatrix();
  gprimeinv = gprime.Inverse();
  m.Multiply(&gprime);
  m.MultiplyLeft(&gprimeinv);

  return true;
}

//_____________________________________________________________________________
bool AlignParam::applyToGeometry() const
{
  /// Apply the current alignment object to the TGeo geometry
  /// This method returns FALSE if the symname of the object was not
  /// valid neither to get a TGeoPEntry nor as a volume path
  //
  if (!gGeoManager || !gGeoManager->IsClosed()) {
    LOG(error) << "Can't apply the alignment object! gGeoManager doesn't exist or it is still open!";
    return false;
  }

  if (gGeoManager->IsLocked()) {
    LOG(error) << "Can't apply the alignment object! Geometry is locked!";
    return false;
  }

  const char* symname = getSymName().c_str();
  const char* path;
  TGeoPhysicalNode* node;
  TGeoPNEntry* pne = gGeoManager->GetAlignableEntry(symname);
  if (pne) {
    path = pne->GetTitle();
    node = gGeoManager->MakeAlignablePN(pne);
  } else {
    LOG(debug) << "The symbolic volume name " << symname
               << " does not correspond to a physical entry. Using it as a volume path!";
    path = symname;
    if (!gGeoManager->CheckPath(path)) {
      LOG(error) << "Volume path " << path << " is not valid";
      return false;
    }
    if (gGeoManager->GetListOfPhysicalNodes()->FindObject(path)) {
      LOG(error) << "Volume path " << path << " has been misaligned already!";
      return false;
    }
    node = (TGeoPhysicalNode*)gGeoManager->MakePhysicalNode(path);
  }

  if (!node) {
    LOG(error) << "Volume path " << path << " is not valid";
    return false;
  }

  //  double threshold = 0.001;

  TGeoHMatrix gprime = *node->GetMatrix();
  TGeoHMatrix align = createMatrix();
  gprime.MultiplyLeft(&align);
  TGeoHMatrix* ginv = new TGeoHMatrix; // TGeoPhysicalNode takes and manages raw pointer, need naked new!
  TGeoHMatrix* g = node->GetMatrix(node->GetLevel() - 1);
  *ginv = g->Inverse();
  *ginv *= gprime;

  LOG(debug) << "Aligning volume " << symname;

  node->Align(ginv);

  return true;
}

//_____________________________________________________________________________
int AlignParam::getLevel() const
{
  /// Return the geometry level of the alignable volume to which
  /// the alignment object is associated; this is the number of
  /// slashes in the corresponding volume path
  //
  if (!gGeoManager) {
    LOG(error) << "gGeoManager doesn't exist or it is still open: unable to return meaningful level value.";
    return -1;
  }
  const char* symname = getSymName().c_str();
  const char* path;
  TGeoPNEntry* pne = gGeoManager->GetAlignableEntry(symname);
  if (pne) {
    path = pne->GetTitle();
  } else {
    path = symname;
  }

  TString pathStr = path;
  int lev = pathStr.CountChar('/');
  return (pathStr[0] != '/') ? ++lev : lev;
}

//_____________________________________________________________________________
void AlignParam::print() const
{
  // print parameters
  printf("%s : %6d | X: %+e Y: %+e Z: %+e | pitch: %+e roll: %+e yaw: %e\n", getSymName().c_str(), getAlignableID(), getX(),
         getY(), getZ(), getPsi(), getTheta(), getPhi());
}

//_____________________________________________________________________________
void AlignParam::setGlobalParams(double x, double y, double z, double psi, double theta, double phi)
{
  /// set parameters of global delta
  setTranslation(x, y, z);
  setRotation(psi, theta, phi);
}

//_____________________________________________________________________________
void AlignParam::setRotation(double psi, double theta, double phi)
{
  /// set global delta rotations angles in radian
  mPsi = psi;
  mTheta = theta;
  mPhi = phi;
}

//_____________________________________________________________________________
void AlignParam::setTranslation(double x, double y, double z)
{
  /// set global delta displacements in cm
  mX = x;
  mY = y;
  mZ = z;
}

//_____________________________________________________________________________
void AlignParam::setGlobalParams(const TGeoMatrix& m)
{
  /// set params from the matrix of global delta
  setTranslation(m);
  setRotation(m);
}

//___________________________________________________
void AlignParam::setMatrixTranslation(double x, double y, double z, TGeoHMatrix& dest) const
{
  /// apply translation to matrix
  double tra[3] = {x, y, z};
  dest.SetTranslation(tra);
}

//_____________________________________________________________________________
bool AlignParam::setLocalTranslation(double x, double y, double z)
{
  /// Set the global delta transformation by passing the three shifts giving
  /// the translation in the local reference system of the alignable
  /// volume (known by TGeo geometry).
  /// In case that the TGeo was not initialized or not closed,
  /// returns false and the object parameters are not set.

  TGeoHMatrix m;
  double tr[3] = {x, y, z};
  m.SetTranslation(tr);

  return setLocalParams(m);
}

//_____________________________________________________________________________
bool AlignParam::setLocalTranslation(const TGeoMatrix& m)
{
  /// Set the global delta transformation by passing the matrix of
  /// the local delta transformation and taking its translational part
  /// In case that the TGeo was not initialized or not closed,
  /// returns false and the object parameters are not set.

  TGeoHMatrix mtr;
  mtr.SetTranslation(m.GetTranslation());
  return setLocalParams(mtr);
}

//_____________________________________________________________________________
bool AlignParam::setLocalRotation(double psi, double theta, double phi)
{
  /// Set the global delta transformation by passing the three angles giving
  /// the rotation in the local reference system of the alignable
  /// volume (known by TGeo geometry).
  /// In case that the TGeo was not initialized or not closed,
  /// returns false and the object parameters are not set.

  TGeoHMatrix m;
  setMatrixRotation(psi, theta, phi, m);
  return setLocalParams(m);
}

//_____________________________________________________________________________
bool AlignParam::setLocalRotation(const TGeoMatrix& m)
{
  /// Set the global delta transformation by passing the matrix of
  /// the local delta transformation and taking its rotational part
  /// In case that the TGeo was not initialized or not closed,
  /// returns false and the object parameters are not set.

  TGeoHMatrix rotm;
  rotm.SetRotation(m.GetRotationMatrix());
  return setLocalParams(rotm);
}
