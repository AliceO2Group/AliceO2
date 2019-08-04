// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file AlignParam.cxx
/// \brief Implementation of the base alignment parameters class

#include <FairLogger.h>
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
  : TNamed(symname, "")
{
  /// standard constructor with 3 translation + 3 rotation parameters
  /// If the user explicitly sets the global variable to false then the
  /// parameters are interpreted as giving the local transformation.
  /// This requires to have a gGeoMenager active instance, otherwise the
  /// constructor will fail (no object created)

  setAlignableID(algID);

  if (global) {
    setTranslation(x, y, z);
    setRotation(psi, theta, phi);
  } else {
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
  Double_t sinpsi = std::sin(psi);
  Double_t cospsi = std::cos(psi);
  Double_t sinthe = std::sin(theta);
  Double_t costhe = std::cos(theta);
  Double_t sinphi = std::sin(phi);
  Double_t cosphi = std::cos(phi);
  rot[0] = costhe * cosphi;
  rot[1] = -costhe * sinphi;
  rot[2] = sinthe;
  rot[3] = sinpsi * sinthe * cosphi + cospsi * sinphi;
  rot[4] = -sinpsi * sinthe * sinphi + cospsi * cosphi;
  rot[5] = -costhe * sinpsi;
  rot[6] = -cospsi * sinthe * cosphi + sinpsi * sinphi;
  rot[7] = cospsi * sinthe * sinphi + sinpsi * cosphi;
  rot[8] = costhe * cospsi;
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
      setRotation(psi, theta, phi);
    }
  } else {
    mPsi = mTheta = mPhi = 0.;
  }
  return true;
}

//_____________________________________________________________________________
bool AlignParam::matrixToAngles(const double* rot, double& psi, double& theta, double& phi)
{
  /// Calculates the Euler angles in "x y z" notation
  /// using the rotation matrix
  /// Returns false in case the rotation angles can not be
  /// extracted from the matrix
  //
  if (std::abs(rot[0]) < 1e-7 || std::abs(rot[8]) < 1e-7) {
    LOG(ERROR) << "Failed to extract roll-pitch-yall angles!" << FairLogger::endl;
    return false;
  }
  psi = std::atan2(-rot[5], rot[8]);
  theta = std::asin(rot[2]);
  phi = std::atan2(-rot[1], rot[0]);
  return true;
}

//_____________________________________________________________________________
bool AlignParam::setLocalParams(double x, double y, double z, double psi, double theta, double phi)
{
  /// Set the global delta transformation by passing the parameters
  /// for the local delta transformation (3 shifts and 3 angles).
  /// In case that the TGeo was not initialized or not closed,
  /// returns false and the object parameters are not set.

  TGeoHMatrix m;
  Double_t tr[3] = {x, y, z};
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
    LOG(ERROR) << "Can't set the local alignment object parameters! gGeoManager doesn't exist or it is still open!"
               << FairLogger::endl;
    return false;
  }

  const char* symname = getSymName();
  TGeoHMatrix gprime, gprimeinv;
  TGeoPhysicalNode* pn = nullptr;
  TGeoPNEntry* pne = gGeoManager->GetAlignableEntry(symname);
  if (pne) {
    pn = pne->GetPhysicalNode();
    if (pn) {
      if (pn->IsAligned()) {
        LOG(WARNING) << "Volume " << symname << " has been misaligned already!" << FairLogger::endl;
      }
      gprime = *pn->GetMatrix();
    } else {
      gprime = pne->GetGlobalOrig();
    }
  } else {
    LOG(WARNING) << "The symbolic volume name " << symname
                 << " does not correspond to a physical entry. Using it as volume path!" << FairLogger::endl;
    if (!gGeoManager->cd(symname)) {
      LOG(ERROR) << "Volume name or path " << symname << " is not valid!" << FairLogger::endl;
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

  return setLocalParams(m1);
}

//_____________________________________________________________________________
bool AlignParam::createLocalMatrix(TGeoHMatrix& m) const
{
  // Get the matrix for the local delta transformation.
  // In case that the TGeo was not initialized or not closed,
  // returns false and the object parameters are not set.
  //
  if (!gGeoManager || !gGeoManager->IsClosed()) {
    LOG(ERROR) << "Can't get the local alignment object parameters! gGeoManager doesn't exist or it is still open!"
               << FairLogger::endl;
    return false;
  }

  const char* symname = getSymName();
  TGeoPhysicalNode* node;
  TGeoPNEntry* pne = gGeoManager->GetAlignableEntry(symname);
  if (pne) {
    if (!pne->GetPhysicalNode()) {
      node = gGeoManager->MakeAlignablePN(pne);
    } else {
      node = pne->GetPhysicalNode();
    }
  } else {
    LOG(WARNING) << "The symbolic volume name " << symname
                 << " does not correspond to a physical entry. Using it as volume path!" << FairLogger::endl;
    node = (TGeoPhysicalNode*)gGeoManager->MakePhysicalNode(symname);
  }

  if (!node) {
    LOG(ERROR) << "Volume name or path " << symname << " is not valid!" << FairLogger::endl;
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
bool AlignParam::applyToGeometry(bool ovlpcheck, double ovlToler)
{
  /// Apply the current alignment object to the TGeo geometry
  /// This method returns FALSE if the symname of the object was not
  /// valid neither to get a TGeoPEntry nor as a volume path
  //
  if (!gGeoManager || !gGeoManager->IsClosed()) {
    LOG(ERROR) << "Can't apply the alignment object! gGeoManager doesn't exist or it is still open!"
               << FairLogger::endl;
    return false;
  }

  if (gGeoManager->IsLocked()) {
    LOG(ERROR) << "Can't apply the alignment object! Geometry is locked!" << FairLogger::endl;
    return false;
  }

  const char* symname = getSymName();
  const char* path;
  TGeoPhysicalNode* node;
  TGeoPNEntry* pne = gGeoManager->GetAlignableEntry(symname);
  if (pne) {
    path = pne->GetTitle();
    node = gGeoManager->MakeAlignablePN(pne);
  } else {
    LOG(DEBUG) << "The symbolic volume name " << symname
               << " does not correspond to a physical entry. Using it as a volume path!" << FairLogger::endl;
    path = symname;
    if (!gGeoManager->CheckPath(path)) {
      LOG(ERROR) << "Volume path " << path << " is not valid" << FairLogger::endl;
      return false;
    }
    if (gGeoManager->GetListOfPhysicalNodes()->FindObject(path)) {
      LOG(ERROR) << "Volume path " << path << " has been misaligned already!" << FairLogger::endl;
      return false;
    }
    node = (TGeoPhysicalNode*)gGeoManager->MakePhysicalNode(path);
  }

  if (!node) {
    LOG(ERROR) << "Volume path " << path << " is not valid" << FairLogger::endl;
    return false;
  }

  //  Double_t threshold = 0.001;

  TGeoHMatrix gprime = *node->GetMatrix();
  TGeoHMatrix align = createMatrix();
  gprime.MultiplyLeft(&align);
  TGeoHMatrix* ginv = new TGeoHMatrix; // TGeoPhysicalNode takes and manages raw pointer, need naked new!
  TGeoHMatrix* g = node->GetMatrix(node->GetLevel() - 1);
  *ginv = g->Inverse();
  *ginv *= gprime;

  LOG(DEBUG) << "Aligning volume " << symname << FairLogger::endl;

  if (ovlpcheck) {
    node->Align(ginv, nullptr, true, ovlToler);
    TObjArray* ovlpArray = gGeoManager->GetListOfOverlaps();
    Int_t nOvlp = ovlpArray->GetEntriesFast();
    if (nOvlp) {
      LOG(INFO) << "Misalignment of node " << node->GetName() << " generated the following " << nOvlp
                << "overlaps/extrusions:" << FairLogger::endl;
      for (int i = 0; i < nOvlp; i++) {
        ((TGeoOverlap*)ovlpArray->UncheckedAt(i))->PrintInfo();
      }
    }
  } else {
    node->Align(ginv, nullptr, false, ovlToler);
  }

  return true;
}

//_____________________________________________________________________________
int AlignParam::Compare(const TObject* obj) const
{
  /// Compare the levels of two alignment objects
  /// Used in the sorting during the application of alignment
  /// objects to the geometry

  int level = getLevel();
  int level2 = ((AlignParam*)obj)->getLevel();
  if (level == level2)
    return 0;
  else
    return ((level > level2) ? 1 : -1);
}

//_____________________________________________________________________________
int AlignParam::getLevel() const
{
  /// Return the geometry level of the alignable volume to which
  /// the alignment object is associated; this is the number of
  /// slashes in the corresponding volume path
  //
  if (!gGeoManager) {
    LOG(ERROR) << "gGeoManager doesn't exist or it is still open: unable to return meaningful level value."
               << FairLogger::endl;
    return -1;
  }
  const char* symname = getSymName();
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
void AlignParam::Print(const Option_t*) const
{
  // print parameters
  printf("%s : %6d | X: %+e Y: %+e Z: %+e | pitch: %+e roll: %+e yaw: %e\n", getSymName(), getAlignableID(), getX(),
         getY(), getZ(), getPsi(), getTheta(), getPhi());
}
