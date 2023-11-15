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

/// \file GeometryManager.cxx
/// \brief Implementation of the GeometryManager class

#include <fairlogger/Logger.h>  // for LOG
#include <TCollection.h> // for TIter
#include <TFile.h>
#include <TGeoMatrix.h>       // for TGeoHMatrix
#include <TGeoNode.h>         // for TGeoNode
#include <TGeoPhysicalNode.h> // for TGeoPhysicalNode, TGeoPNEntry
#include <string>
#include <cassert>
#include <cstddef> // for NULL
#include <numeric>

#include "DetectorsBase/GeometryManager.h"
#include "DetectorsCommonDataFormats/AlignParam.h"
#include "CommonUtils/NameConf.h"
#include "DetectorsBase/Aligner.h"

using namespace o2::detectors;
using namespace o2::base;

/// Implementation of GeometryManager, the geometry manager class which interfaces to TGeo and
/// the look-up table mapping unique volume indices to symbolic volume names. For that, it
/// collects several static methods
std::mutex GeometryManager::sTGMutex;

//______________________________________________________________________
Bool_t GeometryManager::getOriginalMatrix(const char* symname, TGeoHMatrix& m)
{
  m.Clear();
  if (!gGeoManager || !gGeoManager->IsClosed()) {
    LOG(error) << "No active geometry or geometry not yet closed!";
    ;
    return kFALSE;
  }
  std::lock_guard<std::mutex> guard(sTGMutex);
  if (!gGeoManager->GetListOfPhysicalNodes()) {
    LOG(warning) << "gGeoManager doesn't contain any aligned nodes!";

    if (!gGeoManager->cd(symname)) {
      LOG(error) << "Volume path " << symname << " not valid!";
      return kFALSE;
    } else {
      m = *gGeoManager->GetCurrentMatrix();
      return kTRUE;
    }
  }

  TGeoPNEntry* pne = gGeoManager->GetAlignableEntry(symname);
  const char* path = nullptr;

  if (pne) {
    m = *pne->GetGlobalOrig();
    return kTRUE;
  } else {
    LOG(warning) << "The symbolic volume name " << symname
                 << "does not correspond to a physical entry. Using it as a volume path!";
    path = symname;
  }

  return getOriginalMatrixFromPath(path, m);
}

//______________________________________________________________________
Bool_t GeometryManager::getOriginalMatrixFromPath(const char* path, TGeoHMatrix& m)
{
  m.Clear();

  if (!gGeoManager || !gGeoManager->IsClosed()) {
    LOG(error) << "Can't get the original global matrix! gGeoManager doesn't exist or it is still opened!";
    return kFALSE;
  }
  std::lock_guard<std::mutex> guard(sTGMutex);
  if (!gGeoManager->CheckPath(path)) {
    LOG(error) << "Volume path " << path << " not valid!";
    return kFALSE;
  }

  TIter next(gGeoManager->GetListOfPhysicalNodes());
  gGeoManager->cd(path);

  while (gGeoManager->GetLevel()) {
    TGeoPhysicalNode* physNode = nullptr;
    next.Reset();
    TGeoNode* node = gGeoManager->GetCurrentNode();

    while ((physNode = (TGeoPhysicalNode*)next())) {
      if (physNode->GetNode() == node) {
        break;
      }
    }

    TGeoMatrix* lm = nullptr;
    if (physNode) {
      lm = physNode->GetOriginalMatrix();
      if (!lm) {
        lm = node->GetMatrix();
      }
    } else {
      lm = node->GetMatrix();
    }

    m.MultiplyLeft(lm);

    gGeoManager->CdUp();
  }
  return kTRUE;
}

//______________________________________________________________________
TGeoHMatrix* GeometryManager::getMatrix(TGeoPNEntry* pne)
{
  // Get the global transformation matrix for a given PNEntry
  // by quering the TGeoManager

  if (!gGeoManager || !gGeoManager->IsClosed()) {
    LOG(error) << "Can't get the global matrix! gGeoManager doesn't exist or it is still opened!";
    return nullptr;
  }

  // if matrix already known --> return it
  TGeoPhysicalNode* pnode = pne->GetPhysicalNode();
  if (pnode) {
    return pnode->GetMatrix();
  }

  // otherwise calculate it from title and attach via TGeoPhysicalNode
  pne->SetPhysicalNode(new TGeoPhysicalNode(pne->GetTitle()));
  return pne->GetPhysicalNode()->GetMatrix();
}

//______________________________________________________________________
TGeoHMatrix* GeometryManager::getMatrix(const char* symname)
{
  // Get the global transformation matrix for a given alignable volume
  //  identified by its symbolic name 'symname' by quering the TGeoManager

  if (!gGeoManager || !gGeoManager->IsClosed()) {
    LOG(error) << "No active geometry or geometry not yet closed!";
    return nullptr;
  }

  TGeoPNEntry* pne = gGeoManager->GetAlignableEntry(symname);
  if (!pne) {
    return nullptr;
  }

  return getMatrix(pne);
}

//______________________________________________________________________
const char* GeometryManager::getSymbolicName(DetID detid, int sensid)
{
  /**
   * Get symoblic name of sensitive volume sensid of detector detid
   **/
  int id = getSensID(detid, sensid);
  TGeoPNEntry* pne = gGeoManager->GetAlignableEntryByUID(id);
  if (!pne) {
    LOG(error) << "Failed to find alignable entry with index " << id << ": Det" << detid << " Sens.Vol:" << sensid << ") !";
    return nullptr;
  }
  return pne->GetName();
}

TGeoPNEntry* GeometryManager::getPNEntry(DetID detid, Int_t sensid)
{
  /**
   * Get PN Entry of sensitive volume sensid of detector detid
   **/
  int id = getSensID(detid, sensid);
  TGeoPNEntry* pne = gGeoManager->GetAlignableEntryByUID(id);
  if (!pne) {
    LOG(error) << "The sens.vol " << sensid << " of det " << detid << " does not correspond to a physical entry!";
  }
  return pne;
}

//______________________________________________________________________
TGeoHMatrix* GeometryManager::getMatrix(DetID detid, Int_t sensid)
{
  /**
   * Get position matrix (aligned) of sensitive volume sensid of detector detid. Slow
   **/
  static TGeoHMatrix matTmp;
  TGeoPNEntry* pne = getPNEntry(detid, sensid);
  if (!pne) {
    return nullptr;
  }

  TGeoPhysicalNode* pnode = pne->GetPhysicalNode();
  if (pnode) {
    return pnode->GetMatrix();
  }

  const char* path = pne->GetTitle();
  gGeoManager->PushPath(); // Preserve the modeler state.
  if (!gGeoManager->cd(path)) {
    gGeoManager->PopPath();
    LOG(error) << "Volume path " << path << " not valid!";
    return nullptr;
  }
  matTmp = *gGeoManager->GetCurrentMatrix();
  gGeoManager->PopPath();
  return &matTmp;
}

//______________________________________________________________________
Bool_t GeometryManager::getOriginalMatrix(DetID detid, int sensid, TGeoHMatrix& m)
{
  /**
   * Get position matrix (original) of sensitive volume sensid of detector detid. Slow
   **/
  m.Clear();

  const char* symname = getSymbolicName(detid, sensid);
  if (!symname) {
    return kFALSE;
  }

  return getOriginalMatrix(symname, m);
}

//______________________________________________________________________
bool GeometryManager::applyAlignment(const std::vector<const std::vector<o2::detectors::AlignParam>*> algPars)
{
  /// misalign geometry with alignment objects from the array, optionaly check overlaps
  for (auto dv : algPars) {
    if (dv && !applyAlignment(*dv)) {
      return false;
    }
  }
  return true;
}

//______________________________________________________________________
bool GeometryManager::applyAlignment(const std::vector<o2::detectors::AlignParam>& algPars)
{
  /// misalign geometry with alignment objects from the array, optionaly check overlaps
  int nvols = algPars.size();
  std::vector<int> ord(nvols);
  std::iota(std::begin(ord), std::end(ord), 0); // sort to apply alignment in correct hierarchy
  std::sort(std::begin(ord), std::end(ord), [&algPars](int a, int b) { return algPars[a].getLevel() < algPars[b].getLevel(); });

  bool res = true;
  for (int i = 0; i < nvols; i++) {
    if (!algPars[ord[i]].applyToGeometry()) {
      res = false;
      LOG(error) << "Error applying alignment object for volume" << algPars[ord[i]].getSymName();
    }
  }
  return res;
}

// ================= methods for nested MatBudgetExt class ================

//______________________________________________________________________
void GeometryManager::MatBudgetExt::normalize(double step)
{
  double nrm = 1. / step;
  meanRho *= nrm;
  meanA *= nrm;
  meanZ *= nrm;
  meanZ2A *= nrm;
  if (nrm > 0.) {
    length = step;
  }
}

//______________________________________________________________________
void GeometryManager::accountMaterial(const TGeoMaterial* material, GeometryManager::MatBudgetExt& bd)
{
  bd.meanRho = material->GetDensity();
  bd.meanX2X0 = material->GetRadLen();
  bd.meanA = material->GetA();
  bd.meanZ = material->GetZ();
  if (material->IsMixture()) {
    TGeoMixture* mixture = (TGeoMixture*)material;
    Double_t norm = 0.;
    bd.meanZ2A = 0.;
    for (Int_t iel = 0; iel < mixture->GetNelements(); iel++) {
      norm += mixture->GetWmixt()[iel];
      bd.meanZ2A += mixture->GetZmixt()[iel] * mixture->GetWmixt()[iel] / mixture->GetAmixt()[iel];
    }
    bd.meanZ2A /= norm;
  } else {
    bd.meanZ2A = bd.meanZ / bd.meanA;
  }
}

//_____________________________________________________________________________________
GeometryManager::MatBudgetExt GeometryManager::meanMaterialBudgetExt(float x0, float y0, float z0, float x1, float y1, float z1)
{
  //
  // TODO? It seems there is no real nead for extended material budget, consider eliminating it
  //
  // Calculate mean material budget and material properties (extended version) between
  //    the points "0" and "1".
  //
  // see MatBudgetExt data members for provided information
  //
  //  Origin:  Marian Ivanov, Marian.Ivanov@cern.ch
  //
  //  Corrections and improvements by
  //        Andrea Dainese, Andrea.Dainese@lnl.infn.it,
  //        Andrei Gheata,  Andrei.Gheata@cern.ch
  //
  //  Ported to O2: ruben.shahoyan@cern.ch
  //
  if (!gGeoManager) {
    throw std::runtime_error("meanMaterialBudgetExt requires geometry loaded");
  }
  double length, startD[3] = {x0, y0, z0};
  double dir[3] = {x1 - x0, y1 - y0, z1 - z0};
  if ((length = dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]) < TGeoShape::Tolerance() * TGeoShape::Tolerance()) {
    return MatBudgetExt(); // return empty struct
  }
  length = TMath::Sqrt(length);
  double invlen = 1. / length;
  for (int i = 3; i--;) {
    dir[i] *= invlen;
  }
  std::lock_guard<std::mutex> guard(sTGMutex);
  // Initialize start point and direction
  TGeoNode* currentnode = gGeoManager->InitTrack(startD, dir);
  if (!currentnode) {
    LOG(error) << "start point out of geometry: " << x0 << ':' << y0 << ':' << z0;
    return MatBudgetExt(); // return empty struct
  }

  MatBudgetExt budTotal, budStep;
  accountMaterial(currentnode->GetVolume()->GetMedium()->GetMaterial(), budStep);
  budStep.length = length;

  // Locate next boundary within length without computing safety.
  // Propagate either with length (if no boundary found) or just cross boundary
  gGeoManager->FindNextBoundaryAndStep(length, kFALSE);
  Double_t stepTot = 0.0; // Step made
  Double_t step = gGeoManager->GetStep();
  // If no boundary within proposed length, return current step data
  if (!gGeoManager->IsOnBoundary()) {
    budStep.meanX2X0 = budStep.length / budStep.meanX2X0;
    return MatBudgetExt(budStep);
  }
  // Try to cross the boundary and see what is next
  Int_t nzero = 0;
  while (length > TGeoShape::Tolerance()) {
    if (step < 2. * TGeoShape::Tolerance()) {
      nzero++;
    } else {
      nzero = 0;
    }
    if (nzero > 3) {
      // This means navigation has problems on one boundary
      // Try to cross by making a small step
      const double* curPos = gGeoManager->GetCurrentPoint();
      LOG(warning) << "Cannot cross boundary at (" << curPos[0] << ',' << curPos[1] << ',' << curPos[2] << ')';
      budTotal.normalize(stepTot);
      budTotal.nCross = -1; // flag failed navigation
      return MatBudgetExt(budTotal);
    }
    stepTot += step;

    budTotal.meanRho += step * budStep.meanRho;
    budTotal.meanX2X0 += step / budStep.meanX2X0;
    budTotal.meanA += step * budStep.meanA;
    budTotal.meanZ += step * budStep.meanZ;
    budTotal.meanZ2A += step * budStep.meanZ2A;
    budTotal.nCross++;

    if (step >= length) {
      break;
    }
    currentnode = gGeoManager->GetCurrentNode();
    if (!currentnode) {
      break;
    }
    length -= step;
    accountMaterial(currentnode->GetVolume()->GetMedium()->GetMaterial(), budStep);
    gGeoManager->FindNextBoundaryAndStep(length, kFALSE);
    step = gGeoManager->GetStep();
  }
  budTotal.normalize(stepTot);
  return MatBudgetExt(budTotal);
}

//_____________________________________________________________________________________
o2::base::MatBudget GeometryManager::meanMaterialBudget(float x0, float y0, float z0, float x1, float y1, float z1)
{
  //
  // Calculate mean material budget and material properties between
  //    the points "0" and "1".
  //
  //  see MatBudget data members for provided information
  //
  //  Origin:  Marian Ivanov, Marian.Ivanov@cern.ch
  //
  //  Corrections and improvements by
  //        Andrea Dainese, Andrea.Dainese@lnl.infn.it,
  //        Andrei Gheata,  Andrei.Gheata@cern.ch
  //
  //  Ported to O2: ruben.shahoyan@cern.ch
  //

  double length, startD[3] = {x0, y0, z0};
  double dir[3] = {x1 - x0, y1 - y0, z1 - z0};
  if ((length = dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]) < TGeoShape::Tolerance() * TGeoShape::Tolerance()) {
    return o2::base::MatBudget(); // return empty struct
  }
  length = TMath::Sqrt(length);
  double invlen = 1. / length;
  for (int i = 3; i--;) {
    dir[i] *= invlen;
  }
  std::lock_guard<std::mutex> guard(sTGMutex);
  // Initialize start point and direction
  TGeoNode* currentnode = gGeoManager->InitTrack(startD, dir);
  if (!currentnode) {
    LOG(error) << "start point out of geometry: " << x0 << ':' << y0 << ':' << z0;
    return o2::base::MatBudget(); // return empty struct
  }

  o2::base::MatBudget budTotal, budStep;
  accountMaterial(currentnode->GetVolume()->GetMedium()->GetMaterial(), budStep);
  budStep.length = length;

  // Locate next boundary within length without computing safety.
  // Propagate either with length (if no boundary found) or just cross boundary
  gGeoManager->FindNextBoundaryAndStep(length, kFALSE);
  Double_t stepTot = 0.0; // Step made
  Double_t step = gGeoManager->GetStep();
  // If no boundary within proposed length, return current step data
  if (!gGeoManager->IsOnBoundary()) {
    budStep.meanX2X0 = budStep.length / budStep.meanX2X0;
    return o2::base::MatBudget(budStep);
  }
  // Try to cross the boundary and see what is next
  Int_t nzero = 0;
  while (length > TGeoShape::Tolerance()) {
    if (step < 2. * TGeoShape::Tolerance()) {
      nzero++;
    } else {
      nzero = 0;
    }
    if (nzero > 3) {
      // This means navigation has problems on one boundary
      // Try to cross by making a small step
      const double* curPos = gGeoManager->GetCurrentPoint();
      LOG(warning) << "Cannot cross boundary at (" << curPos[0] << ',' << curPos[1] << ',' << curPos[2] << ')';
      budTotal.meanRho /= stepTot;
      budTotal.length = stepTot;
      return o2::base::MatBudget(budTotal);
    }
    stepTot += step;

    budTotal.meanRho += step * budStep.meanRho;
    budTotal.meanX2X0 += step / budStep.meanX2X0;

    if (step >= length) {
      break;
    }
    currentnode = gGeoManager->GetCurrentNode();
    if (!currentnode) {
      break;
    }
    length -= step;
    accountMaterial(currentnode->GetVolume()->GetMedium()->GetMaterial(), budStep);
    gGeoManager->FindNextBoundaryAndStep(length, kFALSE);
    step = gGeoManager->GetStep();
  }
  budTotal.meanRho /= stepTot;
  budTotal.length = stepTot;
  return o2::base::MatBudget(budTotal);
}

//_________________________________
void GeometryManager::applyMisalignent(bool applyMisalignment)
{
  ///< load geometry from file
  if (!isGeometryLoaded()) {
    LOG(fatal) << "geometry is not loaded";
  }
  if (applyMisalignment) {
    auto& aligner = Aligner::Instance();
    aligner.applyAlignment();
  }
}

//_________________________________
void GeometryManager::loadGeometry(std::string_view simPrefix, bool applyMisalignment, bool preferAlignedFile)
{
  auto loadGeom = [](const std::string_view fname) {
    LOG(info) << "Loading geometry from " << fname;
    TFile flGeom(fname.data());
    if (flGeom.IsZombie()) {
      LOG(fatal) << "Failed to open file " << fname;
    }
    // try under the standard CCDB name
    if (!flGeom.Get(std::string(o2::base::NameConf::CCDBOBJECT).c_str()) &&
        !flGeom.Get(std::string(o2::base::NameConf::GEOMOBJECTNAME_FAIR).c_str())) {
      LOG(fatal) << "Did not find geometry named " << o2::base::NameConf::CCDBOBJECT << " or " << o2::base::NameConf::GEOMOBJECTNAME_FAIR;
    }
  };

  if (preferAlignedFile) {
    ///< load directly from aligned file and apply alignment on top
    loadGeom(o2::base::NameConf::getAlignedGeomFileName(simPrefix));
  } else {
    ///< load geometry from unaligned file and apply alignment on top
    loadGeom(o2::base::NameConf::getGeomFileName(simPrefix));
    applyMisalignent(applyMisalignment);
  }
}
