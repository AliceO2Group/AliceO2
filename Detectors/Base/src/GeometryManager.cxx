// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GeometryManager.cxx
/// \brief Implementation of the GeometryManager class

#include "DetectorsBase/GeometryManager.h"

#include "FairLogger.h" // for LOG

#include "TCollection.h"      // for TIter
#include "TGeoManager.h"      // for TGeoManager
#include "TGeoMatrix.h"       // for TGeoHMatrix
#include "TGeoNode.h"         // for TGeoNode
#include "TGeoPhysicalNode.h" // for TGeoPhysicalNode, TGeoPNEntry
#include "TObjArray.h"        // for TObjArray
#include "TObject.h"          // for TObject

#include <cassert>
#include <cstddef> // for NULL

using namespace o2::Base;

ClassImp(o2::Base::GeometryManager)

  TGeoManager* GeometryManager::sGeometry = nullptr;

/// Implementation of GeometryManager, the geometry manager class which interfaces to TGeo and
/// the look-up table mapping unique volume indices to symbolic volume names. For that, it
/// collects several static methods

GeometryManager::GeometryManager()
{
  /// default constructor

  /// make sure detectors masks can be encoded in the combined det+sensor id
  static_assert(sizeof(Int_t) * 8 - sDetOffset > DetID::getNDetectors(),
                "N detectors exceeds available N bits for their encoding");
}

Bool_t GeometryManager::getOriginalMatrix(const char* symname, TGeoHMatrix& m)
{
  m.Clear();

  if (!sGeometry || !sGeometry->IsClosed()) {
    LOG(ERROR) << "No active geometry or geometry not yet closed!" << FairLogger::endl;
    return kFALSE;
  }

  if (!sGeometry->GetListOfPhysicalNodes()) {
    LOG(WARNING) << "gGeoManager doesn't contain any aligned nodes!" << FairLogger::endl;

    if (!sGeometry->cd(symname)) {
      LOG(ERROR) << "Volume path " << symname << " not valid!" << FairLogger::endl;
      return kFALSE;
    } else {
      m = *sGeometry->GetCurrentMatrix();
      return kTRUE;
    }
  }

  TGeoPNEntry* pne = sGeometry->GetAlignableEntry(symname);
  const char* path = nullptr;

  if (pne) {
    m = *pne->GetGlobalOrig();
    return kTRUE;
  } else {
    LOG(WARNING) << "The symbolic volume name " << symname
                 << "does not correspond to a physical entry. Using it as a volume path!" << FairLogger::endl;
    path = symname;
  }

  return getOriginalMatrixFromPath(path, m);
}

Bool_t GeometryManager::getOriginalMatrixFromPath(const char* path, TGeoHMatrix& m)
{
  m.Clear();

  if (!sGeometry || !sGeometry->IsClosed()) {
    LOG(ERROR) << "Can't get the original global matrix! gGeoManager doesn't exist or it is still opened!"
               << FairLogger::endl;
    return kFALSE;
  }

  if (!sGeometry->CheckPath(path)) {
    LOG(ERROR) << "Volume path " << path << " not valid!" << FairLogger::endl;
    return kFALSE;
  }

  TIter next(sGeometry->GetListOfPhysicalNodes());
  sGeometry->cd(path);

  while (sGeometry->GetLevel()) {
    TGeoPhysicalNode* physNode = nullptr;
    next.Reset();
    TGeoNode* node = sGeometry->GetCurrentNode();

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

    sGeometry->CdUp();
  }
  return kTRUE;
}

const char* GeometryManager::getSymbolicName(DetID detid, int sensid)
{
  /**
   * Get symoblic name of sensitive volume sensid of detector detid
   **/
  int id = getSensID(detid, sensid);
  TGeoPNEntry* pne = gGeoManager->GetAlignableEntryByUID(id);
  if (!pne) {
    LOG(ERROR) << "Failed to find alignable entry with index " << id << ": Det" << detid << " Sens.Vol:" << sensid
               << ") !" << FairLogger::endl;
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
    LOG(ERROR) << "The sens.vol " << sensid << " of det " << detid << " does not correspond to a physical entry!"
               << FairLogger::endl;
  }
  return pne;
}

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
    LOG(ERROR) << "Volume path " << path << " not valid!" << FairLogger::endl;
    return nullptr;
  }
  matTmp = *gGeoManager->GetCurrentMatrix();
  gGeoManager->PopPath();
  return &matTmp;
}

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
