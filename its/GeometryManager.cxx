/**************************************************************************
 * Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 *                                                                        *
 * Author: The ALICE Off-line Project.                                    *
 * Contributors are mentioned in the code where appropriate.              *
 *                                                                        *
 * Permission to use, copy, modify and distribute this software and its   *
 * documentation strictly for non-commercial purposes is hereby granted   *
 * without fee, provided that the above copyright notice appears in all   *
 * copies and that both the copyright notice and this permission notice   *
 * appear in the supporting documentation. The authors make no claims     *
 * about the suitability of this software for any purpose. It is          *
 * provided "as is" without express or implied warranty.                  *
 **************************************************************************/
//-------------------------------------------------------------------------
//   Implementation of GeometryManager, the geometry manager class 
//   which interfaces to TGeo and the look-up table mapping unique
//   volume indices to symbolic volume names. For that it collects
//   several static methods.
//-------------------------------------------------------------------------

#include <TClass.h>
#include <TFile.h>
#include <TGeoManager.h>
#include <TObjString.h>
#include <TGeoPhysicalNode.h>
#include <TClonesArray.h>
#include <TGeoMatrix.h>
#include <TGeoPhysicalNode.h>
#include <TSystem.h>
#include <TStopwatch.h>
#include <TGeoOverlap.h>
#include <TPluginManager.h>
#include <TROOT.h>

#include "GeometryManager.h"

#include "FairLogger.h"

using namespace AliceO2::ITS;

ClassImp(GeometryManager)

TGeoManager* GeometryManager::fgGeometry = 0x0;

GeometryManager::GeometryManager():
  TObject()
{
  // default constructor
}

GeometryManager::~GeometryManager()
{
  // dummy destructor
}

Bool_t GeometryManager::GetOrigGlobalMatrix(const char* symname, TGeoHMatrix &m) 
{
  // Get the global transformation matrix (ideal geometry) for a given alignable volume
  // The alignable volume is identified by 'symname' which has to be either a valid symbolic
  // name, the query being performed after alignment, or a valid volume path if the query is
  // performed before alignment.
  //
  m.Clear();

  if (!fgGeometry || !fgGeometry->IsClosed()) {
    LOG(ERROR) << "No active geometry or geometry not yet closed!" << FairLogger::endl;
    return kFALSE;
  }
  if (!fgGeometry->GetListOfPhysicalNodes()) {
     LOG(WARNING) << "gGeoManager doesn't contain any aligned nodes!" << FairLogger::endl;
    if (!fgGeometry->cd(symname)) {
      LOG(ERROR) << "Volume path " << symname << " not valid!" << FairLogger::endl;
      return kFALSE;
    }
    else {
      m = *fgGeometry->GetCurrentMatrix();
      return kTRUE;
    }
  }

  TGeoPNEntry* pne = fgGeometry->GetAlignableEntry(symname);
  const char* path = NULL;
  if(pne){
    m = *pne->GetGlobalOrig();
    return kTRUE;
  }else{
    LOG(WARNING) << "The symbolic volume name " << symname
                 << "does not correspond to a physical entry. Using it as a volume path!"
                 << FairLogger::endl;
    path=symname;
  }

  return GetOrigGlobalMatrixFromPath(path,m);
}

Bool_t GeometryManager::GetOrigGlobalMatrixFromPath(const char *path, TGeoHMatrix &m)
{
  // The method returns the global matrix for the volume identified by 
  // 'path' in the ideal detector geometry.
  // The output global matrix is stored in 'm'.
  // Returns kFALSE in case TGeo has not been initialized or the volume
  // path is not valid.
  //
  m.Clear();

  if (!fgGeometry || !fgGeometry->IsClosed()) {
    LOG(ERROR) << "Can't get the original global matrix! gGeoManager doesn't exist or it is still opened!"
               << FairLogger::endl;
    return kFALSE;
  }

  if (!fgGeometry->CheckPath(path)) {
    LOG(ERROR) << "Volume path " << path << " not valid!" << FairLogger::endl;
    return kFALSE;
  }

  TIter next(fgGeometry->GetListOfPhysicalNodes());
  fgGeometry->cd(path);

  while(fgGeometry->GetLevel()){

    TGeoPhysicalNode *physNode = NULL;
    next.Reset();
    TGeoNode *node = fgGeometry->GetCurrentNode();
    while ((physNode=(TGeoPhysicalNode*)next())) 
      if (physNode->GetNode() == node) break;

    TGeoMatrix *lm = NULL;
    if (physNode) {
      lm = physNode->GetOriginalMatrix();
      if (!lm) lm = node->GetMatrix();
    } else
      lm = node->GetMatrix();

    m.MultiplyLeft(lm);

    fgGeometry->CdUp();
  }

  return kTRUE;
}
