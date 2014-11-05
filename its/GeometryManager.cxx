/// \file GeometryManager.cxx
/// \brief Implementation of the GeometryManager class

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

ClassImp(AliceO2::ITS::GeometryManager)

TGeoManager* GeometryManager::mGeometry = 0x0;

/// Implementation of GeometryManager, the geometry manager class which interfaces to TGeo and
/// the look-up table mapping unique volume indices to symbolic volume names. For that, it
/// collects several static methods
GeometryManager::GeometryManager() : TObject()
{
}

GeometryManager::~GeometryManager()
{
}

Bool_t GeometryManager::GetOriginalGlobalMatrix(const char* symname, TGeoHMatrix& m)
{
  m.Clear();

  if (!mGeometry || !mGeometry->IsClosed()) {
    LOG(ERROR) << "No active geometry or geometry not yet closed!" << FairLogger::endl;
    return kFALSE;
  }

  if (!mGeometry->GetListOfPhysicalNodes()) {
    LOG(WARNING) << "gGeoManager doesn't contain any aligned nodes!" << FairLogger::endl;

    if (!mGeometry->cd(symname)) {
      LOG(ERROR) << "Volume path " << symname << " not valid!" << FairLogger::endl;
      return kFALSE;
    } else {
      m = *mGeometry->GetCurrentMatrix();
      return kTRUE;
    }
  }

  TGeoPNEntry* pne = mGeometry->GetAlignableEntry(symname);
  const char* path = NULL;

  if (pne) {
    m = *pne->GetGlobalOrig();
    return kTRUE;
  } else {
    LOG(WARNING) << "The symbolic volume name " << symname
                 << "does not correspond to a physical entry. Using it as a volume path!"
                 << FairLogger::endl;
    path = symname;
  }

  return GetOriginalGlobalMatrixFromPath(path, m);
}

Bool_t GeometryManager::GetOriginalGlobalMatrixFromPath(const char* path, TGeoHMatrix& m)
{
  m.Clear();

  if (!mGeometry || !mGeometry->IsClosed()) {
    LOG(ERROR)
      << "Can't get the original global matrix! gGeoManager doesn't exist or it is still opened!"
      << FairLogger::endl;
    return kFALSE;
  }

  if (!mGeometry->CheckPath(path)) {
    LOG(ERROR) << "Volume path " << path << " not valid!" << FairLogger::endl;
    return kFALSE;
  }

  TIter next(mGeometry->GetListOfPhysicalNodes());
  mGeometry->cd(path);

  while (mGeometry->GetLevel()) {
    TGeoPhysicalNode* physNode = NULL;
    next.Reset();
    TGeoNode* node = mGeometry->GetCurrentNode();

    while ((physNode = (TGeoPhysicalNode*)next())) {
      if (physNode->GetNode() == node) {
        break;
      }
    }

    TGeoMatrix* lm = NULL;
    if (physNode) {
      lm = physNode->GetOriginalMatrix();
      if (!lm) {
        lm = node->GetMatrix();
      }
    } else {
      lm = node->GetMatrix();
    }

    m.MultiplyLeft(lm);

    mGeometry->CdUp();
  }
  return kTRUE;
}
