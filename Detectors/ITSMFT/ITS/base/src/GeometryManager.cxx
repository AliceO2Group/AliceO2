/// \file GeometryManager.cxx
/// \brief Implementation of the GeometryManager class

#include "ITSBase/GeometryManager.h"

#include "FairLogger.h" // for LOG

#include "TCollection.h"      // for TIter
#include "TGeoManager.h"      // for TGeoManager
#include "TGeoMatrix.h"       // for TGeoHMatrix
#include "TGeoNode.h"         // for TGeoNode
#include "TGeoPhysicalNode.h" // for TGeoPhysicalNode, TGeoPNEntry
#include "TObjArray.h"        // for TObjArray
#include "TObject.h"          // for TObject

#include <cstddef> // for NULL

using namespace o2::ITS;

ClassImp(o2::ITS::GeometryManager)

  TGeoManager* GeometryManager::mGeometry = nullptr;

/// Implementation of GeometryManager, the geometry manager class which interfaces to TGeo and
/// the look-up table mapping unique volume indices to symbolic volume names. For that, it
/// collects several static methods
GeometryManager::GeometryManager() : TObject() {}
GeometryManager::~GeometryManager() = default;
Bool_t GeometryManager::getOriginalGlobalMatrix(const char* symname, TGeoHMatrix& m)
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
  const char* path = nullptr;

  if (pne) {
    m = *pne->GetGlobalOrig();
    return kTRUE;
  } else {
    LOG(WARNING) << "The symbolic volume name " << symname
                 << "does not correspond to a physical entry. Using it as a volume path!" << FairLogger::endl;
    path = symname;
  }

  return getOriginalGlobalMatrixFromPath(path, m);
}

Bool_t GeometryManager::getOriginalGlobalMatrixFromPath(const char* path, TGeoHMatrix& m)
{
  m.Clear();

  if (!mGeometry || !mGeometry->IsClosed()) {
    LOG(ERROR) << "Can't get the original global matrix! gGeoManager doesn't exist or it is still opened!"
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
    TGeoPhysicalNode* physNode = nullptr;
    next.Reset();
    TGeoNode* node = mGeometry->GetCurrentNode();

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

    mGeometry->CdUp();
  }
  return kTRUE;
}
