/// \file GeometryHandler.cxx
/// \brief Implementation of the GeometryHandler class
/// \author F. Uhlig <f.uhlig@gsi.de>

#include "ITSSimulation/GeometryHandler.h"

#include "FairLogger.h" // for FairLogger, etc

#include "TGeoBBox.h"    // for TGeoBBox
#include "TGeoManager.h" // for TGeoManager, gGeoManager
#include "TGeoNode.h"    // for TGeoNode
#include "TGeoVolume.h"  // for TGeoVolume
#include "TVirtualMC.h"  // for TVirtualMC, gMC

#include <cstdio>  // for printf
#include <cstring> // for NULL, strlen, strncpy
#include <iostream> // for cout, endl
#include <map>      // for map
#include <utility>  // for pair

using std::map;
using std::pair;
using std::cout;
using std::endl;

using namespace o2::ITS;

GeometryHandler::GeometryHandler()
  : TObject(),
    mIsSimulation(kFALSE),
    mLastUsedDetectorId(0),
    mGeometryPathHash(0),
    mCurrentVolume(nullptr),
    mVolumeShape(nullptr),
    mGlobalCentre(),
    mGlobalMatrix(nullptr)
{
}

Int_t GeometryHandler::Init(Bool_t isSimulation)
{
  //  Int_t geoVersion = CheckGeometryVersion();

  mIsSimulation = isSimulation;

  return 1;
}

void GeometryHandler::localToGlobal(Double_t *local, Double_t *global, Int_t detectorId)
{
  TString path = constructFullPathFromDetectorId(detectorId);
  navigateTo(path);
  gGeoManager->LocalToMaster(local, global);
}

TString GeometryHandler::constructFullPathFromDetectorId(Int_t detectorId)
{
  TString volumeString = "/cave_1/tutorial4_0/tut4_det_";
  TString volumePath = volumeString;
  volumePath += detectorId;
  return volumePath;
}

Int_t GeometryHandler::getUniqueDetectorId(TString volumeName)
{
  if (mGeometryPathHash != volumeName.Hash()) {
    navigateTo(volumeName);
  }
  return getUniqueDetectorId();
}

Int_t GeometryHandler::getUniqueDetectorId()
{
  Int_t detectorNumber = 0;

  currentVolumeOffId(0, detectorNumber);

  return detectorNumber;
}

Int_t GeometryHandler::volumeIdGeo(const char *name) const
{
  Int_t uid = gGeoManager->GetUID(name);
  if (uid < 0) {
    printf("VolId: Volume %s not found\n", name);
    return 0;
  }
  return uid;
}

Int_t GeometryHandler::volumeId(const Text_t *name) const
{
  if (mIsSimulation) {
    return TVirtualMC::GetMC()->VolId(name);
  } else {
    char sname[20];
    Int_t length = strlen(name) - 1;

    if (name[length] != ' ') {
      return volumeIdGeo(name);
    }

    strncpy(sname, name, length);
    sname[length] = 0;
    return volumeIdGeo(sname);
  }
}

Int_t GeometryHandler::currentVolumeId(Int_t &copy) const
{
  if (mIsSimulation) {
    return TVirtualMC::GetMC()->CurrentVolID(copy);
  } else {
    if (gGeoManager->IsOutside()) {
      return 0;
    }
    TGeoNode *node = gGeoManager->GetCurrentNode();
    copy = node->GetNumber();
    Int_t id = node->GetVolume()->GetNumber();
    return id;
  }
}

//_____________________________________________________________________________
Int_t GeometryHandler::currentVolumeOffId(Int_t off, Int_t &copy) const
{
  if (mIsSimulation) {
    return TVirtualMC::GetMC()->CurrentVolOffID(off, copy);
  } else {
    if (off < 0 || off > gGeoManager->GetLevel()) {
      return 0;
    }

    if (off == 0) {
      return currentVolumeId(copy);
    }

    TGeoNode *node = gGeoManager->GetMother(off);

    if (!node) {
      return 0;
    }

    copy = node->GetNumber();
    return node->GetVolume()->GetNumber();
  }
}

const char *GeometryHandler::currentVolumeName() const
{
  if (mIsSimulation) {
    return TVirtualMC::GetMC()->CurrentVolName();
  } else {
    if (gGeoManager->IsOutside()) {
      return gGeoManager->GetTopVolume()->GetName();
    }

    return gGeoManager->GetCurrentVolume()->GetName();
  }
}

const char *GeometryHandler::currentVolumeOffName(Int_t off) const
{
  if (mIsSimulation) {
    return TVirtualMC::GetMC()->CurrentVolOffName(off);
  } else {
    if (off < 0 || off > gGeoManager->GetLevel()) {
      return nullptr;
    }

    if (off == 0) {
      return currentVolumeName();
    }

    TGeoNode *node = gGeoManager->GetMother(off);

    if (!node) {
      return nullptr;
    }

    return node->GetVolume()->GetName();
  }
}

void GeometryHandler::navigateTo(TString volumeName)
{
  if (mIsSimulation) {
    LOG(FATAL) << "This method is not supported in simulation mode" << FairLogger::endl;
  } else {
    gGeoManager->cd(volumeName.Data());
    mGeometryPathHash = volumeName.Hash();
    mCurrentVolume = gGeoManager->GetCurrentVolume();
    mVolumeShape = (TGeoBBox *) mCurrentVolume->GetShape();
    Double_t local[3] = {0., 0., 0.}; // Local centre of volume
    gGeoManager->LocalToMaster(local, mGlobalCentre);
    LOG(DEBUG2) << "Pos: " << mGlobalCentre[0] << " , " << mGlobalCentre[1] << " , "
                << mGlobalCentre[2] << FairLogger::endl;
    //    mGlobalMatrix = gGeoManager->GetCurrentMatrix();
  }
}

ClassImp(o2::ITS::GeometryHandler)
