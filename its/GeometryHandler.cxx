/// \file GeometryHandler.cxx
/// \brief Implementation of the GeometryHandler class
/// \author F. Uhlig <f.uhlig@gsi.de>

#include "GeometryHandler.h"

#include "FairLogger.h" // for FairLogger, etc

#include "TGeoBBox.h"    // for TGeoBBox
#include "TGeoManager.h" // for TGeoManager, gGeoManager
#include "TGeoNode.h"    // for TGeoNode
#include "TGeoVolume.h"  // for TGeoVolume
#include "TVirtualMC.h"  // for TVirtualMC, gMC

#include <stdio.h>  // for printf
#include <string.h> // for NULL, strlen, strncpy
#include <iostream> // for cout, endl
#include <map>      // for map
#include <utility>  // for pair

using std::map;
using std::pair;
using std::cout;
using std::endl;

using namespace AliceO2::ITS;

GeometryHandler::GeometryHandler()
  : TObject(),
    mIsSimulation(kFALSE),
    mLastUsedDetectorId(0),
    mGeometryPathHash(0),
    mCurrentVolume(NULL),
    mVolumeShape(NULL),
    mGlobalCentre(),
    mGlobalMatrix(NULL)
{
}

Int_t GeometryHandler::Init(Bool_t isSimulation)
{
  //  Int_t geoVersion = CheckGeometryVersion();

  mIsSimulation = isSimulation;

  return 1;
}

void GeometryHandler::LocalToGlobal(Double_t* local, Double_t* global, Int_t detectorId)
{
  TString path = ConstructFullPathFromDetectorId(detectorId);
  NavigateTo(path);
  gGeoManager->LocalToMaster(local, global);
}

TString GeometryHandler::ConstructFullPathFromDetectorId(Int_t detectorId)
{
  TString volumeString = "/cave_1/tutorial4_0/tut4_det_";
  TString volumePath = volumeString;
  volumePath += detectorId;
  return volumePath;
}

Int_t GeometryHandler::GetUniqueDetectorId(TString volumeName)
{
  if (mGeometryPathHash != volumeName.Hash()) {
    NavigateTo(volumeName);
  }
  return GetUniqueDetectorId();
}

Int_t GeometryHandler::GetUniqueDetectorId()
{
  Int_t detectorNumber = 0;

  CurrentVolumeOffId(0, detectorNumber);

  return detectorNumber;
}

Int_t GeometryHandler::VolumeIdGeo(const char* name) const
{
  Int_t uid = gGeoManager->GetUID(name);
  if (uid < 0) {
    printf("VolId: Volume %s not found\n", name);
    return 0;
  }
  return uid;
}

Int_t GeometryHandler::VolumeId(const Text_t* name) const
{
  if (mIsSimulation) {
    return gMC->VolId(name);
  } else {
    char sname[20];
    Int_t length = strlen(name) - 1;

    if (name[length] != ' ') {
      return VolumeIdGeo(name);
    }

    strncpy(sname, name, length);
    sname[length] = 0;
    return VolumeIdGeo(sname);
  }
}

Int_t GeometryHandler::CurrentVolumeId(Int_t& copy) const
{
  if (mIsSimulation) {
    return gMC->CurrentVolID(copy);
  } else {
    if (gGeoManager->IsOutside()) {
      return 0;
    }
    TGeoNode* node = gGeoManager->GetCurrentNode();
    copy = node->GetNumber();
    Int_t id = node->GetVolume()->GetNumber();
    return id;
  }
}

//_____________________________________________________________________________
Int_t GeometryHandler::CurrentVolumeOffId(Int_t off, Int_t& copy) const
{
  if (mIsSimulation) {
    return gMC->CurrentVolOffID(off, copy);
  } else {
    if (off < 0 || off > gGeoManager->GetLevel()) {
      return 0;
    }

    if (off == 0) {
      return CurrentVolumeId(copy);
    }

    TGeoNode* node = gGeoManager->GetMother(off);

    if (!node) {
      return 0;
    }

    copy = node->GetNumber();
    return node->GetVolume()->GetNumber();
  }
}

const char* GeometryHandler::CurrentVolumeName() const
{
  if (mIsSimulation) {
    return gMC->CurrentVolName();
  } else {
    if (gGeoManager->IsOutside()) {
      return gGeoManager->GetTopVolume()->GetName();
    }

    return gGeoManager->GetCurrentVolume()->GetName();
  }
}

const char* GeometryHandler::CurrentVolumeOffName(Int_t off) const
{
  if (mIsSimulation) {
    return gMC->CurrentVolOffName(off);
  } else {
    if (off < 0 || off > gGeoManager->GetLevel()) {
      return 0;
    }

    if (off == 0) {
      return CurrentVolumeName();
    }

    TGeoNode* node = gGeoManager->GetMother(off);

    if (!node) {
      return 0;
    }

    return node->GetVolume()->GetName();
  }
}

void GeometryHandler::NavigateTo(TString volumeName)
{
  if (mIsSimulation) {
    LOG(FATAL) << "This method is not supported in simulation mode" << FairLogger::endl;
  } else {
    gGeoManager->cd(volumeName.Data());
    mGeometryPathHash = volumeName.Hash();
    mCurrentVolume = gGeoManager->GetCurrentVolume();
    mVolumeShape = (TGeoBBox*)mCurrentVolume->GetShape();
    Double_t local[3] = { 0., 0., 0. }; // Local centre of volume
    gGeoManager->LocalToMaster(local, mGlobalCentre);
    LOG(DEBUG2) << "Pos: " << mGlobalCentre[0] << " , " << mGlobalCentre[1] << " , "
                << mGlobalCentre[2] << FairLogger::endl;
    //    mGlobalMatrix = gGeoManager->GetCurrentMatrix();
  }
}

ClassImp(AliceO2::ITS::GeometryHandler)
