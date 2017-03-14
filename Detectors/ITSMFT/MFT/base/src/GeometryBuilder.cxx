/// \file GeometryBuilder.cxx
/// \brief Class describing MFT Geometry Builder
/// \author Raphael Tieulent <raphael.tieulent@cern.ch>

#include "MFTBase/Constants.h"
#include "MFTBase/Geometry.h"
#include "MFTBase/GeometryTGeo.h"
#include "MFTBase/GeometryBuilder.h"
#include "MFTBase/Segmentation.h"
#include "MFTBase/HalfSegmentation.h"
#include "MFTBase/HalfDetector.h"
#include "MFTBase/HalfCone.h"

#include "TGeoVolume.h"
#include "TGeoManager.h"

#include "FairLogger.h"

using namespace AliceO2::MFT;

ClassImp(AliceO2::MFT::GeometryBuilder)

//_____________________________________________________________________________
GeometryBuilder::GeometryBuilder():TNamed()
{
  // default constructor

}

//_____________________________________________________________________________
GeometryBuilder::~GeometryBuilder()
{
  // destructor

}

//_____________________________________________________________________________
/// \brief Build the MFT Geometry
void GeometryBuilder::BuildGeometry()
{

  Geometry *mftGeo = Geometry::Instance();

  TGeoVolume *volMFT = new TGeoVolumeAssembly(GeometryTGeo::GetVolumeName());

  LOG(INFO) << "GeometryBuilder::BuildGeometry volume name = " << GeometryTGeo::GetVolumeName() << FairLogger::endl;

  TGeoVolume *vALIC = gGeoManager->GetVolume("cave");
  if (!vALIC) {
    LOG(FATAL) << "Could not find the top volume" << FairLogger::endl;
  }

  Info("BuildGeometry",Form("gGeoManager name is %s title is %s \n",gGeoManager->GetName(),gGeoManager->GetTitle()),0,0);

  Segmentation *seg = mftGeo->GetSegmentation();
  
  for (int iHalf = 0; iHalf < 2; iHalf++) {
    HalfSegmentation *halfSeg = seg->GetHalf(iHalf);
    HalfDetector *halfMFT = new HalfDetector(halfSeg);
    volMFT->AddNode(halfMFT->GetVolume(),iHalf,halfSeg->GetTransformation());
    delete halfMFT;
  }

  /// \todo Add the service, Barrel, etc Those objects will probably be defined into the COMMON ITSMFT area.
  
  HalfCone * halfCone = new HalfCone();
  TGeoVolumeAssembly * halfCone1 = halfCone->CreateHalfCone(0);
  TGeoVolumeAssembly * halfCone2 = halfCone->CreateHalfCone(1);
  volMFT->AddNode(halfCone1,1);
  volMFT->AddNode(halfCone2,1);
  
  vALIC->AddNode(volMFT,0);

}

