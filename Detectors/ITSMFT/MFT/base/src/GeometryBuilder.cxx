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

using namespace o2::MFT;

ClassImp(o2::MFT::GeometryBuilder)

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
void GeometryBuilder::buildGeometry()
{

  Geometry *mftGeo = Geometry::instance();

  TGeoVolume *volMFT = new TGeoVolumeAssembly(GeometryTGeo::getVolumeName());

  LOG(INFO) << "GeometryBuilder::BuildGeometry volume name = " << GeometryTGeo::getVolumeName() << FairLogger::endl;

  TGeoVolume *vALIC = gGeoManager->GetVolume("cave");
  if (!vALIC) {
    LOG(FATAL) << "Could not find the top volume" << FairLogger::endl;
  }

  Info("BuildGeometry",Form("gGeoManager name is %s title is %s \n",gGeoManager->GetName(),gGeoManager->GetTitle()),0,0);

  Segmentation *seg = mftGeo->getSegmentation();
  
  for (int iHalf = 0; iHalf < 2; iHalf++) {
    HalfSegmentation *halfSeg = seg->getHalf(iHalf);
    auto *halfMFT = new HalfDetector(halfSeg);
    volMFT->AddNode(halfMFT->getVolume(),iHalf,halfSeg->getTransformation());
    delete halfMFT;
  }

  /// \todo Add the service, Barrel, etc Those objects will probably be defined into the COMMON ITSMFT area.
  /*
  auto * halfCone = new HalfCone();
  TGeoVolumeAssembly * halfCone1 = halfCone->createHalfCone(0);
  TGeoVolumeAssembly * halfCone2 = halfCone->createHalfCone(1);
  volMFT->AddNode(halfCone1,1);
  volMFT->AddNode(halfCone2,1);
  */
  vALIC->AddNode(volMFT,0);

}

