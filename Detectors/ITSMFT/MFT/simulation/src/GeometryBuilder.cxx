/// \file GeometryBuilder.cxx
/// \brief Class describing MFT Geometry Builder
/// \author Raphael Tieulent <raphael.tieulent@cern.ch>

#include "MFTBase/Constants.h"
#include "MFTSimulation/Geometry.h"
#include "MFTSimulation/GeometryTGeo.h"
#include "MFTSimulation/GeometryBuilder.h"
#include "MFTSimulation/Segmentation.h"
#include "MFTSimulation/HalfSegmentation.h"
#include "MFTSimulation/HalfDetector.h"

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

  Info("BuildGeometry",Form("gGeoManager name is %s title is %s \n",gGeoManager->GetName(),gGeoManager->GetTitle()),0,0);

  Segmentation *seg = mftGeo->GetSegmentation();
  
  for (int iHalf = 0; iHalf < 2; iHalf++) {
    HalfSegmentation *halfSeg = seg->GetHalf(iHalf);
    HalfDetector *halfMFT = new HalfDetector(halfSeg);
    volMFT->AddNode(halfMFT->GetVolume(),iHalf,halfSeg->GetTransformation());
    delete halfMFT;
  }

  gGeoManager->GetVolume("cave")->AddNode(volMFT,0);

}

