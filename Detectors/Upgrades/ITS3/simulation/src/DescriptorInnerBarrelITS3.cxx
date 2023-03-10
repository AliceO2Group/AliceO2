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

#include "FairDetector.h"      // for FairDetector
#include <fairlogger/Logger.h> // for LOG, LOG_IF
#include "FairRootManager.h"   // for FairRootManager
#include "FairRun.h"           // for FairRun
#include "FairRuntimeDb.h"     // for FairRuntimeDb
#include "FairVolume.h"        // for FairVolume
#include "FairRootManager.h"

#include "TGeoManager.h"     // for TGeoManager, gGeoManager
#include "TGeoTube.h"        // for TGeoTube
#include "TGeoPcon.h"        // for TGeoPcon
#include "TGeoVolume.h"      // for TGeoVolume, TGeoVolumeAssembly
#include "TString.h"         // for TString, operator+
#include "TVirtualMC.h"      // for gMC, TVirtualMC
#include "TVirtualMCStack.h" // for TVirtualMCStack

#include "ITS3Simulation/DescriptorInnerBarrelITS3.h"
#include "ITS3Simulation/DescriptorInnerBarrelITS3Param.h"
#include "ITS3Simulation/ITS3Services.h"
#include "ITS3Base/SegmentationSuperAlpide.h"

using namespace o2::its;
using namespace o2::its3;

/// \cond CLASSIMP
ClassImp(DescriptorInnerBarrelITS3);
/// \endcond

//________________________________________________________________
void DescriptorInnerBarrelITS3::configure()
{
  // set version
  auto& param = DescriptorInnerBarrelITS3Param::Instance();
  int buildLevel = param.mBuildLevel;
  if (param.getITS3LayerConfigString() != "") {
    LOG(info) << "Instance \'DescriptorInnerBarrelITS3\' class with following parameters";
    LOG(info) << param;
    setVersion(param.getITS3LayerConfigString());
  } else {
    LOG(info) << "Instance \'DescriptorInnerBarrelITS3\' class with following parameters";
    LOG(info) << "DescriptorInnerBarrelITS3.mVersion : " << mVersion;
    LOG(info) << "DescriptorInnerBarrelITS3.mBuildLevel : " << buildLevel;
  }

  if (mVersion == "ThreeLayersNoDeadZones") {
    mNumLayers = 3;
  } else if (mVersion == "ThreeLayers") {
    mNumLayers = 3;
  } else if (mVersion == "FourLayers") {
    mNumLayers = 4;
  } else if (mVersion == "FiveLayers") {
    mNumLayers = 5;
  }

  // build ITS3 upgrade detector
  mLayer.resize(mNumLayers);
  mLayerRadii.resize(mNumLayers);
  mLayerZLen.resize(mNumLayers);
  mChipThickness.resize(mNumLayers);
  mChipTypeID.resize(mNumLayers);
  mGap.resize(mNumLayers);
  mNumSubSensorsHalfLayer.resize(mNumLayers);
  mFringeChipWidth.resize(mNumLayers);
  mMiddleChipWidth.resize(mNumLayers);
  mHeightStripFoam.resize(mNumLayers);
  mLengthSemiCircleFoam.resize(mNumLayers);
  mThickGluedFoam.resize(mNumLayers);
  mBuildLevel.resize(mNumLayers);

  const double safety = 0.5;

  // radius, length, gap, num of chips in half layer, fringe chip width, middle chip width, strip foam height, semi-cicle foam length, guled foam width
  std::vector<std::array<double, 10>> IBtdr5dat;
  IBtdr5dat.emplace_back(std::array<double, 10>{1.8f, 27.15, 0.1, 3., 0.06, 0.128, 0.25, 0.8, 0.022, -999.});
  IBtdr5dat.emplace_back(std::array<double, 10>{2.4f, 27.15, 0.1, 4., 0.06, 0.128, 0.25, 0.8, 0.022, -999.});
  IBtdr5dat.emplace_back(std::array<double, 10>{3.0f, 27.15, 0.1, 5., 0.06, 0.128, 0.25, 0.8, 0.022, -999.});
  IBtdr5dat.emplace_back(std::array<double, 10>{6.0f, 27.15, 0.1, 5., 0.06, 0.128, 0.25, 0.8, 0.022, 0.05});

  if (mVersion == "ThreeLayersNoDeadZones") {

    mWrapperMinRadius = IBtdr5dat[0][0] - safety;

    for (auto idLayer{0u}; idLayer < mNumLayers; ++idLayer) {
      mLayerRadii[idLayer] = IBtdr5dat[idLayer][0];
      mLayerZLen[idLayer] = IBtdr5dat[idLayer][1];
      mChipThickness[idLayer] = SegmentationSuperAlpide::mDetectorLayerThickness;
      mGap[idLayer] = IBtdr5dat[idLayer][2];
      mChipTypeID[idLayer] = 0;
      mHeightStripFoam[idLayer] = IBtdr5dat[idLayer][6];
      mLengthSemiCircleFoam[idLayer] = IBtdr5dat[idLayer][7];
      mThickGluedFoam[idLayer] = IBtdr5dat[idLayer][8];
      mBuildLevel[idLayer] = buildLevel;
      LOGP(info, "ITS3 L# {} R:{} Dthick:{} Gap:{} StripFoamHeight:{} SemiCircleFoamLength:{} ThickGluedFoam:{}",
           idLayer, mLayerRadii[idLayer], mChipThickness[idLayer], mGap[idLayer],
           mHeightStripFoam[idLayer], mLengthSemiCircleFoam[idLayer], mThickGluedFoam[idLayer]);
    }
  } else if (mVersion == "ThreeLayers") {

    mWrapperMinRadius = IBtdr5dat[0][0] - safety;

    for (auto idLayer{0u}; idLayer < mNumLayers; ++idLayer) {
      mLayerRadii[idLayer] = IBtdr5dat[idLayer][0];
      mLayerZLen[idLayer] = IBtdr5dat[idLayer][1];
      mChipThickness[idLayer] = SegmentationSuperAlpide::mDetectorLayerThickness;
      mNumSubSensorsHalfLayer[idLayer] = (int)IBtdr5dat[idLayer][3];
      mFringeChipWidth[idLayer] = IBtdr5dat[idLayer][4];
      mMiddleChipWidth[idLayer] = IBtdr5dat[idLayer][5];
      mGap[idLayer] = IBtdr5dat[idLayer][2];
      mChipTypeID[idLayer] = 0;
      mHeightStripFoam[idLayer] = IBtdr5dat[idLayer][6];
      mLengthSemiCircleFoam[idLayer] = IBtdr5dat[idLayer][7];
      mThickGluedFoam[idLayer] = IBtdr5dat[idLayer][8];
      mBuildLevel[idLayer] = buildLevel;
      LOGP(info, "ITS3 L# {} R:{} Dthick:{} Gap:{} NSubSensors:{} FringeChipWidth:{} MiddleChipWidth:{} StripFoamHeight:{} SemiCircleFoamLength:{} ThickGluedFoam:{}",
           idLayer, mLayerRadii[idLayer], mChipThickness[idLayer], mGap[idLayer],
           mNumSubSensorsHalfLayer[idLayer], mFringeChipWidth[idLayer], mMiddleChipWidth[idLayer],
           mHeightStripFoam[idLayer], mLengthSemiCircleFoam[idLayer], mThickGluedFoam[idLayer]);
    }
  } else if (mVersion == "FourLayers") {

    mWrapperMinRadius = IBtdr5dat[0][0] - safety;

    for (auto idLayer{0u}; idLayer < mNumLayers; ++idLayer) {
      mLayerRadii[idLayer] = IBtdr5dat[idLayer][0];
      mLayerZLen[idLayer] = IBtdr5dat[idLayer][1];
      mChipThickness[idLayer] = SegmentationSuperAlpide::mDetectorLayerThickness;
      mNumSubSensorsHalfLayer[idLayer] = (int)IBtdr5dat[idLayer][3];
      mFringeChipWidth[idLayer] = IBtdr5dat[idLayer][4];
      mMiddleChipWidth[idLayer] = IBtdr5dat[idLayer][5];
      mGap[idLayer] = IBtdr5dat[idLayer][2];
      mChipTypeID[idLayer] = 0;
      mHeightStripFoam[idLayer] = IBtdr5dat[idLayer][6];
      mLengthSemiCircleFoam[idLayer] = IBtdr5dat[idLayer][7];
      mThickGluedFoam[idLayer] = IBtdr5dat[idLayer][8];
      mBuildLevel[idLayer] = buildLevel;
      if (idLayer == 3) {
        mGapXDirection4thLayer = IBtdr5dat[idLayer][9];
        LOGP(info, "ITS3 L# {} R:{} Dthick:{} Gap:{} NSubSensors:{} FringeChipWidth:{} MiddleChipWidth:{} StripFoamHeight:{} SemiCircleFoamLength:{} ThickGluedFoam:{}, GapXDirection4thLayer:{}",
             3, mLayerRadii[idLayer], mChipThickness[idLayer], mGap[idLayer],
             mNumSubSensorsHalfLayer[idLayer], mFringeChipWidth[idLayer], mMiddleChipWidth[idLayer],
             mHeightStripFoam[idLayer], mLengthSemiCircleFoam[idLayer], mThickGluedFoam[idLayer], mGapXDirection4thLayer);
      } else {
        LOGP(info, "ITS3 L# {} R:{} Dthick:{} Gap:{} NSubSensors:{} FringeChipWidth:{} MiddleChipWidth:{} StripFoamHeight:{} SemiCircleFoamLength:{} ThickGluedFoam:{}",
             idLayer, mLayerRadii[idLayer], mChipThickness[idLayer], mGap[idLayer],
             mNumSubSensorsHalfLayer[idLayer], mFringeChipWidth[idLayer], mMiddleChipWidth[idLayer],
             mHeightStripFoam[idLayer], mLengthSemiCircleFoam[idLayer], mThickGluedFoam[idLayer]);
      }
    }
  } else if (mVersion == "FiveLayers") {
    LOGP(fatal, "ITS3 version FiveLayers not yet implemented.");
  } else {
    LOGP(fatal, "ITS3 version {} not supported.", mVersion.data());
  }
}

//________________________________________________________________
ITS3Layer* DescriptorInnerBarrelITS3::createLayer(int idLayer, TGeoVolume* dest)
{
  if (idLayer >= mNumLayers) {
    LOGP(fatal, "Trying to define layer {} of inner barrel, but only {} layers expected!", idLayer, mNumLayers);
    return nullptr;
  }

  mLayer[idLayer] = new ITS3Layer(idLayer);
  mLayer[idLayer]->setChipThick(mChipThickness[idLayer]);
  mLayer[idLayer]->setLayerRadius(mLayerRadii[idLayer]);
  mLayer[idLayer]->setLayerZLen(mLayerZLen[idLayer]);
  mLayer[idLayer]->setGapBetweenEmispheres(mGap[idLayer]);
  mLayer[idLayer]->setChipID(mChipTypeID[idLayer]);
  mLayer[idLayer]->setHeightStripFoam(mHeightStripFoam[idLayer]);
  mLayer[idLayer]->setLengthSemiCircleFoam(mLengthSemiCircleFoam[idLayer]);
  mLayer[idLayer]->setThickGluedFoam(mThickGluedFoam[idLayer]);
  mLayer[idLayer]->setBuildLevel(mBuildLevel[idLayer]);
  if (mVersion == "ThreeLayersNoDeadZones") {
    mLayer[idLayer]->createLayer(dest);
  } else if (mVersion == "ThreeLayers") {
    mLayer[idLayer]->setFringeChipWidth(mFringeChipWidth[idLayer]);
    mLayer[idLayer]->setMiddleChipWidth(mMiddleChipWidth[idLayer]);
    mLayer[idLayer]->setNumSubSensorsHalfLayer(mNumSubSensorsHalfLayer[idLayer]);
    mLayer[idLayer]->createLayerWithDeadZones(dest);
  } else if (mVersion == "FourLayers") {
    mLayer[idLayer]->setFringeChipWidth(mFringeChipWidth[idLayer]);
    mLayer[idLayer]->setMiddleChipWidth(mMiddleChipWidth[idLayer]);
    mLayer[idLayer]->setNumSubSensorsHalfLayer(mNumSubSensorsHalfLayer[idLayer]);
    if (idLayer != 3) {
      mLayer[idLayer]->createLayerWithDeadZones(dest);
    } else if (idLayer == 3) {
      mLayer[idLayer]->setGapXDirection(mGapXDirection4thLayer);
      mLayer[idLayer]->create4thLayer(dest);
    }
  }

  return mLayer[idLayer]; // is this needed?
}

//________________________________________________________________
void DescriptorInnerBarrelITS3::createServices(TGeoVolume* dest)
{
  //
  // Creates the Inner Barrel Service structures
  //

  std::unique_ptr<ITS3Services> mServicesGeometry(new ITS3Services());
  TGeoVolume* cyss = mServicesGeometry.get()->createCYSSAssembly();
  dest->AddNode(cyss, 1, nullptr);
}
