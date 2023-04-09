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

#include "ITS3Base/DescriptorInnerBarrelITS3Param.h"
#include "ITS3Base/SegmentationSuperAlpide.h"
#include "ITS3Simulation/DescriptorInnerBarrelITS3.h"
#include "ITS3Simulation/ITS3Services.h"

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
  auto gapZ = param.mGapZ;
  auto gapPhi = param.mGapPhi;
  auto radii = param.mRadii;
  auto length = param.mLength;
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
  mChipTypeID.resize(mNumLayers);
  mGapZ.resize(mNumLayers);
  mGapPhi.resize(mNumLayers);
  mNumSubSensorsHalfLayer.resize(mNumLayers);
  mFringeChipWidth.resize(mNumLayers);
  mMiddleChipWidth.resize(mNumLayers);
  mHeightStripFoam.resize(mNumLayers);
  mLengthSemiCircleFoam.resize(mNumLayers);
  mThickGluedFoam.resize(mNumLayers);
  mBuildLevel.resize(mNumLayers);

  const double safety = 0.5;

  // radius, length, gap in z, gap in phi, num of chips in half layer, fringe chip width, middle chip width, strip foam height, semi-cicle foam length, guled foam width, gapx for 4th layer
  std::vector<std::array<double, 11>> IBtdr5dat;
  IBtdr5dat.emplace_back(std::array<double, 11>{radii[0], length, gapZ[0], gapPhi[0], 3.f, 0.06f, 0.128f, 0.25f, 0.8f, 0.022f, 0.f});
  IBtdr5dat.emplace_back(std::array<double, 11>{radii[1], length, gapZ[1], gapPhi[1], 4.f, 0.06f, 0.128f, 0.25f, 0.8f, 0.022f, 0.f});
  IBtdr5dat.emplace_back(std::array<double, 11>{radii[2], length, gapZ[2], gapPhi[2], 5.f, 0.06f, 0.128f, 0.25f, 0.8f, 0.022f, 0.f});
  IBtdr5dat.emplace_back(std::array<double, 11>{radii[3], length, gapZ[3], gapPhi[3], 5.f, 0.06f, 0.128f, 0.25f, 0.8f, 0.022f, 0.05f});

  if (mVersion == "ThreeLayersNoDeadZones") {

    mWrapperMinRadius = IBtdr5dat[0][0] - safety;

    for (auto idLayer{0u}; idLayer < mNumLayers; ++idLayer) {
      mLayerRadii[idLayer] = IBtdr5dat[idLayer][0];
      mLayerZLen[idLayer] = IBtdr5dat[idLayer][1];
      mGapZ[idLayer] = IBtdr5dat[idLayer][2];
      mGapPhi[idLayer] = IBtdr5dat[idLayer][3];
      mChipTypeID[idLayer] = 0;
      mHeightStripFoam[idLayer] = IBtdr5dat[idLayer][7];
      mLengthSemiCircleFoam[idLayer] = IBtdr5dat[idLayer][8];
      mThickGluedFoam[idLayer] = IBtdr5dat[idLayer][9];
      mBuildLevel[idLayer] = buildLevel;
      LOGP(info, "ITS3 L# {} R:{} Gap:{} StripFoamHeight:{} SemiCircleFoamLength:{} ThickGluedFoam:{}",
           idLayer, mLayerRadii[idLayer], mGapZ[idLayer],
           mHeightStripFoam[idLayer], mLengthSemiCircleFoam[idLayer], mThickGluedFoam[idLayer]);
    }
  } else if (mVersion == "ThreeLayers") {

    mWrapperMinRadius = IBtdr5dat[0][0] - safety;

    for (auto idLayer{0u}; idLayer < mNumLayers; ++idLayer) {
      mLayerRadii[idLayer] = IBtdr5dat[idLayer][0];
      mLayerZLen[idLayer] = IBtdr5dat[idLayer][1];
      mNumSubSensorsHalfLayer[idLayer] = (int)IBtdr5dat[idLayer][4];
      mFringeChipWidth[idLayer] = IBtdr5dat[idLayer][5];
      mMiddleChipWidth[idLayer] = IBtdr5dat[idLayer][6];
      mGapZ[idLayer] = IBtdr5dat[idLayer][2];
      mGapPhi[idLayer] = IBtdr5dat[idLayer][3];
      mChipTypeID[idLayer] = 0;
      mHeightStripFoam[idLayer] = IBtdr5dat[idLayer][7];
      mLengthSemiCircleFoam[idLayer] = IBtdr5dat[idLayer][8];
      mThickGluedFoam[idLayer] = IBtdr5dat[idLayer][9];
      mBuildLevel[idLayer] = buildLevel;
      LOGP(info, "ITS3 L# {} R:{} Gap:{} NSubSensors:{} FringeChipWidth:{} MiddleChipWidth:{} StripFoamHeight:{} SemiCircleFoamLength:{} ThickGluedFoam:{}",
           idLayer, mLayerRadii[idLayer], mGapZ[idLayer],
           mNumSubSensorsHalfLayer[idLayer], mFringeChipWidth[idLayer], mMiddleChipWidth[idLayer],
           mHeightStripFoam[idLayer], mLengthSemiCircleFoam[idLayer], mThickGluedFoam[idLayer]);
    }
  } else if (mVersion == "FourLayers") {

    mWrapperMinRadius = IBtdr5dat[0][0] - safety;

    for (auto idLayer{0u}; idLayer < mNumLayers; ++idLayer) {
      mLayerRadii[idLayer] = IBtdr5dat[idLayer][0];
      mLayerZLen[idLayer] = IBtdr5dat[idLayer][1];
      mNumSubSensorsHalfLayer[idLayer] = (int)IBtdr5dat[idLayer][4];
      mFringeChipWidth[idLayer] = IBtdr5dat[idLayer][5];
      mMiddleChipWidth[idLayer] = IBtdr5dat[idLayer][6];
      mGapZ[idLayer] = IBtdr5dat[idLayer][2];
      mGapPhi[idLayer] = IBtdr5dat[idLayer][3];
      mChipTypeID[idLayer] = 0;
      mHeightStripFoam[idLayer] = IBtdr5dat[idLayer][7];
      mLengthSemiCircleFoam[idLayer] = IBtdr5dat[idLayer][8];
      mThickGluedFoam[idLayer] = IBtdr5dat[idLayer][9];
      mBuildLevel[idLayer] = buildLevel;
      mGapXDirection4thLayer = IBtdr5dat[idLayer][10];
      LOGP(info, "ITS3 L# {} R:{} Gap:{} NSubSensors:{} FringeChipWidth:{} MiddleChipWidth:{} StripFoamHeight:{} SemiCircleFoamLength:{} ThickGluedFoam:{}",
           idLayer, mLayerRadii[idLayer], mGapZ[idLayer],
           mNumSubSensorsHalfLayer[idLayer], mFringeChipWidth[idLayer], mMiddleChipWidth[idLayer],
           mHeightStripFoam[idLayer], mLengthSemiCircleFoam[idLayer], mThickGluedFoam[idLayer]);
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
  mLayer[idLayer]->setLayerRadius(mLayerRadii[idLayer]);
  mLayer[idLayer]->setLayerZLen(mLayerZLen[idLayer]);
  mLayer[idLayer]->setGapBetweenEmispheres(mGapZ[idLayer]);
  mLayer[idLayer]->setGapBetweenEmispheresInPhi(mGapPhi[idLayer]);
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
