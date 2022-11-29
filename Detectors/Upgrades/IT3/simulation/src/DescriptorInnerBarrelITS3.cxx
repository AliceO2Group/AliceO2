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
#include "ITS3Base/SegmentationSuperAlpide.h"

using namespace o2::its;
using namespace o2::its3;

/// \cond CLASSIMP
ClassImp(DescriptorInnerBarrelITS3);
/// \endcond

//________________________________________________________________
DescriptorInnerBarrelITS3::DescriptorInnerBarrelITS3(Version version) : DescriptorInnerBarrel()
{
  //
  // Standard constructor
  //
  switch (version) {
    case ThreeLayersNoDeadZones: {
      mNumLayers = 3;
      break;
    }
    case ThreeLayers: {
      mNumLayers = 3;
      break;
    }
    case FourLayers: {
      mNumLayers = 4;
      break;
    }
    case FiveLayers: {
      mNumLayers = 5;
      break;
    }
  }

  mSensorLayerThickness = SegmentationSuperAlpide::SensorLayerThickness;
}

//________________________________________________________________
void DescriptorInnerBarrelITS3::configure()
{
  // build ITS3 upgrade detector
  mLayer.resize(mNumLayers);
  mLayerRadii.resize(mNumLayers);
  mLayerZLen.resize(mNumLayers);
  mDetectorThickness.resize(mNumLayers);
  mChipTypeID.resize(mNumLayers);
  mGap.resize(mNumLayers);

  const double safety = 0.5;

  std::vector<std::array<double, 3>> IBtdr5dat; // radius, length, gap
  IBtdr5dat.emplace_back(std::array<double, 3>{1.8f, 27.15, 0.1});
  IBtdr5dat.emplace_back(std::array<double, 3>{2.4f, 27.15, 0.1});
  IBtdr5dat.emplace_back(std::array<double, 3>{3.0f, 27.15, 0.1});

  switch (mVersion) {
    case ThreeLayersNoDeadZones: {

      mWrapperMinRadius = IBtdr5dat[0][0] - safety;
      mWrapperMaxRadius = IBtdr5dat[mNumLayers - 1][0] + safety;

      for (auto idLayer{0u}; idLayer < IBtdr5dat.size(); ++idLayer) {
        mLayerRadii[idLayer] = IBtdr5dat[idLayer][0];
        mLayerZLen[idLayer] = IBtdr5dat[idLayer][1];
        mDetectorThickness[idLayer] = mSensorLayerThickness;
        mGap[idLayer] = 0.1;
        mChipTypeID[idLayer] = 0;
        LOGP(info, "ITS3 L# {} R:{} Dthick:{} Gap:{} ", idLayer, mLayerRadii[idLayer], mDetectorThickness[idLayer], mGap[idLayer]);
      }
      break;
    }
    case ThreeLayers: {
      LOGP(fatal, "ITS3 version ThreeLayers not yet implemented.");
      break;
    }
    case FourLayers: {
      LOGP(fatal, "ITS3 version FourLayers not yet implemented.");
      break;
    }
    case FiveLayers: {
      LOGP(fatal, "ITS3 version FourLayers not yet implemented.");
      break;
    }
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
  mLayer[idLayer]->setSensorThick(mDetectorThickness[idLayer]);
  mLayer[idLayer]->setLayerRadius(mLayerRadii[idLayer]);
  mLayer[idLayer]->setLayerZLen(mLayerZLen[idLayer]);
  mLayer[idLayer]->setGapBetweenEmispheres(mGap[idLayer]);
  mLayer[idLayer]->setChipID(mChipTypeID[idLayer]);
  mLayer[idLayer]->createLayer(dest);

  return mLayer[idLayer]; // is this needed?
}
