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

using namespace o2::its;
using namespace o2::its3;

/// \cond CLASSIMP
ClassImp(DescriptorInnerBarrelITS3);
/// \endcond

//________________________________________________________________
DescriptorInnerBarrelITS3::DescriptorInnerBarrelITS3(int nlayers) : DescriptorInnerBarrel(nlayers)
{
  //
  // Standard constructor
  //

  fSensorLayerThickness = 30.e-4;
}

//________________________________________________________________
void DescriptorInnerBarrelITS3::ConfigureITS3()
{
  // build ITS3 upgrade detector
  fLayerRadii.resize(fNumLayers);
  fLayerZLen.resize(fNumLayers);
  fDetectorThickness.resize(fNumLayers);
  fChipTypeID.resize(fNumLayers);
  fBuildLevel.resize(fNumLayers);

  std::vector<std::array<double, 2>> IBtdr5dat; // 18 24
  IBtdr5dat.emplace_back(std::array<double, 2>{1.8f, 27.15});
  IBtdr5dat.emplace_back(std::array<double, 2>{2.4f, 27.15});
  IBtdr5dat.emplace_back(std::array<double, 2>{3.0f, 27.15});
  if (fNumLayers == 4)
    IBtdr5dat.emplace_back(std::array<double, 2>{7.0f, 27.15});

  const double safety = 0.5;
  fWrapperMinRadius = IBtdr5dat[0][0] - safety;
  fWrapperMaxRadius = IBtdr5dat[fNumLayers - 1][0] + safety;
  fWrapperZSpan = 70.;

  for (auto idLayer{0u}; idLayer < IBtdr5dat.size(); ++idLayer) {
    fLayerRadii[idLayer] = IBtdr5dat[idLayer][0];
    fLayerZLen[idLayer] = IBtdr5dat[idLayer][1];
    fDetectorThickness[idLayer] = fSensorLayerThickness;
    fChipTypeID[idLayer] = 0;
    fBuildLevel[idLayer] = 0;
  }
}

//________________________________________________________________
void DescriptorInnerBarrelITS3::GetConfigurationLayers(std::vector<double>& radii, std::vector<double>& zlen, std::vector<double>& thickness, std::vector<int>& chipID, std::vector<int>& buildlev)
{
  radii = fLayerRadii;
  zlen = fLayerZLen;
  thickness = fDetectorThickness;
  chipID = fChipTypeID;
  buildlev = fBuildLevel;
}
