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

/// \file GeometryTGeo.cxx
/// \brief Implementation of the GeometryTGeo class
/// \author cvetan.cheshkov@cern.ch - 15/02/2007
/// \author ruben.shahoyan@cern.ch - adapted to ITSupg 18/07/2012
/// \author rafael.pezzi@cern.ch - adapted to ALICE 3 EndCaps 14/02/2021

// ATTENTION: In opposite to old AliITSgeomTGeo, all indices start from 0, not from 1!!!

#include "FT3Base/GeometryTGeo.h"
#include "DetectorsBase/GeometryManager.h"
#include "MathUtils/Cartesian.h"

#include <fairlogger/Logger.h> // for LOG

#include <TGeoBBox.h>         // for TGeoBBox
#include <TGeoManager.h>      // for gGeoManager, TGeoManager
#include <TGeoPhysicalNode.h> // for TGeoPNEntry, TGeoPhysicalNode
#include <TGeoShape.h>        // for TGeoShape
#include <TMath.h>            // for Nint, ATan2, RadToDeg
#include <TString.h>          // for TString, Form
#include "TClass.h"           // for TClass
#include "TGeoMatrix.h"       // for TGeoHMatrix
#include "TGeoNode.h"         // for TGeoNode, TGeoNodeMatrix
#include "TGeoVolume.h"       // for TGeoVolume
#include "TMathBase.h"        // for Max
#include "TObjArray.h"        // for TObjArray
#include "TObject.h"          // for TObject

#include <cctype> // for isdigit
#include <algorithm>
#include <cstdio>  // for snprintf, NULL, printf
#include <cstring> // for strstr, strlen
#include <map>

using namespace TMath;
using namespace o2::ft3;
using namespace o2::detectors;

ClassImp(o2::ft3::GeometryTGeo);

std::unique_ptr<o2::ft3::GeometryTGeo> GeometryTGeo::sInstance;

std::string GeometryTGeo::sVolumeName = "FT3V";          ///< Mother volume name
std::string GeometryTGeo::sInnerVolumeName = "FT3Inner"; ///< Mother inner volume name
std::string GeometryTGeo::sLayerName = "FT3Layer";       ///< Layer name
std::string GeometryTGeo::sChipName = "FT3Chip";         ///< Chip name
std::string GeometryTGeo::sSensorName = "FT3Sensor";     ///< Sensor name
std::string GeometryTGeo::sPassiveName = "FT3Passive";   ///< Passive material name

//__________________________________________________________________________
GeometryTGeo::GeometryTGeo(bool build, int loadTrans) : DetMatrixCache(DetID::FT3)
{
  // default c-tor, if build is true, the structures will be filled and the transform matrices
  // will be cached
  if (sInstance) {
    LOG(fatal) << "Invalid use of public constructor: o2::ft3::GeometryTGeo instance exists";
    // throw std::runtime_error("Invalid use of public constructor: o2::ft3::GeometryTGeo instance exists");
  }

  if (build) {
    Build(loadTrans);
  }
}

void GeometryTGeo::adopt(GeometryTGeo* raw)
{
  sInstance.reset(raw);
}

//__________________________________________________________________________
void GeometryTGeo::Build(int loadTrans)
{
  if (isBuilt()) {
    LOG(warning) << "Already built";
    return; // already initialized
  }

  if (!gGeoManager) {
    // RSTODO: in future there will be a method to load matrices from the CDB
    LOG(fatal) << "Geometry is not loaded";
  }

  mNumberOfLayersPerSide = 0;
  for (int dir = 0; dir < 2; ++dir) {
    while (gGeoManager->GetVolume(Form("%s%d_%d", getFT3LayerPattern(), dir, mNumberOfLayersPerSide))) {
      ++mNumberOfLayersPerSide;
    }
    if (mNumberOfLayersPerSide > 0) {
      break;
    }
  }

  std::map<int, std::map<int, int>> chipsPerStavePerLayer;
  auto* volumes = gGeoManager->GetListOfVolumes();
  const int nVolumes = volumes ? volumes->GetEntriesFast() : 0;
  for (int i = 0; i < nVolumes; ++i) {
    auto* volume = static_cast<TGeoVolume*>(volumes->At(i));
    if (!volume) {
      continue;
    }
    const std::string name = volume->GetName();
    if (name.find("FT3Sensor") != 0 || name.find("Inactive") != std::string::npos || name.find("inactive") != std::string::npos) {
      continue;
    }
    int layer = -1, stave = 0, chip = 0;
    extractChipIdsFT3(name, layer, stave, chip);
    if (layer < 0) {
      continue;
    }
    chipsPerStavePerLayer[layer][stave] = std::max(chipsPerStavePerLayer[layer][stave], chip + 1);
  }

  int numberOfChips = 0;
  int numberOfStaves = 0;
  mFirstChipIndexLayer.clear();
  mFirstStaveIndexLayer.clear();
  mFirstChipIndexStave.clear();
  mFirstChipIndexLayer.push_back(0);
  mFirstStaveIndexLayer.push_back(0);
  mFirstChipIndexStave.push_back(0);
  for (const auto& [layer, chipsPerStave] : chipsPerStavePerLayer) {
    for (const auto& [stave, nChips] : chipsPerStave) {
      numberOfChips += nChips;
      mFirstChipIndexStave.push_back(numberOfChips);
      ++numberOfStaves;
    }
    mFirstStaveIndexLayer.push_back(numberOfStaves);
    mFirstChipIndexLayer.push_back(numberOfChips);
  }

  setSize(numberOfChips);
  fillMatrixCache(loadTrans);
}

//__________________________________________________________________________
const char* GeometryTGeo::composeSymNameLayer(Int_t d, Int_t lr)
{
  return Form("%s/%s%d", composeSymNameFT3(d), getFT3LayerPattern(), lr);
}

//__________________________________________________________________________
const char* GeometryTGeo::composeSymNameChip(Int_t d, Int_t lr)
{
  return Form("%s/%s%d", composeSymNameLayer(d, lr), getFT3ChipPattern(), lr);
}

//__________________________________________________________________________
const char* GeometryTGeo::composeSymNameSensor(Int_t d, Int_t lr)
{
  return Form("%s/%s%d", composeSymNameChip(d, lr), getFT3SensorPattern(), lr);
}

//__________________________________________________________________________
void GeometryTGeo::fillMatrixCache(int mask)
{
  if (mSize < 1) {
    LOG(warning) << "The method Build was not called yet";
    Build(mask);
    return;
  }
  if ((mask & o2::math_utils::bit2Mask(o2::math_utils::TransformType::L2G)) && !getCacheL2G().isFilled()) {
    LOGP(info, "Loading {} L2G matrix cache with {} FT3 sensor slots", getName(), mSize);
    getCacheL2G().setSize(mSize);
  }
}

//__________________________________________________________________________
void GeometryTGeo::Print(Option_t*) const
{
  if (!isBuilt()) {
    LOGP(info, "{} geometry is not built yet", getName());
    return;
  }
  LOGP(info, "Summary of GeometryTGeo: {}", getName());
  LOGP(info, "Number of FT3 layers: {}", getNumberOfLayers());
  LOGP(info, "Number of FT3 layers per side: {}", mNumberOfLayersPerSide);
  LOGP(info, "Number of FT3 chips: {}", getNumberOfChips());
  for (int layer = 0; layer < getNumberOfLayers(); ++layer) {
    const int nStaves = mFirstStaveIndexLayer[layer + 1] - mFirstStaveIndexLayer[layer];
    const int nChips = mFirstChipIndexLayer[layer + 1] - mFirstChipIndexLayer[layer];
    LOGP(info, "Layer {}: {} staves, {} chips", layer, nStaves, nChips);
  }
}

int GeometryTGeo::getLayer(int index) const
{
  for (int layer = 0; layer + 1 < mFirstChipIndexLayer.size(); ++layer) {
    if (index >= mFirstChipIndexLayer[layer] && index < mFirstChipIndexLayer[layer + 1]) {
      return layer;
    }
  }
  return -1;
}

int GeometryTGeo::getStave(int index) const
{
  const int layer = getLayer(index);
  if (layer < 0) {
    return -1;
  }
  for (int stave = mFirstStaveIndexLayer[layer]; stave < mFirstStaveIndexLayer[layer + 1]; ++stave) {
    if (index >= mFirstChipIndexStave[stave] && index < mFirstChipIndexStave[stave + 1]) {
      return stave - mFirstStaveIndexLayer[layer];
    }
  }
  return -1;
}

int GeometryTGeo::getChip(int index) const
{
  const int layer = getLayer(index);
  const int stave = getStave(index);
  if (layer < 0 || stave < 0) {
    return -1;
  }
  return index - mFirstChipIndexStave[mFirstStaveIndexLayer[layer] + stave];
}

int GeometryTGeo::getChipIndex(int layer, int stave, int chip) const
{
  if (layer < 0 || layer + 1 >= mFirstStaveIndexLayer.size()) {
    return -1;
  }
  const int absStave = mFirstStaveIndexLayer[layer] + stave;
  if (absStave < mFirstStaveIndexLayer[layer] || absStave >= mFirstStaveIndexLayer[layer + 1]) {
    return -1;
  }
  const int chipIndex = mFirstChipIndexStave[absStave] + chip;
  return chipIndex < mFirstChipIndexStave[absStave + 1] ? chipIndex : -1;
}

int GeometryTGeo::getChipIndex(const std::string& volName) const
{
  int layer = -1, stave = 0, chip = 0;
  extractChipIdsFT3(volName, layer, stave, chip);
  return getChipIndex(layer, stave, chip);
}

void GeometryTGeo::extractChipIdsFT3(std::string const& volName, int& layer, int& stave, int& chip) const
{
  layer = -1;
  stave = 0;
  chip = 0;
  if (volName.find("FT3Sensor_Active") == 0) {
    int idx = volName.find('_') + 1;
    idx = volName.find('_', idx) + 1;
    const int direction = std::stoi(volName.substr(idx));
    idx = volName.find('_', idx) + 1;
    layer = std::stoi(volName.substr(idx));
    idx = volName.find('_', idx) + 1;
    stave = std::stoi(volName.substr(idx));
    idx = volName.find('_', idx) + 1;
    chip = std::stoi(volName.substr(idx));
    if (direction == 1) {
      layer += mNumberOfLayersPerSide;
    }
    return;
  }
  if (volName.find("FT3Sensor_front_") == 0 || volName.find("FT3Sensor_back_") == 0) {
    int idx = volName.find('_') + 1;
    idx = volName.find('_', idx) + 1;
    layer = std::stoi(volName.substr(idx));
    idx = volName.find('_', idx) + 1;
    const int direction = std::stoi(volName.substr(idx));
    idx = volName.find('_', idx) + 1;
    chip = std::stoi(volName.substr(idx));
    if (direction == 1) {
      layer += mNumberOfLayersPerSide;
    }
    return;
  }
  if (volName.find("FT3Sensor_") == 0) {
    int idx = std::string("FT3Sensor_").size();
    const int direction = std::stoi(volName.substr(idx));
    idx = volName.find('_', idx) + 1;
    layer = std::stoi(volName.substr(idx));
    if (direction == 1) {
      layer += mNumberOfLayersPerSide;
    }
  }
}
