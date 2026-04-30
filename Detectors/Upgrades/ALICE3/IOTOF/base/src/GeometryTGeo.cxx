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

#include <IOTOFBase/GeometryTGeo.h>
#include <TGeoManager.h>

namespace o2
{
namespace iotof
{
std::unique_ptr<o2::iotof::GeometryTGeo> GeometryTGeo::sInstance;

// Common i/oTOF
std::string GeometryTGeo::sIOTOFVolumeName = "IOTOFV";

// Inner TOF
std::string GeometryTGeo::sITOFLayerName = "ITOFLayer";
std::string GeometryTGeo::sITOFStaveName = "ITOFStave";
std::string GeometryTGeo::sITOFModuleName = "ITOFModule";
std::string GeometryTGeo::sITOFChipName = "ITOFChip";
std::string GeometryTGeo::sITOFSensorName = "ITOFSensor";

// Outer TOF
std::string GeometryTGeo::sOTOFLayerName = "OTOFLayer";
std::string GeometryTGeo::sOTOFStaveName = "OTOFStave";
std::string GeometryTGeo::sOTOFModuleName = "OTOFModule";
std::string GeometryTGeo::sOTOFChipName = "OTOFChip";
std::string GeometryTGeo::sOTOFSensorName = "OTOFSensor";

// Forward TOF
std::string GeometryTGeo::sFTOFLayerName = "FTOFLayer";
std::string GeometryTGeo::sFTOFChipName = "FTOFChip";
std::string GeometryTGeo::sFTOFSensorName = "FTOFSensor";

// Backward TOF
std::string GeometryTGeo::sBTOFLayerName = "BTOFLayer";
std::string GeometryTGeo::sBTOFChipName = "BTOFChip";
std::string GeometryTGeo::sBTOFSensorName = "BTOFSensor";

GeometryTGeo::GeometryTGeo(bool build, int loadTrans) : DetMatrixCache()
{
  if (sInstance) {
    LOGP(fatal, "Invalid use of public constructor: o2::iotof::GeometryTGeo instance exists");
  }
  if (build) {
    Build(loadTrans);
  }
}

int GeometryTGeo::extractNumberOfStavesIOTOF(int lay) const
{
  int numberOfStaves{0};

  std::string layName = lay == 0 ? getITOFLayerPattern() : getOTOFLayerPattern();
  TGeoVolume* layV = gGeoManager->GetVolume(layName.c_str());

  // LOG(info) << "lay name = " << layV->GetName();

  TObjArray* nodes = layV->GetNodes();
  int nNodes = nodes->GetEntriesFast();

  for (int j{0}; j < nNodes; ++j) {
    if (strstr(nodes->At(j)->GetName(), lay == 0 ? getITOFStavePattern() : getOTOFStavePattern()) != nullptr) {
      numberOfStaves++;
    }
  }

  return numberOfStaves;
}

int GeometryTGeo::extractNumberOfModulesIOTOF(int lay) const
{
  int numberOfModules{0};

  std::string staveName = lay == 0 ? getITOFStavePattern() : getOTOFStavePattern();
  TGeoVolume* staveV = gGeoManager->GetVolume(staveName.c_str());

  TObjArray* nodes = staveV->GetNodes();
  int nNodes = nodes->GetEntriesFast();

  for (int j{0}; j < nNodes; ++j) {
    if (strstr(nodes->At(j)->GetName(), lay == 0 ? getITOFModulePattern() : getOTOFModulePattern()) != nullptr) {
      numberOfModules++;
    }
  }

  return numberOfModules;
}

int GeometryTGeo::extractNumberOfChipsPerModuleIOTOF(int lay) const
{
  int numberOfChips{0};

  std::string moduleName = lay == 0 ? getITOFModulePattern() : getOTOFModulePattern();
  TGeoVolume* moduleV = gGeoManager->GetVolume(moduleName.c_str());

  TObjArray* nodes = moduleV->GetNodes();
  int nNodes = nodes->GetEntriesFast();

  for (int j{0}; j < nNodes; ++j) {
    if (strstr(nodes->At(j)->GetName(), lay == 0 ? getITOFChipPattern() : getOTOFChipPattern()) != nullptr) {
      numberOfChips++;
    }
  }

  return numberOfChips;
}

int GeometryTGeo::extractNumberOfChipsFTOF() const
{
  return 0;
}

int GeometryTGeo::extractNumberOfChipsBTOF() const
{
  return 0;
}

void GeometryTGeo::Build(int loadTrans)
{
  if (isBuilt()) {
    LOGP(warning, "Already built");
    return; // already initialized
  }

  if (!gGeoManager) {
    LOGP(fatal, "Geometry is not loaded");
  }

  // Inner/outer TOF
  for (int j{0}; j < 2; ++j) {
    mNumberOfStavesIOTOF[j] = extractNumberOfStavesIOTOF(j);
    mNumberOfModulesIOTOF[j] = extractNumberOfModulesIOTOF(j);
    mNumberOfChipsPerModuleIOTOF[j] = extractNumberOfChipsPerModuleIOTOF(j);
  }

  // Forward TOF
  mNumberOfChipsFTOF = extractNumberOfChipsFTOF();

  // Backward TOF
  mNumberOfChipsBTOF = extractNumberOfChipsBTOF();

  // LOG(info) << "stavesITOF = " << mNumberOfStavesITOF << ", stavesOTOF = " << mNumberOfStavesOTOF;
  // LOG(info) << "modulesITOF = " << mNumberOfModulesITOF << ", modulesOTOF = " << mNumberOfModulesOTOF;
  // LOG(info) << "chipsITOF = " << mNumberOfChipsITOF << ", chipsOTOF = " << mNumberOfChipsOTOF;

  int numberOfChips{0};
  for (int j{0}; j < 2; ++j) {
    mNumberOfChipsIOTOF[j] = mNumberOfStavesIOTOF[j] * mNumberOfModulesIOTOF[j] * mNumberOfChipsPerModuleIOTOF[j];
    numberOfChips += mNumberOfChipsIOTOF[j];
    mLastChipIndex[j] = numberOfChips - 1;
  }

  // LOG(info) << "numberOfChipsITOF = " << mNumberOfChipsIOTOF[0] << ", numberOfChipsOTOF = " << mNumberOfChipsIOTOF[1] << ", numberOfChips = " << numberOfChips;

  setSize(numberOfChips);
  fillMatrixCache(loadTrans);
}

void GeometryTGeo::fillMatrixCache(int mask)
{
}

GeometryTGeo* GeometryTGeo::Instance()
{
  if (!sInstance) {
    sInstance = std::unique_ptr<GeometryTGeo>(new GeometryTGeo(true, 0));
  }
  return sInstance.get();
}

} // namespace iotof
} // namespace o2
