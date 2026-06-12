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
#include <IOTOFBase/IOTOFBaseParam.h>
#include <TGeoManager.h>
#include <TMath.h>

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

  std::string layName = lay == 0 ? GeometryTGeo::getITOFLayerPattern() : GeometryTGeo::getOTOFLayerPattern();
  TGeoVolume* layV = gGeoManager->GetVolume(layName.c_str());
  if (layV == nullptr) {
    LOG(fatal) << "Can't find volume " << layName;
    return -1;
  }

  TObjArray* nodes = layV->GetNodes();
  int nNodes = nodes->GetEntriesFast();

  for (int j{0}; j < nNodes; ++j) {
    if (strstr(nodes->At(j)->GetName(), lay == 0 ? GeometryTGeo::getITOFStavePattern() : GeometryTGeo::getOTOFStavePattern()) != nullptr) {
      numberOfStaves++;
    }
  }

  return numberOfStaves;
}

int GeometryTGeo::extractNumberOfModulesIOTOF(int lay) const
{
  int numberOfModules{0};

  std::string staveName = lay == 0 ? GeometryTGeo::getITOFStavePattern() : GeometryTGeo::getOTOFStavePattern();
  TGeoVolume* staveV = gGeoManager->GetVolume(staveName.c_str());
  if (staveV == nullptr) {
    LOG(fatal) << "Can't find volume " << staveName;
    return -1;
  }

  TObjArray* nodes = staveV->GetNodes();
  int nNodes = nodes->GetEntriesFast();

  for (int j{0}; j < nNodes; ++j) {
    if (strstr(nodes->At(j)->GetName(), lay == 0 ? GeometryTGeo::getITOFModulePattern() : GeometryTGeo::getOTOFModulePattern()) != nullptr) {
      numberOfModules++;
    }
  }

  return numberOfModules;
}

int GeometryTGeo::extractNumberOfChipsPerModuleIOTOF(int lay) const
{
  int numberOfChips{0};

  std::string moduleName = lay == 0 ? GeometryTGeo::getITOFModulePattern() : GeometryTGeo::getOTOFModulePattern();
  TGeoVolume* moduleV = gGeoManager->GetVolume(moduleName.c_str());
  if (moduleV == nullptr) {
    LOG(fatal) << "Can't find volume " << moduleName;
    return -1;
  }

  TObjArray* nodes = moduleV->GetNodes();
  int nNodes = nodes->GetEntriesFast();

  for (int j{0}; j < nNodes; ++j) {
    if (strstr(nodes->At(j)->GetName(), lay == 0 ? GeometryTGeo::getITOFChipPattern() : GeometryTGeo::getOTOFChipPattern()) != nullptr) {
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

int GeometryTGeo::getIOTOFFirstChipIndex(int lay) const
{
  return lay == 0 ? 0 : mLastChipIndex[0] + 1;
}

int GeometryTGeo::getIOTOFLayer(int index) const
{
  if (index < 0 || index > mLastChipIndex[1]) {
    LOG(fatal) << "Invalid chip index " << index;
    return -1;
  }
  return index > mLastChipIndex[0] ? 1 : 0;
}

int GeometryTGeo::getIOTOFChipIndex(int lay, int sta, int mod, int chip) const
{
  return getIOTOFFirstChipIndex(lay) + (sta - 1) * mNumberOfChipsPerStaveIOTOF[lay] + (mod - 1) * mNumberOfChipsPerModuleIOTOF[lay] + (chip - 1);
}

bool GeometryTGeo::getIOTOFChipId(int index, int& lay, int& sta, int& mod, int& chip) const
{
  lay = getIOTOFLayer(index);
  index -= getIOTOFFirstChipIndex(lay);
  sta = mNumberOfStavesIOTOF[lay] > 0 ? index / mNumberOfChipsPerStaveIOTOF[lay] : -1;
  index %= mNumberOfChipsPerStaveIOTOF[lay];
  mod = mNumberOfModulesIOTOF[lay] > 0 ? index / mNumberOfChipsPerModuleIOTOF[lay] : -1;
  chip = index % mNumberOfChipsPerModuleIOTOF[lay];
  return true;
}

TString GeometryTGeo::getMatrixPath(int index) const
{
  int lay, sta, mod, chip;
  getIOTOFChipId(index, lay, sta, mod, chip);

  TString path = Form("/cave_1/barrel_1/%s_2/", GeometryTGeo::getIOTOFVolPattern());
  sta += 1;
  mod += 1;
  chip += 1;

  if (lay == 0) {
    path += Form("%s_1/", GeometryTGeo::getITOFLayerPattern());
    if (mNumberOfStavesIOTOF[lay] > 0) {
      path += Form("%s_%d/", GeometryTGeo::getITOFStavePattern(), sta);
    }
    if (mNumberOfModulesIOTOF[lay] > 0) {
      path += Form("%s_%d/", GeometryTGeo::getITOFModulePattern(), mod);
    }
    if (mNumberOfChipsPerModuleIOTOF[lay] > 0) {
      path += Form("%s_%d/%s_1", GeometryTGeo::getITOFChipPattern(), chip, GeometryTGeo::getITOFSensorPattern());
    }
  } else {
    path += Form("%s_1/", GeometryTGeo::getOTOFLayerPattern());
    if (mNumberOfStavesIOTOF[lay] > 0) {
      path += Form("%s_%d/", GeometryTGeo::getOTOFStavePattern(), sta);
    }
    if (mNumberOfModulesIOTOF[lay] > 0) {
      path += Form("%s_%d/", GeometryTGeo::getOTOFModulePattern(), mod);
    }
    if (mNumberOfChipsPerModuleIOTOF[lay] > 0) {
      path += Form("%s_%d/%s_1", GeometryTGeo::getOTOFChipPattern(), chip, GeometryTGeo::getOTOFSensorPattern());
    }
  }

  return path;
}

TGeoHMatrix* GeometryTGeo::extractMatrixSensor(int index) const
{
  auto path = getMatrixPath(index);

  static TGeoHMatrix matTmp;
  gGeoManager->PushPath();

  if (!gGeoManager->cd(path.Data())) {
    gGeoManager->PopPath();
    LOG(error) << "Error in cd-ing to " << path.Data();
    return nullptr;
  }

  matTmp = *gGeoManager->GetCurrentMatrix();
  // LOG(info) << "Path = " << path.Data();

  // Restore the modeler state
  gGeoManager->PopPath();

  // account for the difference between physical sensitive layer (where charge collection is simulated) and effective sensor thicknesses
  // TODO: apply translation by the effective sensor thickness, not yet done (see ITS)

  return &matTmp;
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

  auto& iotofPars = IOTOFBaseParam::Instance();
  if (!iotofPars.segmentedInnerTOF && !iotofPars.segmentedOuterTOF) {
    return;
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

  int numberOfChips{0};
  for (int j{0}; j < 2; ++j) {
    mNumberOfChipsPerStaveIOTOF[j] = mNumberOfModulesIOTOF[j] * mNumberOfChipsPerModuleIOTOF[j];
    mNumberOfChipsIOTOF[j] = mNumberOfStavesIOTOF[j] * mNumberOfChipsPerStaveIOTOF[j];
    numberOfChips += mNumberOfChipsIOTOF[j];
    mLastChipIndex[j] = numberOfChips - 1;
  }

  LOG(info) << "numberOfChipsITOF = " << mNumberOfChipsIOTOF[0] << ", numberOfChipsOTOF = " << mNumberOfChipsIOTOF[1] << ", numberOfChips = " << numberOfChips << ", mNumberOfChipesPerStaveITOF" << mNumberOfChipsPerStaveIOTOF[0];

  setSize(numberOfChips);
  defineSensors();
  fillTrackingFramesCache();
  fillMatrixCache(loadTrans);
  //  fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::L2G));
}

void GeometryTGeo::defineSensors()
{
  for (int i = 0; i < mSize; i++) {
    sensors.push_back(i);
  }
}

void GeometryTGeo::fillTrackingFramesCache()
{
  // fill for every sensor of IOTOF its tracking frame parameters
  if (!isTrackingFrameCached() && !sensors.empty()) {
    size_t newSize = sensors.size();
    mCacheRefX.resize(newSize);
    mCacheRefAlpha.resize(newSize);
    for (int i = 0; i < newSize; i++) {
      int sensorId = sensors[i];
      extractSensorXAlpha(sensorId, mCacheRefX[i], mCacheRefAlpha[i]);
    }
  }
}

void GeometryTGeo::fillMatrixCache(int mask)
{
  if (mSize < 1) {
    LOG(warning) << "The method Build was not called yet";
    Build(mask);
    return;
  }

  LOG(debug) << "Filling matrix cache for " << getName() << " with mask " << mask;

  if ((mask & o2::math_utils::bit2Mask(o2::math_utils::TransformType::L2G)) && !getCacheL2G().isFilled()) {
    // Matrices for Local (Sensor!!! rather than the full chip) to Global frame transformation
    LOG(info) << "Loading " << getName() << " L2G matrices from TGeo; there are " << mSize << " matrices";
    auto& cacheL2G = getCacheL2G();
    cacheL2G.setSize(mSize);

    for (int i = 0; i < mSize; i++) {
      TGeoHMatrix* hm = extractMatrixSensor(i);
      cacheL2G.setMatrix(o2::math_utils::Transform3D(*hm), i);
    }
  }

  // build T2L matrices for IOTOF
  if ((mask & o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L)) && !getCacheT2L().isFilled()) {
    LOGP(info, "Loading {} T2L matrices from TGeo for IOTOF", getName());
    if (sensors.size()) {
      int m_Size = sensors.size();
      auto& cacheT2L = getCacheT2L();
      cacheT2L.setSize(m_Size);
      for (int i = 0; i < m_Size; i++) {
        int sensorID = sensors[i];
        TGeoHMatrix& hm = createT2LMatrix(sensorID);
        cacheT2L.setMatrix(Mat3D(hm), i);
      }
    }
  }
}

void GeometryTGeo::extractSensorXAlpha(int chipID, float& x, float& alp)
{
  double locA[3] = {-100., 0., 0.}, locB[3] = {100., 0., 0.}, gloA[3], gloB[3];
  double xp{0}, yp{0};

  const TGeoHMatrix* matL2G = extractMatrixSensor(chipID);
  matL2G->LocalToMaster(locA, gloA);
  matL2G->LocalToMaster(locB, gloB);
  double dx = gloB[0] - gloA[0], dy = gloB[1] - gloA[1];
  double t = (gloB[0] * dx + gloB[1] * dy) / (dx * dx + dy * dy);
  xp = gloB[0] - dx * t;
  yp = gloB[1] - dy * t;

  alp = std::atan2(yp, xp);
  x = std::hypot(xp, yp);
  o2::math_utils::bringTo02Pi(alp);
}

TGeoHMatrix& GeometryTGeo::createT2LMatrix(int chipID)
{
  static TGeoHMatrix t2l;
  t2l.Clear();
  float alpha = getSensorRefAlpha(chipID);
  t2l.RotateZ(alpha * TMath::RadToDeg());
  const TGeoHMatrix* matL2G = extractMatrixSensor(chipID);
  const TGeoHMatrix& matL2Gi = matL2G->Inverse();
  t2l.MultiplyLeft(&matL2Gi);
  return t2l;
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
