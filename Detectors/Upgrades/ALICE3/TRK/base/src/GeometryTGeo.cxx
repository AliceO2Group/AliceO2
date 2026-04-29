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

#include <TRKBase/GeometryTGeo.h>
#include <TGeoManager.h>
#include "TRKBase/SegmentationChip.h"
#include "TRKBase/Specs.h"
#include <TMath.h>

#include <limits>

using Segmentation = o2::trk::SegmentationChip;

namespace o2
{
namespace trk
{
std::unique_ptr<o2::trk::GeometryTGeo> GeometryTGeo::sInstance;

// Names
std::string GeometryTGeo::sVolumeName = "TRKV";
std::string GeometryTGeo::sServiceVolName = "TRKService";
std::string GeometryTGeo::sLayerName = "TRKLayer";
std::string GeometryTGeo::sPetalAssemblyName = "PETAL";
std::string GeometryTGeo::sPetalName = "PETALCASE";
std::string GeometryTGeo::sPetalDiskName = "DISK";
std::string GeometryTGeo::sPetalLayerName = "LAYER";
std::string GeometryTGeo::sStaveName = "TRKStave";
std::string GeometryTGeo::sHalfStaveName = "TRKHalfStave";
std::string GeometryTGeo::sModuleName = "TRKModule";
std::string GeometryTGeo::sChipName = "TRKChip";
std::string GeometryTGeo::sSensorName = "TRKSensor";
std::string GeometryTGeo::sDeadzoneName = "TRKDeadzone";
std::string GeometryTGeo::sMetalStackName = "TRKMetalStack";

std::string GeometryTGeo::sWrapperVolumeName = "TRKUWrapVol"; ///< Wrapper volume name, not implemented at the moment

o2::trk::GeometryTGeo::~GeometryTGeo()
{
  if (!mOwner) {
    mOwner = true;
    sInstance.release();
  }
}
GeometryTGeo::GeometryTGeo(bool build, int loadTrans) : DetMatrixCache(detectors::DetID::TRK)
{
  if (sInstance) {
    LOGP(fatal, "Invalid use of public constructor: o2::trk::GeometryTGeo instance exists");
  }
  mLayerToWrapper.fill(-1);
  if (build) {
    Build(loadTrans);
  }
}

//__________________________________________________________________________
void GeometryTGeo::Build(int loadTrans)
{
  ///// current geometry organization:
  ///// total elements = x staves (*2 half staves if staggered geometry) * ML+OT layers + 4 petal cases * (3 layers + 6 disks)
  ///// indexing from 0 to 35: VD petals -> layers -> disks
  ///// indexing from 36 to y: MLOT staves

  if (isBuilt()) {
    LOGP(warning, "Already built");
    return; // already initialized
  }

  if (gGeoManager == nullptr) {
    LOGP(fatal, "Geometry is not loaded");
  }

  mLayoutMLOT = o2::trk::TRKBaseParam::Instance().getLayoutMLOT();

  LOG(debug) << "Overall layout ML and OT: " << mLayoutMLOT;

  mNumberOfLayersMLOT = extractNumberOfLayersMLOT();
  mNumberOfPetalsVD = extractNumberOfPetalsVD();
  mNumberOfActivePartsVD = extractNumberOfActivePartsVD();
  mNumberOfLayersVD = extractNumberOfLayersVD();
  mNumberOfDisksVD = extractNumberOfDisksVD();

  mNumberOfStaves.resize(mNumberOfLayersMLOT);
  mNumberOfHalfStaves.resize(mNumberOfLayersMLOT);
  mNumberOfModules.resize(mNumberOfLayersMLOT);
  mNumberOfChips.resize(mNumberOfLayersMLOT);

  mNumberOfChipsPerLayerVD.resize(mNumberOfLayersVD);
  mNumberOfChipsPerLayerMLOT.resize(mNumberOfLayersMLOT);
  mNumbersOfChipPerDiskVD.resize(mNumberOfDisksVD);
  mNumberOfChipsPerPetalVD.resize(mNumberOfPetalsVD);

  mLastChipIndex.resize(mNumberOfPetalsVD + mNumberOfLayersMLOT);
  mLastChipIndexVD.resize(mNumberOfPetalsVD);
  mLastChipIndexMLOT.resize(mNumberOfLayersMLOT); /// ML and OT are part of TRK as the same detector, without disks

  for (int i = 0; i < mNumberOfLayersMLOT; i++) {
    if (mLayoutMLOT == eMLOTLayout::kCylindrical) {
      mNumberOfStaves[i] = 1;
      mNumberOfHalfStaves[i] = 1;
      mNumberOfModules[i] = 1;
      mNumberOfChips[i] = 1;
    } else {
      mNumberOfStaves[i] = extractNumberOfStavesMLOT(i);
      mNumberOfHalfStaves[i] = extractNumberOfHalfStavesMLOT(i);
      mNumberOfModules[i] = extractNumberOfModulesMLOT(i);
      mNumberOfChips[i] = extractNumberOfChipsMLOT(i);
    }
  }

  int numberOfChipsTotal = 0;

  /// filling the information for the VD
  for (int i = 0; i < mNumberOfPetalsVD; i++) {
    mNumberOfChipsPerPetalVD[i] = extractNumberOfChipsPerPetalVD();
    numberOfChipsTotal += mNumberOfChipsPerPetalVD[i];
    mLastChipIndex[i] = numberOfChipsTotal - 1;
    mLastChipIndexVD[i] = numberOfChipsTotal - 1;
  }

  /// filling the information for the MLOT
  for (int i = 0; i < mNumberOfLayersMLOT; i++) {
    mNumberOfChipsPerLayerMLOT[i] = mNumberOfStaves[i] * mNumberOfHalfStaves[i] * mNumberOfModules[i] * mNumberOfChips[i];
    numberOfChipsTotal += mNumberOfChipsPerLayerMLOT[i];
    mLastChipIndex[i + mNumberOfPetalsVD] = numberOfChipsTotal - 1;
    mLastChipIndexMLOT[i] = numberOfChipsTotal - 1;
  }

  setSize(numberOfChipsTotal);
  defineMLOTSensors();
  fillTrackingFramesCacheMLOT();
  fillMatrixCache(loadTrans);
}

//__________________________________________________________________________
int GeometryTGeo::getSubDetID(int index) const
{
  if (index <= mLastChipIndexVD[mLastChipIndexVD.size() - 1]) {
    return 0;
  } else if (index > mLastChipIndexVD[mLastChipIndexVD.size() - 1]) {
    return 1;
  }
  return -1; /// not found
}

//__________________________________________________________________________
int GeometryTGeo::getPetalCase(int index) const
{
  int petalcase = 0;

  int subDetID = getSubDetID(index);
  if (subDetID == 1) {
    return -1;
  } else if (index <= mLastChipIndexVD[mNumberOfPetalsVD - 1]) {
    while (index > mLastChipIndexVD[petalcase]) {
      petalcase++;
    }
  }
  return petalcase;
}

//__________________________________________________________________________
int GeometryTGeo::getDisk(int index) const
{
  int subDetID = getSubDetID(index);
  int petalcase = getPetalCase(index);

  if (subDetID == 0) { /// VD
    if (index % mNumberOfChipsPerPetalVD[petalcase] < mNumberOfLayersVD) {
      return -1; /// layers
    }
    return (index % mNumberOfChipsPerPetalVD[petalcase]) - mNumberOfLayersVD;
  }

  return -1; /// not found or ML/OT
}

//__________________________________________________________________________
int GeometryTGeo::getLayer(int index) const
{
  int subDetID = getSubDetID(index);
  int petalcase = getPetalCase(index);
  int lay = 0;

  if (subDetID == 0) { /// VD
    if (index % mNumberOfChipsPerPetalVD[petalcase] >= mNumberOfLayersVD) {
      return -1; /// disks
    }
    return index % mNumberOfChipsPerPetalVD[petalcase];
  } else if (subDetID == 1) { /// MLOT
    while (index > mLastChipIndex[lay]) {
      lay++;
    }
    return lay - mNumberOfPetalsVD; /// numeration of MLOT layers starting from 0
  }
  return -1; /// -1 if not found
}
//__________________________________________________________________________
int GeometryTGeo::getLayerTRK(int index) const
{
  if (getDisk(index) != -1) {
    return -1; /// disks do not have a global layer index
  }
  int subDetID = getSubDetID(index);
  return subDetID * o2::trk::constants::VD::petal::nLayers + getLayer(index); // MLOT: offset by number of VD layers
}
//__________________________________________________________________________
int GeometryTGeo::getStave(int index) const
{
  int subDetID = getSubDetID(index);
  int lay = getLayer(index);
  int petalcase = getPetalCase(index);

  if (subDetID == 0) { /// VD
    return -1;
  } else if (subDetID == 1) { /// MLOT
    int lay = getLayer(index);
    index -= getFirstChipIndex(lay, petalcase, subDetID); // get the index of the sensing element in the layer

    const int Nhs = mNumberOfHalfStaves[lay];
    const int Nmod = mNumberOfModules[lay];
    const int Nchip = mNumberOfChips[lay];

    if (Nhs == 2) {
      int chipsPerModule = Nchip;
      int chipsPerHalfStave = Nmod * chipsPerModule;
      int chipsPerStave = Nhs * chipsPerHalfStave;
      return index / chipsPerStave;
    } else if (Nhs == 1) {
      int chipsPerModule = Nchip;
      int chipsPerStave = Nmod * chipsPerModule;
      return index / chipsPerStave;
    }
  }
  return -1;
}

//__________________________________________________________________________
int GeometryTGeo::getHalfStave(int index) const
{
  int subDetID = getSubDetID(index);
  int lay = getLayer(index);
  int petalcase = getPetalCase(index);

  if (subDetID == 0) { /// VD
    return -1;
  } else if (subDetID == 1) { /// MLOT
    int lay = getLayer(index);
    index -= getFirstChipIndex(lay, petalcase, subDetID); // get the index of the sensing element in the layer

    const int Nhs = mNumberOfHalfStaves[lay];
    const int Nmod = mNumberOfModules[lay];
    const int Nchip = mNumberOfChips[lay];

    int chipsPerModule = Nchip;
    int chipsPerHalfStave = Nmod * chipsPerModule;
    int chipsPerStave = Nhs * chipsPerHalfStave;

    int rem = index % chipsPerStave;
    return rem / chipsPerHalfStave; // 0 = left, 1 = right
  }
  return -1;
}

//__________________________________________________________________________
int GeometryTGeo::getModule(int index) const
{
  int subDetID = getSubDetID(index);
  int lay = getLayer(index);
  int petalcase = getPetalCase(index);

  if (subDetID == 0) { /// VD
    return -1;
  } else if (subDetID == 1) { /// MLOT
    int lay = getLayer(index);
    index -= getFirstChipIndex(lay, petalcase, subDetID); // get the index of the sensing element in the layer

    const int Nhs = mNumberOfHalfStaves[lay];
    const int Nmod = mNumberOfModules[lay];
    const int Nchip = mNumberOfChips[lay];

    if (Nhs == 2) {
      int chipsPerModule = Nchip;
      int chipsPerHalfStave = Nmod * chipsPerModule;
      int rem = index % (Nhs * chipsPerHalfStave);
      rem = rem % chipsPerHalfStave;
      return rem / chipsPerModule;
    } else if (Nhs == 1) {
      int chipsPerModule = Nchip;
      int rem = index % (Nmod * chipsPerModule);
      return rem / chipsPerModule;
    }
  }
  return -1;
}

//__________________________________________________________________________
int GeometryTGeo::getChip(int index) const
{
  int subDetID = getSubDetID(index);
  int lay = getLayer(index);
  int petalcase = getPetalCase(index);

  if (subDetID == 0) { /// VD
    return -1;
  } else if (subDetID == 1) { /// MLOT
    int lay = getLayer(index);
    index -= getFirstChipIndex(lay, petalcase, subDetID); // get the index of the sensing element in the layer

    const int Nhs = mNumberOfHalfStaves[lay];
    const int Nmod = mNumberOfModules[lay];
    const int Nchip = mNumberOfChips[lay];

    if (Nhs == 2) {
      int chipsPerModule = Nchip;
      return index % chipsPerModule;
    } else if (Nhs == 1) {
      int chipsPerModule = Nchip;
      return index % chipsPerModule;
    }
  }
  return -1;
}

//__________________________________________________________________________
unsigned short GeometryTGeo::getChipIndex(int subDetID, int petalcase, int disk, int lay, int stave, int halfstave, int mod, int chip) const
{
  if (subDetID == 0) { // VD
    if (lay == -1) {   // disk
      return getFirstChipIndex(lay, petalcase, subDetID) + mNumberOfLayersVD + disk;
    } else { // layer
      return getFirstChipIndex(lay, petalcase, subDetID) + lay;
    }
  } else if (subDetID == 1) {                 // MLOT
    const int Nhs = mNumberOfHalfStaves[lay]; // 1 or 2
    const int Nmod = mNumberOfModules[lay];   // module per half-stave (per stave if Nhs==1)
    const int Nchip = mNumberOfChips[lay];    // chips per module

    if (Nhs == 2) { // staggered geometry: layer -> stave -> halfstave -> mod -> chip
      int chipsPerModule = Nchip;
      int chipsPerHalfStave = Nmod * chipsPerModule;
      int chipsPerStave = Nhs * chipsPerHalfStave;
      return getFirstChipIndex(lay, petalcase, subDetID) + stave * chipsPerStave + halfstave * chipsPerHalfStave + mod * chipsPerModule + chip;
    } else if (Nhs == 1) { // turbo geometry: layer -> stave -> mod -> chip (no halfstave)
      int chipsPerModule = Nchip;
      int chipsPerStave = Nmod * chipsPerModule;
      return getFirstChipIndex(lay, petalcase, subDetID) + stave * chipsPerStave + mod * chipsPerModule + chip;
    }
  }

  LOGP(warning, "Chip index not found for subDetID %d, petalcase %d, disk %d, layer %d, stave %d, halfstave %d, module %d, chip %d, returning numeric limit", subDetID, petalcase, disk, lay, stave, halfstave, mod, chip);
  return std::numeric_limits<unsigned short>::max(); // not found
}

//__________________________________________________________________________
unsigned short GeometryTGeo::getChipIndex(int subDetID, int volume, int lay, int stave, int halfstave, int mod, int chip) const
{
  if (subDetID == 0) { // VD
    return volume;     /// In the current configuration for VD, each volume is the sensor element = chip. // TODO: when the geometry naming scheme will be changed, change this method

  } else if (subDetID == 1) {                 // MLOT
    const int Nhs = mNumberOfHalfStaves[lay]; // 1 or 2
    const int Nmod = mNumberOfModules[lay];   // module per half-stave (per stave if Nhs==1)
    const int Nchip = mNumberOfChips[lay];    // chips per module

    if (Nhs == 2) { // staggered geometry: layer -> stave -> halfstave -> mod -> chip
      int chipsPerModule = Nchip;
      int chipsPerHalfStave = Nmod * chipsPerModule;
      int chipsPerStave = Nhs * chipsPerHalfStave;
      return getFirstChipIndex(lay, -1, subDetID) + stave * chipsPerStave + halfstave * chipsPerHalfStave + mod * chipsPerModule + chip;
    } else if (Nhs == 1) { // turbo geometry: layer -> stave -> mod -> chip (no halfstave)
      int chipsPerModule = Nchip;
      int chipsPerStave = Nmod * chipsPerModule;
      return getFirstChipIndex(lay, -1, subDetID) + stave * chipsPerStave + mod * chipsPerModule + chip;
    }
  }

  LOGP(warning, "Chip index not found for subDetID %d, volume %d, layer %d, stave %d, halfstave %d, module %d, chip %d, returning numeric limit", subDetID, volume, lay, stave, halfstave, mod, chip);
  return std::numeric_limits<unsigned short>::max(); // not found
}

//__________________________________________________________________________
bool GeometryTGeo::getChipID(int index, int& subDetID, int& petalcase, int& disk, int& lay, int& stave, int& halfstave, int& mod, int& chip) const
{
  subDetID = getSubDetID(index);
  petalcase = getPetalCase(index);
  disk = getDisk(index);
  lay = getLayer(index);
  stave = getStave(index);
  if (mNumberOfHalfStaves[lay] == 2) {
    halfstave = getHalfStave(index);
  } else {
    halfstave = 0; // if not staggered geometry, return 0
  }
  halfstave = getHalfStave(index);
  mod = getModule(index);
  chip = getChip(index);

  return kTRUE;
}

//__________________________________________________________________________
TString GeometryTGeo::getMatrixPath(int index) const
{

  int subDetID, petalcase, disk, layer, stave, halfstave, mod, chip;
  getChipID(index, subDetID, petalcase, disk, layer, stave, halfstave, mod, chip);

  // PrintChipID(index, subDetID, petalcase, disk, layer, stave, halfstave, mod, chip);

  TString path = Form("/cave_1/barrel_1/%s_2/", GeometryTGeo::getTRKVolPattern());

  // build the path
  if (subDetID == 0) { // VD
    if (disk >= 0) {
      path += Form("%s_%d_%d/", getTRKPetalAssemblyPattern(), petalcase, petalcase + 1);                                               // PETAL_n
      path += Form("%s%d_%s%d_1/", getTRKPetalPattern(), petalcase, getTRKPetalDiskPattern(), disk);                                   // PETALCASEx_DISKy_1
      path += Form("%s%d_%s%d_%s%d_1/", getTRKPetalPattern(), petalcase, getTRKPetalDiskPattern(), disk, getTRKChipPattern(), disk);   // PETALCASEx_DISKy_TRKChipy_1
      path += Form("%s%d_%s%d_%s%d_1/", getTRKPetalPattern(), petalcase, getTRKPetalDiskPattern(), disk, getTRKSensorPattern(), disk); // PETALCASEx_DISKy_TRKSensory_1
    } else if (layer >= 0) {
      path += Form("%s_%d_%d/", getTRKPetalAssemblyPattern(), petalcase, petalcase + 1);               // PETAL_n
      path += Form("%s%d_%s%d_1/", getTRKPetalPattern(), petalcase, getTRKPetalLayerPattern(), layer); // PETALCASEx_LAYERy_1
      // path += Form("%s%d_%s%d_%s%d_1/", getTRKPetalPattern(), petalcase, getTRKPetalLayerPattern(), layer, getTRKStavePattern(), layer);  // PETALCASEx_LAYERy_TRKStavey_1
      path += Form("%s%d_%s%d_%s%d_1/", getTRKPetalPattern(), petalcase, getTRKPetalLayerPattern(), layer, getTRKChipPattern(), layer);   // PETALCASEx_LAYERy_TRKChipy_1
      path += Form("%s%d_%s%d_%s%d_1/", getTRKPetalPattern(), petalcase, getTRKPetalLayerPattern(), layer, getTRKSensorPattern(), layer); // PETALCASEx_LAYERy_TRKSensory_1
    }
  } else if (subDetID == 1) {                             // MLOT
    path += Form("%s%d_1/", getTRKLayerPattern(), layer); // TRKLayerx_1
    if (mLayoutMLOT == eMLOTLayout::kCylindrical) {
      path += Form("%s%d_1/", getTRKSensorPattern(), layer); // TRKSensorx_1
    } else {
      path += Form("%s%d_%d/", getTRKStavePattern(), layer, stave);           // TRKStavex_y
      if (mNumberOfHalfStaves[layer] == 2) {                                  // staggered geometry
        path += Form("%s%d_%d/", getTRKHalfStavePattern(), layer, halfstave); // TRKHalfStavex_y
      }
      path += Form("%s%d_%d/", getTRKModulePattern(), layer, mod); // TRKModulx_y
      path += Form("%s%d_%d/", getTRKChipPattern(), layer, chip);  // TRKChipx_y
      path += Form("%s%d_1/", getTRKSensorPattern(), layer);       // TRKSensorx_1
    }
  }
  return path;
}

//__________________________________________________________________________
TGeoHMatrix* GeometryTGeo::extractMatrixSensor(int index) const
{
  // extract matrix transforming from the PHYSICAL sensor frame to global one
  // Note, the if the effective sensitive layer thickness is smaller than the
  // total physical sensor tickness, this matrix is biased and connot be used
  // directly for transformation from sensor frame to global one.
  // Therefore we need to add a shift

  auto path = getMatrixPath(index);

  static TGeoHMatrix matTmp;
  gGeoManager->PushPath(); // Preserve the modeler state.

  if (!gGeoManager->cd(path.Data())) {
    gGeoManager->PopPath();
    LOG(error) << "Error in cd-ing to " << path.Data();
    return nullptr;
  } // end if !gGeoManager

  matTmp = *gGeoManager->GetCurrentMatrix(); // matrix may change after cd

  // RSS
  // matTmp.Print();
  // Restore the modeler state.
  gGeoManager->PopPath();

  static int chipInGlo{0};

  /// TODO:
  // account for the difference between physical sensitive layer (where charge collection is simulated) and effective sensor thicknesses
  // in the VD case this will be accounted by specialized functions during the clusterization (following what it is done for ITS3)
  // this can be done once the right sensor thickness is in place in the geometry
  // double delta = 0.;
  // if (getSubDetID(index) == 1){ /// ML/OT
  //   delta = Segmentation::SensorLayerThicknessVD - Segmentation::SiliconTickness;
  //   static TGeoTranslation tra(0., 0.5 * delta, 0.);
  //   matTmp *= tra;
  // }
  // std::cout<<"-----"<<std::endl;
  // matTmp.Print();

  return &matTmp;
}

//__________________________________________________________________________
void GeometryTGeo::defineMLOTSensors()
{
  for (int i = 0; i < mSize; i++) {
    if (getSubDetID(i) == 0) {
      continue;
    }
    sensorsMLOT.push_back(i);
  }
}

//__________________________________________________________________________
void GeometryTGeo::fillTrackingFramesCacheMLOT()
{
  // fill for every sensor of ML & OT its tracking frame parameters
  if (!isTrackingFrameCachedMLOT() && !sensorsMLOT.empty()) {
    size_t newSize = sensorsMLOT.size();
    mCacheRefXMLOT.resize(newSize);
    mCacheRefAlphaMLOT.resize(newSize);
    for (int i = 0; i < newSize; i++) {
      int sensorId = sensorsMLOT[i];
      extractSensorXAlphaMLOT(sensorId, mCacheRefXMLOT[i], mCacheRefAlphaMLOT[i]);
    }
  }
}

//__________________________________________________________________________
void GeometryTGeo::fillMatrixCache(int mask)
{
  if (mSize < 1) {
    LOG(warning) << "The method Build was not called yet";
    Build(mask);
    return;
  }

  // build matrices
  if ((mask & o2::math_utils::bit2Mask(o2::math_utils::TransformType::L2G)) && !getCacheL2G().isFilled()) {
    // Matrices for Local (Sensor!!! rather than the full chip) to Global frame transformation
    LOGP(info, "Loading {} L2G matrices from TGeo; there are {} matrices", getName(), mSize);
    auto& cacheL2G = getCacheL2G();
    cacheL2G.setSize(mSize);

    for (int i = 0; i < mSize; i++) { /// here get the matrices for det ID between 0 and 257 (mSize = 258 at the moment)
      TGeoHMatrix* hm = extractMatrixSensor(i);
      cacheL2G.setMatrix(Mat3D(*hm), i);
    }
  }

  // build T2L matrices for ML & OT !! VD is yet to be implemented once its geometry will be more refined
  if ((mask & o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L)) && !getCacheT2L().isFilled()) {
    LOGP(info, "Loading {} T2L matrices from TGeo for ML & OT", getName());
    if (sensorsMLOT.size()) {
      int m_Size = sensorsMLOT.size();
      auto& cacheT2L = getCacheT2L();
      cacheT2L.setSize(m_Size);
      for (int i = 0; i < m_Size; i++) {
        int sensorID = sensorsMLOT[i];
        TGeoHMatrix& hm = createT2LMatrixMLOT(sensorID);
        cacheT2L.setMatrix(Mat3D(hm), i); // here, sensorIDs from 0 to 374, sensorIDs shifted to 36 !
      }
    }
  }

  // TODO: build matrices for the cases T2L, T2G and T2GRot when needed
}

//__________________________________________________________________________

#ifdef ENABLE_UPGRADES
const char* GeometryTGeo::composeSymNameLayer(int d, int layer)
{
  return Form("%s/%s%d", composeSymNameTRK(d), getTRKLayerPattern(), layer);
}
#endif

const char* GeometryTGeo::composeSymNameStave(int d, int layer)
{
  return Form("%s/%s%d", composeSymNameLayer(d, layer), getTRKStavePattern(), layer);
}

const char* GeometryTGeo::composeSymNameModule(int d, int layer)
{
  return Form("%s/%s%d", composeSymNameStave(d, layer), getTRKModulePattern(), layer);
}

const char* GeometryTGeo::composeSymNameChip(int d, int layer)
{
  return Form("%s/%s%d", composeSymNameStave(d, layer), getTRKChipPattern(), layer);
}

const char* GeometryTGeo::composeSymNameSensor(int d, int layer)
{
  return Form("%s/%s%d", composeSymNameChip(d, layer), getTRKSensorPattern(), layer);
}

//__________________________________________________________________________
int GeometryTGeo::extractVolumeCopy(const char* name, const char* prefix) const
{
  TString nms = name;
  if (!nms.BeginsWith(prefix)) {
    return -1;
  }
  nms.Remove(0, strlen(prefix));
  if (!isdigit(nms.Data()[0])) {
    return -1;
  }
  return nms.Atoi();
}

//__________________________________________________________________________
int GeometryTGeo::extractNumberOfLayersMLOT()
{
  int numberOfLayers = 0;
  TGeoVolume* trkV = gGeoManager->GetVolume(getTRKVolPattern());
  if (trkV == nullptr) {
    LOG(fatal) << getName() << " volume " << getTRKVolPattern() << " is not in the geometry";
  }

  // Loop on all TRKV nodes, count Layer volumes by checking names
  // Build on the fly layer - wrapper correspondence
  TObjArray* nodes = trkV->GetNodes();
  // nodes->Print();
  int nNodes = nodes->GetEntriesFast();
  for (int j = 0; j < nNodes; j++) {
    int lrID = -1;
    auto nd = dynamic_cast<TGeoNode*>(nodes->At(j));
    const char* name = nd->GetName();
    if (strstr(name, getTRKLayerPattern()) != nullptr) {
      numberOfLayers++;
      if ((lrID = extractVolumeCopy(name, GeometryTGeo::getTRKLayerPattern())) < 0) {
        LOG(fatal) << "Failed to extract layer ID from the " << name;
      }
      mLayerToWrapper[lrID] = -1;                                 // not wrapped
    } else if (strstr(name, getTRKWrapVolPattern()) != nullptr) { // this is a wrapper volume, may cointain layers
      int wrID = -1;
      if ((wrID = extractVolumeCopy(name, GeometryTGeo::getTRKWrapVolPattern())) < 0) {
        LOG(fatal) << "Failed to extract wrapper ID from the " << name;
      }
      TObjArray* nodesW = nd->GetNodes();
      int nNodesW = nodesW->GetEntriesFast();

      for (int jw = 0; jw < nNodesW; jw++) {
        auto ndW = dynamic_cast<TGeoNode*>(nodesW->At(jw))->GetName();
        if (strstr(ndW, getTRKLayerPattern()) != nullptr) {
          if ((lrID = extractVolumeCopy(ndW, GeometryTGeo::getTRKLayerPattern())) < 0) {
            LOGP(fatal, "Failed to extract layer ID from wrapper volume '{}' from one of its nodes '{}'", name, ndW);
          }
          numberOfLayers++;
          mLayerToWrapper[lrID] = wrID;
        }
      }
    }
  }
  return numberOfLayers;
}

//__________________________________________________________________________
int GeometryTGeo::extractNumberOfPetalsVD() const
{
  int numberOfPetals = 0;
  TGeoVolume* trkV = gGeoManager->GetVolume(getTRKVolPattern());
  if (!trkV) {
    LOGP(fatal, "{} volume {} is not in the geometry", getName(), getTRKVolPattern());
    return 0;
  }

  // Loop on all TRKV nodes, count PETAL assemblies and their contents
  TObjArray* nodes = trkV->GetNodes();
  if (!nodes) {
    LOGP(warning, "{} volume has no child nodes", getTRKVolPattern());
    return 0;
  }

  LOGP(info, "Searching for petal assemblies in {} (pattern: {})",
       getTRKVolPattern(), getTRKPetalAssemblyPattern());

  for (int j = 0; j < nodes->GetEntriesFast(); j++) {
    auto* nd = dynamic_cast<TGeoNode*>(nodes->At(j));
    const char* name = nd->GetName();

    if (strstr(name, getTRKPetalAssemblyPattern()) != nullptr) {
      numberOfPetals++;
      LOGP(info, "Found petal assembly: {}", name);

      // Get petal volume and its nodes for debugging
      TGeoVolume* petalVol = nd->GetVolume();
      if (petalVol) {
        TObjArray* petalNodes = petalVol->GetNodes();
        if (petalNodes) {
          LOGP(debug, "Petal {} contains {} child nodes", name, petalNodes->GetEntriesFast());
          // Print all nodes in this petal
          for (int k = 0; k < petalNodes->GetEntriesFast(); k++) {
            auto* petalNode = dynamic_cast<TGeoNode*>(petalNodes->At(k));
            LOGP(debug, "  Node {}: {}", k, petalNode->GetName());
          }
        } else {
          LOGP(warning, "Petal {} has no child nodes", name);
        }
      } else {
        LOGP(warning, "Petal {} has no volume", name);
      }
    }
  }

  if (numberOfPetals == 0) {
    LOGP(warning, "No petal assemblies found in geometry");
  } else {
    LOGP(info, "Found {} petal assemblies", numberOfPetals);
  }

  return numberOfPetals;
}

//__________________________________________________________________________
int GeometryTGeo::extractNumberOfActivePartsVD() const
{
  // The number of active parts returned here is 36 = 4 petals * (3 layers + 6 disks)
  int numberOfParts = 0;
  TGeoVolume* vdV = gGeoManager->GetVolume(getTRKVolPattern());
  if (!vdV) {
    LOGP(fatal, "{} volume {} is not in the geometry", getName(), getTRKVolPattern());
    return 0;
  }

  // Find first petal to count its active parts
  TObjArray* nodes = vdV->GetNodes();
  if (!nodes) {
    LOGP(warning, "{} volume has no child nodes", getTRKVolPattern());
    return 0;
  }

  bool petalFound = false;

  for (int j = 0; j < nodes->GetEntriesFast(); j++) {
    auto* nd = dynamic_cast<TGeoNode*>(nodes->At(j));
    const char* name = nd->GetName();
    if (strstr(name, getTRKPetalAssemblyPattern()) == nullptr) {
      continue;
    }

    petalFound = true;
    LOGP(info, "Counting active parts in petal: {}", name);

    // Found a petal, count its layers and disks
    TGeoVolume* petalVol = nd->GetVolume();
    if (!petalVol) {
      LOGP(warning, "Petal {} has no volume", name);
      break;
    }

    TObjArray* petalNodes = petalVol->GetNodes();
    if (!petalNodes) {
      LOGP(warning, "Petal {} has no child nodes", name);
      break;
    }

    for (int k = 0; k < petalNodes->GetEntriesFast(); k++) {
      auto* petalNode = dynamic_cast<TGeoNode*>(petalNodes->At(k));
      const char* nodeName = petalNode->GetName();

      if (strstr(nodeName, getTRKPetalLayerPattern()) != nullptr ||
          strstr(nodeName, getTRKPetalDiskPattern()) != nullptr) {
        numberOfParts++;
        LOGP(debug, "Found active part in {}: {}", name, nodeName);
      }
    }
    // We only need to check one petal as they're identical
    break;
  }

  if (!petalFound) {
    LOGP(warning, "No petal assembly found matching pattern '{}'", getTRKPetalAssemblyPattern());
    return 0;
  }

  if (numberOfParts == 0) {
    LOGP(warning, "No active parts (layers/disks) found in petal");
    return 0;
  }

  // Multiply by number of petals since all petals are identical
  int totalParts = numberOfParts * mNumberOfPetalsVD;
  LOGP(info, "Total number of active parts: {} ({}*{})",
       totalParts, numberOfParts, mNumberOfPetalsVD);
  return totalParts;
}

//__________________________________________________________________________
int GeometryTGeo::extractNumberOfDisksVD() const
{
  // Count disks in the first petal (all petals are identical)
  int numberOfDisks = 0;
  TGeoVolume* vdV = gGeoManager->GetVolume(getTRKVolPattern());
  if (!vdV) {
    LOGP(fatal, "{} volume {} is not in the geometry", getName(), getTRKVolPattern());
    return 0;
  }

  // Find first petal
  TObjArray* nodes = vdV->GetNodes();
  if (!nodes) {
    LOGP(warning, "{} volume has no child nodes", getTRKVolPattern());
    return 0;
  }

  bool petalFound = false;

  for (int j = 0; j < nodes->GetEntriesFast(); j++) {
    auto* nd = dynamic_cast<TGeoNode*>(nodes->At(j));
    if (strstr(nd->GetName(), getTRKPetalAssemblyPattern()) == nullptr) {
      continue;
    }

    petalFound = true;
    LOGP(info, "Counting disks in petal: {}", nd->GetName());

    // Count disks in this petal
    TGeoVolume* petalVol = nd->GetVolume();
    if (!petalVol) {
      LOGP(warning, "Petal {} has no volume", nd->GetName());
      break;
    }

    TObjArray* petalNodes = petalVol->GetNodes();
    if (!petalNodes) {
      LOGP(warning, "Petal {} has no child nodes", nd->GetName());
      break;
    }

    for (int k = 0; k < petalNodes->GetEntriesFast(); k++) {
      auto* petalNode = dynamic_cast<TGeoNode*>(petalNodes->At(k));
      if (strstr(petalNode->GetName(), getTRKPetalDiskPattern()) != nullptr) {
        numberOfDisks++;
        LOGP(info, "Found disk in {} : {}", nd->GetName(), petalNode->GetName());
      }
    }
    // One petal is enough
    break;
  }

  if (!petalFound) {
    LOGP(warning, "No petal assembly found matching pattern '{}'", getTRKPetalAssemblyPattern());
  }

  if (numberOfDisks == 0) {
    LOGP(warning, "No disks found in VD geometry");
  }

  return numberOfDisks;
}

//__________________________________________________________________________
int GeometryTGeo::extractNumberOfLayersVD() const
{
  // Count layers in the first petal (all petals are identical)
  int numberOfLayers = 0;
  TGeoVolume* vdV = gGeoManager->GetVolume(getTRKVolPattern());
  if (!vdV) {
    LOGP(fatal, "{} volume {} is not in the geometry", getName(), getTRKVolPattern());
    return 0;
  }

  // Find first petal
  TObjArray* nodes = vdV->GetNodes();
  if (!nodes) {
    LOGP(warning, "{} volume has no child nodes", getTRKVolPattern());
    return 0;
  }

  bool petalFound = false;

  for (int j = 0; j < nodes->GetEntriesFast(); j++) {
    auto* nd = dynamic_cast<TGeoNode*>(nodes->At(j));
    if (strstr(nd->GetName(), getTRKPetalAssemblyPattern()) == nullptr) {
      continue;
    }

    petalFound = true;
    LOGP(info, "Counting layers in petal: {}", nd->GetName());

    // Count layers in this petal
    TGeoVolume* petalVol = nd->GetVolume();
    if (!petalVol) {
      LOGP(warning, "Petal {} has no volume", nd->GetName());
      break;
    }

    TObjArray* petalNodes = petalVol->GetNodes();
    if (!petalNodes) {
      LOGP(warning, "Petal {} has no child nodes", nd->GetName());
      break;
    }

    for (int k = 0; k < petalNodes->GetEntriesFast(); k++) {
      auto* petalNode = dynamic_cast<TGeoNode*>(petalNodes->At(k));
      if (strstr(petalNode->GetName(), getTRKPetalLayerPattern()) != nullptr) {
        numberOfLayers++;
        LOGP(info, "Found layer in {} : {}", nd->GetName(), petalNode->GetName());
      }
    }
    // One petal is enough
    break;
  }

  if (!petalFound) {
    LOGP(warning, "No petal assembly found matching pattern '{}'", getTRKPetalAssemblyPattern());
  }

  if (numberOfLayers == 0) {
    LOGP(warning, "No layers found in VD geometry");
  }

  return numberOfLayers;
}

//__________________________________________________________________________
int GeometryTGeo::extractNumberOfChipsPerPetalVD() const
{
  // The number of chips per petal returned here is 9 for each layer = number of layers + number of quarters of disks per petal
  int numberOfChips = 0;
  TGeoVolume* vdV = gGeoManager->GetVolume(getTRKVolPattern());
  if (!vdV) {
    LOGP(fatal, "{} volume {} is not in the geometry", getName(), getTRKVolPattern());
    return 0;
  }

  // Find first petal assembly
  TObjArray* nodes = vdV->GetNodes();
  if (!nodes) {
    LOGP(warning, "{} volume has no child nodes", getTRKVolPattern());
    return 0;
  }

  bool petalFound = false;

  for (int j = 0; j < nodes->GetEntriesFast(); j++) {
    auto* nd = dynamic_cast<TGeoNode*>(nodes->At(j));
    const char* name = nd->GetName();
    if (strstr(name, getTRKPetalAssemblyPattern()) == nullptr) {
      continue;
    }

    petalFound = true;
    LOGP(info, "Counting chips in petal: {}", name);

    // Found a petal, count sensors in its layers and disks
    TGeoVolume* petalVol = nd->GetVolume();
    if (!petalVol) {
      LOGP(warning, "Petal {} has no volume", name);
      break;
    }

    TObjArray* petalNodes = petalVol->GetNodes();
    if (!petalNodes) {
      LOGP(warning, "Petal {} has no child nodes", name);
      break;
    }

    for (int k = 0; k < petalNodes->GetEntriesFast(); k++) {
      auto* petalNode = dynamic_cast<TGeoNode*>(petalNodes->At(k));
      const char* nodeName = petalNode->GetName();
      TGeoVolume* vol = petalNode->GetVolume();

      if (!vol) {
        LOGP(debug, "Node {} has no volume", nodeName);
        continue;
      }

      // Look for sensors in this volume
      TObjArray* subNodes = vol->GetNodes();
      if (!subNodes) {
        LOGP(debug, "Node {} has no sub-nodes", nodeName);
        continue;
      }

      for (int i = 0; i < subNodes->GetEntriesFast(); i++) {
        auto* subNode = dynamic_cast<TGeoNode*>(subNodes->At(i));
        if (strstr(subNode->GetName(), getTRKChipPattern()) != nullptr) {
          numberOfChips++;
          LOGP(debug, "Found chip in {}: {}", nodeName, subNode->GetName());
        }
      }
    }
    // We only need one petal
    break;
  }

  if (!petalFound) {
    LOGP(warning, "No petal assembly found matching pattern '{}'", getTRKPetalAssemblyPattern());
  }

  if (numberOfChips == 0) {
    LOGP(warning, "No chips/sensors found in VD petal");
  }

  LOGP(info, "Number of chips per petal: {}", numberOfChips);
  return numberOfChips;
}

//__________________________________________________________________________
int GeometryTGeo::extractNumberOfStavesMLOT(int lay) const
{
  int numberOfStaves = 0;

  std::string layName = Form("%s%d", getTRKLayerPattern(), lay);
  TGeoVolume* layV = gGeoManager->GetVolume(layName.c_str());

  if (layV == nullptr) {
    LOG(fatal) << getName() << " volume " << getTRKLayerPattern() << " is not in the geometry";
  }

  // Loop on all layV nodes, count Layer volumes by checking names
  TObjArray* nodes = layV->GetNodes();
  // std::cout << "Printing nodes for layer " << lay << std::endl;
  // nodes->Print();
  int nNodes = nodes->GetEntriesFast();

  for (int j = 0; j < nNodes; j++) {
    int lrID = -1;
    auto nd = dynamic_cast<TGeoNode*>(nodes->At(j)); /// layer node
    const char* name = nd->GetName();
    if (strstr(name, getTRKStavePattern()) != nullptr) {
      numberOfStaves++;
    }
  }
  return numberOfStaves;
}

//__________________________________________________________________________
int GeometryTGeo::extractNumberOfHalfStavesMLOT(int lay) const
{
  int numberOfHalfStaves = 0;

  std::string staveName = Form("%s%d", getTRKStavePattern(), lay);
  TGeoVolume* staveV = gGeoManager->GetVolume(staveName.c_str());

  if (staveV == nullptr) {
    LOG(fatal) << getName() << " volume " << getTRKStavePattern() << " is not in the geometry";
  }

  // Loop on all layV nodes, count Layer volumes by checking names
  TObjArray* nodes = staveV->GetNodes();
  // std::cout << "Printing nodes for layer " << lay << std::endl;
  // nodes->Print();
  int nNodes = nodes->GetEntriesFast();

  for (int j = 0; j < nNodes; j++) {
    auto nd = dynamic_cast<TGeoNode*>(nodes->At(j)); /// layer node
    const char* name = nd->GetName();
    if (strstr(name, getTRKHalfStavePattern()) != nullptr) {
      numberOfHalfStaves++;
    }
  }

  if (numberOfHalfStaves == 0) {
    numberOfHalfStaves = 1; /// in case of turbo geometry, there is no half stave volume, but only stave volume
  }
  return numberOfHalfStaves;
}

//__________________________________________________________________________
int GeometryTGeo::extractNumberOfModulesMLOT(int lay) const
{
  int numberOfModules = 0;

  std::string staveName = Form("%s%d", (mNumberOfHalfStaves[lay] == 2 ? getTRKHalfStavePattern() : getTRKStavePattern()), lay);
  TGeoVolume* staveV = gGeoManager->GetVolume(staveName.c_str());

  if (staveV == nullptr) {
    LOG(fatal) << getName() << " volume " << (mNumberOfHalfStaves[lay] == 2 ? getTRKHalfStavePattern() : getTRKStavePattern()) << " is not in the geometry";
  }

  // Loop on all staveV nodes, count Module volumes by checking names
  TObjArray* nodes = staveV->GetNodes();
  int nNodes = nodes->GetEntriesFast();

  for (int j = 0; j < nNodes; j++) {
    auto nd = dynamic_cast<TGeoNode*>(nodes->At(j)); /// stave node
    const char* name = nd->GetName();
    if (strstr(name, getTRKModulePattern()) != nullptr) {
      numberOfModules++;
    }
  }
  return numberOfModules;
}

//__________________________________________________________________________
int GeometryTGeo::extractNumberOfChipsMLOT(int lay) const
{
  int numberOfChips = 0;

  std::string moduleName = Form("%s%d", getTRKModulePattern(), lay);
  TGeoVolume* moduleV = gGeoManager->GetVolume(moduleName.c_str());

  if (moduleV == nullptr) {
    LOG(fatal) << getName() << " volume " << getTRKModulePattern() << " is not in the geometry";
  }

  // Loop on all moduleV nodes, count Chip volumes by checking names
  TObjArray* nodes = moduleV->GetNodes();
  int nNodes = nodes->GetEntriesFast();

  for (int j = 0; j < nNodes; j++) {
    auto nd = dynamic_cast<TGeoNode*>(nodes->At(j)); /// module node
    const char* name = nd->GetName();
    if (strstr(name, getTRKChipPattern()) != nullptr) {
      numberOfChips++;
    }
  }
  return numberOfChips;
}

//__________________________________________________________________________
void GeometryTGeo::PrintChipID(int index, int subDetID, int petalcase, int disk, int lay, int stave, int halfstave, int mod, int chip) const
{
  std::cout << "\nindex = " << index << std::endl;
  std::cout << "subDetID = " << subDetID << std::endl;
  std::cout << "petalcase = " << petalcase << std::endl;
  std::cout << "layer = " << lay << std::endl;
  std::cout << "disk = " << disk << std::endl;
  std::cout << "first chip index = " << getFirstChipIndex(lay, petalcase, subDetID) << std::endl;
  std::cout << "stave = " << stave << std::endl;
  std::cout << "halfstave = " << halfstave << std::endl;
  std::cout << "module = " << mod << std::endl;
  std::cout << "chip = " << chip << std::endl;
}

//__________________________________________________________________________
void GeometryTGeo::Print(Option_t*) const
{
  if (!isBuilt()) {
    LOGF(info, "Geometry not built yet!");
    return;
  }
  std::cout << "Detector ID: " << sInstance.get()->getDetID() << std::endl;

  LOGF(info, "Summary of GeometryTGeo: %s", getName());
  LOGF(info, "Number of layers ML + OT: %d", mNumberOfLayersMLOT);
  LOGF(info, "Number of active parts VD: %d", mNumberOfActivePartsVD);
  LOGF(info, "Number of layers VD: %d", mNumberOfLayersVD);
  LOGF(info, "Number of petals VD: %d", mNumberOfPetalsVD);
  LOGF(info, "Number of disks VD: %d", mNumberOfDisksVD);
  LOGF(info, "Number of chips per petal VD: ");
  for (int i = 0; i < mNumberOfPetalsVD; i++) {
    LOGF(info, "%d", mNumberOfChipsPerPetalVD[i]);
  }
  LOGF(info, "Number of staves and half staves per layer MLOT: ");
  for (int i = 0; i < mNumberOfLayersMLOT; i++) {
    std::string mlot = "";
    mlot = (i < constants::ML::nLayers) ? "ML" : "OT";
    LOGF(info, "Layer: %d, %s, %d staves, %d half staves per stave", i, mlot.c_str(), mNumberOfStaves[i], mNumberOfHalfStaves[i]);
  }
  LOGF(info, "Number of modules per stave (half stave) in each ML(OT) layer: ");
  for (int i = 0; i < mNumberOfLayersMLOT; i++) {
    LOGF(info, "%d", mNumberOfModules[i]);
  }
  LOGF(info, "Number of chips per module MLOT: ");
  for (int i = 0; i < mNumberOfLayersMLOT; i++) {
    LOGF(info, "%d", mNumberOfChips[i]);
  }
  LOGF(info, "Number of chips per layer MLOT: ");
  for (int i = 0; i < mNumberOfLayersMLOT; i++) {
    LOGF(info, "%d", mNumberOfChipsPerLayerMLOT[i]);
  }
  LOGF(info, "Total number of chips: %d", getNumberOfChips());

  std::cout << "mLastChipIndex = [";
  for (int i = 0; i < mLastChipIndex.size(); i++) {
    std::cout << mLastChipIndex[i];
    if (i < mLastChipIndex.size() - 1) {
      std::cout << ", ";
    }
  }
  std::cout << "]" << std::endl;
  std::cout << "mLastChipIndexVD = [";
  for (int i = 0; i < mLastChipIndexVD.size(); i++) {
    std::cout << mLastChipIndexVD[i];
    if (i < mLastChipIndexVD.size() - 1) {
      std::cout << ", ";
    }
  }
  std::cout << "]" << std::endl;
}

//__________________________________________________________________________
int GeometryTGeo::getBarrelLayer(int chipID) const
{
  // for barrel layers only,
  // so it would be consistent with number of layers i.e. from 0 to 10,
  // starting from VD0 to OT10;
  // skip the disks;

  int subDetID = getSubDetID(chipID);
  int subLayerID = getLayer(chipID);

  if (subDetID < 0 || subDetID > 1) {
    LOG(error) << "getBarrelLayer(): Invalid subDetID for barrel: " << subDetID
               << ". Expected values are 0 or 1.";
    return -1;
  }

  if (subLayerID < 0 || subLayerID > 7) {
    LOG(error) << "getBarrelLayer(): Invalid subLayerID for barrel: " << subDetID
               << ". Expected values are between 0 and 7.";
    return -1;
  }

  const int baseOffsets[] = {0, 3};

  return baseOffsets[subDetID] + subLayerID;
}

//__________________________________________________________________________
void GeometryTGeo::extractSensorXAlphaMLOT(int chipID, float& x, float& alp)
{
  // works for ML and OT only, a.k.a flat sensors !!!
  double locA[3] = {-100., 0., 0.}, locB[3] = {100., 0., 0.}, gloA[3], gloB[3];
  double xp{0}, yp{0};

  if (getSubDetID(chipID) == 0) {

    LOG(error) << "extractSensorXAlphaMLOT(): VD layers are not supported yet! chipID = " << chipID;
    return;

  } else { // flat sensors, ML and OT
    const TGeoHMatrix* matL2G = extractMatrixSensor(chipID);
    matL2G->LocalToMaster(locA, gloA);
    matL2G->LocalToMaster(locB, gloB);
    double dx = gloB[0] - gloA[0], dy = gloB[1] - gloA[1];
    double t = (gloB[0] * dx + gloB[1] * dy) / (dx * dx + dy * dy);
    xp = gloB[0] - dx * t;
    yp = gloB[1] - dy * t;
  }

  alp = std::atan2(yp, xp);
  x = std::hypot(xp, yp);
  o2::math_utils::bringTo02Pi(alp);

  /// TODO:
  // once the VD segmentation is done, VD should be added
}

//__________________________________________________________________________
TGeoHMatrix& GeometryTGeo::createT2LMatrixMLOT(int chipID)
{
  // works only for ML & OT
  // for VD is yet to be implemented once we have more refined geometry
  if (getSubDetID(chipID) == 0) {

    LOG(error) << "createT2LMatrixMLOT(): VD layers are not supported yet! chipID = " << chipID
               << "returning dummy values! ";
    static TGeoHMatrix dummy;
    return dummy;

  } else {
    static TGeoHMatrix t2l;
    t2l.Clear();
    float alpha = getSensorRefAlphaMLOT(chipID);
    t2l.RotateZ(alpha * TMath::RadToDeg());
    const TGeoHMatrix* matL2G = extractMatrixSensor(chipID);
    const TGeoHMatrix& matL2Gi = matL2G->Inverse();
    t2l.MultiplyLeft(&matL2Gi);
    return t2l;
  }
}

} // namespace trk
} // namespace o2
