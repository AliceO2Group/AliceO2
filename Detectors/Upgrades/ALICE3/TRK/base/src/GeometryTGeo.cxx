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

using Segmentation = o2::trk::SegmentationChip;

namespace o2
{
namespace trk
{
std::unique_ptr<o2::trk::GeometryTGeo> GeometryTGeo::sInstance;

// Names
std::string GeometryTGeo::sVolumeName = "TRKV";
std::string GeometryTGeo::sLayerName = "TRKLayer";
std::string GeometryTGeo::sPetalAssemblyName = "PETAL";
std::string GeometryTGeo::sPetalName = "PETALCASE";
std::string GeometryTGeo::sPetalDiskName = "DISK";
std::string GeometryTGeo::sPetalLayerName = "LAYER";
std::string GeometryTGeo::sStaveName = "TRKStave";
std::string GeometryTGeo::sChipName = "TRKChip";
std::string GeometryTGeo::sSensorName = "TRKSensor";

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
  ///// total elements = x staves (*2 half staves if staggered geometry) * 8 layers ML+OT + 4 petal cases * (3 layers + 6 disks)
  ///// indexing from 0 to 35: VD petals -> layers -> disks
  ///// indexing from 36 to y: MLOT staves

  if (isBuilt()) {
    LOGP(warning, "Already built");
    return; // already initialized
  }

  if (gGeoManager == nullptr) {
    LOGP(fatal, "Geometry is not loaded");
  }

  mNumberOfLayersMLOT = extractNumberOfLayersMLOT();
  mNumberOfPetalsVD = extractNumberOfPetalsVD();
  mNumberOfActivePartsVD = extractNumberOfActivePartsVD();
  mNumberOfLayersVD = extractNumberOfLayersVD();
  mNumberOfDisksVD = extractNumberOfDisksVD();

  mNumberOfStaves.resize(mNumberOfLayersMLOT);
  mNumberOfHalfStaves.resize(mNumberOfLayersMLOT);
  mLastChipIndex.resize(mNumberOfPetalsVD + mNumberOfLayersMLOT);
  mLastChipIndexVD.resize(mNumberOfPetalsVD);
  mLastChipIndexMLOT.resize(mNumberOfLayersMLOT); /// ML and OT are part of TRK as the same detector, without disks
  mNumberOfChipsPerLayerVD.resize(mNumberOfLayersVD);
  mNumberOfChipsPerLayerMLOT.resize(mNumberOfLayersMLOT);
  mNumbersOfChipPerDiskVD.resize(mNumberOfDisksVD);
  mNumberOfChipsPerPetalVD.resize(mNumberOfPetalsVD);

  for (int i = 0; i < mNumberOfLayersMLOT; i++) {
    std::cout << "Layer MLOT: " << i << std::endl;
    mNumberOfStaves[i] = extractNumberOfStavesMLOT(i);
    mNumberOfHalfStaves[i] = extractNumberOfHalfStavesMLOT(i);
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
    mNumberOfChipsPerLayerMLOT[i] = extractNumberOfStavesMLOT(i) * extractNumberOfHalfStavesMLOT(i); // for the moment, considering 1 half stave = 1 chip. TODO: add the final segmentation in chips
    numberOfChipsTotal += mNumberOfChipsPerLayerMLOT[i];
    mLastChipIndex[i + mNumberOfPetalsVD] = numberOfChipsTotal - 1;
    mLastChipIndexMLOT[i] = numberOfChipsTotal - 1;
  }

  setSize(numberOfChipsTotal); /// temporary, number of chips = number of staves and active parts
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
  }

  else if (index <= mLastChipIndexVD[mNumberOfPetalsVD - 1]) {
    while (index > mLastChipIndexVD[petalcase]) {
      petalcase++;
    }
  }
  return petalcase;
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
    return lay - mNumberOfPetalsVD; /// numeration of MLOT layesrs  starting from 0
  }
  return -1; /// -1 if not found
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
    return index / mNumberOfHalfStaves[lay];
  }
  return -1; /// not found
}

//__________________________________________________________________________
int GeometryTGeo::getHalfStave(int index) const
{
  int subDetID = getSubDetID(index);
  int lay = getLayer(index);
  int petalcase = getPetalCase(index);
  int stave = getStave(index);

  if (subDetID == 0) { /// VD
    return -1;
  } else if (subDetID == 1) { /// MLOT
    int lay = getLayer(index);
    index -= getFirstChipIndex(lay, petalcase, subDetID); // get the index of the sensing element in the layer
    return index % 2;                                     /// 0 = half stave left, 1 = half stave right, as geometry is filled /// TODO: generalize once chips will be in place. Can it be working also with chips?
  }
  return -1; /// not found
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
int GeometryTGeo::getChipIndex(int subDetID, int petalcase, int disk, int lay, int stave, int halfstave) const
{
  if (subDetID == 0) { // VD
    if (lay == -1) {   // disk
      return getFirstChipIndex(lay, petalcase, subDetID) + mNumberOfLayersVD + disk;
    } else { // layer
      return getFirstChipIndex(lay, petalcase, subDetID) + lay;
    }
  } else if (subDetID == 1) {            // MLOT
    if (mNumberOfHalfStaves[lay] == 2) { // staggered geometry
      return getFirstChipIndex(lay, petalcase, subDetID) + stave * mNumberOfHalfStaves[lay] + halfstave;
    } else if (mNumberOfHalfStaves[lay] == 1) { // turbo geometry
      return getFirstChipIndex(lay, petalcase, subDetID) + stave;
    }
  }
  return -1; // not found
}

//__________________________________________________________________________
int GeometryTGeo::getChipIndex(int subDetID, int volume, int lay, int stave, int halfstave) const
{
  if (subDetID == 0) { // VD
    return volume;     /// In the current configuration for VD, each volume is the sensor element = chip. // TODO: when the geometry naming scheme will be changed, change this method

  } else if (subDetID == 1) {            // MLOT
    if (mNumberOfHalfStaves[lay] == 2) { // staggered geometry
      return getFirstChipIndex(lay, -1, subDetID) + stave * mNumberOfHalfStaves[lay] + halfstave;
    } else if (mNumberOfHalfStaves[lay] == 1) { // turbo geometry
      return getFirstChipIndex(lay, -1, subDetID) + stave;
    }
  }
  return -1; // not found
}

//__________________________________________________________________________
bool GeometryTGeo::getChipID(int index, int& subDetID, int& petalcase, int& disk, int& lay, int& stave, int& halfstave) const
{
  subDetID = getSubDetID(index);
  petalcase = getPetalCase(index);
  disk = getDisk(index);
  lay = getLayer(index);
  stave = getStave(index);
  halfstave = getHalfStave(index);

  return kTRUE;
}

//__________________________________________________________________________
TString GeometryTGeo::getMatrixPath(int index) const
{

  int subDetID, petalcase, disk, layer, stave, halfstave; //// TODO: add chips in a second step
  getChipID(index, subDetID, petalcase, disk, layer, stave, halfstave);

  // PrintChipID(index, subDetID, petalcase, disk, layer, stave, halfstave);

  // TString path = "/cave_1/barrel_1/TRKV_2/TRKLayer0_1/TRKStave0_1/TRKChip0_1/TRKSensor0_1/"; /// dummy path, to be used for tests
  TString path = Form("/cave_1/barrel_1/%s_2/", GeometryTGeo::getTRKVolPattern());

  if (subDetID == 0) { // VD
    if (disk >= 0) {
      path += Form("%s%d_%s%d_1/", getTRKPetalPattern(), petalcase, getTRKPetalDiskPattern(), disk);                                   // PETALCASEx_DISKy_1
      path += Form("%s%d_%s%d_%s%d_1/", getTRKPetalPattern(), petalcase, getTRKPetalDiskPattern(), disk, getTRKChipPattern(), disk);   // PETALCASEx_DISKy_TRKChipy_1
      path += Form("%s%d_%s%d_%s%d_1/", getTRKPetalPattern(), petalcase, getTRKPetalDiskPattern(), disk, getTRKSensorPattern(), disk); // PETALCASEx_DISKy_TRKSensory_1
    } else if (layer >= 0) {
      path += Form("%s%d_%s%d_1/", getTRKPetalPattern(), petalcase, getTRKPetalLayerPattern(), layer);                                    // PETALCASEx_LAYERy_1
      path += Form("%s%d_%s%d_%s%d_1/", getTRKPetalPattern(), petalcase, getTRKPetalLayerPattern(), layer, getTRKStavePattern(), layer);  // PETALCASEx_LAYERy_TRKStavey_1
      path += Form("%s%d_%s%d_%s%d_1/", getTRKPetalPattern(), petalcase, getTRKPetalLayerPattern(), layer, getTRKChipPattern(), layer);   // PETALCASEx_LAYERy_TRKChipy_1
      path += Form("%s%d_%s%d_%s%d_1/", getTRKPetalPattern(), petalcase, getTRKPetalLayerPattern(), layer, getTRKSensorPattern(), layer); // PETALCASEx_LAYERy_TRKSensory_1
    }
  } else if (subDetID == 1) {                                          // MLOT
    path += Form("%s%d_1/", getTRKLayerPattern(), layer);              // TRKLayerx_1
    path += Form("%s%d_%d/", getTRKStavePattern(), layer, stave);      // TRKStavex_y
    if (mNumberOfHalfStaves[layer] == 2) {                             // staggered geometry
      path += Form("%s%d_%d/", getTRKChipPattern(), layer, halfstave); // TRKChipx_0/1
    } else if (mNumberOfHalfStaves[layer] == 1) {                      // turbo geometry
      path += Form("%s%d_1/", getTRKChipPattern(), layer);             // TRKChipx_1
    }
    path += Form("%s%d_1/", getTRKSensorPattern(), layer); // TRKSensorx_1
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

  // TODO: build matrices for the cases T2L, T2G and T2GRot when needed
}

//__________________________________________________________________________

#ifdef ENABLE_UPGRADES
const char* GeometryTGeo::composeSymNameLayer(int d, int lr)
{
  return Form("%s/%s%d", composeSymNameTRK(d), getTRKLayerPattern(), lr);
}
#endif

const char* GeometryTGeo::composeSymNameStave(int d, int lr)
{
  return Form("%s/%s%d", composeSymNameLayer(d, lr), getTRKStavePattern(), lr);
}

const char* GeometryTGeo::composeSymNameChip(int d, int lr)
{
  return Form("%s/%s%d", composeSymNameStave(d, lr), getTRKChipPattern(), lr);
}

const char* GeometryTGeo::composeSymNameSensor(int d, int lr)
{
  return Form("%s/%s%d", composeSymNameChip(d, lr), getTRKSensorPattern(), lr);
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
        if (strstr(subNode->GetName(), getTRKSensorPattern()) != nullptr) {
          numberOfChips++;
          LOGP(debug, "Found sensor in {}: {}", nodeName, subNode->GetName());
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
    if (strstr(name, getTRKChipPattern()) != nullptr) {
      numberOfHalfStaves++;
    }
  }
  return numberOfHalfStaves;
}

//__________________________________________________________________________
void GeometryTGeo::PrintChipID(int index, int subDetID, int petalcase, int disk, int lay, int stave, int halfstave) const
{
  std::cout << "\nindex = " << index << std::endl;
  std::cout << "subDetID = " << subDetID << std::endl;
  std::cout << "petalcase = " << petalcase << std::endl;
  std::cout << "layer = " << lay << std::endl;
  std::cout << "disk = " << disk << std::endl;
  std::cout << "first chip index = " << getFirstChipIndex(lay, petalcase, subDetID) << std::endl;
  std::cout << "stave = " << stave << std::endl;
  std::cout << "halfstave = " << halfstave << std::endl;
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
  LOGF(info, "Number of layers ML + OL: %d", mNumberOfLayersMLOT);
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
    mlot = (i < 4) ? "ML" : "OT";
    LOGF(info, "Layer: %d, %s, %d staves, %d half staves per stave", i, mlot.c_str(), mNumberOfStaves[i], mNumberOfHalfStaves[i]);
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

} // namespace trk
} // namespace o2
