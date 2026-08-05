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

// TODO: clean up includes
#include <FT3Base/GeometryTGeo.h>
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

#include <cctype>  // for isdigit
#include <cstdio>  // for snprintf, NULL, printf
#include <cstring> // for strstr, strlen

using namespace TMath;
using namespace o2::detectors;

namespace o2
{
namespace ft3
{
std::unique_ptr<o2::ft3::GeometryTGeo> GeometryTGeo::sInstance;

std::string GeometryTGeo::sVolumeName = "FT3V";          ///< Mother volume name
std::string GeometryTGeo::sInnerVolumeName = "FT3Inner"; ///< Mother inner volume name
std::string GeometryTGeo::sLayerName = "FT3Layer";       ///< Layer name
std::string GeometryTGeo::sChipName = "FT3Chip";         ///< Chip name
// TODO: this is now only used for the not-segmented version; synchronise?
std::string GeometryTGeo::sSensorName = "FT3Sensor";     ///< Sensor name 
std::string GeometryTGeo::sPassiveName = "Passive";   ///< Passive material name

GeometryTGeo::~GeometryTGeo()
{
  if (!mOwner) {
    mOwner = true;
    sInstance.release();
  }
}
//__________________________________________________________________________
GeometryTGeo::GeometryTGeo(bool build, int loadTrans) : DetMatrixCache(detectors::DetID::FT3)
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

  // Forward discs part
  //int sensIdx = 0;
  int totDiscs = 0;
  int absStaveIdx = 0;
  mSize = 0;
  // TODO: clean up initialisation
  if (mChipIdxStave.size() == 0) 
  {
    mChipIdxStave.push_back(0);
  }
  if (mStaveIdxDisc.size() == 0)
  {
    mStaveIdxDisc.push_back(0);
  }
  for (int iDir = 0; iDir < 2; iDir++) {
    mNumberOfDiscs.push_back(extractNumberOfDiscs(iDir));
    LOG(info) << "direction " << iDir << " has " << mNumberOfDiscs[iDir] << " discs";
    totDiscs += mNumberOfDiscs[iDir];
    
    for (int iDisc = 0; iDisc < mNumberOfDiscs[iDir]; iDisc++) {
      TGeoVolume* ft3V = gGeoManager->GetVolume(getFT3VolPattern());
      if (ft3V == nullptr) {
        LOG(fatal) << getName() << " volume " << getFT3VolPattern() << " is not in the geometry";
      }
      auto layerNode = ft3V->GetNode(Form("%s_1", composeSymNameLayer(iDir,iDisc)));
      if (layerNode == nullptr) LOG(fatal) << "Could not find layer node " << Form("%s_1", composeSymNameLayer(iDir,iDisc));
      auto layerVol = layerNode->GetVolume();
      if (layerVol == nullptr) LOG(fatal) << "Could not find layer volume " << Form("%s_1",composeSymNameLayer(iDir,iDisc));
      TObjArray* nodes = layerVol->GetNodes();
      int nNodes = nodes->GetEntriesFast();
      int nStaves = 0;
      int nSensor = 0;
      std::vector<int> chipsPerStave;
      for (int j = 0; j < nNodes; j++) {
        auto nd = dynamic_cast<TGeoNode*>(nodes->At(j));
        const char* name = nd->GetName();
        if (strstr(name, "FT3Sensor") != nullptr && strstr(name, "Inactive") == nullptr) {
          int direction = 0, layer = 0;
          int stave = 0, chip = 0;
          extractChipIds(name, direction, layer, stave, chip);
          if (stave >= chipsPerStave.size()) {
            chipsPerStave.resize(stave + 1, 0);
            nStaves = stave + 1;
          }
          if (chip + 1 >= chipsPerStave[stave]) {
            chipsPerStave[stave] = chip + 1;
          }
          nSensor++;
        }
      }
      LOG(info) << "direction " << iDir << " disc " << iDisc << " has " << nNodes << " nodes of which " << nSensor << " sensors in " << chipsPerStave.size() << " staves";

      if (nStaves != chipsPerStave.size())
        LOG(info) << "Inconsistency in stave count " << nStaves << " " << chipsPerStave.size();
      mChipIdxStave.resize(absStaveIdx + chipsPerStave.size() + 1);
      mNumberOfStavesPerDisc.push_back(chipsPerStave.size()); // TODO: remove this? Or remove StaveIdxDisc
      int totSensor = 0;
      for (int nChips: chipsPerStave) {
        LOG(debug) << "Absolute Stave ID " << absStaveIdx << " : " << nChips << " sensors";
        totSensor += nChips;
        if (absStaveIdx) mChipIdxStave[absStaveIdx + 1] = mChipIdxStave[absStaveIdx] + nChips;
        absStaveIdx++;
      }
      if (totSensor != nSensor) LOG(info) << "Inconsistency in sensor count " << nSensor << " " << totSensor;
      LOG(debug) << " adding stave Idx " << absStaveIdx << " to disc array; element " << mStaveIdxDisc.size();
      mStaveIdxDisc.push_back(absStaveIdx);
      mNumberOfChipsPerDisc.push_back(totSensor);
      mSize += totSensor;
      LOG(info) << "Total sensors so far " << mSize;
    }
  }
  //mSize = mChipStaveIds.size();
  LOG(info) << "Total sensors " << mSize;
  LOG(info) << "Length of stave-disc array " << mStaveIdxDisc.size();
  fillMatrixCache(loadTrans); // Check whether this causes trouble
}

//__________________________________________________________________________
const char* GeometryTGeo::composeSymNameLayer(int direction, int layerNumber)
{
  return Form("%s%d_%d",GeometryTGeo::getFT3LayerPattern(), direction, layerNumber);
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
int GeometryTGeo::extractNumberOfDiscs(int dir) {
  int numDiscs = 0;
  while (gGeoManager->GetVolume(composeSymNameLayer(dir,numDiscs))) {numDiscs++;} // Check maybe subvolume?
  return numDiscs; // Assume same # layers on both sides
}
//__________________________________________________________________________
int GeometryTGeo::extractNumberOfChips(int dir, int layer) {
  int numSensors = 0;
  TGeoVolume* ft3V = gGeoManager->GetVolume(getFT3VolPattern());
  if (ft3V == nullptr) {
    LOG(fatal) << getName() << " volume " << getFT3VolPattern() << " is not in the geometry";
  }
  auto layerVol = ft3V->GetNode(Form("%s_1",composeSymNameLayer(dir,layer)))->GetVolume();
  TObjArray* nodes = layerVol->GetNodes();
  int nNodes = nodes->GetEntriesFast();
  int nSensor = 0;
  for (int j = 0; j < nNodes; j++) {
    auto nd = dynamic_cast<TGeoNode*>(nodes->At(j));
    const char* name = nd->GetName();
    if (strstr(name, "FT3Sensor") != nullptr && strstr(name, "Inactive") == nullptr) {
      nSensor++;
    }
  }
  LOG(info) << "direction " << dir << " layer " << layer << " has " << nNodes << " nodes of which " << nSensor << " sensors";
  return nSensor;
}
//__________________________________________________________________________
int GeometryTGeo::extractChipId(std::string const volName) {
  if (volName.find("FT3Sensor_Active")==0) {
    return std::stoi(volName.substr(volName.rfind('_')+1));
  }
  LOG(error) << "Not a sensor volume " << volName;
  return -1;
}
void GeometryTGeo::extractStaveChipId(std::string const volName, int &stave, int &chip) {
  if (volName.find("FT3Sensor_Active")==0) {
    int idx = volName.rfind('_');
    chip = std::stoi(volName.substr(idx+1));
    idx = volName.rfind('_',idx);
    stave = std::stoi(volName.substr(idx+1));
  }
  else {
    LOG(error) << "Not a sensor volume " << volName;
    stave = -1; chip = -1;
  }
}
void GeometryTGeo::extractChipIds(std::string const volName, int &direction, int &layer, int &stave, int &chip) {
  if (volName.find("FT3Sensor_Active")==0) {
    int idx = volName.find('_') + 1;
    idx = volName.find('_', idx) + 1;
    direction = std::stoi(volName.substr(idx));
    idx = volName.find('_', idx) + 1;
    layer = std::stoi(volName.substr(idx));
    idx = volName.find('_', idx) + 1;
    stave = std::stoi(volName.substr(idx));
    idx = volName.find('_', idx) + 1;
    chip = std::stoi(volName.substr(idx));
  }
  else {
    LOG(error) << "Not a sensor volume " << volName;
    direction = -1;
  }
}

int GeometryTGeo::getChipIndex(int dir, int layer, int stave, int chip) const {
  int absDisc = layer;
  if (dir == 1) absDisc += mNumberOfDiscs[0];
  return mChipIdxStave[mStaveIdxDisc[absDisc] + stave] + chip;
}

int GeometryTGeo::getLayer(int chipIdx) const {
  int lay = mNumberOfDiscs[0] + mNumberOfDiscs[1] - 1;
  while (chipIdx < mChipIdxStave[mStaveIdxDisc[lay]] && lay > 0) {
   lay--;
  }
  return lay;
}
 

// retrieve local stave number from chip ID
int GeometryTGeo::getStave(int chipIdx) const {
  int lay = getLayer(chipIdx);
  int absStave = mStaveIdxDisc[lay];
  while (chipIdx >= mChipIdxStave[absStave] && absStave < mStaveIdxDisc[lay + 1]) {
    absStave++;
  }
  return absStave - 1 - mStaveIdxDisc[lay];
}

// retrieve local chip number on stave from chip ID
int GeometryTGeo::getChipOnStave(int chipIdx) const {
  int lay = getLayer(chipIdx);
  int stave = getStave(chipIdx);
  return chipIdx - mChipIdxStave[mStaveIdxDisc[lay] + stave];
}

std::string GeometryTGeo::getMatrixPath(int direction, int layer, int stave, int chip) const
{
 
  // PrintChipID(index, subDetID, petalcase, disk, layer, stave, halfstave, mod, chip);

  std::string path = Form("/cave_1/barrel_1/%s_2/", GeometryTGeo::getFT3VolPattern());

  // Stave name: std::string stave_volume_name =
  //    "Stave_" + std::to_string(i_stave) + "_" + std::to_string(layerNumber) +
  //    "_" + std::to_string(direction);
  // Sensors directly placed in layer volume?

  path += Form("%s%d_%d_1/", getFT3LayerPattern(), direction, layer); // TRKLayerx_1
  //std::string sensorName = std::string("FT3Sensor_") + std::to_string(layer) + "_" + std::to_string(direction) + "_" + std::to_string(mChipStaveIds[index]) + "_" + index;
  path += Form("FT3Sensor_Active_%d_%d_%d_%d_%d", direction, layer, stave, chip, chip);
  /*
  if (mLayoutMLOT == FT3Layout::kCylindrical) {
    // TODO: fix this caser?
    path += Form("%s%d_1/", getTRKSensorPattern(), layer); // TRKSensorx_1
  } else {
    path += Form("%s%d_%d/", getFT3StavePattern(), layer, stave); 
    path += Form("%s%d_%d/", getFT3ModulePattern(), layer, mod);
    path += Form("%s%d_%d_1", getFT3ChipPattern(), layer, chipID);
  }
  */
  return path;
}

//__________________________________________________________________________
void GeometryTGeo::fillMatrixCache(int mask)
{
  // populate matrix cache for requested transformations
  //
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
    auto& cacheT2L = getCacheT2L();
    cacheT2L.setSize(mSize);
    mCacheRefAlphaDiscs.resize(mSize, 0);

    double locA[3] = {-100., 0., 0.}, locB[3] = {100., 0., 0.}, gloA[3], gloB[3];
    double xp{0}, yp{0};

    gGeoManager->PushPath();
    LOG(info) << " Number of directions " << mNumberOfDiscs.size();
    int nTotDisc = mNumberOfDiscs[0] + mNumberOfDiscs[1];
    for (int absDisc = 0; absDisc < nTotDisc; absDisc++) {
      int direction = 0;
      int layer = absDisc;
      if (absDisc >= mNumberOfDiscs[0]) {
        direction = 1;
        layer = absDisc - mNumberOfDiscs[0];
      }
      LOG(info) << "Direction " << direction << " layer " << layer;
      if (absDisc >= mNumberOfStavesPerDisc.size()) LOG(fatal) << "Not enough entries in mNumberOfStavesPerDisc " << absDisc << " " << mNumberOfStavesPerDisc.size();
      for (int stave = 0; stave < mNumberOfStavesPerDisc[absDisc]; stave++ ) {
        int absStave = mStaveIdxDisc[absDisc] + stave;
        if (absStave + 1 >= mChipIdxStave.size())
          LOG(fatal) << "Attempting to get absStave + 1 from index array size " << mChipIdxStave.size();  
        int nChip = mChipIdxStave[absStave + 1] - mChipIdxStave[absStave]; // TODO: this is too often == 0
        LOG(debug) << "Getting matrices for direction " << direction << " layer " << layer << " stave " << stave << " : " << nChip << " chips";
        for (int chip = 0; chip < nChip; chip++ ) {
          int chipIdx = getChipIndex(direction, layer, stave, chip);  
          if (!gGeoManager->cd(getMatrixPath(direction, layer, stave, chip).c_str()))
            LOG(fatal) << "Geometry path not found " << getMatrixPath(direction, layer, stave, chip);
          const TGeoHMatrix* matL2G = gGeoManager->GetCurrentMatrix();
          if (chipIdx >= mSize)
            LOG(fatal) << "ChipIdx " << chipIdx << " out of range " << mSize; 
          cacheL2G.setMatrix(Mat3D(*matL2G), chipIdx);

          matL2G->LocalToMaster(locA, gloA);
          matL2G->LocalToMaster(locB, gloB);
          double dx = gloB[0] - gloA[0], dy = gloB[1] - gloA[1];
          double t = (gloB[0] * dx + gloB[1] * dy) / (dx * dx + dy * dy);
          xp = gloB[0] - dx * t;
          yp = gloB[1] - dy * t;
          float alp = std::atan2(yp, xp);
          mCacheRefXDiscs.push_back(std::hypot(xp, yp));
          o2::math_utils::bringTo02Pi(alp);
          mCacheRefAlphaDiscs[chipIdx] = alp;

          static TGeoHMatrix t2l;
          t2l.Clear();
          t2l.RotateZ(mCacheRefAlphaDiscs[chipIdx] * TMath::RadToDeg());  // TODO: do we need this cache?
          const TGeoHMatrix& matL2Gi = matL2G->Inverse();
          t2l.MultiplyLeft(&matL2Gi);
          cacheT2L.setMatrix(Mat3D(t2l),chipIdx); // TODO: may need deref with *
        }
      }
    }
    gGeoManager->PopPath();
  }
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
  LOGF(info, "Number of disks: %d + %d", mNumberOfDiscs[0], mNumberOfDiscs[1]);
  LOGF(info, "Total number of sensors: %d", mSize);
}

}
}