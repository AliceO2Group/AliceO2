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

// ATTENTION: In opposite to old AliITSgeomTGeo, all indices start from 0, not from 1!!!

#include <fairlogger/Logger.h> // for LOG
#include "ITSBase/GeometryTGeo.h"
#include "DetectorsBase/GeometryManager.h"
#include "ITSMFTBase/SegmentationAlpide.h"
#include "MathUtils/Cartesian.h"

#ifdef ENABLE_UPGRADES
#include "ITS3Base/SpecsV2.h"
#include "ITS3Base/SegmentationSuperAlpide.h"
using SuperSegmentation = o2::its3::SegmentationSuperAlpide;
#endif

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
#include <algorithm>

using namespace TMath;
using namespace o2::its;
using namespace o2::detectors;

using Segmentation = o2::itsmft::SegmentationAlpide;

ClassImp(o2::its::GeometryTGeo);

std::unique_ptr<o2::its::GeometryTGeo> GeometryTGeo::sInstance;
o2::its::GeometryTGeo::~GeometryTGeo()
{
  if (!mOwner) {
    mOwner = true;
    sInstance.release();
  }
}

std::string GeometryTGeo::sVolumeName = "ITSV";               ///< Mother volume name
std::string GeometryTGeo::sLayerName = "ITSULayer";           ///< Layer name
std::string GeometryTGeo::sHalfBarrelName = "ITSUHalfBarrel"; ///< HalfBarrel name
std::string GeometryTGeo::sStaveName = "ITSUStave";           ///< Stave name
std::string GeometryTGeo::sHalfStaveName = "ITSUHalfStave";   ///< HalfStave name
std::string GeometryTGeo::sModuleName = "ITSUModule";         ///< Module name
std::string GeometryTGeo::sChipName = "ITSUChip";             ///< Chip name
std::string GeometryTGeo::sSensorName = "ITSUSensor";         ///< Sensor name
std::string GeometryTGeo::sWrapperVolumeName = "ITSUWrapVol"; ///< Wrapper volume name

const std::string GeometryTGeo::sLayerNameITS3 = "ITS3Layer";           ///< Layer name for ITS3
const std::string GeometryTGeo::sHalfBarrelNameITS3 = "ITS3CarbonForm"; ///< HalfBarrel name for ITS3
const std::string GeometryTGeo::sStaveNameITS3 = "ITS3Chip";            ///< Stave name for ITS3
const std::string GeometryTGeo::sHalfStaveNameITS3 = "ITS3Segment";     ///< HalfStave name for ITS3
const std::string GeometryTGeo::sModuleNameITS3 = "ITS3RSU";            ///< Module name for ITS3
const std::string GeometryTGeo::sChipNameITS3 = "ITS3Tile";             ///< Chip name for ITS3
const std::string GeometryTGeo::sSensorNameITS3 = "ITS3PixelArray";     ///< Sensor name for ITS3

//__________________________________________________________________________
GeometryTGeo::GeometryTGeo(bool build, int loadTrans) : o2::itsmft::GeometryTGeo(DetID::ITS)
{
  // default c-tor, if build is true, the structures will be filled and the transform matrices
  // will be cached
  if (sInstance) {
    LOG(fatal) << "Invalid use of public constructor: o2::its::GeometryTGeo instance exists";
    // throw std::runtime_error("Invalid use of public constructor: o2::its::GeometryTGeo instance exists");
  }

  mLayerToWrapper.fill(-1);
  if (build) {
    Build(loadTrans);
  }
}

//__________________________________________________________________________
void GeometryTGeo::adopt(GeometryTGeo* raw, bool canDelete)
{
  // adopt the unique instance from external raw pointer (to be used only to read saved instance from file)
  if (sInstance) {
    LOG(fatal) << "No adoption: o2::its::GeometryTGeo instance exists";
  }
  sInstance = std::unique_ptr<o2::its::GeometryTGeo>(raw);
  sInstance->mOwner = canDelete;
}

//__________________________________________________________________________
int GeometryTGeo::getChipIndex(int lay, int hba, int sta, int chipInStave) const
{
  return getFirstChipIndex(lay) + mNumberOfChipsPerHalfBarrel[lay] * hba + mNumberOfChipsPerStave[lay] * sta + chipInStave;
}

//__________________________________________________________________________
int GeometryTGeo::getChipIndex(int lay, int hba, int sta, int substa, int chipInSStave) const
{
  int n = getFirstChipIndex(lay) + mNumberOfChipsPerHalfBarrel[lay] * hba + mNumberOfChipsPerStave[lay] * sta + chipInSStave;
  if (mNumberOfHalfStaves[lay] && substa > 0) {
    n += mNumberOfChipsPerHalfStave[lay] * substa;
  }
  return n;
}

//__________________________________________________________________________
int GeometryTGeo::getChipIndex(int lay, int hba, int sta, int substa, int md, int chipInMod) const
{
  if (mNumberOfHalfStaves[lay] == 0) {
    return getChipIndex(lay, substa, md, chipInMod);
  } else {
    int n = getFirstChipIndex(lay) + mNumberOfChipsPerHalfBarrel[lay] * hba + mNumberOfChipsPerStave[lay] * sta + chipInMod;
    if (mNumberOfHalfStaves[lay] && substa > 0) {
      n += mNumberOfChipsPerHalfStave[lay] * substa;
    }
    if (mNumberOfModules[lay] && md > 0) {
      n += mNumberOfChipsPerModule[lay] * md;
    }
    return n;
  }
}

//__________________________________________________________________________
bool GeometryTGeo::getLayer(int index, int& lay, int& indexInLr) const
{
  lay = getLayer(index);
  indexInLr = index - getFirstChipIndex(lay);
  return kTRUE;
}

//__________________________________________________________________________
int GeometryTGeo::getLayer(int index) const
{
  int lay = 0;
  while (index > mLastChipIndex[lay]) {
    lay++;
  }
  return lay;
}

//__________________________________________________________________________
int GeometryTGeo::getHalfBarrel(int index) const
{
  int lay = 0;
  while (index > mLastChipIndex[lay]) {
    lay++;
  }
  index -= getFirstChipIndex(lay);
  return index / mNumberOfChipsPerHalfBarrel[lay];
}

//__________________________________________________________________________
int GeometryTGeo::getStave(int index) const
{
  int lay = 0;
  while (index > mLastChipIndex[lay]) {
    lay++;
  }
  index -= getFirstChipIndex(lay);
  return index / mNumberOfChipsPerStave[lay];
}

//__________________________________________________________________________
int GeometryTGeo::getHalfStave(int index) const
{
  int lay = 0;
  while (index > mLastChipIndex[lay]) {
    lay++;
  }
  if (mNumberOfHalfStaves[lay] < 0) {
    return -1;
  }
  index -= getFirstChipIndex(lay);
  index %= mNumberOfChipsPerStave[lay];
  return index / mNumberOfChipsPerHalfStave[lay];
}

//__________________________________________________________________________
int GeometryTGeo::getModule(int index) const
{
  int lay = 0;
  while (index > mLastChipIndex[lay]) {
    lay++;
  }
  if (mNumberOfModules[lay] < 0) {
    return 0;
  }
  index -= getFirstChipIndex(lay);
  index %= mNumberOfChipsPerStave[lay];
  if (mNumberOfHalfStaves[lay] != 0) {
    index %= mNumberOfChipsPerHalfStave[lay];
  }
  return index / mNumberOfChipsPerModule[lay];
}

//__________________________________________________________________________
int GeometryTGeo::getChipIdInLayer(int index) const
{
  int lay = 0;
  while (index > mLastChipIndex[lay]) {
    lay++;
  }
  index -= getFirstChipIndex(lay);
  return index;
}

//__________________________________________________________________________
int GeometryTGeo::getChipIdInStave(int index) const
{
  int lay = 0;
  while (index > mLastChipIndex[lay]) {
    lay++;
  }
  index -= getFirstChipIndex(lay);
  return index % mNumberOfChipsPerStave[lay];
}

//__________________________________________________________________________
int GeometryTGeo::getChipIdInHalfStave(int index) const
{
  int lay = 0;
  while (index > mLastChipIndex[lay]) {
    lay++;
  }
  index -= getFirstChipIndex(lay);
  return index % mNumberOfChipsPerHalfStave[lay];
}

//__________________________________________________________________________
int GeometryTGeo::getChipIdInModule(int index) const
{
  int lay = 0;
  while (index > mLastChipIndex[lay]) {
    lay++;
  }
  index -= getFirstChipIndex(lay);
  return index % mNumberOfChipsPerModule[lay];
}

//__________________________________________________________________________
bool GeometryTGeo::getChipId(int index, int& lay, int& sta, int& hsta, int& mod, int& chip) const
{
  lay = getLayer(index);
  index -= getFirstChipIndex(lay);
  sta = index / mNumberOfChipsPerStave[lay];
  index %= mNumberOfChipsPerStave[lay];
  hsta = mNumberOfHalfStaves[lay] > 0 ? index / mNumberOfChipsPerHalfStave[lay] : -1;
  index %= mNumberOfChipsPerHalfStave[lay];
  mod = mNumberOfModules[lay] > 0 ? index / mNumberOfChipsPerModule[lay] : -1;
  chip = index % mNumberOfChipsPerModule[lay];

  return kTRUE;
}

//__________________________________________________________________________
bool GeometryTGeo::getChipId(int index, int& lay, int& hba, int& sta, int& hsta, int& mod, int& chip) const
{
  lay = getLayer(index);
  index -= getFirstChipIndex(lay);
  hba = mNumberOfHalfBarrels > 0 ? index / mNumberOfChipsPerHalfBarrel[lay] : -1;
  index %= mNumberOfChipsPerHalfBarrel[lay];
  sta = index / mNumberOfChipsPerStave[lay];
  index %= mNumberOfChipsPerStave[lay];
  hsta = mNumberOfHalfStaves[lay] > 0 ? index / mNumberOfChipsPerHalfStave[lay] : -1;
  index %= mNumberOfChipsPerHalfStave[lay];
  mod = mNumberOfModules[lay] > 0 ? index / mNumberOfChipsPerModule[lay] : -1;
  chip = index % mNumberOfChipsPerModule[lay];

  return kTRUE;
}

//__________________________________________________________________________
const char* GeometryTGeo::composeSymNameITS(bool isITS3)
{
  if (isITS3) {
#ifdef ENABLE_UPGRADES
    return o2::detectors::DetID(o2::detectors::DetID::IT3).getName();
#endif
  }

  return o2::detectors::DetID(o2::detectors::DetID::ITS).getName();
}

//__________________________________________________________________________
const char* GeometryTGeo::composeSymNameLayer(int lr, bool isITS3)
{
  return Form("%s/%s%d", composeSymNameITS(), isITS3 ? getITS3LayerPattern() : getITSLayerPattern(), lr);
}

//__________________________________________________________________________
const char* GeometryTGeo::composeSymNameHalfBarrel(int lr, int hbarrel, bool isITS3)
{
  return hbarrel >= 0 ? Form("%s/%s%d", composeSymNameLayer(lr, isITS3), isITS3 ? getITS3HalfBarrelPattern() : getITSHalfBarrelPattern(), hbarrel)
                      : composeSymNameLayer(lr);
}

//__________________________________________________________________________
const char* GeometryTGeo::composeSymNameStave(int lr, int hbarrel, int stave, bool isITS3)
{
  return Form("%s/%s%d", composeSymNameHalfBarrel(lr, hbarrel, isITS3), isITS3 ? getITS3StavePattern() : getITSStavePattern(), stave);
}

//__________________________________________________________________________
const char* GeometryTGeo::composeSymNameHalfStave(int lr, int hba, int stave, int substave, bool isITS3)
{
  return substave >= 0 ? Form("%s/%s%d", composeSymNameStave(lr, hba, stave, isITS3), isITS3 ? getITS3HalfStavePattern() : getITSHalfStavePattern(), substave)
                       : composeSymNameStave(lr, hba, stave, isITS3);
}

//__________________________________________________________________________
const char* GeometryTGeo::composeSymNameModule(int lr, int hba, int stave, int substave, int mod, bool isITS3)
{
  return mod >= 0 ? Form("%s/%s%d", composeSymNameHalfStave(lr, hba, stave, substave, isITS3), isITS3 ? getITS3ModulePattern() : getITSModulePattern(), mod)
                  : composeSymNameHalfStave(lr, hba, stave, substave, isITS3);
}

//__________________________________________________________________________
const char* GeometryTGeo::composeSymNameChip(int lr, int hba, int sta, int substave, int mod, int chip, bool isITS3)
{
  return Form("%s/%s%d", composeSymNameModule(lr, hba, sta, substave, mod, isITS3), isITS3 ? getITS3ChipPattern() : getITSChipPattern(), chip);
}

TString GeometryTGeo::getMatrixPath(int index) const
{
  int lay, hba, stav, sstav, mod, chipInMod;
  getChipId(index, lay, hba, stav, sstav, mod, chipInMod);

  int wrID = mLayerToWrapper[lay];

  TString path = Form("/cave_1/barrel_1/%s_2/", GeometryTGeo::getITSVolPattern());

  if (wrID >= 0) {
    path += Form("%s%d_1/", getITSWrapVolPattern(), wrID);
  }

  if (!mIsLayerITS3[lay]) {
    path +=
      Form("%s%d_1/", getITSLayerPattern(), lay);
    if (mNumberOfHalfBarrels > 0) {
      path += Form("%s%d_%d/", getITSHalfBarrelPattern(), lay, hba);
    }
    path +=
      Form("%s%d_%d/", getITSStavePattern(), lay, stav);

    if (mNumberOfHalfStaves[lay] > 0) {
      path += Form("%s%d_%d/", getITSHalfStavePattern(), lay, sstav);
    }
    if (mNumberOfModules[lay] > 0) {
      path += Form("%s%d_%d/", getITSModulePattern(), lay, mod);
    }
    path += Form("%s%d_%d/%s%d_1", getITSChipPattern(), lay, chipInMod, getITSSensorPattern(), lay);
  } else {
    // hba = carbonform
    // stav = 0
    // sstav = segment
    // mod = rsu
    // chipInMod = tile
    // sensor = pixelarray
    path += Form("%s_0/", getITS3LayerPattern(lay));
    path += Form("%s_%d/", getITS3CarbonFormPattern(lay), hba);
    path += Form("%s_0/", getITS3ChipPattern(lay));
    path += Form("%s_%d/", getITS3SegmentPattern(lay), sstav);
    path += Form("%s_%d/", getITS3RSUPattern(lay), mod);
    path += Form("%s_%d/", getITS3TilePattern(lay), chipInMod);
    path += Form("%s_0", getITS3PixelArrayPattern(lay));
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
  //
  // Therefore we need to add a shift
  auto path = getMatrixPath(index);

  static TGeoHMatrix matTmp;
  gGeoManager->PushPath();

  if (!gGeoManager->cd(path.Data())) {
    gGeoManager->PopPath();
    LOG(error) << "Error in cd-ing to " << path.Data();
    return nullptr;
  } // end if !gGeoManager

  matTmp = *gGeoManager->GetCurrentMatrix(); // matrix may change after cd

  // RSS
  // printf("%d/%d/%d %s\n", lay, stav, detInSta, path.Data());
  // matTmp.Print();
  // Restore the modeler state.
  gGeoManager->PopPath();

  static int chipInGlo{0};

  // account for the difference between physical sensitive layer (where charge collection is simulated) and effective sensor thicknesses
  double delta = Segmentation::SensorLayerThickness - Segmentation::SensorLayerThicknessEff;
#ifdef ENABLE_UPGRADES
  if (mIsLayerITS3[getLayer(index)]) {
    delta = its3::SegmentationSuperAlpide::mSensorLayerThickness - its3::SegmentationSuperAlpide::mSensorLayerThicknessEff;
  }
#endif

  static TGeoTranslation tra(0., 0.5 * delta, 0.);

  matTmp *= tra;

  return &matTmp;
}

//__________________________________________________________________________
float GeometryTGeo::getAlphaFromGlobalITS3(const o2::math_utils::Point3D<float>& gloXYZ)
{
  // since ITS3 is cylindrical alpha = phi in global
  auto alpha = std::atan2(gloXYZ.Y(), gloXYZ.X());
  o2::math_utils::bringTo02Pi(alpha);
  return alpha;
}

//__________________________________________________________________________
const o2::math_utils::Transform3D GeometryTGeo::getT2LMatrixITS3(int isn, float alpha)
{
  // create for sensor isn the TGeo matrix for Tracking to Local frame transformations
  static TGeoHMatrix t2l;
  t2l.Clear();
  t2l.RotateZ(alpha * RadToDeg()); // rotate in direction of normal to the tangent to the cylinder
  const TGeoHMatrix& matL2G = getMatrixL2G(isn);
  const auto& matL2Gi = matL2G.Inverse();
  t2l.MultiplyLeft(&matL2Gi);
  return Mat3D(t2l);
}

//__________________________________________________________________________
void GeometryTGeo::Build(int loadTrans)
{
  if (isBuilt()) {
    LOG(warning) << "Already built";
    return; // already initialized
  }

  if (gGeoManager == nullptr) {
    // RSTODO: in future there will be a method to load matrices from the CDB
    LOG(fatal) << "Geometry is not loaded";
  }

  mIsLayerITS3.fill(false);
  mNumberOfLayers = extractNumberOfLayers();
  if (mNumberOfLayers == 0) {
    return;
  }

  mNumberOfStaves.resize(mNumberOfLayers);
  mNumberOfHalfStaves.resize(mNumberOfLayers);
  mNumberOfModules.resize(mNumberOfLayers);
  mNumberOfChipsPerModule.resize(mNumberOfLayers);
  mNumberOfChipRowsPerModule.resize(mNumberOfLayers);
  mNumberOfChipsPerHalfStave.resize(mNumberOfLayers);
  mNumberOfChipsPerStave.resize(mNumberOfLayers);
  mNumberOfChipsPerHalfBarrel.resize(mNumberOfLayers);
  mNumberOfChipsPerLayer.resize(mNumberOfLayers);
  mLastChipIndex.resize(mNumberOfLayers);
  int numberOfChips = 0;

  mNumberOfHalfBarrels = extractNumberOfHalfBarrels();
  for (int i = 0; i < mNumberOfLayers; i++) {
    mNumberOfStaves[i] = extractNumberOfStaves(i);
    mNumberOfHalfStaves[i] = extractNumberOfHalfStaves(i);
    mNumberOfModules[i] = extractNumberOfModules(i);
    mNumberOfChipsPerModule[i] = extractNumberOfChipsPerModule(i, mNumberOfChipRowsPerModule[i]);
    mNumberOfChipsPerHalfStave[i] = mNumberOfChipsPerModule[i] * Max(1, mNumberOfModules[i]);
    mNumberOfChipsPerStave[i] = mNumberOfChipsPerHalfStave[i] * Max(1, mNumberOfHalfStaves[i]);
    mNumberOfChipsPerLayer[i] = mNumberOfChipsPerStave[i] * mNumberOfStaves[i];
    mNumberOfChipsPerHalfBarrel[i] = mNumberOfChipsPerLayer[i] / Max(1, mNumberOfHalfBarrels);
    numberOfChips += mNumberOfChipsPerLayer[i];
    mLastChipIndex[i] = numberOfChips - 1;
  }

  LOGP(debug, "Summary of extracted Geometry:");
  LOGP(debug, "  There are {} Layers and {} HalfBarrels", mNumberOfLayers, mNumberOfHalfBarrels);
  for (int i = 0; i < mNumberOfLayers; i++) {
    LOGP(debug, "    Layer {}: {:*^30}", i, "START");
    LOGP(debug, "      - mNumberOfStaves={}", mNumberOfStaves[i]);
    LOGP(debug, "        - mNumberOfChipsPerStave={}", mNumberOfChipsPerStave[i]);
    LOGP(debug, "      - mNumberOfHalfStaves={}", mNumberOfHalfStaves[i]);
    LOGP(debug, "        - mNumberOfChipsPerHalfStave={}", mNumberOfChipsPerHalfStave[i]);
    LOGP(debug, "      - mNumberOfModules={}", mNumberOfModules[i]);
    LOGP(debug, "        - mNumberOfChipsPerModules={}", mNumberOfChipsPerModule[i]);
    LOGP(debug, "        - mNumberOfChipsPerLayer={}", mNumberOfChipsPerLayer[i]);
    LOGP(debug, "        - mNumberOfChipsPerHalfBarrel={}", mNumberOfChipsPerHalfBarrel[i]);
    LOGP(debug, "      - mLastChipIndex={}", mLastChipIndex[i]);
    LOGP(debug, "    Layer {}: {:*^30}", i, "END");
  }
  LOGP(debug, "In total there {} chips registered", numberOfChips);

#ifdef ENABLE_UPGRADES
  if (std::any_of(mIsLayerITS3.cbegin(), mIsLayerITS3.cend(), [](auto b) { return b; })) {
    LOGP(info, "Found active IT3 layers -> Renaming Detector ITS to IT3");
    mDetID = DetID::IT3;
  }
#endif

  setSize(numberOfChips);
  fillTrackingFramesCache();
  fillMatrixCache(loadTrans);
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

    for (int i = 0; i < mSize; i++) {
      TGeoHMatrix* hm = extractMatrixSensor(i);
      cacheL2G.setMatrix(Mat3D(*hm), i);
    }
  }

  if ((mask & o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L)) && !getCacheT2L().isFilled()) {
    // matrices for Tracking to Local (Sensor!!! rather than the full chip) frame transformation
    LOGP(info, "Loading {} T2L matrices from TGeo", getName());
    auto& cacheT2L = getCacheT2L();
    cacheT2L.setSize(mSize);
    for (int i = 0; i < mSize; i++) {
      TGeoHMatrix& hm = createT2LMatrix(i);
      cacheT2L.setMatrix(Mat3D(hm), i);
    }
  }

  if ((mask & o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2GRot)) && !getCacheT2GRot().isFilled()) {
    // 2D rotation matrices for Tracking frame to Global rotations
    LOGP(info, "Loading {} T2G rotation 2D matrices", getName());
    auto& cacheT2Gr = getCacheT2GRot();
    cacheT2Gr.setSize(mSize);
    for (int i = 0; i < mSize; i++) {
      cacheT2Gr.setMatrix(Rot2D(getSensorRefAlpha(i)), i);
    }
  }

  if ((mask & o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2G)) && !getCacheT2G().isFilled()) {
    LOG(debug) << "It is faster to use 2D rotation for T2G instead of full Transform3D matrices";
    // matrices for Tracking to Global frame transformation
    LOGP(info, "Creating {} T2G matrices from TGeo", getName());
    auto& cacheT2G = getCacheT2G();
    cacheT2G.setSize(mSize);

    for (int i = 0; i < mSize; i++) {
      /*
  TGeoHMatrix& mat = createT2LMatrix(i);
  mat.MultiplyLeft(extractMatrixSensor(i));
      */
      Rot2D r(getSensorRefAlpha(i));
      Mat3D mat{};
      mat.SetComponents(r.getCos(), -r.getSin(), 0., 0., r.getSin(), r.getCos(), 0., 0., 0., 0., 1., 0.);
      cacheT2G.setMatrix(mat, i);
    }
  }
}

//__________________________________________________________________________
void GeometryTGeo::fillTrackingFramesCache()
{
  // fill for every sensor its tracking frame parameteres
  if (!isTrackingFrameCached()) {
    // special cache for sensors tracking frame X and alpha params
    mCacheRefX.resize(mSize);
    mCacheRefAlpha.resize(mSize);
    for (int i = 0; i < mSize; i++) {
      extractSensorXAlpha(i, mCacheRefX[i], mCacheRefAlpha[i]);
    }
  }
}

//__________________________________________________________________________
int GeometryTGeo::extractNumberOfLayers()
{
  int numberOfLayers = 0;

  TGeoVolume* itsV = gGeoManager->GetVolume(getITSVolPattern());
  if (itsV == nullptr) {
    LOG(fatal) << getName() << " volume " << getITSVolPattern() << " is not in the geometry";
  }

  // Loop on all ITSV nodes, count Layer volumes by checking names
  // Build on the fly layer - wrapper correspondence
  TObjArray* nodes = itsV->GetNodes();
  int nNodes = nodes->GetEntriesFast();
  for (int j = 0; j < nNodes; j++) {
    int lrID = -1;
    auto nd = dynamic_cast<TGeoNode*>(nodes->At(j));
    const char* name = nd->GetName();

    if ((strstr(name, getITSLayerPattern()) != nullptr) || (strstr(name, getITS3LayerPattern()) != nullptr)) {
      numberOfLayers++;
      if ((lrID = extractVolumeCopy(name, GeometryTGeo::getITSLayerPattern())) < 0) {
        if ((lrID = extractVolumeCopy(name, GeometryTGeo::getITS3LayerPattern())) < 0) {
          LOG(fatal) << "Failed to extract layer ID from the " << name;
        }
        mIsLayerITS3[lrID] = true;
      }
      mLayerToWrapper[lrID] = -1;                                 // not wrapped
    } else if (strstr(name, getITSWrapVolPattern()) != nullptr) { // this is a wrapper volume, may cointain layers
      int wrID = -1;
      if ((wrID = extractVolumeCopy(name, GeometryTGeo::getITSWrapVolPattern())) < 0) {
        LOG(fatal) << "Failed to extract wrapper ID from the " << name;
      }
      TObjArray* nodesW = nd->GetNodes();
      int nNodesW = nodesW->GetEntriesFast();

      for (int jw = 0; jw < nNodesW; jw++) {
        auto ndW = dynamic_cast<TGeoNode*>(nodesW->At(jw))->GetName();
        if ((strstr(ndW, getITSLayerPattern()) != nullptr) || (strstr(ndW, getITS3LayerPattern()) != nullptr)) {
          if ((lrID = extractVolumeCopy(ndW, GeometryTGeo::getITSLayerPattern())) < 0) {
            if ((lrID = extractVolumeCopy(ndW, GeometryTGeo::getITS3LayerPattern())) < 0) {
              LOGP(fatal, "Failed to extract layer ID from wrapper volume '{}' from one of its nodes '{}'", name, ndW);
            }
            mIsLayerITS3[lrID] = true;
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
int GeometryTGeo::extractNumberOfHalfBarrels() const
{
  // We take in account that we always have 2 and only 2 half barrels
  int numberOfHalfBarrels = 2;

  return numberOfHalfBarrels;
}

//__________________________________________________________________________
int GeometryTGeo::extractNumberOfStaves(int lay) const
{
  int numberOfStaves = 0;
  char hbarnam[30];
  if (mNumberOfHalfBarrels == 0) {
    snprintf(hbarnam, 30, "%s%d", mIsLayerITS3[lay] ? getITS3LayerPattern() : getITSLayerPattern(), lay);
  } else {
    snprintf(hbarnam, 30, "%s%d", mIsLayerITS3[lay] ? getITS3HalfBarrelPattern() : getITSHalfBarrelPattern(), lay);
  }
  TGeoVolume* volHb = gGeoManager->GetVolume(hbarnam);
  if (volHb == nullptr) {
    LOGP(fatal, "Can't find '{}' volume (ITS3={})", hbarnam, mIsLayerITS3[lay]);
    return -1;
  }

  // Loop on all half barrel nodes, count Stave volumes by checking names
  int nNodes = volHb->GetNodes()->GetEntries();
  for (int j = 0; j < nNodes; j++) {
    // LOG(info) << "L" << lay << " " << j << " of " << nNodes << " "
    //           << volHb->GetNodes()->At(j)->GetName() << " "
    //           << mIsLayerITS3[lay] ? getITS3StavePattern() : getITSStavePattern() << " -> " << numberOfStaves;
    if (strstr(volHb->GetNodes()->At(j)->GetName(), mIsLayerITS3[lay] ? getITS3StavePattern() : getITSStavePattern()) != nullptr) {
      numberOfStaves++;
    }
  }
  return mNumberOfHalfBarrels > 0 ? (numberOfStaves * mNumberOfHalfBarrels) : numberOfStaves;
}

//__________________________________________________________________________
int GeometryTGeo::extractNumberOfHalfStaves(int lay) const
{
  if (sHalfStaveName.empty()) {
    return 0; // for the setup w/o substave defined the stave and the substave is the same thing
  }
  int nSS = 0;
  char stavnam[30];
  snprintf(stavnam, 30, "%s%d", mIsLayerITS3[lay] ? getITS3StavePattern() : getITSStavePattern(), lay);
  TGeoVolume* volLd = gGeoManager->GetVolume(stavnam);
  if (volLd == nullptr) {
    LOG(fatal) << "can't find volume " << stavnam;
  }
  // Loop on all stave nodes, count Chip volumes by checking names
  int nNodes = volLd->GetNodes()->GetEntries();
  for (int j = 0; j < nNodes; j++) {
    if (strstr(volLd->GetNodes()->At(j)->GetName(), mIsLayerITS3[lay] ? getITS3HalfStavePattern() : getITSHalfStavePattern()) != nullptr) {
      nSS++;
    }
  }
  return nSS;
}

//__________________________________________________________________________
int GeometryTGeo::extractNumberOfModules(int lay) const
{
  if (sModuleName.empty()) {
    return 0;
  }

  char stavnam[30];
  TGeoVolume* volLd = nullptr;

  if (!sHalfStaveName.empty()) {
    snprintf(stavnam, 30, "%s%d", mIsLayerITS3[lay] ? getITS3HalfStavePattern() : getITSHalfStavePattern(), lay);
    volLd = gGeoManager->GetVolume(stavnam);
  }
  if (!volLd) { // no substaves, check staves
    snprintf(stavnam, 30, "%s%d", mIsLayerITS3[lay] ? getITS3StavePattern() : getITSStavePattern(), lay);
    volLd = gGeoManager->GetVolume(stavnam);
  }
  if (!volLd) {
    return 0;
  }

  int nMod = 0;

  // Loop on all substave nodes, count module volumes by checking names
  int nNodes = volLd->GetNodes()->GetEntries();

  for (int j = 0; j < nNodes; j++) {
    if (strstr(volLd->GetNodes()->At(j)->GetName(), mIsLayerITS3[lay] ? getITS3ModulePattern() : getITSModulePattern())) {
      nMod++;
    }
  }
  return nMod;
}

//__________________________________________________________________________
int GeometryTGeo::extractNumberOfChipsPerModule(int lay, int& nrow) const
{
#ifdef ENABLE_UPGRADES
  // FS: TODO
  // For now we hardcode ITS3 number of chips is eq. to the number of tiles per RSU
  // The test in the end does not work for ITS3.
  if (mIsLayerITS3[lay]) {
    nrow = o2::its3::constants::pixelarray::nRows;
    return o2::its3::constants::rsu::nTiles;
  }
#endif
  int numberOfChips = 0;
  char stavnam[30];
  TGeoVolume* volLd = nullptr;

  if (!sModuleName.empty()) {
    snprintf(stavnam, 30, "%s%d", getITSModulePattern(), lay);
    volLd = gGeoManager->GetVolume(stavnam);
  }
  if (!volLd) { // no modules on this layer, check substaves
    if (!sHalfStaveName.empty()) {
      snprintf(stavnam, 30, "%s%d", getITSHalfStavePattern(), lay);
      volLd = gGeoManager->GetVolume(stavnam);
    }
  }
  if (!volLd) { // no substaves on this layer, check staves
    snprintf(stavnam, 30, "%s%d", getITSStavePattern(), lay);
    volLd = gGeoManager->GetVolume(stavnam);
  }
  if (!volLd) {
    LOG(fatal) << "can't find volume containing chips on layer " << lay;
  }

  // Loop on all stave nodes, count Chip volumes by checking names
  int nNodes = volLd->GetNodes()->GetEntries();

  double xmin = 1e9, xmax = -1e9, zmin = 1e9, zmax = -1e9;
  double lab[3], loc[3] = {0, 0, 0};
  double dx = -1, dz = -1;

  for (int j = 0; j < nNodes; j++) {
    TGeoNodeMatrix* node = (TGeoNodeMatrix*)volLd->GetNodes()->At(j);
    if (strstr(node->GetName(), getITSChipPattern()) == nullptr) {
      continue;
    }
    node->LocalToMaster(loc, lab);
    if (lab[0] > xmax) {
      xmax = lab[0];
    }
    if (lab[0] < xmin) {
      xmin = lab[0];
    }
    if (lab[2] > zmax) {
      zmax = lab[2];
    }
    if (lab[2] < zmin) {
      zmin = lab[2];
    }

    numberOfChips++;

    if (dx < 0) {
      TGeoShape* chShape = node->GetVolume()->GetShape();
      TGeoBBox* bbox = dynamic_cast<TGeoBBox*>(chShape);
      if (!bbox) {
        LOG(fatal) << "Chip " << node->GetName() << " volume is of unprocessed shape " << chShape->IsA()->GetName();
      } else {
        dx = 2 * bbox->GetDX();
        dz = 2 * bbox->GetDZ();
      }
    }
  }

  double spanX = xmax - xmin;
  double spanZ = zmax - zmin;
  nrow = TMath::Nint(spanX / dx + 1);
  int ncol = TMath::Nint(spanZ / dz + 1);
  if (nrow * ncol != numberOfChips) {
    LOG(error) << "Inconsistency between Nchips=" << numberOfChips << " and Nrow*Ncol=" << nrow << "*" << ncol << "->"
               << nrow * ncol << "\n"
               << "Extracted chip dimensions (x,z): " << dx << " " << dz << " Module Span: " << spanX << " " << spanZ << "\n"
               << "xmin=" << xmin << "   xmax=" << xmax
               << "   zmin=" << zmin << "   zmax=" << zmax;
  }
  return numberOfChips;
}

//__________________________________________________________________________
int GeometryTGeo::extractLayerChipType(int lay) const
{
  char stavnam[30];
  snprintf(stavnam, 30, "%s%d", mIsLayerITS3[lay] ? getITS3LayerPattern() : getITSLayerPattern(), lay);
  TGeoVolume* volLd = gGeoManager->GetVolume(stavnam);
  if (!volLd) {
    LOG(fatal) << "can't find volume " << stavnam;
    return -1;
  }
  return volLd->GetUniqueID();
}

//__________________________________________________________________________
void GeometryTGeo::Print(Option_t*) const
{
  if (!isBuilt()) {
    LOGF(info, "Geometry not built yet!");
    return;
  }

  LOGF(info, "Summary of GeometryTGeo: %s", getName());
  LOGF(info, "NLayers:%d NChips:%d\n", mNumberOfLayers, getNumberOfChips());
  for (int i = 0; i < mNumberOfLayers; i++) {
    LOGF(info,
         "Lr%2d\tNStav:%2d\tNChips:%2d "
         "(%dx%-2d)\tNMod:%d\tNSubSt:%d\tNSt:%3d\tChip#:%5d:%-5d\tWrapVol:%d",
         i, mNumberOfStaves[i], mNumberOfChipsPerModule[i], mNumberOfChipRowsPerModule[i],
         mNumberOfChipRowsPerModule[i] ? mNumberOfChipsPerModule[i] / mNumberOfChipRowsPerModule[i] : 0,
         mNumberOfModules[i], mNumberOfHalfStaves[i], mNumberOfStaves[i], getFirstChipIndex(i), getLastChipIndex(i),
         mLayerToWrapper[i]);
  }
}

//__________________________________________________________________________
void GeometryTGeo::extractSensorXAlpha(int isn, float& x, float& alp)
{
  // calculate r and phi of the impact of the normal on the sensor
  // (i.e. phi of the tracking frame alpha and X of the sensor in this frame)

  const TGeoHMatrix* matL2G = extractMatrixSensor(isn);
  double locA[3] = {-100., 0., 0.}, locB[3] = {100., 0., 0.}, gloA[3], gloB[3];
  int iLayer = getLayer(isn);

#ifdef ENABLE_UPGRADES
  if (mIsLayerITS3[iLayer]) {
    // in this case we need the line tangent to the circumference
    double radius = 0.;
    radius = o2::its3::constants::radii[iLayer];
    locA[1] = radius;
    locB[1] = radius;
  }
#endif

  matL2G->LocalToMaster(locA, gloA);
  matL2G->LocalToMaster(locB, gloB);
  double dx = gloB[0] - gloA[0], dy = gloB[1] - gloA[1];
  double t = (gloB[0] * dx + gloB[1] * dy) / (dx * dx + dy * dy);
  double xp = gloB[0] - dx * t, yp = gloB[1] - dy * t;
  x = Sqrt(xp * xp + yp * yp);
  alp = ATan2(yp, xp);
  o2::math_utils::bringTo02Pi(alp);
}

//__________________________________________________________________________
TGeoHMatrix& GeometryTGeo::createT2LMatrix(int isn)
{
  // create for sensor isn the TGeo matrix for Tracking to Local frame transformations
  static TGeoHMatrix t2l;
  float x = 0.f, alp = 0.f;
  extractSensorXAlpha(isn, x, alp);
  t2l.Clear();
  t2l.RotateZ(alp * RadToDeg()); // rotate in direction of normal to the sensor plane
  const TGeoHMatrix* matL2G = extractMatrixSensor(isn);
  const TGeoHMatrix& matL2Gi = matL2G->Inverse();
  t2l.MultiplyLeft(&matL2Gi);
  return t2l;
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
