// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GeometryTGeo.cxx
/// \brief Implementation of the GeometryTGeo class
/// \author cvetan.cheshkov@cern.ch - 15/02/2007
/// \author ruben.shahoyan@cern.ch - adapted to ITSupg 18/07/2012

// ATTENTION: In opposite to old AliITSgeomTGeo, all indices start from 0, not from 1!!!

#include "ITSBase/GeometryTGeo.h"
#include "DetectorsBase/GeometryManager.h"
#include "ITSMFTBase/SegmentationAlpide.h"
#include "MathUtils/Cartesian3D.h"

#include "FairLogger.h" // for LOG

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
using namespace o2::its;
using namespace o2::detectors;
using namespace o2::utils;

using Segmentation = o2::itsmft::SegmentationAlpide;

ClassImp(o2::its::GeometryTGeo);

std::unique_ptr<o2::its::GeometryTGeo> GeometryTGeo::sInstance;

std::string GeometryTGeo::sVolumeName = "ITSV";               ///< Mother volume name
std::string GeometryTGeo::sLayerName = "ITSULayer";           ///< Layer name
std::string GeometryTGeo::sStaveName = "ITSUStave";           ///< Stave name
std::string GeometryTGeo::sHalfStaveName = "ITSUHalfStave";   ///< HalfStave name
std::string GeometryTGeo::sModuleName = "ITSUModule";         ///< Module name
std::string GeometryTGeo::sChipName = "ITSUChip";             ///< Chip name
std::string GeometryTGeo::sSensorName = "ITSUSensor";         ///< Sensor name
std::string GeometryTGeo::sWrapperVolumeName = "ITSUWrapVol"; ///< Wrapper volume name

//__________________________________________________________________________
GeometryTGeo::GeometryTGeo(bool build, int loadTrans) : o2::itsmft::GeometryTGeo(DetID::ITS)
{
  // default c-tor, if build is true, the structures will be filled and the transform matrices
  // will be cached
  if (sInstance) {
    LOG(FATAL) << "Invalid use of public constructor: o2::its::GeometryTGeo instance exists";
    // throw std::runtime_error("Invalid use of public constructor: o2::its::GeometryTGeo instance exists");
  }

  for (int i = MAXLAYERS; i--;) {
    mLayerToWrapper[i] = -1;
  }
  if (build) {
    Build(loadTrans);
  }
}

//__________________________________________________________________________
void GeometryTGeo::adopt(GeometryTGeo* raw)
{
  // adopt the unique instance from external raw pointer (to be used only to read saved instance from file)
  if (sInstance) {
    LOG(FATAL) << "No adoption: o2::its::GeometryTGeo instance exists";
  }
  sInstance = std::unique_ptr<o2::its::GeometryTGeo>(raw);
}

//__________________________________________________________________________
int GeometryTGeo::getChipIndex(int lay, int sta, int chipInStave) const
{
  return getFirstChipIndex(lay) + mNumberOfChipsPerStave[lay] * sta + chipInStave;
}

//__________________________________________________________________________
int GeometryTGeo::getChipIndex(int lay, int sta, int substa, int chipInSStave) const
{
  int n = getFirstChipIndex(lay) + mNumberOfChipsPerStave[lay] * sta + chipInSStave;
  if (mNumberOfHalfStaves[lay] && substa > 0) {
    n += mNumberOfChipsPerHalfStave[lay] * substa;
  }
  return n;
}

//__________________________________________________________________________
int GeometryTGeo::getChipIndex(int lay, int sta, int substa, int md, int chipInMod) const
{
  int n = getFirstChipIndex(lay) + mNumberOfChipsPerStave[lay] * sta + chipInMod;
  if (mNumberOfHalfStaves[lay] && substa > 0) {
    n += mNumberOfChipsPerHalfStave[lay] * substa;
  }
  if (mNumberOfModules[lay] && md > 0) {
    n += mNumberOfChipsPerModule[lay] * md;
  }
  return n;
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
  if (mNumberOfHalfStaves[lay]) {
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
const char* GeometryTGeo::composeSymNameLayer(int lr)
{
  return Form("%s/%s%d", composeSymNameITS(), getITSLayerPattern(), lr);
}

//__________________________________________________________________________
const char* GeometryTGeo::composeSymNameStave(int lr, int stave)
{
  return Form("%s/%s%d", composeSymNameLayer(lr), getITSStavePattern(), stave);
}

//__________________________________________________________________________
const char* GeometryTGeo::composeSymNameHalfStave(int lr, int stave, int substave)
{
  return substave >= 0 ? Form("%s/%s%d", composeSymNameStave(lr, stave), getITSHalfStavePattern(), substave)
                       : composeSymNameStave(lr, stave);
}

//__________________________________________________________________________
const char* GeometryTGeo::composeSymNameModule(int lr, int stave, int substave, int mod)
{
  return mod >= 0 ? Form("%s/%s%d", composeSymNameHalfStave(lr, stave, substave), getITSModulePattern(), mod)
                  : composeSymNameHalfStave(lr, stave, substave);
}

//__________________________________________________________________________
const char* GeometryTGeo::composeSymNameChip(int lr, int sta, int substave, int mod, int chip)
{
  return Form("%s/%s%d", composeSymNameModule(lr, sta, substave, mod), getITSChipPattern(), chip);
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

  int lay, stav, sstav, mod, chipInMod;
  getChipId(index, lay, stav, sstav, mod, chipInMod);

  int wrID = mLayerToWrapper[lay];

  TString path = Form("/cave_1/%s_2/", GeometryTGeo::getITSVolPattern());

  if (wrID >= 0) {
    path += Form("%s%d_1/", getITSWrapVolPattern(), wrID);
  }

  path +=
    Form("%s%d_1/%s%d_%d/", GeometryTGeo::getITSLayerPattern(), lay, GeometryTGeo::getITSStavePattern(), lay, stav);

  if (mNumberOfHalfStaves[lay] > 0) {
    path += Form("%s%d_%d/", GeometryTGeo::getITSHalfStavePattern(), lay, sstav);
  }
  if (mNumberOfModules[lay] > 0) {
    path += Form("%s%d_%d/", GeometryTGeo::getITSModulePattern(), lay, mod);
  }
  path +=
    Form("%s%d_%d/%s%d_1", GeometryTGeo::getITSChipPattern(), lay, chipInMod, GeometryTGeo::getITSSensorPattern(), lay);

  static TGeoHMatrix matTmp;
  gGeoManager->PushPath();

  if (!gGeoManager->cd(path.Data())) {
    gGeoManager->PopPath();
    LOG(ERROR) << "Error in cd-ing to " << path.Data();
    return nullptr;
  } // end if !gGeoManager

  matTmp = *gGeoManager->GetCurrentMatrix(); // matrix may change after cd
  // RSS
  //  printf("%d/%d/%d %s\n",lay,stav,detInSta,path.Data());
  //  mat->Print();
  // Restore the modeler state.
  gGeoManager->PopPath();

  // account for the difference between physical sensitive layer (where charge collection is simulated) and effective sensor ticknesses
  static TGeoTranslation tra(0., 0.5 * (Segmentation::SensorLayerThickness - Segmentation::SensorLayerThicknessEff), 0.);

  matTmp *= tra;

  return &matTmp;
}

//__________________________________________________________________________
void GeometryTGeo::Build(int loadTrans)
{
  if (isBuilt()) {
    LOG(WARNING) << "Already built";
    return; // already initialized
  }

  if (!gGeoManager) {
    // RSTODO: in future there will be a method to load matrices from the CDB
    LOG(FATAL) << "Geometry is not loaded";
  }

  mNumberOfLayers = extractNumberOfLayers();
  if (!mNumberOfLayers) {
    return;
  }

  mNumberOfStaves.resize(mNumberOfLayers);
  mNumberOfHalfStaves.resize(mNumberOfLayers);
  mNumberOfModules.resize(mNumberOfLayers);
  mNumberOfChipsPerModule.resize(mNumberOfLayers);
  mNumberOfChipRowsPerModule.resize(mNumberOfLayers);
  mNumberOfChipsPerHalfStave.resize(mNumberOfLayers);
  mNumberOfChipsPerStave.resize(mNumberOfLayers);
  mNumberOfChipsPerLayer.resize(mNumberOfLayers);
  mLastChipIndex.resize(mNumberOfLayers);
  int numberOfChips = 0;

  for (int i = 0; i < mNumberOfLayers; i++) {
    mNumberOfStaves[i] = extractNumberOfStaves(i);
    mNumberOfHalfStaves[i] = extractNumberOfHalfStaves(i);
    mNumberOfModules[i] = extractNumberOfModules(i);
    mNumberOfChipsPerModule[i] = extractNumberOfChipsPerModule(i, mNumberOfChipRowsPerModule[i]);
    mNumberOfChipsPerHalfStave[i] = mNumberOfChipsPerModule[i] * Max(1, mNumberOfModules[i]);
    mNumberOfChipsPerStave[i] = mNumberOfChipsPerHalfStave[i] * Max(1, mNumberOfHalfStaves[i]);
    mNumberOfChipsPerLayer[i] = mNumberOfChipsPerStave[i] * mNumberOfStaves[i];
    numberOfChips += mNumberOfChipsPerLayer[i];
    mLastChipIndex[i] = numberOfChips - 1;
  }
  setSize(numberOfChips);
  fillTrackingFramesCache();
  //
  fillMatrixCache(loadTrans);
}

//__________________________________________________________________________
void GeometryTGeo::fillMatrixCache(int mask)
{
  // populate matrix cache for requested transformations
  //
  if (mSize < 1) {
    LOG(WARNING) << "The method Build was not called yet";
    Build(mask);
    return;
  }

  // build matrices
  if ((mask & o2::utils::bit2Mask(o2::TransformType::L2G)) && !getCacheL2G().isFilled()) {
    // Matrices for Local (Sensor!!! rather than the full chip) to Global frame transformation
    LOG(INFO) << "Loading ITS L2G matrices from TGeo";
    auto& cacheL2G = getCacheL2G();
    cacheL2G.setSize(mSize);

    for (int i = 0; i < mSize; i++) {
      TGeoHMatrix* hm = extractMatrixSensor(i);
      cacheL2G.setMatrix(Mat3D(*hm), i);
    }
  }

  if ((mask & o2::utils::bit2Mask(o2::TransformType::T2L)) && !getCacheT2L().isFilled()) {
    // matrices for Tracking to Local (Sensor!!! rather than the full chip) frame transformation
    LOG(INFO) << "Loading ITS T2L matrices from TGeo";
    auto& cacheT2L = getCacheT2L();
    cacheT2L.setSize(mSize);
    for (int i = 0; i < mSize; i++) {
      TGeoHMatrix& hm = createT2LMatrix(i);
      cacheT2L.setMatrix(Mat3D(hm), i);
    }
  }

  if ((mask & o2::utils::bit2Mask(o2::TransformType::T2G)) && !getCacheT2G().isFilled()) {
    LOG(WARNING) << "It is faster to use 2D rotation for T2G instead of full Transform3D matrices";
    // matrices for Tracking to Global frame transformation
    LOG(INFO) << "Loading ITS T2G matrices from TGeo";
    auto& cacheT2G = getCacheT2G();
    cacheT2G.setSize(mSize);

    for (int i = 0; i < mSize; i++) {
      TGeoHMatrix& mat = createT2LMatrix(i);
      mat.MultiplyLeft(extractMatrixSensor(i));
      cacheT2G.setMatrix(Mat3D(mat), i);
    }
  }

  if ((mask & o2::utils::bit2Mask(o2::TransformType::T2GRot)) && !getCacheT2GRot().isFilled()) {
    // 2D rotation matrices for Tracking frame to Global rotations
    LOG(INFO) << "Loading ITS T2G rotation 2D matrices";
    auto& cacheT2Gr = getCacheT2GRot();
    cacheT2Gr.setSize(mSize);
    for (int i = 0; i < mSize; i++) {
      cacheT2Gr.setMatrix(Rot2D(getSensorRefAlpha(i)), i);
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
  if (!itsV) {
    LOG(FATAL) << "ITS volume " << getITSVolPattern() << " is not in the geometry";
  }

  // Loop on all ITSV nodes, count Layer volumes by checking names
  // Build on the fly layer - wrapper correspondence
  TObjArray* nodes = itsV->GetNodes();
  int nNodes = nodes->GetEntriesFast();

  for (int j = 0; j < nNodes; j++) {
    int lrID = -1;
    TGeoNode* nd = (TGeoNode*)nodes->At(j);
    const char* name = nd->GetName();

    if (strstr(name, getITSLayerPattern())) {
      numberOfLayers++;
      if ((lrID = extractVolumeCopy(name, GeometryTGeo::getITSLayerPattern())) < 0) {
        LOG(FATAL) << "Failed to extract layer ID from the " << name;
        exit(1);
      }

      mLayerToWrapper[lrID] = -1;                      // not wrapped
    } else if (strstr(name, getITSWrapVolPattern())) { // this is a wrapper volume, may cointain layers
      int wrID = -1;
      if ((wrID = extractVolumeCopy(name, GeometryTGeo::getITSWrapVolPattern())) < 0) {
        LOG(FATAL) << "Failed to extract wrapper ID from the " << name;
        exit(1);
      }

      TObjArray* nodesW = nd->GetNodes();
      int nNodesW = nodesW->GetEntriesFast();

      for (int jw = 0; jw < nNodesW; jw++) {
        TGeoNode* ndW = (TGeoNode*)nodesW->At(jw);
        if (strstr(ndW->GetName(), getITSLayerPattern())) {
          if ((lrID = extractVolumeCopy(ndW->GetName(), GeometryTGeo::getITSLayerPattern())) < 0) {
            LOG(FATAL) << "Failed to extract layer ID from the " << name;
            exit(1);
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
int GeometryTGeo::extractNumberOfStaves(int lay) const
{
  int numberOfStaves = 0;
  char laynam[30];
  snprintf(laynam, 30, "%s%d", getITSLayerPattern(), lay);
  TGeoVolume* volLr = gGeoManager->GetVolume(laynam);
  if (!volLr) {
    LOG(FATAL) << "can't find " << laynam << " volume";
    return -1;
  }

  // Loop on all layer nodes, count Stave volumes by checking names
  int nNodes = volLr->GetNodes()->GetEntries();
  for (int j = 0; j < nNodes; j++) {
    // LOG(INFO) << "L" << lay << " " << j << " of " << nNodes << " "
    //           << volLr->GetNodes()->At(j)->GetName() << " "
    //           << getITSStavePattern() << " -> " << numberOfStaves;
    if (strstr(volLr->GetNodes()->At(j)->GetName(), getITSStavePattern())) {
      numberOfStaves++;
    }
  }
  return numberOfStaves;
}

//__________________________________________________________________________
int GeometryTGeo::extractNumberOfHalfStaves(int lay) const
{
  if (sHalfStaveName.empty()) {
    return 0; // for the setup w/o substave defined the stave and the substave is the same thing
  }
  int nSS = 0;
  char stavnam[30];
  snprintf(stavnam, 30, "%s%d", getITSStavePattern(), lay);
  TGeoVolume* volLd = gGeoManager->GetVolume(stavnam);
  if (!volLd) {
    LOG(FATAL) << "can't find volume " << stavnam;
  }
  // Loop on all stave nodes, count Chip volumes by checking names
  int nNodes = volLd->GetNodes()->GetEntries();
  for (int j = 0; j < nNodes; j++) {
    if (strstr(volLd->GetNodes()->At(j)->GetName(), getITSHalfStavePattern())) {
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
    snprintf(stavnam, 30, "%s%d", getITSHalfStavePattern(), lay);
    volLd = gGeoManager->GetVolume(stavnam);
  }
  if (!volLd) { // no substaves, check staves
    snprintf(stavnam, 30, "%s%d", getITSStavePattern(), lay);
    volLd = gGeoManager->GetVolume(stavnam);
  }
  if (!volLd) {
    return 0;
  }

  int nMod = 0;

  // Loop on all substave nodes, count module volumes by checking names
  int nNodes = volLd->GetNodes()->GetEntries();

  for (int j = 0; j < nNodes; j++) {
    if (strstr(volLd->GetNodes()->At(j)->GetName(), getITSModulePattern())) {
      nMod++;
    }
  }
  return nMod;
}

//__________________________________________________________________________
int GeometryTGeo::extractNumberOfChipsPerModule(int lay, int& nrow) const
{
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
    LOG(FATAL) << "can't find volume containing chips on layer " << lay;
  }

  // Loop on all stave nodes, count Chip volumes by checking names
  int nNodes = volLd->GetNodes()->GetEntries();

  double xmin = 1e9, xmax = -1e9, zmin = 1e9, zmax = -1e9;
  double lab[3], loc[3] = {0, 0, 0};
  double dx = -1, dz = -1;

  for (int j = 0; j < nNodes; j++) {
    //    AliInfo(Form("L%d %d of %d %s %s ->
    // %d",lay,j,nNodes,volLd->GetNodes()->At(j)->GetName(),GetITSChipPattern(),numberOfChips));
    TGeoNodeMatrix* node = (TGeoNodeMatrix*)volLd->GetNodes()->At(j);
    if (!strstr(node->GetName(), getITSChipPattern())) {
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
        LOG(FATAL) << "Chip " << node->GetName() << " volume is of unprocessed shape " << chShape->IsA()->GetName();
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
    LOG(ERROR) << "Inconsistency between Nchips=" << numberOfChips << " and Nrow*Ncol=" << nrow << "*" << ncol << "->"
               << nrow * ncol << FairLogger::endl
               << "Extracted chip dimensions (x,z): " << dx << " " << dz << " Module Span: " << spanX << " " << spanZ;
  }
  return numberOfChips;
}

//__________________________________________________________________________
int GeometryTGeo::extractLayerChipType(int lay) const
{
  char stavnam[30];
  snprintf(stavnam, 30, "%s%d", getITSLayerPattern(), lay);
  TGeoVolume* volLd = gGeoManager->GetVolume(stavnam);
  if (!volLd) {
    LOG(FATAL) << "can't find volume " << stavnam;
    return -1;
  }
  return volLd->GetUniqueID();
}

//__________________________________________________________________________
void GeometryTGeo::Print(Option_t*) const
{
  printf("NLayers:%d NChips:%d\n", mNumberOfLayers, getNumberOfChips());
  if (!isBuilt())
    return;

  for (int i = 0; i < mNumberOfLayers; i++) {
    printf(
      "Lr%2d\tNStav:%2d\tNChips:%2d "
      "(%dx%-2d)\tNMod:%d\tNSubSt:%d\tNSt:%3d\tChip#:%5d:%-5d\tWrapVol:%d\n",
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
  double locA[3] = {-100., 0., 0.}, locB[3] = {100., 0., 0.}, gloA[3], gloB[3];
  const TGeoHMatrix* matL2G = extractMatrixSensor(isn);

  matL2G->LocalToMaster(locA, gloA);
  matL2G->LocalToMaster(locB, gloB);
  double dx = gloB[0] - gloA[0], dy = gloB[1] - gloA[1];
  double t = (gloB[0] * dx + gloB[1] * dy) / (dx * dx + dy * dy);
  double xp = gloB[0] - dx * t, yp = gloB[1] - dy * t;
  x = Sqrt(xp * xp + yp * yp);
  alp = ATan2(yp, xp);
  BringTo02Pi(alp);
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
