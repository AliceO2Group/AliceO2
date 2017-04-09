/// \file GeometryTGeo.cxx
/// \brief Implementation of the GeometryTGeo class
/// \author cvetan.cheshkov@cern.ch - 15/02/2007
/// \author ruben.shahoyan@cern.ch - adapted to ITSupg 18/07/2012

// ATTENTION: In opposite to old AliITSgeomTGeo, all indices start from 0, not from 1!!!
#include "ITSBase/GeometryTGeo.h"
#include "ITSBase/GeometryManager.h"
#include "ITSMFTBase/Segmentation.h"
#include "ITSMFTBase/SegmentationPixel.h"

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

using o2::ITSMFT::Segmentation;
using o2::ITSMFT::SegmentationPixel;
using namespace TMath;
using namespace o2::ITS;

ClassImp(o2::ITS::GeometryTGeo)

  UInt_t GeometryTGeo::mUIDShift = 16; // bit shift to go from mod.id to modUUID for TGeo
TString GeometryTGeo::mVolumeName = "ITSV";
TString GeometryTGeo::mLayerName = "ITSULayer";
TString GeometryTGeo::mStaveName = "ITSUStave";
TString GeometryTGeo::mHalfStaveName = "ITSUHalfStave";
TString GeometryTGeo::mModuleName = "ITSUModule";
TString GeometryTGeo::mChipName = "ITSUChip";
TString GeometryTGeo::mSensorName = "ITSUSensor";
TString GeometryTGeo::mWrapperVolumeName = "ITSUWrapVol";
TString GeometryTGeo::mChipTypeName[GeometryTGeo::kNChipTypes] = { "Pix" };

TString GeometryTGeo::mSegmentationFileName = "itsSegmentations.root";

GeometryTGeo::GeometryTGeo(Bool_t build, Bool_t loadSegmentations)
  : mVersion(kITSVNA),
    mNumberOfLayers(0),
    mNumberOfChips(0),
    mNumberOfStaves(nullptr),
    mNumberOfHalfStaves(nullptr),
    mNumberOfModules(nullptr),
    mNumberOfChipsPerModule(nullptr),
    mNumberOfChipRowsPerModule(nullptr),
    mNumberOfChipsPerHalfStave(nullptr),
    mNumberOfChipsPerStave(nullptr),
    mNumberOfChipsPerLayer(nullptr),
    mLayerChipType(nullptr),
    mLastChipIndex(nullptr),
    mSensorMatrices(nullptr),
    mTrackingToLocalMatrices(nullptr),
    mSegmentations(nullptr)
{
  // default c-tor
  for (int i = gMaxLayers; i--;) {
    mLayerToWrapper[i] = -1;
  }
  if (build) {
    Build(loadSegmentations);
  }
}

GeometryTGeo::GeometryTGeo(const GeometryTGeo& src)
  : TObject(src),
    mVersion(src.mVersion),
    mNumberOfLayers(src.mNumberOfLayers),
    mNumberOfChips(src.mNumberOfChips),
    mNumberOfStaves(nullptr),
    mNumberOfHalfStaves(nullptr),
    mNumberOfModules(nullptr),
    mNumberOfChipsPerModule(nullptr),
    mNumberOfChipRowsPerModule(nullptr),
    mNumberOfChipsPerHalfStave(nullptr),
    mNumberOfChipsPerStave(nullptr),
    mNumberOfChipsPerLayer(nullptr),
    mLayerChipType(nullptr),
    mLastChipIndex(nullptr),
    mSensorMatrices(nullptr),
    mTrackingToLocalMatrices(nullptr),
    mSegmentations(nullptr)
{
  // copy c-tor
  if (mNumberOfLayers) {
    mNumberOfStaves = new Int_t[mNumberOfLayers];
    mNumberOfChipsPerModule = new Int_t[mNumberOfLayers];
    mNumberOfChipRowsPerModule = new Int_t[mNumberOfLayers];
    mLayerChipType = new Int_t[mNumberOfLayers];
    mLastChipIndex = new Int_t[mNumberOfLayers];
    mNumberOfChipsPerHalfStave = new Int_t[mNumberOfLayers];
    mNumberOfChipsPerStave = new Int_t[mNumberOfLayers];
    mNumberOfChipsPerLayer = new Int_t[mNumberOfLayers];

    for (int i = mNumberOfLayers; i--;) {
      mNumberOfStaves[i] = src.mNumberOfStaves[i];
      mNumberOfHalfStaves[i] = src.mNumberOfHalfStaves[i];
      mNumberOfModules[i] = src.mNumberOfModules[i];
      mNumberOfChipsPerModule[i] = src.mNumberOfChipsPerModule[i];
      mNumberOfChipRowsPerModule[i] = src.mNumberOfChipRowsPerModule[i];
      mNumberOfChipsPerHalfStave[i] = src.mNumberOfChipsPerHalfStave[i];
      mNumberOfChipsPerStave[i] = src.mNumberOfChipsPerStave[i];
      mNumberOfChipsPerLayer[i] = src.mNumberOfChipsPerLayer[i];
      mLayerChipType[i] = src.mLayerChipType[i];
      mLastChipIndex[i] = src.mLastChipIndex[i];
    }
    if (src.mSensorMatrices) {
      mSensorMatrices = new TObjArray(mNumberOfChips);
      mSensorMatrices->SetOwner(kTRUE);
      for (int i = 0; i < mNumberOfChips; i++) {
        const TGeoHMatrix* mat = (TGeoHMatrix*)src.mSensorMatrices->At(i);
        mSensorMatrices->AddAt(new TGeoHMatrix(*mat), i);
      }
    }
    if (src.mTrackingToLocalMatrices) {
      mTrackingToLocalMatrices = new TObjArray(mNumberOfChips);
      mTrackingToLocalMatrices->SetOwner(kTRUE);
      for (int i = 0; i < mNumberOfChips; i++) {
        const TGeoHMatrix* mat = (TGeoHMatrix*)src.mTrackingToLocalMatrices->At(i);
        mTrackingToLocalMatrices->AddAt(new TGeoHMatrix(*mat), i);
      }
    }
    if (src.mSegmentations) {
      int sz = src.mSegmentations->GetEntriesFast();
      mSegmentations = new TObjArray(sz);
      mSegmentations->SetOwner(kTRUE);
      for (int i = 0; i < sz; i++) {
        Segmentation* sg = (Segmentation*)src.mSegmentations->UncheckedAt(i);
        if (!sg) {
          continue;
        }
        mSegmentations->AddAt(sg->Clone(), i);
      }
    }
  }
  for (int i = gMaxLayers; i--;) {
    mLayerToWrapper[i] = src.mLayerToWrapper[i];
  }
}

GeometryTGeo::~GeometryTGeo()
{
  // d-tor
  delete[] mNumberOfStaves;
  delete[] mNumberOfHalfStaves;
  delete[] mNumberOfModules;
  delete[] mLayerChipType;
  delete[] mNumberOfChipsPerModule;
  delete[] mNumberOfChipRowsPerModule;
  delete[] mNumberOfChipsPerHalfStave;
  delete[] mNumberOfChipsPerStave;
  delete[] mNumberOfChipsPerLayer;
  delete[] mLastChipIndex;
  delete mTrackingToLocalMatrices;
  delete mSensorMatrices;
  delete mSegmentations;
}

GeometryTGeo& GeometryTGeo::operator=(const GeometryTGeo& src)
{
  // cp op.
  if (this != &src) {
    delete[] mNumberOfStaves;
    delete[] mNumberOfHalfStaves;
    delete[] mNumberOfModules;
    delete[] mLayerChipType;
    delete[] mNumberOfChipsPerModule;
    delete[] mNumberOfChipRowsPerModule;
    delete[] mNumberOfChipsPerHalfStave;
    delete[] mNumberOfChipsPerStave;
    delete[] mNumberOfChipsPerLayer;
    delete[] mLastChipIndex;
    mNumberOfStaves = mNumberOfHalfStaves = mNumberOfModules = mLayerChipType = mNumberOfChipsPerModule =
      mLastChipIndex = nullptr;
    mVersion = src.mVersion;
    mNumberOfLayers = src.mNumberOfLayers;
    mNumberOfChips = src.mNumberOfChips;
    if (src.mSensorMatrices) {
      delete mSensorMatrices;
      mSensorMatrices = new TObjArray(mNumberOfChips);
      mSensorMatrices->SetOwner(kTRUE);
      for (int i = 0; i < mNumberOfChips; i++) {
        const TGeoHMatrix* mat = (TGeoHMatrix*)src.mSensorMatrices->At(i);
        mSensorMatrices->AddAt(new TGeoHMatrix(*mat), i);
      }
    }
    if (src.mTrackingToLocalMatrices) {
      delete mTrackingToLocalMatrices;
      mTrackingToLocalMatrices = new TObjArray(mNumberOfChips);
      mTrackingToLocalMatrices->SetOwner(kTRUE);
      for (int i = 0; i < mNumberOfChips; i++) {
        const TGeoHMatrix* mat = (TGeoHMatrix*)src.mTrackingToLocalMatrices->At(i);
        mTrackingToLocalMatrices->AddAt(new TGeoHMatrix(*mat), i);
      }
    }
    if (src.mSegmentations) {
      int sz = src.mSegmentations->GetEntriesFast();
      mSegmentations = new TObjArray(sz);
      mSegmentations->SetOwner(kTRUE);
      for (int i = 0; i < sz; i++) {
        Segmentation* sg = (Segmentation*)src.mSegmentations->UncheckedAt(i);
        if (!sg) {
          continue;
        }
        mSegmentations->AddAt(sg->Clone(), i);
      }
    }

    if (mNumberOfLayers) {
      mNumberOfStaves = new Int_t[mNumberOfLayers];
      mNumberOfHalfStaves = new Int_t[mNumberOfLayers];
      mNumberOfModules = new Int_t[mNumberOfLayers];
      mNumberOfChipsPerModule = new Int_t[mNumberOfLayers];
      mNumberOfChipRowsPerModule = new Int_t[mNumberOfLayers];
      mNumberOfChipsPerHalfStave = new Int_t[mNumberOfLayers];
      mNumberOfChipsPerStave = new Int_t[mNumberOfLayers];
      mNumberOfChipsPerLayer = new Int_t[mNumberOfLayers];
      mLayerChipType = new Int_t[mNumberOfLayers];
      mLastChipIndex = new Int_t[mNumberOfLayers];
      for (int i = mNumberOfLayers; i--;) {
        mNumberOfStaves[i] = src.mNumberOfStaves[i];
        mNumberOfHalfStaves[i] = src.mNumberOfHalfStaves[i];
        mNumberOfModules[i] = src.mNumberOfModules[i];
        mNumberOfChipsPerModule[i] = src.mNumberOfChipsPerModule[i];
        mNumberOfChipRowsPerModule[i] = src.mNumberOfChipRowsPerModule[i];
        mNumberOfChipsPerHalfStave[i] = src.mNumberOfChipsPerHalfStave[i];
        mNumberOfChipsPerStave[i] = src.mNumberOfChipsPerStave[i];
        mNumberOfChipsPerLayer[i] = src.mNumberOfChipsPerLayer[i];
        mLayerChipType[i] = src.mLayerChipType[i];
        mLastChipIndex[i] = src.mLastChipIndex[i];
      }
    }
  }
  return *this;
}

Int_t GeometryTGeo::getChipIndex(Int_t lay, Int_t sta, Int_t chipInStave) const
{
  return getFirstChipIndex(lay) + mNumberOfChipsPerStave[lay] * sta + chipInStave;
}

Int_t GeometryTGeo::getChipIndex(Int_t lay, Int_t sta, Int_t substa, Int_t chipInSStave) const
{
  int n = getFirstChipIndex(lay) + mNumberOfChipsPerStave[lay] * sta + chipInSStave;
  if (mNumberOfHalfStaves[lay] && substa > 0) {
    n += mNumberOfChipsPerHalfStave[lay] * substa;
  }
  return n;
}

Int_t GeometryTGeo::getChipIndex(Int_t lay, Int_t sta, Int_t substa, Int_t md, Int_t chipInMod) const
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

Bool_t GeometryTGeo::getLayer(Int_t index, Int_t& lay, Int_t& indexInLr) const
{
  lay = getLayer(index);
  indexInLr = index - getFirstChipIndex(lay);
  return kTRUE;
}

Int_t GeometryTGeo::getLayer(Int_t index) const
{
  int lay = 0;
  while (index > mLastChipIndex[lay]) {
    lay++;
  }
  return lay;
}

Int_t GeometryTGeo::getStave(Int_t index) const
{
  int lay = 0;
  while (index > mLastChipIndex[lay]) {
    lay++;
  }
  index -= getFirstChipIndex(lay);
  return index / mNumberOfChipsPerStave[lay];
}

Int_t GeometryTGeo::getHalfStave(Int_t index) const
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

Int_t GeometryTGeo::getModule(Int_t index) const
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

Int_t GeometryTGeo::getChipIdInLayer(Int_t index) const
{
  int lay = 0;
  while (index > mLastChipIndex[lay]) {
    lay++;
  }
  index -= getFirstChipIndex(lay);
  return index;
}

Int_t GeometryTGeo::getChipIdInStave(Int_t index) const
{
  int lay = 0;
  while (index > mLastChipIndex[lay]) {
    lay++;
  }
  index -= getFirstChipIndex(lay);
  return index % mNumberOfChipsPerStave[lay];
}

Int_t GeometryTGeo::getChipIdInHalfStave(Int_t index) const
{
  int lay = 0;
  while (index > mLastChipIndex[lay]) {
    lay++;
  }
  index -= getFirstChipIndex(lay);
  return index % mNumberOfChipsPerHalfStave[lay];
}

Int_t GeometryTGeo::getChipIdInModule(Int_t index) const
{
  int lay = 0;
  while (index > mLastChipIndex[lay]) {
    lay++;
  }
  index -= getFirstChipIndex(lay);
  return index % mNumberOfChipsPerModule[lay];
}

Bool_t GeometryTGeo::getChipId(Int_t index, Int_t& lay, Int_t& sta, Int_t& hsta, Int_t& mod, Int_t& chip) const
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

const char* GeometryTGeo::getSymbolicName(Int_t index) const
{
  Int_t lay, index2;
  if (!getLayer(index, lay, index2)) {
    return nullptr;
  }
  // return
  // GeometryManager::SymName((GeometryManager::ELayerID)((lay-1)+GeometryManager::kSPD1),index2);
  // RS: this is not optimal, but we cannod access directly GeometryManager, since the latter has
  // hardwired layers
  //  TGeoPNEntry* pne = gGeoManager->GetAlignableEntryByUID(
  // GeometryManager::layerToVolUID(lay+1,index2) );
  TGeoPNEntry* pne = gGeoManager->GetAlignableEntryByUID(chipVolUID(index));
  if (!pne) {
    LOG(ERROR) << "Failed to find alignable entry with index " << index << ": (Lr" << lay << " Chip:" << index2 << ") !"
               << FairLogger::endl;
    return nullptr;
  }
  return pne->GetName();
}

const char* GeometryTGeo::composeSymNameITS() { return "ITS"; }
const char* GeometryTGeo::composeSymNameLayer(Int_t lr)
{
  return Form("%s/%s%d", composeSymNameITS(), getITSLayerPattern(), lr);
}

const char* GeometryTGeo::composeSymNameStave(Int_t lr, Int_t stave)
{
  return Form("%s/%s%d", composeSymNameLayer(lr), getITSStavePattern(), stave);
}

const char* GeometryTGeo::composeSymNameHalfStave(Int_t lr, Int_t stave, Int_t substave)
{
  return substave >= 0 ? Form("%s/%s%d", composeSymNameStave(lr, stave), getITSHalfStavePattern(), substave)
                       : composeSymNameStave(lr, stave);
}

const char* GeometryTGeo::composeSymNameModule(Int_t lr, Int_t stave, Int_t substave, Int_t mod)
{
  return mod >= 0 ? Form("%s/%s%d", composeSymNameHalfStave(lr, stave, substave), getITSModulePattern(), mod)
                  : composeSymNameHalfStave(lr, stave, substave);
}

const char* GeometryTGeo::composeSymNameChip(Int_t lr, Int_t sta, Int_t substave, Int_t mod, Int_t chip)
{
  return Form("%s/%s%d", composeSymNameModule(lr, sta, substave, mod), getITSChipPattern(), chip);
}

TGeoHMatrix* GeometryTGeo::GetMatrix(Int_t index) const
{
  static TGeoHMatrix matTmp;
  TGeoPNEntry* pne = getPNEntry(index);
  if (!pne) {
    return nullptr;
  }

  TGeoPhysicalNode* pnode = pne->GetPhysicalNode();
  if (pnode) {
    return pnode->GetMatrix();
  }

  const char* path = pne->GetTitle();
  gGeoManager->PushPath(); // Preserve the modeler state.
  if (!gGeoManager->cd(path)) {
    gGeoManager->PopPath();
    LOG(ERROR) << "Volume path " << path << " not valid!" << FairLogger::endl;
    return nullptr;
  }
  matTmp = *gGeoManager->GetCurrentMatrix();
  gGeoManager->PopPath();
  return &matTmp;
}

Bool_t GeometryTGeo::GetTranslation(Int_t index, Double_t t[3]) const
{
  TGeoHMatrix* m = GetMatrix(index);
  if (!m) {
    return kFALSE;
  }

  Double_t* trans = m->GetTranslation();
  for (Int_t i = 0; i < 3; i++) {
    t[i] = trans[i];
  }

  return kTRUE;
}

Bool_t GeometryTGeo::getRotation(Int_t index, Double_t r[9]) const
{
  TGeoHMatrix* m = GetMatrix(index);
  if (!m) {
    return kFALSE;
  }

  Double_t* rot = m->GetRotationMatrix();
  for (Int_t i = 0; i < 9; i++) {
    r[i] = rot[i];
  }

  return kTRUE;
}

Bool_t GeometryTGeo::GetOriginalMatrix(Int_t index, TGeoHMatrix& m) const
{
  m.Clear();

  const char* symname = getSymbolicName(index);
  if (!symname) {
    return kFALSE;
  }

  return GeometryManager::getOriginalGlobalMatrix(symname, m);
}

Bool_t GeometryTGeo::getOriginalTranslation(Int_t index, Double_t t[3]) const
{
  TGeoHMatrix m;
  if (!GetOriginalMatrix(index, m)) {
    return kFALSE;
  }

  Double_t* trans = m.GetTranslation();
  for (Int_t i = 0; i < 3; i++) {
    t[i] = trans[i];
  }

  return kTRUE;
}

Bool_t GeometryTGeo::getOriginalRotation(Int_t index, Double_t r[9]) const
{
  TGeoHMatrix m;
  if (!GetOriginalMatrix(index, m)) {
    return kFALSE;
  }

  Double_t* rot = m.GetRotationMatrix();
  for (Int_t i = 0; i < 9; i++) {
    r[i] = rot[i];
  }

  return kTRUE;
}

TGeoHMatrix* GeometryTGeo::extractMatrixTrackingToLocal(Int_t index) const
{
  TGeoPNEntry* pne = getPNEntry(index);
  if (!pne) {
    return nullptr;
  }

  TGeoHMatrix* m = (TGeoHMatrix*)pne->GetMatrix();
  if (!m) {
    LOG(ERROR) << "TGeoPNEntry (" << pne->GetName() << ") contains no matrix !" << FairLogger::endl;
  }

  return m;
}

Bool_t GeometryTGeo::getTrackingMatrix(Int_t index, TGeoHMatrix& m)
{
  m.Clear();

  TGeoHMatrix* m1 = GetMatrix(index);
  if (!m1) {
    return kFALSE;
  }

  const TGeoHMatrix* m2 = getMatrixT2L(index);
  if (!m2) {
    return kFALSE;
  }

  m = *m1;
  m.Multiply(m2);

  return kTRUE;
}

TGeoHMatrix* GeometryTGeo::extractMatrixSensor(Int_t index) const
{
  Int_t lay, stav, sstav, mod, chipInMod;
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
    LOG(ERROR) << "Error in cd-ing to " << path.Data() << FairLogger::endl;
    return nullptr;
  } // end if !gGeoManager

  matTmp = *gGeoManager->GetCurrentMatrix(); // matrix may change after cd
  // RSS
  //  printf("%d/%d/%d %s\n",lay,stav,detInSta,path.Data());
  //  mat->Print();
  // Restore the modeler state.
  gGeoManager->PopPath();
  return &matTmp;
}

TGeoPNEntry* GeometryTGeo::getPNEntry(Int_t index) const
{
  if (index >= mNumberOfChips) {
    LOG(ERROR) << "Invalid ITS chip index: " << index << " (0 -> " << mNumberOfChips << ") !" << FairLogger::endl;
    return nullptr;
  }

  if (!gGeoManager || !gGeoManager->IsClosed()) {
    LOG(ERROR) << "Can't get the matrix! gGeoManager doesn't exist or it is still opened!" << FairLogger::endl;
    return nullptr;
  }
  TGeoPNEntry* pne = gGeoManager->GetAlignableEntryByUID(chipVolUID(index));
  //  TGeoPNEntry* pne = gGeoManager->GetAlignableEntry(getSymbolicName(index));

  if (!pne) {
    LOG(ERROR) << "The index " << index << " does not correspond to a physical entry!" << FairLogger::endl;
  }
  return pne;
}

void GeometryTGeo::Build(Bool_t loadSegmentations)
{
  if (mVersion != kITSVNA) {
    LOG(WARNING) << "Already built" << FairLogger::endl;
    return; // already initialized
  }
  if (!gGeoManager) {
    LOG(FATAL) << "Geometry is not loaded" << FairLogger::endl;
  }

  mNumberOfLayers = extractNumberOfLayers();
  if (!mNumberOfLayers) {
    return;
  }

  mNumberOfStaves = new Int_t[mNumberOfLayers];
  mNumberOfHalfStaves = new Int_t[mNumberOfLayers];
  mNumberOfModules = new Int_t[mNumberOfLayers];
  mNumberOfChipsPerModule = new Int_t[mNumberOfLayers];
  mNumberOfChipRowsPerModule = new Int_t[mNumberOfLayers];
  mNumberOfChipsPerHalfStave = new Int_t[mNumberOfLayers];
  mNumberOfChipsPerStave = new Int_t[mNumberOfLayers];
  mNumberOfChipsPerLayer = new Int_t[mNumberOfLayers];
  mLayerChipType = new Int_t[mNumberOfLayers];
  mLastChipIndex = new Int_t[mNumberOfLayers];
  mNumberOfChips = 0;

  for (int i = 0; i < mNumberOfLayers; i++) {
    mLayerChipType[i] = extractLayerChipType(i);
    mNumberOfStaves[i] = extractNumberOfStaves(i);
    mNumberOfHalfStaves[i] = extractNumberOfHalfStaves(i);
    mNumberOfModules[i] = extractNumberOfModules(i);
    mNumberOfChipsPerModule[i] = extractNumberOfChipsPerModule(i, mNumberOfChipRowsPerModule[i]);
    mNumberOfChipsPerHalfStave[i] = mNumberOfChipsPerModule[i] * Max(1, mNumberOfModules[i]);
    mNumberOfChipsPerStave[i] = mNumberOfChipsPerHalfStave[i] * Max(1, mNumberOfHalfStaves[i]);
    mNumberOfChipsPerLayer[i] = mNumberOfChipsPerStave[i] * mNumberOfStaves[i];
    mNumberOfChips += mNumberOfChipsPerLayer[i];
    mLastChipIndex[i] = mNumberOfChips - 1;
  }

  fetchMatrices();
  mVersion = kITSVUpg;

  if (loadSegmentations) { // fetch segmentations
    mSegmentations = new TObjArray();
    SegmentationPixel::loadSegmentations(mSegmentations, getITSsegmentationFileName());
  }
}

Int_t GeometryTGeo::extractNumberOfLayers()
{
  Int_t numberOfLayers = 0;

  TGeoVolume* itsV = gGeoManager->GetVolume(getITSVolPattern());
  if (!itsV) {
    LOG(FATAL) << "ITS volume " << getITSVolPattern() << " is not in the geometry" << FairLogger::endl;
  }
  setUIDShift(itsV->GetUniqueID());

  // Loop on all ITSV nodes, count Layer volumes by checking names
  // Build on the fly layer - wrapper correspondence
  TObjArray* nodes = itsV->GetNodes();
  Int_t nNodes = nodes->GetEntriesFast();

  for (Int_t j = 0; j < nNodes; j++) {
    int lrID = -1;
    TGeoNode* nd = (TGeoNode*)nodes->At(j);
    const char* name = nd->GetName();

    if (strstr(name, getITSLayerPattern())) {
      numberOfLayers++;
      if ((lrID = extractVolumeCopy(name, GeometryTGeo::getITSLayerPattern())) < 0) {
        LOG(FATAL) << "Failed to extract layer ID from the " << name << FairLogger::endl;
        exit(1);
      }

      mLayerToWrapper[lrID] = -1;                      // not wrapped
    } else if (strstr(name, getITSWrapVolPattern())) { // this is a wrapper volume, may cointain layers
      int wrID = -1;
      if ((wrID = extractVolumeCopy(name, GeometryTGeo::getITSWrapVolPattern())) < 0) {
        LOG(FATAL) << "Failed to extract wrapper ID from the " << name << FairLogger::endl;
        exit(1);
      }

      TObjArray* nodesW = nd->GetNodes();
      int nNodesW = nodesW->GetEntriesFast();

      for (Int_t jw = 0; jw < nNodesW; jw++) {
        TGeoNode* ndW = (TGeoNode*)nodesW->At(jw);
        if (strstr(ndW->GetName(), getITSLayerPattern())) {
          if ((lrID = extractVolumeCopy(ndW->GetName(), GeometryTGeo::getITSLayerPattern())) < 0) {
            LOG(FATAL) << "Failed to extract layer ID from the " << name << FairLogger::endl;
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

Int_t GeometryTGeo::extractNumberOfStaves(Int_t lay) const
{
  Int_t numberOfStaves = 0;
  char laynam[30];
  snprintf(laynam, 30, "%s%d", getITSLayerPattern(), lay);
  TGeoVolume* volLr = gGeoManager->GetVolume(laynam);
  if (!volLr) {
    LOG(FATAL) << "can't find " << laynam << " volume" << FairLogger::endl;
    return -1;
  }

  // Loop on all layer nodes, count Stave volumes by checking names
  Int_t nNodes = volLr->GetNodes()->GetEntries();
  for (Int_t j = 0; j < nNodes; j++) {
    // LOG(INFO) << "L" << lay << " " << j << " of " << nNodes << " "
    //           << volLr->GetNodes()->At(j)->GetName() << " "
    //           << getITSStavePattern() << " -> " << numberOfStaves << FairLogger::endl;
    if (strstr(volLr->GetNodes()->At(j)->GetName(), getITSStavePattern())) {
      numberOfStaves++;
    }
  }
  return numberOfStaves;
}

Int_t GeometryTGeo::extractNumberOfHalfStaves(Int_t lay) const
{
  if (mHalfStaveName.IsNull()) {
    return 0; // for the setup w/o substave defined the stave and the substave is the same thing
  }
  Int_t nSS = 0;
  char stavnam[30];
  snprintf(stavnam, 30, "%s%d", getITSStavePattern(), lay);
  TGeoVolume* volLd = gGeoManager->GetVolume(stavnam);
  if (!volLd) {
    LOG(FATAL) << "can't find volume " << stavnam << FairLogger::endl;
  }
  // Loop on all stave nodes, count Chip volumes by checking names
  Int_t nNodes = volLd->GetNodes()->GetEntries();
  for (Int_t j = 0; j < nNodes; j++) {
    if (strstr(volLd->GetNodes()->At(j)->GetName(), getITSHalfStavePattern())) {
      nSS++;
    }
  }
  return nSS;
}

Int_t GeometryTGeo::extractNumberOfModules(Int_t lay) const
{
  if (mModuleName.IsNull()) {
    return 0;
  }

  char stavnam[30];
  TGeoVolume* volLd = nullptr;

  if (!mHalfStaveName.IsNull()) {
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

  Int_t nMod = 0;

  // Loop on all substave nodes, count module volumes by checking names
  Int_t nNodes = volLd->GetNodes()->GetEntries();

  for (Int_t j = 0; j < nNodes; j++) {
    if (strstr(volLd->GetNodes()->At(j)->GetName(), getITSModulePattern())) {
      nMod++;
    }
  }
  return nMod;
}

Int_t GeometryTGeo::extractNumberOfChipsPerModule(Int_t lay, int& nrow) const
{
  Int_t numberOfChips = 0;
  char stavnam[30];
  TGeoVolume* volLd = nullptr;

  if (!mModuleName.IsNull()) {
    snprintf(stavnam, 30, "%s%d", getITSModulePattern(), lay);
    volLd = gGeoManager->GetVolume(stavnam);
  }
  if (!volLd) { // no modules on this layer, check substaves
    if (!mHalfStaveName.IsNull()) {
      snprintf(stavnam, 30, "%s%d", getITSHalfStavePattern(), lay);
      volLd = gGeoManager->GetVolume(stavnam);
    }
  }
  if (!volLd) { // no substaves on this layer, check staves
    snprintf(stavnam, 30, "%s%d", getITSStavePattern(), lay);
    volLd = gGeoManager->GetVolume(stavnam);
  }
  if (!volLd) {
    LOG(FATAL) << "can't find volume containing chips on layer " << lay << FairLogger::endl;
  }

  // Loop on all stave nodes, count Chip volumes by checking names
  Int_t nNodes = volLd->GetNodes()->GetEntries();

  double xmin = 1e9, xmax = -1e9, zmin = 1e9, zmax = -1e9;
  double lab[3], loc[3] = { 0, 0, 0 };
  double dx = -1, dz = -1;

  for (Int_t j = 0; j < nNodes; j++) {
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
        LOG(FATAL) << "Chip " << node->GetName() << " volume is of unprocessed shape " << chShape->IsA()->GetName()
                   << FairLogger::endl;
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
               << "Extracted chip dimensions (x,z): " << dx << " " << dz << " Module Span: " << spanX << " " << spanZ
               << FairLogger::endl;
  }
  return numberOfChips;
}

Int_t GeometryTGeo::extractLayerChipType(Int_t lay) const
{
  char stavnam[30];
  snprintf(stavnam, 30, "%s%d", getITSLayerPattern(), lay);
  TGeoVolume* volLd = gGeoManager->GetVolume(stavnam);
  if (!volLd) {
    LOG(FATAL) << "can't find volume " << stavnam << FairLogger::endl;
    return -1;
  }
  return volLd->GetUniqueID();
}

UInt_t GeometryTGeo::composeChipTypeId(UInt_t segmId)
{
  if (segmId >= kMaxSegmPerChipType) {
    LOG(FATAL) << "Id=" << segmId << " is >= max.allowed " << kMaxSegmPerChipType << FairLogger::endl;
  }
  return segmId + kChipTypePix * kMaxSegmPerChipType;
}

void GeometryTGeo::Print(Option_t*) const
{
  printf("Geometry version %d, NLayers:%d NChips:%d\n", mVersion, mNumberOfLayers, mNumberOfChips);
  if (mVersion == kITSVNA) {
    return;
  }
  for (int i = 0; i < mNumberOfLayers; i++) {
    printf(
      "Lr%2d\tNStav:%2d\tNChips:%2d "
      "(%dx%-2d)\tNMod:%d\tNSubSt:%d\tNSt:%3d\tChipType:%3d\tChip#:%5d:%-5d\tWrapVol:%d\n",
      i, mNumberOfStaves[i], mNumberOfChipsPerModule[i], mNumberOfChipRowsPerModule[i],
      mNumberOfChipRowsPerModule[i] ? mNumberOfChipsPerModule[i] / mNumberOfChipRowsPerModule[i] : 0,
      mNumberOfModules[i], mNumberOfHalfStaves[i], mNumberOfStaves[i], mLayerChipType[i], getFirstChipIndex(i),
      getLastChipIndex(i), mLayerToWrapper[i]);
  }
}

void GeometryTGeo::fetchMatrices()
{
  if (!gGeoManager) {
    LOG(FATAL) << "Geometry is not loaded" << FairLogger::endl;
  }
  mSensorMatrices = new TObjArray(mNumberOfChips);
  mSensorMatrices->SetOwner(kTRUE);
  for (int i = 0; i < mNumberOfChips; i++) {
    mSensorMatrices->AddAt(new TGeoHMatrix(*extractMatrixSensor(i)), i);
  }
  createT2LMatrices();
}

void GeometryTGeo::createT2LMatrices()
{
  // create tracking to local (Sensor!) matrices
  mTrackingToLocalMatrices = new TObjArray(mNumberOfChips);
  mTrackingToLocalMatrices->SetOwner(kTRUE);
  TGeoHMatrix matLtoT;
  double locA[3] = { -100, 0, 0 }, locB[3] = { 100, 0, 0 }, gloA[3], gloB[3];
  for (int isn = 0; isn < mNumberOfChips; isn++) {
    const TGeoHMatrix* matSens = getMatrixSensor(isn);
    if (!matSens) {
      LOG(FATAL) << "Failed to get matrix for sensor " << isn << FairLogger::endl;
      return;
    }
    matSens->LocalToMaster(locA, gloA);
    matSens->LocalToMaster(locB, gloB);
    double dx = gloB[0] - gloA[0];
    double dy = gloB[1] - gloA[1];
    double t = (gloB[0] * dx + gloB[1] * dy) / (dx * dx + dy * dy), x = gloB[0] - dx * t, y = gloB[1] - dy * t;
    auto* t2l = new TGeoHMatrix();
    t2l->RotateZ(ATan2(y, x) * RadToDeg()); // rotate in direction of normal to the sensor plane
    t2l->SetDx(x);
    t2l->SetDy(y);
    t2l->MultiplyLeft(&matSens->Inverse());
    mTrackingToLocalMatrices->AddAt(t2l, isn);
    /*
    const double *gtrans = matSens->GetTranslation();
    memcpy(&rotMatrix[0], matSens->GetRotationMatrix(), 9*sizeof(Double_t));
    Double_t al = -ATan2(rotMatrix[1],rotMatrix[0]);
    Double_t rSens = Sqrt(gtrans[0]*gtrans[0] + gtrans[1]*gtrans[1]);
    Double_t tanAl = ATan2(gtrans[1],gtrans[0]) - Pi()/2; //angle of tangent
    Double_t alTr = tanAl - al;

    // The X axis of tracking frame must always look outward
    loc[1] = rSens/2;
    matSens->LocalToMaster(loc,glo);
    double rPos = Sqrt(glo[0]*glo[0] + glo[1]*glo[1]);
    Bool_t rotOutward = rPos>rSens ? kFALSE : kTRUE;

    // Transformation matrix
    matLtoT.Clear();
    matLtoT.SetDx(-rSens*Sin(alTr)); // translation
    matLtoT.SetDy(0.);
    matLtoT.SetDz(gtrans[2]);
    // Rotation matrix
    rotMatrix[0]= 0;  rotMatrix[1]= 1;  rotMatrix[2]= 0; // + rotation
    rotMatrix[3]=-1;  rotMatrix[4]= 0;  rotMatrix[5]= 0;
    rotMatrix[6]= 0;  rotMatrix[7]= 0;  rotMatrix[8]= 1;

    TGeoRotation rot;
    rot.SetMatrix(rotMatrix);
    matLtoT.MultiplyLeft(&rot);
    if (rotOutward) matLtoT.RotateZ(180.);
    // Inverse transformation Matrix
    mTrackingToLocalMatrices->AddAt(new TGeoHMatrix(matLtoT.Inverse()),isn);
    */
  }
}

//______________________________________________________________________
Int_t GeometryTGeo::extractVolumeCopy(const char* name, const char* prefix) const
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
