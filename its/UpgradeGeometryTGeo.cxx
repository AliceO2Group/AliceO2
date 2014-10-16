/// \file UpgradeGeometryTGeo.cxx
/// \brief Implementation of the UpgradeGeometryTGeo class
/// \author cvetan.cheshkov@cern.ch - 15/02/2007
/// \author ruben.shahoyan@cern.ch - adapted to ITSupg 18/07/2012

// ATTENTION: In opposite to old AliITSgeomTGeo, all indices start from 0, not from 1!!!

#include <TClass.h>
#include <TString.h>
#include <TGeoManager.h>
#include <TGeoPhysicalNode.h>
#include <TGeoShape.h>
#include <TGeoBBox.h>
#include <TDatime.h>
#include <TMath.h>
#include <TSystem.h>

#include "GeometryManager.h"
#include "Segmentation.h"
#include "UpgradeGeometryTGeo.h"
#include "UpgradeSegmentationPixel.h"

#include "FairLogger.h"

using namespace TMath;
using namespace AliceO2::ITS;

ClassImp(UpgradeGeometryTGeo)

UInt_t UpgradeGeometryTGeo::mUIDShift = 16; // bit shift to go from mod.id to modUUID for TGeo
TString UpgradeGeometryTGeo::mVolumeName = "ITSV";
TString UpgradeGeometryTGeo::mLayerName = "ITSULayer";
TString UpgradeGeometryTGeo::mStaveName = "ITSUStave";
TString UpgradeGeometryTGeo::mHalfStaveName = "ITSUHalfStave";
TString UpgradeGeometryTGeo::mModuleName = "ITSUModule";
TString UpgradeGeometryTGeo::mChipName = "ITSUChip";
TString UpgradeGeometryTGeo::mSensorName = "ITSUSensor";
TString UpgradeGeometryTGeo::mWrapperVolumeName = "ITSUWrapVol";
TString UpgradeGeometryTGeo::mChipTypeName[UpgradeGeometryTGeo::kNChipTypes] = { "Pix" };

TString UpgradeGeometryTGeo::mSegmentationFileName = "itsSegmentations.root";

UpgradeGeometryTGeo::UpgradeGeometryTGeo(Bool_t build, Bool_t loadSegmentations)
    : mVersion(kITSVNA)
    , mNumberOfLayers(0)
    , mNumberOfChips(0)
    , mNumberOfStaves(0)
    , mNumberOfHalfStaves(0)
    , mNumberOfModules(0)
    , mNumberOfChipsPerModule(0)
    , mNumberOfChipRowsPerModule(0)
    , mNumberOfChipsPerHalfStave(0)
    , mNumberOfChipsPerStave(0)
    , mNumberOfChipsPerLayer(0)
    , mLayerChipType(0)
    , mLastChipIndex(0)
    , mSensorMatrices(0)
    , mTrackingToLocalMatrices(0)
    , mSegmentations(0)
{
  // default c-tor
  for (int i = gMaxLayers; i--;) {
    mLayerToWrapper[i] = -1;
  }
  if (build) {
    Build(loadSegmentations);
  }
}

UpgradeGeometryTGeo::UpgradeGeometryTGeo(const UpgradeGeometryTGeo& src)
    : TObject(src)
    , mVersion(src.mVersion)
    , mNumberOfLayers(src.mNumberOfLayers)
    , mNumberOfChips(src.mNumberOfChips)
    , mNumberOfStaves(0)
    , mNumberOfHalfStaves(0)
    , mNumberOfModules(0)
    , mNumberOfChipsPerModule(0)
    , mNumberOfChipRowsPerModule(0)
    , mNumberOfChipsPerHalfStave(0)
    , mNumberOfChipsPerStave(0)
    , mNumberOfChipsPerLayer(0)
    , mLayerChipType(0)
    , mLastChipIndex(0)
    , mSensorMatrices(0)
    , mTrackingToLocalMatrices(0)
    , mSegmentations(0)
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

UpgradeGeometryTGeo::~UpgradeGeometryTGeo()
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

UpgradeGeometryTGeo& UpgradeGeometryTGeo::operator=(const UpgradeGeometryTGeo& src)
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
    mNumberOfStaves = mNumberOfHalfStaves = mNumberOfModules = mLayerChipType =
        mNumberOfChipsPerModule = mLastChipIndex = 0;
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

Int_t UpgradeGeometryTGeo::GetChipIndex(Int_t lay, Int_t sta, Int_t chipInStave) const
{
  return GetFirstChipIndex(lay) + mNumberOfChipsPerStave[lay] * sta + chipInStave;
}

Int_t UpgradeGeometryTGeo::GetChipIndex(Int_t lay, Int_t sta, Int_t substa, Int_t chipInSStave)
    const
{
  int n = GetFirstChipIndex(lay) + mNumberOfChipsPerStave[lay] * sta + chipInSStave;
  if (mNumberOfHalfStaves[lay] && substa > 0) {
    n += mNumberOfChipsPerHalfStave[lay] * substa;
  }
  return n;
}

Int_t UpgradeGeometryTGeo::GetChipIndex(Int_t lay, Int_t sta, Int_t substa, Int_t md,
                                        Int_t chipInMod) const
{
  int n = GetFirstChipIndex(lay) + mNumberOfChipsPerStave[lay] * sta + chipInMod;
  if (mNumberOfHalfStaves[lay] && substa > 0) {
    n += mNumberOfChipsPerHalfStave[lay] * substa;
  }
  if (mNumberOfModules[lay] && md > 0) {
    n += mNumberOfChipsPerModule[lay] * md;
  }
  return n;
}

Bool_t UpgradeGeometryTGeo::GetLayer(Int_t index, Int_t& lay, Int_t& indexInLr) const
{
  lay = GetLayer(index);
  indexInLr = index - GetFirstChipIndex(lay);
  return kTRUE;
}

Int_t UpgradeGeometryTGeo::GetLayer(Int_t index) const
{
  int lay = 0;
  while (index > mLastChipIndex[lay]) {
    lay++;
  }
  return lay;
}

Int_t UpgradeGeometryTGeo::GetStave(Int_t index) const
{
  int lay = 0;
  while (index > mLastChipIndex[lay]) {
    lay++;
  }
  index -= GetFirstChipIndex(lay);
  return index / mNumberOfChipsPerStave[lay];
}

Int_t UpgradeGeometryTGeo::GetHalfStave(Int_t index) const
{
  int lay = 0;
  while (index > mLastChipIndex[lay]) {
    lay++;
  }
  if (mNumberOfHalfStaves[lay] < 0) {
    return -1;
  }
  index -= GetFirstChipIndex(lay);
  index %= mNumberOfChipsPerStave[lay];
  return index / mNumberOfChipsPerHalfStave[lay];
}

Int_t UpgradeGeometryTGeo::GetModule(Int_t index) const
{
  int lay = 0;
  while (index > mLastChipIndex[lay]) {
    lay++;
  }
  if (mNumberOfModules[lay] < 0) {
    return 0;
  }
  index -= GetFirstChipIndex(lay);
  index %= mNumberOfChipsPerStave[lay];
  if (mNumberOfHalfStaves[lay]) {
    index %= mNumberOfChipsPerHalfStave[lay];
  }
  return index / mNumberOfChipsPerModule[lay];
}

Int_t UpgradeGeometryTGeo::GetChipIdInLayer(Int_t index) const
{
  int lay = 0;
  while (index > mLastChipIndex[lay]) {
    lay++;
  }
  index -= GetFirstChipIndex(lay);
  return index;
}

Int_t UpgradeGeometryTGeo::GetChipIdInStave(Int_t index) const
{
  int lay = 0;
  while (index > mLastChipIndex[lay]) {
    lay++;
  }
  index -= GetFirstChipIndex(lay);
  return index % mNumberOfChipsPerStave[lay];
}

Int_t UpgradeGeometryTGeo::GetChipIdInHalfStave(Int_t index) const
{
  int lay = 0;
  while (index > mLastChipIndex[lay]) {
    lay++;
  }
  index -= GetFirstChipIndex(lay);
  return index % mNumberOfChipsPerHalfStave[lay];
}

Int_t UpgradeGeometryTGeo::GetChipIdInModule(Int_t index) const
{
  int lay = 0;
  while (index > mLastChipIndex[lay]) {
    lay++;
  }
  index -= GetFirstChipIndex(lay);
  return index % mNumberOfChipsPerModule[lay];
}

Bool_t UpgradeGeometryTGeo::GetChipId(Int_t index, Int_t& lay, Int_t& sta, Int_t& hsta, Int_t& mod,
                                      Int_t& chip) const
{
  lay = GetLayer(index);
  index -= GetFirstChipIndex(lay);
  sta = index / mNumberOfChipsPerStave[lay];
  index %= mNumberOfChipsPerStave[lay];
  hsta = mNumberOfHalfStaves[lay] > 0 ? index / mNumberOfChipsPerHalfStave[lay] : -1;
  index %= mNumberOfChipsPerHalfStave[lay];
  mod = mNumberOfModules[lay] > 0 ? index / mNumberOfChipsPerModule[lay] : -1;
  chip = index % mNumberOfChipsPerModule[lay];

  return kTRUE;
}

const char* UpgradeGeometryTGeo::GetSymbolicName(Int_t index) const
{
  Int_t lay, index2;
  if (!GetLayer(index, lay, index2)) {
    return NULL;
  }
  // return
  // GeometryManager::SymName((GeometryManager::ELayerID)((lay-1)+GeometryManager::kSPD1),index2);
  // RS: this is not optimal, but we cannod access directly GeometryManager, since the latter has
  // hardwired layers
  //  TGeoPNEntry* pne = gGeoManager->GetAlignableEntryByUID(
  // GeometryManager::LayerToVolUID(lay+1,index2) );
  TGeoPNEntry* pne = gGeoManager->GetAlignableEntryByUID(ChipVolUID(index));
  if (!pne) {
    LOG(ERROR) << "Failed to find alignable entry with index " << index << ": (Lr" << lay
               << " Chip:" << index2 << ") !" << FairLogger::endl;
    return NULL;
  }
  return pne->GetName();
}

const char* UpgradeGeometryTGeo::ComposeSymNameITS()
{
  return "ITS";
}

const char* UpgradeGeometryTGeo::ComposeSymNameLayer(Int_t lr)
{
  return Form("%s/%s%d", ComposeSymNameITS(), GetITSLayerPattern(), lr);
}

const char* UpgradeGeometryTGeo::ComposeSymNameStave(Int_t lr, Int_t stave)
{
  return Form("%s/%s%d", ComposeSymNameLayer(lr), GetITSStavePattern(), stave);
}

const char* UpgradeGeometryTGeo::ComposeSymNameHalfStave(Int_t lr, Int_t stave, Int_t substave)
{
  return substave >= 0
             ? Form("%s/%s%d", ComposeSymNameStave(lr, stave), GetITSHalfStavePattern(), substave)
             : ComposeSymNameStave(lr, stave);
}

const char* UpgradeGeometryTGeo::ComposeSymNameModule(Int_t lr, Int_t stave, Int_t substave,
                                                      Int_t mod)
{
  return mod >= 0 ? Form("%s/%s%d", ComposeSymNameHalfStave(lr, stave, substave),
                         GetITSModulePattern(), mod)
                  : ComposeSymNameHalfStave(lr, stave, substave);
}

const char* UpgradeGeometryTGeo::ComposeSymNameChip(Int_t lr, Int_t sta, Int_t substave, Int_t mod,
                                                    Int_t chip)
{
  return Form("%s/%s%d", ComposeSymNameModule(lr, sta, substave, mod), GetITSChipPattern(), chip);
}

TGeoHMatrix* UpgradeGeometryTGeo::GetMatrix(Int_t index) const
{
  static TGeoHMatrix matTmp;
  TGeoPNEntry* pne = GetPNEntry(index);
  if (!pne) {
    return NULL;
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
    return NULL;
  }
  matTmp = *gGeoManager->GetCurrentMatrix();
  gGeoManager->PopPath();
  return &matTmp;
}

Bool_t UpgradeGeometryTGeo::GetTranslation(Int_t index, Double_t t[3]) const
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

Bool_t UpgradeGeometryTGeo::GetRotation(Int_t index, Double_t r[9]) const
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

Bool_t UpgradeGeometryTGeo::GetOriginalMatrix(Int_t index, TGeoHMatrix& m) const
{
  m.Clear();

  const char* symname = GetSymbolicName(index);
  if (!symname) {
    return kFALSE;
  }

  return GeometryManager::GetOriginalGlobalMatrix(symname, m);
}

Bool_t UpgradeGeometryTGeo::GetOriginalTranslation(Int_t index, Double_t t[3]) const
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

Bool_t UpgradeGeometryTGeo::GetOriginalRotation(Int_t index, Double_t r[9]) const
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

TGeoHMatrix* UpgradeGeometryTGeo::ExtractMatrixTrackingToLocal(Int_t index) const
{
  TGeoPNEntry* pne = GetPNEntry(index);
  if (!pne) {
    return NULL;
  }

  TGeoHMatrix* m = (TGeoHMatrix*)pne->GetMatrix();
  if (!m) {
    LOG(ERROR) << "TGeoPNEntry (" << pne->GetName() << ") contains no matrix !" << FairLogger::endl;
  }

  return m;
}

Bool_t UpgradeGeometryTGeo::GetTrackingMatrix(Int_t index, TGeoHMatrix& m)
{
  m.Clear();

  TGeoHMatrix* m1 = GetMatrix(index);
  if (!m1) {
    return kFALSE;
  }

  const TGeoHMatrix* m2 = GetMatrixT2L(index);
  if (!m2) {
    return kFALSE;
  }

  m = *m1;
  m.Multiply(m2);

  return kTRUE;
}

TGeoHMatrix* UpgradeGeometryTGeo::ExtractMatrixSensor(Int_t index) const
{
  Int_t lay, stav, sstav, mod, chipInMod;
  GetChipId(index, lay, stav, sstav, mod, chipInMod);

  int wrID = mLayerToWrapper[lay];

  TString path = Form("/cave_1/%s_2/", UpgradeGeometryTGeo::GetITSVolPattern());

  if (wrID >= 0) {
    path += Form("%s%d_1/", GetITSWrapVolPattern(), wrID);
  }

  path += Form("%s%d_1/%s%d_%d/", UpgradeGeometryTGeo::GetITSLayerPattern(), lay,
               UpgradeGeometryTGeo::GetITSStavePattern(), lay, stav);

  if (mNumberOfHalfStaves[lay] > 0) {
    path += Form("%s%d_%d/", UpgradeGeometryTGeo::GetITSHalfStavePattern(), lay, sstav);
  }
  if (mNumberOfModules[lay] > 0) {
    path += Form("%s%d_%d/", UpgradeGeometryTGeo::GetITSModulePattern(), lay, mod);
  }
  path += Form("%s%d_%d/%s%d_1", UpgradeGeometryTGeo::GetITSChipPattern(), lay, chipInMod,
               UpgradeGeometryTGeo::GetITSSensorPattern(), lay);

  static TGeoHMatrix matTmp;
  gGeoManager->PushPath();

  if (!gGeoManager->cd(path.Data())) {
    gGeoManager->PopPath();
    LOG(ERROR) << "Error in cd-ing to " << path.Data() << FairLogger::endl;
    return 0;
  } // end if !gGeoManager

  matTmp = *gGeoManager->GetCurrentMatrix(); // matrix may change after cd
  // RSS
  //  printf("%d/%d/%d %s\n",lay,stav,detInSta,path.Data());
  //  mat->Print();
  // Restore the modeler state.
  gGeoManager->PopPath();
  return &matTmp;
}

TGeoPNEntry* UpgradeGeometryTGeo::GetPNEntry(Int_t index) const
{
  if (index >= mNumberOfChips) {
    LOG(ERROR) << "Invalid ITS chip index: " << index << " (0 -> " << mNumberOfChips << ") !"
               << FairLogger::endl;
    return NULL;
  }

  if (!gGeoManager || !gGeoManager->IsClosed()) {
    LOG(ERROR) << "Can't get the matrix! gGeoManager doesn't exist or it is still opened!"
               << FairLogger::endl;
    return NULL;
  }
  TGeoPNEntry* pne = gGeoManager->GetAlignableEntryByUID(ChipVolUID(index));
  //  TGeoPNEntry* pne = gGeoManager->GetAlignableEntry(GetSymbolicName(index));

  if (!pne) {
    LOG(ERROR) << "The index " << index << " does not correspond to a physical entry!"
               << FairLogger::endl;
  }
  return pne;
}

void UpgradeGeometryTGeo::Build(Bool_t loadSegmentations)
{
  if (mVersion != kITSVNA) {
    LOG(WARNING) << "Already built" << FairLogger::endl;
    return; // already initialized
  }
  if (!gGeoManager) {
    LOG(FATAL) << "Geometry is not loaded" << FairLogger::endl;
  }

  mNumberOfLayers = ExtractNumberOfLayers();
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
    mLayerChipType[i] = ExtractLayerChipType(i);
    mNumberOfStaves[i] = ExtractNumberOfStaves(i);
    mNumberOfHalfStaves[i] = ExtractNumberOfHalfStaves(i);
    mNumberOfModules[i] = ExtractNumberOfModules(i);
    mNumberOfChipsPerModule[i] = ExtractNumberOfChipsPerModule(i, mNumberOfChipRowsPerModule[i]);
    mNumberOfChipsPerHalfStave[i] = mNumberOfChipsPerModule[i] * Max(1, mNumberOfModules[i]);
    mNumberOfChipsPerStave[i] = mNumberOfChipsPerHalfStave[i] * Max(1, mNumberOfHalfStaves[i]);
    mNumberOfChipsPerLayer[i] = mNumberOfChipsPerStave[i] * mNumberOfStaves[i];
    mNumberOfChips += mNumberOfChipsPerLayer[i];
    mLastChipIndex[i] = mNumberOfChips - 1;
  }

  FetchMatrices();
  mVersion = kITSVUpg;

  if (loadSegmentations) { // fetch segmentations
    mSegmentations = new TObjArray();
    UpgradeSegmentationPixel::LoadSegmentations(mSegmentations, GetITSsegmentationFileName());
  }
}

Int_t UpgradeGeometryTGeo::ExtractNumberOfLayers()
{
  Int_t numberOfLayers = 0;

  TGeoVolume* itsV = gGeoManager->GetVolume(GetITSVolPattern());
  if (!itsV) {
    LOG(FATAL) << "ITS volume " << GetITSVolPattern() << " is not in the geometry"
               << FairLogger::endl;
  }
  SetUIDShift(itsV->GetUniqueID());

  // Loop on all ITSV nodes, count Layer volumes by checking names
  // Build on the fly layer - wrapper correspondence
  TObjArray* nodes = itsV->GetNodes();
  Int_t nNodes = nodes->GetEntriesFast();

  for (Int_t j = 0; j < nNodes; j++) {
    int lrID = -1;
    TGeoNode* nd = (TGeoNode*)nodes->At(j);
    const char* name = nd->GetName();

    if (strstr(name, GetITSLayerPattern())) {
      numberOfLayers++;
      if ((lrID = ExtractVolumeCopy(name, UpgradeGeometryTGeo::GetITSLayerPattern())) < 0) {
        LOG(FATAL) << "Failed to extract layer ID from the " << name << FairLogger::endl;
        exit(1);
      }

      mLayerToWrapper[lrID] = -1; // not wrapped
    }
    else if (strstr(name,
                      GetITSWrapVolPattern())) { // this is a wrapper volume, may cointain layers
      int wrID = -1;
      if ((wrID = ExtractVolumeCopy(name, UpgradeGeometryTGeo::GetITSWrapVolPattern())) < 0) {
        LOG(FATAL) << "Failed to extract wrapper ID from the " << name << FairLogger::endl;
        exit(1);
      }

      TObjArray* nodesW = nd->GetNodes();
      int nNodesW = nodesW->GetEntriesFast();

      for (Int_t jw = 0; jw < nNodesW; jw++) {
        TGeoNode* ndW = (TGeoNode*)nodesW->At(jw);
        if (strstr(ndW->GetName(), GetITSLayerPattern())) {
          if ((lrID = ExtractVolumeCopy(ndW->GetName(),
                                        UpgradeGeometryTGeo::GetITSLayerPattern())) < 0) {
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

Int_t UpgradeGeometryTGeo::ExtractNumberOfStaves(Int_t lay) const
{
  Int_t numberOfStaves = 0;
  char laynam[30];
  snprintf(laynam, 30, "%s%d", GetITSLayerPattern(), lay);
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
    //           << GetITSStavePattern() << " -> " << numberOfStaves << FairLogger::endl;
    if (strstr(volLr->GetNodes()->At(j)->GetName(), GetITSStavePattern())) {
      numberOfStaves++;
    }
  }
  return numberOfStaves;
}

Int_t UpgradeGeometryTGeo::ExtractNumberOfHalfStaves(Int_t lay) const
{
  if (mHalfStaveName.IsNull()) {
    return 0; // for the setup w/o substave defined the stave and the substave is the same thing
  }
  Int_t nSS = 0;
  char stavnam[30];
  snprintf(stavnam, 30, "%s%d", GetITSStavePattern(), lay);
  TGeoVolume* volLd = gGeoManager->GetVolume(stavnam);
  if (!volLd) {
    LOG(FATAL) << "can't find volume " << stavnam << FairLogger::endl;
  }
  // Loop on all stave nodes, count Chip volumes by checking names
  Int_t nNodes = volLd->GetNodes()->GetEntries();
  for (Int_t j = 0; j < nNodes; j++) {
    if (strstr(volLd->GetNodes()->At(j)->GetName(), GetITSHalfStavePattern())) {
      nSS++;
    }
  }
  return nSS;
}

Int_t UpgradeGeometryTGeo::ExtractNumberOfModules(Int_t lay) const
{
  if (mModuleName.IsNull()) {
    return 0;
  }

  char stavnam[30];
  TGeoVolume* volLd = 0;

  if (!mHalfStaveName.IsNull()) {
    snprintf(stavnam, 30, "%s%d", GetITSHalfStavePattern(), lay);
    volLd = gGeoManager->GetVolume(stavnam);
  }
  if (!volLd) { // no substaves, check staves
    snprintf(stavnam, 30, "%s%d", GetITSStavePattern(), lay);
    volLd = gGeoManager->GetVolume(stavnam);
  }
  if (!volLd) {
    return 0;
  }

  Int_t nMod = 0;

  // Loop on all substave nodes, count module volumes by checking names
  Int_t nNodes = volLd->GetNodes()->GetEntries();

  for (Int_t j = 0; j < nNodes; j++) {
    if (strstr(volLd->GetNodes()->At(j)->GetName(), GetITSModulePattern())) {
      nMod++;
    }
  }
  return nMod;
}

Int_t UpgradeGeometryTGeo::ExtractNumberOfChipsPerModule(Int_t lay, int& nrow) const
{
  Int_t numberOfChips = 0;
  char stavnam[30];
  TGeoVolume* volLd = 0;

  if (!mModuleName.IsNull()) {
    snprintf(stavnam, 30, "%s%d", GetITSModulePattern(), lay);
    volLd = gGeoManager->GetVolume(stavnam);
  }
  if (!volLd) { // no modules on this layer, check substaves
    if (!mHalfStaveName.IsNull()) {
      snprintf(stavnam, 30, "%s%d", GetITSHalfStavePattern(), lay);
      volLd = gGeoManager->GetVolume(stavnam);
    }
  }
  if (!volLd) { // no substaves on this layer, check staves
    snprintf(stavnam, 30, "%s%d", GetITSStavePattern(), lay);
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
    if (!strstr(node->GetName(), GetITSChipPattern())) {
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
        LOG(FATAL) << "Chip " << node->GetName() << " volume is of unprocessed shape "
                   << chShape->IsA()->GetName() << FairLogger::endl;
      }
      else {
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
    LOG(ERROR) << "Inconsistency between Nchips=" << numberOfChips << " and Nrow*Ncol=" << nrow
               << "*" << ncol << "->" << nrow * ncol << FairLogger::endl
               << "Extracted chip dimensions (x,z): " << dx << " " << dz
               << " Module Span: " << spanX << " " << spanZ << FairLogger::endl;
  }
  return numberOfChips;
}

Int_t UpgradeGeometryTGeo::ExtractLayerChipType(Int_t lay) const
{
  char stavnam[30];
  snprintf(stavnam, 30, "%s%d", GetITSLayerPattern(), lay);
  TGeoVolume* volLd = gGeoManager->GetVolume(stavnam);
  if (!volLd) {
    LOG(FATAL) << "can't find volume " << stavnam << FairLogger::endl;
    return -1;
  }
  return volLd->GetUniqueID();
}

UInt_t UpgradeGeometryTGeo::ComposeChipTypeId(UInt_t segmId)
{
  if (segmId >= kMaxSegmPerChipType) {
    LOG(FATAL) << "Id=" << segmId << " is >= max.allowed " << kMaxSegmPerChipType
               << FairLogger::endl;
  }
  return segmId + kChipTypePix * kMaxSegmPerChipType;
}

void UpgradeGeometryTGeo::Print(Option_t*) const
{
  printf("Geometry version %d, NLayers:%d NChips:%d\n", mVersion, mNumberOfLayers, mNumberOfChips);
  if (mVersion == kITSVNA) {
    return;
  }
  for (int i = 0; i < mNumberOfLayers; i++) {
    printf("Lr%2d\tNStav:%2d\tNChips:%2d "
           "(%dx%-2d)\tNMod:%d\tNSubSt:%d\tNSt:%3d\tChipType:%3d\tChip#:%5d:%-5d\tWrapVol:%d\n",
           i, mNumberOfStaves[i], mNumberOfChipsPerModule[i], mNumberOfChipRowsPerModule[i],
           mNumberOfChipRowsPerModule[i]
               ? mNumberOfChipsPerModule[i] / mNumberOfChipRowsPerModule[i]
               : 0,
           mNumberOfModules[i], mNumberOfHalfStaves[i], mNumberOfStaves[i], mLayerChipType[i],
           GetFirstChipIndex(i), GetLastChipIndex(i), mLayerToWrapper[i]);
  }
}

void UpgradeGeometryTGeo::FetchMatrices()
{
  if (!gGeoManager) {
    LOG(FATAL) << "Geometry is not loaded" << FairLogger::endl;
  }
  mSensorMatrices = new TObjArray(mNumberOfChips);
  mSensorMatrices->SetOwner(kTRUE);
  for (int i = 0; i < mNumberOfChips; i++) {
    mSensorMatrices->AddAt(new TGeoHMatrix(*ExtractMatrixSensor(i)), i);
  }
  CreateT2LMatrices();
}

void UpgradeGeometryTGeo::CreateT2LMatrices()
{
  // create tracking to local (Sensor!) matrices
  mTrackingToLocalMatrices = new TObjArray(mNumberOfChips);
  mTrackingToLocalMatrices->SetOwner(kTRUE);
  TGeoHMatrix matLtoT;
  double locA[3] = { -100, 0, 0 }, locB[3] = { 100, 0, 0 }, gloA[3], gloB[3];
  for (int isn = 0; isn < mNumberOfChips; isn++) {
    const TGeoHMatrix* matSens = GetMatrixSensor(isn);
    if (!matSens) {
      LOG(FATAL) << "Failed to get matrix for sensor " << isn << FairLogger::endl;
      return;
    }
    matSens->LocalToMaster(locA, gloA);
    matSens->LocalToMaster(locB, gloB);
    double dx = gloB[0] - gloA[0];
    double dy = gloB[1] - gloA[1];
    double t = (gloB[0] * dx + gloB[1] * dy) / (dx * dx + dy * dy), x = gloB[0] - dx * t,
           y = gloB[1] - dy * t;
    TGeoHMatrix* t2l = new TGeoHMatrix();
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
Int_t UpgradeGeometryTGeo::ExtractVolumeCopy(const char* name, const char* prefix) const
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
