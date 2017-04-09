/// \file SegmentationPixel.cxx
/// \brief Implementation of the SegmentationPixel class

#include "ITSMFTBase/SegmentationPixel.h"
//#include "ITSBase/GeometryTGeo.h"

#include <TFile.h>     // for TFile
#include <TObjArray.h> // for TObjArray
#include <TString.h>   // for TString
#include <TSystem.h>   // for TSystem, gSystem
#include <cstdio>     // for printf
#include "TMathBase.h" // for Abs, Max, Min
#include "TObject.h"   // for TObject

using namespace TMath;
using namespace o2::ITSMFT;

ClassImp(o2::ITSMFT::SegmentationPixel)

  const char* SegmentationPixel::sSegmentationsListName = "Segmentations";

SegmentationPixel::SegmentationPixel(UInt_t id, int nchips, int ncol, int nrow, float pitchX, float pitchZ,
                                     float thickness, float pitchLftC, float pitchRgtC, float edgL, float edgR,
                                     float edgT, float edgB)
  : Segmentation(),
    mGuardLeft(edgL),
    mGuardRight(edgR),
    mGuardTop(edgT),
    mGuardBottom(edgB),
    mShiftLocalX(0.5 * (edgT - edgB)),
    mShiftLocalZ(0.5 * (edgR - edgL)),
    mDxActive(0),
    mDzActive(0),
    mPitchX(pitchX),
    mPitchZ(pitchZ),
    mPitchZLeftColumn(pitchLftC < 0 ? pitchZ : pitchLftC),
    mPitchZRightColumn(pitchRgtC < 0 ? pitchZ : pitchRgtC),
    mChipSizeDZ(0),
    mNumberOfChips(nchips),
    mNumberOfColumnsPerChip(nchips > 0 ? ncol / nchips : 0),
    mNumberOfRows(nrow),
    mNumberOfColumns(ncol),
    mDiodShiftMatNColumn(0),
    mDiodShiftMatNRow(0),
    mDiodShiftMatDimension(0),
    mDiodShiftMatX(nullptr),
    mDiodShiftMatZ(nullptr)
{
  // Default constructor, sizes in cm

 if (nchips) {
//   SetUniqueID(GeometryTGeo::composeChipTypeId(id));
   SetUniqueID(id);
 }

  mChipSizeDZ = (mNumberOfColumnsPerChip - 2) * mPitchZ + mPitchZLeftColumn + mPitchZRightColumn;
  mDxActive = mNumberOfRows * mPitchX;
  mDzActive = mNumberOfChips * mChipSizeDZ;
  setDetectorSize(mDxActive + mGuardTop + mGuardBottom, mDzActive + mGuardLeft + mGuardRight, thickness);
}

SegmentationPixel::~SegmentationPixel()
{
  // d-tor
  delete[] mDiodShiftMatX;
  delete[] mDiodShiftMatZ;
}

void SegmentationPixel::getPadIxz(Float_t x, Float_t z, Int_t& ix, Int_t& iz) const
{
  ix = int(x / mPitchX);
  iz = int(zToColumn(z));

  if (iz < 0) {
    LOG(WARNING) << "Z=" << z << " gives col=" << iz << " outside [0:" << mNumberOfColumns << ")" << FairLogger::endl;
    iz = 0;
  } else if (iz >= mNumberOfColumns) {
    LOG(WARNING) << "Z=" << z << " gives col=" << iz << " outside [0:" << mNumberOfColumns << ")" << FairLogger::endl;
    iz = mNumberOfColumns - 1;
  }
  if (ix < 0) {
    LOG(WARNING) << "X=" << x << " gives row=" << ix << " outside [0:" << mNumberOfRows << ")" << FairLogger::endl;
    ix = 0;
  } else if (ix >= mNumberOfRows) {
    LOG(WARNING) << "X=" << x << " gives row=" << ix << " outside [0:" << mNumberOfRows << ")" << FairLogger::endl;
    ix = mNumberOfRows - 1;
  }
}

void SegmentationPixel::getPadTxz(Float_t& x, Float_t& z) const
{
  x /= mPitchX;
  z = zToColumn(z);
}

void SegmentationPixel::getPadCxz(Int_t ix, Int_t iz, Float_t& x, Float_t& z) const
{
  x = Float_t((ix + 0.5) * mPitchX);
  z = columnToZ(iz);
}

Float_t SegmentationPixel::zToColumn(Float_t z) const
{
  int chip = int(z / mChipSizeDZ);
  float col = chip * mNumberOfColumnsPerChip;
  z -= chip * mChipSizeDZ;
  if (z > mPitchZLeftColumn) {
    col += 1 + (z - mPitchZLeftColumn) / mPitchZ;
  }
  return col;
}

Float_t SegmentationPixel::columnToZ(Int_t col) const
{
  int nchip = col / mNumberOfColumnsPerChip;
  col %= mNumberOfColumnsPerChip;
  float z = nchip * mChipSizeDZ;
  if (col > 0) {
    if (col < mNumberOfColumnsPerChip - 1) {
      z += mPitchZLeftColumn + (col - 0.5) * mPitchZ;
    } else {
      z += mChipSizeDZ - mPitchZRightColumn / 2;
    }
  } else {
    z += mPitchZLeftColumn / 2;
  }
  return z;
}

SegmentationPixel& SegmentationPixel::operator=(const SegmentationPixel& src)
{
  if (this == &src) {
    return *this;
  }
  Segmentation::operator=(src);
  mNumberOfColumns = src.mNumberOfColumns;
  mNumberOfRows = src.mNumberOfRows;
  mNumberOfColumnsPerChip = src.mNumberOfColumnsPerChip;
  mNumberOfChips = src.mNumberOfChips;
  mChipSizeDZ = src.mChipSizeDZ;
  mPitchZRightColumn = src.mPitchZRightColumn;
  mPitchZLeftColumn = src.mPitchZLeftColumn;
  mPitchZ = src.mPitchZ;
  mPitchX = src.mPitchX;
  mShiftLocalX = src.mShiftLocalX;
  mShiftLocalZ = src.mShiftLocalZ;
  mDxActive = src.mDxActive;
  mDzActive = src.mDzActive;

  mGuardBottom = src.mGuardBottom;
  mGuardTop = src.mGuardTop;
  mGuardRight = src.mGuardRight;
  mGuardLeft = src.mGuardLeft;

  mDiodShiftMatNColumn = src.mDiodShiftMatNColumn;
  mDiodShiftMatNRow = src.mDiodShiftMatNRow;
  mDiodShiftMatDimension = src.mDiodShiftMatDimension;
  delete mDiodShiftMatX;
  mDiodShiftMatX = nullptr;
  delete mDiodShiftMatZ;
  mDiodShiftMatZ = nullptr;
  if (mDiodShiftMatDimension) {
    mDiodShiftMatX = new Float_t[mDiodShiftMatDimension];
    mDiodShiftMatZ = new Float_t[mDiodShiftMatDimension];
    for (int i = mDiodShiftMatDimension; i--;) {
      mDiodShiftMatX[i] = src.mDiodShiftMatX[i];
      mDiodShiftMatZ[i] = src.mDiodShiftMatZ[i];
    }
  }
  return *this;
}

SegmentationPixel::SegmentationPixel(const SegmentationPixel& src)
  : Segmentation(src),
    mGuardLeft(src.mGuardLeft),
    mGuardRight(src.mGuardRight),
    mGuardTop(src.mGuardTop),
    mGuardBottom(src.mGuardBottom),
    mShiftLocalX(src.mShiftLocalX),
    mShiftLocalZ(src.mShiftLocalZ),
    mDxActive(src.mDxActive),
    mDzActive(src.mDzActive),
    mPitchX(src.mPitchX),
    mPitchZ(src.mPitchZ),
    mPitchZLeftColumn(src.mPitchZLeftColumn),
    mPitchZRightColumn(src.mPitchZRightColumn),
    mChipSizeDZ(src.mChipSizeDZ),
    mNumberOfChips(src.mNumberOfChips),
    mNumberOfColumnsPerChip(src.mNumberOfColumnsPerChip),
    mNumberOfRows(src.mNumberOfRows),
    mNumberOfColumns(src.mNumberOfColumns),
    mDiodShiftMatNColumn(src.mDiodShiftMatNColumn),
    mDiodShiftMatNRow(src.mDiodShiftMatNRow),
    mDiodShiftMatDimension(src.mDiodShiftMatDimension),
    mDiodShiftMatX(nullptr),
    mDiodShiftMatZ(nullptr)
{
  // copy constructor
  if (mDiodShiftMatDimension) {
    mDiodShiftMatX = new Float_t[mDiodShiftMatDimension];
    mDiodShiftMatZ = new Float_t[mDiodShiftMatDimension];
    for (int i = mDiodShiftMatDimension; i--;) {
      mDiodShiftMatX[i] = src.mDiodShiftMatX[i];
      mDiodShiftMatZ[i] = src.mDiodShiftMatZ[i];
    }
  }
}

Float_t SegmentationPixel::cellSizeX(Int_t) const { return mPitchX; }
Float_t SegmentationPixel::cellSizeZ(Int_t col) const
{
  col %= mNumberOfColumnsPerChip;
  if (!col) {
    return mPitchZLeftColumn;
  }
  if (col == mNumberOfColumnsPerChip - 1) {
    return mPitchZRightColumn;
  }
  return mPitchZ;
}

void SegmentationPixel::neighbours(Int_t iX, Int_t iZ, Int_t* nlist, Int_t xlist[8], Int_t zlist[8]) const
{
  *nlist = 8;
  xlist[0] = xlist[1] = iX;
  xlist[2] = iX - 1;
  xlist[3] = iX + 1;
  zlist[0] = iZ - 1;
  zlist[1] = iZ + 1;
  zlist[2] = zlist[3] = iZ;

  // Diagonal elements
  xlist[4] = iX + 1;
  zlist[4] = iZ + 1;

  xlist[5] = iX - 1;
  zlist[5] = iZ - 1;

  xlist[6] = iX - 1;
  zlist[6] = iZ + 1;

  xlist[7] = iX + 1;
  zlist[7] = iZ - 1;
}

Bool_t SegmentationPixel::localToDetector(Float_t x, Float_t z, Int_t& ix, Int_t& iz) const
{
  x += 0.5 * dxActive() + mShiftLocalX; // get X,Z wrt bottom/left corner
  z += 0.5 * dzActive() + mShiftLocalZ;
  ix = iz = -1;
  if (x < 0 || x > dxActive()) {
    // throw OutOfActiveAreaException(OutOfActiveAreaException::kX, x, 0, dxActive());
    return kFALSE;
  }
  if (z < 0 || z > dzActive()) {
    // throw OutOfActiveAreaException(OutOfActiveAreaException::kZ, z, 0, dzActive());
    return kFALSE;
  }
  ix = int(x / mPitchX);
  iz = zToColumn(z);
  return kTRUE; // Found ix and iz, return.
}

Bool_t SegmentationPixel::detectorToLocal(Int_t ix, Int_t iz, Float_t& x, Float_t& z) const
{
  x = -0.5 * dxActive(); // default value.
  z = -0.5 * dzActive(); // default value.
  if (ix < 0 || ix >= mNumberOfRows) {
    // throw InvalidPixelException(InvalidPixelException::kX, ix, mNumberOfRows);
    return kFALSE;
  } // outside of detector
  if (iz < 0 || iz >= mNumberOfColumns) {
    // throw InvalidPixelException(InvalidPixelException::kZ, iz, mNumberOfColumns);
    return kFALSE;
  }                                         // outside of detector
  x += (ix + 0.5) * mPitchX - mShiftLocalX; // RS: we go to the center of the pad, i.e. + pitch/2, not
  // to the boundary as in SPD
  z += columnToZ(iz) - mShiftLocalZ;
  return kTRUE;
}

void SegmentationPixel::cellBoundries(Int_t ix, Int_t iz, Double_t& xl, Double_t& xu, Double_t& zl, Double_t& zu) const
{
  Float_t x, z;
  detectorToLocal(ix, iz, x, z);

  if (ix < 0 || ix >= mNumberOfRows || iz < 0 || iz >= mNumberOfColumns) {
    xl = xu = -0.5 * Dx(); // default value.
    zl = zu = -0.5 * Dz(); // default value.
    return;                // outside of detctor
  }
  float zpitchH = cellSizeZ(iz) * 0.5;
  float xpitchH = mPitchX * 0.5;
  xl -= xpitchH;
  xu += xpitchH;
  zl -= zpitchH;
  zu += zpitchH;
  return; // Found x and z, return.
}

Int_t SegmentationPixel::getChipFromChannel(Int_t, Int_t iz) const
{
  if (iz >= mNumberOfColumns || iz < 0) {
    throw InvalidPixelException(InvalidPixelException::kZ, iz, mNumberOfColumns);
  }
  return iz / mNumberOfColumnsPerChip;
}

Int_t SegmentationPixel::getChipFromLocal(Float_t, Float_t zloc) const
{
  Int_t ix0, iz;
  try {
    localToDetector(0, zloc, ix0, iz);
  } catch (OutOfActiveAreaException& e) {
    LOG(WARNING) << "Bad local coordinate" << FairLogger::endl;
    return -1;
  }
  return getChipFromChannel(ix0, iz);
}

Int_t SegmentationPixel::getChipsInLocalWindow(Int_t* array, Float_t zmin, Float_t zmax, Float_t, Float_t) const
{
  if (zmin > zmax) {
    LOG(WARNING) << "Bad coordinate limits: zmin>zmax!" << FairLogger::endl;
    return -1;
  }

  Int_t nChipInW = 0;

  Float_t zminDet = -0.5 * dzActive() - mShiftLocalZ;
  Float_t zmaxDet = 0.5 * dzActive() - mShiftLocalZ;
  if (zmin < zminDet) {
    zmin = zminDet;
  }
  if (zmax > zmaxDet) {
    zmax = zmaxDet;
  }

  Int_t n1 = getChipFromLocal(0, zmin);
  array[nChipInW] = n1;
  nChipInW++;

  Int_t n2 = getChipFromLocal(0, zmax);

  if (n2 != n1) {
    Int_t imin = Min(n1, n2);
    Int_t imax = Max(n1, n2);
    for (Int_t ichip = imin; ichip <= imax; ichip++) {
      if (ichip == n1) {
        continue;
      }
      array[nChipInW] = ichip;
      nChipInW++;
    }
  }
  return nChipInW;
}

void SegmentationPixel::Init()
{
  // init settings
}

Bool_t SegmentationPixel::Store(const char* outf)
{
  TString fns = outf;
  gSystem->ExpandPathName(fns);

  if (fns.IsNull()) {
    LOG(FATAL) << "No file name provided" << FairLogger::endl;
    return kFALSE;
  }

  TFile* fout = TFile::Open(fns.Data(), "update");

  if (!fout) {
    LOG(FATAL) << "Failed to open output file " << outf << FairLogger::endl;
    return kFALSE;
  }

  TObjArray* arr = (TObjArray*)fout->Get(sSegmentationsListName);

  int id = GetUniqueID();

  if (!arr) {
    arr = new TObjArray();
  } else if (arr->At(id)) {
    LOG(FATAL) << "Segmentation " << id << " already exists in file " << outf << FairLogger::endl;
    return kFALSE;
  }

  arr->AddAtAndExpand(this, id);
  arr->SetOwner(kTRUE);
  fout->WriteObject(arr, sSegmentationsListName, "kSingleKey");
  fout->Close();
  delete fout;
  arr->RemoveAt(id);
  delete arr;
  LOG(INFO) << "Stored segmentation " << id << " in " << outf << FairLogger::endl;
  return kTRUE;
}

SegmentationPixel* SegmentationPixel::loadWithId(UInt_t id, const char* inpf)
{
  TString fns = inpf;
  gSystem->ExpandPathName(fns);
  if (fns.IsNull()) {
    LOG(FATAL) << "LoadWithId: No file name provided" << FairLogger::endl;
    return nullptr;
  }
  TFile* finp = TFile::Open(fns.Data());
  if (!finp) {
    LOG(FATAL) << "LoadWithId: Failed to open file " << inpf << FairLogger::endl;
    return nullptr;
  }
  TObjArray* arr = (TObjArray*)finp->Get(sSegmentationsListName);
  if (!arr) {
    LOG(FATAL) << "LoadWithId: Failed to find segmenation array " << sSegmentationsListName << " in " << inpf
               << FairLogger::endl;
    return nullptr;
  }
  SegmentationPixel* segm = dynamic_cast<SegmentationPixel*>(arr->At(id));
  if (!segm || segm->GetUniqueID() != id) {
    LOG(FATAL) << "LoadWithId: Failed to find segmenation " << id << " in " << inpf << FairLogger::endl;
    return nullptr;
  }

  arr->RemoveAt(id);
  arr->SetOwner(kTRUE); // to not leave in memory other segmenations
  finp->Close();
  delete finp;
  delete arr;

  return segm;
}

void SegmentationPixel::loadSegmentations(TObjArray* dest, const char* inpf)
{
  if (!dest) {
    return;
  }
  TString fns = inpf;
  gSystem->ExpandPathName(fns);
  if (fns.IsNull()) {
    LOG(FATAL) << "LoadWithId: No file name provided" << FairLogger::endl;
  }
  TFile* finp = TFile::Open(fns.Data());
  if (!finp) {
    LOG(FATAL) << "LoadWithId: Failed to open file " << inpf << FairLogger::endl;
  }
  TObjArray* arr = (TObjArray*)finp->Get(sSegmentationsListName);
  if (!arr) {
    LOG(FATAL) << "LoadWithId: Failed to find segmentation array " << sSegmentationsListName << " in " << inpf
               << FairLogger::endl;
  }
  int nent = arr->GetEntriesFast();
  TObject* segm = nullptr;
  for (int i = nent; i--;) {
    if ((segm = arr->At(i))) {
      dest->AddAtAndExpand(segm, segm->GetUniqueID());
    }
  }
  LOG(INFO) << "LoadSegmentations: Loaded " << arr->GetEntries() << " segmentations from " << inpf << FairLogger::endl;
  arr->SetOwner(kFALSE);
  arr->Clear();
  finp->Close();
  delete finp;
  delete arr;
}

void SegmentationPixel::setDiodShiftMatrix(Int_t nrow, Int_t ncol, const Float_t* shiftX, const Float_t* shiftZ)
{
  if (mDiodShiftMatDimension) {
    delete mDiodShiftMatX;
    delete mDiodShiftMatZ;
    mDiodShiftMatX = mDiodShiftMatZ = nullptr;
  }
  mDiodShiftMatNColumn = ncol;
  mDiodShiftMatNRow = nrow;
  mDiodShiftMatDimension = mDiodShiftMatNColumn * mDiodShiftMatNRow;
  if (mDiodShiftMatDimension) {
    mDiodShiftMatX = new Float_t[mDiodShiftMatDimension];
    mDiodShiftMatZ = new Float_t[mDiodShiftMatDimension];
    for (int ir = 0; ir < mDiodShiftMatNRow; ir++) {
      for (int ic = 0; ic < mDiodShiftMatNColumn; ic++) {
        int cnt = ic + ir * mDiodShiftMatNColumn;
        mDiodShiftMatX[cnt] = shiftX ? shiftX[cnt] : 0.;
        mDiodShiftMatZ[cnt] = shiftZ ? shiftZ[cnt] : 0.;
      }
    }
  }
}

void SegmentationPixel::setDiodShiftMatrix(Int_t nrow, Int_t ncol, const Double_t* shiftX, const Double_t* shiftZ)
{
  if (mDiodShiftMatDimension) {
    delete mDiodShiftMatX;
    delete mDiodShiftMatZ;
    mDiodShiftMatX = mDiodShiftMatZ = nullptr;
  }

  mDiodShiftMatNColumn = ncol;
  mDiodShiftMatNRow = nrow;
  mDiodShiftMatDimension = mDiodShiftMatNColumn * mDiodShiftMatNRow;
  if (mDiodShiftMatDimension) {
    mDiodShiftMatX = new Float_t[mDiodShiftMatDimension];
    mDiodShiftMatZ = new Float_t[mDiodShiftMatDimension];
    for (int ir = 0; ir < mDiodShiftMatNRow; ir++) {
      for (int ic = 0; ic < mDiodShiftMatNColumn; ic++) {
        int cnt = ic + ir * mDiodShiftMatNColumn;
        mDiodShiftMatX[cnt] = shiftX ? shiftX[cnt] : 0.;
        mDiodShiftMatZ[cnt] = shiftZ ? shiftZ[cnt] : 0.;
      }
    }
  }
}

void SegmentationPixel::Print(Option_t* /*option*/) const
{
  const double kmc = 1e4;
  printf("Segmentation %d: Active Size: DX: %.1f DY: %.1f DZ: %.1f | Pitch: X:%.1f Z:%.1f\n", GetUniqueID(),
         kmc * dxActive(), kmc * Dy(), kmc * dzActive(), kmc * cellSizeX(1), kmc * cellSizeZ(1));
  printf(
    "Passive Edges: Bottom: %.1f Right: %.1f Top: %.1f Left: %.1f -> DX: %.1f DZ: %.1f Shift: "
    "x:%.1f z:%.1f\n",
    kmc * mGuardBottom, kmc * mGuardRight, kmc * mGuardTop, kmc * mGuardLeft, kmc * Dx(), kmc * Dz(),
    kmc * mShiftLocalX, kmc * mShiftLocalZ);
  printf("%d chips along Z: chip Ncol=%d Nrow=%d\n", mNumberOfChips, mNumberOfColumnsPerChip, mNumberOfRows);
  if (Abs(mPitchZLeftColumn - mPitchZ) > 1e-5) {
    printf("Special left  column pitch: %.1f\n", mPitchZLeftColumn * kmc);
  }
  if (Abs(mPitchZRightColumn - mPitchZ) > 1e-5) {
    printf("Special right column pitch: %.1f\n", mPitchZRightColumn * kmc);
  }

  if (mDiodShiftMatDimension) {
    double dx, dz = 0;
    printf("Diod shift (fraction of pitch) periodicity pattern (X,Z[row][col])\n");
    for (int irow = 0; irow < mDiodShiftMatNRow; irow++) {
      for (int icol = 0; icol < mDiodShiftMatNColumn; icol++) {
        getDiodShift(irow, icol, dx, dz);
        printf("%.1f/%.1f |", dx, dz);
      }
      printf("\n");
    }
  }
}

void SegmentationPixel::getDiodShift(Int_t row, Int_t col, Float_t& dx, Float_t& dz) const
{
  // obtain optional diod shift
  if (!mDiodShiftMatDimension) {
    dx = dz = 0;
    return;
  }
  int cnt = (col % mDiodShiftMatNColumn) + (row % mDiodShiftMatNRow) * mDiodShiftMatNColumn;
  dx = mDiodShiftMatX[cnt];
  dz = mDiodShiftMatZ[cnt];
}
