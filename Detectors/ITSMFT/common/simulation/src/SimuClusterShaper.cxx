/// \file SimuClusterShaper.cxx
/// \brief Cluster shaper for the ALPIDE response simulation

#include <iostream>
#include <map>
#include <TBits.h>
#include <TRandom.h>

#include "ITSMFTSimulation/SimuClusterShaper.h"

ClassImp(o2::ITSMFT::SimuClusterShaper)

using namespace o2::ITSMFT;

//______________________________________________________________________
SimuClusterShaper::SimuClusterShaper() :
mHitX(0.f),
mHitZ(0.f),
mHitC(0),
mHitR(0),
mFireCenter(false),
mNpixOn(0),
mSeg(nullptr),
mCShape(nullptr) {}


//______________________________________________________________________
SimuClusterShaper::SimuClusterShaper(const UInt_t &cs) {
  mHitX = 0.f;
  mHitZ = 0.f;
  mHitC = 0;
  mHitR = 0;
  mFireCenter = false;
  mNpixOn = cs;
  UInt_t nRows = cs;
  UInt_t nCols = cs;

  mSeg = nullptr;
  mCShape = new ClusterShape(nRows, nCols);
}


//______________________________________________________________________
SimuClusterShaper::~SimuClusterShaper() {
  delete mCShape;
}


//______________________________________________________________________
void SimuClusterShaper::FillClusterRandomly() {
  Int_t matrixSize = mCShape->GetNRows()*mCShape->GetNCols();

  // generate UNIQUE random numbers
  UInt_t i = 0, j = 0;
  auto *bits = new TBits(mNpixOn);

  if (mFireCenter) {
    bits->SetBitNumber(mCShape->GetCenterIndex());
    i++;
  }
  while (i < mNpixOn) {
    j = gRandom->Integer(matrixSize); // [0, matrixSize-1]
    if (bits->TestBitNumber(j)) continue;
    bits->SetBitNumber(j);
    i++;
  }

  Int_t bit = 0;
  for (i = 0; i < mNpixOn; ++i) {
    j = bits->FirstSetBit(bit);
    mCShape->AddShapeValue(j);
    bit = j+1;
  }
  delete bits;
}


//______________________________________________________________________
void SimuClusterShaper::FillClusterSorted() {
  UInt_t matrixSize = mCShape->GetNRows()*mCShape->GetNCols();
  if (matrixSize == 1) {
    mCShape->AddShapeValue(mCShape->GetCenterIndex());
    return;
  }

  ReComputeCenters();

  std::map<Double_t, UInt_t> sortedpix;
  Float_t pX = 0.f, pZ = 0.f;

  for (UInt_t i = 0; i < matrixSize; ++i) {
    UInt_t r = i / mCShape->GetNRows();
    UInt_t c = i % mCShape->GetNRows();
    UInt_t nx = mHitC - mCShape->GetCenterC() + c;
    UInt_t nz = mHitR - mCShape->GetCenterR() + r;
    mSeg->detectorToLocal(nx, nz, pX, pZ);
    Double_t d = sqrt(pow(mHitX-pX,2)+pow(mHitZ-pZ,2));

    // what to do when you reached the border?
    if (d > 1.) continue;
    sortedpix[d] = i;
  }

  // border case
  if (sortedpix.size() < mNpixOn) mNpixOn = sortedpix.size();
  for (std::map<Double_t, UInt_t>::iterator it = sortedpix.begin(); it != std::next(sortedpix.begin(),mNpixOn); ++it) {
    // std::cout << "  " << it->second << std::endl;
    mCShape->AddShapeValue(it->second);
  }
}


//______________________________________________________________________
void SimuClusterShaper::AddNoisePixel() {
  Int_t matrixSize = mCShape->GetNRows()*mCShape->GetNCols();
  UInt_t j = gRandom->Integer(matrixSize); // [0, matrixSize-1]
  while (mCShape->HasElement(j)) {
    j = gRandom->Integer(matrixSize);
  }
  //fCShape->SetShapeValue(i, j);
}


//______________________________________________________________________
void SimuClusterShaper::ReComputeCenters() {
  UInt_t  r  = 0,   c = 0;
  Float_t pX = 0.f, pZ = 0.f;
  mSeg->detectorToLocal(mHitC, mHitR, pX, pZ);

  // c is even
  if (mCShape->GetNCols() % 2 == 0) {
    if (mHitX > pX) { // n/2 - 1
      c = mCShape->GetNCols()/2 - 1;
    } else { // n/2
      c = mCShape->GetNCols()/2;
    }
  } else { // c is odd
    c = (mCShape->GetNCols()-1)/2;
  }

  // r is even
  if (mCShape->GetNRows() % 2 == 0) {
    if (mHitZ > pZ) { // n/2 - 1
      r = mCShape->GetNRows()/2 - 1;
    } else { // n/2
      r = mCShape->GetNRows()/2;
    }
  } else { // r is odd
    r = (mCShape->GetNRows()-1)/2;
  }

  mCShape->SetCenter(r, c);
}
