/// \file SimuClusterShaper.cxx
/// \brief Cluster shaper for the ALPIDE response simulation

#include <iostream>
#include <TBits.h>
#include <TRandom.h>

#include "ITSMFTSimulation/SimuClusterShaper.h"

ClassImp(o2::ITSMFT::SimuClusterShaper)

using namespace o2::ITSMFT;

//______________________________________________________________________
SimuClusterShaper::SimuClusterShaper() :
mFireCenter(false),
mNpixOn(0),
mCShape(nullptr) {}


//______________________________________________________________________
SimuClusterShaper::SimuClusterShaper(const UInt_t &cs) {
  mFireCenter = false;
  mNpixOn = cs;
  UInt_t nRows = 0;
  UInt_t nCols = 0;
  while (nRows*nCols < cs) {
    nRows += 1;
    nCols += 1;
  }
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
void SimuClusterShaper::AddNoisePixel() {
  Int_t matrixSize = mCShape->GetNRows()*mCShape->GetNCols();
  UInt_t j = gRandom->Integer(matrixSize); // [0, matrixSize-1]
  while (mCShape->HasElement(j)) {
    j = gRandom->Integer(matrixSize);
  }
  //fCShape->SetShapeValue(i, j);
}
