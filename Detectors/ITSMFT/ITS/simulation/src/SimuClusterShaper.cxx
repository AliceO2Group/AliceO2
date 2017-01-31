/// \file SimuClusterShaper.cxx
/// \brief Cluster shaper for the ALPIDE response simulation

#include <iostream>
#include <TBits.h>
#include <TRandom.h>

#include "ITSSimulation/SimuClusterShaper.h"

ClassImp(AliceO2::ITS::SimuClusterShaper)

using namespace AliceO2::ITS;

//______________________________________________________________________
SimuClusterShaper::SimuClusterShaper() :
fCShape(0) {}


//______________________________________________________________________
SimuClusterShaper::SimuClusterShaper(const UInt_t &cs) {
  fNpixOn = cs;
  UInt_t nRows = 0;
  UInt_t nCols = 0;
  while (nRows*nCols < cs) {
    nRows += 1;
    nCols += 1;
  }
  fCShape = new ClusterShape(nRows, nCols, fNpixOn);
}


//______________________________________________________________________
SimuClusterShaper::~SimuClusterShaper() {
  delete fCShape;
}


//______________________________________________________________________
void SimuClusterShaper::FillClusterRandomly() {
  Int_t matrixSize = fCShape->GetNRows()*fCShape->GetNCols();

  // generate UNIQUE random numbers
  UInt_t i = 0;
  TBits *bits = new TBits(fNpixOn);
  while (i < fNpixOn) {
    UInt_t j = gRandom->Integer(matrixSize); // [0, matrixSize-1]
    if (bits->TestBitNumber(j)) continue;
    bits->SetBitNumber(j);
    i++;
  }

  Int_t bit = 0;
  for (i = 0; i < fNpixOn; ++i) {
    UInt_t j = bits->FirstSetBit(bit);
    fCShape->SetShapeValue(i, j);
    bit = j+1;
  }
  delete bits;
}


//______________________________________________________________________
void SimuClusterShaper::AddNoisePixel() {
  Int_t matrixSize = fCShape->GetNRows()*fCShape->GetNCols();
  UInt_t j = gRandom->Integer(matrixSize); // [0, matrixSize-1]
  while (fCShape->HasElement(j)) {
    j = gRandom->Integer(matrixSize);
  }
  //fCShape->SetShapeValue(i, j);
}
