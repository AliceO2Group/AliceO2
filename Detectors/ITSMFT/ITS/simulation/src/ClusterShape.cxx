/// \file ClusterShape.cxx
/// \brief Cluster shape class for the ALPIDE response simulation

#include <iostream>
#include <TBits.h>
#include <TRandom.h>

#include "ITSSimulation/ClusterShape.h"

ClassImp(AliceO2::ITS::ClusterShape)

using namespace AliceO2::ITS;

//______________________________________________________________________
ClusterShape::ClusterShape() :
fNrows(0),
fNcols(0),
fNFPix(0),
fShape(0)
{}


//______________________________________________________________________
ClusterShape::ClusterShape(UInt_t Nrows, UInt_t Ncols, UInt_t NFPix) {
  fNrows = Nrows;
  fNcols = Ncols;
  fNFPix = NFPix;
  fShape = new UInt_t[fNFPix];
}


//______________________________________________________________________
ClusterShape::~ClusterShape() {}


//______________________________________________________________________
Long64_t ClusterShape::GetShapeID() {
  // DJBX33X
  Long64_t id = 5381;
  for (UInt_t i = 0; i < fNFPix; ++i) {
    id = ((id << 5) + id) ^ fShape[i];
  }
  return id;
}
