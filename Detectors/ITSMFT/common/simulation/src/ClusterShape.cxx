/// \file ClusterShape.cxx
/// \brief Cluster shape class for the ALPIDE response simulation

#include <iostream>
#include <TBits.h>
#include <TRandom.h>

#include "ITSMFTSimulation/ClusterShape.h"

ClassImp(AliceO2::ITSMFT::ClusterShape)

using namespace AliceO2::ITSMFT;

//______________________________________________________________________
ClusterShape::ClusterShape() :
fNrows(0),
fNcols(0) {
  fShape.clear();
}


//______________________________________________________________________
ClusterShape::ClusterShape(UInt_t Nrows, UInt_t Ncols) :
fNrows(Nrows),
fNcols(Ncols) {
  fShape.clear();
}


//______________________________________________________________________
ClusterShape::ClusterShape(UInt_t Nrows, UInt_t Ncols, const std::vector<UInt_t>& Shape) :
fNrows(Nrows),
fNcols(Ncols) {
  fShape = Shape;
}


//______________________________________________________________________
ClusterShape::~ClusterShape() {}


//______________________________________________________________________
Bool_t ClusterShape::IsValidShape() {
  // Check the size
  if (fShape.size() > fNrows*fNcols) return false;

  // Check for duplicates and the validity of the position
  std::sort(fShape.begin(), fShape.end());
  for (size_t i = 0; i < fShape.size() - 1; i++) {
    if (fShape[i] >= fNrows*fNcols || fShape[i+1] >= fNrows*fNcols) return false;
    if (fShape[i] == fShape[i+1]) return false;
  }

  return true;
}


//______________________________________________________________________
Long64_t ClusterShape::GetShapeID() const {
  // DJBX33X
  Long64_t id = 5381;
  id = ((id << 5) + id) ^ fNrows;
  id = ((id << 5) + id) ^ fNcols;
  for (UInt_t i = 0; i < fShape.size(); ++i) {
    id = ((id << 5) + id) ^ fShape[i];
  }
  return id;
}


//______________________________________________________________________
Bool_t ClusterShape::HasElement(UInt_t value) const {
  for (auto & el : fShape) {
    if (el > value) break;
    if (el == value) return true;
  }
  return false;
}
