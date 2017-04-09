/// \file ClusterShape.cxx
/// \brief Cluster shape class for the ALPIDE response simulation

#include <iostream>
#include <algorithm>
#include <TBits.h>
#include <TRandom.h>

#include "ITSMFTSimulation/ClusterShape.h"

ClassImp(o2::ITSMFT::ClusterShape)

using namespace o2::ITSMFT;

//______________________________________________________________________
ClusterShape::ClusterShape() :
mNrows(0),
mNcols(0) {
  mShape.clear();
}


//______________________________________________________________________
ClusterShape::ClusterShape(UInt_t Nrows, UInt_t Ncols) :
mNrows(Nrows),
mNcols(Ncols) {
  mShape.clear();
}


//______________________________________________________________________
ClusterShape::ClusterShape(UInt_t Nrows, UInt_t Ncols, const std::vector<UInt_t>& Shape) :
mNrows(Nrows),
mNcols(Ncols) {
  mShape = Shape;
}


//______________________________________________________________________
ClusterShape::~ClusterShape() = default;


//______________________________________________________________________
Bool_t ClusterShape::IsValidShape() {
  // Check the size
  if (mShape.size() > mNrows*mNcols) return false;

  // Check for duplicates and the validity of the position
  std::sort(mShape.begin(), mShape.end());
  for (size_t i = 0; i < mShape.size() - 1; i++) {
    if (mShape[i] >= mNrows*mNcols || mShape[i+1] >= mNrows*mNcols) return false;
    if (mShape[i] == mShape[i+1]) return false;
  }

  return true;
}


//______________________________________________________________________
Long64_t ClusterShape::GetShapeID() const {
  // DJBX33X
  Long64_t id = 5381;
  id = ((id << 5) + id) ^ mNrows;
  id = ((id << 5) + id) ^ mNcols;
  for (UInt_t i = 0; i < mShape.size(); ++i) {
    id = ((id << 5) + id) ^ mShape[i];
  }
  return id;
}


//______________________________________________________________________
Bool_t ClusterShape::HasElement(UInt_t value) const {
  for (auto & el : mShape) {
    if (el > value) break;
    if (el == value) return true;
  }
  return false;
}
