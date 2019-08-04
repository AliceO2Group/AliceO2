// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file ClusterShape.cxx
/// \brief Cluster shape class for the ALPIDE response simulation

#include <iostream>
#include <algorithm>
#include <TBits.h>
#include <TRandom.h>

#include "ITSMFTSimulation/ClusterShape.h"

ClassImp(o2::itsmft::ClusterShape);

using namespace o2::itsmft;

//______________________________________________________________________
ClusterShape::ClusterShape() : mNrows(0),
                               mNcols(0),
                               mCenterR(0),
                               mCenterC(0)
{
  mShape.clear();
}

//______________________________________________________________________
ClusterShape::ClusterShape(UInt_t Nrows, UInt_t Ncols) : mNrows(Nrows),
                                                         mNcols(Ncols)
{
  mCenterR = ComputeCenter(Nrows);
  mCenterC = ComputeCenter(Ncols);
  mShape.clear();
}

//______________________________________________________________________
ClusterShape::ClusterShape(UInt_t Nrows, UInt_t Ncols, const std::vector<UInt_t>& Shape) : mNrows(Nrows),
                                                                                           mNcols(Ncols)
{
  mCenterR = ComputeCenter(Nrows);
  mCenterC = ComputeCenter(Ncols);
  mShape = Shape;
}

//______________________________________________________________________
ClusterShape::~ClusterShape() = default;

//______________________________________________________________________
Bool_t ClusterShape::IsValidShape()
{
  // Check the size
  if (mShape.size() > mNrows * mNcols)
    return false;

  // Check for duplicates and the validity of the position
  std::sort(mShape.begin(), mShape.end());
  for (size_t i = 0; i < mShape.size() - 1; i++) {
    if (mShape[i] >= mNrows * mNcols || mShape[i + 1] >= mNrows * mNcols)
      return false;
    if (mShape[i] == mShape[i + 1])
      return false;
  }

  return true;
}

//______________________________________________________________________
Long64_t ClusterShape::GetShapeID() const
{
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
Bool_t ClusterShape::HasElement(UInt_t value) const
{
  for (auto& el : mShape) {
    if (el > value)
      break;
    if (el == value)
      return true;
  }
  return false;
}

//______________________________________________________________________
UInt_t ClusterShape::ComputeCenter(UInt_t n)
{
  UInt_t c = 0;
  if (n % 2 == 0) {
    UInt_t r = gRandom->Integer(2); // 0 or 1
    c = (UInt_t)r + n / 2;
  } else {
    c = (UInt_t)(n + 1) / 2;
  }
  return c - 1; // 0-based
}
