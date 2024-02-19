// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file  TPCMShapeCorrection.cxx
/// \brief Definition of TPCMShapeCorrection class
///
/// \author Matthias Kleiner <mkleiner@ikf.uni-frankfurt.de>

#include "TPCCalibration/TPCMShapeCorrection.h"
#include <TFile.h>
#include <TTree.h>
#include "Framework/Logger.h"

using namespace o2::tpc;

void TPCMShapeCorrection::dumpToFile(const char* file, const char* name)
{
  TFile out(file, "RECREATE");
  TTree tree(name, name);
  tree.SetAutoSave(0);
  tree.Branch("TPCMShapeCorrection", this);
  tree.Fill();
  out.WriteObject(&tree, name);
}

void TPCMShapeCorrection::dumpToFileSlices(const char* file, const char* name, int nSlices)
{
  const int nPotentials = mMShapes.mTimeMS.size();
  const int nValuesPerSlice = std::round(nPotentials / float(nSlices) + 0.5);
  for (int i = 0; i < nSlices; ++i) {
    TPCMShapeCorrection corr = *this;
    const int idx = i * nValuesPerSlice;
    const int nOffsStart = (i == 0) ? 0 : 1;
    const size_t idxStart = idx - nOffsStart;
    const int nOffsEnd = (i == nSlices - 1) ? 0 : 1;
    int idxEnd = (idx + nValuesPerSlice + 2 * nOffsEnd);
    idxEnd = std::clamp(idxEnd, 0, nPotentials);

    corr.mMShapes.mTimeMS = std::vector<double>(mMShapes.mTimeMS.begin() + idxStart, mMShapes.mTimeMS.begin() + idxEnd);
    corr.mMShapes.mBoundaryPotentialIFC = std::vector<BoundaryPotentialIFC>(mMShapes.mBoundaryPotentialIFC.begin() + idxStart, mMShapes.mBoundaryPotentialIFC.begin() + idxEnd);

    long tsCCDBStart = corr.mMShapes.mTimeMS[nOffsStart];
    if (i == 0) {
      tsCCDBStart -= 2. * mMaxDeltaTimeMS;
    }

    long tsCCDBEnd = corr.mMShapes.mTimeMS[corr.mMShapes.mTimeMS.size() - 1 - nOffsEnd];
    // add additional validation offset for last object
    if (i == nSlices - 1) {
      tsCCDBEnd += 2. * mMaxDeltaTimeMS;
    }
    const std::string fileOut = fmt::format("{}_{}_{}_{}.root", file, i, tsCCDBStart, tsCCDBEnd);
    corr.dumpToFile(fileOut.data(), name);
  }
}

void TPCMShapeCorrection::loadFromFile(const char* inpf, const char* name)
{
  TFile out(inpf, "READ");
  TTree* tree = (TTree*)out.Get(name);
  setFromTree(*tree);
}

void TPCMShapeCorrection::setFromTree(TTree& tpcMShapeTree)
{
  TPCMShapeCorrection* mshapeTmp = this;
  tpcMShapeTree.SetBranchAddress("TPCMShapeCorrection", &mshapeTmp);
  const int entries = tpcMShapeTree.GetEntries();
  if (entries > 0) {
    tpcMShapeTree.GetEntry(0);
  } else {
    LOGP(error, "TPCMShapeCorrection not found in input file");
  }
  tpcMShapeTree.SetBranchAddress("TPCMShapeCorrection", nullptr);
}

BoundaryPotentialIFC TPCMShapeCorrection::getBoundaryPotential(const double timestamp) const
{
  const auto& time = mMShapes.mTimeMS;
  // find closest stored M-Shape
  const auto lower = std::lower_bound(time.begin(), time.end(), timestamp);
  const int idx = std::distance(time.begin(), lower);
  if ((idx > 0) && (idx < time.size())) {
    // if idx > 0 check preceeding value
    double diff1 = std::abs(timestamp - *lower);
    double diff2 = std::abs(timestamp - *(lower - 1));
    if (diff2 < diff1) {
      if (diff2 < mMaxDeltaTimeMS) {
        return mMShapes.mBoundaryPotentialIFC[idx - 1];
      }
    } else {
      if (diff1 < mMaxDeltaTimeMS) {
        return mMShapes.mBoundaryPotentialIFC[idx];
      }
    }
  }
  return BoundaryPotentialIFC{};
}
