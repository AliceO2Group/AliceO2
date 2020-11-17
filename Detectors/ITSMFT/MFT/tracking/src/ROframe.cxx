// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
///
/// \file ROframe.cxx
///

#include "MFTTracking/ROframe.h"

#include <iostream>

namespace o2
{
namespace mft
{

ROframe::ROframe(const Int_t ROframeId) : mROframeId{ROframeId}
{
}

Int_t ROframe::getTotalClusters() const
{
  size_t totalClusters{0};
  for (auto& clusters : mClusters) {
    totalClusters += clusters.size();
  }
  return Int_t(totalClusters);
}

void ROframe::initialize()
{
  sortClusters();
}

void ROframe::sortClusters()
{
  Int_t nClsInLayer, binPrevIndex, clsMinIndex, clsMaxIndex, jClsLayer;
  // sort the clusters in R-Phi
  for (Int_t iLayer = 0; iLayer < constants::mft::LayersNumber; ++iLayer) {
    if (mClusters[iLayer].size() == 0) {
      continue;
    }
    // sort clusters in layer according to the bin index
    sort(mClusters[iLayer].begin(), mClusters[iLayer].end(),
         [](Cluster& c1, Cluster& c2) { return c1.indexTableBin < c2.indexTableBin; });
    // find the cluster local index range in each bin
    // index = element position in the vector
    nClsInLayer = mClusters[iLayer].size();
    binPrevIndex = mClusters[iLayer].at(0).indexTableBin;
    clsMinIndex = 0;
    for (jClsLayer = 1; jClsLayer < nClsInLayer; ++jClsLayer) {
      if (mClusters[iLayer].at(jClsLayer).indexTableBin == binPrevIndex) {
        continue;
      }

      clsMaxIndex = jClsLayer - 1;

      mClusterBinIndexRange[iLayer][binPrevIndex] = std::pair<Int_t, Int_t>(clsMinIndex, clsMaxIndex);

      binPrevIndex = mClusters[iLayer].at(jClsLayer).indexTableBin;
      clsMinIndex = jClsLayer;
    } // clusters

    // last cluster
    clsMaxIndex = jClsLayer - 1;

    mClusterBinIndexRange[iLayer][binPrevIndex] = std::pair<Int_t, Int_t>(clsMinIndex, clsMaxIndex);
  } // layers
}

} // namespace mft
} // namespace o2
