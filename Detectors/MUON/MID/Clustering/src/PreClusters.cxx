// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/Clustering/src/PreClusters.cxx
/// \brief  Implementation of the pre-clusters for MID
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   05 July 2018
#include "MIDClustering/PreClusters.h"

namespace o2
{
namespace mid
{
//______________________________________________________________________________
PreClusters::PreClusters()
  : mNPreClustersBP(), mPreClustersNBP(), mPreClustersBP()
{
  /// Default constructor
}

//______________________________________________________________________________
bool PreClusters::init()
{
  /// Initializes the class

  // prepare storage of PreClusters
  mPreClustersNBP.reserve(100);
  for (int icol = 0; icol < 7; ++icol) {
    mPreClustersBP[icol].reserve(20);
  }

  return true;
}

//______________________________________________________________________________
PreClusters::PreClusterNBP* PreClusters::nextPreClusterNBP()
{
  /// Iterates on pre-clusters in the bending plane
  if (mNPreClustersNBP >= static_cast<int>(mPreClustersNBP.size())) {
    mPreClustersNBP.emplace_back(PreClusterNBP());
  }
  return &(mPreClustersNBP[mNPreClustersNBP++]);
}

//______________________________________________________________________________
PreClusters::PreClusterBP* PreClusters::nextPreClusterBP(int icolumn)
{
  /// Iterates on pre-clusters in the bending plane
  if (mNPreClustersBP[icolumn] >= static_cast<int>(mPreClustersBP[icolumn].size())) {
    mPreClustersBP[icolumn].emplace_back(PreClusterBP());
  }
  return &(mPreClustersBP[icolumn][mNPreClustersBP[icolumn]++]);
}

//______________________________________________________________________________
std::vector<int> PreClusters::getNeighbours(int icolumn, int idx)
{
  /// Gets the neighbour pre-cluster in the BP i the next column
  std::vector<int> neighbours;
  if (icolumn == 6) {
    return neighbours;
  }
  PreClusterBP& pcB = mPreClustersBP[icolumn][idx];
  for (int ib = 0; ib < mNPreClustersBP[icolumn + 1]; ++ib) {
    PreClusterBP& neigh = mPreClustersBP[icolumn + 1][idx];
    if (neigh.area.getYmin() > pcB.area.getYmax()) {
      continue;
    }
    if (neigh.area.getYmax() < pcB.area.getYmin()) {
      continue;
    }
    neighbours.push_back(ib);
  }
  return neighbours;
}

//______________________________________________________________________________
void PreClusters::reset()
{
  /// Resets number of pre-clusters
  mDEId = 99;
  mNPreClustersNBP = 0;
  mNPreClustersBP.fill(0);
}
} // namespace mid
} // namespace o2
