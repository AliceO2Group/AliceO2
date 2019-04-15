// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/Clustering/src/PreClustersDE.cxx
/// \brief  Structure with pre-clusters in the MID detection element
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   05 July 2018
#include "MIDClustering/PreClustersDE.h"

namespace o2
{
namespace mid
{

//______________________________________________________________________________
bool PreClustersDE::init()
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
std::vector<int> PreClustersDE::getNeighbours(int icolumn, int idx) const
{
  /// Gets the neighbour pre-cluster in the BP in icolumn + 1
  std::vector<int> neighbours;
  if (icolumn == 6) {
    return neighbours;
  }
  const BP& pcB = mPreClustersBP[icolumn][idx];
  for (int ib = 0; ib < mPreClustersBP[icolumn + 1].size(); ++ib) {
    const BP& neigh = mPreClustersBP[icolumn + 1][idx];
    if (neigh.area.getYmin() > pcB.area.getYmax()) {
      continue;
    }
    if (neigh.area.getYmax() < pcB.area.getYmin()) {
      continue;
    }
    neighbours.emplace_back(ib);
  }
  return neighbours;
}

//______________________________________________________________________________
void PreClustersDE::reset()
{
  /// Resets number of pre-clusters
  mDEId = 99;
  mPreClustersNBP.clear();
  for (int icol = 0; icol < 7; ++icol) {
    mPreClustersBP[icol].clear();
  }
}
} // namespace mid
} // namespace o2
