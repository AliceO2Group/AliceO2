// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file TrackTPC.cxx
/// \brief Implementation of the TPC track
/// \author Thomas Klemenz, TU Muenchen, thomas.klemenz@tum.de

#include "TPCReconstruction/TrackTPC.h"

using namespace o2::TPC;


float TrackTPC::getTruncatedMean(float low, float high, int type, int removeRows, int *nclPID) const
{
  std::vector<float> values;

  for (auto &clusterObject : mClusterVector) {
    values.push_back((type == 0)?clusterObject.getQmax():clusterObject.getQ());
  }

  std::sort(values.begin(), values.end());

  float dEdx = 0.f;
  int nClustersTrunc = 0;
  int nClustersUsed = static_cast<int>(values.size());

  for (int icl=0; icl<nClustersUsed; ++icl) {
    if (icl<std::round(low*nClustersUsed)) continue;
    if (icl>std::round(high*nClustersUsed)) break;

    dEdx+=values[icl];
    ++nClustersTrunc;
  }

  if (nClustersTrunc>0){
    dEdx/=nClustersTrunc;
  }

  if (nclPID) (*nclPID)=nClustersTrunc;

  return dEdx;
}

void TrackTPC::resetClusterReferences(int nClusters)
{
  mNClusters = nClusters;
  mClusterReferences.resize(nClusters + (nClusters + 1) / 2);
}
