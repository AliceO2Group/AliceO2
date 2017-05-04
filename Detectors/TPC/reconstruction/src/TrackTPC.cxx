/// \file TrackTPC.cxx
/// \brief Implementation of the TPC track
/// \author Thomas Klemenz, TU Muenchen, thomas.klemenz@tum.de

#include "TPCReconstruction/TrackTPC.h"

using namespace o2::TPC;

float TrackTPC::GetTruncatedMean(float low, float high, int type, int removeRows, int *nclPID) const
{
  std::vector<float> values;

  for (auto clusterObject : *mClusterArray) {
    Cluster *inputcluster = static_cast<Cluster *>(clusterObject);
    values.push_back((type == 0)?inputcluster->getQmax():inputcluster->getQ());
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
