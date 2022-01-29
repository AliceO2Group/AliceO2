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
///

#include "ITStrackingGPU/TimeFrameGPU.h"

namespace o2
{
namespace its
{
namespace gpu
{
template <int NLayers>
TimeFrameGPU<NLayers>::TimeFrameGPU()
{
  for (auto iLayer{0}; iLayer < NLayers; ++iLayer) {
    mClustersGPU[iLayer] = Vector<Cluster>{(int)5e5, (int)5e5};
    mTrackingFrameInfoGPU[iLayer] = Vector<TrackingFrameInfo>{(int)5e5, (int)5e5};
    mClusterExternalIndicesGPU[iLayer] = Vector<int>{(int)5e5, (int)5e5};
    mROframesClustersGPU[iLayer] = Vector<int>{(int)5e5, (int)5e5};
  }
}

template <int NLayers>
TimeFrameGPU<NLayers>::~TimeFrameGPU()
{
}

template <int NLayers>
void TimeFrameGPU<NLayers>::loadToDevice()
{
  for (auto iLayer{0}; iLayer < NLayers; ++iLayer) {
    mClustersGPU[iLayer].reset(mClusters[iLayer].data(), mClusters[iLayer].size());
    mTrackingFrameInfoGPU[iLayer].reset(mTrackingFrameInfo[iLayer].data(), mTrackingFrameInfo[iLayer].size());
    mClusterExternalIndicesGPU[iLayer].reset(mClusterExternalIndices[iLayer].data(), mClusterExternalIndices[iLayer].size());
    mROframesClustersGPU[iLayer].reset(mROframesClusters[iLayer].data(), mROframesClusters[iLayer].size());
  }
}

template class TimeFrameGPU<7>;
} // namespace gpu
} // namespace its
} // namespace o2