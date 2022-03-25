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
#include "fairlogger/Logger.h"
#include "ITStrackingGPU/Utils.h"

namespace o2
{
namespace its
{
namespace gpu
{

template <int NLayers>
TimeFrameGPU<NLayers>::TimeFrameGPU()
{
  LOGP(info, ">>> Building TimeFrameGPU for {} layers", NLayers);
  for (int iLayer{0}; iLayer < NLayers; ++iLayer) {
    mClustersD[iLayer] = Vector<Cluster>{(int)4e4, (int)4e4};
    mTrackingFrameInfoD[iLayer] = Vector<TrackingFrameInfo>{(int)5e5, (int)5e5};
    mClusterExternalIndicesD[iLayer] = Vector<int>{(int)5e5, (int)5e5};
    mROframesClustersD[iLayer] = Vector<int>{(int)5e5, (int)5e5};
  }
}

template <int NLayers>
void TimeFrameGPU<NLayers>::loadToDevice(const int maxLayers)
{
  LOGP(info, ">>> Loading data on device");

  for (int iLayer{0}; iLayer < maxLayers; ++iLayer) {
    LOGP(info, "Size: {}, {:f} MB, {} layers", mClusters[iLayer].size(), (float)mClusters[iLayer].size() * (float)sizeof(Cluster) / (float)(1024 * 1024), mClusters.size());
    // mClustersD[iLayer].reset(v.data(), static_cast<int>(mClusters[iLayer].size()));
    // mTrackingFrameInfoD[iLayer].reset(mTrackingFrameInfo[iLayer].data(), static_cast<int>(mTrackingFrameInfo[iLayer].size()));
    // mClusterExternalIndicesD[iLayer].reset(mClusterExternalIndices[iLayer].data(), static_cast<int>(mClusterExternalIndices[iLayer].size()));
    // mROframesClustersD[iLayer].reset(mROframesClusters[iLayer].data(), static_cast<int>(mROframesClusters[iLayer].size()));
  }
}

template <int NLayers>
void TimeFrameGPU<NLayers>::initialise(const int iteration,
                                       const MemoryParameters& memParam,
                                       const TrackingParameters& trkParam,
                                       const int maxLayers)
{
  LOGP(info, ">>> Called GPU initalise");
  o2::its::TimeFrame::initialise(iteration, memParam, trkParam, maxLayers);
  loadToDevice(maxLayers);
}

template <int NLayers>
TimeFrameGPU<NLayers>::~TimeFrameGPU()
{
}
template class TimeFrameGPU<7>;
} // namespace gpu
} // namespace its
} // namespace o2