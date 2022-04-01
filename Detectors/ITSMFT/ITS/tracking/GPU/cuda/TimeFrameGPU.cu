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

#include "fairlogger/Logger.h"

#include "ITStracking/Constants.h"
#include "ITStracking/Configuration.h"

#include "ITStrackingGPU/Utils.h"
#include "ITStrackingGPU/TimeFrameGPU.h"

namespace o2
{
namespace its
{
using namespace constants::its2;

namespace gpu
{

template <int NLayers>
TimeFrameGPU<NLayers>::TimeFrameGPU()
{
  for (int iLayer{0}; iLayer < NLayers; ++iLayer) { // Tracker and vertexer
    mClustersD[iLayer] = Vector<Cluster>{conf.clustersPerLayerCapacity, conf.clustersPerLayerCapacity};
    mTrackingFrameInfoD[iLayer] = Vector<TrackingFrameInfo>{conf.clustersPerLayerCapacity, conf.clustersPerLayerCapacity};
    mClusterExternalIndicesD[iLayer] = Vector<int>{conf.clustersPerLayerCapacity, conf.clustersPerLayerCapacity};
    mROframesClustersD[iLayer] = Vector<int>{conf.clustersPerROfCapacity, conf.clustersPerROfCapacity};
    // mIndexTablesD[iLayer] = Vector<int>{ZBins * PhiBins + 1};
    if (iLayer < NLayers - 1) {
      mTrackletsD[iLayer] = Vector<Tracklet>{conf.clustersPerLayerCapacity * conf.maxTrackletsPerCluster,
                                             conf.clustersPerLayerCapacity * conf.maxTrackletsPerCluster};
    }
  }

  for (auto iComb{0}; iComb < 2; ++iComb) { // Vertexer only
    mNTrackletsPerClusterD[iComb] = Vector<int>{conf.clustersPerLayerCapacity, conf.clustersPerLayerCapacity};
  }
  mIndexTablesLayer0D = Vector<int>{ZBins * PhiBins + 1};
}

template <int NLayers>
void TimeFrameGPU<NLayers>::loadToDevice(const int maxLayers)
{
  for (int iLayer{0}; iLayer < maxLayers; ++iLayer) {
    mClustersD[iLayer].reset(mClusters[iLayer].data(), static_cast<int>(mClusters[iLayer].size()));
    mROframesClustersD[iLayer].reset(mROframesClusters[iLayer].data(), static_cast<int>(mROframesClusters[iLayer].size()));
  }
  if (maxLayers == NLayers) {
    // Tracker-only: we don't need to copy data in vertexer
    for (int iLayer{0}; iLayer < maxLayers; ++iLayer) {
      mTrackingFrameInfoD[iLayer].reset(mTrackingFrameInfo[iLayer].data(), static_cast<int>(mTrackingFrameInfo[iLayer].size()));
      mClusterExternalIndicesD[iLayer].reset(mClusterExternalIndices[iLayer].data(), static_cast<int>(mClusterExternalIndices[iLayer].size()));
    }
  }
}

template <int NLayers>
void TimeFrameGPU<NLayers>::initialise(const int iteration,
                                       const MemoryParameters& memParam,
                                       const TrackingParameters& trkParam,
                                       const int maxLayers)
{
  o2::its::TimeFrame::initialise(iteration, memParam, trkParam, maxLayers);
  checkBufferSizes();
  loadToDevice(maxLayers);
}

template <int NLayers>
TimeFrameGPU<NLayers>::~TimeFrameGPU()
{
}

template <int NLayers>
void TimeFrameGPU<NLayers>::checkBufferSizes()
{
  for (int iLayer{0}; iLayer < NLayers; ++iLayer) {
    if (mClusters[iLayer].size() > conf.clustersPerLayerCapacity) {
      LOGP(info, "Number of clusters on layer {} is {} and exceeds the GPU configuration defined one: {}", iLayer, mClusters[iLayer].size(), conf.clustersPerLayerCapacity);
    }
    if (mTrackingFrameInfo[iLayer].size() > conf.clustersPerLayerCapacity) {
      LOGP(info, "Number of tracking frame info on layer {} is {} and exceeds the GPU configuration defined one: {}", iLayer, mTrackingFrameInfo[iLayer].size(), conf.clustersPerLayerCapacity);
    }
    if (mClusterExternalIndices[iLayer].size() > conf.clustersPerLayerCapacity) {
      LOGP(info, "Number of external indices on layer {} is {} and exceeds the GPU configuration defined one: {}", iLayer, mClusterExternalIndices[iLayer].size(), conf.clustersPerLayerCapacity);
    }
    if (mROframesClusters[iLayer].size() > conf.clustersPerROfCapacity) {
      LOGP(info, "Size of clusters per roframe on layer {} is {} and exceeds the GPU configuration defined one: {}", iLayer, mROframesClusters[iLayer].size(), conf.clustersPerROfCapacity);
    }
  }
}
template class TimeFrameGPU<7>;
} // namespace gpu
} // namespace its
} // namespace o2