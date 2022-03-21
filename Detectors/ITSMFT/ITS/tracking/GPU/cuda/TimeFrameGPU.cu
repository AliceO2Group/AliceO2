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
  for (auto iLayer{0}; iLayer < NLayers; ++iLayer) {
    cudaMalloc(reinterpret_cast<void**>(&(mClustersD[iLayer])), sizeof(Cluster) * 5e5);
    cudaMalloc(reinterpret_cast<void**>(&(mTrackingFrameInfoD[iLayer])), sizeof(TrackingFrameInfo) * 5e5);
    cudaMalloc(reinterpret_cast<void**>(&(mClusterExternalIndicesD[iLayer])), sizeof(int) * 5e5);
    cudaMalloc(reinterpret_cast<void**>(&(mROframesClustersD[iLayer])), sizeof(int) * 5e5);
  }
}

template <int NLayers>
void TimeFrameGPU<NLayers>::loadToDevice(const int maxLayers)
{
  LOGP(info, ">>> Loading data on device");
  for (auto iLayer{0}; iLayer < std::min(maxLayers, NLayers); ++iLayer) {
    cudaMemcpyAsync(mClustersD[iLayer], mClusters[iLayer].data(), mClusters[iLayer].size() * sizeof(Cluster), cudaMemcpyHostToDevice);
    // cudaMemcpyAsync(mTrackingFrameInfoD[iLayer], mTrackingFrameInfo[iLayer].data(), mTrackingFrameInfo[iLayer].size() * sizeof(TrackingFrameInfo), cudaMemcpyHostToDevice);
    // cudaMemcpyAsync(mClusterExternalIndicesD[iLayer], mClusterExternalIndices[iLayer].data(), mClusterExternalIndices[iLayer].size() * sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpyAsync(mROframesClustersD[iLayer], mROframesClusters[iLayer].data(), mROframesClusters[iLayer].size() * sizeof(int), cudaMemcpyHostToDevice);
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
  for (auto iLayer{0}; iLayer < NLayers; ++iLayer) {
    cudaFree(mClustersD[iLayer]);
    cudaFree(mTrackingFrameInfoD[iLayer]);
    cudaFree(mClusterExternalIndicesD[iLayer]);
    cudaFree(mROframesClustersD[iLayer]);
  }
}
template class TimeFrameGPU<7>;
} // namespace gpu
} // namespace its
} // namespace o2