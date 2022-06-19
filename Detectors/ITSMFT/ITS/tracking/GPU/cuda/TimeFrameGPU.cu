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

#include <fmt/format.h>
#include <sstream>

#include "ITStracking/Constants.h"

#include "ITStrackingGPU/Utils.h"
#include "ITStrackingGPU/TimeFrameGPU.h"

namespace o2
{
namespace its
{
using constants::MB;
namespace gpu
{

GPUh() void gpuThrowOnError()
{
  cudaError_t error = cudaGetLastError();

  if (error != cudaSuccess) {
    std::ostringstream errorString{};
    errorString << GPU_ARCH << " API returned error  [" << cudaGetErrorString(error) << "] (code " << error << ")" << std::endl;
    throw std::runtime_error{errorString.str()};
  }
}

template <int NLayers>
TimeFrameGPU<NLayers>::TimeFrameGPU()
{
  getDeviceMemory(); // We don't check if we can store the data in the GPU for the moment, only log it.

  for (int iLayer{0}; iLayer < NLayers; ++iLayer) { // Tracker and vertexer
    mClustersD[iLayer] = Vector<Cluster>{mConfig.clustersPerLayerCapacity, mConfig.clustersPerLayerCapacity};
    mTrackingFrameInfoD[iLayer] = Vector<TrackingFrameInfo>{mConfig.clustersPerLayerCapacity, mConfig.clustersPerLayerCapacity};
    mClusterExternalIndicesD[iLayer] = Vector<int>{mConfig.clustersPerLayerCapacity, mConfig.clustersPerLayerCapacity};
    mROframesClustersD[iLayer] = Vector<int>{mConfig.clustersPerROfCapacity, mConfig.clustersPerROfCapacity};
    if (iLayer < NLayers - 1) {
      mTrackletsD[iLayer] = Vector<Tracklet>{mConfig.trackletsCapacity,
                                             mConfig.trackletsCapacity};
    }
  }

  for (auto iComb{0}; iComb < 2; ++iComb) { // Vertexer only
    mNTrackletsPerClusterD[iComb] = Vector<int>{mConfig.clustersPerLayerCapacity, mConfig.clustersPerLayerCapacity};
  }
  mIndexTablesLayer0D = Vector<int>{mConfig.nMaxROFs * (ZBins * PhiBins + 1), mConfig.nMaxROFs * (ZBins * PhiBins + 1)};
  mIndexTablesLayer2D = Vector<int>{mConfig.nMaxROFs * (ZBins * PhiBins + 1), mConfig.nMaxROFs * (ZBins * PhiBins + 1)};
  mLines = Vector<Line>{mConfig.trackletsCapacity, mConfig.trackletsCapacity};
  mNFoundLines = Vector<int>{mConfig.clustersPerLayerCapacity, mConfig.clustersPerLayerCapacity};
  mNExclusiveFoundLines = Vector<int>{mConfig.clustersPerLayerCapacity, mConfig.clustersPerLayerCapacity};
  mUsedTracklets = Vector<unsigned char>{mConfig.trackletsCapacity, mConfig.trackletsCapacity};
  discardResult(cudaMalloc(&mCUBTmpBuffers, mConfig.nMaxROFs * mConfig.tmpCUBBufferSize));
  mXYCentroids = Vector<float>{2 * mConfig.nMaxROFs * mConfig.maxCentroidsXYCapacity, 2 * mConfig.nMaxROFs * mConfig.maxCentroidsXYCapacity};
  mZCentroids = Vector<float>{mConfig.nMaxROFs * mConfig.maxLinesCapacity, mConfig.nMaxROFs * mConfig.maxLinesCapacity};
  for (size_t i{0}; i < 3; ++i) {
    mXYZHistograms[i] = Vector<int>{mConfig.nMaxROFs * mConfig.histConf.nBinsXYZ[i], mConfig.nMaxROFs * mConfig.histConf.nBinsXYZ[i]};
  }
  mTmpVertexPositionBins = Vector<cub::KeyValuePair<int, int>>{3 * mConfig.nMaxROFs, 3 * mConfig.nMaxROFs};
  mBeamPosition = Vector<float>{2 * mConfig.nMaxROFs, 2 * mConfig.nMaxROFs};
  mGPUVertices = Vector<Vertex>{mConfig.nMaxROFs * mConfig.maxVerticesCapacity, mConfig.nMaxROFs * mConfig.maxVerticesCapacity};
}

template <int NLayers>
float TimeFrameGPU<NLayers>::getDeviceMemory()
{
  float totalMemory{0};
  totalMemory += NLayers * mConfig.clustersPerLayerCapacity * sizeof(Cluster);
  totalMemory += NLayers * mConfig.clustersPerLayerCapacity * sizeof(TrackingFrameInfo);
  totalMemory += NLayers * mConfig.clustersPerLayerCapacity * sizeof(int);
  totalMemory += NLayers * mConfig.clustersPerROfCapacity * sizeof(int);
  totalMemory += (NLayers - 1) * mConfig.trackletsCapacity * sizeof(Tracklet);
  totalMemory += 2 * mConfig.clustersPerLayerCapacity * sizeof(int);
  totalMemory += 2 * mConfig.nMaxROFs * (ZBins * PhiBins + 1) * sizeof(int);
  totalMemory += mConfig.trackletsCapacity * sizeof(Line);
  totalMemory += mConfig.clustersPerLayerCapacity * sizeof(int);
  totalMemory += mConfig.clustersPerLayerCapacity * sizeof(int);
  totalMemory += mConfig.trackletsCapacity * sizeof(unsigned char);
  totalMemory += mConfig.nMaxROFs * mConfig.tmpCUBBufferSize * sizeof(int);
  totalMemory += 2 * mConfig.nMaxROFs * mConfig.maxCentroidsXYCapacity * sizeof(float);
  totalMemory += mConfig.nMaxROFs * mConfig.maxLinesCapacity * sizeof(float);
  for (size_t i{0}; i < 3; ++i) {
    totalMemory += mConfig.nMaxROFs * mConfig.histConf.nBinsXYZ[i] * sizeof(float);
  }
  totalMemory += 3 * mConfig.nMaxROFs * sizeof(cub::KeyValuePair<int, int>);
  totalMemory += 2 * mConfig.nMaxROFs * sizeof(float);
  totalMemory += mConfig.nMaxROFs * mConfig.maxVerticesCapacity * sizeof(Vertex);

  LOGP(debug, "Total requested memory for GPU: {:.2f} MB", totalMemory / MB);
  LOGP(debug, "\t- Clusters: {:.2f} MB", NLayers * mConfig.clustersPerLayerCapacity * sizeof(Cluster) / MB);
  LOGP(debug, "\t- Tracking frame info: {:.2f} MB", NLayers * mConfig.clustersPerLayerCapacity * sizeof(TrackingFrameInfo) / MB);
  LOGP(debug, "\t- Cluster external indices: {:.2f} MB", NLayers * mConfig.clustersPerLayerCapacity * sizeof(int) / MB);
  LOGP(debug, "\t- Clusters per ROf: {:.2f} MB", NLayers * mConfig.clustersPerROfCapacity * sizeof(int) / MB);
  LOGP(debug, "\t- Tracklets: {:.2f} MB", (NLayers - 1) * mConfig.trackletsCapacity * sizeof(Tracklet) / MB);
  LOGP(debug, "\t- N tracklets per cluster: {:.2f} MB", 2 * mConfig.clustersPerLayerCapacity * sizeof(int) / MB);
  LOGP(debug, "\t- Index tables: {:.2f} MB", 2 * mConfig.nMaxROFs * (ZBins * PhiBins + 1) * sizeof(int) / MB);
  LOGP(debug, "\t- Lines: {:.2f} MB", mConfig.trackletsCapacity * sizeof(Line) / MB);
  LOGP(debug, "\t- N found lines: {:.2f} MB", mConfig.clustersPerLayerCapacity * sizeof(int) / MB);
  LOGP(debug, "\t- N exclusive-scan found lines: {:.2f} MB", mConfig.clustersPerLayerCapacity * sizeof(int) / MB);
  LOGP(debug, "\t- Used tracklets: {:.2f} MB", mConfig.trackletsCapacity * sizeof(unsigned char) / MB);
  LOGP(debug, "\t- CUB tmp buffers: {:.2f} MB", mConfig.nMaxROFs * mConfig.tmpCUBBufferSize / MB);
  LOGP(debug, "\t- XY centroids: {:.2f} MB", 2 * mConfig.nMaxROFs * mConfig.maxCentroidsXYCapacity * sizeof(float) / MB);
  LOGP(debug, "\t- Z centroids: {:.2f} MB", mConfig.nMaxROFs * mConfig.maxLinesCapacity * sizeof(float) / MB);
  LOGP(debug, "\t- XY histograms: {:.2f} MB", 2 * mConfig.nMaxROFs * mConfig.histConf.nBinsXYZ[0] * sizeof(int) / MB);
  LOGP(debug, "\t- Z histograms: {:.2f} MB", mConfig.nMaxROFs * mConfig.histConf.nBinsXYZ[2] * sizeof(int) / MB);
  LOGP(debug, "\t- TMP Vertex position bins: {:.2f} MB", 3 * mConfig.nMaxROFs * sizeof(cub::KeyValuePair<int, int>) / MB);
  LOGP(debug, "\t- Beam positions: {:.2f} MB", 2 * mConfig.nMaxROFs * sizeof(float) / MB);
  LOGP(debug, "\t- Vertices: {:.2f} MB", mConfig.nMaxROFs * mConfig.maxVerticesCapacity * sizeof(Vertex) / MB);

  return totalMemory;
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
  } else {
    mIndexTablesLayer0D.reset(getIndexTableWhole(0).data(), static_cast<int>(getIndexTableWhole(0).size()));
    mIndexTablesLayer2D.reset(getIndexTableWhole(2).data(), static_cast<int>(getIndexTableWhole(2).size()));
  }
  gpuThrowOnError();
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
TimeFrameGPU<NLayers>::~TimeFrameGPU() = default;

template <int NLayers>
void TimeFrameGPU<NLayers>::checkBufferSizes()
{
  for (int iLayer{0}; iLayer < NLayers; ++iLayer) {
    if (mClusters[iLayer].size() > mConfig.clustersPerLayerCapacity) {
      LOGP(error, "Number of clusters on layer {} is {} and exceeds the GPU configuration defined one: {}", iLayer, mClusters[iLayer].size(), mConfig.clustersPerLayerCapacity);
    }
    if (mTrackingFrameInfo[iLayer].size() > mConfig.clustersPerLayerCapacity) {
      LOGP(error, "Number of tracking frame info on layer {} is {} and exceeds the GPU configuration defined one: {}", iLayer, mTrackingFrameInfo[iLayer].size(), mConfig.clustersPerLayerCapacity);
    }
    if (mClusterExternalIndices[iLayer].size() > mConfig.clustersPerLayerCapacity) {
      LOGP(error, "Number of external indices on layer {} is {} and exceeds the GPU configuration defined one: {}", iLayer, mClusterExternalIndices[iLayer].size(), mConfig.clustersPerLayerCapacity);
    }
    if (mROframesClusters[iLayer].size() > mConfig.clustersPerROfCapacity) {
      LOGP(error, "Size of clusters per roframe on layer {} is {} and exceeds the GPU configuration defined one: {}", iLayer, mROframesClusters[iLayer].size(), mConfig.clustersPerROfCapacity);
    }
  }
  if (mNrof > mConfig.nMaxROFs) {
    LOGP(error, "Number of ROFs in timeframe is {} and exceeds the GPU configuration defined one: {}", mNrof, mConfig.nMaxROFs);
  }
}

template class TimeFrameGPU<7>;
} // namespace gpu
} // namespace its
} // namespace o2
