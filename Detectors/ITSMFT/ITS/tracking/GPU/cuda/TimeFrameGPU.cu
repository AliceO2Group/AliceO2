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

#include <thrust/fill.h>

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
using utils::host::checkGPUError;
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
  mIsGPU = true;
  // getDeviceMemory(); To be updated
}

template <int NLayers>
float TimeFrameGPU<NLayers>::getDeviceMemory()
{
  // We don't check if we can store the data in the GPU for the moment, only log it.
  float totalMemory{0};
  totalMemory += NLayers * mConfig.clustersPerLayerCapacity * sizeof(Cluster);
  totalMemory += NLayers * mConfig.clustersPerLayerCapacity * sizeof(unsigned char);
  totalMemory += NLayers * mConfig.clustersPerLayerCapacity * sizeof(TrackingFrameInfo);
  totalMemory += NLayers * mConfig.clustersPerLayerCapacity * sizeof(int);
  totalMemory += NLayers * mConfig.clustersPerROfCapacity * sizeof(int);
  totalMemory += (NLayers - 1) * mConfig.trackletsCapacity * sizeof(Tracklet);
  totalMemory += (NLayers - 1) * mConfig.nMaxROFs * (256 * 128 + 1) * sizeof(int);
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

  LOG(info) << fmt::format("Total requested memory for GPU: {:.2f} MB", totalMemory / MB);
  LOG(info) << fmt::format("\t- Clusters: {:.2f} MB", NLayers * mConfig.clustersPerLayerCapacity * sizeof(Cluster) / MB);
  LOG(info) << fmt::format("\t- Used clusters: {:.2f} MB", NLayers * mConfig.clustersPerLayerCapacity * sizeof(unsigned char) / MB);
  LOG(info) << fmt::format("\t- Tracking frame info: {:.2f} MB", NLayers * mConfig.clustersPerLayerCapacity * sizeof(TrackingFrameInfo) / MB);
  LOG(info) << fmt::format("\t- Cluster external indices: {:.2f} MB", NLayers * mConfig.clustersPerLayerCapacity * sizeof(int) / MB);
  LOG(info) << fmt::format("\t- Clusters per ROf: {:.2f} MB", NLayers * mConfig.clustersPerROfCapacity * sizeof(int) / MB);
  LOG(info) << fmt::format("\t- Tracklets: {:.2f} MB", (NLayers - 1) * mConfig.trackletsCapacity * sizeof(Tracklet) / MB);
  LOG(info) << fmt::format("\t- Tracklet index tables: {:.2f} MB", (NLayers - 1) * mConfig.nMaxROFs * (256 * 128 + 1) * sizeof(int) / MB);
  LOG(info) << fmt::format("\t- N tracklets per cluster: {:.2f} MB", 2 * mConfig.clustersPerLayerCapacity * sizeof(int) / MB);
  LOG(info) << fmt::format("\t- Index tables: {:.2f} MB", 2 * mConfig.nMaxROFs * (ZBins * PhiBins + 1) * sizeof(int) / MB);
  LOG(info) << fmt::format("\t- Lines: {:.2f} MB", mConfig.trackletsCapacity * sizeof(Line) / MB);
  LOG(info) << fmt::format("\t- N found lines: {:.2f} MB", mConfig.clustersPerLayerCapacity * sizeof(int) / MB);
  LOG(info) << fmt::format("\t- N exclusive-scan found lines: {:.2f} MB", mConfig.clustersPerLayerCapacity * sizeof(int) / MB);
  LOG(info) << fmt::format("\t- Used tracklets: {:.2f} MB", mConfig.trackletsCapacity * sizeof(unsigned char) / MB);
  LOG(info) << fmt::format("\t- CUB tmp buffers: {:.2f} MB", mConfig.nMaxROFs * mConfig.tmpCUBBufferSize / MB);
  LOG(info) << fmt::format("\t- XY centroids: {:.2f} MB", 2 * mConfig.nMaxROFs * mConfig.maxCentroidsXYCapacity * sizeof(float) / MB);
  LOG(info) << fmt::format("\t- Z centroids: {:.2f} MB", mConfig.nMaxROFs * mConfig.maxLinesCapacity * sizeof(float) / MB);
  LOG(info) << fmt::format("\t- XY histograms: {:.2f} MB", 2 * mConfig.nMaxROFs * mConfig.histConf.nBinsXYZ[0] * sizeof(int) / MB);
  LOG(info) << fmt::format("\t- Z histograms: {:.2f} MB", mConfig.nMaxROFs * mConfig.histConf.nBinsXYZ[2] * sizeof(int) / MB);
  LOG(info) << fmt::format("\t- TMP Vertex position bins: {:.2f} MB", 3 * mConfig.nMaxROFs * sizeof(cub::KeyValuePair<int, int>) / MB);
  LOG(info) << fmt::format("\t- Beam positions: {:.2f} MB", 2 * mConfig.nMaxROFs * sizeof(float) / MB);
  LOG(info) << fmt::format("\t- Vertices: {:.2f} MB", mConfig.nMaxROFs * mConfig.maxVerticesCapacity * sizeof(Vertex) / MB);

  return totalMemory;
}

template <int NLayers>
template <unsigned char isTracker>
void TimeFrameGPU<NLayers>::initialiseDevice(const TrackingParameters& trkParam)
{
  mTrackletSizeHost.resize(NLayers - 1, 0);
  mCellSizeHost.resize(NLayers - 2, 0);
  for (int iLayer{0}; iLayer < NLayers - 1; ++iLayer) { // Tracker and vertexer
    mTrackletsD[iLayer] = Vector<Tracklet>{mConfig.trackletsCapacity, mConfig.trackletsCapacity};
    auto thrustTrackletsBegin = thrust::device_ptr<Tracklet>(mTrackletsD[iLayer].get());
    auto thrustTrackletsEnd = thrustTrackletsBegin + mConfig.trackletsCapacity;
    thrust::fill(thrustTrackletsBegin, thrustTrackletsEnd, Tracklet{});
    mTrackletsLookupTablesD[iLayer].resetWithInt(mClusters[iLayer].size());
    if (iLayer < NLayers - 2) {
      mCellsD[iLayer] = Vector<Cell>{mConfig.validatedTrackletsCapacity, mConfig.validatedTrackletsCapacity};
      mCellsLookupTablesD[iLayer] = Vector<int>{mConfig.cellsLUTsize, mConfig.cellsLUTsize};
      mCellsLookupTablesD[iLayer].resetWithInt(mConfig.cellsLUTsize);
    }
  }

  for (auto iComb{0}; iComb < 2; ++iComb) { // Vertexer only
    mNTrackletsPerClusterD[iComb] = Vector<int>{mConfig.clustersPerLayerCapacity, mConfig.clustersPerLayerCapacity};
  }
  mLines = Vector<Line>{mConfig.trackletsCapacity, mConfig.trackletsCapacity};
  mNFoundLines = Vector<int>{mConfig.clustersPerLayerCapacity, mConfig.clustersPerLayerCapacity};
  mNFoundLines.resetWithInt(mConfig.clustersPerLayerCapacity);
  mNExclusiveFoundLines = Vector<int>{mConfig.clustersPerLayerCapacity, mConfig.clustersPerLayerCapacity};
  mNExclusiveFoundLines.resetWithInt(mConfig.clustersPerLayerCapacity);
  mUsedTracklets = Vector<unsigned char>{mConfig.trackletsCapacity, mConfig.trackletsCapacity};
  discardResult(cudaMalloc(&mCUBTmpBuffers, mConfig.nMaxROFs * mConfig.tmpCUBBufferSize));
  discardResult(cudaMalloc(&mDeviceFoundTracklets, (NLayers - 1) * sizeof(int)));
  discardResult(cudaMemset(mDeviceFoundTracklets, 0, (NLayers - 1) * sizeof(int)));
  discardResult(cudaMalloc(&mDeviceFoundCells, (NLayers - 2) * sizeof(int)));
  discardResult(cudaMemset(mDeviceFoundCells, 0, (NLayers - 2) * sizeof(int)));
  mXYCentroids = Vector<float>{2 * mConfig.nMaxROFs * mConfig.maxCentroidsXYCapacity, 2 * mConfig.nMaxROFs * mConfig.maxCentroidsXYCapacity};
  mZCentroids = Vector<float>{mConfig.nMaxROFs * mConfig.maxLinesCapacity, mConfig.nMaxROFs * mConfig.maxLinesCapacity};
  for (size_t i{0}; i < 3; ++i) {
    mXYZHistograms[i] = Vector<int>{mConfig.nMaxROFs * mConfig.histConf.nBinsXYZ[i], mConfig.nMaxROFs * mConfig.histConf.nBinsXYZ[i]};
  }
  mTmpVertexPositionBins = Vector<cub::KeyValuePair<int, int>>{3 * mConfig.nMaxROFs, 3 * mConfig.nMaxROFs};
  mBeamPosition = Vector<float>{2 * mConfig.nMaxROFs, 2 * mConfig.nMaxROFs};
  mGPUVertices = Vector<Vertex>{mConfig.nMaxROFs * mConfig.maxVerticesCapacity, mConfig.nMaxROFs * mConfig.maxVerticesCapacity};
  //////////////////////////////////////////////////////////////////////////////
  constexpr int layers = isTracker ? NLayers : 3;
  for (int iLayer{0}; iLayer < layers; ++iLayer) {
    mClustersD[iLayer].reset(mClusters[iLayer].data(), static_cast<int>(mClusters[iLayer].size()));
  }
  if constexpr (isTracker) {
    StaticTrackingParameters<NLayers> pars;
    pars.set(trkParam);
    checkGPUError(cudaMalloc(reinterpret_cast<void**>(&mDeviceTrackingParams), sizeof(gpu::StaticTrackingParameters<NLayers>)), __FILE__, __LINE__);
    checkGPUError(cudaMalloc(reinterpret_cast<void**>(&mDeviceIndexTableUtils), sizeof(IndexTableUtils)), __FILE__, __LINE__);
    checkGPUError(cudaMemcpy(mDeviceTrackingParams, &pars, sizeof(gpu::StaticTrackingParameters<NLayers>), cudaMemcpyHostToDevice), __FILE__, __LINE__);
    checkGPUError(cudaMemcpy(mDeviceIndexTableUtils, &mIndexTableUtils, sizeof(IndexTableUtils), cudaMemcpyHostToDevice), __FILE__, __LINE__);
    // Tracker-only: we don't need to copy data in vertexer
    for (int iLayer{0}; iLayer < NLayers; ++iLayer) {
      mUsedClustersD[iLayer].reset(mUsedClusters[iLayer].data(), static_cast<int>(mUsedClusters[iLayer].size()));
      mTrackingFrameInfoD[iLayer].reset(mTrackingFrameInfo[iLayer].data(), static_cast<int>(mTrackingFrameInfo[iLayer].size()));
      mClusterExternalIndicesD[iLayer].reset(mClusterExternalIndices[iLayer].data(), static_cast<int>(mClusterExternalIndices[iLayer].size()));
      mROframesClustersD[iLayer].reset(mROframesClusters[iLayer].data(), static_cast<int>(mROframesClusters[iLayer].size()));
      mIndexTablesD[iLayer].reset(mIndexTables[iLayer].data(), static_cast<int>(mIndexTables[iLayer].size()));
    }
  } else {
    mIndexTablesD[0].reset(getIndexTableWhole(0).data(), static_cast<int>(getIndexTableWhole(0).size()));
    mIndexTablesD[2].reset(getIndexTableWhole(2).data(), static_cast<int>(getIndexTableWhole(2).size()));
  }

  gpuThrowOnError();
}

template <int NLayers>
void TimeFrameGPU<NLayers>::initialise(const int iteration,
                                       const TrackingParameters& trkParam,
                                       const int maxLayers)
{
  o2::its::TimeFrame::initialise(iteration, trkParam, maxLayers);
  checkBufferSizes();
  if (maxLayers < NLayers) {
    initialiseDevice<false>(trkParam); // vertexer
  } else {
    initialiseDevice<true>(trkParam); // tracker
  }
}

template <int NLayers>
TimeFrameGPU<NLayers>::~TimeFrameGPU()
{
  discardResult(cudaFree(mCUBTmpBuffers));
  discardResult(cudaFree(mDeviceFoundTracklets));
  discardResult(cudaFree(mDeviceTrackingParams));
  discardResult(cudaFree(mDeviceIndexTableUtils));
  discardResult(cudaFree(mDeviceFoundCells));
}

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
