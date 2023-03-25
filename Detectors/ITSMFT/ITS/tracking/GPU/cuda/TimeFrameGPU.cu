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
#include <thrust/fill.h>
#include <thrust/execution_policy.h>

#include "ITStracking/Constants.h"

#include "ITStrackingGPU/Utils.h"
#include "ITStrackingGPU/TimeFrameGPU.h"
#include "ITStrackingGPU/TracerGPU.h"

#include <unistd.h>
#include <thread>

#ifndef __HIPCC__
#define THRUST_NAMESPACE thrust::cuda
#else
#define THRUST_NAMESPACE thrust::hip
#endif

namespace o2
{
namespace its
{
using constants::GB;
using constants::MB;

namespace gpu
{
using utils::checkGPUError;
/////////////////////////////////////////////////////////////////////////////////////////
// GpuChunk
/////////////////////////////////////////////////////////////////////////////////////////
template <int nLayers>
GpuTimeFrameChunk<nLayers>::~GpuTimeFrameChunk()
{
  if (mAllocated) {
    for (int i = 0; i < nLayers; ++i) {
      checkGPUError(cudaFree(mClustersDevice[i]));
      checkGPUError(cudaFree(mTrackingFrameInfoDevice[i]));
      checkGPUError(cudaFree(mClusterExternalIndicesDevice[i]));
      checkGPUError(cudaFree(mIndexTablesDevice[i]));
      if (i < nLayers - 1) {
        checkGPUError(cudaFree(mTrackletsDevice[i]));
        checkGPUError(cudaFree(mTrackletsLookupTablesDevice[i]));
        if (i < nLayers - 2) {
          checkGPUError(cudaFree(mCellsDevice[i]));
          checkGPUError(cudaFree(mCellsLookupTablesDevice[i]));
        }
      }
    }
    checkGPUError(cudaFree(mCUBTmpBufferDevice));
    checkGPUError(cudaFree(mFoundTrackletsDevice));
    checkGPUError(cudaFree(mFoundCellsDevice));
  }
}

template <int nLayers>
void GpuTimeFrameChunk<nLayers>::allocate(const size_t nrof, Stream& stream)
{
  RANGE("device_partition_allocation", 2);
  mNRof = nrof;
  for (int i = 0; i < nLayers; ++i) {
    checkGPUError(cudaMallocAsync(reinterpret_cast<void**>(&(mClustersDevice[i])), sizeof(Cluster) * mTFGPUParams->clustersPerROfCapacity * nrof, stream.get()));
    checkGPUError(cudaMallocAsync(reinterpret_cast<void**>(&(mTrackingFrameInfoDevice[i])), sizeof(TrackingFrameInfo) * mTFGPUParams->clustersPerROfCapacity * nrof, stream.get()));
    checkGPUError(cudaMallocAsync(reinterpret_cast<void**>(&(mClusterExternalIndicesDevice[i])), sizeof(int) * mTFGPUParams->clustersPerROfCapacity * nrof, stream.get()));
    checkGPUError(cudaMallocAsync(reinterpret_cast<void**>(&(mIndexTablesDevice[i])), sizeof(int) * (256 * 128 + 1) * nrof, stream.get()));
    if (i < nLayers - 1) {
      checkGPUError(cudaMallocAsync(reinterpret_cast<void**>(&(mTrackletsLookupTablesDevice[i])), sizeof(int) * mTFGPUParams->clustersPerROfCapacity * nrof, stream.get()));
      checkGPUError(cudaMallocAsync(reinterpret_cast<void**>(&(mTrackletsDevice[i])), sizeof(Tracklet) * mTFGPUParams->maxTrackletsPerCluster * mTFGPUParams->clustersPerROfCapacity * nrof, stream.get()));
      if (i < nLayers - 2) {
        checkGPUError(cudaMallocAsync(reinterpret_cast<void**>(&(mCellsLookupTablesDevice[i])), sizeof(int) * mTFGPUParams->validatedTrackletsCapacity * nrof, stream.get()));
        checkGPUError(cudaMallocAsync(reinterpret_cast<void**>(&(mCellsDevice[i])), sizeof(Cell) * mTFGPUParams->validatedTrackletsCapacity * nrof, stream.get()));
        if (i < 2) {
          checkGPUError(cudaMallocAsync(reinterpret_cast<void**>(&(mNTrackletsPerClusterDevice[i])), sizeof(int) * mTFGPUParams->clustersPerROfCapacity * nrof, stream.get()));
        }
      }
    }
  }
  checkGPUError(cudaMallocAsync(reinterpret_cast<void**>(&mCUBTmpBufferDevice), mTFGPUParams->tmpCUBBufferSize * nrof, stream.get()));
  checkGPUError(cudaMallocAsync(reinterpret_cast<void**>(&mLinesDevice), sizeof(Line) * mTFGPUParams->maxTrackletsPerCluster * mTFGPUParams->clustersPerROfCapacity * nrof, stream.get()));
  checkGPUError(cudaMallocAsync(reinterpret_cast<void**>(&mNFoundLinesDevice), sizeof(int) * mTFGPUParams->clustersPerROfCapacity * nrof, stream.get()));
  checkGPUError(cudaMallocAsync(reinterpret_cast<void**>(&mNExclusiveFoundLinesDevice), sizeof(int) * mTFGPUParams->clustersPerROfCapacity * nrof + 1, stream.get())); // + 1 for cub::DeviceScan::ExclusiveSum, to cover cases where we have maximum number of clusters per ROF
  checkGPUError(cudaMallocAsync(reinterpret_cast<void**>(&mUsedTrackletsDevice), sizeof(unsigned char) * mTFGPUParams->maxTrackletsPerCluster * mTFGPUParams->clustersPerROfCapacity * nrof, stream.get()));
  checkGPUError(cudaMallocAsync(reinterpret_cast<void**>(&mClusteredLinesDevice), sizeof(int) * mTFGPUParams->clustersPerROfCapacity * mTFGPUParams->maxTrackletsPerCluster * nrof, stream.get()));

  /// Invariant allocations
  checkGPUError(cudaMallocAsync(reinterpret_cast<void**>(&mFoundTrackletsDevice), (nLayers - 1) * sizeof(int) * nrof, stream.get())); // No need to reset, we always read it after writing
  checkGPUError(cudaMallocAsync(reinterpret_cast<void**>(&mFoundCellsDevice), (nLayers - 2) * sizeof(int) * nrof, stream.get()));

  mAllocated = true;
}

template <int nLayers>
void GpuTimeFrameChunk<nLayers>::reset(const Task task, Stream& stream)
{
  RANGE("buffer_reset", 0);
  if ((bool)task) { // Vertexer-only initialisation (cannot be constexpr: due to the presence of gpu raw calls can't be put in header)
    for (int i = 0; i < 2; i++) {
      auto thrustTrackletsBegin = thrust::device_ptr<Tracklet>(mTrackletsDevice[i]);
      auto thrustTrackletsEnd = thrustTrackletsBegin + mTFGPUParams->maxTrackletsPerCluster * mTFGPUParams->clustersPerROfCapacity * mNRof;
      thrust::fill(THRUST_NAMESPACE::par.on(stream.get()), thrustTrackletsBegin, thrustTrackletsEnd, Tracklet{});
      checkGPUError(cudaMemsetAsync(mNTrackletsPerClusterDevice[i], 0, sizeof(int) * mTFGPUParams->clustersPerROfCapacity * mNRof, stream.get()));
    }
    checkGPUError(cudaMemsetAsync(mUsedTrackletsDevice, false, sizeof(unsigned char) * mTFGPUParams->maxTrackletsPerCluster * mTFGPUParams->clustersPerROfCapacity * mNRof, stream.get()));
    checkGPUError(cudaMemsetAsync(mClusteredLinesDevice, -1, sizeof(int) * mTFGPUParams->clustersPerROfCapacity * mTFGPUParams->maxTrackletsPerCluster * mNRof, stream.get()));
  } else {
    for (int i = 0; i < nLayers; ++i) {
      if (i < nLayers - 1) {
        checkGPUError(cudaMemsetAsync(mTrackletsLookupTablesDevice[i], 0, sizeof(int) * mTFGPUParams->clustersPerROfCapacity * mNRof, stream.get()));
        auto thrustTrackletsBegin = thrust::device_ptr<Tracklet>(mTrackletsDevice[i]);
        auto thrustTrackletsEnd = thrustTrackletsBegin + mTFGPUParams->maxTrackletsPerCluster * mTFGPUParams->clustersPerROfCapacity * mNRof;
        thrust::fill(THRUST_NAMESPACE::par.on(stream.get()), thrustTrackletsBegin, thrustTrackletsEnd, Tracklet{});
        if (i < nLayers - 2) {
          checkGPUError(cudaMemsetAsync(mCellsLookupTablesDevice[i], 0, sizeof(int) * mTFGPUParams->cellsLUTsize * mNRof, stream.get()));
        }
      }
    }
    checkGPUError(cudaMemsetAsync(mFoundCellsDevice, 0, (nLayers - 2) * sizeof(int), stream.get()));
  }
}

template <int nLayers>
size_t GpuTimeFrameChunk<nLayers>::computeScalingSizeBytes(const int nrof, const TimeFrameGPUParameters& config)
{
  size_t rofsize = nLayers * sizeof(int);                                                                      // number of clusters per ROF
  rofsize += nLayers * sizeof(Cluster) * config.clustersPerROfCapacity;                                        // clusters
  rofsize += nLayers * sizeof(TrackingFrameInfo) * config.clustersPerROfCapacity;                              // tracking frame info
  rofsize += nLayers * sizeof(int) * config.clustersPerROfCapacity;                                            // external cluster indices
  rofsize += nLayers * sizeof(int) * (256 * 128 + 1);                                                          // index tables
  rofsize += (nLayers - 1) * sizeof(int) * config.clustersPerROfCapacity;                                      // tracklets lookup tables
  rofsize += (nLayers - 1) * sizeof(Tracklet) * config.maxTrackletsPerCluster * config.clustersPerROfCapacity; // tracklets
  rofsize += 2 * sizeof(int) * config.clustersPerROfCapacity;                                                  // tracklets found per cluster (vertexer)
  rofsize += sizeof(unsigned char) * config.maxTrackletsPerCluster * config.clustersPerROfCapacity;            // used tracklets (vertexer)
  rofsize += (nLayers - 2) * sizeof(int) * config.validatedTrackletsCapacity;                                  // cells lookup tables
  rofsize += (nLayers - 2) * sizeof(Cell) * config.validatedTrackletsCapacity;                                 // cells
  rofsize += sizeof(Line) * config.maxTrackletsPerCluster * config.clustersPerROfCapacity;                     // lines
  rofsize += sizeof(int) * config.clustersPerROfCapacity;                                                      // found lines
  rofsize += sizeof(int) * config.clustersPerROfCapacity;                                                      // found lines exclusive sum
  rofsize += sizeof(int) * config.clustersPerROfCapacity * config.maxTrackletsPerCluster;                      // lines used in clusterlines

  rofsize += (nLayers - 1) * sizeof(int); // total found tracklets
  rofsize += (nLayers - 2) * sizeof(int); // total found cells

  return rofsize * nrof;
}

template <int nLayers>
size_t GpuTimeFrameChunk<nLayers>::computeFixedSizeBytes(const TimeFrameGPUParameters& config)
{
  size_t total = config.tmpCUBBufferSize;                  // CUB tmp buffers
  total += sizeof(gpu::StaticTrackingParameters<nLayers>); // static parameters loaded once
  return total;
}

template <int nLayers>
size_t GpuTimeFrameChunk<nLayers>::computeRofPerChunk(const TimeFrameGPUParameters& config, const size_t m)
{
  return (m * GB / (float)(config.nTimeFrameChunks) - GpuTimeFrameChunk<nLayers>::computeFixedSizeBytes(config)) / (float)GpuTimeFrameChunk<nLayers>::computeScalingSizeBytes(1, config);
}

/// Interface
template <int nLayers>
Cluster* GpuTimeFrameChunk<nLayers>::getDeviceClusters(const int layer)
{
  return mClustersDevice[layer];
}

template <int nLayers>
TrackingFrameInfo* GpuTimeFrameChunk<nLayers>::getDeviceTrackingFrameInfo(const int layer)
{
  return mTrackingFrameInfoDevice[layer];
}

template <int nLayers>
int* GpuTimeFrameChunk<nLayers>::getDeviceClusterExternalIndices(const int layer)
{
  return mClusterExternalIndicesDevice[layer];
}

template <int nLayers>
int* GpuTimeFrameChunk<nLayers>::getDeviceIndexTables(const int layer)
{
  return mIndexTablesDevice[layer];
}

template <int nLayers>
Tracklet* GpuTimeFrameChunk<nLayers>::getDeviceTracklets(const int layer)
{
  return mTrackletsDevice[layer];
}

template <int nLayers>
int* GpuTimeFrameChunk<nLayers>::getDeviceTrackletsLookupTables(const int layer)
{
  return mTrackletsLookupTablesDevice[layer];
}

template <int nLayers>
Cell* GpuTimeFrameChunk<nLayers>::getDeviceCells(const int layer)
{
  return mCellsDevice[layer];
}

template <int nLayers>
int* GpuTimeFrameChunk<nLayers>::getDeviceCellsLookupTables(const int layer)
{
  return mCellsLookupTablesDevice[layer];
}

// Load data
template <int nLayers>
size_t GpuTimeFrameChunk<nLayers>::loadDataOnDevice(const size_t startRof, const size_t maxRof, const int maxLayers, Stream& stream)
{
  RANGE("load_clusters_data", 5);
  auto nRofs = std::min(maxRof - startRof, mNRof);
  mNPopulatedRof = mTimeFramePtr->getNClustersROFrange(startRof, nRofs, 0).size();
  for (int i = 0; i < maxLayers; ++i) {
    mHostClusters[i] = mTimeFramePtr->getClustersPerROFrange(startRof, nRofs, i);
    mHostIndexTables[i] = mTimeFramePtr->getIndexTablePerROFrange(startRof, nRofs, i);
    if (mHostClusters[i].size() > mTFGPUParams->clustersPerROfCapacity * nRofs) {
      LOGP(warning, "Clusters on layer {} exceed the expected value, resizing to config value: {}, will lose information!", i, mTFGPUParams->clustersPerROfCapacity * nRofs);
    }
    checkGPUError(cudaMemcpyAsync(mClustersDevice[i],
                                  mHostClusters[i].data(),
                                  (int)std::min(mHostClusters[i].size(), mTFGPUParams->clustersPerROfCapacity * nRofs) * sizeof(Cluster),
                                  cudaMemcpyHostToDevice, stream.get()));
    if (mHostIndexTables[i].data()) {
      checkGPUError(cudaMemcpyAsync(mIndexTablesDevice[i],
                                    mHostIndexTables[i].data(),
                                    mHostIndexTables[i].size() * sizeof(int),
                                    cudaMemcpyHostToDevice, stream.get()));
    }
  }
  return mNPopulatedRof; // return the number of ROFs we loaded the data for.
}

/////////////////////////////////////////////////////////////////////////////////////////
// TimeFrameGPU
/////////////////////////////////////////////////////////////////////////////////////////
template <int nLayers>
TimeFrameGPU<nLayers>::TimeFrameGPU()
{
  mIsGPU = true;
  utils::getDeviceProp(0, true);
}

template <int nLayers>
TimeFrameGPU<nLayers>::~TimeFrameGPU() = default;

template <int nLayers>
void TimeFrameGPU<nLayers>::registerHostMemory(const int maxLayers)
{
  if (mHostRegistered) {
    return;
  } else {
    mHostRegistered = true;
  }
  for (auto iLayer{0}; iLayer < maxLayers; ++iLayer) {
    checkGPUError(cudaHostRegister(mClusters[iLayer].data(), mClusters[iLayer].size() * sizeof(Cluster), cudaHostRegisterPortable));
    checkGPUError(cudaHostRegister(mNClustersPerROF[iLayer].data(), mNClustersPerROF[iLayer].size() * sizeof(int), cudaHostRegisterPortable));
    checkGPUError(cudaHostRegister(mIndexTables[iLayer].data(), (mStaticTrackingParams.ZBins * mStaticTrackingParams.PhiBins + 1) * mNrof * sizeof(int), cudaHostRegisterPortable));
  }
  checkGPUError(cudaHostRegister(mHostNTracklets.data(), (nLayers - 1) * mGpuParams.nTimeFrameChunks * sizeof(int), cudaHostRegisterPortable));
}

template <int nLayers>
void TimeFrameGPU<nLayers>::unregisterHostMemory(const int maxLayers)
{
  if (!mHostRegistered) {
    return;
  }
  for (auto iLayer{0}; iLayer < maxLayers; ++iLayer) {
    checkGPUError(cudaHostUnregister(mClusters[iLayer].data()));
    checkGPUError(cudaHostUnregister(mNClustersPerROF[iLayer].data()));
    checkGPUError(cudaHostUnregister(mIndexTables[iLayer].data()));
  }
  checkGPUError(cudaHostUnregister(mHostNTracklets.data()));
  mHostRegistered = false;
}

template <int nLayers>
void TimeFrameGPU<nLayers>::initialise(const int iteration,
                                       const TrackingParameters& trkParam,
                                       const int maxLayers,
                                       IndexTableUtils* utils,
                                       const TimeFrameGPUParameters* gpuParam)
{
  mGpuStreams.resize(mGpuParams.nTimeFrameChunks);
  mHostNTracklets.resize((nLayers - 1) * mGpuParams.nTimeFrameChunks, 0);

  auto init = [&](int p) -> void {
    this->initDevice(p, utils, trkParam, *gpuParam, maxLayers, iteration);
  };
  std::thread t1{init, mGpuParams.nTimeFrameChunks};
  RANGE("tf_cpu_initialisation", 1);
  o2::its::TimeFrame::initialise(iteration, trkParam, maxLayers);
  registerHostMemory(maxLayers);
  t1.join();
}

template <int nLayers>
void TimeFrameGPU<nLayers>::wipe(const int maxLayers)
{
  unregisterHostMemory(maxLayers);
}

template <int nLayers>
void TimeFrameGPU<nLayers>::initDevice(const int chunks,
                                       IndexTableUtils* utils,
                                       const TrackingParameters& trkParam,
                                       const TimeFrameGPUParameters& gpuParam,
                                       const int maxLayers,
                                       const int iteration)
{
  mStaticTrackingParams.ZBins = trkParam.ZBins;
  mStaticTrackingParams.PhiBins = trkParam.PhiBins;
  if (mFirstInit) {
    mGpuParams = gpuParam;
    if (mGpuParams.maxGPUMemoryGB < 0) {
      // Adaptive to available memory, hungry mode
      size_t free;
      checkGPUError(cudaMemGetInfo(&free, nullptr));
      mAvailMemGB = (double)free / GB;
      LOGP(info, "Hungry memory mode requested, found {} free GB, going to use all of them", mAvailMemGB);
    } else {
      mAvailMemGB = mGpuParams.maxGPUMemoryGB;
      LOGP(info, "Fixed memory mode requested, will try to use {} GB", mAvailMemGB);
    }
    checkGPUError(cudaMalloc(reinterpret_cast<void**>(&mTrackingParamsDevice), sizeof(gpu::StaticTrackingParameters<nLayers>)));
    checkGPUError(cudaMemcpy(mTrackingParamsDevice, &mStaticTrackingParams, sizeof(gpu::StaticTrackingParameters<nLayers>), cudaMemcpyHostToDevice));
    if (utils) { // If utils is not nullptr, then its gpu vertexing
      mIndexTableUtils = *utils;
      checkGPUError(cudaMalloc(reinterpret_cast<void**>(&mIndexTableUtilsDevice), sizeof(IndexTableUtils)));
    } else { // GPU tracking otherwise
      mIndexTableUtils.setTrackingParameters(trkParam);
    }

    mMemChunks.resize(chunks, GpuTimeFrameChunk<nLayers>{static_cast<TimeFrame*>(this), mGpuParams});
    mVerticesInChunks.resize(chunks);
    mNVerticesInChunks.resize(chunks);
    mLabelsInChunks.resize(chunks);
    LOGP(debug, "Size of fixed part is: {} MB", GpuTimeFrameChunk<nLayers>::computeFixedSizeBytes(mGpuParams) / MB);
    LOGP(debug, "Size of scaling part is: {} MB", GpuTimeFrameChunk<nLayers>::computeScalingSizeBytes(GpuTimeFrameChunk<nLayers>::computeRofPerChunk(mGpuParams, mAvailMemGB), mGpuParams) / MB);
    LOGP(info, "Allocating {} chunks of {} rofs capacity each.", chunks, GpuTimeFrameChunk<nLayers>::computeRofPerChunk(mGpuParams, mAvailMemGB));

    for (int iChunk{0}; iChunk < mMemChunks.size(); ++iChunk) {
      mMemChunks[iChunk].allocate(GpuTimeFrameChunk<nLayers>::computeRofPerChunk(mGpuParams, mAvailMemGB), mGpuStreams[iChunk]);
    }
    for (auto iLayer{0}; iLayer < nLayers; ++iLayer) {
      checkGPUError(cudaMalloc(reinterpret_cast<void**>(&mROframesClustersDevice[iLayer]), mROframesClusters[iLayer].size() * sizeof(int)));
      checkGPUError(cudaMalloc(reinterpret_cast<void**>(&(mUsedClustersDevice[iLayer])), sizeof(unsigned char) * mGpuParams.clustersPerROfCapacity * mNrof));
    }
    checkGPUError(cudaMalloc(reinterpret_cast<void**>(&mVerticesDevice), sizeof(Vertex) * mGpuParams.maxVerticesCapacity));
    checkGPUError(cudaMalloc(reinterpret_cast<void**>(&mROframesPVDevice), sizeof(int) * (mNrof + 1)));

    mFirstInit = false;
  }
  if (maxLayers < nLayers) { // Vertexer
    for (auto iLayer{0}; iLayer < nLayers; ++iLayer) {
      checkGPUError(cudaMemcpy(mROframesClustersDevice[iLayer], mROframesClusters[iLayer].data(), mROframesClusters[iLayer].size() * sizeof(int), cudaMemcpyHostToDevice));
    }
  } else { // Tracker
    checkGPUError(cudaMemcpy(mVerticesDevice, mPrimaryVertices.data(), sizeof(Vertex) * mPrimaryVertices.size(), cudaMemcpyHostToDevice));
    checkGPUError(cudaMemcpy(mROframesPVDevice, mROframesPV.data(), sizeof(int) * mROframesPV.size(), cudaMemcpyHostToDevice));
    if (!iteration) {
      for (auto iLayer{0}; iLayer < nLayers; ++iLayer) {
        checkGPUError(cudaMemset(mUsedClustersDevice[iLayer], 0, sizeof(unsigned char) * mGpuParams.clustersPerROfCapacity * mNrof));
      }
    }
  }
  checkGPUError(cudaMemcpy(mIndexTableUtilsDevice, &mIndexTableUtils, sizeof(IndexTableUtils), cudaMemcpyHostToDevice));
}

template <int nLayers>
unsigned char* TimeFrameGPU<nLayers>::getDeviceUsedClusters(const int layer)
{
  return mUsedClustersDevice[layer];
}

template <int nLayers>
gsl::span<int> TimeFrameGPU<nLayers>::getHostNTracklets(const int chunkId)
{
  return gsl::span<int>(mHostNTracklets.data() + (nLayers - 1) * chunkId, nLayers - 1);
}

template class TimeFrameGPU<7>;
template class GpuTimeFrameChunk<7>;
} // namespace gpu
} // namespace its
} // namespace o2