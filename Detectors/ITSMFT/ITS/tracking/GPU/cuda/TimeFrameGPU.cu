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

template <int nLayers>
struct StaticTrackingParameters {
  StaticTrackingParameters<nLayers>& operator=(const StaticTrackingParameters<nLayers>& t) = default;
  void set(const TrackingParameters& pars)
  {
    ClusterSharing = pars.ClusterSharing;
    MinTrackLength = pars.MinTrackLength;
    NSigmaCut = pars.NSigmaCut;
    PVres = pars.PVres;
    DeltaROF = pars.DeltaROF;
    ZBins = pars.ZBins;
    PhiBins = pars.PhiBins;
    CellDeltaTanLambdaSigma = pars.CellDeltaTanLambdaSigma;
  }

  /// General parameters
  int ClusterSharing = 0;
  int MinTrackLength = nLayers;
  float NSigmaCut = 5;
  float PVres = 1.e-2f;
  int DeltaROF = 0;
  int ZBins{256};
  int PhiBins{128};

  /// Cell finding cuts
  float CellDeltaTanLambdaSigma = 0.007f;
};

/////////////////////////////////////////////////////////////////////////////////////////
// GpuPartition
template <int nLayers>
GpuTimeFramePartition<nLayers>::~GpuTimeFramePartition()
{
  if (mAllocated) {
    for (int i = 0; i < nLayers; ++i) {
      checkGPUError(cudaFree(mROframesClustersDevice[i]));
      checkGPUError(cudaFree(mClustersDevice[i]));
      checkGPUError(cudaFree(mUsedClustersDevice[i]));
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
void GpuTimeFramePartition<nLayers>::allocate(const size_t nrof, Stream& stream)
{
  RANGE("device_partition_allocation", 2);
  mNRof = nrof;
  for (int i = 0; i < nLayers; ++i) {
    checkGPUError(cudaMallocAsync(reinterpret_cast<void**>(&(mROframesClustersDevice[i])), sizeof(int) * nrof, stream.get()));
    checkGPUError(cudaMallocAsync(reinterpret_cast<void**>(&(mClustersDevice[i])), sizeof(Cluster) * mTFGconf->clustersPerROfCapacity * nrof, stream.get()));
    checkGPUError(cudaMallocAsync(reinterpret_cast<void**>(&(mUsedClustersDevice[i])), sizeof(unsigned char) * mTFGconf->clustersPerROfCapacity * nrof, stream.get()));
    checkGPUError(cudaMallocAsync(reinterpret_cast<void**>(&(mTrackingFrameInfoDevice[i])), sizeof(TrackingFrameInfo) * mTFGconf->clustersPerROfCapacity * nrof, stream.get()));
    checkGPUError(cudaMallocAsync(reinterpret_cast<void**>(&(mClusterExternalIndicesDevice[i])), sizeof(int) * mTFGconf->clustersPerROfCapacity * nrof, stream.get()));
    checkGPUError(cudaMallocAsync(reinterpret_cast<void**>(&(mIndexTablesDevice[i])), sizeof(int) * (256 * 128 + 1) * nrof, stream.get()));
    if (i < nLayers - 1) {
      checkGPUError(cudaMallocAsync(reinterpret_cast<void**>(&(mTrackletsLookupTablesDevice[i])), sizeof(int) * mTFGconf->clustersPerROfCapacity * nrof, stream.get()));
      checkGPUError(cudaMallocAsync(reinterpret_cast<void**>(&(mTrackletsDevice[i])), sizeof(Tracklet) * mTFGconf->maxTrackletsPerCluster * mTFGconf->clustersPerROfCapacity * nrof, stream.get()));
      if (i < nLayers - 2) {
        checkGPUError(cudaMallocAsync(reinterpret_cast<void**>(&(mCellsLookupTablesDevice[i])), sizeof(int) * mTFGconf->validatedTrackletsCapacity * nrof, stream.get()));
        checkGPUError(cudaMallocAsync(reinterpret_cast<void**>(&(mCellsDevice[i])), sizeof(Cell) * mTFGconf->validatedTrackletsCapacity * nrof, stream.get()));
        if (i < 2) {
          checkGPUError(cudaMallocAsync(reinterpret_cast<void**>(&(mNTrackletsPerClusterDevice[i])), sizeof(int) * mTFGconf->clustersPerROfCapacity * nrof, stream.get()));
        }
      }
    }
  }
  checkGPUError(cudaMallocAsync(reinterpret_cast<void**>(&mCUBTmpBufferDevice), mTFGconf->tmpCUBBufferSize * nrof, stream.get()));
  checkGPUError(cudaMallocAsync(reinterpret_cast<void**>(&mLinesDevice), sizeof(Line) * mTFGconf->maxTrackletsPerCluster * mTFGconf->clustersPerROfCapacity * nrof, stream.get()));
  checkGPUError(cudaMallocAsync(reinterpret_cast<void**>(&mNFoundLinesDevice), sizeof(int) * mTFGconf->clustersPerROfCapacity * nrof, stream.get()));
  checkGPUError(cudaMallocAsync(reinterpret_cast<void**>(&mNExclusiveFoundLinesDevice), sizeof(int) * mTFGconf->clustersPerROfCapacity * nrof, stream.get()));

  /// Invariant allocations
  checkGPUError(cudaMallocAsync(reinterpret_cast<void**>(&mFoundTrackletsDevice), (nLayers - 1) * sizeof(int) * nrof, stream.get()));
  checkGPUError(cudaMallocAsync(reinterpret_cast<void**>(&mFoundCellsDevice), (nLayers - 2) * sizeof(int) * nrof, stream.get()));

  mAllocated = true;
}

template <int nLayers>
void GpuTimeFramePartition<nLayers>::reset(const size_t nrof, const Task task, Stream& stream)
{
  RANGE("buffer_reset", 0);
  if ((bool)task) { // Vertexer-only initialisation (cannot be constexpr: due to the presence of gpu raw calls can't be put in header)
    std::vector<std::thread> t;
    for (int i = 0; i < 2; i++) {
      auto thrustTrackletsBegin = thrust::device_ptr<Tracklet>(mTrackletsDevice[i]);
      auto thrustTrackletsEnd = thrustTrackletsBegin + mTFGconf->maxTrackletsPerCluster * mTFGconf->clustersPerROfCapacity * nrof;
      thrust::fill(THRUST_NAMESPACE::par.on(stream.get()), thrustTrackletsBegin, thrustTrackletsEnd, Tracklet{});
      checkGPUError(cudaMemsetAsync(mTrackletsLookupTablesDevice[i], 0, sizeof(int) * mTFGconf->clustersPerROfCapacity * nrof, stream.get()));
      checkGPUError(cudaMemsetAsync(mNTrackletsPerClusterDevice[i], 0, sizeof(int) * mTFGconf->clustersPerROfCapacity * nrof, stream.get()));
    }
  } else {
    for (int i = 0; i < nLayers - 1; ++i) {
      checkGPUError(cudaMemsetAsync(mTrackletsLookupTablesDevice[i], 0, sizeof(int) * mTFGconf->clustersPerROfCapacity * nrof, stream.get()));
      auto thrustTrackletsBegin = thrust::device_ptr<Tracklet>(mTrackletsDevice[i]);
      auto thrustTrackletsEnd = thrustTrackletsBegin + mTFGconf->maxTrackletsPerCluster * mTFGconf->clustersPerROfCapacity * nrof;
      thrust::fill(THRUST_NAMESPACE::par.on(stream.get()), thrustTrackletsBegin, thrustTrackletsEnd, Tracklet{});
      if (i < nLayers - 2) {
        checkGPUError(cudaMemsetAsync(mCellsLookupTablesDevice[i], 0, sizeof(int) * mTFGconf->cellsLUTsize * nrof, stream.get()));
      }
    }
    checkGPUError(cudaMemsetAsync(mFoundCellsDevice, 0, (nLayers - 2) * sizeof(int), stream.get()));
  }
}

template <int nLayers>
size_t GpuTimeFramePartition<nLayers>::computeScalingSizeBytes(const int nrof, const TimeFrameGPUConfig& config)
{
  size_t rofsize = nLayers * sizeof(int);                                                                      // number of clusters per ROF
  rofsize += nLayers * sizeof(Cluster) * config.clustersPerROfCapacity;                                        // clusters
  rofsize += nLayers * sizeof(unsigned char) * config.clustersPerROfCapacity;                                  // used clusters flags
  rofsize += nLayers * sizeof(TrackingFrameInfo) * config.clustersPerROfCapacity;                              // tracking frame info
  rofsize += nLayers * sizeof(int) * config.clustersPerROfCapacity;                                            // external cluster indices
  rofsize += nLayers * sizeof(int) * (256 * 128 + 1);                                                          // index tables
  rofsize += (nLayers - 1) * sizeof(int) * config.clustersPerROfCapacity;                                      // tracklets lookup tables
  rofsize += (nLayers - 1) * sizeof(Tracklet) * config.maxTrackletsPerCluster * config.clustersPerROfCapacity; // tracklets
  rofsize += 2 * sizeof(int) * config.clustersPerROfCapacity;                                                  // tracklets found per cluster (vertexer)
  rofsize += (nLayers - 2) * sizeof(int) * config.validatedTrackletsCapacity;                                  // cells lookup tables
  rofsize += (nLayers - 2) * sizeof(Cell) * config.validatedTrackletsCapacity;                                 // cells
  rofsize += sizeof(Line) * config.maxTrackletsPerCluster * config.clustersPerROfCapacity;                     // lines
  rofsize += sizeof(int) * config.clustersPerROfCapacity;                                                      // found lines
  rofsize += sizeof(int) * config.clustersPerROfCapacity;                                                      // found lines exclusive sum

  rofsize += (nLayers - 1) * sizeof(int); // total found tracklets
  rofsize += (nLayers - 2) * sizeof(int); // total found cells

  return rofsize * nrof;
}

template <int nLayers>
size_t GpuTimeFramePartition<nLayers>::computeFixedSizeBytes(const TimeFrameGPUConfig& config)
{
  size_t total = config.tmpCUBBufferSize;                  // CUB tmp buffers
  total += sizeof(gpu::StaticTrackingParameters<nLayers>); // static parameters loaded once
  return total;
}

template <int nLayers>
size_t GpuTimeFramePartition<nLayers>::computeRofPerPartition(const TimeFrameGPUConfig& config, const size_t m)
{
  return (m * GB / (float)(config.nTimeFramePartitions) - GpuTimeFramePartition<nLayers>::computeFixedSizeBytes(config)) / (float)GpuTimeFramePartition<nLayers>::computeScalingSizeBytes(1, config);
}

/// Interface
template <int nLayers>
int* GpuTimeFramePartition<nLayers>::getDeviceROframesClusters(const int layer)
{
  return mROframesClustersDevice[layer];
}

template <int nLayers>
Cluster* GpuTimeFramePartition<nLayers>::getDeviceClusters(const int layer)
{
  return mClustersDevice[layer];
}

template <int nLayers>
unsigned char* GpuTimeFramePartition<nLayers>::getDeviceUsedClusters(const int layer)
{
  return mUsedClustersDevice[layer];
}

template <int nLayers>
TrackingFrameInfo* GpuTimeFramePartition<nLayers>::getDeviceTrackingFrameInfo(const int layer)
{
  return mTrackingFrameInfoDevice[layer];
}

template <int nLayers>
int* GpuTimeFramePartition<nLayers>::getDeviceClusterExternalIndices(const int layer)
{
  return mClusterExternalIndicesDevice[layer];
}

template <int nLayers>
int* GpuTimeFramePartition<nLayers>::getDeviceIndexTables(const int layer)
{
  return mIndexTablesDevice[layer];
}

template <int nLayers>
Tracklet* GpuTimeFramePartition<nLayers>::getDeviceTracklets(const int layer)
{
  return mTrackletsDevice[layer];
}

template <int nLayers>
int* GpuTimeFramePartition<nLayers>::getDeviceTrackletsLookupTables(const int layer)
{
  return mTrackletsLookupTablesDevice[layer];
}

template <int nLayers>
Cell* GpuTimeFramePartition<nLayers>::getDeviceCells(const int layer)
{
  return mCellsDevice[layer];
}

template <int nLayers>
int* GpuTimeFramePartition<nLayers>::getDeviceCellsLookupTables(const int layer)
{
  return mCellsLookupTablesDevice[layer];
}

// Load data
template <int nLayers>
size_t GpuTimeFramePartition<nLayers>::copyDeviceData(const size_t startRof, const int maxLayers, Stream& stream)
{
  RANGE("load_clusters_data", 5);
  for (int i = 0; i < maxLayers; ++i) {
    mHostClusters[i] = mTimeFramePtr->getClustersPerROFrange(startRof, mNRof, i);
    mHostROframesClusters[i] = mTimeFramePtr->getROframesClustersPerROFrange(startRof, mNRof, i);
    if (mHostClusters[i].size() > mTFGconf->clustersPerROfCapacity * mNRof) {
      LOGP(warning, "Excess of expected clusters on layer {}, resizing to config value: {}, will lose information!", i, mTFGconf->clustersPerROfCapacity * mNRof);
    }
    checkGPUError(cudaMemcpyAsync(mClustersDevice[i], mHostClusters[i].data(), (int)std::min(mHostClusters[i].size(), mTFGconf->clustersPerROfCapacity * mNRof) * sizeof(Cluster), cudaMemcpyHostToDevice, stream.get()));
    checkGPUError(cudaMemcpyAsync(mROframesClustersDevice[i], mHostROframesClusters[i].data(), mHostROframesClusters[i].size() * sizeof(int), cudaMemcpyHostToDevice, stream.get()));
  }
  return mHostROframesClusters[0].size(); // We want to return for how much ROFs we loaded the data.
}

/////////////////////////////////////////////////////////////////////////////////////////
// TimeFrameGPU
template <int nLayers>
TimeFrameGPU<nLayers>::TimeFrameGPU()
{
  mIsGPU = true;
  utils::getDeviceProp(0, true);
  if (mGpuConfig.maxGPUMemoryGB < 0) {
    // Adaptive to available memory, hungry mode
    size_t free;
    checkGPUError(cudaMemGetInfo(&free, nullptr));
    mAvailMemGB = (double)free / GB;
    LOGP(info, "Hungry memory mode requested, found {} free GB, going to use all of them", mAvailMemGB);
  } else {
    mAvailMemGB = mGpuConfig.maxGPUMemoryGB;
    LOGP(info, "Fixed memory mode requested, will try to use {} GB", mAvailMemGB);
  }
}

template <int nLayers>
TimeFrameGPU<nLayers>::~TimeFrameGPU()
{
  // checkGPUError(cudaFree(mTrackingParamsDevice));
}

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
    checkGPUError(cudaHostRegister(mROframesClusters[iLayer].data(), mROframesClusters[iLayer].size() * sizeof(int), cudaHostRegisterPortable));
  }
}

template <int nLayers>
void TimeFrameGPU<nLayers>::unregisterHostMemory(const int maxLayers)
{
  if (!mHostRegistered) {
    return;
  }
  for (auto iLayer{0}; iLayer < maxLayers; ++iLayer) {
    checkGPUError(cudaHostUnregister(mClusters[iLayer].data()));
    checkGPUError(cudaHostUnregister(mROframesClusters[iLayer].data()));
  }
}

template <int nLayers>
void TimeFrameGPU<nLayers>::initialise(const int iteration,
                                       const TrackingParameters& trkParam,
                                       const int maxLayers,
                                       const IndexTableUtils* utils)
{
  mGpuStreams.resize(mGpuConfig.nTimeFramePartitions);
  auto init = [&](int p) -> void {
    this->initDevice(p, utils, maxLayers);
  };
  std::thread t1{init, mGpuConfig.nTimeFramePartitions};
  RANGE("tf_cpu_initialisation", 1);
  o2::its::TimeFrame::initialise(iteration, trkParam, maxLayers);
  registerHostMemory(maxLayers);
  t1.join();
}

template <int nLayers>
void TimeFrameGPU<nLayers>::initDevice(const int partitions, const IndexTableUtils* utils, const int maxLayers)
{
  StaticTrackingParameters<nLayers> pars;
  checkGPUError(cudaMalloc(reinterpret_cast<void**>(&mTrackingParamsDevice), sizeof(gpu::StaticTrackingParameters<nLayers>)));
  checkGPUError(cudaMemcpy(mTrackingParamsDevice, &pars, sizeof(gpu::StaticTrackingParameters<nLayers>), cudaMemcpyHostToDevice));
  if (utils) {
    checkGPUError(cudaMalloc(reinterpret_cast<void**>(&mDeviceIndexTableUtils), sizeof(IndexTableUtils)));
    checkGPUError(cudaMemcpy(mDeviceIndexTableUtils, utils, sizeof(IndexTableUtils), cudaMemcpyHostToDevice));
  }
  mMemPartitions.resize(partitions, GpuTimeFramePartition<nLayers>{static_cast<TimeFrame*>(this), mGpuConfig});
  LOGP(debug, "Size of fixed part is: {} MB", GpuTimeFramePartition<nLayers>::computeFixedSizeBytes(mGpuConfig) / MB);
  LOGP(debug, "Size of scaling part is: {} MB", GpuTimeFramePartition<nLayers>::computeScalingSizeBytes(GpuTimeFramePartition<nLayers>::computeRofPerPartition(mGpuConfig, mAvailMemGB), mGpuConfig) / MB);
  LOGP(info, "Allocating {} partitions counting {} rofs each.", partitions, GpuTimeFramePartition<nLayers>::computeRofPerPartition(mGpuConfig, mAvailMemGB));

  initDevicePartitions(GpuTimeFramePartition<nLayers>::computeRofPerPartition(mGpuConfig, mAvailMemGB), maxLayers);
}

template <int nLayers>
void TimeFrameGPU<nLayers>::initDevicePartitions(const int nRof, const int maxLayers)
{
  if (mDeviceInitialised) {
    return;
  } else {
    mDeviceInitialised = true;
  }
  if (!mMemPartitions.size()) {
    LOGP(fatal, "gpu-tracking: TimeFrame GPU partitions not created");
  }
  for (int iPartition{0}; iPartition < mMemPartitions.size(); ++iPartition) {
    mMemPartitions[iPartition].allocate(nRof, mGpuStreams[iPartition]);
    mMemPartitions[iPartition].reset(nRof, maxLayers < nLayers ? gpu::Task::Vertexer : gpu::Task::Tracker, mGpuStreams[iPartition]);
  }
}

template class TimeFrameGPU<7>;
template class GpuTimeFramePartition<7>;
} // namespace gpu
} // namespace its
} // namespace o2
