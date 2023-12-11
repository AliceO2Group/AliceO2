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
#ifndef __HIPCC__
#include "GPUReconstructionCUDADef.h" // This file should come first, if included
#endif

#include <thrust/fill.h>
#include <thrust/execution_policy.h>

#include "ITStracking/Constants.h"

#include "ITStrackingGPU/Utils.h"
#include "ITStrackingGPU/TimeFrameGPU.h"
#include "ITStrackingGPU/TracerGPU.h"

#include <unistd.h>
#include <thread>

#define GPUCA_TPC_GEOMETRY_O2 // To set working switch in GPUTPCGeometry whose else statement is bugged
#define GPUCA_O2_INTERFACE    // To suppress errors related to the weird dependency between itsgputracking and GPUTracking

#ifndef __HIPCC__
#define THRUST_NAMESPACE thrust::cuda
#else
#define THRUST_NAMESPACE thrust::hip
// clang-format off
#ifndef GPUCA_NO_CONSTANT_MEMORY
  #ifdef GPUCA_CONSTANT_AS_ARGUMENT
    #define GPUCA_CONSMEM_PTR const GPUConstantMemCopyable gGPUConstantMemBufferByValue,
    #define GPUCA_CONSMEM_CALL gGPUConstantMemBufferHost,
    #define GPUCA_CONSMEM (const_cast<GPUConstantMem&>(gGPUConstantMemBufferByValue.v))
  #else
    #define GPUCA_CONSMEM_PTR
    #define GPUCA_CONSMEM_CALL
    #define GPUCA_CONSMEM (gGPUConstantMemBuffer.v)
  #endif
#else
  #define GPUCA_CONSMEM_PTR const GPUConstantMem *gGPUConstantMemBuffer,
  #define GPUCA_CONSMEM_CALL me->mDeviceConstantMem,
  #define GPUCA_CONSMEM const_cast<GPUConstantMem&>(*gGPUConstantMemBuffer)
#endif
#define GPUCA_KRNL_BACKEND_CLASS GPUReconstructionHIPBackend
// clang-format on
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

void* DefaultGPUAllocator::allocate(size_t size)
{
  LOGP(info, "Called DefaultGPUAllocator::allocate with size {}", size);
  return nullptr; // to be implemented
}

/////////////////////////////////////////////////////////////////////////////////////////
// GpuChunk
/////////////////////////////////////////////////////////////////////////////////////////
template <int nLayers>
GpuTimeFrameChunk<nLayers>::~GpuTimeFrameChunk()
{
  if (mAllocated) {
    for (int i = 0; i < nLayers; ++i) {
      checkGPUError(cudaFree(mClustersDevice[i]));
      // checkGPUError(cudaFree(mTrackingFrameInfoDevice[i]));
      checkGPUError(cudaFree(mClusterExternalIndicesDevice[i]));
      checkGPUError(cudaFree(mIndexTablesDevice[i]));
      if (i < nLayers - 1) {
        checkGPUError(cudaFree(mTrackletsDevice[i]));
        checkGPUError(cudaFree(mTrackletsLookupTablesDevice[i]));
        if (i < nLayers - 2) {
          checkGPUError(cudaFree(mCellsDevice[i]));
          checkGPUError(cudaFree(mCellsLookupTablesDevice[i]));
          checkGPUError(cudaFree(mRoadsLookupTablesDevice[i]));
          if (i < nLayers - 3) {
            checkGPUError(cudaFree(mNeighboursCellLookupTablesDevice[i]));
            checkGPUError(cudaFree(mNeighboursCellDevice[i]));
          }
        }
      }
    }
    // checkGPUError(cudaFree(mRoadsDevice));
    checkGPUError(cudaFree(mCUBTmpBufferDevice));
    checkGPUError(cudaFree(mFoundTrackletsDevice));
    checkGPUError(cudaFree(mNFoundCellsDevice));
    checkGPUError(cudaFree(mCellsDeviceArray));
    checkGPUError(cudaFree(mNeighboursCellDeviceArray));
    checkGPUError(cudaFree(mNeighboursCellLookupTablesDeviceArray));
  }
}

template <int nLayers>
void GpuTimeFrameChunk<nLayers>::allocate(const size_t nrof, Stream& stream)
{
  RANGE("device_partition_allocation", 2);
  mNRof = nrof;
  for (int i = 0; i < nLayers; ++i) {
    checkGPUError(cudaMallocAsync(reinterpret_cast<void**>(&(mClustersDevice[i])), sizeof(Cluster) * mTFGPUParams->clustersPerROfCapacity * nrof, stream.get()));
    // checkGPUError(cudaMallocAsync(reinterpret_cast<void**>(&(mTrackingFrameInfoDevice[i])), sizeof(TrackingFrameInfo) * mTFGPUParams->clustersPerROfCapacity * nrof, stream.get()));
    checkGPUError(cudaMallocAsync(reinterpret_cast<void**>(&(mClusterExternalIndicesDevice[i])), sizeof(int) * mTFGPUParams->clustersPerROfCapacity * nrof, stream.get()));
    checkGPUError(cudaMallocAsync(reinterpret_cast<void**>(&(mIndexTablesDevice[i])), sizeof(int) * (256 * 128 + 1) * nrof, stream.get()));
    if (i < nLayers - 1) {
      checkGPUError(cudaMallocAsync(reinterpret_cast<void**>(&(mTrackletsLookupTablesDevice[i])), sizeof(int) * mTFGPUParams->clustersPerROfCapacity * nrof, stream.get()));
      checkGPUError(cudaMallocAsync(reinterpret_cast<void**>(&(mTrackletsDevice[i])), sizeof(Tracklet) * mTFGPUParams->maxTrackletsPerCluster * mTFGPUParams->clustersPerROfCapacity * nrof, stream.get()));
      if (i < nLayers - 2) {
        checkGPUError(cudaMallocAsync(reinterpret_cast<void**>(&(mCellsLookupTablesDevice[i])), sizeof(int) * mTFGPUParams->validatedTrackletsCapacity * nrof, stream.get()));
        checkGPUError(cudaMallocAsync(reinterpret_cast<void**>(&(mCellsDevice[i])), sizeof(CellSeed) * mTFGPUParams->maxNeighboursSize * nrof, stream.get()));
        checkGPUError(cudaMallocAsync(reinterpret_cast<void**>(&mRoadsLookupTablesDevice[i]), sizeof(int) * mTFGPUParams->maxNeighboursSize * nrof, stream.get()));
        if (i < nLayers - 3) {
          checkGPUError(cudaMallocAsync(reinterpret_cast<void**>(&(mNeighboursCellLookupTablesDevice[i])), sizeof(int) * mTFGPUParams->maxNeighboursSize * nrof, stream.get()));
          checkGPUError(cudaMallocAsync(reinterpret_cast<void**>(&(mNeighboursCellDevice[i])), sizeof(int) * mTFGPUParams->maxNeighboursSize * nrof, stream.get()));
        }
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
  // checkGPUError(cudaMallocAsync(reinterpret_cast<void**>(&mRoadsDevice), sizeof(Road<nLayers - 2>) * mTFGPUParams->maxRoadPerRofSize * nrof, stream.get()));

  /// Invariant allocations
  checkGPUError(cudaMallocAsync(reinterpret_cast<void**>(&mFoundTrackletsDevice), (nLayers - 1) * sizeof(int) * nrof, stream.get())); // No need to reset, we always read it after writing
  checkGPUError(cudaMallocAsync(reinterpret_cast<void**>(&mNFoundCellsDevice), (nLayers - 2) * sizeof(int) * nrof, stream.get()));
  checkGPUError(cudaMallocAsync(reinterpret_cast<void**>(&mCellsDeviceArray), (nLayers - 2) * sizeof(CellSeed*), stream.get()));
  checkGPUError(cudaMallocAsync(reinterpret_cast<void**>(&mNeighboursCellDeviceArray), (nLayers - 3) * sizeof(int*), stream.get()));
  checkGPUError(cudaMallocAsync(reinterpret_cast<void**>(&mNeighboursCellLookupTablesDeviceArray), (nLayers - 3) * sizeof(int*), stream.get()));

  /// Copy pointers of allocated memory to regrouping arrays
  checkGPUError(cudaMemcpyAsync(mCellsDeviceArray, mCellsDevice.data(), (nLayers - 2) * sizeof(CellSeed*), cudaMemcpyHostToDevice, stream.get()));
  checkGPUError(cudaMemcpyAsync(mNeighboursCellDeviceArray, mNeighboursCellDevice.data(), (nLayers - 3) * sizeof(int*), cudaMemcpyHostToDevice, stream.get()));
  checkGPUError(cudaMemcpyAsync(mNeighboursCellLookupTablesDeviceArray, mNeighboursCellLookupTablesDevice.data(), (nLayers - 3) * sizeof(int*), cudaMemcpyHostToDevice, stream.get()));

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
          checkGPUError(cudaMemsetAsync(mRoadsLookupTablesDevice[i], 0, sizeof(int) * mTFGPUParams->maxNeighboursSize * mNRof, stream.get()));
          if (i < nLayers - 3) {
            checkGPUError(cudaMemsetAsync(mNeighboursCellLookupTablesDevice[i], 0, sizeof(int) * mTFGPUParams->maxNeighboursSize * mNRof, stream.get()));
            checkGPUError(cudaMemsetAsync(mNeighboursCellDevice[i], 0, sizeof(int) * mTFGPUParams->maxNeighboursSize * mNRof, stream.get()));
          }
        }
      }
    }
    checkGPUError(cudaMemsetAsync(mNFoundCellsDevice, 0, (nLayers - 2) * sizeof(int), stream.get()));
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
  rofsize += (nLayers - 2) * sizeof(CellSeed) * config.maxNeighboursSize;                                      // cells
  rofsize += (nLayers - 3) * sizeof(int) * config.maxNeighboursSize;                                           // cell neighbours lookup tables
  rofsize += (nLayers - 3) * sizeof(int) * config.maxNeighboursSize;                                           // cell neighbours
  rofsize += sizeof(Road<nLayers - 2>) * config.maxRoadPerRofSize;                                             // roads
  rofsize += (nLayers - 2) * sizeof(int) * config.maxNeighboursSize;                                           // road LUT
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

// template <int nLayers>
// TrackingFrameInfo* GpuTimeFrameChunk<nLayers>::getDeviceTrackingFrameInfo(const int layer)
// {
//   return mTrackingFrameInfoDevice[layer];
// }

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
CellSeed* GpuTimeFrameChunk<nLayers>::getDeviceCells(const int layer)
{
  return mCellsDevice[layer];
}

template <int nLayers>
int* GpuTimeFrameChunk<nLayers>::getDeviceCellsLookupTables(const int layer)
{
  return mCellsLookupTablesDevice[layer];
}

template <int nLayers>
int* GpuTimeFrameChunk<nLayers>::getDeviceCellNeigboursLookupTables(const int layer)
{
  return mNeighboursCellLookupTablesDevice[layer];
}

template <int nLayers>
int* GpuTimeFrameChunk<nLayers>::getDeviceCellNeighbours(const int layer)
{
  return mNeighboursCellDevice[layer];
}

template <int nLayers>
int* GpuTimeFrameChunk<nLayers>::getDeviceRoadsLookupTables(const int layer)
{
  return mRoadsLookupTablesDevice[layer];
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
void TimeFrameGPU<nLayers>::allocMemAsync(void** ptr, size_t size, Stream* strPtr, bool extAllocator)
{
  if (extAllocator) {
    *ptr = mAllocator->allocate(size);
  } else {
    LOGP(debug, "Calling default CUDA allocator");
    checkGPUError(cudaMallocAsync(reinterpret_cast<void**>(ptr), size, strPtr->get()));
  }
}

template <int nLayers>
void TimeFrameGPU<nLayers>::setDevicePropagator(const o2::base::PropagatorImpl<float>* propagator)
{
  mPropagatorDevice = propagator;
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
    checkGPUError(cudaHostRegister(mNClustersPerROF[iLayer].data(), mNClustersPerROF[iLayer].size() * sizeof(int), cudaHostRegisterPortable));
    checkGPUError(cudaHostRegister(mIndexTables[iLayer].data(), (mStaticTrackingParams.ZBins * mStaticTrackingParams.PhiBins + 1) * mNrof * sizeof(int), cudaHostRegisterPortable));
  }
  checkGPUError(cudaHostRegister(mHostNTracklets.data(), (nLayers - 1) * mGpuParams.nTimeFrameChunks * sizeof(int), cudaHostRegisterPortable));
  checkGPUError(cudaHostRegister(mHostNCells.data(), (nLayers - 2) * mGpuParams.nTimeFrameChunks * sizeof(int), cudaHostRegisterPortable));
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
  checkGPUError(cudaHostUnregister(mHostNCells.data()));
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
  mHostNCells.resize((nLayers - 2) * mGpuParams.nTimeFrameChunks, 0);

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
void TimeFrameGPU<nLayers>::initialiseHybrid(const int iteration,
                                             const TrackingParameters& trkParam,
                                             const int maxLayers,
                                             IndexTableUtils* utils,
                                             const TimeFrameGPUParameters* gpuParam)
{
  mGpuStreams.resize(mGpuParams.nTimeFrameChunks);
  o2::its::TimeFrame::initialise(iteration, trkParam, maxLayers);
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
void TimeFrameGPU<nLayers>::loadUnsortedClustersDevice()
{
  for (auto iLayer{0}; iLayer < nLayers; ++iLayer) {
    LOGP(debug, "gpu-transfer: loading {} unsorted clusters on layer {}, for {} MB.", mUnsortedClusters[iLayer].size(), iLayer, mUnsortedClusters[iLayer].size() * sizeof(Cluster) / MB);
    allocMemAsync(reinterpret_cast<void**>(&mUnsortedClustersDevice[iLayer]), mUnsortedClusters[iLayer].size() * sizeof(Cluster), nullptr, getExtAllocator());
    // Register and move data
    checkGPUError(cudaHostRegister(mUnsortedClusters[iLayer].data(), mUnsortedClusters[iLayer].size() * sizeof(Cluster), cudaHostRegisterPortable));
    checkGPUError(cudaMemcpyAsync(mUnsortedClustersDevice[iLayer], mUnsortedClusters[iLayer].data(), mUnsortedClusters[iLayer].size() * sizeof(Cluster), cudaMemcpyHostToDevice, mGpuStreams[0].get()));
  }
  allocMemAsync(reinterpret_cast<void**>(&mUnsortedClustersDeviceArray), nLayers * sizeof(Cluster*), nullptr, getExtAllocator());
  checkGPUError(cudaHostRegister(mUnsortedClustersDevice.data(), nLayers * sizeof(Cluster*), cudaHostRegisterPortable));
  checkGPUError(cudaMemcpyAsync(mUnsortedClustersDeviceArray, mUnsortedClustersDevice.data(), nLayers * sizeof(Cluster*), cudaMemcpyHostToDevice, mGpuStreams[0].get()));
}

template <int nLayers>
void TimeFrameGPU<nLayers>::loadClustersDevice()
{
  for (auto iLayer{0}; iLayer < nLayers; ++iLayer) {
    LOGP(debug, "gpu-transfer: loading {} clusters on layer {}, for {} MB.", mClusters[iLayer].size(), iLayer, mClusters[iLayer].size() * sizeof(Cluster) / MB);
    allocMemAsync(reinterpret_cast<void**>(&mClustersDevice[iLayer]), mClusters[iLayer].size() * sizeof(Cluster), nullptr, getExtAllocator());
    // Register and move data
    checkGPUError(cudaHostRegister(mClusters[iLayer].data(), mClusters[iLayer].size() * sizeof(Cluster), cudaHostRegisterPortable));
    checkGPUError(cudaMemcpyAsync(mClustersDevice[iLayer], mClusters[iLayer].data(), mClusters[iLayer].size() * sizeof(Cluster), cudaMemcpyHostToDevice, mGpuStreams[0].get()));
  }
  allocMemAsync(reinterpret_cast<void**>(&mClustersDeviceArray), nLayers * sizeof(Cluster*), nullptr, getExtAllocator());
  checkGPUError(cudaHostRegister(mClustersDevice.data(), nLayers * sizeof(Cluster*), cudaHostRegisterPortable));
  checkGPUError(cudaMemcpyAsync(mClustersDeviceArray, mClustersDevice.data(), nLayers * sizeof(Cluster*), cudaMemcpyHostToDevice, mGpuStreams[0].get()));
}

template <int nLayers>
void TimeFrameGPU<nLayers>::loadTrackingFrameInfoDevice()
{
  for (auto iLayer{0}; iLayer < nLayers; ++iLayer) {
    LOGP(debug, "gpu-transfer: loading {} tfinfo on layer {}, for {} MB.", mTrackingFrameInfo[iLayer].size(), iLayer, mTrackingFrameInfo[iLayer].size() * sizeof(TrackingFrameInfo) / MB);
    allocMemAsync(reinterpret_cast<void**>(&mTrackingFrameInfoDevice[iLayer]), mTrackingFrameInfo[iLayer].size() * sizeof(TrackingFrameInfo), nullptr, getExtAllocator());
    // Register and move data
    checkGPUError(cudaHostRegister(mTrackingFrameInfo[iLayer].data(), mTrackingFrameInfo[iLayer].size() * sizeof(TrackingFrameInfo), cudaHostRegisterPortable));
    checkGPUError(cudaMemcpyAsync(mTrackingFrameInfoDevice[iLayer], mTrackingFrameInfo[iLayer].data(), mTrackingFrameInfo[iLayer].size() * sizeof(TrackingFrameInfo), cudaMemcpyHostToDevice, mGpuStreams[0].get()));
  }
  allocMemAsync(reinterpret_cast<void**>(&mTrackingFrameInfoDeviceArray), nLayers * sizeof(TrackingFrameInfo*), nullptr, getExtAllocator());
  checkGPUError(cudaHostRegister(mTrackingFrameInfoDevice.data(), nLayers * sizeof(TrackingFrameInfo*), cudaHostRegisterPortable));
  checkGPUError(cudaMemcpyAsync(mTrackingFrameInfoDeviceArray, mTrackingFrameInfoDevice.data(), nLayers * sizeof(TrackingFrameInfo*), cudaMemcpyHostToDevice, mGpuStreams[0].get()));
}

template <int nLayers>
void TimeFrameGPU<nLayers>::loadTrackletsDevice()
{
  for (auto iLayer{0}; iLayer < nLayers - 1; ++iLayer) {
    LOGP(debug, "gpu-transfer: loading {} tracklets on layer {}, for {} MB.", mTracklets[iLayer].size(), iLayer, mTracklets[iLayer].size() * sizeof(Tracklet) / MB);
    allocMemAsync(reinterpret_cast<void**>(&mTrackletsDevice[iLayer]), mTracklets[iLayer].size() * sizeof(Tracklet), nullptr, getExtAllocator());
    // Register and move data
    checkGPUError(cudaHostRegister(mTracklets[iLayer].data(), mTracklets[iLayer].size() * sizeof(Tracklet), cudaHostRegisterPortable));
    checkGPUError(cudaMemcpyAsync(mTrackletsDevice[iLayer], mTracklets[iLayer].data(), mTracklets[iLayer].size() * sizeof(Tracklet), cudaMemcpyHostToDevice, mGpuStreams[0].get()));
  }
  allocMemAsync(reinterpret_cast<void**>(&mTrackletsDeviceArray), (nLayers - 1) * sizeof(Tracklet*), nullptr, getExtAllocator());
  checkGPUError(cudaHostRegister(mTrackletsDevice.data(), (nLayers - 1) * sizeof(Tracklet*), cudaHostRegisterPortable));
  checkGPUError(cudaMemcpyAsync(mTrackletsDeviceArray, mTrackletsDevice.data(), (nLayers - 1) * sizeof(Tracklet*), cudaMemcpyHostToDevice, mGpuStreams[0].get()));
}

template <int nLayers>
void TimeFrameGPU<nLayers>::loadCellsDevice()
{
  for (auto iLayer{0}; iLayer < nLayers - 2; ++iLayer) {
    LOGP(debug, "gpu-transfer: loading {} cell seeds on layer {}, for {} MB.", mCells[iLayer].size(), iLayer, mCells[iLayer].size() * sizeof(CellSeed) / MB);
    allocMemAsync(reinterpret_cast<void**>(&mCellsDevice[iLayer]), mCells[iLayer].size() * sizeof(CellSeed), nullptr, getExtAllocator());
    // Register and move data
    checkGPUError(cudaHostRegister(mCells[iLayer].data(), mCells[iLayer].size() * sizeof(CellSeed), cudaHostRegisterPortable));
    checkGPUError(cudaMemcpyAsync(mCellsDevice[iLayer], mCells[iLayer].data(), mCells[iLayer].size() * sizeof(CellSeed), cudaMemcpyHostToDevice, mGpuStreams[0].get()));
  }
  allocMemAsync(reinterpret_cast<void**>(&mCellsDeviceArray), (nLayers - 2) * sizeof(CellSeed*), nullptr, getExtAllocator());
  checkGPUError(cudaHostRegister(mCellsDevice.data(), (nLayers - 2) * sizeof(CellSeed*), cudaHostRegisterPortable));
  checkGPUError(cudaMemcpyAsync(mCellsDeviceArray, mCellsDevice.data(), (nLayers - 2) * sizeof(CellSeed*), cudaMemcpyHostToDevice, mGpuStreams[0].get()));
}

template <int nLayers>
void TimeFrameGPU<nLayers>::loadRoadsDevice()
{
  LOGP(debug, "gpu-transfer: loading {} roads, for {} MB.", mRoads.size(), mRoads.size() * sizeof(Road<nLayers - 2>) / MB);
  allocMemAsync(reinterpret_cast<void**>(&mRoadsDevice), mRoads.size() * sizeof(Road<nLayers - 2>), &(mGpuStreams[0]), false);
  checkGPUError(cudaHostRegister(mRoads.data(), mRoads.size() * sizeof(Road<nLayers - 2>), cudaHostRegisterPortable));
  checkGPUError(cudaMemcpyAsync(mRoadsDevice, mRoads.data(), mRoads.size() * sizeof(Road<nLayers - 2>), cudaMemcpyHostToDevice, mGpuStreams[0].get()));
}

template <int nLayers>
void TimeFrameGPU<nLayers>::loadTrackSeedsDevice(std::vector<CellSeed>& seeds)
{
  LOGP(debug, "gpu-transfer: loading {} track seeds, for {} MB.", seeds.size(), seeds.size() * sizeof(CellSeed) / MB);
  allocMemAsync(reinterpret_cast<void**>(&mTrackSeedsDevice), seeds.size() * sizeof(CellSeed), &(mGpuStreams[0]), false);
  checkGPUError(cudaHostRegister(seeds.data(), seeds.size() * sizeof(CellSeed), cudaHostRegisterPortable));
  checkGPUError(cudaMemcpyAsync(mTrackSeedsDevice, seeds.data(), seeds.size() * sizeof(CellSeed), cudaMemcpyHostToDevice, mGpuStreams[0].get()));
}

template <int nLayers>
void TimeFrameGPU<nLayers>::createTrackITSExtDevice(const std::vector<CellSeed>& seeds)
{
  mTrackITSExt.clear();
  mTrackITSExt.resize(seeds.size());
  LOGP(debug, "gpu-allocation: reserving {} tracks, for {} MB.", seeds.size(), seeds.size() * sizeof(o2::its::TrackITSExt) / MB);
  allocMemAsync(reinterpret_cast<void**>(&mTrackITSExtDevice), seeds.size() * sizeof(o2::its::TrackITSExt), &(mGpuStreams[0]), false);
  checkGPUError(cudaMemsetAsync(mTrackITSExtDevice, 0, seeds.size() * sizeof(o2::its::TrackITSExt), mGpuStreams[0].get()));
  checkGPUError(cudaHostRegister(mTrackITSExt.data(), seeds.size() * sizeof(o2::its::TrackITSExt), cudaHostRegisterPortable));
}

template <int nLayers>
void TimeFrameGPU<nLayers>::createTrackITSExtDevice()
{
  mTrackITSExt.clear();
  mTrackITSExt.resize(mRoads.size());
  LOGP(debug, "gpu-allocation: reserving {} tracks, for {} MB.", mRoads.size(), mRoads.size() * sizeof(o2::its::TrackITSExt) / MB);
  allocMemAsync(reinterpret_cast<void**>(&mTrackITSExtDevice), mRoads.size() * sizeof(o2::its::TrackITSExt), &(mGpuStreams[0]), false);
  checkGPUError(cudaMemsetAsync(mTrackITSExtDevice, 0, mRoads.size() * sizeof(o2::its::TrackITSExt), mGpuStreams[0].get()));
  checkGPUError(cudaHostRegister(mTrackITSExt.data(), mRoads.size() * sizeof(o2::its::TrackITSExt), cudaHostRegisterPortable));
}

template <int nLayers>
void TimeFrameGPU<nLayers>::downloadTrackITSExtDevice()
{
  LOGP(debug, "gpu-transfer: downloading {} tracks, for {} MB.", mTrackITSExt.size(), mTrackITSExt.size() * sizeof(o2::its::TrackITSExt) / MB);
  checkGPUError(cudaMemcpyAsync(mTrackITSExt.data(), mTrackITSExtDevice, mTrackITSExt.size() * sizeof(o2::its::TrackITSExt), cudaMemcpyDeviceToHost, mGpuStreams[0].get()));
  checkGPUError(cudaHostUnregister(mTrackITSExt.data()));
  discardResult(cudaDeviceSynchronize());
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

template <int nLayers>
gsl::span<int> TimeFrameGPU<nLayers>::getHostNCells(const int chunkId)
{
  return gsl::span<int>(mHostNCells.data() + (nLayers - 2) * chunkId, nLayers - 2);
}

template class TimeFrameGPU<7>;
template class GpuTimeFrameChunk<7>;
} // namespace gpu
} // namespace its
} // namespace o2