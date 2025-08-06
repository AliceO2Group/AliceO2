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
#include <cuda_runtime.h>
#include <thrust/fill.h>
#include <thrust/execution_policy.h>

#include "ITStracking/Constants.h"
#include "ITStracking/BoundedAllocator.h"

#include "ITStrackingGPU/Utils.h"
#include "ITStrackingGPU/TimeFrameGPU.h"
#include "ITStrackingGPU/TracerGPU.h"

#include <unistd.h>
#include <vector>
#include <fmt/format.h>

#include "GPUCommonDef.h"
#include "GPUCommonMath.h"
#include "GPUCommonLogger.h"
#include "GPUCommonHelpers.h"

namespace o2::its::gpu
{

#ifdef ITS_MEASURE_GPU_TIME
class GPUTimer
{
 public:
  GPUTimer(Streams& streams, const std::string& name)
    : mName(name)
  {
    for (size_t i{0}; i < streams.size(); ++i) {
      mStreams.push_back(streams[i].get());
    }
    startTimers();
  }
  GPUTimer(Streams& streams, const std::string& name, size_t end, size_t start = 0)
    : mName(name)
  {
    for (size_t sta{start}; sta < end; ++sta) {
      mStreams.push_back(streams[sta].get());
    }
    startTimers();
  }
  GPUTimer(Stream& stream, const std::string& name)
    : mName(name)
  {
    mStreams.push_back(stream.get());
    startTimers();
  }
  ~GPUTimer()
  {
    for (size_t i{0}; i < mStreams.size(); ++i) {
      GPUChkErrS(cudaEventRecord(mStops[i], mStreams[i]));
      GPUChkErrS(cudaEventSynchronize(mStops[i]));
      float ms = 0.0f;
      GPUChkErrS(cudaEventElapsedTime(&ms, mStarts[i], mStops[i]));
      LOGP(info, "Elapsed time for {}:{} {} ms", mName, i, ms);
      GPUChkErrS(cudaEventDestroy(mStarts[i]));
      GPUChkErrS(cudaEventDestroy(mStops[i]));
    }
  }

  void startTimers()
  {
    mStarts.resize(mStreams.size());
    mStops.resize(mStreams.size());
    for (size_t i{0}; i < mStreams.size(); ++i) {
      GPUChkErrS(cudaEventCreate(&mStarts[i]));
      GPUChkErrS(cudaEventCreate(&mStops[i]));
      GPUChkErrS(cudaEventRecord(mStarts[i], mStreams[i]));
    }
  }

 private:
  std::string mName;
  std::vector<cudaEvent_t> mStarts, mStops;
  std::vector<cudaStream_t> mStreams;
};

#define GPULog(...) LOGP(info, __VA_ARGS__)
#else // ITS_MEASURE_GPU_TIME not defined
class GPUTimer
{
 public:
  template <typename... Args>
  GPUTimer(Args&&...)
  {
  }
};

#define GPULog(...)
#endif

template <int nLayers>
TimeFrameGPU<nLayers>::TimeFrameGPU()
{
  this->mIsGPU = true;
}

template <int nLayers>
TimeFrameGPU<nLayers>::~TimeFrameGPU() = default;

template <int nLayers>
void TimeFrameGPU<nLayers>::allocMemAsync(void** ptr, size_t size, Stream& stream, bool extAllocator)
{
  if (extAllocator) {
    *ptr = this->mAllocator->allocate(size);
  } else {
    GPULog("Calling default CUDA allocator");
    GPUChkErrS(cudaMallocAsync(reinterpret_cast<void**>(ptr), size, stream.get()));
  }
}

template <int nLayers>
void TimeFrameGPU<nLayers>::allocMem(void** ptr, size_t size, bool extAllocator)
{
  if (extAllocator) {
    *ptr = this->mAllocator->allocate(size);
  } else {
    GPULog("Calling default CUDA allocator");
    GPUChkErrS(cudaMalloc(reinterpret_cast<void**>(ptr), size));
  }
}

template <int nLayers>
void TimeFrameGPU<nLayers>::setDevicePropagator(const o2::base::PropagatorImpl<float>* propagator)
{
  this->mPropagatorDevice = propagator;
}

template <int nLayers>
void TimeFrameGPU<nLayers>::loadIndexTableUtils(const int iteration)
{
  GPUTimer timer(mGpuStreams[0], "loading indextable utils");
  if (!iteration) {
    GPULog("gpu-allocation: allocating IndexTableUtils buffer, for {:.2f} MB.", sizeof(IndexTableUtils) / constants::MB);
    allocMem(reinterpret_cast<void**>(&mIndexTableUtilsDevice), sizeof(IndexTableUtils), this->getExtAllocator());
  }
  GPULog("gpu-transfer: loading IndexTableUtils object, for {:.2f} MB.", sizeof(IndexTableUtils) / constants::MB);
  GPUChkErrS(cudaMemcpy(mIndexTableUtilsDevice, &(this->mIndexTableUtils), sizeof(IndexTableUtils), cudaMemcpyHostToDevice));
}

template <int nLayers>
void TimeFrameGPU<nLayers>::loadUnsortedClustersDevice(const int iteration)
{
  if (!iteration) {
    GPUTimer timer(mGpuStreams[0], "loading unsorted clusters");
    for (int iLayer{0}; iLayer < nLayers; ++iLayer) {
      GPULog("gpu-transfer: loading {} unsorted clusters on layer {}, for {:.2f} MB.", this->mUnsortedClusters[iLayer].size(), iLayer, this->mUnsortedClusters[iLayer].size() * sizeof(Cluster) / constants::MB);
      allocMemAsync(reinterpret_cast<void**>(&mUnsortedClustersDevice[iLayer]), this->mUnsortedClusters[iLayer].size() * sizeof(Cluster), mGpuStreams[iLayer], this->getExtAllocator());
      GPUChkErrS(cudaHostRegister(this->mUnsortedClusters[iLayer].data(), this->mUnsortedClusters[iLayer].size() * sizeof(Cluster), cudaHostRegisterPortable));
      GPUChkErrS(cudaMemcpyAsync(mUnsortedClustersDevice[iLayer], this->mUnsortedClusters[iLayer].data(), this->mUnsortedClusters[iLayer].size() * sizeof(Cluster), cudaMemcpyHostToDevice, mGpuStreams[iLayer].get()));
    }
    mGpuStreams.sync();
    allocMem(reinterpret_cast<void**>(&mUnsortedClustersDeviceArray), nLayers * sizeof(Cluster*), this->getExtAllocator());
    GPUChkErrS(cudaHostRegister(mUnsortedClustersDevice.data(), nLayers * sizeof(Cluster*), cudaHostRegisterPortable));
    GPUChkErrS(cudaMemcpy(mUnsortedClustersDeviceArray, mUnsortedClustersDevice.data(), nLayers * sizeof(Cluster*), cudaMemcpyHostToDevice));
  }
}

template <int nLayers>
void TimeFrameGPU<nLayers>::loadClustersDevice(const int iteration)
{
  if (!iteration) {
    GPUTimer timer(mGpuStreams[0], "loading sorted clusters");
    for (auto iLayer{0}; iLayer < nLayers; ++iLayer) {
      GPULog("gpu-transfer: loading {} clusters on layer {}, for {:.2f} MB.", this->mClusters[iLayer].size(), iLayer, this->mClusters[iLayer].size() * sizeof(Cluster) / constants::MB);
      allocMemAsync(reinterpret_cast<void**>(&mClustersDevice[iLayer]), this->mClusters[iLayer].size() * sizeof(Cluster), mGpuStreams[iLayer], this->getExtAllocator());
      GPUChkErrS(cudaHostRegister(this->mClusters[iLayer].data(), this->mClusters[iLayer].size() * sizeof(Cluster), cudaHostRegisterPortable));
      GPUChkErrS(cudaMemcpyAsync(mClustersDevice[iLayer], this->mClusters[iLayer].data(), this->mClusters[iLayer].size() * sizeof(Cluster), cudaMemcpyHostToDevice, mGpuStreams[iLayer].get()));
    }
    mGpuStreams.sync();
    allocMem(reinterpret_cast<void**>(&mClustersDeviceArray), nLayers * sizeof(Cluster*), this->getExtAllocator());
    GPUChkErrS(cudaHostRegister(mClustersDevice.data(), nLayers * sizeof(Cluster*), cudaHostRegisterPortable));
    GPUChkErrS(cudaMemcpy(mClustersDeviceArray, mClustersDevice.data(), nLayers * sizeof(Cluster*), cudaMemcpyHostToDevice));
  }
}

template <int nLayers>
void TimeFrameGPU<nLayers>::loadClustersIndexTables(const int iteration)
{
  if (!iteration) {
    GPUTimer timer(mGpuStreams[0], "loading sorted clusters");
    for (auto iLayer{0}; iLayer < nLayers; ++iLayer) {
      GPULog("gpu-transfer: loading clusters indextable for layer {} with {} elements, for {:.2f} MB.", iLayer, this->mIndexTables[iLayer].size(), this->mIndexTables[iLayer].size() * sizeof(int) / constants::MB);
      allocMemAsync(reinterpret_cast<void**>(&mClustersIndexTablesDevice[iLayer]), this->mIndexTables[iLayer].size() * sizeof(int), mGpuStreams[iLayer], this->getExtAllocator());
      GPUChkErrS(cudaMemcpyAsync(mClustersIndexTablesDevice[iLayer], this->mIndexTables[iLayer].data(), this->mIndexTables[iLayer].size() * sizeof(int), cudaMemcpyHostToDevice, mGpuStreams[iLayer].get()));
    }
    mGpuStreams.sync();
    allocMem(reinterpret_cast<void**>(&mClustersIndexTablesDeviceArray), nLayers * sizeof(int), this->getExtAllocator());
    GPUChkErrS(cudaMemcpyAsync(mClustersIndexTablesDeviceArray, mClustersIndexTablesDevice.data(), nLayers * sizeof(int*), cudaMemcpyHostToDevice));
  }
}

template <int nLayers>
void TimeFrameGPU<nLayers>::createUsedClustersDevice(const int iteration)
{
  if (!iteration) {
    GPUTimer timer(mGpuStreams[0], "creating used clusters flags");
    for (auto iLayer{0}; iLayer < nLayers; ++iLayer) {
      GPULog("gpu-transfer: creating {} used clusters flags on layer {}, for {:.2f} MB.", this->mUsedClusters[iLayer].size(), iLayer, this->mUsedClusters[iLayer].size() * sizeof(unsigned char) / constants::MB);
      allocMemAsync(reinterpret_cast<void**>(&mUsedClustersDevice[iLayer]), this->mUsedClusters[iLayer].size() * sizeof(unsigned char), mGpuStreams[iLayer], this->getExtAllocator());
      GPUChkErrS(cudaMemsetAsync(mUsedClustersDevice[iLayer], 0, this->mUsedClusters[iLayer].size() * sizeof(unsigned char), mGpuStreams[iLayer].get()));
    }
    mGpuStreams.sync();
    allocMem(reinterpret_cast<void**>(&mUsedClustersDeviceArray), nLayers * sizeof(unsigned char*), this->getExtAllocator());
    GPUChkErrS(cudaMemcpyAsync(mUsedClustersDeviceArray, mUsedClustersDevice.data(), nLayers * sizeof(unsigned char*), cudaMemcpyHostToDevice, mGpuStreams[0].get()));
  }
}

template <int nLayers>
void TimeFrameGPU<nLayers>::loadUsedClustersDevice()
{
  GPUTimer timer(mGpuStreams[0], "loading used clusters flags");
  for (auto iLayer{0}; iLayer < nLayers; ++iLayer) {
    GPULog("gpu-transfer: loading {} used clusters flags on layer {}, for {:.2f} MB.", this->mUsedClusters[iLayer].size(), iLayer, this->mClusters[iLayer].size() * sizeof(unsigned char) / constants::MB);
    GPUChkErrS(cudaMemcpyAsync(mUsedClustersDevice[iLayer], this->mUsedClusters[iLayer].data(), this->mUsedClusters[iLayer].size() * sizeof(unsigned char), cudaMemcpyHostToDevice, mGpuStreams[0].get()));
  }
}

template <int nLayers>
void TimeFrameGPU<nLayers>::loadROframeClustersDevice(const int iteration)
{
  if (!iteration) {
    GPUTimer timer(mGpuStreams[0], "loading ROframe clusters");
    for (auto iLayer{0}; iLayer < nLayers; ++iLayer) {
      GPULog("gpu-transfer: loading {} ROframe clusters info on layer {}, for {:.2f} MB.", this->mROFramesClusters[iLayer].size(), iLayer, this->mROFramesClusters[iLayer].size() * sizeof(int) / constants::MB);
      allocMemAsync(reinterpret_cast<void**>(&mROFramesClustersDevice[iLayer]), this->mROFramesClusters[iLayer].size() * sizeof(int), mGpuStreams[iLayer], this->getExtAllocator());
      GPUChkErrS(cudaMemcpyAsync(mROFramesClustersDevice[iLayer], this->mROFramesClusters[iLayer].data(), this->mROFramesClusters[iLayer].size() * sizeof(int), cudaMemcpyHostToDevice, mGpuStreams[0].get()));
    }
    mGpuStreams.sync();
    allocMem(reinterpret_cast<void**>(&mROFrameClustersDeviceArray), nLayers * sizeof(int*), this->getExtAllocator());
    GPUChkErrS(cudaMemcpy(mROFrameClustersDeviceArray, mROFramesClustersDevice.data(), nLayers * sizeof(int*), cudaMemcpyHostToDevice));
  }
}

template <int nLayers>
void TimeFrameGPU<nLayers>::loadTrackingFrameInfoDevice(const int iteration)
{
  GPUTimer timer(mGpuStreams[0], "loading trackingframeinfo");
  if (!iteration) {
    for (auto iLayer{0}; iLayer < nLayers; ++iLayer) {
      GPULog("gpu-transfer: loading {} tfinfo on layer {}, for {:.2f} MB.", this->mTrackingFrameInfo[iLayer].size(), iLayer, this->mTrackingFrameInfo[iLayer].size() * sizeof(TrackingFrameInfo) / constants::MB);
      allocMemAsync(reinterpret_cast<void**>(&mTrackingFrameInfoDevice[iLayer]), this->mTrackingFrameInfo[iLayer].size() * sizeof(TrackingFrameInfo), mGpuStreams[iLayer], this->getExtAllocator());
      GPUChkErrS(cudaHostRegister(this->mTrackingFrameInfo[iLayer].data(), this->mTrackingFrameInfo[iLayer].size() * sizeof(TrackingFrameInfo), cudaHostRegisterPortable));
      GPUChkErrS(cudaMemcpyAsync(mTrackingFrameInfoDevice[iLayer], this->mTrackingFrameInfo[iLayer].data(), this->mTrackingFrameInfo[iLayer].size() * sizeof(TrackingFrameInfo), cudaMemcpyHostToDevice, mGpuStreams[iLayer].get()));
    }
    mGpuStreams.sync();
    allocMemAsync(reinterpret_cast<void**>(&mTrackingFrameInfoDeviceArray), nLayers * sizeof(TrackingFrameInfo*), mGpuStreams[0], this->getExtAllocator());
    GPUChkErrS(cudaHostRegister(mTrackingFrameInfoDevice.data(), nLayers * sizeof(TrackingFrameInfo*), cudaHostRegisterPortable));
    GPUChkErrS(cudaMemcpyAsync(mTrackingFrameInfoDeviceArray, mTrackingFrameInfoDevice.data(), nLayers * sizeof(TrackingFrameInfo*), cudaMemcpyHostToDevice, mGpuStreams[0].get()));
  }
}

template <int nLayers>
void TimeFrameGPU<nLayers>::loadMultiplicityCutMask(const int iteration)
{
  if (!iteration || iteration == 3) { // we need to re-load the swapped mult-mask in upc iteration
    GPUTimer timer(mGpuStreams[0], "loading multiplicity cut mask");
    GPULog("gpu-transfer: iteration {} loading multiplicity cut mask with {} elements, for {:.2f} MB.", iteration, this->mMultiplicityCutMask.size(), this->mMultiplicityCutMask.size() * sizeof(bool) / constants::MB);
    if (!iteration) { // only allocate on first call
      allocMem(reinterpret_cast<void**>(&mMultMaskDevice), this->mMultiplicityCutMask.size() * sizeof(uint8_t), this->getExtAllocator());
    }
    GPUChkErrS(cudaMemcpy(mMultMaskDevice, this->mMultiplicityCutMask.data(), this->mMultiplicityCutMask.size() * sizeof(uint8_t), cudaMemcpyHostToDevice));
  }
}

template <int nLayers>
void TimeFrameGPU<nLayers>::loadVertices(const int iteration)
{
  if (!iteration) {
    GPUTimer timer(mGpuStreams[0], "loading seeding vertices");
    GPULog("gpu-transfer: loading {} ROframes vertices, for {:.2f} MB.", this->mROFramesPV.size(), this->mROFramesPV.size() * sizeof(int) / constants::MB);
    allocMem(reinterpret_cast<void**>(&mROFramesPVDevice), this->mROFramesPV.size() * sizeof(int), this->getExtAllocator());
    GPUChkErrS(cudaMemcpy(mROFramesPVDevice, this->mROFramesPV.data(), this->mROFramesPV.size() * sizeof(int), cudaMemcpyHostToDevice));
    GPULog("gpu-transfer: loading {} seeding vertices, for {:.2f} MB.", this->mPrimaryVertices.size(), this->mPrimaryVertices.size() * sizeof(Vertex) / constants::MB);
    allocMem(reinterpret_cast<void**>(&mPrimaryVerticesDevice), this->mPrimaryVertices.size() * sizeof(Vertex), this->getExtAllocator());
    GPUChkErrS(cudaMemcpy(mPrimaryVerticesDevice, this->mPrimaryVertices.data(), this->mPrimaryVertices.size() * sizeof(Vertex), cudaMemcpyHostToDevice));
  }
}

template <int nLayers>
void TimeFrameGPU<nLayers>::createTrackletsLUTDevice(const int iteration)
{
  GPUTimer timer(mGpuStreams[0], "creating tracklets LUTs");
  for (auto iLayer{0}; iLayer < nLayers - 1; ++iLayer) {
    const int ncls = this->mClusters[iLayer].size() + 1;
    if (!iteration) {
      GPULog("gpu-transfer: creating tracklets LUT for {} elements on layer {}, for {:.2f} MB.", ncls, iLayer, ncls * sizeof(int) / constants::MB);
      allocMemAsync(reinterpret_cast<void**>(&mTrackletsLUTDevice[iLayer]), ncls * sizeof(int), mGpuStreams[iLayer], this->getExtAllocator());
    }
    GPUChkErrS(cudaMemsetAsync(mTrackletsLUTDevice[iLayer], 0, ncls * sizeof(int), mGpuStreams[iLayer].get()));
  }
  if (!iteration) {
    allocMemAsync(reinterpret_cast<void**>(&mTrackletsLUTDeviceArray), (nLayers - 1) * sizeof(int*), mGpuStreams[0], this->getExtAllocator());
    GPUChkErrS(cudaMemcpyAsync(mTrackletsLUTDeviceArray, mTrackletsLUTDevice.data(), mTrackletsLUTDevice.size() * sizeof(int*), cudaMemcpyHostToDevice, mGpuStreams[0].get()));
  }
}

template <int nLayers>
void TimeFrameGPU<nLayers>::createTrackletsBuffers()
{
  for (int iLayer{0}; iLayer < nLayers - 1; ++iLayer) {
    GPUTimer timer(mGpuStreams[iLayer], "creating tracklet buffers");
    mNTracklets[iLayer] = 0;
    GPUChkErrS(cudaMemcpyAsync(&mNTracklets[iLayer], mTrackletsLUTDevice[iLayer] + this->mClusters[iLayer].size(), sizeof(int), cudaMemcpyDeviceToHost, mGpuStreams[iLayer].get()));
    GPULog("gpu-transfer: creating tracklets buffer for {} elements on layer {}, for {:.2f} MB.", mNTracklets[iLayer], iLayer, mNTracklets[iLayer] * sizeof(Tracklet) / constants::MB);
    allocMemAsync(reinterpret_cast<void**>(&mTrackletsDevice[iLayer]), mNTracklets[iLayer] * sizeof(Tracklet), mGpuStreams[iLayer], this->getExtAllocator());
  }
}

template <int nLayers>
void TimeFrameGPU<nLayers>::loadTrackletsDevice()
{
  GPUTimer timer(mGpuStreams, "loading tracklets", nLayers - 1);
  for (auto iLayer{0}; iLayer < nLayers - 1; ++iLayer) {
    GPULog("gpu-transfer: loading {} tracklets on layer {}, for {:.2f} MB.", this->mTracklets[iLayer].size(), iLayer, this->mTracklets[iLayer].size() * sizeof(Tracklet) / constants::MB);
    GPUChkErrS(cudaHostRegister(this->mTracklets[iLayer].data(), this->mTracklets[iLayer].size() * sizeof(Tracklet), cudaHostRegisterPortable));
    GPUChkErrS(cudaMemcpyAsync(mTrackletsDevice[iLayer], this->mTracklets[iLayer].data(), this->mTracklets[iLayer].size() * sizeof(Tracklet), cudaMemcpyHostToDevice, mGpuStreams[iLayer].get()));
  }
}

template <int nLayers>
void TimeFrameGPU<nLayers>::loadTrackletsLUTDevice()
{
  GPUTimer timer(mGpuStreams, "loading tracklets", nLayers - 2);
  for (auto iLayer{0}; iLayer < nLayers - 2; ++iLayer) {
    GPULog("gpu-transfer: loading tracklets LUT for {} elements on layer {}, for {:.2f} MB", this->mTrackletsLookupTable[iLayer].size(), iLayer + 1, this->mTrackletsLookupTable[iLayer].size() * sizeof(int) / constants::MB);
    GPUChkErrS(cudaHostRegister(this->mTrackletsLookupTable[iLayer].data(), this->mTrackletsLookupTable[iLayer].size() * sizeof(int), cudaHostRegisterPortable));
    GPUChkErrS(cudaMemcpyAsync(mTrackletsLUTDevice[iLayer + 1], this->mTrackletsLookupTable[iLayer].data(), this->mTrackletsLookupTable[iLayer].size() * sizeof(int), cudaMemcpyHostToDevice, mGpuStreams[iLayer].get()));
  }
  GPUChkErrS(cudaHostRegister(mTrackletsLUTDevice.data(), (nLayers - 1) * sizeof(int*), cudaHostRegisterPortable));
  GPUChkErrS(cudaMemcpyAsync(mTrackletsLUTDeviceArray, mTrackletsLUTDevice.data(), (nLayers - 1) * sizeof(int*), cudaMemcpyHostToDevice, mGpuStreams[0].get()));
}

template <int nLayers>
void TimeFrameGPU<nLayers>::createNeighboursIndexTablesDevice()
{
  GPUTimer timer(mGpuStreams[0], "creating cells neighbours");
  // Here we do also the creation of the CellsDeviceArray, as the cells buffers are populated separately in the previous steps.
  allocMemAsync(reinterpret_cast<void**>(&mCellsDeviceArray), (nLayers - 2) * sizeof(CellSeed*), mGpuStreams[0], this->getExtAllocator());
  GPUChkErrS(cudaHostRegister(mCellsDevice.data(), (nLayers - 2) * sizeof(CellSeed*), cudaHostRegisterPortable));
  GPUChkErrS(cudaMemcpyAsync(mCellsDeviceArray, mCellsDevice.data(), (nLayers - 2) * sizeof(CellSeed*), cudaMemcpyHostToDevice, mGpuStreams[0].get()));
  for (auto iLayer{0}; iLayer < nLayers - 2; ++iLayer) {
    GPULog("gpu-transfer: loading neighbours LUT for {} elements on layer {}, for {:.2f} MB.", mNCells[iLayer], iLayer, mNCells[iLayer] * sizeof(CellSeed) / constants::MB);
    allocMemAsync(reinterpret_cast<void**>(&mNeighboursIndexTablesDevice[iLayer]), (mNCells[iLayer] + 1) * sizeof(int), mGpuStreams[0], this->getExtAllocator());
    GPUChkErrS(cudaMemsetAsync(mNeighboursIndexTablesDevice[iLayer], 0, (mNCells[iLayer] + 1) * sizeof(int), mGpuStreams[0].get()));
    if (iLayer < nLayers - 3) {
      mNNeighbours[iLayer] = 0;
    }
  }
}

template <int nLayers>
void TimeFrameGPU<nLayers>::createNeighboursLUTDevice(const int layer, const unsigned int nCells)
{
  GPUTimer timer(mGpuStreams[0], "reserving neighboursLUT");
  GPULog("gpu-allocation: reserving neighbours LUT for {} elements on layer {} , for {:.2f} MB.", nCells + 1, layer, (nCells + 1) * sizeof(int) / constants::MB);
  allocMemAsync(reinterpret_cast<void**>(&mNeighboursLUTDevice[layer]), (nCells + 1) * sizeof(int), mGpuStreams[0], this->getExtAllocator()); // We need one element more to move exc -> inc
  GPUChkErrS(cudaMemsetAsync(mNeighboursLUTDevice[layer], 0, (nCells + 1) * sizeof(int), mGpuStreams[0].get()));
}

template <int nLayers>
void TimeFrameGPU<nLayers>::loadCellsDevice()
{
  GPUTimer timer(mGpuStreams, "loading cell seeds", nLayers - 2);
  for (auto iLayer{0}; iLayer < nLayers - 2; ++iLayer) {
    GPULog("gpu-transfer: loading {} cell seeds on layer {}, for {:.2f} MB.", this->mCells[iLayer].size(), iLayer, this->mCells[iLayer].size() * sizeof(CellSeed) / constants::MB);
    allocMemAsync(reinterpret_cast<void**>(&mCellsDevice[iLayer]), this->mCells[iLayer].size() * sizeof(CellSeed), mGpuStreams[iLayer], this->getExtAllocator());
    allocMemAsync(reinterpret_cast<void**>(&mNeighboursIndexTablesDevice[iLayer]), (this->mCells[iLayer].size() + 1) * sizeof(int), mGpuStreams[iLayer], this->getExtAllocator()); // accessory for the neigh. finding.
    GPUChkErrS(cudaMemsetAsync(mNeighboursIndexTablesDevice[iLayer], 0, (this->mCells[iLayer].size() + 1) * sizeof(int), mGpuStreams[iLayer].get()));
    GPUChkErrS(cudaMemcpyAsync(mCellsDevice[iLayer], this->mCells[iLayer].data(), this->mCells[iLayer].size() * sizeof(CellSeed), cudaMemcpyHostToDevice, mGpuStreams[iLayer].get()));
  }
  allocMemAsync(reinterpret_cast<void**>(&mCellsDeviceArray), (nLayers - 2) * sizeof(CellSeed*), mGpuStreams[0], this->getExtAllocator());
  GPUChkErrS(cudaMemcpyAsync(mCellsDeviceArray, mCellsDevice.data(), (nLayers - 2) * sizeof(CellSeed*), cudaMemcpyHostToDevice, mGpuStreams[0].get()));
}

template <int nLayers>
void TimeFrameGPU<nLayers>::createCellsLUTDevice()
{
  GPUTimer timer(mGpuStreams, "creating cells LUTs", nLayers - 2);
  for (auto iLayer{0}; iLayer < nLayers - 2; ++iLayer) {
    GPULog("gpu-transfer: creating cell LUT for {} elements on layer {}, for {:.2f} MB.", mNTracklets[iLayer] + 1, iLayer, (mNTracklets[iLayer] + 1) * sizeof(int) / constants::MB);
    allocMemAsync(reinterpret_cast<void**>(&mCellsLUTDevice[iLayer]), (mNTracklets[iLayer] + 1) * sizeof(int), mGpuStreams[iLayer], this->getExtAllocator());
    GPUChkErrS(cudaMemsetAsync(mCellsLUTDevice[iLayer], 0, (mNTracklets[iLayer] + 1) * sizeof(int), mGpuStreams[iLayer].get()));
  }
  allocMemAsync(reinterpret_cast<void**>(&mCellsLUTDeviceArray), (nLayers - 2) * sizeof(int*), mGpuStreams[0], this->getExtAllocator());
  GPUChkErrS(cudaMemcpyAsync(mCellsLUTDeviceArray, mCellsLUTDevice.data(), mCellsLUTDevice.size() * sizeof(int*), cudaMemcpyHostToDevice, mGpuStreams[0].get()));
}

template <int nLayers>
void TimeFrameGPU<nLayers>::createCellsBuffers(const int layer)
{
  GPUTimer timer(mGpuStreams[0], "creating cells buffers");
  mNCells[layer] = 0;
  GPUChkErrS(cudaMemcpyAsync(&mNCells[layer], mCellsLUTDevice[layer] + mNTracklets[layer], sizeof(int), cudaMemcpyDeviceToHost, mGpuStreams[layer].get()));
  GPULog("gpu-transfer: creating cell buffer for {} elements on layer {}, for {:.2f} MB.", mNCells[layer], layer, mNCells[layer] * sizeof(CellSeed) / constants::MB);
  allocMemAsync(reinterpret_cast<void**>(&mCellsDevice[layer]), mNCells[layer] * sizeof(CellSeed), mGpuStreams[layer], this->getExtAllocator());
}

template <int nLayers>
void TimeFrameGPU<nLayers>::loadCellsLUTDevice()
{
  GPUTimer timer(mGpuStreams, "loading cells LUTs", nLayers - 3);
  for (auto iLayer{0}; iLayer < nLayers - 3; ++iLayer) {
    GPULog("gpu-transfer: loading cell LUT for {} elements on layer {}, for {:.2f} MB.", this->mCellsLookupTable[iLayer].size(), iLayer, this->mCellsLookupTable[iLayer].size() * sizeof(int) / constants::MB);
    GPUChkErrS(cudaHostRegister(this->mCellsLookupTable[iLayer].data(), this->mCellsLookupTable[iLayer].size() * sizeof(int), cudaHostRegisterPortable));
    GPUChkErrS(cudaMemcpyAsync(mCellsLUTDevice[iLayer + 1], this->mCellsLookupTable[iLayer].data(), this->mCellsLookupTable[iLayer].size() * sizeof(int), cudaMemcpyHostToDevice, mGpuStreams[iLayer].get()));
  }
}

template <int nLayers>
void TimeFrameGPU<nLayers>::loadRoadsDevice()
{
  GPUTimer timer(mGpuStreams[0], "loading roads device");
  GPULog("gpu-transfer: loading {} roads, for {:.2f} MB.", this->mRoads.size(), this->mRoads.size() * sizeof(Road<nLayers - 2>) / constants::MB);
  allocMemAsync(reinterpret_cast<void**>(&mRoadsDevice), this->mRoads.size() * sizeof(Road<nLayers - 2>), mGpuStreams[0], this->getExtAllocator());
  GPUChkErrS(cudaHostRegister(this->mRoads.data(), this->mRoads.size() * sizeof(Road<nLayers - 2>), cudaHostRegisterPortable));
  GPUChkErrS(cudaMemcpyAsync(mRoadsDevice, this->mRoads.data(), this->mRoads.size() * sizeof(Road<nLayers - 2>), cudaMemcpyHostToDevice, mGpuStreams[0].get()));
}

template <int nLayers>
void TimeFrameGPU<nLayers>::loadTrackSeedsDevice(bounded_vector<CellSeed>& seeds)
{
  GPUTimer timer(mGpuStreams[0], "loading track seeds");
  GPULog("gpu-transfer: loading {} track seeds, for {:.2f} MB.", seeds.size(), seeds.size() * sizeof(CellSeed) / constants::MB);
  allocMemAsync(reinterpret_cast<void**>(&mTrackSeedsDevice), seeds.size() * sizeof(CellSeed), mGpuStreams[0], this->getExtAllocator());
  GPUChkErrS(cudaHostRegister(seeds.data(), seeds.size() * sizeof(CellSeed), cudaHostRegisterPortable));
  GPUChkErrS(cudaMemcpyAsync(mTrackSeedsDevice, seeds.data(), seeds.size() * sizeof(CellSeed), cudaMemcpyHostToDevice, mGpuStreams[0].get()));
}

template <int nLayers>
void TimeFrameGPU<nLayers>::createNeighboursDevice(const unsigned int layer, const unsigned int nNeighbours)
{
  GPUTimer timer(mGpuStreams[0], "reserving neighbours");
  GPULog("gpu-allocation: reserving {} neighbours (pairs), for {:.2f} MB.", nNeighbours, nNeighbours * sizeof(gpuPair<int, int>) / constants::MB);
  allocMemAsync(reinterpret_cast<void**>(&mNeighbourPairsDevice[layer]), nNeighbours * sizeof(gpuPair<int, int>), mGpuStreams[0], this->getExtAllocator());
  GPUChkErrS(cudaMemsetAsync(mNeighbourPairsDevice[layer], -1, nNeighbours * sizeof(gpuPair<int, int>), mGpuStreams[0].get()));
  GPULog("gpu-allocation: reserving {} neighbours, for {:.2f} MB.", nNeighbours, nNeighbours * sizeof(gpuPair<int, int>) / constants::MB);
  allocMemAsync(reinterpret_cast<void**>(&mNeighboursDevice[layer]), nNeighbours * sizeof(int), mGpuStreams[0], this->getExtAllocator());
}

template <int nLayers>
void TimeFrameGPU<nLayers>::createNeighboursDevice(const unsigned int layer, std::vector<std::pair<int, int>>& neighbours)
{
  GPUTimer timer(mGpuStreams[0], "reserving neighbours");
  this->mCellsNeighbours[layer].clear();
  this->mCellsNeighbours[layer].resize(neighbours.size());
  GPULog("gpu-allocation: reserving {} neighbours (pairs), for {:.2f} MB.", neighbours.size(), neighbours.size() * sizeof(gpuPair<int, int>) / constants::MB);
  allocMemAsync(reinterpret_cast<void**>(&mNeighbourPairsDevice[layer]), neighbours.size() * sizeof(gpuPair<int, int>), mGpuStreams[0], this->getExtAllocator());
  GPUChkErrS(cudaMemsetAsync(mNeighbourPairsDevice[layer], -1, neighbours.size() * sizeof(gpuPair<int, int>), mGpuStreams[0].get()));
  GPULog("gpu-allocation: reserving {} neighbours, for {:.2f} MB.", neighbours.size(), neighbours.size() * sizeof(gpuPair<int, int>) / constants::MB);
  allocMemAsync(reinterpret_cast<void**>(&mNeighboursDevice[layer]), neighbours.size() * sizeof(int), mGpuStreams[0], this->getExtAllocator());
}

template <int nLayers>
void TimeFrameGPU<nLayers>::createNeighboursDeviceArray()
{
  GPUTimer timer(mGpuStreams[0], "reserving neighbours");
  allocMemAsync(reinterpret_cast<void**>(&mNeighboursDeviceArray), (nLayers - 2) * sizeof(int*), mGpuStreams[0], this->getExtAllocator());
  GPUChkErrS(cudaMemcpyAsync(mNeighboursDeviceArray, mNeighboursDevice.data(), (nLayers - 2) * sizeof(int*), cudaMemcpyHostToDevice, mGpuStreams[0].get()));
}

template <int nLayers>
void TimeFrameGPU<nLayers>::createTrackITSExtDevice(bounded_vector<CellSeed>& seeds)
{
  GPUTimer timer(mGpuStreams[0], "reserving tracks");
  mTrackITSExt = bounded_vector<TrackITSExt>(seeds.size(), {}, this->getMemoryPool().get());
  GPULog("gpu-allocation: reserving {} tracks, for {:.2f} MB.", seeds.size(), seeds.size() * sizeof(o2::its::TrackITSExt) / constants::MB);
  allocMemAsync(reinterpret_cast<void**>(&mTrackITSExtDevice), seeds.size() * sizeof(o2::its::TrackITSExt), mGpuStreams[0], this->getExtAllocator());
  GPUChkErrS(cudaMemsetAsync(mTrackITSExtDevice, 0, seeds.size() * sizeof(o2::its::TrackITSExt), mGpuStreams[0].get()));
  GPUChkErrS(cudaHostRegister(mTrackITSExt.data(), seeds.size() * sizeof(o2::its::TrackITSExt), cudaHostRegisterPortable));
}

template <int nLayers>
void TimeFrameGPU<nLayers>::downloadCellsDevice()
{
  GPUTimer timer(mGpuStreams, "downloading cells", nLayers - 2);
  for (int iLayer{0}; iLayer < nLayers - 2; ++iLayer) {
    GPULog("gpu-transfer: downloading {} cells on layer: {}, for {:.2f} MB.", mNCells[iLayer], iLayer, mNCells[iLayer] * sizeof(CellSeed) / constants::MB);
    this->mCells[iLayer].resize(mNCells[iLayer]);
    GPUChkErrS(cudaMemcpyAsync(this->mCells[iLayer].data(), this->mCellsDevice[iLayer], mNCells[iLayer] * sizeof(CellSeed), cudaMemcpyDeviceToHost, mGpuStreams[iLayer].get()));
  }
}

template <int nLayers>
void TimeFrameGPU<nLayers>::downloadCellsLUTDevice()
{
  GPUTimer timer(mGpuStreams, "downloading cell luts", nLayers - 3);
  for (auto iLayer{0}; iLayer < nLayers - 3; ++iLayer) {
    GPULog("gpu-transfer: downloading cells lut on layer {} for {} elements", iLayer, (mNTracklets[iLayer + 1] + 1));
    this->mCellsLookupTable[iLayer].resize(mNTracklets[iLayer + 1] + 1);
    GPUChkErrS(cudaMemcpyAsync(this->mCellsLookupTable[iLayer].data(), mCellsLUTDevice[iLayer + 1], (mNTracklets[iLayer + 1] + 1) * sizeof(int), cudaMemcpyDeviceToHost, mGpuStreams[iLayer].get()));
  }
}

template <int nLayers>
void TimeFrameGPU<nLayers>::downloadCellsNeighboursDevice(std::vector<bounded_vector<std::pair<int, int>>>& neighbours, const int layer)
{
  GPUTimer timer(mGpuStreams[0], fmt::format("downloading neighbours from layer {}", layer));
  GPULog("gpu-transfer: downloading {} neighbours, for {:.2f} MB.", neighbours[layer].size(), neighbours[layer].size() * sizeof(std::pair<int, int>) / constants::MB);
  // TODO: something less dangerous than assuming the same memory layout of std::pair and gpuPair... or not? :)
  GPUChkErrS(cudaMemcpyAsync(neighbours[layer].data(), mNeighbourPairsDevice[layer], neighbours[layer].size() * sizeof(gpuPair<int, int>), cudaMemcpyDeviceToHost, mGpuStreams[0].get()));
}

template <int nLayers>
void TimeFrameGPU<nLayers>::downloadNeighboursLUTDevice(bounded_vector<int>& lut, const int layer)
{
  GPUTimer timer(mGpuStreams[0], fmt::format("downloading neighbours LUT from layer {}", layer));
  GPULog("gpu-transfer: downloading neighbours LUT for {} elements on layer {}, for {:.2f} MB.", lut.size(), layer, lut.size() * sizeof(int) / constants::MB);
  GPUChkErrS(cudaMemcpyAsync(lut.data(), mNeighboursLUTDevice[layer], lut.size() * sizeof(int), cudaMemcpyDeviceToHost, mGpuStreams[0].get()));
}

template <int nLayers>
void TimeFrameGPU<nLayers>::downloadTrackITSExtDevice(bounded_vector<CellSeed>& seeds)
{
  GPUTimer timer(mGpuStreams[0], "downloading tracks");
  GPULog("gpu-transfer: downloading {} tracks, for {:.2f} MB.", mTrackITSExt.size(), mTrackITSExt.size() * sizeof(o2::its::TrackITSExt) / constants::MB);
  GPUChkErrS(cudaMemcpyAsync(mTrackITSExt.data(), mTrackITSExtDevice, seeds.size() * sizeof(o2::its::TrackITSExt), cudaMemcpyDeviceToHost, mGpuStreams[0].get()));
  GPUChkErrS(cudaHostUnregister(mTrackITSExt.data()));
  GPUChkErrS(cudaHostUnregister(seeds.data()));
}

template <int nLayers>
void TimeFrameGPU<nLayers>::unregisterRest()
{
  GPUTimer timer(mGpuStreams[0], "unregistering rest of the host memory");
  GPULog("unregistering rest of the host memory...");
  GPUChkErrS(cudaHostUnregister(mCellsDevice.data()));
  // GPUChkErrS(cudaHostUnregister(mTrackletsDevice.data()));
}

template <int nLayers>
void TimeFrameGPU<nLayers>::unregisterHostMemory(const int maxLayers)
{
  GPUTimer timer(mGpuStreams[0], "unregistering host memory");
  GPULog("unregistering host memory");
  for (auto iLayer{0}; iLayer < nLayers; ++iLayer) {
    GPUChkErrS(cudaHostUnregister(this->mUnsortedClusters[iLayer].data()));
    GPUChkErrS(cudaHostUnregister(this->mClusters[iLayer].data()));
    GPUChkErrS(cudaHostUnregister(this->mTrackingFrameInfo[iLayer].data()));
  }
  GPUChkErrS(cudaHostUnregister(mTrackingFrameInfoDevice.data()));
  GPUChkErrS(cudaHostUnregister(mUnsortedClustersDevice.data()));
  GPUChkErrS(cudaHostUnregister(mClustersDevice.data()));
}

template <int nLayers>
void TimeFrameGPU<nLayers>::initialise(const int iteration,
                                       const TrackingParameters& trkParam,
                                       const int maxLayers,
                                       IndexTableUtils* utils,
                                       const TimeFrameGPUParameters* gpuParam)
{
  mGpuStreams.resize(nLayers);
  o2::its::TimeFrame<nLayers>::initialise(iteration, trkParam, maxLayers);
}

template <int nLayers>
void TimeFrameGPU<nLayers>::syncStreams()
{
  mGpuStreams.sync();
}

template <int nLayers>
void TimeFrameGPU<nLayers>::wipe()
{
  unregisterHostMemory(0);
  o2::its::TimeFrame<nLayers>::wipe();
}

template class TimeFrameGPU<7>;
} // namespace o2::its::gpu
