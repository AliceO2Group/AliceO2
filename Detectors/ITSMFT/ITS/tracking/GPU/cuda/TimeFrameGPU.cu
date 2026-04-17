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

#include <unistd.h>
#include <vector>

#include "ITStrackingGPU/TimeFrameGPU.h"
#include "ITStracking/Constants.h"
#include "ITStracking/BoundedAllocator.h"
#include "ITStrackingGPU/Utils.h"

#include "GPUCommonDef.h"
#include "GPUCommonMath.h"
#include "GPUCommonLogger.h"
#include "GPUCommonHelpers.h"
#include "utils/strtag.h"

namespace o2::its::gpu
{

template <int NLayers>
void TimeFrameGPU<NLayers>::allocMemAsync(void** ptr, size_t size, Stream& stream, bool extAllocator, int32_t type)
{
  if (extAllocator) {
    *ptr = (this->mExternalAllocator)->allocate(size, type);
  } else {
    GPULog("Calling default CUDA allocator");
    GPUChkErrS(cudaMallocAsync(reinterpret_cast<void**>(ptr), size, stream.get()));
  }
}

template <int NLayers>
void TimeFrameGPU<NLayers>::allocMem(void** ptr, size_t size, bool extAllocator, int32_t type)
{
  if (extAllocator) {
    *ptr = (this->mExternalAllocator)->allocate(size, type);
  } else {
    GPULog("Calling default CUDA allocator");
    GPUChkErrS(cudaMalloc(reinterpret_cast<void**>(ptr), size));
  }
}

template <int NLayers>
void TimeFrameGPU<NLayers>::loadIndexTableUtils(const int iteration)
{
  GPUTimer timer("loading indextable utils");
  if (!iteration) {
    GPULog("gpu-allocation: allocating IndexTableUtils buffer, for {:.2f} MB.", sizeof(IndexTableUtilsN) / constants::MB);
    allocMem(reinterpret_cast<void**>(&mIndexTableUtilsDevice), sizeof(IndexTableUtilsN), this->hasFrameworkAllocator());
  }
  GPULog("gpu-transfer: loading IndexTableUtils object, for {:.2f} MB.", sizeof(IndexTableUtilsN) / constants::MB);
  GPUChkErrS(cudaMemcpy(mIndexTableUtilsDevice, &(this->mIndexTableUtils), sizeof(IndexTableUtilsN), cudaMemcpyHostToDevice));
}

template <int NLayers>
void TimeFrameGPU<NLayers>::createUnsortedClustersDeviceArray(const int iteration, const int maxLayers)
{
  if (!iteration) {
    GPUTimer timer("creating unsorted clusters array");
    allocMem(reinterpret_cast<void**>(&mUnsortedClustersDeviceArray), NLayers * sizeof(Cluster*), this->hasFrameworkAllocator());
    GPUChkErrS(cudaHostRegister(mUnsortedClustersDevice.data(), NLayers * sizeof(Cluster*), cudaHostRegisterPortable));
    mPinnedUnsortedClusters.set(NLayers);
    if (!this->hasFrameworkAllocator()) {
      for (auto iLayer{0}; iLayer < o2::gpu::CAMath::Min(maxLayers, NLayers); ++iLayer) {
        GPUChkErrS(cudaHostRegister(this->mUnsortedClusters[iLayer].data(), this->mUnsortedClusters[iLayer].size() * sizeof(Cluster), cudaHostRegisterPortable));
        mPinnedUnsortedClusters.set(iLayer);
      }
    }
  }
}

template <int NLayers>
void TimeFrameGPU<NLayers>::loadUnsortedClustersDevice(const int iteration, const int layer)
{
  if (!iteration) {
    GPUTimer timer(mGpuStreams[layer], "loading unsorted clusters", layer);
    GPULog("gpu-transfer: loading {} unsorted clusters on layer {}, for {:.2f} MB.", this->mUnsortedClusters[layer].size(), layer, this->mUnsortedClusters[layer].size() * sizeof(Cluster) / constants::MB);
    allocMemAsync(reinterpret_cast<void**>(&mUnsortedClustersDevice[layer]), this->mUnsortedClusters[layer].size() * sizeof(Cluster), mGpuStreams[layer], this->hasFrameworkAllocator());
    GPUChkErrS(cudaMemcpyAsync(mUnsortedClustersDevice[layer], this->mUnsortedClusters[layer].data(), this->mUnsortedClusters[layer].size() * sizeof(Cluster), cudaMemcpyHostToDevice, mGpuStreams[layer].get()));
    GPUChkErrS(cudaMemcpyAsync(&mUnsortedClustersDeviceArray[layer], &mUnsortedClustersDevice[layer], sizeof(Cluster*), cudaMemcpyHostToDevice, mGpuStreams[layer].get()));
  }
}

template <int NLayers>
void TimeFrameGPU<NLayers>::createClustersDeviceArray(const int iteration, const int maxLayers)
{
  if (!iteration) {
    GPUTimer timer("creating sorted clusters array");
    allocMem(reinterpret_cast<void**>(&mClustersDeviceArray), NLayers * sizeof(Cluster*), this->hasFrameworkAllocator());
    GPUChkErrS(cudaHostRegister(mClustersDevice.data(), NLayers * sizeof(Cluster*), cudaHostRegisterPortable));
    mPinnedClusters.set(NLayers);
    if (!this->hasFrameworkAllocator()) {
      for (auto iLayer{0}; iLayer < o2::gpu::CAMath::Min(maxLayers, NLayers); ++iLayer) {
        GPUChkErrS(cudaHostRegister(this->mClusters[iLayer].data(), this->mClusters[iLayer].size() * sizeof(Cluster), cudaHostRegisterPortable));
        mPinnedClusters.set(iLayer);
      }
    }
  }
}

template <int NLayers>
void TimeFrameGPU<NLayers>::loadClustersDevice(const int iteration, const int layer)
{
  if (!iteration) {
    GPUTimer timer(mGpuStreams[layer], "loading sorted clusters", layer);
    GPULog("gpu-transfer: loading {} clusters on layer {}, for {:.2f} MB.", this->mClusters[layer].size(), layer, this->mClusters[layer].size() * sizeof(Cluster) / constants::MB);
    allocMemAsync(reinterpret_cast<void**>(&mClustersDevice[layer]), this->mClusters[layer].size() * sizeof(Cluster), mGpuStreams[layer], this->hasFrameworkAllocator());
    GPUChkErrS(cudaMemcpyAsync(mClustersDevice[layer], this->mClusters[layer].data(), this->mClusters[layer].size() * sizeof(Cluster), cudaMemcpyHostToDevice, mGpuStreams[layer].get()));
    GPUChkErrS(cudaMemcpyAsync(&mClustersDeviceArray[layer], &mClustersDevice[layer], sizeof(Cluster*), cudaMemcpyHostToDevice, mGpuStreams[layer].get()));
  }
}

template <int NLayers>
void TimeFrameGPU<NLayers>::createClustersIndexTablesArray(const int iteration)
{
  if (!iteration) {
    GPUTimer timer("creating clustersindextable array");
    allocMem(reinterpret_cast<void**>(&mClustersIndexTablesDeviceArray), NLayers * sizeof(int*), this->hasFrameworkAllocator());
    GPUChkErrS(cudaHostRegister(mClustersIndexTablesDevice.data(), NLayers * sizeof(int*), cudaHostRegisterPortable));
    mPinnedClustersIndexTables.set(NLayers);
    if (!this->hasFrameworkAllocator()) {
      for (auto iLayer{0}; iLayer < NLayers; ++iLayer) {
        GPUChkErrS(cudaHostRegister(this->mIndexTables[iLayer].data(), this->mIndexTables[iLayer].size() * sizeof(int), cudaHostRegisterPortable));
        mPinnedClustersIndexTables.set(iLayer);
      }
    }
  }
}

template <int NLayers>
void TimeFrameGPU<NLayers>::loadClustersIndexTables(const int iteration, const int layer)
{
  if (!iteration) {
    GPUTimer timer(mGpuStreams[layer], "loading sorted clusters", layer);
    GPULog("gpu-transfer: loading clusters indextable for layer {} with {} elements, for {:.2f} MB.", layer, this->mIndexTables[layer].size(), this->mIndexTables[layer].size() * sizeof(int) / constants::MB);
    allocMemAsync(reinterpret_cast<void**>(&mClustersIndexTablesDevice[layer]), this->mIndexTables[layer].size() * sizeof(int), mGpuStreams[layer], this->hasFrameworkAllocator());
    GPUChkErrS(cudaMemcpyAsync(mClustersIndexTablesDevice[layer], this->mIndexTables[layer].data(), this->mIndexTables[layer].size() * sizeof(int), cudaMemcpyHostToDevice, mGpuStreams[layer].get()));
    GPUChkErrS(cudaMemcpyAsync(&mClustersIndexTablesDeviceArray[layer], &mClustersIndexTablesDevice[layer], sizeof(int*), cudaMemcpyHostToDevice, mGpuStreams[layer].get()));
  }
}

template <int NLayers>
void TimeFrameGPU<NLayers>::createUsedClustersDeviceArray(const int iteration, const int maxLayers)
{
  if (!iteration) {
    GPUTimer timer("creating used clusters flags");
    allocMem(reinterpret_cast<void**>(&mUsedClustersDeviceArray), NLayers * sizeof(uint8_t*), this->hasFrameworkAllocator());
    GPUChkErrS(cudaHostRegister(mUsedClustersDevice.data(), NLayers * sizeof(uint8_t*), cudaHostRegisterPortable));
    mPinnedUsedClusters.set(NLayers);
    if (!this->hasFrameworkAllocator()) {
      for (auto iLayer{0}; iLayer < o2::gpu::CAMath::Min(maxLayers, NLayers); ++iLayer) {
        GPUChkErrS(cudaHostRegister(this->mUsedClusters[iLayer].data(), this->mUsedClusters[iLayer].size() * sizeof(uint8_t), cudaHostRegisterPortable));
        mPinnedUsedClusters.set(iLayer);
      }
    }
  }
}

template <int NLayers>
void TimeFrameGPU<NLayers>::createUsedClustersDevice(const int iteration, const int layer)
{
  if (!iteration) {
    GPUTimer timer(mGpuStreams[layer], "creating used clusters flags", layer);
    GPULog("gpu-transfer: creating {} used clusters flags on layer {}, for {:.2f} MB.", this->mUsedClusters[layer].size(), layer, this->mUsedClusters[layer].size() * sizeof(unsigned char) / constants::MB);
    allocMemAsync(reinterpret_cast<void**>(&mUsedClustersDevice[layer]), this->mUsedClusters[layer].size() * sizeof(unsigned char), mGpuStreams[layer], this->hasFrameworkAllocator());
    GPUChkErrS(cudaMemsetAsync(mUsedClustersDevice[layer], 0, this->mUsedClusters[layer].size() * sizeof(unsigned char), mGpuStreams[layer].get()));
    GPUChkErrS(cudaMemcpyAsync(&mUsedClustersDeviceArray[layer], &mUsedClustersDevice[layer], sizeof(unsigned char*), cudaMemcpyHostToDevice, mGpuStreams[layer].get()));
  }
}

template <int NLayers>
void TimeFrameGPU<NLayers>::loadUsedClustersDevice()
{
  for (auto iLayer{0}; iLayer < NLayers; ++iLayer) {
    GPUTimer timer(mGpuStreams[iLayer], "loading used clusters flags", iLayer);
    GPULog("gpu-transfer: loading {} used clusters flags on layer {}, for {:.2f} MB.", this->mUsedClusters[iLayer].size(), iLayer, this->mUsedClusters[iLayer].size() * sizeof(unsigned char) / constants::MB);
    GPUChkErrS(cudaMemcpyAsync(mUsedClustersDevice[iLayer], this->mUsedClusters[iLayer].data(), this->mUsedClusters[iLayer].size() * sizeof(unsigned char), cudaMemcpyHostToDevice, mGpuStreams[iLayer].get()));
  }
}

template <int NLayers>
void TimeFrameGPU<NLayers>::createROFrameClustersDeviceArray(const int iteration)
{
  if (!iteration) {
    GPUTimer timer("creating ROFrame clusters array");
    allocMem(reinterpret_cast<void**>(&mROFramesClustersDeviceArray), NLayers * sizeof(int*), this->hasFrameworkAllocator());
    GPUChkErrS(cudaHostRegister(mROFramesClustersDevice.data(), NLayers * sizeof(int*), cudaHostRegisterPortable));
    mPinnedROFramesClusters.set(NLayers);
    if (!this->hasFrameworkAllocator()) {
      for (auto iLayer{0}; iLayer < NLayers; ++iLayer) {
        GPUChkErrS(cudaHostRegister(this->mROFramesClusters[iLayer].data(), this->mROFramesClusters[iLayer].size() * sizeof(int), cudaHostRegisterPortable));
        mPinnedROFramesClusters.set(iLayer);
      }
    }
  }
}

template <int NLayers>
void TimeFrameGPU<NLayers>::loadROFrameClustersDevice(const int iteration, const int layer)
{
  if (!iteration) {
    GPUTimer timer(mGpuStreams[layer], "loading ROframe clusters", layer);
    GPULog("gpu-transfer: loading {} ROframe clusters info on layer {}, for {:.2f} MB.", this->mROFramesClusters[layer].size(), layer, this->mROFramesClusters[layer].size() * sizeof(int) / constants::MB);
    allocMemAsync(reinterpret_cast<void**>(&mROFramesClustersDevice[layer]), this->mROFramesClusters[layer].size() * sizeof(int), mGpuStreams[layer], this->hasFrameworkAllocator());
    GPUChkErrS(cudaMemcpyAsync(mROFramesClustersDevice[layer], this->mROFramesClusters[layer].data(), this->mROFramesClusters[layer].size() * sizeof(int), cudaMemcpyHostToDevice, mGpuStreams[layer].get()));
    GPUChkErrS(cudaMemcpyAsync(&mROFramesClustersDeviceArray[layer], &mROFramesClustersDevice[layer], sizeof(int*), cudaMemcpyHostToDevice, mGpuStreams[layer].get()));
  }
}

template <int NLayers>
void TimeFrameGPU<NLayers>::createTrackingFrameInfoDeviceArray(const int iteration)
{
  if (!iteration) {
    GPUTimer timer("creating trackingframeinfo array");
    allocMem(reinterpret_cast<void**>(&mTrackingFrameInfoDeviceArray), NLayers * sizeof(TrackingFrameInfo*), this->hasFrameworkAllocator());
    GPUChkErrS(cudaHostRegister(mTrackingFrameInfoDevice.data(), NLayers * sizeof(TrackingFrameInfo*), cudaHostRegisterPortable));
    mPinnedTrackingFrameInfo.set(NLayers);
    if (!this->hasFrameworkAllocator()) {
      for (auto iLayer{0}; iLayer < NLayers; ++iLayer) {
        GPUChkErrS(cudaHostRegister(this->mTrackingFrameInfo[iLayer].data(), this->mTrackingFrameInfo[iLayer].size() * sizeof(TrackingFrameInfo), cudaHostRegisterPortable));
        mPinnedTrackingFrameInfo.set(iLayer);
      }
    }
  }
}

template <int NLayers>
void TimeFrameGPU<NLayers>::loadTrackingFrameInfoDevice(const int iteration, const int layer)
{
  if (!iteration) {
    GPUTimer timer(mGpuStreams[layer], "loading trackingframeinfo", layer);
    GPULog("gpu-transfer: loading {} tfinfo on layer {}, for {:.2f} MB.", this->mTrackingFrameInfo[layer].size(), layer, this->mTrackingFrameInfo[layer].size() * sizeof(TrackingFrameInfo) / constants::MB);
    allocMemAsync(reinterpret_cast<void**>(&mTrackingFrameInfoDevice[layer]), this->mTrackingFrameInfo[layer].size() * sizeof(TrackingFrameInfo), mGpuStreams[layer], this->hasFrameworkAllocator());
    GPUChkErrS(cudaMemcpyAsync(mTrackingFrameInfoDevice[layer], this->mTrackingFrameInfo[layer].data(), this->mTrackingFrameInfo[layer].size() * sizeof(TrackingFrameInfo), cudaMemcpyHostToDevice, mGpuStreams[layer].get()));
    GPUChkErrS(cudaMemcpyAsync(&mTrackingFrameInfoDeviceArray[layer], &mTrackingFrameInfoDevice[layer], sizeof(TrackingFrameInfo*), cudaMemcpyHostToDevice, mGpuStreams[layer].get()));
  }
}

template <int NLayers>
void TimeFrameGPU<NLayers>::loadROFCutMask(const int iteration)
{
  if (!iteration || iteration == 3) { // we need to re-load the swapped mult-mask in upc iteration
    GPUTimer timer("loading multiplicity cut mask");
    const auto& hostTable = *(this->mROFMask);
    const auto hostView = hostTable.getView();
    using TableEntry = ROFMaskTable<NLayers>::TableEntry;
    using TableIndex = ROFMaskTable<NLayers>::TableIndex;
    TableEntry* d_flatTable{nullptr};
    TableIndex* d_indices{nullptr};
    GPULog("gpu-transfer: iteration {} loading multiplicity cut mask with {} elements, for {:.2f} MB.",
           iteration, hostTable.getFlatMaskSize(), hostTable.getFlatMaskSize() * sizeof(TableEntry) / constants::MB);
    allocMem(reinterpret_cast<void**>(&d_flatTable), hostTable.getFlatMaskSize() * sizeof(TableEntry), this->hasFrameworkAllocator());
    allocMem(reinterpret_cast<void**>(&d_indices), NLayers * sizeof(uint32_t), this->hasFrameworkAllocator());
    GPUChkErrS(cudaMemcpy(d_indices, hostView.mLayerROFOffsets, NLayers * sizeof(TableIndex), cudaMemcpyHostToDevice));
    // Re-copy the flat mask on every qualifying iteration (e.g. after swapMasks() for UPC)
    GPUChkErrS(cudaMemcpy(d_flatTable, hostView.mFlatMask, hostTable.getFlatMaskSize() * sizeof(TableEntry), cudaMemcpyHostToDevice));
    mDeviceROFMaskTableView = hostTable.getDeviceView(d_flatTable, d_indices);
  }
}

template <int NLayers>
void TimeFrameGPU<NLayers>::loadVertices(const int iteration)
{
  if (!iteration) {
    GPUTimer timer("loading seeding vertices");
    GPULog("gpu-transfer: loading {} seeding vertices, for {:.2f} MB.", this->mPrimaryVertices.size(), this->mPrimaryVertices.size() * sizeof(Vertex) / constants::MB);
    allocMem(reinterpret_cast<void**>(&mPrimaryVerticesDevice), this->mPrimaryVertices.size() * sizeof(Vertex), this->hasFrameworkAllocator());
    GPUChkErrS(cudaMemcpy(mPrimaryVerticesDevice, this->mPrimaryVertices.data(), this->mPrimaryVertices.size() * sizeof(Vertex), cudaMemcpyHostToDevice));
  }
}

template <int NLayers>
void TimeFrameGPU<NLayers>::loadROFOverlapTable(const int iteration)
{
  if (!iteration) {
    GPUTimer timer("initialising device view of ROFOverlapTable");
    const auto& hostTable = this->getROFOverlapTable();
    const auto& hostView = this->getROFOverlapTableView();
    using TableEntry = ROFOverlapTable<NLayers>::TableEntry;
    using TableIndex = ROFOverlapTable<NLayers>::TableIndex;
    using LayerTiming = o2::its::LayerTiming;
    TableEntry* d_flatTable{nullptr};
    TableIndex* d_indices{nullptr};
    LayerTiming* d_layers{nullptr};
    size_t flatTableSize = hostTable.getFlatTableSize();
    allocMem(reinterpret_cast<void**>(&d_flatTable), flatTableSize * sizeof(TableEntry), this->hasFrameworkAllocator());
    GPUChkErrS(cudaMemcpy(d_flatTable, hostView.mFlatTable, flatTableSize * sizeof(TableEntry), cudaMemcpyHostToDevice));
    allocMem(reinterpret_cast<void**>(&d_indices), hostTable.getIndicesSize() * sizeof(TableIndex), this->hasFrameworkAllocator());
    GPUChkErrS(cudaMemcpy(d_indices, hostView.mIndices, hostTable.getIndicesSize() * sizeof(TableIndex), cudaMemcpyHostToDevice));
    allocMem(reinterpret_cast<void**>(&d_layers), NLayers * sizeof(LayerTiming), this->hasFrameworkAllocator());
    GPUChkErrS(cudaMemcpy(d_layers, hostView.mLayers, NLayers * sizeof(LayerTiming), cudaMemcpyHostToDevice));
    mDeviceROFOverlapTableView = hostTable.getDeviceView(d_flatTable, d_indices, d_layers);
  }
}

template <int NLayers>
void TimeFrameGPU<NLayers>::loadROFVertexLookupTable(const int iteration)
{
  if (!iteration) {
    GPUTimer timer("initialising device view of ROFVertexLookupTable");
    const auto& hostTable = this->getROFVertexLookupTable();
    const auto& hostView = this->getROFVertexLookupTableView();
    using TableEntry = ROFVertexLookupTable<NLayers>::TableEntry;
    using TableIndex = ROFVertexLookupTable<NLayers>::TableIndex;
    using LayerTiming = o2::its::LayerTiming;
    TableEntry* d_flatTable{nullptr};
    TableIndex* d_indices{nullptr};
    LayerTiming* d_layers{nullptr};
    size_t flatTableSize = hostTable.getFlatTableSize();
    allocMem(reinterpret_cast<void**>(&d_flatTable), flatTableSize * sizeof(TableEntry), this->hasFrameworkAllocator());
    GPUChkErrS(cudaMemcpy(d_flatTable, hostView.mFlatTable, flatTableSize * sizeof(TableEntry), cudaMemcpyHostToDevice));
    allocMem(reinterpret_cast<void**>(&d_indices), hostTable.getIndicesSize() * sizeof(TableIndex), this->hasFrameworkAllocator());
    GPUChkErrS(cudaMemcpy(d_indices, hostView.mIndices, hostTable.getIndicesSize() * sizeof(TableIndex), cudaMemcpyHostToDevice));
    allocMem(reinterpret_cast<void**>(&d_layers), NLayers * sizeof(LayerTiming), this->hasFrameworkAllocator());
    GPUChkErrS(cudaMemcpy(d_layers, hostView.mLayers, NLayers * sizeof(LayerTiming), cudaMemcpyHostToDevice));
    mDeviceROFVertexLookupTableView = hostTable.getDeviceView(d_flatTable, d_indices, d_layers);
  }
}

template <int NLayers>
void TimeFrameGPU<NLayers>::updateROFVertexLookupTable(const int iteration)
{
  const auto& hostTable = this->getROFVertexLookupTable();
  if (!iteration) {
    GPUTimer timer("updating device view of ROFVertexLookupTable");
    const auto& hostView = this->getROFVertexLookupTableView();
    using TableEntry = ROFVertexLookupTable<NLayers>::TableEntry;
    TableEntry* d_flatTable{nullptr};
    size_t flatTableSize = hostTable.getFlatTableSize();
    allocMem(reinterpret_cast<void**>(&d_flatTable), flatTableSize * sizeof(TableEntry), this->hasFrameworkAllocator());
    GPUChkErrS(cudaMemcpy(d_flatTable, hostView.mFlatTable, flatTableSize * sizeof(TableEntry), cudaMemcpyHostToDevice));
    mDeviceROFVertexLookupTableView = hostTable.getDeviceView(d_flatTable, hostView.mIndices, hostView.mLayers);
  }
}

template <int NLayers>
void TimeFrameGPU<NLayers>::createTrackletsLUTDeviceArray(const int iteration)
{
  if (!iteration) {
    allocMem(reinterpret_cast<void**>(&mTrackletsLUTDeviceArray), (NLayers - 1) * sizeof(int*), this->hasFrameworkAllocator());
  }
}

template <int NLayers>
void TimeFrameGPU<NLayers>::createTrackletsLUTDevice(const int iteration, const int layer)
{
  GPUTimer timer(mGpuStreams[layer], "creating tracklets LUTs", layer);
  const int ncls = this->mClusters[layer].size() + 1;
  if (!iteration) {
    GPULog("gpu-allocation: creating tracklets LUT for {} elements on layer {}, for {:.2f} MB.", ncls, layer, ncls * sizeof(int) / constants::MB);
    allocMemAsync(reinterpret_cast<void**>(&mTrackletsLUTDevice[layer]), ncls * sizeof(int), mGpuStreams[layer], this->hasFrameworkAllocator());
    GPUChkErrS(cudaMemcpyAsync(&mTrackletsLUTDeviceArray[layer], &mTrackletsLUTDevice[layer], sizeof(int*), cudaMemcpyHostToDevice, mGpuStreams[layer].get()));
  }
  GPUChkErrS(cudaMemsetAsync(mTrackletsLUTDevice[layer], 0, ncls * sizeof(int), mGpuStreams[layer].get()));
}

template <int NLayers>
void TimeFrameGPU<NLayers>::createTrackletsBuffersArray(const int iteration)
{
  if (!iteration) {
    GPUTimer timer("creating tracklet buffers array");
    allocMem(reinterpret_cast<void**>(&mTrackletsDeviceArray), (NLayers - 1) * sizeof(Tracklet*), this->hasFrameworkAllocator());
  }
}

template <int NLayers>
void TimeFrameGPU<NLayers>::createTrackletsBuffers(const int layer)
{
  GPUTimer timer(mGpuStreams[layer], "creating tracklet buffers", layer);
  mNTracklets[layer] = 0;
  GPUChkErrS(cudaMemcpyAsync(&mNTracklets[layer], mTrackletsLUTDevice[layer] + this->mClusters[layer].size(), sizeof(int), cudaMemcpyDeviceToHost, mGpuStreams[layer].get()));
  mGpuStreams[layer].sync(); // ensure number of tracklets is correct
  GPULog("gpu-transfer: creating tracklets buffer for {} elements on layer {}, for {:.2f} MB.", mNTracklets[layer], layer, mNTracklets[layer] * sizeof(Tracklet) / constants::MB);
  allocMemAsync(reinterpret_cast<void**>(&mTrackletsDevice[layer]), mNTracklets[layer] * sizeof(Tracklet), mGpuStreams[layer], this->hasFrameworkAllocator(), (o2::gpu::GPUMemoryResource::MEMORY_GPU | o2::gpu::GPUMemoryResource::MEMORY_STACK));
  GPUChkErrS(cudaMemsetAsync(mTrackletsDevice[layer], 0, mNTracklets[layer] * sizeof(Tracklet), mGpuStreams[layer].get()));
  GPUChkErrS(cudaMemcpyAsync(&mTrackletsDeviceArray[layer], &mTrackletsDevice[layer], sizeof(Tracklet*), cudaMemcpyHostToDevice, mGpuStreams[layer].get()));
}

template <int NLayers>
void TimeFrameGPU<NLayers>::loadTrackletsDevice()
{
  GPUTimer timer(mGpuStreams, "loading tracklets", NLayers - 1);
  for (auto iLayer{0}; iLayer < NLayers - 1; ++iLayer) {
    GPULog("gpu-transfer: loading {} tracklets on layer {}, for {:.2f} MB.", this->mTracklets[iLayer].size(), iLayer, this->mTracklets[iLayer].size() * sizeof(Tracklet) / constants::MB);
    GPUChkErrS(cudaHostRegister(this->mTracklets[iLayer].data(), this->mTracklets[iLayer].size() * sizeof(Tracklet), cudaHostRegisterPortable));
    GPUChkErrS(cudaMemcpyAsync(mTrackletsDevice[iLayer], this->mTracklets[iLayer].data(), this->mTracklets[iLayer].size() * sizeof(Tracklet), cudaMemcpyHostToDevice, mGpuStreams[iLayer].get()));
  }
}

template <int NLayers>
void TimeFrameGPU<NLayers>::loadTrackletsLUTDevice()
{
  GPUTimer timer("loading tracklets");
  for (auto iLayer{0}; iLayer < NLayers - 2; ++iLayer) {
    GPULog("gpu-transfer: loading tracklets LUT for {} elements on layer {}, for {:.2f} MB", this->mTrackletsLookupTable[iLayer].size(), iLayer + 1, this->mTrackletsLookupTable[iLayer].size() * sizeof(int) / constants::MB);
    GPUChkErrS(cudaMemcpyAsync(mTrackletsLUTDevice[iLayer + 1], this->mTrackletsLookupTable[iLayer].data(), this->mTrackletsLookupTable[iLayer].size() * sizeof(int), cudaMemcpyHostToDevice, mGpuStreams[iLayer].get()));
  }
  mGpuStreams.sync();
  GPUChkErrS(cudaMemcpy(mTrackletsLUTDeviceArray, mTrackletsLUTDevice.data(), (NLayers - 1) * sizeof(int*), cudaMemcpyHostToDevice));
}

template <int NLayers>
void TimeFrameGPU<NLayers>::createNeighboursIndexTablesDevice(const int layer)
{
  GPUTimer timer(mGpuStreams[layer], "creating cells neighbours", layer);
  GPULog("gpu-transfer: reserving neighbours LUT for {} elements on layer {}, for {:.2f} MB.", mNCells[layer] + 1, layer, (mNCells[layer] + 1) * sizeof(int) / constants::MB);
  allocMemAsync(reinterpret_cast<void**>(&mNeighboursIndexTablesDevice[layer]), (mNCells[layer] + 1) * sizeof(int), mGpuStreams[layer], this->hasFrameworkAllocator(), (o2::gpu::GPUMemoryResource::MEMORY_GPU | o2::gpu::GPUMemoryResource::MEMORY_STACK));
  GPUChkErrS(cudaMemsetAsync(mNeighboursIndexTablesDevice[layer], 0, (mNCells[layer] + 1) * sizeof(int), mGpuStreams[layer].get()));
}

template <int NLayers>
void TimeFrameGPU<NLayers>::createNeighboursLUTDevice(const int layer, const unsigned int nCells)
{
  GPUTimer timer(mGpuStreams[layer], "reserving neighboursLUT");
  GPULog("gpu-allocation: reserving neighbours LUT for {} elements on layer {} , for {:.2f} MB.", nCells + 1, layer, (nCells + 1) * sizeof(int) / constants::MB);
  allocMemAsync(reinterpret_cast<void**>(&mNeighboursLUTDevice[layer]), (nCells + 1) * sizeof(int), mGpuStreams[layer], this->hasFrameworkAllocator(), (o2::gpu::GPUMemoryResource::MEMORY_GPU | o2::gpu::GPUMemoryResource::MEMORY_STACK)); // We need one element more to move exc -> inc
  GPUChkErrS(cudaMemsetAsync(mNeighboursLUTDevice[layer], 0, (nCells + 1) * sizeof(int), mGpuStreams[layer].get()));
}

template <int NLayers>
void TimeFrameGPU<NLayers>::loadCellsDevice()
{
  GPUTimer timer(mGpuStreams, "loading cell seeds", NLayers - 2);
  for (auto iLayer{0}; iLayer < NLayers - 2; ++iLayer) {
    GPULog("gpu-transfer: loading {} cell seeds on layer {}, for {:.2f} MB.", this->mCells[iLayer].size(), iLayer, this->mCells[iLayer].size() * sizeof(CellSeedN) / constants::MB);
    allocMemAsync(reinterpret_cast<void**>(&mCellsDevice[iLayer]), this->mCells[iLayer].size() * sizeof(CellSeedN), mGpuStreams[iLayer], this->hasFrameworkAllocator());
    allocMemAsync(reinterpret_cast<void**>(&mNeighboursIndexTablesDevice[iLayer]), (this->mCells[iLayer].size() + 1) * sizeof(int), mGpuStreams[iLayer], this->hasFrameworkAllocator()); // accessory for the neigh. finding.
    GPUChkErrS(cudaMemsetAsync(mNeighboursIndexTablesDevice[iLayer], 0, (this->mCells[iLayer].size() + 1) * sizeof(int), mGpuStreams[iLayer].get()));
    GPUChkErrS(cudaMemcpyAsync(mCellsDevice[iLayer], this->mCells[iLayer].data(), this->mCells[iLayer].size() * sizeof(CellSeedN), cudaMemcpyHostToDevice, mGpuStreams[iLayer].get()));
  }
}

template <int NLayers>
void TimeFrameGPU<NLayers>::createCellsLUTDeviceArray(const int iteration)
{
  if (!iteration) {
    GPUTimer timer("creating cells LUTs array");
    allocMem(reinterpret_cast<void**>(&mCellsLUTDeviceArray), (NLayers - 2) * sizeof(int*), this->hasFrameworkAllocator());
  }
}

template <int NLayers>
void TimeFrameGPU<NLayers>::createCellsLUTDevice(const int layer)
{
  GPUTimer timer(mGpuStreams[layer], "creating cells LUTs", layer);
  GPULog("gpu-transfer: creating cell LUT for {} elements on layer {}, for {:.2f} MB.", mNTracklets[layer] + 1, layer, (mNTracklets[layer] + 1) * sizeof(int) / constants::MB);
  allocMemAsync(reinterpret_cast<void**>(&mCellsLUTDevice[layer]), (mNTracklets[layer] + 1) * sizeof(int), mGpuStreams[layer], this->hasFrameworkAllocator(), (o2::gpu::GPUMemoryResource::MEMORY_GPU | o2::gpu::GPUMemoryResource::MEMORY_STACK));
  GPUChkErrS(cudaMemsetAsync(mCellsLUTDevice[layer], 0, (mNTracklets[layer] + 1) * sizeof(int), mGpuStreams[layer].get()));
  GPUChkErrS(cudaMemcpyAsync(&mCellsLUTDeviceArray[layer], &mCellsLUTDevice[layer], sizeof(int*), cudaMemcpyHostToDevice, mGpuStreams[layer].get()));
}

template <int NLayers>
void TimeFrameGPU<NLayers>::createCellsBuffersArray(const int iteration)
{
  if (!iteration) {
    GPUTimer timer("creating cells buffers array");
    allocMem(reinterpret_cast<void**>(&mCellsDeviceArray), (NLayers - 2) * sizeof(CellSeedN*), this->hasFrameworkAllocator());
    GPUChkErrS(cudaMemcpy(mCellsDeviceArray, mCellsDevice.data(), mCellsDevice.size() * sizeof(CellSeedN*), cudaMemcpyHostToDevice));
  }
}

template <int NLayers>
void TimeFrameGPU<NLayers>::createCellsBuffers(const int layer)
{
  GPUTimer timer(mGpuStreams[layer], "creating cells buffers");
  mNCells[layer] = 0;
  GPUChkErrS(cudaMemcpyAsync(&mNCells[layer], mCellsLUTDevice[layer] + mNTracklets[layer], sizeof(int), cudaMemcpyDeviceToHost, mGpuStreams[layer].get()));
  mGpuStreams[layer].sync(); // ensure number of cells is correct
  GPULog("gpu-transfer: creating cell buffer for {} elements on layer {}, for {:.2f} MB.", mNCells[layer], layer, mNCells[layer] * sizeof(CellSeedN) / constants::MB);
  allocMemAsync(reinterpret_cast<void**>(&mCellsDevice[layer]), mNCells[layer] * sizeof(CellSeedN), mGpuStreams[layer], this->hasFrameworkAllocator(), (o2::gpu::GPUMemoryResource::MEMORY_GPU | o2::gpu::GPUMemoryResource::MEMORY_STACK));
  GPUChkErrS(cudaMemsetAsync(mCellsDevice[layer], 0, mNCells[layer] * sizeof(CellSeedN), mGpuStreams[layer].get()));
  GPUChkErrS(cudaMemcpyAsync(&mCellsDeviceArray[layer], &mCellsDevice[layer], sizeof(CellSeedN*), cudaMemcpyHostToDevice, mGpuStreams[layer].get()));
}

template <int NLayers>
void TimeFrameGPU<NLayers>::loadCellsLUTDevice()
{
  GPUTimer timer(mGpuStreams, "loading cells LUTs", NLayers - 3);
  for (auto iLayer{0}; iLayer < NLayers - 3; ++iLayer) {
    GPULog("gpu-transfer: loading cell LUT for {} elements on layer {}, for {:.2f} MB.", this->mCellsLookupTable[iLayer].size(), iLayer, this->mCellsLookupTable[iLayer].size() * sizeof(int) / constants::MB);
    GPUChkErrS(cudaHostRegister(this->mCellsLookupTable[iLayer].data(), this->mCellsLookupTable[iLayer].size() * sizeof(int), cudaHostRegisterPortable));
    GPUChkErrS(cudaMemcpyAsync(mCellsLUTDevice[iLayer + 1], this->mCellsLookupTable[iLayer].data(), this->mCellsLookupTable[iLayer].size() * sizeof(int), cudaMemcpyHostToDevice, mGpuStreams[iLayer].get()));
  }
}

template <int NLayers>
void TimeFrameGPU<NLayers>::loadTrackSeedsDevice(bounded_vector<CellSeedN>& seeds)
{
  GPUTimer timer("loading track seeds");
  GPULog("gpu-transfer: loading {} track seeds, for {:.2f} MB.", seeds.size(), seeds.size() * sizeof(CellSeedN) / constants::MB);
  allocMem(reinterpret_cast<void**>(&mTrackSeedsDevice), seeds.size() * sizeof(CellSeedN), this->hasFrameworkAllocator(), (o2::gpu::GPUMemoryResource::MEMORY_GPU | o2::gpu::GPUMemoryResource::MEMORY_STACK));
  GPUChkErrS(cudaMemcpy(mTrackSeedsDevice, seeds.data(), seeds.size() * sizeof(CellSeedN), cudaMemcpyHostToDevice));
  GPULog("gpu-transfer: creating {} track seeds LUT, for {:.2f} MB.", seeds.size() + 1, (seeds.size() + 1) * sizeof(int) / constants::MB);
  allocMem(reinterpret_cast<void**>(&mTrackSeedsLUTDevice), (seeds.size() + 1) * sizeof(int), this->hasFrameworkAllocator(), (o2::gpu::GPUMemoryResource::MEMORY_GPU | o2::gpu::GPUMemoryResource::MEMORY_STACK));
  GPUChkErrS(cudaMemset(mTrackSeedsLUTDevice, 0, (seeds.size() + 1) * sizeof(int)));
}

template <int NLayers>
void TimeFrameGPU<NLayers>::createNeighboursDevice(const unsigned int layer)
{
  GPUTimer timer(mGpuStreams[layer], "reserving neighbours", layer);
  this->mNNeighbours[layer] = 0;
  GPUChkErrS(cudaMemcpyAsync(&(this->mNNeighbours[layer]), &(mNeighboursLUTDevice[layer][this->mNCells[layer + 1] - 1]), sizeof(unsigned int), cudaMemcpyDeviceToHost, mGpuStreams[layer].get()));
  mGpuStreams[layer].sync(); // ensure number of neighbours is correct
  GPULog("gpu-allocation: reserving {} neighbours (pairs), for {:.2f} MB.", this->mNNeighbours[layer], (this->mNNeighbours[layer]) * sizeof(gpuPair<int, int>) / constants::MB);
  allocMemAsync(reinterpret_cast<void**>(&mNeighbourPairsDevice[layer]), (this->mNNeighbours[layer]) * sizeof(gpuPair<int, int>), mGpuStreams[layer], this->hasFrameworkAllocator(), (o2::gpu::GPUMemoryResource::MEMORY_GPU | o2::gpu::GPUMemoryResource::MEMORY_STACK));
  GPUChkErrS(cudaMemsetAsync(mNeighbourPairsDevice[layer], -1, (this->mNNeighbours[layer]) * sizeof(gpuPair<int, int>), mGpuStreams[layer].get()));
  GPULog("gpu-allocation: reserving {} neighbours, for {:.2f} MB.", this->mNNeighbours[layer], (this->mNNeighbours[layer]) * sizeof(gpuPair<int, int>) / constants::MB);
  allocMemAsync(reinterpret_cast<void**>(&mNeighboursDevice[layer]), (this->mNNeighbours[layer]) * sizeof(int), mGpuStreams[layer], this->hasFrameworkAllocator(), (o2::gpu::GPUMemoryResource::MEMORY_GPU | o2::gpu::GPUMemoryResource::MEMORY_STACK));
}

template <int NLayers>
void TimeFrameGPU<NLayers>::createTrackITSExtDevice(const size_t nSeeds)
{
  GPUTimer timer("reserving tracks");
  mNTracks = 0;
  GPUChkErrS(cudaMemcpy(&mNTracks, mTrackSeedsLUTDevice + nSeeds, sizeof(int), cudaMemcpyDeviceToHost));
  GPULog("gpu-allocation: reserving {} tracks, for {:.2f} MB.", mNTracks, mNTracks * sizeof(o2::its::TrackITSExt) / constants::MB);
  mTrackITSExt = bounded_vector<TrackITSExt>(mNTracks, {}, this->getMemoryPool().get());
  allocMem(reinterpret_cast<void**>(&mTrackITSExtDevice), mNTracks * sizeof(o2::its::TrackITSExt), this->hasFrameworkAllocator(), (o2::gpu::GPUMemoryResource::MEMORY_GPU | o2::gpu::GPUMemoryResource::MEMORY_STACK));
  GPUChkErrS(cudaMemset(mTrackITSExtDevice, 0, mNTracks * sizeof(o2::its::TrackITSExt)));
}

template <int NLayers>
void TimeFrameGPU<NLayers>::downloadCellsDevice()
{
  GPUTimer timer(mGpuStreams, "downloading cells", NLayers - 2);
  for (int iLayer{0}; iLayer < NLayers - 2; ++iLayer) {
    GPULog("gpu-transfer: downloading {} cells on layer: {}, for {:.2f} MB.", mNCells[iLayer], iLayer, mNCells[iLayer] * sizeof(CellSeedN) / constants::MB);
    this->mCells[iLayer].resize(mNCells[iLayer]);
    GPUChkErrS(cudaMemcpyAsync(this->mCells[iLayer].data(), this->mCellsDevice[iLayer], mNCells[iLayer] * sizeof(CellSeedN), cudaMemcpyDeviceToHost, mGpuStreams[iLayer].get()));
  }
}

template <int NLayers>
void TimeFrameGPU<NLayers>::downloadCellsLUTDevice()
{
  GPUTimer timer(mGpuStreams, "downloading cell luts", NLayers - 3);
  for (auto iLayer{0}; iLayer < NLayers - 3; ++iLayer) {
    GPULog("gpu-transfer: downloading cells lut on layer {} for {} elements", iLayer, (mNTracklets[iLayer + 1] + 1));
    this->mCellsLookupTable[iLayer].resize(mNTracklets[iLayer + 1] + 1);
    GPUChkErrS(cudaMemcpyAsync(this->mCellsLookupTable[iLayer].data(), mCellsLUTDevice[iLayer + 1], (mNTracklets[iLayer + 1] + 1) * sizeof(int), cudaMemcpyDeviceToHost, mGpuStreams[iLayer].get()));
  }
}

template <int NLayers>
void TimeFrameGPU<NLayers>::downloadCellsNeighboursDevice(std::vector<bounded_vector<std::pair<int, int>>>& neighbours, const int layer)
{
  GPUTimer timer(mGpuStreams[layer], "downloading neighbours from layer", layer);
  GPULog("gpu-transfer: downloading {} neighbours, for {:.2f} MB.", neighbours[layer].size(), neighbours[layer].size() * sizeof(std::pair<int, int>) / constants::MB);
  GPUChkErrS(cudaMemcpyAsync(neighbours[layer].data(), mNeighbourPairsDevice[layer], neighbours[layer].size() * sizeof(gpuPair<int, int>), cudaMemcpyDeviceToHost, mGpuStreams[layer].get()));
}

template <int NLayers>
void TimeFrameGPU<NLayers>::downloadNeighboursLUTDevice(bounded_vector<int>& lut, const int layer)
{
  GPUTimer timer(mGpuStreams[layer], "downloading neighbours LUT from layer", layer);
  GPULog("gpu-transfer: downloading neighbours LUT for {} elements on layer {}, for {:.2f} MB.", lut.size(), layer, lut.size() * sizeof(int) / constants::MB);
  GPUChkErrS(cudaMemcpyAsync(lut.data(), mNeighboursLUTDevice[layer], lut.size() * sizeof(int), cudaMemcpyDeviceToHost, mGpuStreams[layer].get()));
}

template <int NLayers>
void TimeFrameGPU<NLayers>::downloadTrackITSExtDevice()
{
  GPUTimer timer("downloading tracks");
  GPULog("gpu-transfer: downloading {} tracks, for {:.2f} MB.", mTrackITSExt.size(), mTrackITSExt.size() * sizeof(o2::its::TrackITSExt) / constants::MB);
  GPUChkErrS(cudaMemcpy(mTrackITSExt.data(), mTrackITSExtDevice, mTrackITSExt.size() * sizeof(o2::its::TrackITSExt), cudaMemcpyDeviceToHost));
}

template <int NLayers>
void TimeFrameGPU<NLayers>::unregisterHostMemory(const int maxLayers)
{
  GPUTimer timer("unregistering host memory");
  GPULog("unregistering host memory");

  auto checkedUnregisterEntry = [](auto& bits, auto& vec, int layer) {
    if (bits.test(layer)) {
      GPUChkErrS(cudaHostUnregister(vec[layer].data()));
      bits.reset(layer);
    }
  };
  auto checkedUnregisterArray = [](auto& bits, auto& vec) {
    if (bits.test(NLayers)) {
      GPUChkErrS(cudaHostUnregister(vec.data()));
      bits.reset(NLayers);
    }
  };

  for (auto iLayer{0}; iLayer < NLayers; ++iLayer) {
    checkedUnregisterEntry(mPinnedUsedClusters, this->mUsedClusters, iLayer);
    checkedUnregisterEntry(mPinnedUnsortedClusters, this->mUnsortedClusters, iLayer);
    checkedUnregisterEntry(mPinnedClusters, this->mClusters, iLayer);
    checkedUnregisterEntry(mPinnedClustersIndexTables, this->mIndexTables, iLayer);
    checkedUnregisterEntry(mPinnedTrackingFrameInfo, this->mTrackingFrameInfo, iLayer);
    checkedUnregisterEntry(mPinnedROFramesClusters, this->mROFramesClusters, iLayer);
  }
  checkedUnregisterArray(mPinnedUsedClusters, mUsedClustersDevice);
  checkedUnregisterArray(mPinnedUnsortedClusters, mUnsortedClustersDevice);
  checkedUnregisterArray(mPinnedClusters, mClustersDevice);
  checkedUnregisterArray(mPinnedClustersIndexTables, mClustersIndexTablesDevice);
  checkedUnregisterArray(mPinnedTrackingFrameInfo, mTrackingFrameInfoDevice);
  checkedUnregisterArray(mPinnedROFramesClusters, mROFramesClustersDevice);
}

namespace detail
{
template <std::size_t I>
constexpr uint64_t makeIterTag()
{
  static_assert(I < 10);
  constexpr char tag[] = {'I', 'T', 'S', 'I', 'T', 'E', 'R', char('0' + I), '\0'};
  return qStr2Tag(tag);
}
template <std::size_t... I>
constexpr auto makeIterTags(std::index_sequence<I...>)
{
  return std::array<uint64_t, sizeof...(I)>{makeIterTag<I>()...};
}
constexpr auto kIterTags = makeIterTags(std::make_index_sequence<constants::MaxIter>{});
} // namespace detail

template <int NLayers>
void TimeFrameGPU<NLayers>::pushMemoryStack(const int iteration)
{
  // mark the beginning of memory marked with MEMORY_STACK that can be discarded
  // after doing one iteration
  (this->mExternalAllocator)->pushTagOnStack(detail::kIterTags[iteration]);
}

template <int NLayers>
void TimeFrameGPU<NLayers>::popMemoryStack(const int iteration)
{
  // pop all memory on the stack from this iteration
  (this->mExternalAllocator)->popTagOffStack(detail::kIterTags[iteration]);
}

template <int NLayers>
void TimeFrameGPU<NLayers>::initialise(const int iteration,
                                       const TrackingParameters& trkParam,
                                       const int maxLayers)
{
  mGpuStreams.resize(NLayers);
  o2::its::TimeFrame<NLayers>::initialise(iteration, trkParam, maxLayers, false);
}

template <int NLayers>
void TimeFrameGPU<NLayers>::syncStream(const size_t stream)
{
  mGpuStreams[stream].sync();
}

template <int NLayers>
void TimeFrameGPU<NLayers>::syncStreams(const bool device)
{
  mGpuStreams.sync(device);
}

template <int NLayers>
void TimeFrameGPU<NLayers>::waitEvent(const int stream, const int event)
{
  mGpuStreams.waitEvent(stream, event);
}

template <int NLayers>
void TimeFrameGPU<NLayers>::recordEvent(const int event)
{
  mGpuStreams[event].record();
}

template <int NLayers>
void TimeFrameGPU<NLayers>::recordEvents(const int start, const int end)
{
  for (int i{start}; i < end; ++i) {
    recordEvent(i);
  }
}

template <int NLayers>
void TimeFrameGPU<NLayers>::wipe()
{
  unregisterHostMemory(0);
  o2::its::TimeFrame<NLayers>::wipe();
}

template class TimeFrameGPU<7>;
} // namespace o2::its::gpu
