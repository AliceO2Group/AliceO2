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

namespace o2::its::gpu
{

template <int nLayers>
TimeFrameGPU<nLayers>::TimeFrameGPU()
{
  this->mIsGPU = true;
}

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
void TimeFrameGPU<nLayers>::loadIndexTableUtils(const int iteration)
{
  GPUTimer timer("loading indextable utils");
  if (!iteration) {
    GPULog("gpu-allocation: allocating IndexTableUtils buffer, for {:.2f} MB.", sizeof(IndexTableUtilsN) / constants::MB);
    allocMem(reinterpret_cast<void**>(&mIndexTableUtilsDevice), sizeof(IndexTableUtilsN), this->getExtAllocator());
  }
  GPULog("gpu-transfer: loading IndexTableUtils object, for {:.2f} MB.", sizeof(IndexTableUtilsN) / constants::MB);
  GPUChkErrS(cudaMemcpy(mIndexTableUtilsDevice, &(this->mIndexTableUtils), sizeof(IndexTableUtilsN), cudaMemcpyHostToDevice));
}

template <int nLayers>
void TimeFrameGPU<nLayers>::createUnsortedClustersDeviceArray(const int iteration)
{
  if (!iteration) {
    GPUTimer timer("creating unsorted clusters array");
    allocMem(reinterpret_cast<void**>(&mUnsortedClustersDeviceArray), nLayers * sizeof(Cluster*), this->getExtAllocator());
    GPUChkErrS(cudaHostRegister(mUnsortedClustersDevice.data(), nLayers * sizeof(Cluster*), cudaHostRegisterPortable));
    mPinnedUnsortedClusters.set(nLayers);
    for (auto iLayer{0}; iLayer < nLayers; ++iLayer) {
      GPUChkErrS(cudaHostRegister(this->mUnsortedClusters[iLayer].data(), this->mUnsortedClusters[iLayer].size() * sizeof(Cluster), cudaHostRegisterPortable));
      mPinnedUnsortedClusters.set(iLayer);
    }
  }
}

template <int nLayers>
void TimeFrameGPU<nLayers>::loadUnsortedClustersDevice(const int iteration, const int layer)
{
  if (!iteration) {
    GPUTimer timer(mGpuStreams[layer], "loading unsorted clusters", layer);
    GPULog("gpu-transfer: loading {} unsorted clusters on layer {}, for {:.2f} MB.", this->mUnsortedClusters[layer].size(), layer, this->mUnsortedClusters[layer].size() * sizeof(Cluster) / constants::MB);
    allocMemAsync(reinterpret_cast<void**>(&mUnsortedClustersDevice[layer]), this->mUnsortedClusters[layer].size() * sizeof(Cluster), mGpuStreams[layer], this->getExtAllocator());
    GPUChkErrS(cudaMemcpyAsync(mUnsortedClustersDevice[layer], this->mUnsortedClusters[layer].data(), this->mUnsortedClusters[layer].size() * sizeof(Cluster), cudaMemcpyHostToDevice, mGpuStreams[layer].get()));
    GPUChkErrS(cudaMemcpyAsync(&mUnsortedClustersDeviceArray[layer], &mUnsortedClustersDevice[layer], sizeof(Cluster*), cudaMemcpyHostToDevice, mGpuStreams[layer].get()));
  }
}

template <int nLayers>
void TimeFrameGPU<nLayers>::createClustersDeviceArray(const int iteration)
{
  if (!iteration) {
    GPUTimer timer("creating sorted clusters array");
    allocMem(reinterpret_cast<void**>(&mClustersDeviceArray), nLayers * sizeof(Cluster*), this->getExtAllocator());
    GPUChkErrS(cudaHostRegister(mClustersDevice.data(), nLayers * sizeof(Cluster*), cudaHostRegisterPortable));
    mPinnedClusters.set(nLayers);
    for (auto iLayer{0}; iLayer < nLayers; ++iLayer) {
      GPUChkErrS(cudaHostRegister(this->mClusters[iLayer].data(), this->mClusters[iLayer].size() * sizeof(Cluster), cudaHostRegisterPortable));
      mPinnedClusters.set(iLayer);
    }
  }
}

template <int nLayers>
void TimeFrameGPU<nLayers>::loadClustersDevice(const int iteration, const int layer)
{
  if (!iteration) {
    GPUTimer timer(mGpuStreams[layer], "loading sorted clusters", layer);
    GPULog("gpu-transfer: loading {} clusters on layer {}, for {:.2f} MB.", this->mClusters[layer].size(), layer, this->mClusters[layer].size() * sizeof(Cluster) / constants::MB);
    allocMemAsync(reinterpret_cast<void**>(&mClustersDevice[layer]), this->mClusters[layer].size() * sizeof(Cluster), mGpuStreams[layer], this->getExtAllocator());
    GPUChkErrS(cudaMemcpyAsync(mClustersDevice[layer], this->mClusters[layer].data(), this->mClusters[layer].size() * sizeof(Cluster), cudaMemcpyHostToDevice, mGpuStreams[layer].get()));
    GPUChkErrS(cudaMemcpyAsync(&mClustersDeviceArray[layer], &mClustersDevice[layer], sizeof(Cluster*), cudaMemcpyHostToDevice, mGpuStreams[layer].get()));
  }
}

template <int nLayers>
void TimeFrameGPU<nLayers>::createClustersIndexTablesArray(const int iteration)
{
  if (!iteration) {
    GPUTimer timer("creating clustersindextable array");
    allocMem(reinterpret_cast<void**>(&mClustersIndexTablesDeviceArray), nLayers * sizeof(int*), this->getExtAllocator());
    GPUChkErrS(cudaHostRegister(mClustersIndexTablesDevice.data(), nLayers * sizeof(int*), cudaHostRegisterPortable));
    mPinnedClustersIndexTables.set(nLayers);
    for (auto iLayer{0}; iLayer < nLayers; ++iLayer) {
      GPUChkErrS(cudaHostRegister(this->mIndexTables[iLayer].data(), this->mIndexTables[iLayer].size() * sizeof(int), cudaHostRegisterPortable));
      mPinnedClustersIndexTables.set(iLayer);
    }
  }
}

template <int nLayers>
void TimeFrameGPU<nLayers>::loadClustersIndexTables(const int iteration, const int layer)
{
  if (!iteration) {
    GPUTimer timer(mGpuStreams[layer], "loading sorted clusters", layer);
    GPULog("gpu-transfer: loading clusters indextable for layer {} with {} elements, for {:.2f} MB.", layer, this->mIndexTables[layer].size(), this->mIndexTables[layer].size() * sizeof(int) / constants::MB);
    allocMemAsync(reinterpret_cast<void**>(&mClustersIndexTablesDevice[layer]), this->mIndexTables[layer].size() * sizeof(int), mGpuStreams[layer], this->getExtAllocator());
    GPUChkErrS(cudaMemcpyAsync(mClustersIndexTablesDevice[layer], this->mIndexTables[layer].data(), this->mIndexTables[layer].size() * sizeof(int), cudaMemcpyHostToDevice, mGpuStreams[layer].get()));
    GPUChkErrS(cudaMemcpyAsync(&mClustersIndexTablesDeviceArray[layer], &mClustersIndexTablesDevice[layer], sizeof(int*), cudaMemcpyHostToDevice, mGpuStreams[layer].get()));
  }
}

template <int nLayers>
void TimeFrameGPU<nLayers>::createUsedClustersDeviceArray(const int iteration)
{
  if (!iteration) {
    GPUTimer timer("creating used clusters flags");
    allocMem(reinterpret_cast<void**>(&mUsedClustersDeviceArray), nLayers * sizeof(unsigned char*), this->getExtAllocator());
    GPUChkErrS(cudaHostRegister(mUsedClustersDevice.data(), nLayers * sizeof(unsigned char*), cudaHostRegisterPortable));
    mPinnedUsedClusters.set(nLayers);
    for (auto iLayer{0}; iLayer < nLayers; ++iLayer) {
      GPUChkErrS(cudaHostRegister(this->mUsedClusters[iLayer].data(), this->mUsedClusters[iLayer].size() * sizeof(unsigned char), cudaHostRegisterPortable));
      mPinnedUsedClusters.set(iLayer);
    }
  }
}

template <int nLayers>
void TimeFrameGPU<nLayers>::createUsedClustersDevice(const int iteration, const int layer)
{
  if (!iteration) {
    GPUTimer timer(mGpuStreams[layer], "creating used clusters flags", layer);
    GPULog("gpu-transfer: creating {} used clusters flags on layer {}, for {:.2f} MB.", this->mUsedClusters[layer].size(), layer, this->mUsedClusters[layer].size() * sizeof(unsigned char) / constants::MB);
    allocMemAsync(reinterpret_cast<void**>(&mUsedClustersDevice[layer]), this->mUsedClusters[layer].size() * sizeof(unsigned char), mGpuStreams[layer], this->getExtAllocator());
    GPUChkErrS(cudaMemsetAsync(mUsedClustersDevice[layer], 0, this->mUsedClusters[layer].size() * sizeof(unsigned char), mGpuStreams[layer].get()));
    GPUChkErrS(cudaMemcpyAsync(&mUsedClustersDeviceArray[layer], &mUsedClustersDevice[layer], sizeof(unsigned char*), cudaMemcpyHostToDevice, mGpuStreams[layer].get()));
  }
}

template <int nLayers>
void TimeFrameGPU<nLayers>::loadUsedClustersDevice()
{
  for (auto iLayer{0}; iLayer < nLayers; ++iLayer) {
    GPUTimer timer(mGpuStreams[iLayer], "loading used clusters flags", iLayer);
    GPULog("gpu-transfer: loading {} used clusters flags on layer {}, for {:.2f} MB.", this->mUsedClusters[iLayer].size(), iLayer, this->mUsedClusters[iLayer].size() * sizeof(unsigned char) / constants::MB);
    GPUChkErrS(cudaMemcpyAsync(mUsedClustersDevice[iLayer], this->mUsedClusters[iLayer].data(), this->mUsedClusters[iLayer].size() * sizeof(unsigned char), cudaMemcpyHostToDevice, mGpuStreams[iLayer].get()));
  }
}

template <int nLayers>
void TimeFrameGPU<nLayers>::createROFrameClustersDeviceArray(const int iteration)
{
  if (!iteration) {
    GPUTimer timer("creating ROFrame clusters array");
    allocMem(reinterpret_cast<void**>(&mROFramesClustersDeviceArray), nLayers * sizeof(int*), this->getExtAllocator());
    GPUChkErrS(cudaHostRegister(mROFramesClustersDevice.data(), nLayers * sizeof(int*), cudaHostRegisterPortable));
    mPinnedROFramesClusters.set(nLayers);
    for (auto iLayer{0}; iLayer < nLayers; ++iLayer) {
      GPUChkErrS(cudaHostRegister(this->mROFramesClusters[iLayer].data(), this->mROFramesClusters[iLayer].size() * sizeof(int), cudaHostRegisterPortable));
      mPinnedROFramesClusters.set(iLayer);
    }
  }
}

template <int nLayers>
void TimeFrameGPU<nLayers>::loadROFrameClustersDevice(const int iteration, const int layer)
{
  if (!iteration) {
    GPUTimer timer(mGpuStreams[layer], "loading ROframe clusters", layer);
    GPULog("gpu-transfer: loading {} ROframe clusters info on layer {}, for {:.2f} MB.", this->mROFramesClusters[layer].size(), layer, this->mROFramesClusters[layer].size() * sizeof(int) / constants::MB);
    allocMemAsync(reinterpret_cast<void**>(&mROFramesClustersDevice[layer]), this->mROFramesClusters[layer].size() * sizeof(int), mGpuStreams[layer], this->getExtAllocator());
    GPUChkErrS(cudaMemcpyAsync(mROFramesClustersDevice[layer], this->mROFramesClusters[layer].data(), this->mROFramesClusters[layer].size() * sizeof(int), cudaMemcpyHostToDevice, mGpuStreams[layer].get()));
    GPUChkErrS(cudaMemcpyAsync(&mROFramesClustersDeviceArray[layer], &mROFramesClustersDevice[layer], sizeof(int*), cudaMemcpyHostToDevice, mGpuStreams[layer].get()));
  }
}

template <int nLayers>
void TimeFrameGPU<nLayers>::createTrackingFrameInfoDeviceArray(const int iteration)
{
  if (!iteration) {
    GPUTimer timer("creating trackingframeinfo array");
    allocMem(reinterpret_cast<void**>(&mTrackingFrameInfoDeviceArray), nLayers * sizeof(TrackingFrameInfo*), this->getExtAllocator());
    GPUChkErrS(cudaHostRegister(mTrackingFrameInfoDevice.data(), nLayers * sizeof(TrackingFrameInfo*), cudaHostRegisterPortable));
    mPinnedTrackingFrameInfo.set(nLayers);
    for (auto iLayer{0}; iLayer < nLayers; ++iLayer) {
      GPUChkErrS(cudaHostRegister(this->mTrackingFrameInfo[iLayer].data(), this->mTrackingFrameInfo[iLayer].size() * sizeof(TrackingFrameInfo), cudaHostRegisterPortable));
      mPinnedTrackingFrameInfo.set(iLayer);
    }
  }
}

template <int nLayers>
void TimeFrameGPU<nLayers>::loadTrackingFrameInfoDevice(const int iteration, const int layer)
{
  if (!iteration) {
    GPUTimer timer(mGpuStreams[layer], "loading trackingframeinfo", layer);
    GPULog("gpu-transfer: loading {} tfinfo on layer {}, for {:.2f} MB.", this->mTrackingFrameInfo[layer].size(), layer, this->mTrackingFrameInfo[layer].size() * sizeof(TrackingFrameInfo) / constants::MB);
    allocMemAsync(reinterpret_cast<void**>(&mTrackingFrameInfoDevice[layer]), this->mTrackingFrameInfo[layer].size() * sizeof(TrackingFrameInfo), mGpuStreams[layer], this->getExtAllocator());
    GPUChkErrS(cudaMemcpyAsync(mTrackingFrameInfoDevice[layer], this->mTrackingFrameInfo[layer].data(), this->mTrackingFrameInfo[layer].size() * sizeof(TrackingFrameInfo), cudaMemcpyHostToDevice, mGpuStreams[layer].get()));
    GPUChkErrS(cudaMemcpyAsync(&mTrackingFrameInfoDeviceArray[layer], &mTrackingFrameInfoDevice[layer], sizeof(TrackingFrameInfo*), cudaMemcpyHostToDevice, mGpuStreams[layer].get()));
  }
}

template <int nLayers>
void TimeFrameGPU<nLayers>::loadMultiplicityCutMask(const int iteration)
{
  if (!iteration || iteration == 3) { // we need to re-load the swapped mult-mask in upc iteration
    GPUTimer timer("loading multiplicity cut mask");
    GPULog("gpu-transfer: iteration {} loading multiplicity cut mask with {} elements, for {:.2f} MB.", iteration, this->mMultiplicityCutMask.size(), this->mMultiplicityCutMask.size() * sizeof(uint8_t) / constants::MB);
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
    GPUTimer timer("loading seeding vertices");
    GPULog("gpu-transfer: loading {} ROframes vertices, for {:.2f} MB.", this->mROFramesPV.size(), this->mROFramesPV.size() * sizeof(int) / constants::MB);
    allocMem(reinterpret_cast<void**>(&mROFramesPVDevice), this->mROFramesPV.size() * sizeof(int), this->getExtAllocator());
    GPUChkErrS(cudaMemcpy(mROFramesPVDevice, this->mROFramesPV.data(), this->mROFramesPV.size() * sizeof(int), cudaMemcpyHostToDevice));
    GPULog("gpu-transfer: loading {} seeding vertices, for {:.2f} MB.", this->mPrimaryVertices.size(), this->mPrimaryVertices.size() * sizeof(Vertex) / constants::MB);
    allocMem(reinterpret_cast<void**>(&mPrimaryVerticesDevice), this->mPrimaryVertices.size() * sizeof(Vertex), this->getExtAllocator());
    GPUChkErrS(cudaMemcpy(mPrimaryVerticesDevice, this->mPrimaryVertices.data(), this->mPrimaryVertices.size() * sizeof(Vertex), cudaMemcpyHostToDevice));
  }
}

template <int nLayers>
void TimeFrameGPU<nLayers>::createTrackletsLUTDeviceArray(const int iteration)
{
  if (!iteration) {
    allocMem(reinterpret_cast<void**>(&mTrackletsLUTDeviceArray), (nLayers - 1) * sizeof(int*), this->getExtAllocator());
  }
}

template <int nLayers>
void TimeFrameGPU<nLayers>::createTrackletsLUTDevice(const int iteration, const int layer)
{
  GPUTimer timer(mGpuStreams[layer], "creating tracklets LUTs", layer);
  const int ncls = this->mClusters[layer].size() + 1;
  if (!iteration) {
    GPULog("gpu-allocation: creating tracklets LUT for {} elements on layer {}, for {:.2f} MB.", ncls, layer, ncls * sizeof(int) / constants::MB);
    allocMemAsync(reinterpret_cast<void**>(&mTrackletsLUTDevice[layer]), ncls * sizeof(int), mGpuStreams[layer], this->getExtAllocator());
    GPUChkErrS(cudaMemcpyAsync(&mTrackletsLUTDeviceArray[layer], &mTrackletsLUTDevice[layer], sizeof(int*), cudaMemcpyHostToDevice, mGpuStreams[layer].get()));
  }
  GPUChkErrS(cudaMemsetAsync(mTrackletsLUTDevice[layer], 0, ncls * sizeof(int), mGpuStreams[layer].get()));
}

template <int nLayers>
void TimeFrameGPU<nLayers>::createTrackletsBuffersArray(const int iteration)
{
  if (!iteration) {
    GPUTimer timer("creating tracklet buffers array");
    allocMem(reinterpret_cast<void**>(&mTrackletsDeviceArray), (nLayers - 1) * sizeof(Tracklet*), this->getExtAllocator());
  }
}

template <int nLayers>
void TimeFrameGPU<nLayers>::createTrackletsBuffers(const int layer)
{
  GPUTimer timer(mGpuStreams[layer], "creating tracklet buffers", layer);
  mNTracklets[layer] = 0;
  GPUChkErrS(cudaMemcpyAsync(&mNTracklets[layer], mTrackletsLUTDevice[layer] + this->mClusters[layer].size(), sizeof(int), cudaMemcpyDeviceToHost, mGpuStreams[layer].get()));
  mGpuStreams[layer].sync(); // ensure number of tracklets is correct
  GPULog("gpu-transfer: creating tracklets buffer for {} elements on layer {}, for {:.2f} MB.", mNTracklets[layer], layer, mNTracklets[layer] * sizeof(Tracklet) / constants::MB);
  allocMemAsync(reinterpret_cast<void**>(&mTrackletsDevice[layer]), mNTracklets[layer] * sizeof(Tracklet), mGpuStreams[layer], this->getExtAllocator());
  GPUChkErrS(cudaMemcpyAsync(&mTrackletsDeviceArray[layer], &mTrackletsDevice[layer], sizeof(Tracklet*), cudaMemcpyHostToDevice, mGpuStreams[layer].get()));
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
  GPUTimer timer("loading tracklets");
  for (auto iLayer{0}; iLayer < nLayers - 2; ++iLayer) {
    GPULog("gpu-transfer: loading tracklets LUT for {} elements on layer {}, for {:.2f} MB", this->mTrackletsLookupTable[iLayer].size(), iLayer + 1, this->mTrackletsLookupTable[iLayer].size() * sizeof(int) / constants::MB);
    GPUChkErrS(cudaMemcpyAsync(mTrackletsLUTDevice[iLayer + 1], this->mTrackletsLookupTable[iLayer].data(), this->mTrackletsLookupTable[iLayer].size() * sizeof(int), cudaMemcpyHostToDevice, mGpuStreams[iLayer].get()));
  }
  mGpuStreams.sync();
  GPUChkErrS(cudaMemcpy(mTrackletsLUTDeviceArray, mTrackletsLUTDevice.data(), (nLayers - 1) * sizeof(int*), cudaMemcpyHostToDevice));
}

template <int nLayers>
void TimeFrameGPU<nLayers>::createNeighboursIndexTablesDevice(const int layer)
{
  GPUTimer timer(mGpuStreams[layer], "creating cells neighbours", layer);
  GPULog("gpu-transfer: reserving neighbours LUT for {} elements on layer {}, for {:.2f} MB.", mNCells[layer] + 1, layer, (mNCells[layer] + 1) * sizeof(int) / constants::MB);
  allocMemAsync(reinterpret_cast<void**>(&mNeighboursIndexTablesDevice[layer]), (mNCells[layer] + 1) * sizeof(int), mGpuStreams[layer], this->getExtAllocator());
  GPUChkErrS(cudaMemsetAsync(mNeighboursIndexTablesDevice[layer], 0, (mNCells[layer] + 1) * sizeof(int), mGpuStreams[layer].get()));
}

template <int nLayers>
void TimeFrameGPU<nLayers>::createNeighboursLUTDevice(const int layer, const unsigned int nCells)
{
  GPUTimer timer(mGpuStreams[layer], "reserving neighboursLUT");
  GPULog("gpu-allocation: reserving neighbours LUT for {} elements on layer {} , for {:.2f} MB.", nCells + 1, layer, (nCells + 1) * sizeof(int) / constants::MB);
  allocMemAsync(reinterpret_cast<void**>(&mNeighboursLUTDevice[layer]), (nCells + 1) * sizeof(int), mGpuStreams[layer], this->getExtAllocator()); // We need one element more to move exc -> inc
  GPUChkErrS(cudaMemsetAsync(mNeighboursLUTDevice[layer], 0, (nCells + 1) * sizeof(int), mGpuStreams[layer].get()));
}

template <int nLayers>
void TimeFrameGPU<nLayers>::loadCellsDevice()
{
  GPUTimer timer(mGpuStreams, "loading cell seeds", nLayers - 2);
  for (auto iLayer{0}; iLayer < nLayers - 2; ++iLayer) {
    GPULog("gpu-transfer: loading {} cell seeds on layer {}, for {:.2f} MB.", this->mCells[iLayer].size(), iLayer, this->mCells[iLayer].size() * sizeof(CellSeedN) / constants::MB);
    allocMemAsync(reinterpret_cast<void**>(&mCellsDevice[iLayer]), this->mCells[iLayer].size() * sizeof(CellSeedN), mGpuStreams[iLayer], this->getExtAllocator());
    allocMemAsync(reinterpret_cast<void**>(&mNeighboursIndexTablesDevice[iLayer]), (this->mCells[iLayer].size() + 1) * sizeof(int), mGpuStreams[iLayer], this->getExtAllocator()); // accessory for the neigh. finding.
    GPUChkErrS(cudaMemsetAsync(mNeighboursIndexTablesDevice[iLayer], 0, (this->mCells[iLayer].size() + 1) * sizeof(int), mGpuStreams[iLayer].get()));
    GPUChkErrS(cudaMemcpyAsync(mCellsDevice[iLayer], this->mCells[iLayer].data(), this->mCells[iLayer].size() * sizeof(CellSeedN), cudaMemcpyHostToDevice, mGpuStreams[iLayer].get()));
  }
}

template <int nLayers>
void TimeFrameGPU<nLayers>::createCellsLUTDeviceArray(const int iteration)
{
  if (!iteration) {
    GPUTimer timer("creating cells LUTs array");
    allocMem(reinterpret_cast<void**>(&mCellsLUTDeviceArray), (nLayers - 2) * sizeof(int*), this->getExtAllocator());
  }
}

template <int nLayers>
void TimeFrameGPU<nLayers>::createCellsLUTDevice(const int layer)
{
  GPUTimer timer(mGpuStreams[layer], "creating cells LUTs", layer);
  GPULog("gpu-transfer: creating cell LUT for {} elements on layer {}, for {:.2f} MB.", mNTracklets[layer] + 1, layer, (mNTracklets[layer] + 1) * sizeof(int) / constants::MB);
  allocMemAsync(reinterpret_cast<void**>(&mCellsLUTDevice[layer]), (mNTracklets[layer] + 1) * sizeof(int), mGpuStreams[layer], this->getExtAllocator());
  GPUChkErrS(cudaMemsetAsync(mCellsLUTDevice[layer], 0, (mNTracklets[layer] + 1) * sizeof(int), mGpuStreams[layer].get()));
  GPUChkErrS(cudaMemcpyAsync(&mCellsLUTDeviceArray[layer], &mCellsLUTDevice[layer], sizeof(int*), cudaMemcpyHostToDevice, mGpuStreams[layer].get()));
}

template <int nLayers>
void TimeFrameGPU<nLayers>::createCellsBuffersArray(const int iteration)
{
  if (!iteration) {
    GPUTimer timer("creating cells buffers array");
    allocMem(reinterpret_cast<void**>(&mCellsDeviceArray), (nLayers - 2) * sizeof(CellSeedN*), this->getExtAllocator());
    GPUChkErrS(cudaMemcpy(mCellsDeviceArray, mCellsDevice.data(), mCellsDevice.size() * sizeof(CellSeedN*), cudaMemcpyHostToDevice));
  }
}

template <int nLayers>
void TimeFrameGPU<nLayers>::createCellsBuffers(const int layer)
{
  GPUTimer timer(mGpuStreams[layer], "creating cells buffers");
  mNCells[layer] = 0;
  GPUChkErrS(cudaMemcpyAsync(&mNCells[layer], mCellsLUTDevice[layer] + mNTracklets[layer], sizeof(int), cudaMemcpyDeviceToHost, mGpuStreams[layer].get()));
  mGpuStreams[layer].sync(); // ensure number of cells is correct
  GPULog("gpu-transfer: creating cell buffer for {} elements on layer {}, for {:.2f} MB.", mNCells[layer], layer, mNCells[layer] * sizeof(CellSeedN) / constants::MB);
  allocMemAsync(reinterpret_cast<void**>(&mCellsDevice[layer]), mNCells[layer] * sizeof(CellSeedN), mGpuStreams[layer], this->getExtAllocator());
  GPUChkErrS(cudaMemcpyAsync(&mCellsDeviceArray[layer], &mCellsDevice[layer], sizeof(CellSeedN*), cudaMemcpyHostToDevice, mGpuStreams[layer].get()));
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
  GPUTimer timer("loading roads device");
  GPULog("gpu-transfer: loading {} roads, for {:.2f} MB.", this->mRoads.size(), this->mRoads.size() * sizeof(Road<nLayers - 2>) / constants::MB);
  allocMem(reinterpret_cast<void**>(&mRoadsDevice), this->mRoads.size() * sizeof(Road<nLayers - 2>), this->getExtAllocator());
  GPUChkErrS(cudaHostRegister(this->mRoads.data(), this->mRoads.size() * sizeof(Road<nLayers - 2>), cudaHostRegisterPortable));
  GPUChkErrS(cudaMemcpy(mRoadsDevice, this->mRoads.data(), this->mRoads.size() * sizeof(Road<nLayers - 2>), cudaMemcpyHostToDevice));
}

template <int nLayers>
void TimeFrameGPU<nLayers>::loadTrackSeedsDevice(bounded_vector<CellSeedN>& seeds)
{
  GPUTimer timer("loading track seeds");
  GPULog("gpu-transfer: loading {} track seeds, for {:.2f} MB.", seeds.size(), seeds.size() * sizeof(CellSeedN) / constants::MB);
  allocMem(reinterpret_cast<void**>(&mTrackSeedsDevice), seeds.size() * sizeof(CellSeedN), this->getExtAllocator());
  GPUChkErrS(cudaHostRegister(seeds.data(), seeds.size() * sizeof(CellSeedN), cudaHostRegisterPortable));
  GPUChkErrS(cudaMemcpy(mTrackSeedsDevice, seeds.data(), seeds.size() * sizeof(CellSeedN), cudaMemcpyHostToDevice));
}

template <int nLayers>
void TimeFrameGPU<nLayers>::createNeighboursDevice(const unsigned int layer)
{
  GPUTimer timer(mGpuStreams[layer], "reserving neighbours", layer);
  this->mNNeighbours[layer] = 0;
  GPUChkErrS(cudaMemcpyAsync(&(this->mNNeighbours[layer]), &(mNeighboursLUTDevice[layer][this->mNCells[layer + 1] - 1]), sizeof(unsigned int), cudaMemcpyDeviceToHost, mGpuStreams[layer].get()));
  mGpuStreams[layer].sync(); // ensure number of neighbours is correct
  GPULog("gpu-allocation: reserving {} neighbours (pairs), for {:.2f} MB.", this->mNNeighbours[layer], (this->mNNeighbours[layer]) * sizeof(gpuPair<int, int>) / constants::MB);
  allocMemAsync(reinterpret_cast<void**>(&mNeighbourPairsDevice[layer]), (this->mNNeighbours[layer]) * sizeof(gpuPair<int, int>), mGpuStreams[layer], this->getExtAllocator());
  GPUChkErrS(cudaMemsetAsync(mNeighbourPairsDevice[layer], -1, (this->mNNeighbours[layer]) * sizeof(gpuPair<int, int>), mGpuStreams[layer].get()));
  GPULog("gpu-allocation: reserving {} neighbours, for {:.2f} MB.", this->mNNeighbours[layer], (this->mNNeighbours[layer]) * sizeof(gpuPair<int, int>) / constants::MB);
  allocMemAsync(reinterpret_cast<void**>(&mNeighboursDevice[layer]), (this->mNNeighbours[layer]) * sizeof(int), mGpuStreams[layer], this->getExtAllocator());
}

template <int nLayers>
void TimeFrameGPU<nLayers>::createTrackITSExtDevice(bounded_vector<CellSeedN>& seeds)
{
  GPUTimer timer("reserving tracks");
  mTrackITSExt = bounded_vector<TrackITSExt>(seeds.size(), {}, this->getMemoryPool().get());
  GPULog("gpu-allocation: reserving {} tracks, for {:.2f} MB.", seeds.size(), seeds.size() * sizeof(o2::its::TrackITSExt) / constants::MB);
  allocMem(reinterpret_cast<void**>(&mTrackITSExtDevice), seeds.size() * sizeof(o2::its::TrackITSExt), this->getExtAllocator());
  GPUChkErrS(cudaMemset(mTrackITSExtDevice, 0, seeds.size() * sizeof(o2::its::TrackITSExt)));
  GPUChkErrS(cudaHostRegister(mTrackITSExt.data(), seeds.size() * sizeof(o2::its::TrackITSExt), cudaHostRegisterPortable));
}

template <int nLayers>
void TimeFrameGPU<nLayers>::downloadCellsDevice()
{
  GPUTimer timer(mGpuStreams, "downloading cells", nLayers - 2);
  for (int iLayer{0}; iLayer < nLayers - 2; ++iLayer) {
    GPULog("gpu-transfer: downloading {} cells on layer: {}, for {:.2f} MB.", mNCells[iLayer], iLayer, mNCells[iLayer] * sizeof(CellSeedN) / constants::MB);
    this->mCells[iLayer].resize(mNCells[iLayer]);
    GPUChkErrS(cudaMemcpyAsync(this->mCells[iLayer].data(), this->mCellsDevice[iLayer], mNCells[iLayer] * sizeof(CellSeedN), cudaMemcpyDeviceToHost, mGpuStreams[iLayer].get()));
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
  GPUTimer timer(mGpuStreams[layer], "downloading neighbours from layer", layer);
  GPULog("gpu-transfer: downloading {} neighbours, for {:.2f} MB.", neighbours[layer].size(), neighbours[layer].size() * sizeof(std::pair<int, int>) / constants::MB);
  GPUChkErrS(cudaMemcpyAsync(neighbours[layer].data(), mNeighbourPairsDevice[layer], neighbours[layer].size() * sizeof(gpuPair<int, int>), cudaMemcpyDeviceToHost, mGpuStreams[layer].get()));
}

template <int nLayers>
void TimeFrameGPU<nLayers>::downloadNeighboursLUTDevice(bounded_vector<int>& lut, const int layer)
{
  GPUTimer timer(mGpuStreams[layer], "downloading neighbours LUT from layer", layer);
  GPULog("gpu-transfer: downloading neighbours LUT for {} elements on layer {}, for {:.2f} MB.", lut.size(), layer, lut.size() * sizeof(int) / constants::MB);
  GPUChkErrS(cudaMemcpyAsync(lut.data(), mNeighboursLUTDevice[layer], lut.size() * sizeof(int), cudaMemcpyDeviceToHost, mGpuStreams[layer].get()));
}

template <int nLayers>
void TimeFrameGPU<nLayers>::downloadTrackITSExtDevice(bounded_vector<CellSeedN>& seeds)
{
  GPUTimer timer("downloading tracks");
  GPULog("gpu-transfer: downloading {} tracks, for {:.2f} MB.", mTrackITSExt.size(), mTrackITSExt.size() * sizeof(o2::its::TrackITSExt) / constants::MB);
  GPUChkErrS(cudaMemcpy(mTrackITSExt.data(), mTrackITSExtDevice, seeds.size() * sizeof(o2::its::TrackITSExt), cudaMemcpyDeviceToHost));
  GPUChkErrS(cudaHostUnregister(mTrackITSExt.data()));
  GPUChkErrS(cudaHostUnregister(seeds.data()));
}

template <int nLayers>
void TimeFrameGPU<nLayers>::unregisterHostMemory(const int maxLayers)
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
    if (bits.test(nLayers)) {
      GPUChkErrS(cudaHostUnregister(vec.data()));
      bits.reset(nLayers);
    }
  };

  for (auto iLayer{0}; iLayer < nLayers; ++iLayer) {
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

template <int nLayers>
void TimeFrameGPU<nLayers>::initialise(const int iteration,
                                       const TrackingParameters& trkParam,
                                       const int maxLayers,
                                       IndexTableUtilsN* utils,
                                       const TimeFrameGPUParameters* gpuParam)
{
  mGpuStreams.resize(nLayers);
  o2::its::TimeFrame<nLayers>::initialise(iteration, trkParam, maxLayers);
}

template <int nLayers>
void TimeFrameGPU<nLayers>::syncStream(const size_t stream)
{
  mGpuStreams[stream].sync();
}

template <int nLayers>
void TimeFrameGPU<nLayers>::syncStreams(const bool device)
{
  mGpuStreams.sync(device);
}

template <int nLayers>
void TimeFrameGPU<nLayers>::waitEvent(const int stream, const int event)
{
  mGpuStreams.waitEvent(stream, event);
}

template <int nLayers>
void TimeFrameGPU<nLayers>::recordEvent(const int event)
{
  mGpuStreams[event].record();
}

template <int nLayers>
void TimeFrameGPU<nLayers>::recordEvents(const int start, const int end)
{
  for (int i{start}; i < end; ++i) {
    recordEvent(i);
  }
}

template <int nLayers>
void TimeFrameGPU<nLayers>::wipe()
{
  unregisterHostMemory(0);
  o2::its::TimeFrame<nLayers>::wipe();
}

template class TimeFrameGPU<7>;
} // namespace o2::its::gpu
