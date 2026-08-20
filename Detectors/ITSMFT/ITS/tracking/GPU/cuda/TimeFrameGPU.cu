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

#include <algorithm>
#include <numeric>
#include <array>
#include <type_traits>
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
template <typename T>
T* TimeFrameGPU<NLayers>::allocDevice(const size_t n, const int32_t type)
{
  if (n == 0) {
    return nullptr;
  }
  void* ptr{nullptr};
  if (this->hasFrameworkAllocator()) {
    ptr = (this->mExternalAllocator)->allocate(n * sizeof(T), type);
  } else {
    GPULog("Calling default CUDA allocator");
    GPUChkErrS(cudaMalloc(&ptr, n * sizeof(T)));
  }
  return static_cast<T*>(ptr);
}

template <int NLayers>
template <typename T>
T* TimeFrameGPU<NLayers>::allocDeviceAsync(const size_t n, Stream& stream, const int32_t type)
{
  if (n == 0) {
    return nullptr;
  }
  void* ptr{nullptr};
  if (this->hasFrameworkAllocator()) {
    ptr = (this->mExternalAllocator)->allocate(n * sizeof(T), type);
  } else {
    GPULog("Calling default CUDA allocator");
    GPUChkErrS(cudaMallocAsync(&ptr, n * sizeof(T), stream.get()));
  }
  return static_cast<T*>(ptr);
}

template <int NLayers>
template <typename SlotPtr>
SlotPtr* TimeFrameGPU<NLayers>::allocSlotArray(const size_t n)
{
  auto* array = allocDevice<SlotPtr>(n);
  if (array != nullptr) {
    GPUChkErrS(cudaMemset(array, 0, n * sizeof(SlotPtr)));
  }
  return array;
}

template <int NLayers>
template <typename T>
void TimeFrameGPU<NLayers>::copyToDevice(T* dst, const T* src, const size_t n)
{
  if (n > 0) {
    GPUChkErrS(cudaMemcpy(dst, src, n * sizeof(T), cudaMemcpyHostToDevice));
  }
}

template <int NLayers>
template <typename T>
void TimeFrameGPU<NLayers>::copyFromDevice(T* dst, const T* src, const size_t n)
{
  if (n > 0) {
    GPUChkErrS(cudaMemcpy(dst, src, n * sizeof(T), cudaMemcpyDeviceToHost));
  }
}

template <int NLayers>
template <typename T, typename ArrayT>
void TimeFrameGPU<NLayers>::publishSlot(ArrayT deviceArray, const int slot, T* const& devicePtr, Stream& stream)
{
  GPUChkErrS(cudaMemcpyAsync(&deviceArray[slot], &devicePtr, sizeof(T*), cudaMemcpyHostToDevice, stream.get()));
}

template <int NLayers>
template <typename T, size_t N, typename ArrayT>
T* TimeFrameGPU<NLayers>::createSlot(std::array<T*, N>& slots, ArrayT deviceArray, const int slot, const size_t n,
                                     const char* what, const SlotInit init, const int32_t type)
{
  auto& stream = mGpuStreams[slot];
  GPULog("gpu-allocation: creating {} for {} elements on slot {}, for {:.2f} MB.", what, n, slot, n * sizeof(T) / constants::MB);
  slots[slot] = allocDeviceAsync<T>(n, stream, type);
  if (init == SlotInit::Zero && n > 0) {
    GPUChkErrS(cudaMemsetAsync(slots[slot], 0, n * sizeof(T), stream.get()));
  }
  publishSlot(deviceArray, slot, slots[slot], stream);
  return slots[slot];
}

template <int NLayers>
template <typename T, size_t N, typename ArrayT, typename Container>
void TimeFrameGPU<NLayers>::uploadSlot(std::array<T*, N>& slots, ArrayT deviceArray, const int slot, const Container& host, const char* what)
{
  auto& stream = mGpuStreams[slot];
  GPULog("gpu-transfer: loading {} {} on slot {}, for {:.2f} MB.", host.size(), what, slot, host.size() * sizeof(T) / constants::MB);
  slots[slot] = allocDeviceAsync<T>(host.size(), stream);
  if (!host.empty()) {
    GPUChkErrS(cudaMemcpyAsync(slots[slot], host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice, stream.get()));
  }
  publishSlot(deviceArray, slot, slots[slot], stream);
}

template <int NLayers>
template <typename ArrayT, typename T, size_t N>
void TimeFrameGPU<NLayers>::createPinnedSlotArray(ArrayT& deviceArray, std::array<T*, N>& slots, std::bitset<NLayers + 1>& pinned)
{
  deviceArray = allocSlotArray<std::remove_pointer_t<ArrayT>>(N);
  GPUChkErrS(cudaHostRegister(slots.data(), N * sizeof(T*), cudaHostRegisterPortable));
  pinned.set(NLayers);
}

template <int NLayers>
template <typename Layers>
void TimeFrameGPU<NLayers>::pinHostLayers(Layers& layers, std::bitset<NLayers + 1>& pinned, const int maxLayers)
{
  if (this->hasFrameworkAllocator()) { // the framework already hands out registered memory
    return;
  }
  for (auto iLayer{0}; iLayer < o2::gpu::CAMath::Min(maxLayers, NLayers); ++iLayer) {
    auto& host = layers[iLayer];
    if (host.empty()) { // registering an empty range fails, and the bit must stay clear for wipe()
      continue;
    }
    GPUChkErrS(cudaHostRegister(host.data(), host.size() * sizeof(typename std::decay_t<decltype(host)>::value_type), cudaHostRegisterPortable));
    pinned.set(iLayer);
  }
}

template <int NLayers>
template <typename Table>
typename Table::View TimeFrameGPU<NLayers>::uploadNavigationTable(const Table& table, const typename Table::View& hostView)
{
  auto* dFlatTable = allocDevice<typename Table::TableEntry>(table.getFlatTableSize());
  auto* dIndices = allocDevice<typename Table::TableIndex>(table.getIndicesSize());
  auto* dLayers = allocDevice<o2::its::LayerTiming>(NLayers);
  copyToDevice(dFlatTable, hostView.mFlatTable, table.getFlatTableSize());
  copyToDevice(dIndices, hostView.mIndices, table.getIndicesSize());
  copyToDevice(dLayers, hostView.mLayers, NLayers);
  return table.getDeviceView(dFlatTable, dIndices, dLayers);
}

template <int NLayers>
void TimeFrameGPU<NLayers>::loadIndexTableUtils()
{
  GPUTimer timer("loading indextable utils");
  GPULog("gpu-transfer: loading IndexTableUtils object, for {:.2f} MB.", sizeof(IndexTableUtilsN) / constants::MB);
  mIndexTableUtilsDevice = allocDevice<IndexTableUtilsN>(1);
  copyToDevice(mIndexTableUtilsDevice, &(this->mIndexTableUtils), 1);
}

template <int NLayers>
void TimeFrameGPU<NLayers>::loadIterationParameters(const TrackingParameters& params)
{
  GPUTimer timer("loading iteration parameters");
  const auto& radii = params.LayerRadii;
  const auto& minPts = params.MinPt;
  const auto& xX0 = params.LayerxX0;
  const size_t n = radii.size() + minPts.size() + xX0.size();
  GPULog("gpu-transfer: loading {} iteration parameters, for {:.2f} MB.", n, n * sizeof(float) / constants::MB);
  std::vector<float> staging;
  staging.reserve(n);
  staging.insert(staging.end(), radii.begin(), radii.end());
  staging.insert(staging.end(), minPts.begin(), minPts.end());
  staging.insert(staging.end(), xX0.begin(), xX0.end());
  mIterationParametersDevice = allocDevice<float>(n);
  copyToDevice(mIterationParametersDevice, staging.data(), n);
  mLayerRadiiDevice = mIterationParametersDevice;
  mMinPtsDevice = mLayerRadiiDevice + radii.size();
  mLayerxX0Device = mMinPtsDevice + minPts.size();
}

template <int NLayers>
void TimeFrameGPU<NLayers>::createUnsortedClustersDeviceArray(const int maxLayers)
{
  GPUTimer timer("creating unsorted clusters array");
  createPinnedSlotArray(mUnsortedClustersDeviceArray, mUnsortedClustersDevice, mPinnedUnsortedClusters);
  pinHostLayers(this->mUnsortedClusters, mPinnedUnsortedClusters, maxLayers);
}

template <int NLayers>
void TimeFrameGPU<NLayers>::loadUnsortedClustersDevice(const int layer)
{
  GPUTimer timer(mGpuStreams[layer], "loading unsorted clusters", layer);
  uploadSlot(mUnsortedClustersDevice, mUnsortedClustersDeviceArray, layer, this->mUnsortedClusters[layer], "unsorted clusters");
}

template <int NLayers>
void TimeFrameGPU<NLayers>::createClustersDeviceArray(const int maxLayers)
{
  GPUTimer timer("creating sorted clusters array");
  createPinnedSlotArray(mClustersDeviceArray, mClustersDevice, mPinnedClusters);
  pinHostLayers(this->mClusters, mPinnedClusters, maxLayers);
}

template <int NLayers>
void TimeFrameGPU<NLayers>::loadClustersDevice(const int layer)
{
  GPUTimer timer(mGpuStreams[layer], "loading sorted clusters", layer);
  uploadSlot(mClustersDevice, mClustersDeviceArray, layer, this->mClusters[layer], "sorted clusters");
}

template <int NLayers>
void TimeFrameGPU<NLayers>::createClustersIndexTablesArray(const int maxLayers)
{
  GPUTimer timer("creating clustersindextable array");
  createPinnedSlotArray(mClustersIndexTablesDeviceArray, mClustersIndexTablesDevice, mPinnedClustersIndexTables);
  pinHostLayers(this->mIndexTables, mPinnedClustersIndexTables, maxLayers);
}

template <int NLayers>
void TimeFrameGPU<NLayers>::loadClustersIndexTables(const int layer)
{
  GPUTimer timer(mGpuStreams[layer], "loading clusters indextables", layer);
  uploadSlot(mClustersIndexTablesDevice, mClustersIndexTablesDeviceArray, layer, this->mIndexTables[layer], "clusters indextable entries");
}

template <int NLayers>
void TimeFrameGPU<NLayers>::createUsedClustersDeviceArray(const int maxLayers)
{
  GPUTimer timer("creating used clusters flags");
  createPinnedSlotArray(mUsedClustersDeviceArray, mUsedClustersDevice, mPinnedUsedClusters);
  pinHostLayers(this->mUsedClusters, mPinnedUsedClusters, maxLayers);
}

template <int NLayers>
void TimeFrameGPU<NLayers>::createUsedClustersDevice(const int layer)
{
  GPUTimer timer(mGpuStreams[layer], "creating used clusters flags", layer);
  createSlot(mUsedClustersDevice, mUsedClustersDeviceArray, layer, this->mUsedClusters[layer].size(), "used clusters flags", SlotInit::Zero);
}

template <int NLayers>
void TimeFrameGPU<NLayers>::loadUsedClustersDevice()
{
  for (auto iLayer{0}; iLayer < NLayers; ++iLayer) {
    GPUTimer timer(mGpuStreams[iLayer], "loading used clusters flags", iLayer);
    const auto& used = this->mUsedClusters[iLayer];
    GPULog("gpu-transfer: loading {} used clusters flags on layer {}, for {:.2f} MB.", used.size(), iLayer, used.size() * sizeof(unsigned char) / constants::MB);
    if (!used.empty()) {
      GPUChkErrS(cudaMemcpyAsync(mUsedClustersDevice[iLayer], used.data(), used.size() * sizeof(unsigned char), cudaMemcpyHostToDevice, mGpuStreams[iLayer].get()));
    }
  }
}

template <int NLayers>
void TimeFrameGPU<NLayers>::createROFrameClustersDeviceArray(const int maxLayers)
{
  GPUTimer timer("creating ROFrame clusters array");
  createPinnedSlotArray(mROFramesClustersDeviceArray, mROFramesClustersDevice, mPinnedROFramesClusters);
  pinHostLayers(this->mROFramesClusters, mPinnedROFramesClusters, maxLayers);
}

template <int NLayers>
void TimeFrameGPU<NLayers>::loadROFrameClustersDevice(const int layer)
{
  GPUTimer timer(mGpuStreams[layer], "loading ROframe clusters", layer);
  uploadSlot(mROFramesClustersDevice, mROFramesClustersDeviceArray, layer, this->mROFramesClusters[layer], "ROframe clusters");
}

template <int NLayers>
void TimeFrameGPU<NLayers>::createTrackingFrameInfoDeviceArray(const int maxLayers)
{
  GPUTimer timer("creating trackingframeinfo array");
  createPinnedSlotArray(mTrackingFrameInfoDeviceArray, mTrackingFrameInfoDevice, mPinnedTrackingFrameInfo);
  pinHostLayers(this->mTrackingFrameInfo, mPinnedTrackingFrameInfo, maxLayers);
}

template <int NLayers>
void TimeFrameGPU<NLayers>::loadTrackingFrameInfoDevice(const int layer)
{
  GPUTimer timer(mGpuStreams[layer], "loading trackingframeinfo", layer);
  uploadSlot(mTrackingFrameInfoDevice, mTrackingFrameInfoDeviceArray, layer, this->mTrackingFrameInfo[layer], "tfinfo");
}

template <int NLayers>
void TimeFrameGPU<NLayers>::loadROFCutMask(const int iteration)
{
  GPUTimer timer("loading multiplicity cut mask");
  const auto& hostTable = *(this->mROFMask);
  const auto hostView = hostTable.getView();
  using TableEntry = ROFMaskTable<NLayers>::TableEntry;
  using TableIndex = ROFMaskTable<NLayers>::TableIndex;
  GPULog("gpu-transfer: iteration {} loading multiplicity cut mask with {} elements, for {:.2f} MB.",
         iteration, hostTable.getFlatMaskSize(), hostTable.getFlatMaskSize() * sizeof(TableEntry) / constants::MB);
  auto* dFlatMask = allocDevice<TableEntry>(hostTable.getFlatMaskSize());
  auto* dOffsets = allocDevice<TableIndex>(NLayers + 1); // the view reads the sentinel past the last layer
  copyToDevice(dOffsets, hostView.mLayerROFOffsets, NLayers + 1);
  // Re-copy the flat mask on every qualifying iteration (e.g. after swapMasks() for UPC)
  copyToDevice(dFlatMask, hostView.mFlatMask, hostTable.getFlatMaskSize());
  mDeviceROFMaskTableView = hostTable.getDeviceView(dFlatMask, dOffsets);
}

template <int NLayers>
void TimeFrameGPU<NLayers>::loadVertices()
{
  GPUTimer timer("loading seeding vertices");
  GPULog("gpu-transfer: loading {} seeding vertices, for {:.2f} MB.", this->mPrimaryVertices.size(), this->mPrimaryVertices.size() * sizeof(Vertex) / constants::MB);
  mPrimaryVerticesDevice = allocDevice<Vertex>(this->mPrimaryVertices.size());
  copyToDevice(mPrimaryVerticesDevice, this->mPrimaryVertices.data(), this->mPrimaryVertices.size());
}

template <int NLayers>
void TimeFrameGPU<NLayers>::loadROFOverlapTable()
{
  GPUTimer timer("initialising device view of ROFOverlapTable");
  mDeviceROFOverlapTableView = uploadNavigationTable(this->getROFOverlapTable(), this->getROFOverlapTableView());
}

template <int NLayers>
void TimeFrameGPU<NLayers>::loadROFVertexLookupTable()
{
  GPUTimer timer("initialising device view of ROFVertexLookupTable");
  mDeviceROFVertexLookupTableView = uploadNavigationTable(this->getROFVertexLookupTable(), this->getROFVertexLookupTableView());
}

template <int NLayers>
void TimeFrameGPU<NLayers>::loadTrackingTopologies()
{
  GPUTimer timer("initialising device views of TrackingTopology");
  const auto& hostTopologies = this->getTrackerTopologies();
  mDeviceTrackerTopologyViews.resize(hostTopologies.size());
  using LayerLink = typename TrackingTopologyN::LayerLink;
  using CellTopology = typename TrackingTopologyN::CellTopology;
  using Range = typename TrackingTopologyN::Range;
  using Id = typename TrackingTopologyN::Id;
  for (size_t iteration = 0; iteration < hostTopologies.size(); ++iteration) {
    const auto& topology = hostTopologies[iteration];
    auto* dLinks = allocDevice<LayerLink>(topology.getNLinks());
    auto* dCells = allocDevice<CellTopology>(topology.getNCells());
    auto* dCellsByFirstLinkIndex = allocDevice<Range>(topology.getNLinks());
    auto* dCellsByFirstLink = allocDevice<Id>(topology.getNCellsByFirstLink());
    copyToDevice(dLinks, topology.getLinks().data(), topology.getNLinks());
    copyToDevice(dCells, topology.getCells().data(), topology.getNCells());
    copyToDevice(dCellsByFirstLinkIndex, topology.getCellsByFirstLinkIndex().data(), topology.getNLinks());
    copyToDevice(dCellsByFirstLink, topology.getCellsByFirstLink().data(), topology.getNCellsByFirstLink());
    mDeviceTrackerTopologyViews[iteration] = topology.getDeviceView(dLinks, dCells, dCellsByFirstLinkIndex, dCellsByFirstLink);
  }
  if (!mDeviceTrackerTopologyViews.empty()) {
    mDeviceTrackingTopologyView = mDeviceTrackerTopologyViews.front();
  }
}

template <int NLayers>
void TimeFrameGPU<NLayers>::uploadROFVertexLookupTable()
{
  GPUTimer timer("updating device view of ROFVertexLookupTable");
  const auto& hostTable = this->getROFVertexLookupTable();
  const auto& hostView = this->getROFVertexLookupTableView();
  using TableEntry = ROFVertexLookupTable<NLayers>::TableEntry;
  auto* dFlatTable = allocDevice<TableEntry>(hostTable.getFlatTableSize());
  copyToDevice(dFlatTable, hostView.mFlatTable, hostTable.getFlatTableSize());
  mDeviceROFVertexLookupTableView = hostTable.getDeviceView(dFlatTable, mDeviceROFVertexLookupTableView.mIndices, mDeviceROFVertexLookupTableView.mLayers);
}

template <int NLayers>
void TimeFrameGPU<NLayers>::createTrackletsLUTDeviceArray()
{
  mTrackletsLUTDeviceArray = allocSlotArray<int*>(MaxLinks);
}

template <int NLayers>
void TimeFrameGPU<NLayers>::createTrackletsLUTDevice(bool allocate, const int layer)
{
  GPUTimer timer(mGpuStreams[layer], "creating tracklets LUTs", layer);
  const int fromLayer = this->mTrackingTopologyView.getLink(layer).fromLayer;
  const size_t ncls = this->mClusters[fromLayer].size() + 1;
  if (allocate || mTrackletsLUTDevice[layer] == nullptr) {
    createSlot(mTrackletsLUTDevice, mTrackletsLUTDeviceArray, layer, ncls, "tracklets LUT");
  }
  GPUChkErrS(cudaMemsetAsync(mTrackletsLUTDevice[layer], 0, ncls * sizeof(int), mGpuStreams[layer].get()));
}

template <int NLayers>
void TimeFrameGPU<NLayers>::createTrackletsBuffersArray()
{
  GPUTimer timer("creating tracklet buffers array");
  mTrackletsDeviceArray = allocSlotArray<Tracklet*>(MaxLinks);
}

template <int NLayers>
void TimeFrameGPU<NLayers>::createTrackletsBuffers(const int layer, size_t capacity)
{
  GPUTimer timer(mGpuStreams[layer], "creating tracklet buffers", layer);
  mNTracklets[layer] = 0;
  createSlot(mTrackletsDevice, mTrackletsDeviceArray, layer, capacity, "tracklets buffer", SlotInit::Raw, o2::gpu::GPUMemoryResource::MEMORY_GPU | o2::gpu::GPUMemoryResource::MEMORY_STACK);
}

template <int NLayers>
void TimeFrameGPU<NLayers>::createNeighboursLUTDevice(const int layer, const unsigned int nCells)
{
  GPUTimer timer(mGpuStreams[layer], "reserving neighboursLUT");
  createSlot(mNeighboursLUTDevice, mNeighboursCellLUTDeviceArray, layer, nCells + 1, "neighbours LUT", SlotInit::Zero, o2::gpu::GPUMemoryResource::MEMORY_GPU | o2::gpu::GPUMemoryResource::MEMORY_STACK);
}

template <int NLayers>
void TimeFrameGPU<NLayers>::createCellsLUTDeviceArray()
{
  GPUTimer timer("creating cells LUTs array");
  mCellsLUTDeviceArray = allocSlotArray<int*>(MaxCells);
  mNeighboursCellLUTDeviceArray = allocSlotArray<int*>(MaxCells);
}

template <int NLayers>
void TimeFrameGPU<NLayers>::createCellsLUTDevice(const int layer)
{
  GPUTimer timer(mGpuStreams[layer], "creating cells LUTs", layer);
  const int firstLink = this->mTrackingTopologyView.getCell(layer).firstLink;
  createSlot(mCellsLUTDevice, mCellsLUTDeviceArray, layer, mNTracklets[firstLink] + 1, "cells LUT", SlotInit::Zero, o2::gpu::GPUMemoryResource::MEMORY_GPU | o2::gpu::GPUMemoryResource::MEMORY_STACK);
}

template <int NLayers>
void TimeFrameGPU<NLayers>::createCellsBuffersArray()
{
  GPUTimer timer("creating cells buffers array");
  mCellsDeviceArray = allocSlotArray<CellSeed*>(MaxCells);
  mNeighboursDeviceArray = allocSlotArray<CellNeighbour*>(MaxCells);
}

template <int NLayers>
void TimeFrameGPU<NLayers>::createCellsBuffers(const int layer, size_t capacity)
{
  GPUTimer timer(mGpuStreams[layer], "creating cells buffers");
  mNCells[layer] = 0;
  createSlot(mCellsDevice, mCellsDeviceArray, layer, capacity, "cells buffer", SlotInit::Raw, o2::gpu::GPUMemoryResource::MEMORY_GPU | o2::gpu::GPUMemoryResource::MEMORY_STACK);
}

template <int NLayers>
void TimeFrameGPU<NLayers>::createNeighboursDevice(const unsigned int layer, size_t capacity)
{
  GPUTimer timer(mGpuStreams[layer], "reserving neighbours", layer);
  this->mNNeighbours[layer] = 0;
  createSlot(mNeighboursDevice, mNeighboursDeviceArray, layer, capacity, "neighbours buffer", SlotInit::Raw, o2::gpu::GPUMemoryResource::MEMORY_GPU | o2::gpu::GPUMemoryResource::MEMORY_STACK);
}

template <int NLayers>
void TimeFrameGPU<NLayers>::createTrackSeedsDevice(const size_t capacity)
{
  GPUTimer timer("reserving track seeds");
  GPULog("gpu-allocation: reserving {} track seeds, for {:.2f} MB.", capacity, capacity * sizeof(TrackSeedN) / constants::MB);
  mTrackSeedsDevice = allocDevice<TrackSeedN>(capacity, o2::gpu::GPUMemoryResource::MEMORY_GPU | o2::gpu::GPUMemoryResource::MEMORY_STACK);
}

template <int NLayers>
void TimeFrameGPU<NLayers>::createTrackITSExtDevice(const size_t capacity)
{
  GPUTimer timer("reserving tracks");
  GPULog("gpu-allocation: reserving {} tracks, for {:.2f} MB.", capacity, capacity * sizeof(o2::its::TrackITSExt) / constants::MB);
  mTrackITSExtDevice = allocDevice<o2::its::TrackITSExt>(capacity, o2::gpu::GPUMemoryResource::MEMORY_GPU | o2::gpu::GPUMemoryResource::MEMORY_STACK);
  if (capacity > 0) {
    GPUChkErrS(cudaMemsetAsync(mTrackITSExtDevice, 0, capacity * sizeof(o2::its::TrackITSExt), Stream::DefaultStream));
  }
  GPULog("gpu-allocation: reserving {} track indices, for {:.2f} MB.", capacity, capacity * sizeof(int) / constants::MB);
  mTrackIndicesDevice = allocDevice<int>(capacity, o2::gpu::GPUMemoryResource::MEMORY_GPU | o2::gpu::GPUMemoryResource::MEMORY_STACK);
  mTrackSeedIndicesDevice = allocDevice<int>(capacity, o2::gpu::GPUMemoryResource::MEMORY_GPU | o2::gpu::GPUMemoryResource::MEMORY_STACK);
  mTrackCounterDevice = allocDevice<int>(1, o2::gpu::GPUMemoryResource::MEMORY_GPU | o2::gpu::GPUMemoryResource::MEMORY_STACK);
}

template <int NLayers>
void TimeFrameGPU<NLayers>::createTrackITSExtHost(const size_t nTracks)
{
  GPUTimer timer("reserving host tracks");
  mNTracks = nTracks;
  mTrackITSExt = bounded_vector<TrackITSExt>(nTracks, {}, this->getMemoryPool().get());
  mTrackIndices = bounded_vector<int>(nTracks, 0, this->getMemoryPool().get());
  std::iota(mTrackIndices.begin(), mTrackIndices.end(), 0);
}

template <int NLayers>
void TimeFrameGPU<NLayers>::createTrackExtensionScratchDevice(const int nThreads, const int maxHypotheses)
{
  GPUTimer timer("reserving track extension scratch");
  using Hypothesis = o2::its::TrackExtensionHypothesis<NLayers>;
  const size_t nHypotheses = static_cast<size_t>(std::max(1, nThreads)) * std::max(1, maxHypotheses);
  GPULog("gpu-allocation: reserving {} track extension hypotheses per scratch buffer, for {:.2f} MB each.", nHypotheses, nHypotheses * sizeof(Hypothesis) / constants::MB);
  mActiveTrackExtensionHypothesesDevice = allocDevice<Hypothesis>(nHypotheses, o2::gpu::GPUMemoryResource::MEMORY_GPU | o2::gpu::GPUMemoryResource::MEMORY_STACK);
  mNextTrackExtensionHypothesesDevice = allocDevice<Hypothesis>(nHypotheses, o2::gpu::GPUMemoryResource::MEMORY_GPU | o2::gpu::GPUMemoryResource::MEMORY_STACK);
}

template <int NLayers>
void TimeFrameGPU<NLayers>::downloadTrackITSExtDevice()
{
  GPUTimer timer("downloading tracks");
  GPULog("gpu-transfer: downloading {} tracks, for {:.2f} MB.", mTrackITSExt.size(), mTrackITSExt.size() * sizeof(o2::its::TrackITSExt) / constants::MB);
  copyFromDevice(mTrackITSExt.data(), mTrackITSExtDevice, mTrackITSExt.size());
}

template <int NLayers>
void TimeFrameGPU<NLayers>::unregisterHostMemory()
{
  GPUTimer timer("unregistering host memory");
  GPULog("unregistering host memory");

  auto unpin = [](auto& pinned, auto& layers, auto& slots) {
    for (auto iLayer{0}; iLayer < NLayers; ++iLayer) {
      if (pinned.test(iLayer)) {
        GPUChkErrS(cudaHostUnregister(layers[iLayer].data()));
      }
    }
    if (pinned.test(NLayers)) {
      GPUChkErrS(cudaHostUnregister(slots.data()));
    }
    pinned.reset();
  };

  unpin(mPinnedUsedClusters, this->mUsedClusters, mUsedClustersDevice);
  unpin(mPinnedUnsortedClusters, this->mUnsortedClusters, mUnsortedClustersDevice);
  unpin(mPinnedClusters, this->mClusters, mClustersDevice);
  unpin(mPinnedClustersIndexTables, this->mIndexTables, mClustersIndexTablesDevice);
  unpin(mPinnedTrackingFrameInfo, this->mTrackingFrameInfo, mTrackingFrameInfoDevice);
  unpin(mPinnedROFramesClusters, this->mROFramesClusters, mROFramesClustersDevice);
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
void TimeFrameGPU<NLayers>::initialise(const TrackingParameters& trkParam, int maxLayers)
{
  mGpuStreams.resize(MaxStreams);
  o2::its::TimeFrame<NLayers>::initialise(trkParam, maxLayers);
}

template <int NLayers>
void TimeFrameGPU<NLayers>::initialise(const TrackingParameters& trkParam, int maxLayers, int iteration)
{
  mGpuStreams.resize(MaxStreams);
  o2::its::TimeFrame<NLayers>::initialise(trkParam, maxLayers, iteration);
  if (iteration != constants::UnusedIndex && iteration < static_cast<int>(mDeviceTrackerTopologyViews.size())) {
    mDeviceTrackingTopologyView = mDeviceTrackerTopologyViews[iteration];
  }
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
void TimeFrameGPU<NLayers>::wipe()
{
  unregisterHostMemory();
  o2::its::TimeFrame<NLayers>::wipe();
}

template class TimeFrameGPU<7>;
// ALICE3 upgrade
#ifdef ENABLE_UPGRADES
template class TimeFrameGPU<11>;
template class TimeFrameGPU<13>;
#endif
} // namespace o2::its::gpu
