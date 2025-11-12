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

#ifndef TRACKINGITSGPU_INCLUDE_TIMEFRAMEGPU_H
#define TRACKINGITSGPU_INCLUDE_TIMEFRAMEGPU_H

#include <gsl/gsl>
#include <bitset>

#include "ITStracking/BoundedAllocator.h"
#include "ITStracking/TimeFrame.h"
#include "ITStracking/Configuration.h"
#include "ITStrackingGPU/Utils.h"

namespace o2::its::gpu
{

template <int nLayers = 7>
class TimeFrameGPU final : public TimeFrame<nLayers>
{
  using typename TimeFrame<nLayers>::CellSeedN;
  using typename TimeFrame<nLayers>::IndexTableUtilsN;

 public:
  TimeFrameGPU() = default;
  ~TimeFrameGPU() = default;

  /// Most relevant operations
  void pushMemoryStack(const int);
  void popMemoryStack(const int);
  void registerHostMemory(const int);
  void unregisterHostMemory(const int);
  void initialise(const int, const TrackingParameters&, const int, IndexTableUtilsN* utils = nullptr, const TimeFrameGPUParameters* pars = nullptr);
  void initDevice(IndexTableUtilsN*, const TrackingParameters& trkParam, const TimeFrameGPUParameters&, const int, const int);
  void initDeviceSAFitting();
  void loadIndexTableUtils(const int);
  void loadTrackingFrameInfoDevice(const int, const int);
  void createTrackingFrameInfoDeviceArray(const int);
  void loadUnsortedClustersDevice(const int, const int);
  void createUnsortedClustersDeviceArray(const int, const int = nLayers);
  void loadClustersDevice(const int, const int);
  void createClustersDeviceArray(const int, const int = nLayers);
  void loadClustersIndexTables(const int, const int);
  void createClustersIndexTablesArray(const int);
  void createUsedClustersDevice(const int, const int);
  void createUsedClustersDeviceArray(const int, const int = nLayers);
  void loadUsedClustersDevice();
  void loadROFrameClustersDevice(const int, const int);
  void createROFrameClustersDeviceArray(const int);
  void loadMultiplicityCutMask(const int);
  void loadVertices(const int);

  ///
  void createTrackletsLUTDevice(const int, const int);
  void createTrackletsLUTDeviceArray(const int);
  void loadTrackletsDevice();
  void loadTrackletsLUTDevice();
  void loadCellsDevice();
  void loadCellsLUTDevice();
  void loadTrackSeedsDevice();
  void loadTrackSeedsChi2Device();
  void loadRoadsDevice();
  void loadTrackSeedsDevice(bounded_vector<CellSeedN>&);
  void createTrackletsBuffers(const int);
  void createTrackletsBuffersArray(const int);
  void createCellsBuffers(const int);
  void createCellsBuffersArray(const int);
  void createCellsDevice();
  void createCellsLUTDevice(const int);
  void createCellsLUTDeviceArray(const int);
  void createNeighboursIndexTablesDevice(const int);
  void createNeighboursDevice(const unsigned int layer);
  void createNeighboursLUTDevice(const int, const unsigned int);
  void createTrackITSExtDevice(bounded_vector<CellSeedN>&);
  void downloadTrackITSExtDevice(bounded_vector<CellSeedN>&);
  void downloadCellsNeighboursDevice(std::vector<bounded_vector<std::pair<int, int>>>&, const int);
  void downloadNeighboursLUTDevice(bounded_vector<int>&, const int);
  void downloadCellsDevice();
  void downloadCellsLUTDevice();

  /// Vertexer
  void createVtxTrackletsLUTDevice(const int32_t);
  void createVtxTrackletsBuffers(const int32_t);
  void createVtxLinesLUTDevice(const int32_t);
  void createVtxLinesBuffer(const int32_t);

  /// synchronization
  auto& getStream(const size_t stream) { return mGpuStreams[stream]; }
  auto& getStreams() { return mGpuStreams; }
  void syncStream(const size_t stream);
  void syncStreams(const bool = true);
  void waitEvent(const int, const int);
  void recordEvent(const int);
  void recordEvents(const int = 0, const int = nLayers);

  /// cleanup
  virtual void wipe() final;

  /// interface
  virtual bool isGPU() const noexcept final { return true; }
  virtual const char* getName() const noexcept { return "GPU"; }
  int getNClustersInRofSpan(const int, const int, const int) const;
  IndexTableUtilsN* getDeviceIndexTableUtils() { return mIndexTableUtilsDevice; }
  int* getDeviceROFramesClusters(const int layer) { return mROFramesClustersDevice[layer]; }
  auto& getTrackITSExt() { return mTrackITSExt; }
  Vertex* getDeviceVertices() { return mPrimaryVerticesDevice; }
  int* getDeviceROFramesPV() { return mROFramesPVDevice; }
  unsigned char* getDeviceUsedClusters(const int);
  const o2::base::Propagator* getChainPropagator();

  // Hybrid
  Road<nLayers - 2>* getDeviceRoads() { return mRoadsDevice; }
  TrackITSExt* getDeviceTrackITSExt() { return mTrackITSExtDevice; }
  int* getDeviceNeighboursLUT(const int layer) { return mNeighboursLUTDevice[layer]; }
  gsl::span<int*> getDeviceNeighboursLUTs() { return mNeighboursLUTDevice; }
  gpuPair<int, int>* getDeviceNeighbourPairs(const int layer) { return mNeighbourPairsDevice[layer]; }
  std::array<int*, nLayers - 2>& getDeviceNeighboursAll() { return mNeighboursDevice; }
  int* getDeviceNeighbours(const int layer) { return mNeighboursDevice[layer]; }
  int** getDeviceNeighboursArray() { return mNeighboursDevice.data(); }
  TrackingFrameInfo* getDeviceTrackingFrameInfo(const int);
  const TrackingFrameInfo** getDeviceArrayTrackingFrameInfo() const { return mTrackingFrameInfoDeviceArray; }
  const Cluster** getDeviceArrayClusters() const { return mClustersDeviceArray; }
  const Cluster** getDeviceArrayUnsortedClusters() const { return mUnsortedClustersDeviceArray; }
  const int** getDeviceArrayClustersIndexTables() const { return mClustersIndexTablesDeviceArray; }
  std::vector<unsigned int> getClusterSizes();
  uint8_t** getDeviceArrayUsedClusters() const { return mUsedClustersDeviceArray; }
  const int** getDeviceROFrameClusters() const { return mROFramesClustersDeviceArray; }
  Tracklet** getDeviceArrayTracklets() { return mTrackletsDeviceArray; }
  int** getDeviceArrayTrackletsLUT() const { return mTrackletsLUTDeviceArray; }
  int** getDeviceArrayCellsLUT() const { return mCellsLUTDeviceArray; }
  int** getDeviceArrayNeighboursCellLUT() const { return mNeighboursCellLUTDeviceArray; }
  CellSeedN** getDeviceArrayCells() { return mCellsDeviceArray; }
  CellSeedN* getDeviceTrackSeeds() { return mTrackSeedsDevice; }
  o2::track::TrackParCovF** getDeviceArrayTrackSeeds() { return mCellSeedsDeviceArray; }
  float** getDeviceArrayTrackSeedsChi2() { return mCellSeedsChi2DeviceArray; }
  int* getDeviceNeighboursIndexTables(const int layer) { return mNeighboursIndexTablesDevice[layer]; }
  uint8_t* getDeviceMultCutMask() { return mMultMaskDevice; }

  // Vertexer
  auto& getDeviceNTrackletsPerROF() const noexcept { return mNTrackletsPerROFDevice; }
  auto& getDeviceNTrackletsPerCluster() const noexcept { return mNTrackletsPerClusterDevice; }
  auto& getDeviceNTrackletsPerClusterSum() const noexcept { return mNTrackletsPerClusterSumDevice; }
  int32_t** getDeviceArrayNTrackletsPerROF() const noexcept { return mNTrackletsPerROFDeviceArray; }
  int32_t** getDeviceArrayNTrackletsPerCluster() const noexcept { return mNTrackletsPerClusterDeviceArray; }
  int32_t** getDeviceArrayNTrackletsPerClusterSum() const noexcept { return mNTrackletsPerClusterSumDeviceArray; }
  uint8_t* getDeviceUsedTracklets() const noexcept { return mUsedTrackletsDevice; }
  int32_t* getDeviceNLinesPerCluster() const noexcept { return mNLinesPerClusterDevice; }
  int32_t* getDeviceNLinesPerClusterSum() const noexcept { return mNLinesPerClusterSumDevice; }
  Line* getDeviceLines() const noexcept { return mLinesDevice; }
  gsl::span<int*> getDeviceTrackletsPerROFs() { return mNTrackletsPerROFDevice; }

  void setDevicePropagator(const o2::base::PropagatorImpl<float>* p) final { this->mPropagatorDevice = p; }

  // Host-specific getters
  gsl::span<int, nLayers - 1> getNTracklets() { return mNTracklets; }
  gsl::span<int, nLayers - 2> getNCells() { return mNCells; }
  auto& getArrayNCells() { return mNCells; }
  gsl::span<int, nLayers - 3> getNNeighbours() { return mNNeighbours; }
  auto& getArrayNNeighbours() { return mNNeighbours; }

  // Host-available device getters
  gsl::span<int*> getDeviceTrackletsLUTs() { return mTrackletsLUTDevice; }
  gsl::span<int*> getDeviceCellLUTs() { return mCellsLUTDevice; }
  gsl::span<Tracklet*> getDeviceTracklets() { return mTrackletsDevice; }
  gsl::span<CellSeedN*> getDeviceCells() { return mCellsDevice; }

  // Overridden getters
  int getNumberOfTracklets() const final;
  int getNumberOfCells() const final;
  int getNumberOfNeighbours() const final;

 private:
  void allocMemAsync(void**, size_t, Stream&, bool, int32_t = o2::gpu::GPUMemoryResource::MEMORY_GPU); // Abstract owned and unowned memory allocations on specific stream
  void allocMem(void**, size_t, bool, int32_t = o2::gpu::GPUMemoryResource::MEMORY_GPU);               // Abstract owned and unowned memory allocations on default stream
  TimeFrameGPUParameters mGpuParams;

  // Host-available device buffer sizes
  std::array<int, nLayers - 1> mNTracklets;
  std::array<int, nLayers - 2> mNCells;
  std::array<int, nLayers - 3> mNNeighbours;

  // Device pointers
  IndexTableUtilsN* mIndexTableUtilsDevice;

  // Hybrid pref
  uint8_t* mMultMaskDevice;
  Vertex* mPrimaryVerticesDevice;
  int* mROFramesPVDevice;
  std::array<Cluster*, nLayers> mClustersDevice;
  std::array<Cluster*, nLayers> mUnsortedClustersDevice;
  std::array<int*, nLayers> mClustersIndexTablesDevice;
  std::array<unsigned char*, nLayers> mUsedClustersDevice;
  std::array<int*, nLayers> mROFramesClustersDevice;
  const Cluster** mClustersDeviceArray;
  const Cluster** mUnsortedClustersDeviceArray;
  const int** mClustersIndexTablesDeviceArray;
  uint8_t** mUsedClustersDeviceArray;
  const int** mROFramesClustersDeviceArray;
  std::array<Tracklet*, nLayers - 1> mTrackletsDevice;
  std::array<int*, nLayers - 1> mTrackletsLUTDevice;
  std::array<int*, nLayers - 2> mCellsLUTDevice;
  std::array<int*, nLayers - 3> mNeighboursLUTDevice;

  Tracklet** mTrackletsDeviceArray{nullptr};
  int** mCellsLUTDeviceArray{nullptr};
  int** mNeighboursCellDeviceArray{nullptr};
  int** mNeighboursCellLUTDeviceArray{nullptr};
  int** mTrackletsLUTDeviceArray{nullptr};
  std::array<CellSeedN*, nLayers - 2> mCellsDevice;
  CellSeedN** mCellsDeviceArray;
  std::array<int*, nLayers - 3> mNeighboursIndexTablesDevice;
  CellSeedN* mTrackSeedsDevice{nullptr};
  std::array<o2::track::TrackParCovF*, nLayers - 2> mCellSeedsDevice;
  o2::track::TrackParCovF** mCellSeedsDeviceArray;
  std::array<float*, nLayers - 2> mCellSeedsChi2Device;
  float** mCellSeedsChi2DeviceArray;

  Road<nLayers - 2>* mRoadsDevice;
  TrackITSExt* mTrackITSExtDevice;
  std::array<gpuPair<int, int>*, nLayers - 2> mNeighbourPairsDevice;
  std::array<int*, nLayers - 2> mNeighboursDevice;
  std::array<TrackingFrameInfo*, nLayers> mTrackingFrameInfoDevice;
  const TrackingFrameInfo** mTrackingFrameInfoDeviceArray;

  /// Vertexer
  std::array<int32_t*, 2> mNTrackletsPerROFDevice;
  std::array<int32_t*, 2> mNTrackletsPerClusterDevice;
  std::array<int32_t*, 2> mNTrackletsPerClusterSumDevice;
  uint8_t* mUsedTrackletsDevice;
  int32_t* mNLinesPerClusterDevice;
  int32_t* mNLinesPerClusterSumDevice;
  int32_t** mNTrackletsPerROFDeviceArray;
  int32_t** mNTrackletsPerClusterDeviceArray;
  int32_t** mNTrackletsPerClusterSumDeviceArray;
  Line* mLinesDevice;

  // State
  Streams mGpuStreams;
  std::bitset<nLayers + 1> mPinnedUnsortedClusters{0};
  std::bitset<nLayers + 1> mPinnedClusters{0};
  std::bitset<nLayers + 1> mPinnedClustersIndexTables{0};
  std::bitset<nLayers + 1> mPinnedUsedClusters{0};
  std::bitset<nLayers + 1> mPinnedROFramesClusters{0};
  std::bitset<nLayers + 1> mPinnedTrackingFrameInfo{0};

  // Temporary buffer for storing output tracks from GPU tracking
  bounded_vector<TrackITSExt> mTrackITSExt;
};

template <int nLayers>
inline int TimeFrameGPU<nLayers>::getNClustersInRofSpan(const int rofIdstart, const int rofSpanSize, const int layerId) const
{
  return static_cast<int>(this->mROFramesClusters[layerId][(rofIdstart + rofSpanSize) < this->mROFramesClusters.size() ? rofIdstart + rofSpanSize : this->mROFramesClusters.size() - 1] - this->mROFramesClusters[layerId][rofIdstart]);
}

template <int nLayers>
inline std::vector<unsigned int> TimeFrameGPU<nLayers>::getClusterSizes()
{
  std::vector<unsigned int> sizes(this->mUnsortedClusters.size());
  std::transform(this->mUnsortedClusters.begin(), this->mUnsortedClusters.end(), sizes.begin(),
                 [](const auto& v) { return static_cast<unsigned int>(v.size()); });
  return sizes;
}

template <int nLayers>
inline int TimeFrameGPU<nLayers>::getNumberOfTracklets() const
{
  return std::accumulate(mNTracklets.begin(), mNTracklets.end(), 0);
}

template <int nLayers>
inline int TimeFrameGPU<nLayers>::getNumberOfCells() const
{
  return std::accumulate(mNCells.begin(), mNCells.end(), 0);
}

template <int nLayers>
inline int TimeFrameGPU<nLayers>::getNumberOfNeighbours() const
{
  return std::accumulate(mNNeighbours.begin(), mNNeighbours.end(), 0);
}

} // namespace o2::its::gpu

#endif
