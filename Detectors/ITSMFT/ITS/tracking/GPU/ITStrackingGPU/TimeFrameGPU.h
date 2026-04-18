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

template <int NLayers>
class TimeFrameGPU final : public TimeFrame<NLayers>
{
  using typename TimeFrame<NLayers>::IndexTableUtilsN;
  using typename TimeFrame<NLayers>::ROFOverlapTableN;
  using typename TimeFrame<NLayers>::ROFVertexLookupTableN;
  using typename TimeFrame<NLayers>::ROFMaskTableN;
  using typename TimeFrame<NLayers>::TrackSeedN;

 public:
  TimeFrameGPU() = default;
  ~TimeFrameGPU() final = default;

  /// Most relevant operations
  void pushMemoryStack(const int);
  void popMemoryStack(const int);
  void registerHostMemory(const int);
  void unregisterHostMemory(const int);
  void initialise(const int, const TrackingParameters&, const int);
  void loadIndexTableUtils(const int);
  void loadTrackingFrameInfoDevice(const int, const int);
  void createTrackingFrameInfoDeviceArray(const int);
  void loadUnsortedClustersDevice(const int, const int);
  void createUnsortedClustersDeviceArray(const int, const int = NLayers);
  void loadClustersDevice(const int, const int);
  void createClustersDeviceArray(const int, const int = NLayers);
  void loadClustersIndexTables(const int, const int);
  void createClustersIndexTablesArray(const int);
  void createUsedClustersDevice(const int, const int);
  void createUsedClustersDeviceArray(const int, const int = NLayers);
  void loadUsedClustersDevice();
  void loadROFrameClustersDevice(const int, const int);
  void createROFrameClustersDeviceArray(const int);
  void loadROFCutMask(const int);
  void loadVertices(const int);
  void loadROFOverlapTable(const int);
  void loadROFVertexLookupTable(const int);
  void updateROFVertexLookupTable(const int);

  ///
  void createTrackletsLUTDevice(const int, const int);
  void createTrackletsLUTDeviceArray(const int);
  void loadTrackletsDevice();
  void loadTrackletsLUTDevice();
  void loadCellsDevice();
  void loadCellsLUTDevice();
  void loadTrackSeedsDevice();
  void loadTrackSeedsChi2Device();
  void loadTrackSeedsDevice(bounded_vector<TrackSeedN>&);
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
  void createTrackITSExtDevice(const size_t);
  void downloadTrackITSExtDevice();
  void downloadCellsNeighboursDevice(std::vector<bounded_vector<std::pair<int, int>>>&, const int);
  void downloadNeighboursLUTDevice(bounded_vector<int>&, const int);
  void downloadCellsDevice();
  void downloadCellsLUTDevice();

  /// synchronization
  auto& getStream(const size_t stream) { return mGpuStreams[stream]; }
  auto& getStreams() { return mGpuStreams; }
  void syncStream(const size_t stream);
  void syncStreams(const bool = true);
  void waitEvent(const int, const int);
  void recordEvent(const int);
  void recordEvents(const int = 0, const int = NLayers);

  /// cleanup
  virtual void wipe() final;

  /// interface
  virtual bool isGPU() const noexcept final { return true; }
  virtual const char* getName() const noexcept { return "GPU"; }
  IndexTableUtilsN* getDeviceIndexTableUtils() { return mIndexTableUtilsDevice; }
  const auto getDeviceROFOverlapTableView() { return mDeviceROFOverlapTableView; }
  const auto getDeviceROFVertexLookupTableView() { return mDeviceROFVertexLookupTableView; }
  const auto getDeviceROFMaskTableView() { return mDeviceROFMaskTableView; }
  int* getDeviceROFramesClusters(const int layer) { return mROFramesClustersDevice[layer]; }
  auto& getTrackITSExt() { return mTrackITSExt; }
  Vertex* getDeviceVertices() { return mPrimaryVerticesDevice; }
  int* getDeviceROFramesPV() { return mROFramesPVDevice; }
  unsigned char* getDeviceUsedClusters(const int);
  const o2::base::Propagator* getChainPropagator();

  // Hybrid
  TrackITSExt* getDeviceTrackITSExt() { return mTrackITSExtDevice; }
  int* getDeviceNeighboursLUT(const int layer) { return mNeighboursLUTDevice[layer]; }
  gsl::span<int*> getDeviceNeighboursLUTs() { return mNeighboursLUTDevice; }
  gpuPair<int, int>* getDeviceNeighbourPairs(const int layer) { return mNeighbourPairsDevice[layer]; }
  std::array<int*, NLayers - 2>& getDeviceNeighboursAll() { return mNeighboursDevice; }
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
  CellSeed** getDeviceArrayCells() { return mCellsDeviceArray; }
  TrackSeedN* getDeviceTrackSeeds() { return mTrackSeedsDevice; }
  int* getDeviceTrackSeedsLUT() { return mTrackSeedsLUTDevice; }
  auto getNTrackSeeds() const { return mNTracks; }
  o2::track::TrackParCovF** getDeviceArrayTrackSeeds() { return mCellSeedsDeviceArray; }
  float** getDeviceArrayTrackSeedsChi2() { return mCellSeedsChi2DeviceArray; }
  int* getDeviceNeighboursIndexTables(const int layer) { return mNeighboursIndexTablesDevice[layer]; }

  void setDevicePropagator(const o2::base::PropagatorImpl<float>* p) final { this->mPropagatorDevice = p; }

  // Host-specific getters
  gsl::span<int, NLayers - 1> getNTracklets() { return mNTracklets; }
  gsl::span<int, NLayers - 2> getNCells() { return mNCells; }
  auto& getArrayNCells() { return mNCells; }
  gsl::span<int, NLayers - 3> getNNeighbours() { return mNNeighbours; }
  auto& getArrayNNeighbours() { return mNNeighbours; }

  // Host-available device getters
  gsl::span<int*> getDeviceTrackletsLUTs() { return mTrackletsLUTDevice; }
  gsl::span<int*> getDeviceCellLUTs() { return mCellsLUTDevice; }
  gsl::span<Tracklet*> getDeviceTracklets() { return mTrackletsDevice; }
  gsl::span<CellSeed*> getDeviceCells() { return mCellsDevice; }

  // Overridden getters
  size_t getNumberOfTracklets() const final;
  size_t getNumberOfCells() const final;
  size_t getNumberOfNeighbours() const final;

 private:
  void allocMemAsync(void**, size_t, Stream&, bool, int32_t = o2::gpu::GPUMemoryResource::MEMORY_GPU); // Abstract owned and unowned memory allocations on specific stream
  void allocMem(void**, size_t, bool, int32_t = o2::gpu::GPUMemoryResource::MEMORY_GPU);               // Abstract owned and unowned memory allocations on default stream

  // Host-available device buffer sizes
  std::array<int, NLayers - 1> mNTracklets;
  std::array<int, NLayers - 2> mNCells;
  std::array<int, NLayers - 3> mNNeighbours;

  // Device pointers
  IndexTableUtilsN* mIndexTableUtilsDevice;
  // device navigation views
  ROFOverlapTableN::View mDeviceROFOverlapTableView;
  ROFVertexLookupTableN::View mDeviceROFVertexLookupTableView;
  ROFMaskTableN::View mDeviceROFMaskTableView;

  // Hybrid pref
  Vertex* mPrimaryVerticesDevice;
  int* mROFramesPVDevice;
  std::array<Cluster*, NLayers> mClustersDevice;
  std::array<Cluster*, NLayers> mUnsortedClustersDevice;
  std::array<int*, NLayers> mClustersIndexTablesDevice;
  std::array<unsigned char*, NLayers> mUsedClustersDevice;
  std::array<int*, NLayers> mROFramesClustersDevice;
  const Cluster** mClustersDeviceArray;
  const Cluster** mUnsortedClustersDeviceArray;
  const int** mClustersIndexTablesDeviceArray;
  uint8_t** mUsedClustersDeviceArray;
  const int** mROFramesClustersDeviceArray;
  std::array<Tracklet*, NLayers - 1> mTrackletsDevice;
  std::array<int*, NLayers - 1> mTrackletsLUTDevice;
  std::array<int*, NLayers - 2> mCellsLUTDevice;
  std::array<int*, NLayers - 3> mNeighboursLUTDevice;

  Tracklet** mTrackletsDeviceArray{nullptr};
  int** mCellsLUTDeviceArray{nullptr};
  int** mNeighboursCellDeviceArray{nullptr};
  int** mNeighboursCellLUTDeviceArray{nullptr};
  int** mTrackletsLUTDeviceArray{nullptr};
  std::array<CellSeed*, NLayers - 2> mCellsDevice;
  CellSeed** mCellsDeviceArray;
  std::array<int*, NLayers - 3> mNeighboursIndexTablesDevice;
  TrackSeedN* mTrackSeedsDevice{nullptr};
  int* mTrackSeedsLUTDevice{nullptr};
  unsigned int mNTracks{0};
  std::array<o2::track::TrackParCovF*, NLayers - 2> mCellSeedsDevice;
  o2::track::TrackParCovF** mCellSeedsDeviceArray;
  std::array<float*, NLayers - 2> mCellSeedsChi2Device;
  float** mCellSeedsChi2DeviceArray;

  TrackITSExt* mTrackITSExtDevice;
  std::array<gpuPair<int, int>*, NLayers - 2> mNeighbourPairsDevice;
  std::array<int*, NLayers - 2> mNeighboursDevice;
  std::array<TrackingFrameInfo*, NLayers> mTrackingFrameInfoDevice;
  const TrackingFrameInfo** mTrackingFrameInfoDeviceArray;

  // State
  Streams mGpuStreams;
  std::bitset<NLayers + 1> mPinnedUnsortedClusters{0};
  std::bitset<NLayers + 1> mPinnedClusters{0};
  std::bitset<NLayers + 1> mPinnedClustersIndexTables{0};
  std::bitset<NLayers + 1> mPinnedUsedClusters{0};
  std::bitset<NLayers + 1> mPinnedROFramesClusters{0};
  std::bitset<NLayers + 1> mPinnedTrackingFrameInfo{0};

  // Temporary buffer for storing output tracks from GPU tracking
  bounded_vector<TrackITSExt> mTrackITSExt;
};

template <int NLayers>
inline std::vector<unsigned int> TimeFrameGPU<NLayers>::getClusterSizes()
{
  std::vector<unsigned int> sizes(this->mUnsortedClusters.size());
  std::transform(this->mUnsortedClusters.begin(), this->mUnsortedClusters.end(), sizes.begin(),
                 [](const auto& v) { return static_cast<unsigned int>(v.size()); });
  return sizes;
}

template <int NLayers>
inline size_t TimeFrameGPU<NLayers>::getNumberOfTracklets() const
{
  return std::accumulate(mNTracklets.begin(), mNTracklets.end(), 0);
}

template <int NLayers>
inline size_t TimeFrameGPU<NLayers>::getNumberOfCells() const
{
  return std::accumulate(mNCells.begin(), mNCells.end(), 0);
}

template <int NLayers>
inline size_t TimeFrameGPU<NLayers>::getNumberOfNeighbours() const
{
  return std::accumulate(mNNeighbours.begin(), mNNeighbours.end(), 0);
}

} // namespace o2::its::gpu

#endif
