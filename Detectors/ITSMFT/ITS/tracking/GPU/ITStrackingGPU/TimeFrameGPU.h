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

#include "ITSMFTTracking/BoundedAllocator.h"
#include "ITStracking/TimeFrame.h"
#include "ITStracking/Configuration.h"
#include "ITStracking/TrackExtensionHypothesis.h"
#include "ITStrackingGPU/Utils.h"

namespace o2::its::gpu
{

template <int NLayers>
class TimeFrameGPU : public TimeFrame<NLayers>
{
  using typename TimeFrame<NLayers>::IndexTableUtilsN;
  using typename TimeFrame<NLayers>::ROFOverlapTableN;
  using typename TimeFrame<NLayers>::ROFVertexLookupTableN;
  using typename TimeFrame<NLayers>::ROFMaskTableN;
  using typename TimeFrame<NLayers>::TrackingTopologyN;
  using typename TimeFrame<NLayers>::TrackSeedN;
  static constexpr int MaxLinks = TrackingTopologyN::MaxLinks;
  static constexpr int MaxCells = TrackingTopologyN::MaxCells;
  static constexpr int MaxStreams = MaxCells > NLayers ? MaxCells : NLayers;

 public:
  TimeFrameGPU() = default;
  ~TimeFrameGPU() override = default;

  /// Most relevant operations
  void pushMemoryStack(const int);
  void popMemoryStack(const int);
  void unregisterHostMemory();
  void initialise(const TrackingParameters&, int maxLayers);
  void initialise(const TrackingParameters&, int maxLayers, int iteration);
  void loadIndexTableUtils();
  void loadTrackingTopologies();
  void loadTrackingFrameInfoDevice(const int);
  void createTrackingFrameInfoDeviceArray(const int = NLayers);
  void loadUnsortedClustersDevice(const int);
  void createUnsortedClustersDeviceArray(const int = NLayers);
  void loadClustersDevice(const int);
  void createClustersDeviceArray(const int = NLayers);
  void loadClustersIndexTables(const int);
  void createClustersIndexTablesArray(const int = NLayers);
  void createUsedClustersDevice(const int);
  void createUsedClustersDeviceArray(const int = NLayers);
  void loadUsedClustersDevice();
  void loadROFrameClustersDevice(const int);
  void createROFrameClustersDeviceArray(const int = NLayers);
  void loadROFCutMask(const int);
  void loadVertices();
  void loadROFOverlapTable();
  void loadROFVertexLookupTable();
  void uploadROFVertexLookupTable();
  void loadIterationParameters(const TrackingParameters&);

  ///
  void createTrackletsLUTDevice(bool, const int);
  void createTrackletsLUTDeviceArray();
  void createTrackSeedsDevice(const size_t capacity);
  void createTrackletsBuffers(const int, size_t capacity);
  void createTrackletsBuffersArray();
  void createCellsBuffers(const int, size_t capacity);
  void createCellsBuffersArray();
  void createCellsLUTDevice(const int);
  void createCellsLUTDeviceArray();
  void createNeighboursDevice(const unsigned int layer, size_t capacity);
  void createNeighboursLUTDevice(const int, const unsigned int);
  void createTrackITSExtDevice(const size_t capacity);
  void createTrackITSExtHost(const size_t nTracks);
  void createTrackExtensionScratchDevice(const int nThreads, const int maxHypotheses);
  void downloadTrackITSExtDevice();

  /// synchronization
  auto& getStream(const size_t stream) { return mGpuStreams[stream]; }
  auto& getStreams() { return mGpuStreams; }
  void syncStreams(const bool = true);
  void waitEvent(const int, const int);
  void recordEvent(const int);

  /// cleanup
  virtual void wipe() final;

  /// interface
  virtual bool isGPU() const noexcept final { return true; }
  virtual const char* getName() const noexcept override final { return "GPU"; }
  IndexTableUtilsN* getDeviceIndexTableUtils() { return mIndexTableUtilsDevice; }
  const float* getDeviceLayerRadii() const { return mLayerRadiiDevice; }
  const float* getDeviceMinPts() const { return mMinPtsDevice; }
  const float* getDeviceLayerxX0() const { return mLayerxX0Device; }
  const auto getDeviceROFOverlapTableView() { return mDeviceROFOverlapTableView; }
  const auto getDeviceROFVertexLookupTableView() { return mDeviceROFVertexLookupTableView; }
  const auto getDeviceROFMaskTableView() { return mDeviceROFMaskTableView; }
  const auto getDeviceTrackingTopologyView() const { return mDeviceTrackingTopologyView; }
  auto& getTrackITSExt() { return mTrackITSExt; }
  auto& getTrackIndices() { return mTrackIndices; }
  Vertex* getDeviceVertices() { return mPrimaryVerticesDevice; }

  // Hybrid
  TrackITSExt* getDeviceTrackITSExt() { return mTrackITSExtDevice; }
  int* getDeviceTrackIndices() { return mTrackIndicesDevice; }
  TrackExtensionHypothesis<NLayers>* getDeviceActiveTrackExtensionHypotheses() { return mActiveTrackExtensionHypothesesDevice; }
  TrackExtensionHypothesis<NLayers>* getDeviceNextTrackExtensionHypotheses() { return mNextTrackExtensionHypothesesDevice; }
  int* getDeviceNeighboursLUT(const int layer) { return mNeighboursLUTDevice[layer]; }
  CellNeighbour** getDeviceArrayNeighbours() { return mNeighboursDeviceArray; }
  CellNeighbour* getDeviceNeighbours(const int layer) { return mNeighboursDevice[layer]; }
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
  int* getDeviceTrackSeedIndices() { return mTrackSeedIndicesDevice; }
  int* getDeviceTrackCounter() { return mTrackCounterDevice; }
  auto getNTrackSeeds() const { return mNTracks; }

  void setDevicePropagator(const o2::base::PropagatorImpl<float>* p) final { this->mPropagatorDevice = p; }

  // Host-specific getters
  gsl::span<int> getNTracklets() { return {mNTracklets.data(), static_cast<gsl::span<int>::size_type>(this->mTrackingTopologyView.nLinks)}; }
  gsl::span<int> getNCells() { return {mNCells.data(), static_cast<gsl::span<int>::size_type>(this->mTrackingTopologyView.nCells)}; }
  auto& getArrayNCells() { return mNCells; }
  gsl::span<int> getNNeighbours() { return {mNNeighbours.data(), static_cast<gsl::span<int>::size_type>(this->mTrackingTopologyView.nCells)}; }

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
  enum class SlotInit {
    Raw, ///< whatever the allocator handed back
    Zero ///< cleared on the slot's stream
  };

  template <typename T>
  T* allocDevice(size_t n, int32_t type = o2::gpu::GPUMemoryResource::MEMORY_GPU);
  template <typename T>
  T* allocDeviceAsync(size_t n, Stream&, int32_t type = o2::gpu::GPUMemoryResource::MEMORY_GPU);
  template <typename SlotPtr>
  SlotPtr* allocSlotArray(size_t n);
  template <typename T>
  void copyToDevice(T* dst, const T* src, size_t n);
  template <typename T>
  void copyFromDevice(T* dst, const T* src, size_t n);
  template <typename T, typename ArrayT>
  void publishSlot(ArrayT deviceArray, int slot, T* const& devicePtr, Stream&);
  template <typename T, size_t N, typename ArrayT>
  T* createSlot(std::array<T*, N>& slots, ArrayT deviceArray, int slot, size_t n, const char* what, SlotInit init = SlotInit::Raw, int32_t type = o2::gpu::GPUMemoryResource::MEMORY_GPU);
  template <typename T, size_t N, typename ArrayT, typename Container>
  void uploadSlot(std::array<T*, N>& slots, ArrayT deviceArray, int slot, const Container& host, const char* what);
  template <typename ArrayT, typename T, size_t N>
  void createPinnedSlotArray(ArrayT& deviceArray, std::array<T*, N>& slots, std::bitset<NLayers + 1>& pinned);
  template <typename Layers>
  void pinHostLayers(Layers& layers, std::bitset<NLayers + 1>& pinned, int maxLayers);
  template <typename Table>
  typename Table::View uploadNavigationTable(const Table& table, const typename Table::View& hostView);

  // Host-available device buffer sizes
  std::array<int, MaxLinks> mNTracklets{};
  std::array<int, MaxCells> mNCells{};
  std::array<int, MaxCells> mNNeighbours{};

  // Device pointers
  IndexTableUtilsN* mIndexTableUtilsDevice{nullptr};
  float* mIterationParametersDevice{nullptr};
  const float* mLayerRadiiDevice{nullptr};
  const float* mMinPtsDevice{nullptr};
  const float* mLayerxX0Device{nullptr};
  // device navigation views
  ROFOverlapTableN::View mDeviceROFOverlapTableView;
  ROFVertexLookupTableN::View mDeviceROFVertexLookupTableView;
  ROFMaskTableN::View mDeviceROFMaskTableView;
  std::vector<typename TrackingTopologyN::View> mDeviceTrackerTopologyViews;
  typename TrackingTopologyN::View mDeviceTrackingTopologyView;

  // Hybrid pref
  Vertex* mPrimaryVerticesDevice{nullptr};
  std::array<Cluster*, NLayers> mClustersDevice{};
  std::array<Cluster*, NLayers> mUnsortedClustersDevice{};
  std::array<int*, NLayers> mClustersIndexTablesDevice{};
  std::array<unsigned char*, NLayers> mUsedClustersDevice{};
  std::array<int*, NLayers> mROFramesClustersDevice{};
  const Cluster** mClustersDeviceArray{nullptr};
  const Cluster** mUnsortedClustersDeviceArray{nullptr};
  const int** mClustersIndexTablesDeviceArray{nullptr};
  uint8_t** mUsedClustersDeviceArray{nullptr};
  const int** mROFramesClustersDeviceArray{nullptr};
  std::array<Tracklet*, MaxLinks> mTrackletsDevice{};
  std::array<int*, MaxLinks> mTrackletsLUTDevice{};
  std::array<int*, MaxCells> mCellsLUTDevice{};
  std::array<int*, MaxCells> mNeighboursLUTDevice{};

  Tracklet** mTrackletsDeviceArray{nullptr};
  int** mCellsLUTDeviceArray{nullptr};
  int** mNeighboursCellLUTDeviceArray{nullptr};
  int** mTrackletsLUTDeviceArray{nullptr};
  std::array<CellSeed*, MaxCells> mCellsDevice{};
  CellSeed** mCellsDeviceArray{nullptr};
  TrackSeedN* mTrackSeedsDevice{nullptr};
  int* mTrackSeedIndicesDevice{nullptr}; ///< which seed each emitted track was fitted from
  int* mTrackCounterDevice{nullptr};
  unsigned int mNTracks{0};

  TrackITSExt* mTrackITSExtDevice{nullptr};
  int* mTrackIndicesDevice{nullptr};
  TrackExtensionHypothesis<NLayers>* mActiveTrackExtensionHypothesesDevice{nullptr};
  TrackExtensionHypothesis<NLayers>* mNextTrackExtensionHypothesesDevice{nullptr};
  std::array<CellNeighbour*, MaxCells> mNeighboursDevice{};
  CellNeighbour** mNeighboursDeviceArray{nullptr};
  std::array<TrackingFrameInfo*, NLayers> mTrackingFrameInfoDevice{};
  const TrackingFrameInfo** mTrackingFrameInfoDeviceArray{nullptr};

  // State
  Streams mGpuStreams;
  std::bitset<NLayers + 1> mPinnedUnsortedClusters{0};
  std::bitset<NLayers + 1> mPinnedClusters{0};
  std::bitset<NLayers + 1> mPinnedClustersIndexTables{0};
  std::bitset<NLayers + 1> mPinnedUsedClusters{0};
  std::bitset<NLayers + 1> mPinnedROFramesClusters{0};
  std::bitset<NLayers + 1> mPinnedTrackingFrameInfo{0};

  // Temporary buffer for storing output tracks from GPU tracking
  o2::itsmft::tracking::bounded_vector<TrackITSExt> mTrackITSExt;
  o2::itsmft::tracking::bounded_vector<int> mTrackIndices;
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
  return std::accumulate(mNTracklets.begin(), mNTracklets.begin() + this->mTrackingTopologyView.nLinks, 0);
}

template <int NLayers>
inline size_t TimeFrameGPU<NLayers>::getNumberOfCells() const
{
  return std::accumulate(mNCells.begin(), mNCells.begin() + this->mTrackingTopologyView.nCells, 0);
}

template <int NLayers>
inline size_t TimeFrameGPU<NLayers>::getNumberOfNeighbours() const
{
  return std::accumulate(mNNeighbours.begin(), mNNeighbours.begin() + this->mTrackingTopologyView.nCells, 0);
}

} // namespace o2::its::gpu

#endif
