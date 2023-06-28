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

#ifdef __HIPCC__
#include <hip/hip_runtime.h>
#endif

#include "ITStracking/TimeFrame.h"
#include "ITStracking/Configuration.h"

#include "ITStrackingGPU/ClusterLinesGPU.h"
#include "ITStrackingGPU/Array.h"
#include "ITStrackingGPU/Vector.h"
#include "ITStrackingGPU/Stream.h"

#include <gsl/gsl>

#include "GPUCommonDef.h"
#include "GPUCommonMath.h"
#include "GPUCommonLogger.h"

namespace o2
{

namespace its
{

namespace gpu
{

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

enum class Task {
  Tracker = 0,
  Vertexer = 1
};

template <int nLayers>
class GpuTimeFrameChunk
{
 public:
  static size_t computeScalingSizeBytes(const int, const TimeFrameGPUParameters&);
  static size_t computeFixedSizeBytes(const TimeFrameGPUParameters&);
  static size_t computeRofPerChunk(const TimeFrameGPUParameters&, const size_t);

  GpuTimeFrameChunk() = delete;
  GpuTimeFrameChunk(o2::its::TimeFrame* tf, TimeFrameGPUParameters& conf)
  {
    mTimeFramePtr = tf;
    mTFGPUParams = &conf;
  }
  ~GpuTimeFrameChunk();

  /// Most relevant operations
  void allocate(const size_t, Stream&);
  void reset(const Task, Stream&);
  size_t loadDataOnDevice(const size_t, const size_t, const int, Stream&);

  /// Interface
  Cluster* getDeviceClusters(const int);
  TrackingFrameInfo* getDeviceTrackingFrameInfo(const int);
  int* getDeviceClusterExternalIndices(const int);
  int* getDeviceIndexTables(const int);
  Tracklet* getDeviceTracklets(const int);
  int* getDeviceTrackletsLookupTables(const int);
  Cell* getDeviceCells(const int);
  int* getDeviceCellsLookupTables(const int);
  Road<nLayers - 2>* getDeviceRoads() { return mRoadsDevice; }
  int* getDeviceRoadsLookupTables(const int);
  TimeFrameGPUParameters* getTimeFrameGPUParameters() const { return mTFGPUParams; }

  int* getDeviceCUBTmpBuffer() { return mCUBTmpBufferDevice; }
  int* getDeviceFoundTracklets() { return mFoundTrackletsDevice; }
  int* getDeviceNFoundCells() { return mNFoundCellsDevice; }
  int* getDeviceCellNeigboursLookupTables(const int);
  int* getDeviceCellNeighbours(const int);
  Cell** getDeviceArrayCells() const { return mCellsDeviceArray; }
  int** getDeviceArrayNeighboursCell() const { return mNeighboursCellDeviceArray; }
  int** getDeviceArrayNeighboursCellLUT() const { return mNeighboursCellLookupTablesDeviceArray; }

  /// Vertexer only
  int* getDeviceNTrackletCluster(const int combid) { return mNTrackletsPerClusterDevice[combid]; }
  Line* getDeviceLines() { return mLinesDevice; };
  int* getDeviceNFoundLines() { return mNFoundLinesDevice; }
  int* getDeviceNExclusiveFoundLines() { return mNExclusiveFoundLinesDevice; }
  unsigned char* getDeviceUsedTracklets() { return mUsedTrackletsDevice; }
  int* getDeviceClusteredLines() { return mClusteredLinesDevice; }
  size_t getNPopulatedRof() const { return mNPopulatedRof; }

 private:
  /// Host
  std::array<gsl::span<const Cluster>, nLayers> mHostClusters;
  std::array<gsl::span<const int>, nLayers> mHostIndexTables;

  /// Device
  std::array<Cluster*, nLayers> mClustersDevice;
  std::array<TrackingFrameInfo*, nLayers> mTrackingFrameInfoDevice;
  std::array<int*, nLayers> mClusterExternalIndicesDevice;
  std::array<int*, nLayers> mIndexTablesDevice;
  std::array<Tracklet*, nLayers - 1> mTrackletsDevice;
  std::array<int*, nLayers - 1> mTrackletsLookupTablesDevice;
  std::array<Cell*, nLayers - 2> mCellsDevice;
  Road<nLayers - 2>* mRoadsDevice;
  std::array<int*, nLayers - 2> mCellsLookupTablesDevice;
  std::array<int*, nLayers - 3> mNeighboursCellDevice;
  std::array<int*, nLayers - 3> mNeighboursCellLookupTablesDevice;
  std::array<int*, nLayers - 2> mRoadsLookupTablesDevice;

  // These are to make them accessible using layer index
  Cell** mCellsDeviceArray;
  int** mNeighboursCellDeviceArray;
  int** mNeighboursCellLookupTablesDeviceArray;

  // Small accessory buffers
  int* mCUBTmpBufferDevice;
  int* mFoundTrackletsDevice;
  int* mNFoundCellsDevice;

  /// Vertexer only
  Line* mLinesDevice;
  int* mNFoundLinesDevice;
  int* mNExclusiveFoundLinesDevice;
  unsigned char* mUsedTrackletsDevice;
  std::array<int*, 2> mNTrackletsPerClusterDevice;
  int* mClusteredLinesDevice;

  /// State and configuration
  bool mAllocated = false;
  size_t mNRof = 0;
  size_t mNPopulatedRof = 0;
  o2::its::TimeFrame* mTimeFramePtr = nullptr;
  TimeFrameGPUParameters* mTFGPUParams = nullptr;
};

template <int nLayers = 7>
class TimeFrameGPU : public TimeFrame
{
 public:
  TimeFrameGPU();
  ~TimeFrameGPU();

  /// Most relevant operations
  void registerHostMemory(const int);
  void unregisterHostMemory(const int);
  void initialise(const int, const TrackingParameters&, const int, IndexTableUtils* utils = nullptr, const TimeFrameGPUParameters* pars = nullptr);
  void initDevice(const int, IndexTableUtils*, const TrackingParameters& trkParam, const TimeFrameGPUParameters&, const int, const int);
  void initDeviceChunks(const int, const int);
  template <Task task>
  size_t loadChunkData(const size_t, const size_t, const size_t);
  size_t getNChunks() const { return mMemChunks.size(); }
  GpuTimeFrameChunk<nLayers>& getChunk(const int chunk) { return mMemChunks[chunk]; }
  Stream& getStream(const size_t stream) { return mGpuStreams[stream]; }
  void wipe(const int);

  /// interface
  int getNClustersInRofSpan(const int, const int, const int) const;
  IndexTableUtils* getDeviceIndexTableUtils() { return mIndexTableUtilsDevice; }
  int* getDeviceROframesClusters(const int layer) { return mROframesClustersDevice[layer]; }
  std::vector<std::vector<Vertex>>& getVerticesInChunks() { return mVerticesInChunks; }
  std::vector<std::vector<int>>& getNVerticesInChunks() { return mNVerticesInChunks; }
  std::vector<std::vector<o2::MCCompLabel>>& getLabelsInChunks() { return mLabelsInChunks; }
  int getNAllocatedROFs() const { return mNrof; } // Allocated means maximum nROF for each chunk while populated is the number of loaded ones.
  StaticTrackingParameters<nLayers>* getDeviceTrackingParameters() { return mTrackingParamsDevice; }
  Vertex* getDeviceVertices() { return mVerticesDevice; }
  int* getDeviceROframesPV() { return mROframesPVDevice; }
  unsigned char* getDeviceUsedClusters(const int);

  // Host-specific getters
  gsl::span<int> getHostNTracklets(const int chunkId);
  gsl::span<int> getHostNCells(const int chunkId);

 private:
  bool mHostRegistered = false;
  std::vector<GpuTimeFrameChunk<nLayers>> mMemChunks;
  TimeFrameGPUParameters mGpuParams;
  StaticTrackingParameters<nLayers> mStaticTrackingParams;

  // Device pointers
  StaticTrackingParameters<nLayers>* mTrackingParamsDevice;
  IndexTableUtils* mIndexTableUtilsDevice;
  std::array<int*, nLayers> mROframesClustersDevice;
  std::array<unsigned char*, nLayers> mUsedClustersDevice;
  Vertex* mVerticesDevice;
  int* mROframesPVDevice;

  // State
  std::vector<Stream> mGpuStreams;
  size_t mAvailMemGB;
  bool mFirstInit = true;

  // Output
  std::vector<std::vector<Vertex>> mVerticesInChunks;
  std::vector<std::vector<int>> mNVerticesInChunks;
  std::vector<std::vector<o2::MCCompLabel>> mLabelsInChunks;

  // Host memeory used only in GPU tracking
  std::vector<int> mHostNTracklets;
  std::vector<int> mHostNCells;
};

template <int nLayers>
template <Task task>
size_t TimeFrameGPU<nLayers>::loadChunkData(const size_t chunk, const size_t offset, const size_t maxRofs) // offset: readout frame to start from, maxRofs: to manage boundaries
{
  size_t nRof{0};

  mMemChunks[chunk].reset(task, mGpuStreams[chunk]); // Reset chunks memory
  if constexpr ((bool)task) {
    nRof = mMemChunks[chunk].loadDataOnDevice(offset, maxRofs, 3, mGpuStreams[chunk]);
  } else {
    nRof = mMemChunks[chunk].loadDataOnDevice(offset, maxRofs, nLayers, mGpuStreams[chunk]);
  }
  LOGP(debug, "In chunk {}: loaded {} readout frames starting from {}", chunk, nRof, offset);
  return nRof;
}

template <int nLayers>
inline int TimeFrameGPU<nLayers>::getNClustersInRofSpan(const int rofIdstart, const int rofSpanSize, const int layerId) const
{
  return static_cast<int>(mROframesClusters[layerId][(rofIdstart + rofSpanSize) < mROframesClusters.size() ? rofIdstart + rofSpanSize : mROframesClusters.size() - 1] - mROframesClusters[layerId][rofIdstart]);
}
} // namespace gpu
} // namespace its
} // namespace o2
#endif