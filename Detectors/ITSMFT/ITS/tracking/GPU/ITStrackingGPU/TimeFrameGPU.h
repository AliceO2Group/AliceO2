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
template <int>
class StaticTrackingParameters;

enum class Task {
  Tracker = 0,
  Vertexer = 1
};

template <int nLayers>
class GpuTimeFramePartition
{
 public:
  static size_t computeScalingSizeBytes(const int, const TimeFrameGPUConfig&);
  static size_t computeFixedSizeBytes(const TimeFrameGPUConfig&);
  static size_t computeRofPerPartition(const TimeFrameGPUConfig&, const size_t);

  GpuTimeFramePartition() = delete;
  GpuTimeFramePartition(o2::its::TimeFrame* tf, TimeFrameGPUConfig& conf)
  {
    mTimeFramePtr = tf;
    mTFGconf = &conf;
  }
  ~GpuTimeFramePartition();

  /// Most relevant operations
  void allocate(const size_t, Stream&);
  void reset(const size_t, const Task);
  void copyDeviceData(const size_t, const int, Stream);

  /// Interface
  int* getDeviceROframesClusters(const int);
  Cluster* getDeviceClusters(const int);
  unsigned char* getDeviceUsedClusters(const int);
  TrackingFrameInfo* getDeviceTrackingFrameInfo(const int);
  int* getDeviceClusterExternalIndices(const int);
  int* getDeviceIndexTables(const int);
  Tracklet* getDeviceTracklets(const int);
  int* getDeviceTrackletsLookupTables(const int);
  Cell* getDeviceCells(const int);
  int* getDeviceCellsLookupTables(const int);

  int* getDeviceCUBTmpBuffers() { return mCUBTmpBufferDevice; }
  int* getDeviceFoundTracklets() { return mFoundTrackletsDevice; }
  int* getDeviceFoundCells() { return mFoundCellsDevice; }
  IndexTableUtils* getDeviceIndexTableUtils() { return mIndexTableUtilsDevice; }

  /// Vertexer only
  Line* getDeviceLines() { return mLinesDevice; };
  int* getDeviceNFoundLines() { return mNFoundLinesDevice; }
  int* getDeviceNExclusiveFoundLines() { return mNExclusiveFoundLinesDevice; }
  unsigned char* getDeviceUsedTracklets() { return mUsedTrackletsDevice; }

 private:
  /// Host
  std::array<gsl::span<const Cluster>, nLayers> mHostClusters;

  /// Device
  std::array<int*, nLayers> mROframesClustersDevice; // layers x roframes
  std::array<Cluster*, nLayers> mClustersDevice;
  std::array<unsigned char*, nLayers> mUsedClustersDevice;
  std::array<TrackingFrameInfo*, nLayers> mTrackingFrameInfoDevice;
  std::array<int*, nLayers> mClusterExternalIndicesDevice;
  std::array<int*, nLayers> mIndexTablesDevice;
  std::array<Tracklet*, nLayers - 1> mTrackletsDevice;
  std::array<int*, nLayers - 1> mTrackletsLookupTablesDevice;
  std::array<Cell*, nLayers - 2> mCellsDevice;
  std::array<int*, nLayers - 2> mCellsLookupTablesDevice;

  int* mCUBTmpBufferDevice;
  int* mFoundTrackletsDevice;
  int* mFoundCellsDevice;
  IndexTableUtils* mIndexTableUtilsDevice;

  /// Vertexer only
  Line* mLinesDevice;
  int* mNFoundLinesDevice;
  int* mNExclusiveFoundLinesDevice;
  unsigned char* mUsedTrackletsDevice;

  /// State and configuration
  bool mAllocated = false;
  size_t mNRof = 0;
  o2::its::TimeFrame* mTimeFramePtr = nullptr;
  TimeFrameGPUConfig* mTFGconf = nullptr;
};

template <int nLayers = 7>
class TimeFrameGPU : public TimeFrame
{
 public:
  TimeFrameGPU();
  ~TimeFrameGPU();

  /// Most relevant operations
  void initialise(const int, const TrackingParameters&, const int, const IndexTableUtils* utils = nullptr);
  void initDevice(const int, const IndexTableUtils*, const int);
  void initDevicePartitions(const int, const int);
  template <Task task>
  void loadPartitionData(const size_t);
  size_t getNPartions() const { return mMemPartitions.size(); }

  /// interface
  int getNClustersInRofSpan(const int, const int, const int) const;

 private:
  bool mInitialised = false;
  std::vector<GpuTimeFramePartition<nLayers>> mMemPartitions;
  TimeFrameGPUConfig mGpuConfig;

  // Device pointers
  StaticTrackingParameters<nLayers>* mTrackingParamsDevice;
  IndexTableUtils* mDeviceIndexTableUtils;

  // State
  std::vector<Stream> mGpuStreams;
  size_t mAvailMemGB; //
};

template <int nLayers>
template <Task task>
void TimeFrameGPU<nLayers>::loadPartitionData(const size_t part)
{
  auto startRof = part * GpuTimeFramePartition<nLayers>::computeRofPerPartition(mGpuConfig, mAvailMemGB);
  mMemPartitions[part].reset(GpuTimeFramePartition<nLayers>::computeRofPerPartition(mGpuConfig, mAvailMemGB), task);
  if constexpr ((bool)task) {
    mMemPartitions[part].copyDeviceData(startRof, 3, mGpuStreams[part]);
  } else {
    mMemPartitions[part].copyDeviceData(startRof, nLayers, mGpuStreams[part]);
  }
}

template <int nLayers>
inline int TimeFrameGPU<nLayers>::getNClustersInRofSpan(const int rofIdstart, const int rofSpanSize, const int layerId) const
{
  return static_cast<int>(mROframesClusters[layerId][(rofIdstart + rofSpanSize) < mROframesClusters.size() ? rofIdstart + rofSpanSize : mROframesClusters.size() - 1] - mROframesClusters[layerId][rofIdstart]);
}

// #ifdef __HIPCC__
// #include <hip/hip_runtime.h>
// #endif

// #include "ITStracking/TimeFrame.h"
// #include "ITStracking/Configuration.h"

// #include "ITStrackingGPU/ClusterLinesGPU.h"
// #include "ITStrackingGPU/Stream.h"

// #include "Array.h"
// #include "Vector.h"

// #include "GPUCommonDef.h"
// #include "GPUCommonMath.h"
// #include "GPUCommonLogger.h"

// namespace o2
// {

// namespace its
// {
// using namespace constants::its2;

// class TimeFrameGPUConfig;
// namespace gpu
// {

// template <int nLayers = 7>
// class TimeFrameGPU : public TimeFrame
// {

//  public:
//   TimeFrameGPU();
//   ~TimeFrameGPU();

//   void checkBufferSizes();
//   void initialise(const int iteration,
//                   const TrackingParameters& trkParam,
//                   const int maxLayers);
//   template <unsigned char isTracker = false>
//   void initialiseDevice(const TrackingParameters&);
//   /// Getters
//   float getDeviceMemory();
//   Cluster* getDeviceClustersOnLayer(const int rofId, const int layerId) const;
//   unsigned char* getDeviceUsedClustersOnLayer(const int rofId, const int layerId);
//   int* getDeviceROframesClustersOnLayer(const int layerId) const { return mROframesClustersD[layerId].get(); }
//   int getNClustersLayer(const int rofId, const int layerId) const;
//   TimeFrameGPUConfig& getConfig() { return mConfig; }
//   gpu::Stream& getStream(const int iLayer) { return mStreamArray[iLayer]; }
//   std::vector<int>& getTrackletSizeHost() { return mTrackletSizeHost; }
//   std::vector<int>& getCellSizeHost() { return mCellSizeHost; }

//   // Vertexer only
//   int* getDeviceNTrackletsCluster(int rofId, int combId);
//   int* getDeviceIndexTables(const int layerId) { return mIndexTablesD[layerId].get(); }
//   int* getDeviceIndexTableAtRof(const int layerId, const int rofId) { return mIndexTablesD[layerId].get() + rofId * (ZBins * PhiBins + 1); }
//   unsigned char* getDeviceUsedTracklets(const int rofId);
//   Line* getDeviceLines(const int rofId);
//   Tracklet* getDeviceTrackletsVertexerOnly(const int rofId, const int layerId); // this method uses the cluster table for layer 1 for any layer. It is used for the vertexer only.
//   Tracklet* getDeviceTracklets(const int rofId, const int layerId);
//   Tracklet* getDeviceTrackletsAll(const int layerId);
//   Cell* getDeviceCells(const int layerId);
//   int* getDeviceTrackletsLookupTable(const int rofId, const int layerId);
//   int* getDeviceCellsLookupTable(const int layerId);
//   int* getDeviceNFoundLines(const int rofId);
//   int* getDeviceExclusiveNFoundLines(const int rofId);
//   int* getDeviceCUBBuffer(const size_t rofId);
//   int* getDeviceNFoundTracklets() const { return mDeviceFoundTracklets; };
//   int* getDeviceNFoundCells() const { return mDeviceFoundCells; };
//   float* getDeviceXYCentroids(const int rofId);
//   float* getDeviceZCentroids(const int rofId);
//   int* getDeviceXHistograms(const int rofId);
//   int* getDeviceYHistograms(const int rofId);
//   int* getDeviceZHistograms(const int rofId);
//   gpu::StaticTrackingParameters<nLayers>* getDeviceTrackingParameters() const { return mDeviceTrackingParams; }
//   IndexTableUtils* getDeviceIndexTableUtils() const { return mDeviceIndexTableUtils; }

// #ifdef __HIPCC__
//   hipcub::KeyValuePair<int, int>* getTmpVertexPositionBins(const int rofId);
// #else
//   cub::KeyValuePair<int, int>* getTmpVertexPositionBins(const int rofId);
// #endif
//   float* getDeviceBeamPosition(const int rofId);
//   Vertex* getDeviceVertices(const int rofId);

//  private:
//   TimeFrameGPUConfig mConfig;
//   std::array<gpu::Stream, nLayers + 1> mStreamArray;
//   std::vector<int> mTrackletSizeHost;
//   std::vector<int> mCellSizeHost;
//   // Per-layer information, do not expand at runtime
//   std::array<Vector<Cluster>, nLayers> mClustersD;
//   std::array<Vector<unsigned char>, nLayers> mUsedClustersD;
//   std::array<Vector<TrackingFrameInfo>, nLayers> mTrackingFrameInfoD;
//   std::array<Vector<int>, nLayers - 1> mIndexTablesD;
//   std::array<Vector<int>, nLayers> mClusterExternalIndicesD;
//   std::array<Vector<Tracklet>, nLayers - 1> mTrackletsD;
//   std::array<Vector<int>, nLayers - 1> mTrackletsLookupTablesD;
//   std::array<Vector<Cell>, nLayers - 2> mCellsD;
//   std::array<Vector<int>, nLayers - 2> mCellsLookupTablesD;
//   std::array<Vector<int>, nLayers> mROframesClustersD; // layers x roframes
//   int* mCUBTmpBuffers;
//   int* mDeviceFoundTracklets;
//   int* mDeviceFoundCells;
//   gpu::StaticTrackingParameters<nLayers>* mDeviceTrackingParams;
//   IndexTableUtils* mDeviceIndexTableUtils;

//   // Vertexer only
//   Vector<Line> mLines;
//   Vector<int> mIndexTablesLayer0D;
//   Vector<int> mIndexTablesLayer2D;
//   Vector<int> mNFoundLines;
//   Vector<int> mNExclusiveFoundLines;
//   Vector<unsigned char> mUsedTracklets;
//   Vector<float> mXYCentroids;
//   Vector<float> mZCentroids;
//   std::array<Vector<int>, 2> mNTrackletsPerClusterD;
//   std::array<Vector<int>, 3> mXYZHistograms;
//   Vector<float> mBeamPosition;
//   Vector<Vertex> mGPUVertices;
// #ifdef __HIPCC__
//   Vector<hipcub::KeyValuePair<int, int>> mTmpVertexPositionBins;
// #else
//   Vector<cub::KeyValuePair<int, int>> mTmpVertexPositionBins;
// #endif
// };

// template <int nLayers>
// inline Cluster* TimeFrameGPU<nLayers>::getDeviceClustersOnLayer(const int rofId, const int layerId) const
// {
//   return getPtrFromRuler<Cluster>(rofId, mClustersD[layerId].get(), mROframesClusters[layerId].data());
// }

// template <int nLayers>
// inline unsigned char* TimeFrameGPU<nLayers>::getDeviceUsedClustersOnLayer(const int rofId, const int layerId)
// {
//   return getPtrFromRuler<unsigned char>(rofId, mUsedClustersD[layerId].get(), mROframesClusters[layerId].data());
// }

// template <int nLayers>
// inline int TimeFrameGPU<nLayers>::getNClustersLayer(const int rofId, const int layerId) const
// {
//   return static_cast<int>(mROframesClusters[layerId][rofId + 1] - mROframesClusters[layerId][rofId]);
// }

// template <int nLayers>
// inline int* TimeFrameGPU<nLayers>::getDeviceNTrackletsCluster(int rofId, int combId)
// {
//   return getPtrFromRuler<int>(rofId, mNTrackletsPerClusterD[combId].get(), mROframesClusters[1].data());
// }

// template <int nLayers>
// inline unsigned char* TimeFrameGPU<nLayers>::getDeviceUsedTracklets(const int rofId)
// {
//   return getPtrFromRuler<unsigned char>(rofId, mUsedTracklets.get(), mROframesClusters[1].data(), mConfig.maxTrackletsPerCluster);
// }

// template <int nLayers>
// inline Line* TimeFrameGPU<nLayers>::getDeviceLines(const int rofId)
// {
//   return getPtrFromRuler(rofId, mLines.get(), mROframesClusters[1].data());
// }

// template <int nLayers>
// inline Tracklet* TimeFrameGPU<nLayers>::getDeviceTrackletsVertexerOnly(const int rofId, const int layerId)
// {
//   return getPtrFromRuler(rofId, mTrackletsD[layerId].get(), mROframesClusters[1].data(), mConfig.maxTrackletsPerCluster);
// }

// template <int nLayers>
// inline Tracklet* TimeFrameGPU<nLayers>::getDeviceTracklets(const int rofId, const int layerId)
// {
//   return getPtrFromRuler(rofId, mTrackletsD[layerId].get(), mROframesClusters[layerId].data(), mConfig.maxTrackletsPerCluster);
// }

// template <int nLayers>
// inline Tracklet* TimeFrameGPU<nLayers>::getDeviceTrackletsAll(const int layerId)
// {
//   return mTrackletsD[layerId].get();
// }

// template <int nLayers>
// inline Cell* TimeFrameGPU<nLayers>::getDeviceCells(const int layerId)
// {
//   return mCellsD[layerId].get();
// }

// template <int nLayers>
// inline int* TimeFrameGPU<nLayers>::getDeviceTrackletsLookupTable(const int rofId, const int layerId)
// {
//   return getPtrFromRuler(rofId, mTrackletsLookupTablesD[layerId].get(), mROframesClusters[layerId].data());
// }

// template <int nLayers>
// inline int* TimeFrameGPU<nLayers>::getDeviceCellsLookupTable(const int layerId)
// {
//   return mCellsLookupTablesD[layerId].get();
// }

// template <int nLayers>
// inline int* TimeFrameGPU<nLayers>::getDeviceNFoundLines(const int rofId)
// {
//   return getPtrFromRuler<int>(rofId, mNFoundLines.get(), mROframesClusters[1].data());
// }

// template <int nLayers>
// inline int* TimeFrameGPU<nLayers>::getDeviceExclusiveNFoundLines(const int rofId)
// {
//   return getPtrFromRuler<int>(rofId, mNExclusiveFoundLines.get(), mROframesClusters[1].data());
// }

// template <int nLayers>
// inline int* TimeFrameGPU<nLayers>::getDeviceCUBBuffer(const size_t rofId)
// {
//   return reinterpret_cast<int*>(reinterpret_cast<char*>(mCUBTmpBuffers) + (static_cast<size_t>(rofId * mConfig.tmpCUBBufferSize) & 0xFFFFFFFFFFFFF000));
// }

// template <int nLayers>
// inline float* TimeFrameGPU<nLayers>::getDeviceXYCentroids(const int rofId)
// {
//   return mXYCentroids.get() + 2 * rofId * mConfig.maxCentroidsXYCapacity;
// }

// template <int nLayers>
// inline float* TimeFrameGPU<nLayers>::getDeviceZCentroids(const int rofId)
// {
//   return mZCentroids.get() + rofId * mConfig.maxLinesCapacity;
// }

// template <int nLayers>
// inline int* TimeFrameGPU<nLayers>::getDeviceXHistograms(const int rofId)
// {
//   return mXYZHistograms[0].get() + rofId * mConfig.histConf.nBinsXYZ[0];
// }

// template <int nLayers>
// inline int* TimeFrameGPU<nLayers>::getDeviceYHistograms(const int rofId)
// {
//   return mXYZHistograms[1].get() + rofId * mConfig.histConf.nBinsXYZ[1];
// }

// template <int nLayers>
// inline int* TimeFrameGPU<nLayers>::getDeviceZHistograms(const int rofId)
// {
//   return mXYZHistograms[2].get() + rofId * mConfig.histConf.nBinsXYZ[2];
// }

// template <int nLayers>
// #ifdef __HIPCC__
// inline hipcub::KeyValuePair<int, int>* TimeFrameGPU<nLayers>::getTmpVertexPositionBins(const int rofId)
// #else
// inline cub::KeyValuePair<int, int>* TimeFrameGPU<nLayers>::getTmpVertexPositionBins(const int rofId)
// #endif
// {
//   return mTmpVertexPositionBins.get() + 3 * rofId;
// }

// template <int nLayers>
// inline float* TimeFrameGPU<nLayers>::getDeviceBeamPosition(const int rofId)
// {
//   return mBeamPosition.get() + 2 * rofId;
// }

// template <int nLayers>
// inline Vertex* TimeFrameGPU<nLayers>::getDeviceVertices(const int rofId)
// {
//   return mGPUVertices.get() + rofId * mConfig.maxVerticesCapacity;
// }

} // namespace gpu
} // namespace its
} // namespace o2
#endif