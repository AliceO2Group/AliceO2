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
  void reset(const size_t, const Task, Stream&);
  size_t copyDeviceData(const size_t, const int, Stream&);

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
  int* getDeviceNTrackletCluster(const int combid) { return mNTrackletsPerClusterDevice[combid]; }
  Line* getDeviceLines() { return mLinesDevice; };
  int* getDeviceNFoundLines() { return mNFoundLinesDevice; }
  int* getDeviceNExclusiveFoundLines() { return mNExclusiveFoundLinesDevice; }
  unsigned char* getDeviceUsedTracklets() { return mUsedTrackletsDevice; }

  /// Host
  std::array<gsl::span<const Cluster>, nLayers> mHostClusters;
  std::array<gsl::span<const int>, nLayers> mHostROframesClusters;

  /// Device
  std::array<int*, nLayers> mROframesClustersDevice; // layers x roframes
  std::array<Cluster*, nLayers> mClustersDevice;

 private:
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
  std::array<int*, 2> mNTrackletsPerClusterDevice;

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
  void registerHostMemory(const int);
  void unregisterHostMemory(const int);
  void initialise(const int, const TrackingParameters&, const int, const IndexTableUtils* utils = nullptr);
  void initDevice(const int, const IndexTableUtils*, const int);
  void initDevicePartitions(const int, const int);
  template <Task task>
  size_t loadPartitionData(const size_t, const size_t);
  size_t getNPartions() const { return mMemPartitions.size(); }
  GpuTimeFramePartition<nLayers>& getPartition(const int part) { return mMemPartitions[part]; }
  Stream& getStream(const size_t stream) { return mGpuStreams[stream]; }

  /// interface
  int getNClustersInRofSpan(const int, const int, const int) const;

 private:
  bool mDeviceInitialised = false;
  bool mHostRegistered = false;
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
size_t TimeFrameGPU<nLayers>::loadPartitionData(const size_t part, const size_t offset) // offset: readout frame to start from
{
  size_t nRof{0};

  mMemPartitions[part].reset(GpuTimeFramePartition<nLayers>::computeRofPerPartition(mGpuConfig, mAvailMemGB),
                             task,
                             mGpuStreams[part]); // Reset partitions memory
  if constexpr ((bool)task) {
    nRof = mMemPartitions[part].copyDeviceData(offset, 3, mGpuStreams[part]);
  } else {
    nRof = mMemPartitions[part].copyDeviceData(offset, nLayers, mGpuStreams[part]);
  }
  LOGP(info, "In partition {}: loaded {} readout frames starting from {}", part, nRof, offset);
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