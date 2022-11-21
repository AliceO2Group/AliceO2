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
#include <thrust/fill.h>

#include "ITStracking/Constants.h"

#include "ITStrackingGPU/Utils.h"
#include "ITStrackingGPU/TimeFrameGPU.h"

#include <unistd.h>

namespace o2
{
namespace its
{
using constants::GB;
using constants::MB;
namespace gpu
{
using utils::host::checkGPUError;

template <int nLayers>
struct StaticTrackingParameters {
  StaticTrackingParameters<nLayers>& operator=(const StaticTrackingParameters<nLayers>& t) = default;
  void set(const TrackingParameters& pars);

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

template <int nLayers>
void StaticTrackingParameters<nLayers>::set(const TrackingParameters& pars)
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

/////////////////////////////////////////////////////////////////////////////////////////
// GpuPartition
template <int nLayers>
GpuTimeFramePartition<nLayers>::~GpuTimeFramePartition()
{
  for (int i = 0; i < nLayers; ++i) {
    checkGPUError(cudaFree(mROframesClustersDevice[i]));
    checkGPUError(cudaFree(mClustersDevice[i]));
    checkGPUError(cudaFree(mUsedClustersDevice[i]));
    checkGPUError(cudaFree(mTrackingFrameInfoDevice[i]));
    checkGPUError(cudaFree(mClusterExternalIndicesDevice[i]));
    checkGPUError(cudaFree(mIndexTablesDevice[i]));
    if (i < nLayers - 1) {
      checkGPUError(cudaFree(mTrackletsDevice[i]));
      checkGPUError(cudaFree(mTrackletsLookupTablesDevice[i]));
      if (i < nLayers - 2) {
        checkGPUError(cudaFree(mCellsDevice[i]));
        checkGPUError(cudaFree(mCellsLookupTablesDevice[i]));
      }
    }
  }
  checkGPUError(cudaFree(mCUBTmpBuffers));
  checkGPUError(cudaFree(mFoundTrackletsDevice));
  checkGPUError(cudaFree(mFoundCellsDevice));
}

template <int nLayers>
void GpuTimeFramePartition<nLayers>::init(const size_t nrof, const TimeFrameGPUConfig& config)
{
  for (int i = 0; i < nLayers; ++i) {
    checkGPUError(cudaMalloc(reinterpret_cast<void**>(&(mROframesClustersDevice[i])), sizeof(int) * nrof));
    checkGPUError(cudaMalloc(reinterpret_cast<void**>(&(mClustersDevice[i])), sizeof(Cluster) * config.clustersPerROfCapacity * nrof));
    checkGPUError(cudaMalloc(reinterpret_cast<void**>(&(mUsedClustersDevice[i])), sizeof(unsigned char) * config.clustersPerROfCapacity * nrof));
    checkGPUError(cudaMalloc(reinterpret_cast<void**>(&(mTrackingFrameInfoDevice[i])), sizeof(TrackingFrameInfo) * config.clustersPerROfCapacity * nrof));
    checkGPUError(cudaMalloc(reinterpret_cast<void**>(&(mClusterExternalIndicesDevice[i])), sizeof(int) * config.clustersPerROfCapacity * nrof));
    checkGPUError(cudaMalloc(reinterpret_cast<void**>(&(mIndexTablesDevice[i])), sizeof(int) * (256 * 128 + 1) * nrof));
    if (i < nLayers - 1) {
      checkGPUError(cudaMalloc(reinterpret_cast<void**>(&(mTrackletsLookupTablesDevice[i])), sizeof(int) * config.clustersPerROfCapacity * nrof));
      checkGPUError(cudaMalloc(reinterpret_cast<void**>(&(mTrackletsDevice[i])), sizeof(Tracklet) * config.maxTrackletsPerCluster * config.clustersPerROfCapacity * nrof));
      if (i < nLayers - 2) {
        checkGPUError(cudaMalloc(reinterpret_cast<void**>(&(mCellsLookupTablesDevice[i])), sizeof(int) * config.validatedTrackletsCapacity * nrof));
        checkGPUError(cudaMalloc(reinterpret_cast<void**>(&(mCellsDevice[i])), sizeof(Cell) * config.validatedTrackletsCapacity * nrof));
      }
    }
  }
  checkGPUError(cudaMalloc(reinterpret_cast<void**>(&mCUBTmpBuffers), config.tmpCUBBufferSize * nrof));
  checkGPUError(cudaMalloc(&mFoundTrackletsDevice, (nLayers - 1) * sizeof(int)));
  checkGPUError(cudaMalloc(&mFoundCellsDevice, (nLayers - 2) * sizeof(int)));
}

template <int nLayers>
template <Task task>
void GpuTimeFramePartition<nLayers>::reset(const size_t nrof, const TimeFrameGPUConfig& config)
{
  if constexpr ((bool)task) { // Vertexer-only initialisation
  } else {
    for (int i = 0; i < nLayers - 1; ++i) {
      checkGPUError(cudaMemset(mTrackletsLookupTablesDevice[i], 0, sizeof(int) * config.clustersPerROfCapacity * nrof));
      if (i < nLayers - 2) {
        checkGPUError(cudaMemset(mCellsLookupTablesDevice[i], 0, sizeof(int) * config.validatedTrackletsCapacity * nrof));
      }
    }
    checkGPUError(cudaMemset(mFoundCellsDevice, 0, (nLayers - 2) * sizeof(int)));
  }
}

template <int nLayers>
size_t GpuTimeFramePartition<nLayers>::computeScalingSizeBytes(const int nrof, const TimeFrameGPUConfig& config)
{
  size_t rofsize = nLayers * sizeof(int), total;                                                               // number of clusters per ROF
  rofsize += nLayers * sizeof(Cluster) * config.clustersPerROfCapacity;                                        // clusters
  rofsize += nLayers * sizeof(unsigned char) * config.clustersPerROfCapacity;                                  // used clusters flags
  rofsize += nLayers * sizeof(TrackingFrameInfo) * config.clustersPerROfCapacity;                              // tracking frame info
  rofsize += nLayers * sizeof(int) * config.clustersPerROfCapacity;                                            // external cluster indices
  rofsize += nLayers * sizeof(int) * (256 * 128 + 1);                                                          // index tables
  rofsize += (nLayers - 1) * sizeof(int) * config.clustersPerROfCapacity;                                      // tracklets lookup tables
  rofsize += (nLayers - 1) * sizeof(Tracklet) * config.maxTrackletsPerCluster * config.clustersPerROfCapacity; // tracklets
  rofsize += (nLayers - 2) * sizeof(int) * config.validatedTrackletsCapacity;                                  // cells lookup tables
  rofsize += (nLayers - 2) * sizeof(Cell) * config.validatedTrackletsCapacity;                                 // cells

  return rofsize * nrof;
}

template <int nLayers>
size_t GpuTimeFramePartition<nLayers>::computeFixedSizeBytes(const TimeFrameGPUConfig& config)
{
  size_t total = config.tmpCUBBufferSize; // CUB tmp buffers
  total += (nLayers - 1) * sizeof(int);   // found tracklets
  total += (nLayers - 2) * sizeof(int);   // found cells
  return total;
}

template <int nLayers>
size_t GpuTimeFramePartition<nLayers>::computeRofPerPartition(const TimeFrameGPUConfig& config)
{
  return (config.maxTotalMemoryGB * GB / (float)(config.nStreams * config.partitionStreamRatio) - GpuTimeFramePartition<nLayers>::computeFixedSizeBytes(config)) / (float)GpuTimeFramePartition<nLayers>::computeScalingSizeBytes(1, config);
}

/////////////////////////////////////////////////////////////////////////////////////////
// TimeFrameGPU
template <int nLayers>
TimeFrameGPU<nLayers>::TimeFrameGPU()
{
  mIsGPU = true;
}

template <int nLayers>
TimeFrameGPU<nLayers>::~TimeFrameGPU()
{
  checkGPUError(cudaFree(mTrackingParamsDevice));
}

template <int nLayers>
void TimeFrameGPU<nLayers>::initialise(const int iteration,
                                       const TrackingParameters& trkParam,
                                       const int maxLayers)
{
  initDevice(mGpuConfig.nStreams * mGpuConfig.partitionStreamRatio);
  o2::its::TimeFrame::initialise(iteration, trkParam, maxLayers);
}

template <int nLayers>
void TimeFrameGPU<nLayers>::initDevice(const int partitions)
{
  StaticTrackingParameters<nLayers> pars;
  checkGPUError(cudaMalloc(reinterpret_cast<void**>(&mTrackingParamsDevice), sizeof(gpu::StaticTrackingParameters<nLayers>)));
  checkGPUError(cudaMemcpy(mTrackingParamsDevice, &pars, sizeof(gpu::StaticTrackingParameters<nLayers>), cudaMemcpyHostToDevice));
  mMemPartitions.resize(partitions);
  LOGP(info, "Size of fixed part is: {} MB", GpuTimeFramePartition<nLayers>::computeFixedSizeBytes(mGpuConfig) / MB);
  LOGP(info, "Size of scaling part is: {} MB", GpuTimeFramePartition<nLayers>::computeScalingSizeBytes(GpuTimeFramePartition<nLayers>::computeRofPerPartition(mGpuConfig), mGpuConfig) / MB);
  LOGP(info, "Going to allocate {} partitions of {} rofs each.", partitions, GpuTimeFramePartition<nLayers>::computeRofPerPartition(mGpuConfig));

  initDevicePartitions(GpuTimeFramePartition<nLayers>::computeRofPerPartition(mGpuConfig));
}

template <int nLayers>
void TimeFrameGPU<nLayers>::initDevicePartitions(const int nRof)
{
  if (!mMemPartitions.size()) {
    LOGP(fatal, "gpu-tracking: TimeFrame GPU partitions not created");
  }
  for (auto& partition : mMemPartitions) {
    partition.init(nRof, mGpuConfig);
  }
}

// template <int nLayers>
// float TimeFrameGPU<nLayers>::getDeviceMemory()
// {
//   // We don't check if we can store the data in the GPU for the moment, only log it.
//   float totalMemory{0};
//   totalMemory += nLayers * mConfig.clustersPerLayerCapacity * sizeof(Cluster);
//   totalMemory += nLayers * mConfig.clustersPerLayerCapacity * sizeof(unsigned char);
//   totalMemory += nLayers * mConfig.clustersPerLayerCapacity * sizeof(TrackingFrameInfo);
//   totalMemory += nLayers * mConfig.clustersPerLayerCapacity * sizeof(int);
//   totalMemory += nLayers * mConfig.clustersPerROfCapacity * sizeof(int);
//   totalMemory += (nLayers - 1) * mConfig.trackletsCapacity * sizeof(Tracklet);
//   totalMemory += (nLayers - 1) * mConfig.nMaxROFs * (256 * 128 + 1) * sizeof(int);
//   totalMemory += 2 * mConfig.clustersPerLayerCapacity * sizeof(int);
//   totalMemory += 2 * mConfig.nMaxROFs * (ZBins * PhiBins + 1) * sizeof(int);
//   totalMemory += mConfig.trackletsCapacity * sizeof(Line);
//   totalMemory += mConfig.clustersPerLayerCapacity * sizeof(int);
//   totalMemory += mConfig.clustersPerLayerCapacity * sizeof(int);
//   totalMemory += mConfig.trackletsCapacity * sizeof(unsigned char);
//   totalMemory += mConfig.nMaxROFs * mConfig.tmpCUBBufferSize * sizeof(int);
//   totalMemory += 2 * mConfig.nMaxROFs * mConfig.maxCentroidsXYCapacity * sizeof(float);
//   totalMemory += mConfig.nMaxROFs * mConfig.maxLinesCapacity * sizeof(float);
//   for (size_t i{0}; i < 3; ++i) {
//     totalMemory += mConfig.nMaxROFs * mConfig.histConf.nBinsXYZ[i] * sizeof(float);
//   }
//   totalMemory += 3 * mConfig.nMaxROFs * sizeof(cub::KeyValuePair<int, int>);
//   totalMemory += 2 * mConfig.nMaxROFs * sizeof(float);
//   totalMemory += mConfig.nMaxROFs * mConfig.maxVerticesCapacity * sizeof(Vertex);

//   LOG(info) << fmt::format("Total requested memory for GPU: {:.2f} MB", totalMemory / MB);
//   LOG(info) << fmt::format("\t- Clusters: {:.2f} MB", nLayers * mConfig.clustersPerLayerCapacity * sizeof(Cluster) / MB);
//   LOG(info) << fmt::format("\t- Used clusters: {:.2f} MB", nLayers * mConfig.clustersPerLayerCapacity * sizeof(unsigned char) / MB);
//   LOG(info) << fmt::format("\t- Tracking frame info: {:.2f} MB", nLayers * mConfig.clustersPerLayerCapacity * sizeof(TrackingFrameInfo) / MB);
//   LOG(info) << fmt::format("\t- Cluster external indices: {:.2f} MB", nLayers * mConfig.clustersPerLayerCapacity * sizeof(int) / MB);
//   LOG(info) << fmt::format("\t- Clusters per ROf: {:.2f} MB", nLayers * mConfig.clustersPerROfCapacity * sizeof(int) / MB);
//   LOG(info) << fmt::format("\t- Tracklets: {:.2f} MB", (nLayers - 1) * mConfig.trackletsCapacity * sizeof(Tracklet) / MB);
//   LOG(info) << fmt::format("\t- Tracklet index tables: {:.2f} MB", (nLayers - 1) * mConfig.nMaxROFs * (256 * 128 + 1) * sizeof(int) / MB);
//   LOG(info) << fmt::format("\t- N tracklets per cluster: {:.2f} MB", 2 * mConfig.clustersPerLayerCapacity * sizeof(int) / MB);
//   LOG(info) << fmt::format("\t- Index tables: {:.2f} MB", 2 * mConfig.nMaxROFs * (ZBins * PhiBins + 1) * sizeof(int) / MB);
//   LOG(info) << fmt::format("\t- Lines: {:.2f} MB", mConfig.trackletsCapacity * sizeof(Line) / MB);
//   LOG(info) << fmt::format("\t- N found lines: {:.2f} MB", mConfig.clustersPerLayerCapacity * sizeof(int) / MB);
//   LOG(info) << fmt::format("\t- N exclusive-scan found lines: {:.2f} MB", mConfig.clustersPerLayerCapacity * sizeof(int) / MB);
//   LOG(info) << fmt::format("\t- Used tracklets: {:.2f} MB", mConfig.trackletsCapacity * sizeof(unsigned char) / MB);
//   LOG(info) << fmt::format("\t- CUB tmp buffers: {:.2f} MB", mConfig.nMaxROFs * mConfig.tmpCUBBufferSize / MB);
//   LOG(info) << fmt::format("\t- XY centroids: {:.2f} MB", 2 * mConfig.nMaxROFs * mConfig.maxCentroidsXYCapacity * sizeof(float) / MB);
//   LOG(info) << fmt::format("\t- Z centroids: {:.2f} MB", mConfig.nMaxROFs * mConfig.maxLinesCapacity * sizeof(float) / MB);
//   LOG(info) << fmt::format("\t- XY histograms: {:.2f} MB", 2 * mConfig.nMaxROFs * mConfig.histConf.nBinsXYZ[0] * sizeof(int) / MB);
//   LOG(info) << fmt::format("\t- Z histograms: {:.2f} MB", mConfig.nMaxROFs * mConfig.histConf.nBinsXYZ[2] * sizeof(int) / MB);
//   LOG(info) << fmt::format("\t- TMP Vertex position bins: {:.2f} MB", 3 * mConfig.nMaxROFs * sizeof(cub::KeyValuePair<int, int>) / MB);
//   LOG(info) << fmt::format("\t- Beam positions: {:.2f} MB", 2 * mConfig.nMaxROFs * sizeof(float) / MB);
//   LOG(info) << fmt::format("\t- Vertices: {:.2f} MB", mConfig.nMaxROFs * mConfig.maxVerticesCapacity * sizeof(Vertex) / MB);

//   return totalMemory;
// }

// template <int nLayers>
// template <unsigned char isTracker>
// void TimeFrameGPU<nLayers>::initialiseDevice(const TrackingParameters& trkParam)
// {
//   mTrackletSizeHost.resize(nLayers - 1, 0);
//   mCellSizeHost.resize(nLayers - 2, 0);
//   for (int iLayer{0}; iLayer < nLayers - 1; ++iLayer) { // Tracker and vertexer
//     mTrackletsD[iLayer] = Vector<Tracklet>{mConfig.trackletsCapacity, mConfig.trackletsCapacity};
//     auto thrustTrackletsBegin = thrust::device_ptr<Tracklet>(mTrackletsD[iLayer].get());
//     auto thrustTrackletsEnd = thrustTrackletsBegin + mConfig.trackletsCapacity;
//     thrust::fill(thrustTrackletsBegin, thrustTrackletsEnd, Tracklet{});
//     mTrackletsLookupTablesD[iLayer].resetWithInt(mClusters[iLayer].size());
//     if (iLayer < nLayers - 2) {
//       mCellsD[iLayer] = Vector<Cell>{mConfig.validatedTrackletsCapacity, mConfig.validatedTrackletsCapacity};
//       mCellsLookupTablesD[iLayer] = Vector<int>{mConfig.cellsLUTsize, mConfig.cellsLUTsize};
//       mCellsLookupTablesD[iLayer].resetWithInt(mConfig.cellsLUTsize);
//     }
//   }

//   for (auto iComb{0}; iComb < 2; ++iComb) { // Vertexer only
//     mNTrackletsPerClusterD[iComb] = Vector<int>{mConfig.clustersPerLayerCapacity, mConfig.clustersPerLayerCapacity};
//   }
//   mLines = Vector<Line>{mConfig.trackletsCapacity, mConfig.trackletsCapacity};
//   mNFoundLines = Vector<int>{mConfig.clustersPerLayerCapacity, mConfig.clustersPerLayerCapacity};
//   mNFoundLines.resetWithInt(mConfig.clustersPerLayerCapacity);
//   mNExclusiveFoundLines = Vector<int>{mConfig.clustersPerLayerCapacity, mConfig.clustersPerLayerCapacity};
//   mNExclusiveFoundLines.resetWithInt(mConfig.clustersPerLayerCapacity);
//   mUsedTracklets = Vector<unsigned char>{mConfig.trackletsCapacity, mConfig.trackletsCapacity};
//   discardResult(cudaMalloc(&mCUBTmpBuffers, mConfig.nMaxROFs * mConfig.tmpCUBBufferSize));
//   discardResult(cudaMalloc(&mDeviceFoundTracklets, (nLayers - 1) * sizeof(int)));
//   discardResult(cudaMemset(mDeviceFoundTracklets, 0, (nLayers - 1) * sizeof(int)));
//   discardResult(cudaMalloc(&mDeviceFoundCells, (nLayers - 2) * sizeof(int)));
//   discardResult(cudaMemset(mDeviceFoundCells, 0, (nLayers - 2) * sizeof(int)));
//   mXYCentroids = Vector<float>{2 * mConfig.nMaxROFs * mConfig.maxCentroidsXYCapacity, 2 * mConfig.nMaxROFs * mConfig.maxCentroidsXYCapacity};
//   mZCentroids = Vector<float>{mConfig.nMaxROFs * mConfig.maxLinesCapacity, mConfig.nMaxROFs * mConfig.maxLinesCapacity};
//   for (size_t i{0}; i < 3; ++i) {
//     mXYZHistograms[i] = Vector<int>{mConfig.nMaxROFs * mConfig.histConf.nBinsXYZ[i], mConfig.nMaxROFs * mConfig.histConf.nBinsXYZ[i]};
//   }
//   mTmpVertexPositionBins = Vector<cub::KeyValuePair<int, int>>{3 * mConfig.nMaxROFs, 3 * mConfig.nMaxROFs};
//   mBeamPosition = Vector<float>{2 * mConfig.nMaxROFs, 2 * mConfig.nMaxROFs};
//   mGPUVertices = Vector<Vertex>{mConfig.nMaxROFs * mConfig.maxVerticesCapacity, mConfig.nMaxROFs * mConfig.maxVerticesCapacity};
//   //////////////////////////////////////////////////////////////////////////////
//   constexpr int layers = isTracker ? nLayers : 3;
//   for (int iLayer{0}; iLayer < layers; ++iLayer) {
//     mClustersD[iLayer].reset(mClusters[iLayer].data(), static_cast<int>(mClusters[iLayer].size()));
//   }
//   if constexpr (isTracker) {
//     StaticTrackingParameters<nLayers> pars;
//     pars.set(trkParam);
//     checkGPUError(cudaMalloc(reinterpret_cast<void**>(&mDeviceTrackingParams), sizeof(gpu::StaticTrackingParameters<nLayers>)), __FILE__, __LINE__);
//     checkGPUError(cudaMalloc(reinterpret_cast<void**>(&mDeviceIndexTableUtils), sizeof(IndexTableUtils)), __FILE__, __LINE__);
//     checkGPUError(cudaMemcpy(mDeviceTrackingParams, &pars, sizeof(gpu::StaticTrackingParameters<nLayers>), cudaMemcpyHostToDevice), __FILE__, __LINE__);
//     checkGPUError(cudaMemcpy(mDeviceIndexTableUtils, &mIndexTableUtils, sizeof(IndexTableUtils), cudaMemcpyHostToDevice), __FILE__, __LINE__);
//     // Tracker-only: we don't need to copy data in vertexer
//     for (int iLayer{0}; iLayer < nLayers; ++iLayer) {
//       mUsedClustersD[iLayer].reset(mUsedClusters[iLayer].data(), static_cast<int>(mUsedClusters[iLayer].size()));
//       mTrackingFrameInfoD[iLayer].reset(mTrackingFrameInfo[iLayer].data(), static_cast<int>(mTrackingFrameInfo[iLayer].size()));
//       mClusterExternalIndicesD[iLayer].reset(mClusterExternalIndices[iLayer].data(), static_cast<int>(mClusterExternalIndices[iLayer].size()));
//       mROframesClustersD[iLayer].reset(mROframesClusters[iLayer].data(), static_cast<int>(mROframesClusters[iLayer].size()));
//       mIndexTablesD[iLayer].reset(mIndexTables[iLayer].data(), static_cast<int>(mIndexTables[iLayer].size()));
//     }
//   } else {
//     mIndexTablesD[0].reset(getIndexTableWhole(0).data(), static_cast<int>(getIndexTableWhole(0).size()));
//     mIndexTablesD[2].reset(getIndexTableWhole(2).data(), static_cast<int>(getIndexTableWhole(2).size()));
//   }

//   gpuThrowOnError();
// }

// template <int nLayers>
// void TimeFrameGPU<nLayers>::initialise(const int iteration,
//                                        const TrackingParameters& trkParam,
//                                        const int maxLayers)
// {
//   o2::its::TimeFrame::initialise(iteration, trkParam, maxLayers);
//   checkBufferSizes();
//   if (maxLayers < nLayers) {
//     initialiseDevice<false>(trkParam); // vertexer
//   } else {
//     initialiseDevice<true>(trkParam); // tracker
//   }
// }

// template <int nLayers>
// TimeFrameGPU<nLayers>::~TimeFrameGPU()
// {
//   discardResult(cudaFree(mCUBTmpBuffers));
//   discardResult(cudaFree(mDeviceFoundTracklets));
//   discardResult(cudaFree(mDeviceTrackingParams));
//   discardResult(cudaFree(mDeviceIndexTableUtils));
//   discardResult(cudaFree(mDeviceFoundCells));
// }

// template <int nLayers>
// void TimeFrameGPU<nLayers>::checkBufferSizes()
// {
//   for (int iLayer{0}; iLayer < nLayers; ++iLayer) {
//     if (mClusters[iLayer].size() > mConfig.clustersPerLayerCapacity) {
//       LOGP(error, "Number of clusters on layer {} is {} and exceeds the GPU configuration defined one: {}", iLayer, mClusters[iLayer].size(), mConfig.clustersPerLayerCapacity);
//     }
//     if (mTrackingFrameInfo[iLayer].size() > mConfig.clustersPerLayerCapacity) {
//       LOGP(error, "Number of tracking frame info on layer {} is {} and exceeds the GPU configuration defined one: {}", iLayer, mTrackingFrameInfo[iLayer].size(), mConfig.clustersPerLayerCapacity);
//     }
//     if (mClusterExternalIndices[iLayer].size() > mConfig.clustersPerLayerCapacity) {
//       LOGP(error, "Number of external indices on layer {} is {} and exceeds the GPU configuration defined one: {}", iLayer, mClusterExternalIndices[iLayer].size(), mConfig.clustersPerLayerCapacity);
//     }
//     if (mROframesClusters[iLayer].size() > mConfig.clustersPerROfCapacity) {
//       LOGP(error, "Size of clusters per roframe on layer {} is {} and exceeds the GPU configuration defined one: {}", iLayer, mROframesClusters[iLayer].size(), mConfig.clustersPerROfCapacity);
//     }
//   }
//   if (mNrof > mConfig.nMaxROFs) {
//     LOGP(error, "Number of ROFs in timeframe is {} and exceeds the GPU configuration defined one: {}", mNrof, mConfig.nMaxROFs);
//   }
// }

template class TimeFrameGPU<7>;
} // namespace gpu
} // namespace its
} // namespace o2
