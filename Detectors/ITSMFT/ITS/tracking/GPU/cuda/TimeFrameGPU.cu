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
#include "ITStrackingGPU/TracerGPU.h"

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

/////////////////////////////////////////////////////////////////////////////////////////
// GpuPartition
template <int nLayers>
GpuTimeFramePartition<nLayers>::~GpuTimeFramePartition()
{
  if (mAllocated) {
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
    checkGPUError(cudaFree(mCUBTmpBufferDevice));
    checkGPUError(cudaFree(mFoundTrackletsDevice));
    checkGPUError(cudaFree(mFoundCellsDevice));
  }
}

template <int nLayers>
void GpuTimeFramePartition<nLayers>::allocate(const size_t nrof)
{
  mNRof = nrof;
  for (int i = 0; i < nLayers; ++i) {
    checkGPUError(cudaMalloc(reinterpret_cast<void**>(&(mROframesClustersDevice[i])), sizeof(int) * nrof));
    checkGPUError(cudaMalloc(reinterpret_cast<void**>(&(mClustersDevice[i])), sizeof(Cluster) * mTFGconf->clustersPerROfCapacity * nrof));
    checkGPUError(cudaMalloc(reinterpret_cast<void**>(&(mUsedClustersDevice[i])), sizeof(unsigned char) * mTFGconf->clustersPerROfCapacity * nrof));
    checkGPUError(cudaMalloc(reinterpret_cast<void**>(&(mTrackingFrameInfoDevice[i])), sizeof(TrackingFrameInfo) * mTFGconf->clustersPerROfCapacity * nrof));
    checkGPUError(cudaMalloc(reinterpret_cast<void**>(&(mClusterExternalIndicesDevice[i])), sizeof(int) * mTFGconf->clustersPerROfCapacity * nrof));
    checkGPUError(cudaMalloc(reinterpret_cast<void**>(&(mIndexTablesDevice[i])), sizeof(int) * (256 * 128 + 1) * nrof));
    if (i < nLayers - 1) {
      checkGPUError(cudaMalloc(reinterpret_cast<void**>(&(mTrackletsLookupTablesDevice[i])), sizeof(int) * mTFGconf->clustersPerROfCapacity * nrof));
      checkGPUError(cudaMalloc(reinterpret_cast<void**>(&(mTrackletsDevice[i])), sizeof(Tracklet) * mTFGconf->maxTrackletsPerCluster * mTFGconf->clustersPerROfCapacity * nrof));
      if (i < nLayers - 2) {
        checkGPUError(cudaMalloc(reinterpret_cast<void**>(&(mCellsLookupTablesDevice[i])), sizeof(int) * mTFGconf->validatedTrackletsCapacity * nrof));
        checkGPUError(cudaMalloc(reinterpret_cast<void**>(&(mCellsDevice[i])), sizeof(Cell) * mTFGconf->validatedTrackletsCapacity * nrof));
      }
    }
  }
  checkGPUError(cudaMalloc(reinterpret_cast<void**>(&mCUBTmpBufferDevice), mTFGconf->tmpCUBBufferSize * nrof));
  checkGPUError(cudaMalloc(reinterpret_cast<void**>(&mLinesDevice), sizeof(Line) * mTFGconf->maxTrackletsPerCluster * mTFGconf->clustersPerROfCapacity * nrof));
  checkGPUError(cudaMalloc(reinterpret_cast<void**>(&mNFoundLinesDevice), sizeof(int) * mTFGconf->clustersPerROfCapacity * nrof));
  checkGPUError(cudaMalloc(reinterpret_cast<void**>(&mNExclusiveFoundLinesDevice), sizeof(int) * mTFGconf->clustersPerROfCapacity * nrof));

  /// Invariant allocations
  checkGPUError(cudaMalloc(reinterpret_cast<void**>(&mFoundTrackletsDevice), (nLayers - 1) * sizeof(int) * nrof));
  checkGPUError(cudaMalloc(reinterpret_cast<void**>(&mFoundCellsDevice), (nLayers - 2) * sizeof(int) * nrof));

  mAllocated = true;
}

template <int nLayers>
void GpuTimeFramePartition<nLayers>::reset(const size_t nrof, const Task task)
{
  if ((bool)task) { // Vertexer-only initialisation (cannot be constexpr: due to the presence of gpu raw calls can't be put in header)
    for (int i = 0; i < 2; i++) {
      auto thrustTrackletsBegin = thrust::device_ptr<Tracklet>(mTrackletsDevice[i]);
      auto thrustTrackletsEnd = thrustTrackletsBegin + mTFGconf->maxTrackletsPerCluster * mTFGconf->clustersPerROfCapacity * nrof;
      thrust::fill(thrustTrackletsBegin, thrustTrackletsEnd, Tracklet{});
      checkGPUError(cudaMemset(mTrackletsLookupTablesDevice[i], 0, sizeof(int) * mTFGconf->clustersPerROfCapacity * nrof));
    }
  } else {
    for (int i = 0; i < nLayers - 1; ++i) {
      checkGPUError(cudaMemset(mTrackletsLookupTablesDevice[i], 0, sizeof(int) * mTFGconf->clustersPerROfCapacity * nrof));
      auto thrustTrackletsBegin = thrust::device_ptr<Tracklet>(mTrackletsDevice[i]);
      auto thrustTrackletsEnd = thrustTrackletsBegin + mTFGconf->maxTrackletsPerCluster * mTFGconf->clustersPerROfCapacity * nrof;
      thrust::fill(thrustTrackletsBegin, thrustTrackletsEnd, Tracklet{});
      if (i < nLayers - 2) {
        checkGPUError(cudaMemset(mCellsLookupTablesDevice[i], 0, sizeof(int) * mTFGconf->cellsLUTsize * nrof));
      }
    }
    checkGPUError(cudaMemset(mFoundCellsDevice, 0, (nLayers - 2) * sizeof(int)));
  }
}

template <int nLayers>
size_t GpuTimeFramePartition<nLayers>::computeScalingSizeBytes(const int nrof, const TimeFrameGPUConfig& config)
{
  size_t rofsize = nLayers * sizeof(int);                                                                      // number of clusters per ROF
  rofsize += nLayers * sizeof(Cluster) * config.clustersPerROfCapacity;                                        // clusters
  rofsize += nLayers * sizeof(unsigned char) * config.clustersPerROfCapacity;                                  // used clusters flags
  rofsize += nLayers * sizeof(TrackingFrameInfo) * config.clustersPerROfCapacity;                              // tracking frame info
  rofsize += nLayers * sizeof(int) * config.clustersPerROfCapacity;                                            // external cluster indices
  rofsize += nLayers * sizeof(int) * (256 * 128 + 1);                                                          // index tables
  rofsize += (nLayers - 1) * sizeof(int) * config.clustersPerROfCapacity;                                      // tracklets lookup tables
  rofsize += (nLayers - 1) * sizeof(Tracklet) * config.maxTrackletsPerCluster * config.clustersPerROfCapacity; // tracklets
  rofsize += (nLayers - 2) * sizeof(int) * config.validatedTrackletsCapacity;                                  // cells lookup tables
  rofsize += (nLayers - 2) * sizeof(Cell) * config.validatedTrackletsCapacity;                                 // cells
  rofsize += sizeof(Line) * config.maxTrackletsPerCluster * config.clustersPerROfCapacity;                     // lines
  rofsize += sizeof(int) * config.clustersPerROfCapacity;                                                      // found lines
  rofsize += sizeof(int) * config.clustersPerROfCapacity;                                                      // found lines exclusive sum

  rofsize += (nLayers - 1) * sizeof(int); // total found tracklets
  rofsize += (nLayers - 2) * sizeof(int); // total found cells

  return rofsize * nrof;
}

template <int nLayers>
size_t GpuTimeFramePartition<nLayers>::computeFixedSizeBytes(const TimeFrameGPUConfig& config)
{
  size_t total = config.tmpCUBBufferSize;                  // CUB tmp buffers
  total += sizeof(gpu::StaticTrackingParameters<nLayers>); // static parameters loaded once
  return total;
}

template <int nLayers>
size_t GpuTimeFramePartition<nLayers>::computeRofPerPartition(const TimeFrameGPUConfig& config)
{
  return (config.maxGPUMemoryGB * GB / (float)(config.nTimeFramePartitions) - GpuTimeFramePartition<nLayers>::computeFixedSizeBytes(config)) / (float)GpuTimeFramePartition<nLayers>::computeScalingSizeBytes(1, config);
}

/// Interface
template <int nLayers>
int* GpuTimeFramePartition<nLayers>::getDeviceROframesClusters(const int layer)
{
  return mROframesClustersDevice[layer];
}

template <int nLayers>
Cluster* GpuTimeFramePartition<nLayers>::getDeviceClusters(const int layer)
{
  return mClustersDevice[layer];
}

template <int nLayers>
unsigned char* GpuTimeFramePartition<nLayers>::getDeviceUsedClusters(const int layer)
{
  return mUsedClustersDevice[layer];
}

template <int nLayers>
TrackingFrameInfo* GpuTimeFramePartition<nLayers>::getDeviceTrackingFrameInfo(const int layer)
{
  return mTrackingFrameInfoDevice[layer];
}

template <int nLayers>
int* GpuTimeFramePartition<nLayers>::getDeviceClusterExternalIndices(const int layer)
{
  return mClusterExternalIndicesDevice[layer];
}

template <int nLayers>
int* GpuTimeFramePartition<nLayers>::getDeviceIndexTables(const int layer)
{
  return mIndexTablesDevice[layer];
}

template <int nLayers>
Tracklet* GpuTimeFramePartition<nLayers>::getDeviceTracklets(const int layer)
{
  return mTrackletsDevice[layer];
}

template <int nLayers>
int* GpuTimeFramePartition<nLayers>::getDeviceTrackletsLookupTables(const int layer)
{
  return mTrackletsLookupTablesDevice[layer];
}

template <int nLayers>
Cell* GpuTimeFramePartition<nLayers>::getDeviceCells(const int layer)
{
  return mCellsDevice[layer];
}

template <int nLayers>
int* GpuTimeFramePartition<nLayers>::getDeviceCellsLookupTables(const int layer)
{
  return mCellsLookupTablesDevice[layer];
}

// Load data
template <int nLayers>
void GpuTimeFramePartition<nLayers>::copyDeviceData(const size_t startRof, const int maxLayers)
{
  for (int i = 0; i < maxLayers; ++i) {
    mHostClusters[i] = mTimeFramePtr->getClustersPerROFrange(startRof, mNRof, i);
    if (mHostClusters[i].size() > mTFGconf->clustersPerROfCapacity * mNRof) {
      LOGP(warning, "Excess of expected clusters on layer {}, resizing to config value: {}, will lose information!", i, mTFGconf->clustersPerROfCapacity * mNRof);
    }
    checkGPUError(cudaMemcpy(mClustersDevice[i], mHostClusters[i].data(),
                             (int)std::min(mHostClusters[i].size() * mNRof, mTFGconf->clustersPerROfCapacity * mNRof) * sizeof(Cluster), cudaMemcpyHostToDevice));
  }
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
  // checkGPUError(cudaFree(mTrackingParamsDevice));
}

template <int nLayers>
void TimeFrameGPU<nLayers>::initialise(const int iteration,
                                       const TrackingParameters& trkParam,
                                       const int maxLayers)
{
  initDevice(mGpuConfig.nTimeFramePartitions);
  RANGE("TimeFrame initialisation", 1);
  o2::its::TimeFrame::initialise(iteration, trkParam, maxLayers);
}

template <int nLayers>
void TimeFrameGPU<nLayers>::initDevice(const int partitions)
{
  StaticTrackingParameters<nLayers> pars;
  checkGPUError(cudaMalloc(reinterpret_cast<void**>(&mTrackingParamsDevice), sizeof(gpu::StaticTrackingParameters<nLayers>)));
  checkGPUError(cudaMemcpy(mTrackingParamsDevice, &pars, sizeof(gpu::StaticTrackingParameters<nLayers>), cudaMemcpyHostToDevice));
  mMemPartitions.resize(partitions, GpuTimeFramePartition<nLayers>{static_cast<TimeFrame*>(this), mGpuConfig});
  LOGP(debug, "Size of fixed part is: {} MB", GpuTimeFramePartition<nLayers>::computeFixedSizeBytes(mGpuConfig) / MB);
  LOGP(debug, "Size of scaling part is: {} MB", GpuTimeFramePartition<nLayers>::computeScalingSizeBytes(GpuTimeFramePartition<nLayers>::computeRofPerPartition(mGpuConfig), mGpuConfig) / MB);
  LOGP(info, "Going to allocate {} partitions containing {} rofs each.", partitions, GpuTimeFramePartition<nLayers>::computeRofPerPartition(mGpuConfig));

  initDevicePartitions(GpuTimeFramePartition<nLayers>::computeRofPerPartition(mGpuConfig));
}

template <int nLayers>
void TimeFrameGPU<nLayers>::initDevicePartitions(const int nRof)
{
  if (mInitialised) {
    return;
  } else {
    mInitialised = true;
  }
  if (!mMemPartitions.size()) {
    LOGP(fatal, "gpu-tracking: TimeFrame GPU partitions not created");
  }
  for (auto& partition : mMemPartitions) {
    partition.allocate(nRof);
  }
}

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
//   discardResult(cudaMalloc(&mCUBTmpBufferDevice, mConfig.nMaxROFs * mConfig.tmpCUBBufferSize));
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
//   discardResult(cudaFree(mCUBTmpBufferDevice));
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
template class GpuTimeFramePartition<7>;
} // namespace gpu
} // namespace its
} // namespace o2
