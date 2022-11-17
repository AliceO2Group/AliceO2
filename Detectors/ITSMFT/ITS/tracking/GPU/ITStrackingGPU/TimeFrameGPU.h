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
#include "Array.h"
#include "Vector.h"

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
template <int nLayers>

class GpuTimeFramePartition
{
 public:
  GpuTimeFramePartition() = default;
  ~GpuTimeFramePartition();
  void initDevice(const size_t, const TimeFrameGPUConfig&);

 private:
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

  int* mCUBTmpBuffers;
  int* mFoundTrackletsDevice;
  int* mFoundCellsDevice;
  IndexTableUtils* mIndexTableUtilsDevice;

  /// Vertexer only
  Line* mLines;
  int* mNFoundLines;
  int* mNExclusiveFoundLines;
  unsigned char* mUsedTracklets;
  ///

  size_t mNRof;
};

template <int nLayers = 7>
class TimeFrameGPU : public TimeFrame
{
 public:
  TimeFrameGPU();
  void initDevice();
  void initDevicePartitions(const int nRof)
  {
    for (auto* partition : mMemPartitions) {
      partition->initDevice(nRof, mGpuConfig);
    }
  }

 private:
  std::vector<GpuTimeFramePartition<nLayers>*> mMemPartitions; // Vector of pointers to GPU objects
  TimeFrameGPUConfig mGpuConfig;

  // Device pointers
  gpu::StaticTrackingParameters<nLayers>* mTrackingParamsDevice;
};

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

// template <int nLayers>
// struct StaticTrackingParameters {
//   StaticTrackingParameters<nLayers>& operator=(const StaticTrackingParameters<nLayers>& t) = default;
//   void set(const TrackingParameters& pars);

//   /// General parameters
//   int ClusterSharing = 0;
//   int MinTrackLength = nLayers;
//   float NSigmaCut = 5;
//   float PVres = 1.e-2f;
//   int DeltaROF = 0;
//   int ZBins{256};
//   int PhiBins{128};

//   /// Cell finding cuts
//   float CellDeltaTanLambdaSigma = 0.007f;
// };

// template <int nLayers>
// void StaticTrackingParameters<nLayers>::set(const TrackingParameters& pars)
// {
//   ClusterSharing = pars.ClusterSharing;
//   MinTrackLength = pars.MinTrackLength;
//   NSigmaCut = pars.NSigmaCut;
//   PVres = pars.PVres;
//   DeltaROF = pars.DeltaROF;
//   ZBins = pars.ZBins;
//   PhiBins = pars.PhiBins;
//   CellDeltaTanLambdaSigma = pars.CellDeltaTanLambdaSigma;
// }

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