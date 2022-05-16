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
using namespace constants::its2;

class TimeFrameGPUConfig;
namespace gpu
{

template <class T>
GPUhd() T* getPtrFromRuler(int index, T* src, const int* ruler, const int stride = 1)
{
  return src + ruler[index] * stride;
}

template <int NLayers = 7>
class TimeFrameGPU : public TimeFrame
{

 public:
  TimeFrameGPU();
  ~TimeFrameGPU();
  void initialiseDevice(const int maxLayers);

  void checkBufferSizes();
  void initialise(const int iteration,
                  const TrackingParameters& trkParam,
                  const int maxLayers);
  /// Getters
  float getDeviceMemory();
  Cluster* getDeviceClustersOnLayer(const int rofId, const int layerId) const;
  unsigned char* getDeviceUsedClustersOnLayer(const int rofId, const int layerId);
  int* getDeviceROframesClustersOnLayer(const int layerId) const { return mROframesClustersD[layerId].get(); }  
  int getNClustersLayer(const int rofId, const int layerId) const;
  TimeFrameGPUConfig& getConfig() { return mConfig; }

  // Vertexer only
  int* getDeviceNTrackletsCluster(int rofId, int combId);
  int* getDeviceIndexTableL0(const int rofId) { return mIndexTablesLayer0D.get() + rofId * (ZBins * PhiBins + 1); }
  int* getDeviceIndexTableL2(const int rofId) { return mIndexTablesLayer2D.get() + rofId * (ZBins * PhiBins + 1); }
  unsigned char* getDeviceUsedTracklets(const int rofId);
  Line* getDeviceLines(const int rofId);
  Tracklet* getDeviceTracklets(const int rofId, const int layerId);
  int* getDeviceNFoundLines(const int rofId);
  int* getDeviceExclusiveNFoundLines(const int rofId);
  int* getDeviceCUBBuffer(const size_t rofId);
  float* getDeviceXYCentroids(const int rofId);
  float* getDeviceZCentroids(const int rofId);
  int* getDeviceXHistograms(const int rofId);
  int* getDeviceYHistograms(const int rofId);
  int* getDeviceZHistograms(const int rofId);
#ifdef __HIPCC__
  hipcub::KeyValuePair<int, int>* getTmpVertexPositionBins(const int rofId);
#else
  cub::KeyValuePair<int, int>* getTmpVertexPositionBins(const int rofId);
#endif
  float* getDeviceBeamPosition(const int rofId);
  Vertex* getDeviceVertices(const int rofId);

 private:
  TimeFrameGPUConfig mConfig;

  // Per-layer information, do not expand at runtime
  std::array<Vector<Cluster>, NLayers> mClustersD;
  std::array<Vector<unsigned char>, NLayers> mUsedClustersD;
  std::array<Vector<TrackingFrameInfo>, NLayers> mTrackingFrameInfoD;
  std::array<Vector<int>, NLayers - 1> mIndexTablesD;
  std::array<Vector<int>, NLayers> mClusterExternalIndicesD;
  std::array<Vector<Tracklet>, NLayers - 1> mTrackletsD;
  std::array<Vector<int>, NLayers> mROframesClustersD; // layers x roframes
  int* mCUBTmpBuffers;                                 // don't know whether will be used by the tracker

  // Vertexer only
  Vector<Line> mLines;
  Vector<int> mIndexTablesLayer0D;
  Vector<int> mIndexTablesLayer2D;
  Vector<int> mNFoundLines;
  Vector<int> mNExclusiveFoundLines;
  Vector<unsigned char> mUsedTracklets;
  Vector<float> mXYCentroids;
  Vector<float> mZCentroids;
  std::array<Vector<int>, 2> mNTrackletsPerClusterD;
  std::array<Vector<int>, 3> mXYZHistograms;
  Vector<float> mBeamPosition;
  Vector<Vertex> mGPUVertices;
#ifdef __HIPCC__
  Vector<hipcub::KeyValuePair<int, int>> mTmpVertexPositionBins;
#else
  Vector<cub::KeyValuePair<int, int>> mTmpVertexPositionBins;
#endif
};

template <int NLayers>
inline Cluster* TimeFrameGPU<NLayers>::getDeviceClustersOnLayer(const int rofId, const int layerId) const
{
  return getPtrFromRuler<Cluster>(rofId, mClustersD[layerId].get(), mROframesClusters[layerId].data());
}

template <int NLayers>
inline unsigned char* TimeFrameGPU<NLayers>::getDeviceUsedClustersOnLayer(const int rofId, const int layerId)
{
  return getPtrFromRuler<unsigned char>(rofId, mUsedClustersD[layerId].get(), mROframesClusters[layerId].data());
}

template <int NLayers>
inline int TimeFrameGPU<NLayers>::getNClustersLayer(const int rofId, const int layerId) const
{
  return static_cast<int>(mROframesClusters[layerId][rofId + 1] - mROframesClusters[layerId][rofId]);
}

template <int NLayers>
inline int* TimeFrameGPU<NLayers>::getDeviceNTrackletsCluster(int rofId, int combId)
{
  return getPtrFromRuler<int>(rofId, mNTrackletsPerClusterD[combId].get(), mROframesClusters[1].data());
}

template <int NLayers>
inline unsigned char* TimeFrameGPU<NLayers>::getDeviceUsedTracklets(const int rofId)
{
  return getPtrFromRuler<unsigned char>(rofId, mUsedTracklets.get(), mROframesClusters[1].data(), mConfig.maxTrackletsPerCluster);
}

template <int NLayers>
inline Line* TimeFrameGPU<NLayers>::getDeviceLines(const int rofId)
{
  return getPtrFromRuler(rofId, mLines.get(), mROframesClusters[1].data());
}

template <int NLayers>
inline Tracklet* TimeFrameGPU<NLayers>::getDeviceTracklets(const int rofId, const int layerId)
{
  return getPtrFromRuler(rofId, mTrackletsD[layerId].get(), mROframesClusters[1].data(), mConfig.maxTrackletsPerCluster);
}

template <int NLayers>
inline int* TimeFrameGPU<NLayers>::getDeviceNFoundLines(const int rofId)
{
  return getPtrFromRuler<int>(rofId, mNFoundLines.get(), mROframesClusters[1].data());
}

template <int NLayers>
inline int* TimeFrameGPU<NLayers>::getDeviceExclusiveNFoundLines(const int rofId)
{
  return getPtrFromRuler<int>(rofId, mNExclusiveFoundLines.get(), mROframesClusters[1].data());
}

template <int NLayers>
inline int* TimeFrameGPU<NLayers>::getDeviceCUBBuffer(const size_t rofId)
{
  if (rofId >= mNrof) {
    LOG(error) << "Invalid rofId: " << rofId << "/" << mNrof << ", returning nullptr";
  }
  return reinterpret_cast<int*>(reinterpret_cast<char*>(mCUBTmpBuffers) + (static_cast<size_t>(rofId * mConfig.tmpCUBBufferSize) & 0xFFFFFFFFFFFFF000));
}

template <int NLayers>
inline float* TimeFrameGPU<NLayers>::getDeviceXYCentroids(const int rofId)
{
  return mXYCentroids.get() + 2 * rofId * mConfig.maxCentroidsXYCapacity;
}

template <int NLayers>
inline float* TimeFrameGPU<NLayers>::getDeviceZCentroids(const int rofId)
{
  return mZCentroids.get() + rofId * mConfig.maxLinesCapacity;
}

template <int NLayers>
inline int* TimeFrameGPU<NLayers>::getDeviceXHistograms(const int rofId)
{
  return mXYZHistograms[0].get() + rofId * mConfig.histConf.nBinsXYZ[0];
}

template <int NLayers>
inline int* TimeFrameGPU<NLayers>::getDeviceYHistograms(const int rofId)
{
  return mXYZHistograms[1].get() + rofId * mConfig.histConf.nBinsXYZ[1];
}

template <int NLayers>
inline int* TimeFrameGPU<NLayers>::getDeviceZHistograms(const int rofId)
{
  return mXYZHistograms[2].get() + rofId * mConfig.histConf.nBinsXYZ[2];
}

template <int NLayers>
#ifdef __HIPCC__
inline hipcub::KeyValuePair<int, int>* TimeFrameGPU<NLayers>::getTmpVertexPositionBins(const int rofId)
#else
inline cub::KeyValuePair<int, int>* TimeFrameGPU<NLayers>::getTmpVertexPositionBins(const int rofId)
#endif
{
  return mTmpVertexPositionBins.get() + 3 * rofId;
}

template <int NLayers>
inline float* TimeFrameGPU<NLayers>::getDeviceBeamPosition(const int rofId)
{
  return mBeamPosition.get() + 2 * rofId;
}

template <int NLayers>
inline Vertex* TimeFrameGPU<NLayers>::getDeviceVertices(const int rofId)
{
  return mGPUVertices.get() + rofId * mConfig.maxVerticesCapacity;
}

} // namespace gpu
} // namespace its
} // namespace o2
#endif