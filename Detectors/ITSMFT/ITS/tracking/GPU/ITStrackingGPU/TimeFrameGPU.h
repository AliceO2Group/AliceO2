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

#include "Array.h"
#include "Vector.h"

#include "GPUCommonDef.h"
#include "GPUCommonMath.h"

namespace o2
{

namespace its
{
using namespace constants::its2;

class TimeFrameGPUConfig;
namespace gpu
{
template <int NLayers = 7>
class TimeFrameGPU : public TimeFrame
{

 public:
  TimeFrameGPU();
  ~TimeFrameGPU();
  void loadToDevice(const int maxLayers);

  void checkBufferSizes();
  void initialise(const int iteration,
                  const MemoryParameters& memParam,
                  const TrackingParameters& trkParam,
                  const int maxLayers);
  /// Getters
  float getDeviceMemory();
  Cluster* getDeviceClustersOnLayer(const int rofId, const int layerId) const;
  int getNClustersLayer(const int rofId, const int layerId) const;
  std::array<Vector<Tracklet>, NLayers - 1>& getDeviceTracklets() { return mTrackletsD; }
  const TimeFrameGPUConfig& getConfig() const { return mConfig; }

  // Vertexer only
  int* getDeviceNTrackletsCluster(int rofId, int combId);
  int* getDeviceIndexTableL0(const int rofId) { return mIndexTablesLayer0D.get() + rofId * (ZBins * PhiBins + 1); }
  int* getDeviceIndexTableL2(const int rofId) { return mIndexTablesLayer2D.get() + rofId * (ZBins * PhiBins + 1); }
  unsigned char* getDeviceUsedTracklets(const int rofId);
  Line* getDeviceLines(const int rofid);
  int* getDeviceNFoundLines(const int rofid);
  int* getDeviceExclusiveNFoundLines(const int rofid);

 private:
  TimeFrameGPUConfig mConfig;

  // Per-layer information, do not expand at runtime
  std::array<Vector<Cluster>, NLayers> mClustersD;
  std::array<Vector<TrackingFrameInfo>, NLayers> mTrackingFrameInfoD;
  std::array<Vector<int>, NLayers - 1> mIndexTablesD;
  std::array<Vector<int>, NLayers> mClusterExternalIndicesD;
  std::array<Vector<int>, NLayers> mROframesClustersD;
  std::array<Vector<Tracklet>, NLayers - 1> mTrackletsD;
  Vector<int> mCUBTmpBuffers; // don't know whether will be used by the tracker

  // Vertexer only
  Vector<Line> mLines;
  Vector<int> mIndexTablesLayer0D;
  Vector<int> mIndexTablesLayer2D;
  Vector<int> mNFoundLines;
  Vector<int> mNExclusiveFoundLines;
  Vector<unsigned char> mUsedTracklets;
  std::array<Vector<int>, 2> mNTrackletsPerClusterD;
};

template <int NLayers>
inline Cluster* TimeFrameGPU<NLayers>::getDeviceClustersOnLayer(const int rofId, const int layerId) const
{
  if (rofId < 0 || rofId >= mNrof) {
    std::cout << "Invalid rofId: " << rofId << "/" << mNrof << ", returning nullptr" << std::endl;
    return nullptr;
  }
  return mClustersD[layerId].get() + mROframesClusters[layerId][rofId];
}

template <int NLayers>
inline int TimeFrameGPU<NLayers>::getNClustersLayer(const int rofId, const int layerId) const
{
  if (rofId < 0 || rofId >= mNrof) {
    std::cout << "Invalid rofId: " << rofId << "/" << mNrof << ", returning 0 as value" << std::endl;
    return 0;
  }
  return static_cast<int>(mROframesClusters[layerId][rofId + 1] - mROframesClusters[layerId][rofId]);
}

template <int NLayers>
inline int* TimeFrameGPU<NLayers>::getDeviceNTrackletsCluster(int rofId, int combId)
{
  if (rofId < 0 || rofId >= mNrof) {
    std::cout << "Invalid rofId: " << rofId << "/" << mNrof << ", returning nullptr" << std::endl;
  }
  return mNTrackletsPerClusterD[combId].get() + mROframesClusters[1][rofId];
}

template <int NLayers>
inline unsigned char* TimeFrameGPU<NLayers>::getDeviceUsedTracklets(const int rofId)
{
  if (rofId < 0 || rofId >= mNrof) {
    std::cout << "Invalid rofId: " << rofId << "/" << mNrof << ", returning nullptr" << std::endl;
  }
  return mUsedTracklets.get() + mROframesClusters[1][rofId];
}

template <int NLayers>
inline Line* TimeFrameGPU<NLayers>::getDeviceLines(const int rofid)
{
  if (rofid < 0 || rofid >= mNrof) {
    std::cout << "Invalid rofId: " << rofid << "/" << mNrof << ", returning nullptr" << std::endl;
  }
  return mLines.get() + mROframesClusters[1][rofid];
}

template <int NLayers>
inline int* TimeFrameGPU<NLayers>::getDeviceNFoundLines(const int rofid)
{
  if (rofid < 0 || rofid >= mNrof) {
    std::cout << "Invalid rofId: " << rofid << "/" << mNrof << ", returning nullptr" << std::endl;
  }
  return mNFoundLines.get() + mROframesClusters[1][rofid];
}

template <int NLayers>
inline int* TimeFrameGPU<NLayers>::getDeviceExclusiveNFoundLines(const int rofid)
{
  if (rofid < 0 || rofid >= mNrof) {
    std::cout << "Invalid rofId: " << rofid << "/" << mNrof << ", returning nullptr" << std::endl;
  }
  return mNExclusiveFoundLines.get() + mROframesClusters[1][rofid];
}

} // namespace gpu
} // namespace its
} // namespace o2
#endif