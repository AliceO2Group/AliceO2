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
#include "GPUCommonDef.h"
#include "GPUCommonMath.h"

#include "Array.h"
#include "Vector.h"

namespace o2
{

namespace its
{
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
  Cluster* getDeviceClustersOnLayer(const int rofId, const int layerId) const;
  int getDeviceNClustersLayer(const int rofId, const int layerId) const;
  std::array<Vector<Tracklet>, NLayers - 1>& getDeviceTracklets() { return mTrackletsD; }

  // Vertexer only
  Vector<int>& getDeviceIndexTableL0() { return mIndexTablesLayer0D; }
  Vector<int>& getDeviceIndexTableL2() { return mIndexTablesLayer2D; }
  // std::array<Vector<int>, NLayers - 1>& getDeviceIndexTables(int rofid) { return mIndexTablesD};
  int* getDeviceNTrackletsCluster(int rofId, int combId);

 private:
  TimeFrameGPUConfig conf;

  // Per-layer information, do not expand at runtime
  std::array<Vector<Cluster>, NLayers> mClustersD;
  std::array<Vector<TrackingFrameInfo>, NLayers> mTrackingFrameInfoD;
  std::array<Vector<int>, NLayers - 1> mIndexTablesD;
  std::array<Vector<int>, NLayers> mClusterExternalIndicesD;
  std::array<Vector<int>, NLayers> mROframesClustersD;
  std::array<Vector<Tracklet>, NLayers - 1> mTrackletsD;

  // Vertexer only
  Vector<int> mIndexTablesLayer0D;
  Vector<int> mIndexTablesLayer2D;
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
inline int TimeFrameGPU<NLayers>::getDeviceNClustersLayer(const int rofId, const int layerId) const
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

} // namespace gpu
} // namespace its
} // namespace o2
#endif