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

// #include "ITStracking/Cell.h"
// #include "ITStracking/Cluster.h"
// #include "ITStracking/Configuration.h"
// #include "ITStracking/Constants.h"
// #include "ITStracking/Definitions.h"
// #include "ITStracking/Road.h"
// #include "ITStracking/Tracklet.h"
// #include "ITStracking/IndexTableUtils.h"

// #include "SimulationDataFormat/MCCompLabel.h"
// #include "SimulationDataFormat/MCTruthContainer.h"

// #include "ReconstructionDataFormats/Vertex.h"

#include "ITStracking/TimeFrame.h"
#include "GPUCommonDef.h"
#include "GPUCommonMath.h"

#include "Array.h"
#include "Vector.h"
namespace o2
{

namespace its
{
namespace gpu
{
template <int NLayers = 7>
class TimeFrameGPU : public TimeFrame
{

 public:
  TimeFrameGPU();
  ~TimeFrameGPU();
  void loadToDevice(const int maxLayers);
  void initialise(const int iteration,
                  const MemoryParameters& memParam,
                  const TrackingParameters& trkParam,
                  const int maxLayers);
  Cluster* getDeviceClustersOnLayer(const int rofId, const int layerId) const;
  size_t getDeviceNClustersLayer(const int rofId, const int layerId) const;
  Array<Vector<Tracklet>, NLayers>& getDeviceTracklets() { return mTrackletsD; }

  // Vertexer only
  int* getDeviceIndexTableL0() const { return mIndexTablesLayer0D.get(); }
  int* getDeviceNTrackletsCluster() { return }

 private:
  // Per-layer information, do not expand at runtime
  std::array<Vector<Cluster>, NLayers> mClustersD;
  std::array<Vector<TrackingFrameInfo>, NLayers> mTrackingFrameInfoD;
  std::array<Vector<int>, NLayers - 1> mIndexTablesD;
  std::array<Vector<int>, NLayers> mClusterExternalIndicesD;
  std::array<Vector<int>, NLayers> mROframesClustersD;
  std::array<Vector<Tracklet>, NLayers - 1> mTrackletsD;

  // Vertexer only
  Vector<int> mIndexTablesLayer0D;
  std::vector<std::array<Vector<int>, 2>> mNTrackletsPerClusterD; // TODO: remove in favour of mNTrackletsPerROf
  std::vector < Vector<>
};

template <int NLayers>
inline Cluster* TimeFrameGPU<NLayers>::getDeviceClustersOnLayer(const int rofId, const int layerId) const
{
  if (rofId < 0 || rofId >= mNrof) {
    LOGP(info, "Invalid rofId: {}/{}, passing nullptr", rofId, mNrof);
    return nullptr;
  }
  return mClustersD[layerId].get() + mROframesClusters[layerId][rofId];
}

template <int NLayers>
inline size_t TimeFrameGPU<NLayers>::getDeviceNClustersLayer(const int rofId, const int layerId) const
{
  if (rofId < 0 || rofId >= mNrof) {
    LOGP(info, "Invalid rofId: {}/{}, passing nullptr", rofId, mNrof);
    return nullptr;
  }
  return mROframesClusters[layerId][rofId + 1] - mROframesClusters[layerId][rofId];
}

} // namespace gpu
} // namespace its
} // namespace o2
#endif