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
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
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

 private:
  Array<Vector<Cluster>, NLayers> mClustersD;
  Array<Vector<TrackingFrameInfo>, NLayers> mTrackingFrameInfoD;
  Array<Vector<int>, NLayers> mClusterExternalIndicesD;
  Array<Vector<int>, NLayers> mROframesClustersD;
};

} // namespace gpu
} // namespace its
} // namespace o2
#endif