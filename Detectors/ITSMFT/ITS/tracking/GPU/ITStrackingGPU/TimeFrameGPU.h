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
#include "Array.h"
#include "UniquePointer.h"
#include "Vector.h"
#include "GPUCommonDef.h"
#include "GPUCommonMath.h"

namespace o2
{

// namespace itsmft
// {
// class Cluster;
// class CompClusterExt;
// class TopologyDictionary;
// class ROFRecord;
// } // namespace itsmft

namespace its
{
namespace gpu
{
template <int NLayers>
class TimeFrameGPU : public TimeFrame
{
 public:
  TimeFrameGPU();
  ~TimeFrameGPU();
  void loadToGPU();
  // GPUh() int loadROFrameData(gsl::span<o2::itsmft::ROFRecord> rofs,
  //                     gsl::span<const itsmft::CompClusterExt> clusters,
  //                     gsl::span<const unsigned char>::iterator& pattIt,
  //                     const itsmft::TopologyDictionary& dict,
  //                     const dataformats::MCTruthContainer<MCCompLabel>* mcLabels = nullptr);

 private:
  Array<Vector<TrackingFrameInfo>, NLayers> mTrackingFrameInfoGPU;
  Array<Vector<Cluster>, NLayers> mClustersGPU;
  Array<Vector<int>, NLayers> mClusterExternalIndicesGPU;
  Array<Vector<int>, NLayers> mROframesClustersGPU;
};
} // namespace gpu
} // namespace its
} // namespace o2
#endif