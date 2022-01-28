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

#include "ITStrackingGPU/TimeFrameGPU.h"

// #include "DataFormatsITSMFT/Cluster.h"
// #include "DataFormatsITSMFT/CompCluster.h"
// #include "DataFormatsITSMFT/ROFRecord.h"
// #include "DataFormatsITSMFT/TopologyDictionary.h"
// #include "ITSBase/GeometryTGeo.h"
// #include "ITSMFTBase/SegmentationAlpide.h"

// namespace
// {
// struct ClusterHelper {
//   float phi;
//   float r;
//   int bin;
//   int ind;
// };

// float MSangle(float mass, float p, float xX0)
// {
//   float beta = p / o2::gpu::GPUCommonMath::Hypot(mass, p);
//   return 0.0136f * o2::gpu::GPUCommonMath::Sqrt(xX0) * (1.f + 0.038f * o2::gpu::GPUCommonMath::Log(xX0)) / (beta * p);
// }

// float Sq(float v)
// {
//   return v * v;
// }
// } // namespace

namespace o2
{
namespace its
{
namespace gpu
{
template <int NLayers>
TimeFrameGPU<NLayers>::TimeFrameGPU()
{
  for (auto iLayer{0}; iLayer < NLayers; ++iLayer) {
    std::cout << "Built for layer: " << iLayer << std::endl;
    mClustersGPU[iLayer] = Vector<Cluster>{(int)5e5, (int)5e5};
    mTrackingFrameInfoGPU[iLayer] = Vector<TrackingFrameInfo>{(int)5e5, (int)5e5};
    mClusterExternalIndicesGPU[iLayer] = Vector<int>{(int)5e5, (int)5e5};
    mROframesClustersGPU[iLayer] = Vector<int>{(int)5e5, (int)5e5};
  }
}

template <int NLayers>
TimeFrameGPU<NLayers>::~TimeFrameGPU()
{
}

template <int NLayers>
void TimeFrameGPU<NLayers>::loadToDevice()
{
  for (auto iLayer{0}; iLayer < NLayers; ++iLayer) {
    std::cout << "Loading layer: " << iLayer << std::endl;
    mClustersGPU[iLayer].reset(mClusters[iLayer].data(), mClusters[iLayer].size());
    mTrackingFrameInfoGPU[iLayer].reset(mTrackingFrameInfo[iLayer].data(), mTrackingFrameInfo[iLayer].size());
    mClusterExternalIndicesGPU[iLayer].reset(mClusterExternalIndices[iLayer].data(), mClusterExternalIndices[iLayer].size());
    mROframesClustersGPU[iLayer].reset(mROframesClusters[iLayer].data(), mROframesClusters[iLayer].size());
  }
}

template class TimeFrameGPU<7>;
} // namespace gpu
} // namespace its
} // namespace o2