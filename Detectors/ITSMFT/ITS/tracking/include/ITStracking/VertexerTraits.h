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
/// \file VertexerTraits.h
/// \brief Class to compute the primary vertex in ITS from tracklets
/// \author matteo.concas@cern.ch

#ifndef O2_ITS_TRACKING_VERTEXER_TRAITS_H_
#define O2_ITS_TRACKING_VERTEXER_TRAITS_H_

#include <array>
#include <string>
#include <vector>

#include "ITStracking/Cluster.h"
#include "ITStracking/ClusterLines.h"
#include "ITStracking/Configuration.h"
#include "ITStracking/Definitions.h"
#include "ITStracking/IndexTableUtils.h"
#include "ITStracking/TimeFrame.h"
#include "ITStracking/Tracklet.h"

#include "GPUCommonDef.h"
#include "GPUCommonMath.h"

namespace o2
{
class MCCompLabel;

namespace its
{
class ROframe;
using constants::its::LayersNumberVertexer;

enum class TrackletMode {
  Layer0Layer1 = 0,
  Layer1Layer2 = 2
};

class VertexerTraits
{
 public:
  VertexerTraits() = default;
  virtual ~VertexerTraits() = default;

  GPUhd() static constexpr int4 getEmptyBinsRect()
  {
    return int4{0, 0, 0, 0};
  }
  GPUhd() const int4 getBinsRect(const Cluster&, const int, const float, float maxdeltaz, float maxdeltaphi);
  GPUhd() const int2 getPhiBins(float phi, float deltaPhi);

  GPUhd() static const int4 getBinsRect(const Cluster&, const int, const float, float maxdeltaz, float maxdeltaphi, const IndexTableUtils&);
  GPUhd() static const int2 getPhiBins(float phi, float deltaPhi, const IndexTableUtils&);

  // virtual vertexer interface
  virtual void initialise(const TrackingParameters& trackingParams, const int iteration = 0);
  virtual void computeTracklets(const int iteration = 0);
  virtual void computeTrackletMatching(const int iteration = 0);
  virtual void computeVertices(const int iteration = 0);
  virtual void adoptTimeFrame(TimeFrame* tf);
  virtual void updateVertexingParameters(const std::vector<VertexingParameters>& vrtPar, const TimeFrameGPUParameters& gpuTfPar);
  // Hybrid
  virtual void initialiseHybrid(const TrackingParameters& trackingParams, const int iteration = 0) { initialise(trackingParams, iteration); };
  virtual void computeTrackletsHybrid(const int iteration = 0) { computeTracklets(iteration); };
  virtual void computeTrackletMatchingHybrid(const int iteration = 0) { computeTrackletMatching(iteration); };
  virtual void computeVerticesHybrid(const int iteration = 0) { computeVertices(iteration); };
  virtual void adoptTimeFrameHybrid(TimeFrame* tf) { adoptTimeFrame(tf); };

  void computeVerticesInRof(int,
                            gsl::span<const o2::its::Line>&,
                            std::vector<bool>&,
                            std::vector<o2::its::ClusterLines>&,
                            std::array<float, 2>&,
                            std::vector<Vertex>&,
                            std::vector<int>&,
                            TimeFrame*,
                            std::vector<o2::MCCompLabel>*,
                            const int iteration = 0);

  static const std::vector<std::pair<int, int>> selectClusters(const int* indexTable,
                                                               const std::array<int, 4>& selectedBinsRect,
                                                               const IndexTableUtils& utils);

  // utils
  std::vector<VertexingParameters>& getVertexingParameters() { return mVrtParams; }
  std::vector<VertexingParameters> getVertexingParameters() const { return mVrtParams; }
  void setIsGPU(const unsigned char isgpu) { mIsGPU = isgpu; };
  void setVertexingParameters(std::vector<VertexingParameters>& vertParams) { mVrtParams = vertParams; }
  unsigned char getIsGPU() const { return mIsGPU; };
  void dumpVertexerTraits();
  void setNThreads(int n);
  int getNThreads() const { return mNThreads; }

  template <typename T = o2::MCCompLabel>
  static std::pair<T, float> computeMain(const std::vector<T>& elements)
  {
    T elem;
    size_t maxCount = 0;
    for (auto& element : elements) {
      size_t count = std::count(elements.begin(), elements.end(), element);
      if (count > maxCount) {
        maxCount = count;
        elem = element;
      }
    }
    return std::make_pair(elem, static_cast<float>(maxCount) / elements.size());
  }

 protected:
  unsigned char mIsGPU;
  int mNThreads = 1;

  std::vector<VertexingParameters> mVrtParams;
  IndexTableUtils mIndexTableUtils;
  std::vector<lightVertex> mVertices;

  // Frame related quantities
  TimeFrame* mTimeFrame = nullptr;
};

inline void VertexerTraits::initialise(const TrackingParameters& trackingParams, const int iteration)
{
  mTimeFrame->initialise(0, trackingParams, 3, (bool)(!iteration)); // iteration for initialisation must be 0 for correctly resetting the frame, we need to pass the non-reset flag for vertices as well, tho.
  setIsGPU(false);
}

GPUhdi() const int2 VertexerTraits::getPhiBins(float phi, float dPhi)
{
  return VertexerTraits::getPhiBins(phi, dPhi, mIndexTableUtils);
}

GPUhdi() const int2 VertexerTraits::getPhiBins(float phi, float dPhi, const IndexTableUtils& utils)
{
  return int2{utils.getPhiBinIndex(math_utils::getNormalizedPhi(phi - dPhi)),
              utils.getPhiBinIndex(math_utils::getNormalizedPhi(phi + dPhi))};
}

GPUhdi() const int4 VertexerTraits::getBinsRect(const Cluster& currentCluster, const int layerIndex,
                                                const float directionZIntersection, float maxdeltaz, float maxdeltaphi,
                                                const IndexTableUtils& utils)
{
  const float zRangeMin = directionZIntersection - 2 * maxdeltaz;
  const float phiRangeMin = currentCluster.phi - maxdeltaphi;
  const float zRangeMax = directionZIntersection + 2 * maxdeltaz;
  const float phiRangeMax = currentCluster.phi + maxdeltaphi;

  if (zRangeMax < -utils.getLayerZ(layerIndex + 1) ||
      zRangeMin > utils.getLayerZ(layerIndex + 1) || zRangeMin > zRangeMax) {
    return getEmptyBinsRect();
  }

  return int4{o2::gpu::GPUCommonMath::Max(0, utils.getZBinIndex(layerIndex + 1, zRangeMin)),
              utils.getPhiBinIndex(math_utils::getNormalizedPhi(phiRangeMin)),
              o2::gpu::GPUCommonMath::Min(utils.getNzBins() - 1, utils.getZBinIndex(layerIndex + 1, zRangeMax)),
              utils.getPhiBinIndex(math_utils::getNormalizedPhi(phiRangeMax))};
}

GPUhdi() const int4 VertexerTraits::getBinsRect(const Cluster& currentCluster, const int layerIndex,
                                                const float directionZIntersection, float maxdeltaz, float maxdeltaphi)
{
  return VertexerTraits::getBinsRect(currentCluster, layerIndex, directionZIntersection, maxdeltaz, maxdeltaphi, mIndexTableUtils);
}

inline void VertexerTraits::adoptTimeFrame(TimeFrame* tf) { mTimeFrame = tf; }

} // namespace its
} // namespace o2
#endif
