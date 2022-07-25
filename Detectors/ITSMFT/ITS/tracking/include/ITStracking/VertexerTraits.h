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
  virtual void initialise(const TrackingParameters& trackingParams);
  virtual void computeTracklets();
  virtual void computeTrackletMatching();
  virtual void computeVertices();
  virtual void adoptTimeFrame(TimeFrame* tf);
  virtual void updateVertexingParameters(const VertexingParameters& vrtPar);
  // virtual void computeHistVertices();

  VertexingParameters getVertexingParameters() const { return mVrtParams; }
  static const std::vector<std::pair<int, int>> selectClusters(const int* indexTable,
                                                               const std::array<int, 4>& selectedBinsRect,
                                                               const IndexTableUtils& utils);
  std::vector<lightVertex> getVertices() const { return mVertices; }

  // utils

  void setIsGPU(const unsigned char isgpu) { mIsGPU = isgpu; };
  unsigned char getIsGPU() const { return mIsGPU; };
  void dumpVertexerTraits();

 protected:
  unsigned char mIsGPU;

  VertexingParameters mVrtParams;
  IndexTableUtils mIndexTableUtils;
  std::vector<lightVertex> mVertices;

  // Frame related quantities
  TimeFrame* mTimeFrame = nullptr;
};

inline void VertexerTraits::initialise(const TrackingParameters& trackingParams)
{
  if (!mIndexTableUtils.getNzBins()) {
    updateVertexingParameters(mVrtParams);
  }
  mTimeFrame->initialise(0, trackingParams, 3);
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
