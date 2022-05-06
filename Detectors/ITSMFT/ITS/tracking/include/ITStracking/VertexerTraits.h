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

struct lightVertex {
  lightVertex(float x, float y, float z, std::array<float, 6> rms2, int cont, float avgdis2, int stamp);
  float mX;
  float mY;
  float mZ;
  std::array<float, 6> mRMS2;
  float mAvgDistance2;
  int mContributors;
  int mTimeStamp;
};

struct ClusterMCLabelInfo {
  int TrackId;
  int MotherId;
  int EventId;
  float Pt;
};

enum class VertexerDebug : unsigned int {
  TrackletTreeAll = 0x1 << 1,
  LineTreeAll = 0x1 << 2,
  CombinatoricsTreeAll = 0x1 << 3,
  LineSummaryAll = 0x1 << 4,
  HistCentroids = 0x1 << 5
};

enum class TrackletMode {
  Layer0Layer1 = 0,
  Layer1Layer2 = 2
};

inline lightVertex::lightVertex(float x, float y, float z, std::array<float, 6> rms2, int cont, float avgdis2, int stamp) : mX{x}, mY{y}, mZ{z}, mRMS2{rms2}, mAvgDistance2{avgdis2}, mContributors{cont}, mTimeStamp{stamp}
{
}

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
  virtual void initialise(const MemoryParameters& memParams, const TrackingParameters& trackingParams);
  virtual void computeTracklets();
  virtual void computeTrackletMatching();
  // virtual void computeMCFiltering();
  // virtual void filterTrackletsWithMC(std::vector<Tracklet>&,
  //                                    std::vector<Tracklet>&,
  //                                    std::vector<int>&,
  //                                    std::vector<int>&,
  //                                    const int);

  // virtual void computeTrackletsPureMontecarlo();
  virtual void computeVertices();
  // virtual void computeHistVertices();

  void updateVertexingParameters(const VertexingParameters& vrtPar);
  VertexingParameters getVertexingParameters() const { return mVrtParams; }
  static const std::vector<std::pair<int, int>> selectClusters(const int* indexTable,
                                                               const std::array<int, 4>& selectedBinsRect,
                                                               const IndexTableUtils& utils);
  std::vector<lightVertex> getVertices() const { return mVertices; }

  // utils
  virtual void adoptTimeFrame(TimeFrame* tf);
  void setIsGPU(const unsigned char isgpu) { mIsGPU = isgpu; };
  unsigned char getIsGPU() const { return mIsGPU; };
  void dumpVertexerTraits();

  void setDebugFlag(VertexerDebug flag, const unsigned char on);
  unsigned char isDebugFlag(const VertexerDebug& flags) const;
  unsigned int getDebugFlags() const { return static_cast<unsigned int>(mDBGFlags); }

 protected:
  unsigned char mIsGPU;
  unsigned int mDBGFlags = 0;

  VertexingParameters mVrtParams;
  IndexTableUtils mIndexTableUtils;
  std::vector<lightVertex> mVertices;

  // Frame related quantities
  TimeFrame* mTimeFrame = nullptr;
};

inline void VertexerTraits::initialise(const MemoryParameters& memParams, const TrackingParameters& trackingParams)
{
  if (!mIndexTableUtils.getNzBins()) {
    updateVertexingParameters(mVrtParams);
  }
  mTimeFrame->initialise(0, memParams, trackingParams, 3);
  setIsGPU(false);
}

inline void VertexerTraits::updateVertexingParameters(const VertexingParameters& vrtPar)
{
  mVrtParams = vrtPar;
  mIndexTableUtils.setTrackingParameters(vrtPar);
  mVrtParams.phiSpan = static_cast<int>(std::ceil(mIndexTableUtils.getNphiBins() * mVrtParams.phiCut /
                                                  constants::math::TwoPi));
  mVrtParams.zSpan = static_cast<int>(std::ceil(mVrtParams.zCut * mIndexTableUtils.getInverseZCoordinate(0)));
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

// debug
inline void VertexerTraits::setDebugFlag(VertexerDebug flag, const unsigned char on = true)
{
  if (on) {
    mDBGFlags |= static_cast<unsigned int>(flag);
  } else {
    mDBGFlags &= ~static_cast<unsigned int>(flag);
  }
}

inline unsigned char VertexerTraits::isDebugFlag(const VertexerDebug& flags) const
{
  return mDBGFlags & static_cast<unsigned int>(flags);
}

inline void VertexerTraits::adoptTimeFrame(TimeFrame* tf) { mTimeFrame = tf; }

} // namespace its
} // namespace o2
#endif
