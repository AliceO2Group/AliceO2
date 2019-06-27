// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
///
/// \file VertexerTraits.h
/// \brief
///

#ifndef O2_ITS_TRACKING_VERTEXER_TRAITS_H_
#define O2_ITS_TRACKING_VERTEXER_TRAITS_H_

#include <array>
#include <vector>

#include "ITStracking/Cluster.h"
#include "ITStracking/Configuration.h"
#include "ITStracking/ClusterLines.h"
#include "ITStracking/Definitions.h"
#include "ITStracking/Tracklet.h"

#include "GPUCommonMath.h"

namespace o2
{

namespace its
{

class ROframe;

using constants::IndexTable::PhiBins;
using constants::IndexTable::ZBins;
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

inline lightVertex::lightVertex(float x, float y, float z, std::array<float, 6> rms2, int cont, float avgdis2, int stamp) : mX(x), mY(y), mZ(z), mRMS2(rms2), mAvgDistance2(avgdis2), mContributors(cont), mTimeStamp(stamp)
{
}

class VertexerTraits
{
 public:
  VertexerTraits();
  virtual ~VertexerTraits() = default;
  GPU_HOST_DEVICE static constexpr int4 getEmptyBinsRect() { return int4{ 0, 0, 0, 0 }; }
  GPU_DEVICE static const int4 getBinsRect(const Cluster&, const int, const float, float maxdeltaz, float maxdeltaphi);
  GPU_HOST_DEVICE static const int4 getBinsRect2(const Cluster&, const int, const float, float maxdeltaz, float maxdeltaphi);
  GPU_HOST_DEVICE static const int2 getPhiBins(const int layerIndex, float phi, float deltaPhi);

  // virtual vertexer interface
  virtual void reset();
  virtual void initialise(ROframe*);
  virtual void computeTracklets(const bool useMCLabel = false);
  virtual void computeTrackletsPureMontecarlo();
  virtual void computeVertices();

  void updateVertexingParameters(const VertexingParameters& vrtPar);
  VertexingParameters getVertexingParameters() const { return mVrtParams; }
  static const std::vector<std::pair<int, int>> selectClusters(const std::array<int, ZBins * PhiBins + 1>& indexTable,
                                                               const std::array<int, 4>& selectedBinsRect);
  std::vector<lightVertex> getVertices() const { return mVertices; }

  // utils
  void setIsGPU(const bool);
  void dumpVertexerTraits();
  void arrangeClusters(ROframe*);
  std::vector<int> getMClabelsLayer(const int layer) const;

  // debug starts here
  std::vector<Line> mTracklets;
  std::vector<Tracklet> mComb01;
  std::vector<Tracklet> mComb12;
  std::array<std::vector<Cluster>, constants::its::LayersNumberVertexer> mClusters;
  std::vector<std::array<float, 7>> mDeltaTanlambdas;
  std::vector<std::array<float, 6>> mLinesData;
  std::vector<std::array<float, 4>> mCentroids;
  void processLines();

 protected:
  bool mIsGPU;
  VertexingParameters mVrtParams;
  std::array<std::array<int, ZBins * PhiBins + 1>, LayersNumberVertexer> mIndexTables;
  std::vector<lightVertex> mVertices;

  // Frame related quantities
  std::array<std::vector<bool>, 2> mUsedClusters;
  o2::its::ROframe* mEvent;
  uint32_t mROframe;

  std::array<float, 3> mAverageClustersRadii;
  float mDeltaRadii10, mDeltaRadii21;
  float mMaxDirectorCosine3;
  std::vector<ClusterLines> mTrackletClusters;
};

// inline VertexerTraits::~VertexerTraits()
// {
//   // nothing
// }

inline void VertexerTraits::initialise(ROframe* event)
{
  reset();
  arrangeClusters(event);
}

inline void VertexerTraits::setIsGPU(const bool isgpu)
{
  mIsGPU = isgpu;
}

inline void VertexerTraits::updateVertexingParameters(const VertexingParameters& vrtPar)
{
  mVrtParams = vrtPar;
}

inline GPU_DEVICE const int4 VertexerTraits::getBinsRect(const Cluster& currentCluster, const int layerIndex,
                                                         const float directionZIntersection, float maxdeltaz, float maxdeltaphi)
{
  const float zRangeMin = directionZIntersection - maxdeltaz;
  const float phiRangeMin = currentCluster.phiCoordinate - maxdeltaphi;
  const float zRangeMax = directionZIntersection + maxdeltaz;
  const float phiRangeMax = currentCluster.phiCoordinate + maxdeltaphi;

  if (zRangeMax < -constants::its::LayersZCoordinate()[layerIndex + 1] ||
      zRangeMin > constants::its::LayersZCoordinate()[layerIndex + 1] || zRangeMin > zRangeMax) {

    return getEmptyBinsRect();
  }

  return int4{ gpu::GPUCommonMath::Max(0, IndexTableUtils::getZBinIndex(layerIndex + 1, zRangeMin)),
               IndexTableUtils::getPhiBinIndex(MathUtils::getNormalizedPhiCoordinate(phiRangeMin)),
               gpu::GPUCommonMath::Min(constants::IndexTable::ZBins - 1, IndexTableUtils::getZBinIndex(layerIndex + 1, zRangeMax)),
               IndexTableUtils::getPhiBinIndex(MathUtils::getNormalizedPhiCoordinate(phiRangeMax)) };
}

inline GPU_HOST_DEVICE const int2 VertexerTraits::getPhiBins(const int layerIndex, float phi, float dPhi)
{
  return int2{ IndexTableUtils::getPhiBinIndex(MathUtils::getNormalizedPhiCoordinate(phi - dPhi)),
               IndexTableUtils::getPhiBinIndex(MathUtils::getNormalizedPhiCoordinate(phi + dPhi)) };
}

inline GPU_HOST_DEVICE const int4 VertexerTraits::getBinsRect2(const Cluster& currentCluster, const int layerIndex,
                                                               const float directionZIntersection, float maxdeltaz, float maxdeltaphi)
{
  const float zRangeMin = directionZIntersection - 2 * maxdeltaz;
  const float phiRangeMin = currentCluster.phiCoordinate - maxdeltaphi;
  const float zRangeMax = directionZIntersection + 2 * maxdeltaz;
  const float phiRangeMax = currentCluster.phiCoordinate + maxdeltaphi;

  if (zRangeMax < -constants::its::LayersZCoordinate()[layerIndex + 1] ||
      zRangeMin > constants::its::LayersZCoordinate()[layerIndex + 1] || zRangeMin > zRangeMax) {

    return getEmptyBinsRect();
  }

  return int4{ gpu::GPUCommonMath::Max(0, IndexTableUtils::getZBinIndex(layerIndex + 1, zRangeMin)),
               IndexTableUtils::getPhiBinIndex(MathUtils::getNormalizedPhiCoordinate(phiRangeMin)),
               gpu::GPUCommonMath::Min(constants::IndexTable::ZBins - 1, IndexTableUtils::getZBinIndex(layerIndex + 1, zRangeMax)),
               IndexTableUtils::getPhiBinIndex(MathUtils::getNormalizedPhiCoordinate(phiRangeMax)) };
}
extern "C" VertexerTraits* createVertexerTraits();

} // namespace its
} // namespace o2
#endif /* O2_ITS_TRACKING_VERTEXER_TRAITS_H_ */
