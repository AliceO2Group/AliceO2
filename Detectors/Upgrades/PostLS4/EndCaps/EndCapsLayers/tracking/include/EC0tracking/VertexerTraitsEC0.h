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
/// \file VertexerTraitsEC0.h
/// \brief Class to compute the primary vertex in ITS from tracklets
/// \author matteo.concas@cern.ch

#ifndef O2_EC0_TRACKING_VERTEXER_TRAITS_H_
#define O2_EC0_TRACKING_VERTEXER_TRAITS_H_

#include <array>
#include <string>
#include <vector>

#include "EC0tracking/Cluster.h"
#include "EC0tracking/Configuration.h"
#include "EC0tracking/ClusterLines.h"
#include "EC0tracking/Definitions.h"
#ifdef _ALLOW_DEBUG_TREES_ITS_
#include "EC0tracking/StandaloneDebugger.h"
#endif
#include "EC0tracking/Tracklet.h"

#include "GPUCommonMath.h"
#include "GPUCommonDef.h"

namespace o2
{
class MCCompLabel;

namespace utils
{
class TreeStreamRedirector;
}

namespace ecl
{

class ROframe;

using constants::index_table::PhiBins;
using constants::index_table::ZBins;
using constants::ecl::LayersNumberVertexer;

struct lightVertex {
  lightVertex(float x, float y, float z, std::array<float, 6> rms2, int cont, float avgdis2, int stamp);
#ifdef _ALLOW_DEBUG_TREES_ITS_
  lightVertex(float x, float y, float z, std::array<float, 6> rms2, int cont, float avgdis2, int stamp, int eId, float pur);
#endif
  float mX;
  float mY;
  float mZ;
  std::array<float, 6> mRMS2;
  float mAvgDistance2;
  int mContributors;
  int mTimeStamp;
#ifdef _ALLOW_DEBUG_TREES_ITS_
  float mPurity;
  int mEventId;
#endif
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

inline lightVertex::lightVertex(float x, float y, float z, std::array<float, 6> rms2, int cont, float avgdis2, int stamp) : mX{x}, mY{y}, mZ{z}, mRMS2{rms2}, mAvgDistance2{avgdis2}, mContributors{cont}, mTimeStamp{stamp}
{
}
#ifdef _ALLOW_DEBUG_TREES_ITS_
inline lightVertex::lightVertex(float x, float y, float z, std::array<float, 6> rms2, int cont, float avgdis2, int stamp, int eId, float pur) : mX{x}, mY{y}, mZ{z}, mRMS2{rms2}, mAvgDistance2{avgdis2}, mContributors{cont}, mTimeStamp{stamp}, mEventId{eId}, mPurity{pur}
{
}
#endif

class VertexerTraitsEC0
{
 public:
#ifdef _ALLOW_DEBUG_TREES_ITS_
  VertexerTraitsEC0();
  virtual ~VertexerTraitsEC0();
#else
  VertexerTraitsEC0();
  virtual ~VertexerTraitsEC0() = default;
#endif

  GPUhd() static constexpr int4 getEmptyBinsRect()
  {
    return int4{0, 0, 0, 0};
  }
  GPUhd() static const int4 getBinsRect(const Cluster&, const int, const float, float maxdeltaz, float maxdeltaphi);
  GPUhd() static const int2 getPhiBins(float phi, float deltaPhi);

  // virtual vertexer interface
  virtual void reset();
  virtual void initialise(ROframe*);
  virtual void computeTracklets();
  virtual void computeTrackletMatching();
#ifdef _ALLOW_DEBUG_TREES_ITS_
  virtual void computeMCFiltering();
  virtual void filterTrackletsWithMC(std::vector<Tracklet>&,
                                     std::vector<Tracklet>&,
                                     std::vector<int>&,
                                     std::vector<int>&,
                                     const int);
#endif
  virtual void computeTrackletsPureMontecarlo();
  virtual void computeVertices();
  virtual void computeHistVertices();

  void updateVertexingParameters(const VertexingParameters& vrtPar);
  VertexingParameters getVertexingParameters() const { return mVrtParams; }
  static const std::vector<std::pair<int, int>> selectClusters(const std::array<int, ZBins * PhiBins + 1>& indexTable,
                                                               const std::array<int, 4>& selectedBinsRect);
  std::vector<lightVertex> getVertices() const { return mVertices; }

  // utils
  void setIsGPU(const unsigned char);
  void dumpVertexerTraitsEC0();
  void arrangeClusters(ROframe*);
  std::vector<int> getMClabelsLayer(const int layer) const;

  void setDebugFlag(VertexerDebug flag, const unsigned char on);
  unsigned char isDebugFlag(const VertexerDebug& flags) const;
  unsigned int getDebugFlags() const { return static_cast<unsigned int>(mDBGFlags); }

 protected:
  unsigned char mIsGPU;

  std::vector<Line> mTracklets;
  std::vector<Tracklet> mComb01;
  std::vector<Tracklet> mComb12;
  std::vector<int> mFoundTracklets01;
  std::vector<int> mFoundTracklets12;
  std::array<std::vector<Cluster>, constants::ecl::LayersNumberVertexer> mClusters;

  unsigned int mDBGFlags = 0;

#ifdef _ALLOW_DEBUG_TREES_ITS_
  StandaloneDebugger* mDebugger;
  std::vector<std::array<int, 2>> mAllowedTrackletPairs;
#endif

  VertexingParameters mVrtParams;
  std::array<std::array<int, ZBins * PhiBins + 1>, LayersNumberVertexer> mIndexTables;
  std::vector<lightVertex> mVertices;

  // Frame related quantities
  std::array<std::vector<unsigned char>, 2> mUsedClusters;
  ROframe* mEvent;
  uint32_t mROframe;

  std::array<float, 3> mAverageClustersRadii;
  float mDeltaRadii10, mDeltaRadii21;
  float mMaxDirectorCosine3;
  std::vector<ClusterLines> mTrackletClusters;
};

inline void VertexerTraitsEC0::initialise(ROframe* event)
{
  reset();
  arrangeClusters(event);
  setIsGPU(false);
}

inline void VertexerTraitsEC0::setIsGPU(const unsigned char isgpu)
{
  mIsGPU = isgpu;
}

inline void VertexerTraitsEC0::updateVertexingParameters(const VertexingParameters& vrtPar)
{
  mVrtParams = vrtPar;
}

GPUhdi() const int2 VertexerTraitsEC0::getPhiBins(float phi, float dPhi)
{
  return int2{index_table_utils::getPhiBinIndex(math_utils::getNormalizedPhiCoordinate(phi - dPhi)),
              index_table_utils::getPhiBinIndex(math_utils::getNormalizedPhiCoordinate(phi + dPhi))};
}

GPUhdi() const int4 VertexerTraitsEC0::getBinsRect(const Cluster& currentCluster, const int layerIndex,
                                                const float directionZIntersection, float maxdeltaz, float maxdeltaphi)
{
  const float zRangeMin = directionZIntersection - 2 * maxdeltaz;
  const float phiRangeMin = currentCluster.phiCoordinate - maxdeltaphi;
  const float zRangeMax = directionZIntersection + 2 * maxdeltaz;
  const float phiRangeMax = currentCluster.phiCoordinate + maxdeltaphi;

  if (zRangeMax < -constants::ecl::LayersZCoordinate()[layerIndex + 1] ||
      zRangeMin > constants::ecl::LayersZCoordinate()[layerIndex + 1] || zRangeMin > zRangeMax) {

    return getEmptyBinsRect();
  }

  return int4{gpu::GPUCommonMath::Max(0, index_table_utils::getZBinIndex(layerIndex + 1, zRangeMin)),
              index_table_utils::getPhiBinIndex(math_utils::getNormalizedPhiCoordinate(phiRangeMin)),
              gpu::GPUCommonMath::Min(constants::index_table::ZBins - 1, index_table_utils::getZBinIndex(layerIndex + 1, zRangeMax)),
              index_table_utils::getPhiBinIndex(math_utils::getNormalizedPhiCoordinate(phiRangeMax))};
}

// debug
inline void VertexerTraitsEC0::setDebugFlag(VertexerDebug flag, const unsigned char on = true)
{
  if (on) {
    mDBGFlags |= static_cast<unsigned int>(flag);
  } else {
    mDBGFlags &= ~static_cast<unsigned int>(flag);
  }
}

inline unsigned char VertexerTraitsEC0::isDebugFlag(const VertexerDebug& flags) const
{
  return mDBGFlags & static_cast<unsigned int>(flags);
}

extern "C" VertexerTraitsEC0* createVertexerTraitsEC0();

} // namespace ecl
} // namespace o2
#endif /* O2_EC0_TRACKING_VERTEXER_TRAITS_H_ */
