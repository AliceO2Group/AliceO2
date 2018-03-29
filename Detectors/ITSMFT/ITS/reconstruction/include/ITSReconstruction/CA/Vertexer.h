// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_ITSMFT_RECONSTRUCTION_CA_VERTEXER_H_
#define O2_ITSMFT_RECONSTRUCTION_CA_VERTEXER_H_

#include <vector>
#include <array>
#include <tuple>

#include "ITSReconstruction/CA/Constants.h"
#include "ITSReconstruction/CA/Definitions.h"
#include "ITSReconstruction/CA/ClusterLines.h"
#include "ReconstructionDataFormats/Vertex.h"

namespace o2
{
namespace ITS
{
namespace CA
{
class Cluster;
class Event;
class Line;
using Vertex = o2::dataformats::Vertex<o2::dataformats::TimeStamp<int>>;

class Vertexer
{
 public:
  explicit Vertexer(const Event& event);
  virtual ~Vertexer();
  Vertexer(const Vertexer&) = delete;
  Vertexer& operator=(const Vertexer&) = delete;
  void initialise(const float zCut, const float phiCut, const float pairCut, const float clusterCut,
                  const int clusterContributorsCut);
  void initialise(const std::tuple<float, float, float, float, int> initParams);
  void findTracklets(const bool useMCLabel = false);
  void findVertices();
  void setROFrame(std::uint32_t f) { mROFrame = f; }
  std::uint32_t getROFrame() const { return mROFrame; }
  inline std::vector<Line> getTracklets() { return mTracklets; }
  inline std::array<std::vector<Cluster>, Constants::ITS::LayersNumberVertexer> getClusters() { return mClusters; }
  static const std::vector<std::pair<int, int>> selectClusters(
    const std::array<int, Constants::IndexTable::ZBins * Constants::IndexTable::PhiBins + 1>& indexTable,
    const std::array<int, 4>& selectedBinsRect);
  std::array<std::vector<Cluster>, Constants::ITS::LayersNumberVertexer> mClusters;

  // #ifdef DEBUG_BUILD
  //   void printIndexTables();
  //   void dumpTracklets();
  //   inline std::vector<std::tuple<std::array<float, 3>, int, float>> getLegacyVertices() { return mLegacyVertices; }
  // #else
  //   inline std::vector<std::array<float, 3>> getLegacyVertices() { return mLegacyVertices; }
  // #endif
  std::vector<Vertex>& getVertices() { return mVertices; }

 protected:
  bool mVertexerInitialised{ false };
  bool mTrackletsFound{ false };
  std::vector<bool> mUsedTracklets;
  float mDeltaRadii10, mDeltaRadii21;
  float mZCut, mPhiCut, mPairCut, mClusterCut, mMaxDirectorCosine3;
  int mClusterContributorsCut;
  int mPhiSpan, mZSpan;
  std::array<float, 3> mAverageClustersRadii;
  std::array<float, Constants::ITS::LayersNumber> mITSRadii;
  float mZBinSize;
  Event mEvent;
  // #ifdef DEBUG_BUILD
  //   std::vector<std::tuple<std::array<float, 3>, int, float>> mLegacyVertices;
  // #else
  //   std::vector<std::array<float, 3>> mLegacyVertices;
  // #endif
  std::vector<Vertex> mVertices;
  std::array<std::array<int, Constants::IndexTable::ZBins * Constants::IndexTable::PhiBins + 1>,
             Constants::ITS::LayersNumberVertexer>
    mIndexTables;
  std::uint32_t mROFrame = 0;
  std::vector<Line> mTracklets;
  std::vector<ClusterLines> mTrackletClusters;
  std::vector<std::vector<float>> mDCAMatrix;

  // Debug data structures
  // std::vector<std::array<int, 3>> mTriplets;
  std::vector<float> mZDelta;
  std::vector<float> mPhiDelta;
};

} // namespace CA
} // namespace ITS
} // namespace o2

#endif /* O2_ITSMFT_RECONSTRUCTION_CA_VERTEXER_H_ */