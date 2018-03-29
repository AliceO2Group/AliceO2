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

#include "ITSReconstruction/CA/Constants.h"
#include "ITSReconstruction/CA/Definitions.h"
#include "ITSReconstruction/CA/vertexer/ClusterLines.h"

namespace o2
{
namespace ITS
{
namespace CA
{
class Cluster;
class Event;
class Line;

class Vertexer
{
 public:
  explicit Vertexer(const Event& event);
  virtual ~Vertexer();
  Vertexer(const Vertexer&) = delete;
  Vertexer& operator=(const Vertexer&) = delete;

  void initialise(const float zCut, const float phiCut, const float pairCut, const float clusterCut,
                  const int clusterContributorsCut);
  // void computeTriplets();
  // void checkTriplets();
  void findTracklets();
  void findVertices();
  void generateTracklets();
  // inline std::vector<std::array<int, 3>> getTriplets() { return mTriplets; }
  inline std::vector<Line> getTracklets() { return mTracklets; }
  inline std::vector<std::array<float, 3>> getVertices() { return mVertices; }
  inline std::array<std::vector<Cluster>, Constants::ITS::LayersNumberVertexer> getClusters() { return mClusters; }
  inline std::vector<float> getZDelta() { return mZDelta; }
  inline std::vector<float> getPhiDelta() { return mPhiDelta; }
  void printIndexTables();
  void printVertices();

  static const std::vector<std::pair<int, int>> selectClusters(
    const std::array<int, Constants::IndexTable::ZBins * Constants::IndexTable::PhiBins + 1>& indexTable,
    const std::array<int, 4>& selectedBinsRect);

 protected:
  bool mVertexerInitialised{ false };
  bool mTrackletsFound{ false };
  float mDeltaRadii10, mDeltaRadii21;
  float mZCut, mPhiCut, mPairCut, mClusterCut;
  int mClusterContributorsCut;
  int mPhiSpan, mZSpan;
  std::array<float, 3> mAverageClustersRadii;
  std::array<float, Constants::ITS::LayersNumber> mITSRadii;
  float mZBinSize;
  // std::vector<std::pair<int, int>> mClustersToProcessInner;
  // std::vector<std::pair<int, int>> mClustersToProcessOuter;
  Event mEvent;
  std::vector<std::array<float, 3>> mVertices;
  std::array<std::vector<Cluster>, Constants::ITS::LayersNumberVertexer> mClusters;
  std::array<std::array<int, Constants::IndexTable::ZBins * Constants::IndexTable::PhiBins + 1>,
             Constants::ITS::LayersNumberVertexer>
    mIndexTables;
  
  std::vector<bool> mUsedTracklets;
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