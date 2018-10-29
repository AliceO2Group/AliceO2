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

#include "ITStracking/Constants.h"
#include "ITStracking/Definitions.h"
#include "ITStracking/ClusterLines.h"
#include "ITStracking/ROframe.h"
#include "ReconstructionDataFormats/Vertex.h"

namespace o2
{
namespace ITS
{
class Cluster;
class Line;
using Vertex = o2::dataformats::Vertex<o2::dataformats::TimeStamp<int>>;

class Vertexer
{
 public:
  explicit Vertexer(const ROframe& event);
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
  inline std::vector<Line> const getTracklets() const { return mTracklets; }
  inline std::array<std::vector<Cluster>, Constants::ITS::LayersNumberVertexer> getClusters() const { return mClusters; }
  static const std::vector<std::pair<int, int>> selectClusters(
    const std::array<int, Constants::IndexTable::ZBins * Constants::IndexTable::PhiBins + 1>& indexTable,
    const std::array<int, 4>& selectedBinsRect);
  std::array<std::vector<Cluster>, Constants::ITS::LayersNumberVertexer> mClusters;
  std::vector<Vertex> getVertices() const { return mVertices; }

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
  ROframe mEvent;
  std::vector<Vertex> mVertices;
  std::array<std::array<int, Constants::IndexTable::ZBins * Constants::IndexTable::PhiBins + 1>,
             Constants::ITS::LayersNumberVertexer>
    mIndexTables;
  std::uint32_t mROFrame = 0;
  std::vector<Line> mTracklets;
  std::vector<ClusterLines> mTrackletClusters;
};

} // namespace ITS
} // namespace o2

#endif /* O2_ITSMFT_RECONSTRUCTION_CA_VERTEXER_H_ */