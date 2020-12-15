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
/// \fileStandaloneDebugger.h
/// \brief separate TreeStreamerRedirector class to be used with GPU
/// \author matteo.concas@cern.ch

#ifndef O2_ITS_STANDALONE_DEBUGGER_H_
#define O2_ITS_STANDALONE_DEBUGGER_H_

#include <string>
#include <iterator>

// Tracker
#include "DataFormatsITS/TrackITS.h"
#include "ITStracking/ROframe.h"

namespace o2
{

// class MCCompLabel;

namespace utils
{
class TreeStreamRedirector;
}

// namespace its
// {
// class TrackITSExt;
// }

namespace its
{
class Tracklet;
class Line;
class ROframe;
class ClusterLines;

using constants::its::UnusedIndex;

struct FakeTrackInfo {
 public:
  FakeTrackInfo();
  FakeTrackInfo(const ROframe& event, TrackITSExt& track) : isFake{false}, isAmbiguousId{false}, mainLabel{UnusedIndex, UnusedIndex, UnusedIndex, false}
  {
    occurrences.clear();
    for (size_t iCluster{0}; iCluster < 7; ++iCluster) {
      int extIndex = track.getClusterIndex(iCluster);
      o2::MCCompLabel mcLabel = event.getClusterLabels(iCluster, extIndex);
      bool found = false;

      for (size_t iOcc{0}; iOcc < occurrences.size(); ++iOcc) {
        std::pair<o2::MCCompLabel, int>& occurrence = occurrences[iOcc];
        if (mcLabel == occurrence.first) {
          ++occurrence.second;
          found = true;
        }
      }
      if (!found) {
        occurrences.emplace_back(mcLabel, 1);
      }
    }
    if (occurrences.size() > 1) {
      isFake = true;
    }
    std::sort(std::begin(occurrences), std::end(occurrences), [](auto e1, auto e2) {
      return e1.second > e2.second;
    });
    mainLabel = occurrences[0].first;

    for (auto iOcc{1}; iOcc < occurrences.size(); ++iOcc) {
      if (occurrences[iOcc].second == occurrences[0].second) {
        isAmbiguousId = true;
        break;
      }
    }
    for (auto iCluster{0}; iCluster < TrackITSExt::MaxClusters; ++iCluster) {
      int extIndex = track.getClusterIndex(iCluster);
      o2::MCCompLabel lbl = event.getClusterLabels(iCluster, extIndex);
      if (lbl == mainLabel && occurrences[0].second > 1 && !lbl.isNoise()) { // if 7 fake clusters -> occurrences[0].second = 1
        clusStatuses[iCluster] = 1;
      } else {
        clusStatuses[iCluster] = 0;
        ++nFakeClusters;
      }
    }
  }
  std::vector<std::pair<MCCompLabel, int>> occurrences;
  MCCompLabel mainLabel;
  std::array<int, 7> clusStatuses = {-1, -1, -1, -1, -1, -1, -1};
  bool isFake;
  bool isAmbiguousId;
  int nFakeClusters = 0;
};

class StandaloneDebugger
{
 public:
  explicit StandaloneDebugger(const std::string debugTreeFileName = "dbg_ITS.root");
  ~StandaloneDebugger();
  void setDebugTreeFileName(std::string);

  // Monte carlo oracle
  int getEventId(const int firstClusterId, const int secondClusterId, ROframe* frame);

  // Tree part
  const std::string& getDebugTreeFileName() const { return mDebugTreeFileName; }
  void fillCombinatoricsTree(std::array<std::vector<Cluster>, constants::its::LayersNumberVertexer>&,
                             std::vector<Tracklet>,
                             std::vector<Tracklet>,
                             const ROframe*);
  void fillCombinatoricsMCTree(std::vector<Tracklet>, std::vector<Tracklet>);
  void fillTrackletSelectionTree(std::array<std::vector<Cluster>, constants::its::LayersNumberVertexer>&,
                                 std::vector<Tracklet> comb01,
                                 std::vector<Tracklet> comb12,
                                 std::vector<std::array<int, 2>>,
                                 const ROframe*);
  void fillLinesSummaryTree(std::vector<Line>, const ROframe*);
  void fillPairsInfoTree(std::vector<Line>, const ROframe*);
  void fillLineClustersTree(std::vector<ClusterLines> clusters, const ROframe* event);
  void fillXYZHistogramTree(std::array<std::vector<int>, 3>, const std::array<int, 3>);
  void fillVerticesInfoTree(float x, float y, float z, int size, int rId, int eId, float pur);

  // Tracker debug utilities
  void dumpTrackToBranchWithInfo(const ROframe event, o2::its::TrackITSExt track, std::string branchName);

  static int getBinIndex(const float, const int, const float, const float);

 private:
  std::string mDebugTreeFileName = "dbg_ITS.root"; // output filename
  o2::utils::TreeStreamRedirector* mTreeStream;    // observer
};

inline void StandaloneDebugger::setDebugTreeFileName(const std::string name)
{
  if (!name.empty()) {
    mDebugTreeFileName = name;
  }
}

} // namespace its
} // namespace o2

#endif /*O2_ITS_STANDALONE_DEBUGGER_H_*/