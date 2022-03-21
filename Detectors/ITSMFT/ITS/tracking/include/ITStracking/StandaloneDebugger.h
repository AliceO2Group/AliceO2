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
/// \fileStandaloneDebugger.h
/// \brief separate TreeStreamerRedirector class to be used with GPU
/// \author matteo.concas@cern.ch

#ifndef O2_ITS_STANDALONE_DEBUGGER_H_
#define O2_ITS_STANDALONE_DEBUGGER_H_

#include <string>
#include <iterator>

// Tracker
#if !defined(__CUDACC__) && !defined(__HIPCC__)
#include "ITStracking/ROframe.h"
#endif

#include "DataFormatsITS/TrackITS.h"
#include "ITStracking/PrimaryVertexContext.h"

namespace o2
{

namespace utils
{
class TreeStreamRedirector;
}

namespace its
{
class Tracklet;
class Line;
class ROframe;
class ClusterLines;

using constants::its::UnusedIndex;

// #if !defined(__CUDACC__) && !defined(__HIPCC__)
// template <int numClusters = TrackITSExt::MaxClusters>
// struct FakeTrackInfo {
//  public:
//   FakeTrackInfo();
//   FakeTrackInfo(PrimaryVertexContext* pvc, const ROframe& event, TrackITSExt& track, bool storeClusters) : isFake{false}, isAmbiguousId{false}, mainLabel{UnusedIndex, UnusedIndex, UnusedIndex, false}
//   {
//     occurrences.clear();
//     for (auto& c : clusStatuses) {
//       c = -1;
//     }
//     for (size_t iCluster{0}; iCluster < numClusters; ++iCluster) {
//       int extIndex = track.getClusterIndex(iCluster);
//       if (extIndex == -1) {
//         continue;
//       }
//       o2::MCCompLabel mcLabel = *(event.getClusterLabels(iCluster, extIndex).begin());
//       bool found = false;

//       for (size_t iOcc{0}; iOcc < occurrences.size(); ++iOcc) {
//         std::pair<o2::MCCompLabel, int>& occurrence = occurrences[iOcc];
//         if (mcLabel == occurrence.first) {
//           ++occurrence.second;
//           found = true;
//           break;
//         }
//       }
//       if (!found) {
//         occurrences.emplace_back(mcLabel, 1);
//       }
//     }

//     if (occurrences.size() > 1) {
//       isFake = true;
//     }
//     std::sort(std::begin(occurrences), std::end(occurrences), [](auto e1, auto e2) {
//       return e1.second > e2.second;
//     });
//     mainLabel = occurrences[0].first;

//     for (size_t iOcc{1}; iOcc < occurrences.size(); ++iOcc) {
//       if (occurrences[iOcc].second == occurrences[0].second) {
//         isAmbiguousId = true;
//         break;
//       }
//     }

//     for (size_t iCluster{0}; iCluster < numClusters; ++iCluster) {
//       int extIndex = track.getClusterIndex(iCluster);
//       if (extIndex == -1) {
//         continue;
//       }
//       o2::MCCompLabel lbl = *(event.getClusterLabels(iCluster, extIndex).begin());
//       if (lbl == mainLabel && occurrences[0].second > 1 && !lbl.isNoise()) { // if we have MaxClusters fake clusters -> occurrences[0].second = 1
//         clusStatuses[iCluster] = 1;
//       } else {
//         clusStatuses[iCluster] = 0;
//         ++nFakeClusters;
//       }
//     }
//     if (storeClusters) {
//       for (auto iCluster{0}; iCluster < numClusters; ++iCluster) {
//         const int index = track.getClusterIndex(iCluster);
//         if (index != constants::its::UnusedIndex) {
//           clusters[iCluster] = pvc->getClusters()[iCluster][index];
//           trackingFrameInfos[iCluster] = event.getTrackingFrameInfoOnLayer(iCluster).at(index);
//         }
//       }
//     }
//   }

//   // Data
//   std::vector<std::pair<MCCompLabel, int>>
//     occurrences;
//   MCCompLabel mainLabel;
//   std::array<int, numClusters> clusStatuses;
//   std::array<o2::its::Cluster, numClusters> clusters;
//   std::array<o2::its::TrackingFrameInfo, numClusters> trackingFrameInfos;

//   bool isFake;
//   bool isAmbiguousId;
//   int nFakeClusters = 0;
//   ClassDefNV(FakeTrackInfo, 1);
// }; // namespace its
// #endif

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
  void dumpTrackToBranchWithInfo(std::string branchName, o2::its::TrackITSExt track, const ROframe event, PrimaryVertexContext* pvc, const bool dumpClusters = false);

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