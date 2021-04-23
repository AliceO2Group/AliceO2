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

#if !defined(__CUDACC__) && !defined(__HIPCC__)

#include <gsl/gsl>

template <int numClusters = TrackITSExt::MaxClusters>
struct FakeTrackInfo {
 public:
  FakeTrackInfo() = default;
  FakeTrackInfo(PrimaryVertexContext* pvc, const ROframe& event, TrackITSExt& track, bool storeClusters) : isFake{false},
                                                                                                           isAmbiguousId{false},
                                                                                                           track{track},
                                                                                                           mainLabel{UnusedIndex, UnusedIndex, UnusedIndex, false}
  {
    occurrences.clear();
    mcLabels.clear();
    mcLabels.resize(track.getNumberOfClusters());
    clusters.clear();
    clusters.resize(track.getNumberOfClusters());
    trackingFrameInfos.clear();
    trackingFrameInfos.resize(track.getNumberOfClusters());
    for (auto& c : clusStatuses) {
      c = -1;
    }
    for (size_t iCluster{0}; iCluster < track.getNumberOfClusters(); ++iCluster) {
      int extIndex = track.getClusterIndex(iCluster);
      if (extIndex == -1) {
        continue;
      }
      // Get labels related to this cluster
      auto labels = event.getClusterLabels(iCluster, extIndex);
      for (auto& label : labels) {
        if (label.isSet()) { // Store only set labels
          mcLabels[iCluster].emplace_back(label);
        }
      }

      for (size_t iOcc{0}; iOcc < occurrences.size(); ++iOcc) {
        std::pair<o2::MCCompLabel, int>& occurrence = occurrences[iOcc];
        for (auto& label : labels) {
          if (label == occurrence.first && label.isSet()) {
            ++occurrence.second;
            found = true;
            break;
          }
        }
        if (found) {
          break;
        }
      }
      if (!found) {
        for (auto& label : labels) {
          if (label.isSet()) {
            occurrences.emplace_back(label, 1);
          }
        }
      }
    }

    if (occurrences.size() > 1) {
      isFake = true;
    }
    std::sort(std::begin(occurrences), std::end(occurrences), [](auto e1, auto e2) {
      return e1.second > e2.second;
    });
    mainLabel = occurrences[0].first;

    for (size_t iOcc{1}; iOcc < occurrences.size(); ++iOcc) {
      if (occurrences[iOcc].second == occurrences[0].second) {
        isAmbiguousId = true;
        break;
      }
    }

    // Check status of clusters related to main label
    for (size_t iCluster{0}; iCluster < track.getNumberOfClusters(); ++iCluster) {
      int extIndex = track.getClusterIndex(iCluster);
      if (extIndex == -1) {
        continue;
      }

      auto labels = event.getClusterLabels(iCluster, extIndex);
      bool fake{true};
      for (auto& label : labels) {
        if (label == mainLabel && occurrences[0].second > 1 && !label.isNoise()) { // if we have MaxClusters fake clusters -> occurrences[0].second = 1
          clusStatuses[iCluster] = 1;
          fake = false;
          break;
        }
      }
      if (fake) {
        clusStatuses[iCluster] = 0;
        ++nFakeClusters;
      }
    }

    // Store clusters at convenience
    if (storeClusters) {
      for (auto iCluster{0}; iCluster < track.getNumberOfClusters(); ++iCluster) {
        const int index = track.getClusterIndex(iCluster);
        if (index != constants::its::UnusedIndex) {
          clusters[iCluster] = pvc->getClusters()[iCluster][index];
          trackingFrameInfos[iCluster] = event.getTrackingFrameInfoOnLayer(iCluster).at(index);
        }
      }
    }
  }

  // Data
  std::vector<std::vector<MCCompLabel>> mcLabels; // Multiple labels per cluster
  std::vector<std::pair<MCCompLabel, int>> occurrences;
  MCCompLabel mainLabel;
  std::array<int, numClusters> clusStatuses;
  std::vector<o2::its::Cluster> clusters;
  std::vector<o2::its::TrackingFrameInfo> trackingFrameInfos;
  o2::its::TrackITSExt track;

  bool isFake;
  bool isAmbiguousId;
  int nFakeClusters = 0;
  ClassDefNV(FakeTrackInfo, 1);
};
#endif

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
  void dumpTrackToBranchWithInfo(std::string branchName, int layer, int iteration, o2::its::TrackITSExt track, const ROframe& event, PrimaryVertexContext* pvc, const bool dumpClusters = false);
  void dumpTmpTrackToBranchWithInfo(std::string branchName, int layer, int iteration, o2::its::TrackITSExt track, const ROframe& event, PrimaryVertexContext* pvc, float pChi2, const bool dumpClusters = false);
  void dumpTrkChi2(float chiFake, float chiTrue);

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