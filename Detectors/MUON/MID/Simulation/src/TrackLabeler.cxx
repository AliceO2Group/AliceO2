// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/Simulation/src/TrackLabeler.cxx
/// \brief  Implementation of the TrackLabeler for MID
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   01 March 2018
#include "MIDSimulation/TrackLabeler.h"

namespace o2
{
namespace mid
{

bool TrackLabeler::areBothSidesFired(const gsl::span<const MCClusterLabel>& labels) const
{
  /// Check if the BP and NBP were fired by some track
  bool isFiredBP = false, isFiredNBP = false;
  for (auto& label : labels) {
    if (label.isFiredBP()) {
      isFiredBP = true;
    }
    if (label.isFiredNBP()) {
      isFiredNBP = true;
    }
  }
  return isFiredBP && isFiredNBP;
}

std::vector<MCCompLabel> TrackLabeler::findLabels(const Track& track, const o2::dataformats::MCTruthContainer<MCClusterLabel>& inMCContainer) const
{
  /// Finds the track label from its associated clusters labels
  std::vector<MCCompLabel> allLabels;
  std::vector<int> counts;
  for (int ich = 0; ich < 4; ++ich) {
    auto icl = track.getClusterMatched(ich);
    if (icl < 0) {
      continue;
    }

    // First check if the BP and NBP were ever fired by a track
    // If they are, it means that the chamber was efficient
    bool bothFired = areBothSidesFired(inMCContainer.getLabels(icl));

    for (auto& label : inMCContainer.getLabels(icl)) {
      // This track fired only one cathode
      bool oneFired = (!label.isFiredBP() || !label.isFiredNBP());
      if (bothFired && oneFired) {
        // This condition means that the cluster was made from
        // a track in the BP and a different track in the NBP.
        // This happens when we have two tracks in the same column.
        // This means that the cluster is a fake one, so we discard it
        continue;
      }
      bool isNew = true;
      for (size_t idx = 0; idx < allLabels.size(); ++idx) {
        if (allLabels[idx] == label) {
          ++counts[idx];
          isNew = false;
          break;
        }
      }
      if (isNew) {
        allLabels.emplace_back(label);
        counts.emplace_back(1);
      }
    }
  }

  std::vector<MCCompLabel> labels;

  for (size_t idx = 0; idx < allLabels.size(); ++idx) {
    if (counts[idx] >= 3) {
      labels.emplace_back(allLabels[idx]);
    }
  }

  return std::move(labels);
}

void TrackLabeler::process(gsl::span<const Cluster3D>& clusters, gsl::span<const Track>& tracks, const o2::dataformats::MCTruthContainer<MCClusterLabel>& inMCContainer)
{
  /// Applies labels to the tracks
  mMCTracksLabels.clear();

  for (auto& track : tracks) {
    auto idx = &track - &tracks[0];
    auto labels = findLabels(track, inMCContainer);
    if (labels.empty()) {
      labels.emplace_back(MCCompLabel());
    }
    for (auto& label : labels) {
      mMCTracksLabels.addElement(idx, label);
    }
  }

  // For the moment we store all clusters
  // This can change if we decide to store only associated clusters
  mMCTrackClustersLabels.clear();
  mMCTrackClustersLabels = inMCContainer;
}
} // namespace mid
} // namespace o2
