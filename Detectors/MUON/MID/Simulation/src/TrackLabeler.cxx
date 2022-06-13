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

/// \file   MID/Simulation/src/TrackLabeler.cxx
/// \brief  Implementation of the TrackLabeler for MID
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   01 March 2018
#include "MIDSimulation/TrackLabeler.h"

#include <unordered_map>

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

MCCompLabel TrackLabeler::makeTrackLabel(const Track& track, const o2::dataformats::MCTruthContainer<MCClusterLabel>& inMCContainer) const
{
  /// Finds the track label from its associated clusters labels
  std::unordered_map<MCCompLabel, int> allLabels;
  for (int ich = 0; ich < 4; ++ich) {
    auto icl = track.getClusterMatched(ich);
    if (icl < 0) {
      continue;
    }

    auto clLabels = inMCContainer.getLabels(icl);

    // First check if the BP and NBP were ever fired by a track
    // If they are, it means that the chamber was efficient
    bool bothFired = areBothSidesFired(clLabels);

    for (auto& label : clLabels) {
      // This track fired only one cathode
      bool oneFired = (!label.isFiredBP() || !label.isFiredNBP());
      if (bothFired && oneFired) {
        // This condition means that the cluster was made from
        // a track in the BP and a different track in the NBP.
        // This happens when we have two tracks in the same column.
        // This means that the cluster is a fake one, so we discard it
        continue;
      }
      ++allLabels[label];
    }
  }

  MCCompLabel outLabel;
  int nMatched = 0;
  for (auto& item : allLabels) {
    if (item.second > nMatched) {
      nMatched = item.second;
      outLabel = item.first;
    }
  }

  if (nMatched < 3) {
    outLabel.setFakeFlag();
  }

  return outLabel;
}

void TrackLabeler::process(gsl::span<const Cluster> clusters, gsl::span<const Track> tracks, const o2::dataformats::MCTruthContainer<MCClusterLabel>& inMCContainer)
{
  /// Applies labels to the tracks
  mMCTracksLabels.clear();

  for (auto& track : tracks) {
    mMCTracksLabels.emplace_back(makeTrackLabel(track, inMCContainer));
  }

  // For the moment we store all clusters
  // This can change if we decide to store only associated clusters
  mMCTrackClustersLabels.clear();
  mMCTrackClustersLabels.mergeAtBack(inMCContainer);
}
} // namespace mid
} // namespace o2
