// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/Simulation/src/ClusterLabeler.cxx
/// \brief  Implementation of the ClusterLabeler for MID
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   01 March 2018
#include "MIDSimulation/ClusterLabeler.h"

#include <map>

namespace o2
{
namespace mid
{

void ClusterLabeler::process(gsl::span<const PreCluster> preClusters, const o2::dataformats::MCTruthContainer<MCCompLabel>& inMCContainer, gsl::span<const Cluster2D> clusters, gsl::span<const std::array<size_t, 2>> correlations)
{
  /// Applies labels to the clusters
  mMCContainer.clear();

  std::map<size_t, std::vector<size_t>> reordered;
  for (auto& corr : correlations) {
    reordered[corr[0]].emplace_back(corr[1]);
  }

  for (auto& corr : reordered) {
    for (auto& pcIdx : corr.second) {
      int cathode = static_cast<int>(preClusters[pcIdx].cathode);
      auto labels = inMCContainer.getLabels(pcIdx);
      addLabels(corr.first, cathode, labels);
    }
  }
}

MCClusterLabel* ClusterLabeler::findLabel(size_t idx, const MCCompLabel& pcLabel)
{
  /// Checks if the label is already there
  if (idx >= mMCContainer.getIndexedSize()) {
    return nullptr;
  }

  for (auto& cLabel : mMCContainer.getLabels(idx)) {
    if (pcLabel.compare(cLabel) == 1) {
      return &cLabel;
    }
  }

  return nullptr;
}

void ClusterLabeler::addLabels(size_t idx, int cathode, gsl::span<const MCCompLabel>& labels)
{
  /// Adds the labels
  for (auto& pcLabel : labels) {
    MCClusterLabel* foundLabel = findLabel(idx, pcLabel);
    if (foundLabel) {
      if (cathode == 0) {
        foundLabel->setIsFiredBP(true);
      } else {
        foundLabel->setIsFiredNBP(true);
      }
    } else {
      bool isFiredBP = (cathode == 0);
      bool isFiredNBP = (cathode == 1);
      MCClusterLabel cLabel(pcLabel.getTrackID(), pcLabel.getEventID(), pcLabel.getSourceID(), isFiredBP, isFiredNBP);
      mMCContainer.addElement(idx, cLabel);
    }
  }
}

} // namespace mid
} // namespace o2
