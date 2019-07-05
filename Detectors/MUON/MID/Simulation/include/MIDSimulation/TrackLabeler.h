// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MIDSimulation/TrackLabeler.h
/// \brief  Tracks labeler for MID
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   07 June 2019
#ifndef O2_MID_TRACKLABELER_H
#define O2_MID_TRACKLABELER_H

#include <vector>
#include <gsl/gsl>
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "DataFormatsMID/Track.h"
#include "DataFormatsMID/Cluster3D.h"
#include "MIDSimulation/MCClusterLabel.h"

namespace o2
{
namespace mid
{
class TrackLabeler
{
 public:
  void process(gsl::span<const Cluster3D>& clusters, gsl::span<const Track>& tracks, const o2::dataformats::MCTruthContainer<MCClusterLabel>& inMCContainer);

  /// Returns the tracks labels
  const o2::dataformats::MCTruthContainer<MCCompLabel>& getTracksLabels() { return mMCTracksLabels; }

  /// Returns the cluster labels
  const o2::dataformats::MCTruthContainer<MCClusterLabel>& getTrackClustersLabels() { return mMCTrackClustersLabels; }

 private:
  bool areBothSidesFired(const gsl::span<const MCClusterLabel>& labels) const;
  std::vector<MCCompLabel> findLabels(const Track& track, const o2::dataformats::MCTruthContainer<MCClusterLabel>& inMCContainer) const;

  o2::dataformats::MCTruthContainer<MCCompLabel> mMCTracksLabels;           ///< Track labels
  o2::dataformats::MCTruthContainer<MCClusterLabel> mMCTrackClustersLabels; ///< Cluster labels
};
} // namespace mid
} // namespace o2

#endif /* O2_MID_TRACKLABELER_H */
