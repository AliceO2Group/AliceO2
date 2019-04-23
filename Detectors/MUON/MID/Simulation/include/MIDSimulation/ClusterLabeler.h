// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MIDSimulation/ClusterLabeler.h
/// \brief  ClusterLabeler for MID
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   18 April 2019
#ifndef O2_MID_CLUSTERLABELER_H
#define O2_MID_CLUSTERLABELER_H

#include <array>
#include <gsl/gsl>
#include "DataFormatsMID/Cluster2D.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "MIDClustering/PreCluster.h"
#include "MIDSimulation/MCClusterLabel.h"

namespace o2
{
namespace mid
{
class ClusterLabeler
{
 public:
  void process(gsl::span<const PreCluster> preClusters, const o2::dataformats::MCTruthContainer<MCCompLabel>& inMCContainer, gsl::span<const Cluster2D> clusters, gsl::span<const std::array<size_t, 2>> correlations);

  const o2::dataformats::MCTruthContainer<MCClusterLabel>& getContainer() { return mMCContainer; }

 private:
  MCClusterLabel* findLabel(size_t idx, const MCCompLabel& pcLabel);
  void addLabels(size_t idx, int cathode, gsl::span<const MCCompLabel>& labels);

  o2::dataformats::MCTruthContainer<MCClusterLabel> mMCContainer; ///< Clusters labels
};
} // namespace mid
} // namespace o2

#endif /* O2_MID_CLUSTERLABELER_H */
