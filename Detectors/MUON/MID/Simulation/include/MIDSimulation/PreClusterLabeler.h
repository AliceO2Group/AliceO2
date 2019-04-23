// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MIDSimulation/PreClusterLabeler.h
/// \brief  PreClusterLabeler for MID
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   18 April 2019
#ifndef O2_MID_PRECLUSTERLABELER_H
#define O2_MID_PRECLUSTERLABELER_H

#include <gsl/gsl>
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "MIDClustering/PreCluster.h"
#include "MIDSimulation/MCLabel.h"

namespace o2
{
namespace mid
{
class PreClusterLabeler
{
 public:
  void process(gsl::span<const PreCluster> preClusters, const o2::dataformats::MCTruthContainer<MCLabel>& inMCContainer);

  const o2::dataformats::MCTruthContainer<MCCompLabel>& getContainer() { return mMCContainer; }

 private:
  bool isDuplicated(size_t idx, const MCLabel& label) const;
  bool addLabel(size_t idx, const MCLabel& label);

  o2::dataformats::MCTruthContainer<MCCompLabel> mMCContainer; ///< Labels
};
} // namespace mid
} // namespace o2

#endif /* O2_MID_PRECLUSTERLABELER_H */
