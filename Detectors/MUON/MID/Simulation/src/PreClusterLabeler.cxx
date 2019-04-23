// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/Simulation/src/PreClusterLabeler.cxx
/// \brief  Implementation of the PreClusterLabeler for MID
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   01 March 2018
#include "MIDSimulation/PreClusterLabeler.h"

namespace o2
{
namespace mid
{

void PreClusterLabeler::process(gsl::span<const PreCluster> preClusters, const o2::dataformats::MCTruthContainer<MCLabel>& inMCContainer)
{
  /// Applies labels to the pre-clusters
  mMCContainer.clear();

  for (auto& pc : preClusters) {
    auto idx = &pc - &preClusters[0];
    for (size_t iel = 0; iel < inMCContainer.getNElements(); ++iel) {
      auto label = inMCContainer.getElement(iel);
      if (label.getCathode() != pc.cathode) {
        continue;
      }
      if (label.getDEId() != pc.deId) {
        continue;
      }
      int columnId = label.getColumnId();
      if (columnId < pc.firstColumn || columnId > pc.lastColumn) {
        continue;
      }
      int firstStrip = MCLabel::getStrip(pc.firstStrip, pc.firstLine);
      int lastStrip = MCLabel::getStrip(pc.lastStrip, pc.lastLine);
      if ((columnId == pc.firstColumn && label.getLastStrip() < firstStrip) || (columnId == pc.lastColumn && label.getFirstStrip() > lastStrip)) {
        continue;
      }

      addLabel(idx, label);
    }
  }
}

bool PreClusterLabeler::isDuplicated(size_t idx, const MCLabel& label) const
{
  /// Checks if the label is already there
  if (idx >= mMCContainer.getIndexedSize()) {
    return false;
  }

  for (auto& lb : mMCContainer.getLabels(idx)) {
    if (label.compare(lb) == 1) {
      return true;
    }
  }

  return false;
}

bool PreClusterLabeler::addLabel(size_t idx, const MCLabel& label)
{
  /// Converts MCLabel into MCCompLabel
  if (isDuplicated(idx, label)) {
    return false;
  }
  MCCompLabel lb(label.getTrackID(), label.getEventID(), label.getSourceID());
  mMCContainer.addElement(idx, lb);
  return true;
}

} // namespace mid
} // namespace o2
