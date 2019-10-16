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

void PreClusterLabeler::process(gsl::span<const PreCluster> preClusters, const o2::dataformats::MCTruthContainer<MCLabel>& inMCContainer, gsl::span<const ROFRecord> rofRecordsPC, gsl::span<const ROFRecord> rofRecordsData)
{
  /// Applies labels to the pre-clusters
  mMCContainer.clear();
  auto dataROFIt = rofRecordsData.begin();
  for (auto pcROFIt = rofRecordsPC.begin(); pcROFIt != rofRecordsPC.end(); ++pcROFIt) {
    for (size_t ipc = pcROFIt->firstEntry; ipc < pcROFIt->firstEntry + pcROFIt->nEntries; ++ipc) {
      for (size_t idata = dataROFIt->firstEntry; idata < dataROFIt->firstEntry + dataROFIt->nEntries; ++idata) {
        auto labels = inMCContainer.getLabels(idata);
        for (auto& label : labels) {
          if (label.getCathode() != preClusters[ipc].cathode) {
            continue;
          }
          if (label.getDEId() != preClusters[ipc].deId) {
            continue;
          }
          int columnId = label.getColumnId();
          if (columnId < preClusters[ipc].firstColumn || columnId > preClusters[ipc].lastColumn) {
            continue;
          }
          int firstStrip = MCLabel::getStrip(preClusters[ipc].firstStrip, preClusters[ipc].firstLine);
          int lastStrip = MCLabel::getStrip(preClusters[ipc].lastStrip, preClusters[ipc].lastLine);
          if ((columnId == preClusters[ipc].firstColumn && label.getLastStrip() < firstStrip) || (columnId == preClusters[ipc].lastColumn && label.getFirstStrip() > lastStrip)) {
            continue;
          }

          addLabel(ipc, label);
        }
      }
    }
    ++dataROFIt;
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
  MCCompLabel lb(label.getTrackID(), label.getEventID(), label.getSourceID(), label.isFake());
  mMCContainer.addElement(idx, lb);
  return true;
}

} // namespace mid
} // namespace o2
