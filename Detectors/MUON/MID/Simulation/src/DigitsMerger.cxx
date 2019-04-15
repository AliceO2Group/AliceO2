// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/Simulation/src/DigitsMerger.cxx
/// \brief  Implementation of the digits merger for MID
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   05 March 2018
#include "MIDSimulation/DigitsMerger.h"

namespace o2
{
namespace mid
{
void DigitsMerger::process(const std::vector<ColumnDataMC>& inDigitStore, const o2::dataformats::MCTruthContainer<MCLabel>& inMCContainer, std::vector<ColumnData>& outDigitStore, o2::dataformats::MCTruthContainer<MCLabel>& outMCContainer)
{
  /// Merges the MC digits that are provided per hit
  /// into the format that we expect from data
  /// \param inDigitStore Vector of input MC digits
  /// \param inMCContainer Container with MC labels for input MC digits
  /// \param outDigitStore Vector with merged digits
  /// \param outMCContainer Container with MC labels for merged digits
  outDigitStore.clear();
  outMCContainer.clear();
  mDigitsLabels.clear();

  for (auto inIt = inDigitStore.begin(); inIt != inDigitStore.end(); ++inIt) {
    bool isNew = true;
    size_t idx = inIt - inDigitStore.begin();
    for (auto& pair : mDigitsLabels) {
      auto& outCol = pair.first;
      if (outCol.deId == inIt->deId && outCol.columnId == inIt->columnId) {
        outCol |= (*inIt);
        pair.second.emplace_back(idx);
        isNew = false;
        break;
      }
    }
    if (isNew) {
      std::vector<size_t> vec = { idx };
      mDigitsLabels.emplace_back(std::make_pair(*inIt, vec));
    }
  }

  for (auto pair : mDigitsLabels) {
    outDigitStore.emplace_back(pair.first);
    for (auto labelIdx : pair.second) {
      outMCContainer.addElements(outDigitStore.size() - 1, inMCContainer.getLabels(labelIdx));
    }
  }
}
} // namespace mid
} // namespace o2
