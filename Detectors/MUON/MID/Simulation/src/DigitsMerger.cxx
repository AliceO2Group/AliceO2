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
void DigitsMerger::mergeDigit(size_t idigit, const std::vector<ColumnDataMC>& inDigitStore)
{
  /// Merges the current digit
  for (auto& pair : mDigitsLabels) {
    auto& outCol = pair.first;
    if (outCol.deId == inDigitStore[idigit].deId && outCol.columnId == inDigitStore[idigit].columnId) {
      outCol |= (inDigitStore[idigit]);
      pair.second.emplace_back(idigit);
      return;
    }
  }
  std::vector<size_t> vec = {idigit};
  mDigitsLabels.emplace_back(std::make_pair(inDigitStore[idigit], vec));
}

void DigitsMerger::process(const std::vector<ColumnDataMC>& inDigitStore, const o2::dataformats::MCTruthContainer<MCLabel>& inMCContainer, const std::vector<ROFRecord>& inROFRecords)
{
  /// Merges the MC digits that are provided per hit
  /// into the format that we expect from data
  /// \param inDigitStore Vector of input MC digits
  /// \param inMCContainer Container with MC labels for input MC digits
  /// \param inROFRecords Vector with RO frame records
  mDigitStore.clear();
  mMCContainer.clear();
  mROFRecords.clear();

  for (auto& rofRecord : inROFRecords) {
    mDigitsLabels.clear();
    for (size_t idigit = rofRecord.firstEntry; idigit < rofRecord.firstEntry + rofRecord.nEntries; ++idigit) {
      mergeDigit(idigit, inDigitStore);
    }

    auto firstEntry = mDigitStore.size();
    mROFRecords.emplace_back(rofRecord.interactionRecord, rofRecord.eventType, firstEntry, mDigitsLabels.size());

    for (auto pair : mDigitsLabels) {
      mDigitStore.emplace_back(pair.first);
      for (auto labelIdx : pair.second) {
        mMCContainer.addElements(mDigitStore.size() - 1, inMCContainer.getLabels(labelIdx));
      }
    }
  }
}
} // namespace mid
} // namespace o2
