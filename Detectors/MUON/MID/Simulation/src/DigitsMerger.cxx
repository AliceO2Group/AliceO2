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

/// \file   MID/Simulation/src/DigitsMerger.cxx
/// \brief  Implementation of the digits merger for MID
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   05 March 2018
#include "MIDSimulation/DigitsMerger.h"

namespace o2
{
namespace mid
{

void DigitsMerger::process(const std::vector<ColumnData>& inDigitStore, const o2::dataformats::MCTruthContainer<MCLabel>& inMCContainer, const std::vector<ROFRecord>& inROFRecords, bool mergeInBunchPileup)
{
  process(inDigitStore, inROFRecords, &inMCContainer, mergeInBunchPileup);
}

void DigitsMerger::process(gsl::span<const ColumnData> inDigitStore, gsl::span<const ROFRecord> inROFRecords, const o2::dataformats::MCTruthContainer<MCLabel>* inMCContainer, bool mergeInBunchPileup)
{
  mDigitStore.clear();
  mMCContainer.clear();
  mROFRecords.clear();

  for (auto rofIt = inROFRecords.begin(); rofIt != inROFRecords.end(); ++rofIt) {
    auto nextRofIt = rofIt + 1;
    bool mergeInteractions = mergeInBunchPileup && nextRofIt != inROFRecords.end() && rofIt->interactionRecord == nextRofIt->interactionRecord;

    for (size_t idigit = rofIt->firstEntry, end = rofIt->getEndIndex(); idigit < end; ++idigit) {
      mHandler.merge(inDigitStore[idigit], idigit);
    }

    if (mergeInteractions) {
      continue;
    }

    auto firstEntry = mDigitStore.size();
    auto digits = mHandler.getMerged();
    mROFRecords.emplace_back(rofIt->interactionRecord, rofIt->eventType, firstEntry, digits.size());
    mDigitStore.insert(mDigitStore.end(), digits.begin(), digits.end());

    if (inMCContainer) {
      for (auto& dig : digits) {
        auto indexes = mHandler.getMergedIndexes(dig);
        for (auto labelIdx : indexes) {
          mMCContainer.addElements(mDigitStore.size() - 1, inMCContainer->getLabels(labelIdx));
        }
      }
    }
    mHandler.clear();
  }
}
} // namespace mid
} // namespace o2
