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

/// \file   MIDSimulation/DigitsMerger.h
/// \brief  Digits merger for MID
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   05 March 2018
#ifndef O2_MID_DIGITSMERGER_H
#define O2_MID_DIGITSMERGER_H

#include <vector>
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DataFormatsMID/ROFRecord.h"
#include "DataFormatsMID/ColumnData.h"
#include "DataFormatsMID/MCLabel.h"

namespace o2
{
namespace mid
{
class DigitsMerger
{
 public:
  void process(const std::vector<ColumnData>& inDigitStore, const o2::dataformats::MCTruthContainer<MCLabel>& inMCContainer, const std::vector<ROFRecord>& inROFRecords, bool mergeInBunchPileup = true);

  /// Gets the merged column data
  const std::vector<ColumnData>& getColumnData() const { return mDigitStore; }
  /// Gets the merged MC labels
  const o2::dataformats::MCTruthContainer<MCLabel>& getMCContainer() const { return mMCContainer; }
  /// Gets the merged RO frame records
  const std::vector<ROFRecord>& getROFRecords() const { return mROFRecords; }

 private:
  void mergeDigit(size_t idigit, const std::vector<ColumnData>& inDigitStore);
  std::vector<std::pair<ColumnData, std::vector<size_t>>> mDigitsLabels{}; //! Temporary digits store
  std::vector<ColumnData> mDigitStore{};                                   ///< Digit store
  o2::dataformats::MCTruthContainer<MCLabel> mMCContainer{};               ///< MC Container
  std::vector<ROFRecord> mROFRecords{};                                    ///< RO frame records
};

} // namespace mid
} // namespace o2

#endif /* O2_MID_DIGITSMERGER_H */
