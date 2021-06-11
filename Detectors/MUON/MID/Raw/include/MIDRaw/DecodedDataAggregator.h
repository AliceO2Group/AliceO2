// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MIDRaw/DecodedDataAggregator.h
/// \brief  MID decoded raw data aggregator
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   26 February 2020
#ifndef O2_MID_DECODEDDATAAGGREGATOR_H
#define O2_MID_DECODEDDATAAGGREGATOR_H

#include <vector>
#include <map>
#include <gsl/gsl>
#include "DataFormatsMID/ColumnData.h"
#include "DataFormatsMID/ROFRecord.h"
#include "MIDRaw/CrateMapper.h"
#include "DataFormatsMID/ROBoard.h"

namespace o2
{
namespace mid
{
class DecodedDataAggregator
{
 public:
  void process(gsl::span<const ROBoard> localBoards, gsl::span<const ROFRecord> rofRecords);

  /// Gets the vector of data
  const std::vector<ColumnData>& getData() { return mData; }

  /// Gets the vector of data RO frame records
  const std::vector<ROFRecord>& getROFRecords() { return mROFRecords; }

 private:
  void addData(const ROBoard& col, size_t firstEntry);
  ColumnData& FindColumnData(uint8_t deId, uint8_t columnId, size_t firstEntry);

  std::map<uint64_t, std::vector<size_t>> mOrderIndexes; /// Map for time ordering the entries
  std::vector<ColumnData> mData{};                       /// Vector of output column data
  std::vector<ROFRecord> mROFRecords{};                  /// Vector of ROF records
  CrateMapper mCrateMapper;                              /// Mapper to convert the RO info to ColumnData
};
} // namespace mid
} // namespace o2

#endif /* O2_MID_DECODEDDATAAGGREGATOR_H */
