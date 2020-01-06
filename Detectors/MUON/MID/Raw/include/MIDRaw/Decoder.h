// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MIDRaw/Decoder.h
/// \brief  Mid raw data decoder
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   30 September 2019
#ifndef O2_MID_DECODER_H
#define O2_MID_DECODER_H

#include <cstdint>
#include <vector>
#include <gsl/gsl>
#include "DataFormatsMID/ColumnData.h"
#include "DataFormatsMID/ROFRecord.h"
#include "MIDRaw/CrateMapper.h"
#include "MIDRaw/CRUUserLogicDecoder.h"
#include "MIDRaw/LocalBoardRO.h"
#include "MIDRaw/RawUnit.h"

namespace o2
{
namespace mid
{
class Decoder
{
 public:
  void process(gsl::span<const raw::RawUnit> bytes);
  /// Gets the vector of data
  const std::vector<ColumnData>& getData() { return mData; }

  /// Gets the vector of data RO frame records
  const std::vector<ROFRecord>& getROFRecords() { return mROFRecords; }

 private:
  void addData(const LocalBoardRO& col, size_t firstEntry);
  ColumnData& FindColumnData(uint8_t deId, uint8_t columnId, size_t firstEntry);

  gsl::span<const raw::RawUnit> mBytes{};                /// Vector with encoded information
  CRUUserLogicDecoder mCRUUserLogicDecoder;              /// CRU user logic decoder
  std::map<uint64_t, std::vector<size_t>> mOrderIndexes; /// Map for time ordering the entries
  std::vector<ColumnData> mData{};                       /// Vector of output column data
  std::vector<ROFRecord> mROFRecords{};                  /// Vector of ROF records
  CrateMapper mCrateMapper;                              /// Mapper to convert the RO info to ColumnData
};
} // namespace mid
} // namespace o2

#endif /* O2_MID_DECODER_H */
