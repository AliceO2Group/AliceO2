// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MIDRaw/CRUUserLogicDecoder.h
/// \brief  MID CRU user logic decoder
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   18 November 2019
#ifndef O2_MID_CRUUSERLOGICDECODER_H
#define O2_MID_CRUUSERLOGICDECODER_H

#include <cstdint>
#include <vector>
#include <gsl/gsl>
#include "DataFormatsMID/ROFRecord.h"
#include "MIDRaw/LocalBoardRO.h"
#include "MIDRaw/RawBuffer.h"
#include "MIDRaw/RawUnit.h"

namespace o2
{
namespace mid
{
class CRUUserLogicDecoder
{
 public:
  void process(gsl::span<const raw::RawUnit> bytes);
  /// Gets the vector of data
  const std::vector<LocalBoardRO>& getData() { return mData; }

  /// Gets the vector of data RO frame records
  const std::vector<ROFRecord>& getROFRecords() { return mROFRecords; }

 private:
  RawBuffer<raw::RawUnit> mBuffer{};    /// Raw buffer handler
  std::vector<LocalBoardRO> mData{};    /// Vector of output data
  std::vector<ROFRecord> mROFRecords{}; /// List of ROF records
  uint8_t mCrateId{0};                  /// Crate ID (from RDH)

  bool processBlock();
  void reset();
};
} // namespace mid
} // namespace o2

#endif /* O2_MID_CRUUSERLOGICDECODER_H */
