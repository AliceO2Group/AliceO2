// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MIDRaw/Encoder.h
/// \brief  MID raw data encoder
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   30 September 2019
#ifndef O2_MID_ENCODER_H
#define O2_MID_ENCODER_H

#include <cstdint>
#include <gsl/gsl>
#include <map>
#include "CommonDataFormat/InteractionRecord.h"
#include "DetectorsRaw/HBFUtils.h"
#include "DataFormatsMID/ColumnData.h"
#include "DataFormatsMID/ROFRecord.h"
#include "MIDRaw/CrateMapper.h"
#include "MIDRaw/CrateParameters.h"
#include "MIDRaw/CRUUserLogicEncoder.h"
#include "MIDRaw/LocalBoardRO.h"
#include "MIDRaw/RawUnit.h"

namespace o2
{
namespace mid
{
class Encoder
{
 public:
  void process(gsl::span<const ColumnData> data, const InteractionRecord& ir, EventType eventType = EventType::Standard);
  const std::vector<raw::RawUnit>& getBuffer();
  /// Gets the size in bytes of the buffer
  size_t getBufferSize() { return mBytes.size() * raw::sElementSizeInBytes; }
  /// Sets the next header offset in bytes
  void setHeaderOffset(bool headerOffset = true);
  void clear();

 private:
  bool convertData(gsl::span<const ColumnData> data);
  void newHeader(const InteractionRecord& ir);

  std::array<CRUUserLogicEncoder, crateparams::sNGBTs> mCRUUserLogicEncoders{}; /// Array of encoders per link
  std::map<uint16_t, LocalBoardRO> mROData{};                                   /// Map of data per board
  std::vector<raw::RawUnit> mBytes{};                                           /// Vector with encoded information
  CrateMapper mCrateMapper{};                                                   /// Crate mapper
  o2::raw::HBFUtils mHBFUtils{};                                                /// Utility for HBF
  InteractionRecord mLastIR{};                                                  /// Last interaction record
};
} // namespace mid
} // namespace o2

#endif /* O2_MID_ENCODER_H */
