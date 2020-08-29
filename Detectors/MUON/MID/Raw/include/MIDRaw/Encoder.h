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
#include <array>
#include <map>
#include <gsl/gsl>
#include "CommonDataFormat/InteractionRecord.h"
#include "DetectorsRaw/RawFileWriter.h"
#include "DataFormatsMID/ColumnData.h"
#include "MIDRaw/ColumnDataToLocalBoard.h"
#include "MIDRaw/CrateParameters.h"
#include "MIDRaw/FEEIdConfig.h"
#include "MIDRaw/GBTUserLogicEncoder.h"
#include "MIDRaw/LocalBoardRO.h"

namespace o2
{
namespace mid
{
class Encoder
{
 public:
  void init(const char* filename, int verbosity = 0);
  void process(gsl::span<const ColumnData> data, const InteractionRecord& ir, EventType eventType = EventType::Standard);
  /// Sets the maximum size of the superpage
  void setSuperpageSize(int maxSize) { mRawWriter.setSuperPageSize(maxSize); }

  void finalize(bool closeFile = true);

  auto& getWriter() { return mRawWriter; }

 private:
  void flush(uint16_t feeId, const InteractionRecord& ir);
  void hbTrigger(const InteractionRecord& ir);

  o2::raw::RawFileWriter mRawWriter{o2::header::gDataOriginMID}; /// Raw file writer

  std::map<uint16_t, LocalBoardRO> mROData{}; /// Map of data per board
  ColumnDataToLocalBoard mConverter{};        /// ColumnData to LocalBoardRO converter
  FEEIdConfig mFEEIdConfig{};                 /// Crate FEEId mapper
  InteractionRecord mLastIR{};                /// Last interaction record

  std::array<GBTUserLogicEncoder, crateparams::sNGBTs> mGBTEncoders{}; /// Array of encoders per link
  std::array<uint32_t, crateparams::sNGBTs> mGBTIds{};                 /// Array of GBT Ids
};
} // namespace mid
} // namespace o2

#endif /* O2_MID_ENCODER_H */
