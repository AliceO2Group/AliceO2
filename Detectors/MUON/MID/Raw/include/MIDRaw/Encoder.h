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
#include <vector>
#include <gsl/gsl>
#include "CommonDataFormat/InteractionRecord.h"
#include "DetectorsRaw/RawFileWriter.h"
#include "DataFormatsMID/ColumnData.h"
#include "MIDRaw/ColumnDataToLocalBoard.h"
#include "MIDRaw/CrateParameters.h"
#include "MIDRaw/FEEIdConfig.h"
#include "MIDRaw/GBTUserLogicEncoder.h"
#include "DataFormatsMID/ROBoard.h"

class RDHAny;

namespace o2
{
namespace mid
{
class Encoder
{
 public:
  void init(const char* filename, bool perLink = false, int verbosity = 0, bool debugMode = false);
  void process(gsl::span<const ColumnData> data, const InteractionRecord& ir, EventType eventType = EventType::Standard);
  /// Sets the maximum size of the superpage
  void setSuperpageSize(int maxSize) { mRawWriter.setSuperPageSize(maxSize); }

  void finalize(bool closeFile = true);

  auto& getWriter() { return mRawWriter; }

  void emptyHBFMethod(const o2::header::RDHAny* rdh, std::vector<char>& toAdd) const;

 private:
  void completeWord(std::vector<char>& buffer);
  void writePayload(uint16_t linkId, const InteractionRecord& ir);
  void onOrbitChange(uint32_t orbit);
  /// Returns the interaction record expected for the orbit trigger
  inline InteractionRecord getOrbitIR(uint32_t orbit) const { return {o2::constants::lhc::LHCMaxBunches - 1, orbit}; }

  o2::raw::RawFileWriter mRawWriter{o2::header::gDataOriginMID}; /// Raw file writer

  std::map<uint16_t, ROBoard> mROData{}; /// Map of data per board
  ColumnDataToLocalBoard mConverter{};   /// ColumnData to ROBoard converter
  FEEIdConfig mFEEIdConfig{};            /// Crate FEEId mapper
  InteractionRecord mLastIR{};           /// Last interaction record

  std::array<uint32_t, crateparams::sNGBTs> mGBTIds{};                 /// Array of GBT Ids
  std::array<GBTUserLogicEncoder, crateparams::sNGBTs> mGBTEncoders{}; /// Array of encoders per link
  std::array<std::vector<char>, 4> mOrbitResponse{};                   /// Response to orbit trigger
  std::array<std::vector<char>, 4> mOrbitResponseWord{};               /// CRU word for response to orbit trigger
};
} // namespace mid
} // namespace o2

#endif /* O2_MID_ENCODER_H */
