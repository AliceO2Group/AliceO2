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

/// \file   MIDRaw/Encoder.h
/// \brief  MID raw data encoder
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   30 September 2019
#ifndef O2_MID_ENCODER_H
#define O2_MID_ENCODER_H

#include <cstdint>
#include <array>
#include <map>
#include <string_view>
#include <vector>
#include <gsl/gsl>
#include "CommonDataFormat/InteractionRecord.h"
#include "DetectorsRaw/RawFileWriter.h"
#include "DataFormatsMID/ColumnData.h"
#include "DataFormatsMID/ROBoard.h"
#include "MIDRaw/ColumnDataToLocalBoard.h"
#include "MIDRaw/CrateParameters.h"
#include "MIDRaw/ElectronicsDelay.h"
#include "MIDRaw/FEEIdConfig.h"
#include "MIDRaw/GBTUserLogicEncoder.h"

class RDHAny;

namespace o2
{
namespace mid
{
class Encoder
{
 public:
  void init(std::string_view outDir = ".", std::string_view fileFor = "all", int verbosity = 0, std::vector<ROBoardConfig> configurations = makeDefaultROBoardConfig());
  void process(gsl::span<const ColumnData> data, InteractionRecord ir, EventType eventType = EventType::Standard);
  /// Sets the maximum size of the superpage
  void setSuperpageSize(int maxSize) { mRawWriter.setSuperPageSize(maxSize); }

  void finalize(bool closeFile = true);

  auto& getWriter() { return mRawWriter; }

  void emptyHBFMethod(const o2::header::RDHAny* rdh, std::vector<char>& toAdd) const;

 private:
  void completeWord(std::vector<char>& buffer);
  void writePayload(uint16_t linkId, const InteractionRecord& ir, bool onlyNonEmpty = false);
  void onOrbitChange(uint32_t orbit);
  /// Returns the interaction record expected for the orbit trigger
  inline InteractionRecord getOrbitIR(uint32_t orbit) const { return {o2::constants::lhc::LHCMaxBunches - 1, orbit}; }
  /// Initializes the last interaction record
  void initIR();

  o2::raw::RawFileWriter mRawWriter{o2::header::gDataOriginMID}; /// Raw file writer

  std::map<uint16_t, ROBoard> mROData;                        /// Map of data per board
  ColumnDataToLocalBoard mConverter;                          /// ColumnData to ROBoard converter
  std::unordered_map<uint16_t, std::vector<ROBoard>> mGBTMap; /// ROBoard per GBT link
  FEEIdConfig mFEEIdConfig;                                   /// Crate FEEId mapper
  InteractionRecord mLastIR;                                  /// Last interaction record
  ElectronicsDelay mElectronicsDelay;                         /// Delays in the electronics

  std::array<GBTUserLogicEncoder, crateparams::sNGBTs> mGBTEncoders{}; /// Array of encoders per link
  std::array<std::vector<char>, 4> mOrbitResponse{};                   /// Response to orbit trigger
  std::array<std::vector<char>, 4> mOrbitResponseWord{};               /// CRU word for response to orbit trigger
};
} // namespace mid
} // namespace o2

#endif /* O2_MID_ENCODER_H */
