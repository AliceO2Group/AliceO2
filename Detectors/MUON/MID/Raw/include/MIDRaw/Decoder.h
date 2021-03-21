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
/// \brief  MID raw data decoder
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   18 November 2019
#ifndef O2_MID_DECODER_H
#define O2_MID_DECODER_H

#include <cstdint>
#include <vector>
#include <array>
#include <gsl/gsl>
#include "DataFormatsMID/ROFRecord.h"
#include "DetectorsRaw/RDHUtils.h"
#include "MIDRaw/CrateMasks.h"
#include "MIDRaw/CrateParameters.h"
#include "MIDRaw/ElectronicsDelay.h"
#include "MIDRaw/FEEIdConfig.h"
#include "MIDRaw/GBTDecoder.h"
#include "DataFormatsMID/ROBoard.h"

namespace o2
{
namespace mid
{

class Decoder
{
 public:
  Decoder(bool isDebugMode = false, bool isBare = false, const ElectronicsDelay& electronicsDelay = ElectronicsDelay(), const CrateMasks& crateMasks = CrateMasks(), const FEEIdConfig& feeIdConfig = FEEIdConfig());
  virtual ~Decoder() = default;
  void process(gsl::span<const uint8_t> bytes);
  template <class RDH>
  void process(gsl::span<const uint8_t> payload, const RDH& rdh)
  {
    /// Processes the page
    auto feeId = mGetFEEID(rdh);
    mGBTDecoders[feeId]->process(payload, o2::raw::RDHUtils::getHeartBeatOrbit(rdh), mData, mROFRecords);
  }
  /// Gets the vector of data
  const std::vector<ROBoard>& getData() const { return mData; }

  /// Gets the vector of data RO frame records
  const std::vector<ROFRecord>& getROFRecords() const { return mROFRecords; }

  void clear();

 protected:
  /// Gets the feeID
  std::function<uint16_t(const o2::header::RDHAny& rdh)> mGetFEEID{[](const o2::header::RDHAny& rdh) { return o2::raw::RDHUtils::getFEEID(rdh); }};

  std::array<std::unique_ptr<GBTDecoder>, crateparams::sNGBTs> mGBTDecoders{nullptr}; /// GBT decoders

 private:
  std::vector<ROBoard> mData{};         /// Vector of output data
  std::vector<ROFRecord> mROFRecords{}; /// List of ROF records
};

std::unique_ptr<Decoder> createDecoder(const o2::header::RDHAny& rdh, bool isDebugMode, ElectronicsDelay& electronicsDelay, const CrateMasks& crateMasks, const FEEIdConfig& feeIdConfig);
std::unique_ptr<Decoder> createDecoder(const o2::header::RDHAny& rdh, bool isDebugMode, const char* electronicsDelayFile = "", const char* crateMasksFile = "", const char* feeIdConfigFile = "");

} // namespace mid
} // namespace o2

#endif /* O2_MID_DECODER_H */
