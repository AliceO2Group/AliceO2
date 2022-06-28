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

/// \file   MIDRaw/Decoder.h
/// \brief  MID raw data decoder
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   18 November 2019
#ifndef O2_MID_DECODER_H
#define O2_MID_DECODER_H

#include <cstdint>
#if !defined(MID_RAW_VECTORS)
#include <unordered_map>
#endif
#include <vector>
#include <gsl/gsl>
#include "Framework/Logger.h"
#include "DataFormatsMID/ROFRecord.h"
#include "DetectorsRaw/RDHUtils.h"
#include "MIDRaw/CrateMasks.h"
#include "MIDRaw/CrateParameters.h"
#include "MIDRaw/ElectronicsDelay.h"
#include "MIDRaw/FEEIdConfig.h"
#include "MIDRaw/LinkDecoder.h"
#include "MIDRaw/Utils.h"
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
    auto feeId = o2::raw::RDHUtils::getFEEID(rdh);
#if defined(MID_RAW_VECTORS)
    mLinkDecoders[feeId]->process(payload, o2::raw::RDHUtils::getHeartBeatOrbit(rdh), o2::raw::RDHUtils::getTriggerType(rdh), mData, mROFRecords);
#else
    auto linkDecoder = mLinkDecoders.find(feeId);
    if (linkDecoder != mLinkDecoders.end()) {
      linkDecoder->second->process(payload, o2::raw::RDHUtils::getHeartBeatOrbit(rdh), o2::raw::RDHUtils::getTriggerType(rdh), mData, mROFRecords);
    } else {
      LOG(alarm) << "Unexpected feeId " << feeId << " in RDH";
    }
#endif
  }
  /// Gets the vector of data
  const std::vector<ROBoard>& getData() const { return mData; }

  /// Gets the vector of data RO frame records
  const std::vector<ROFRecord>& getROFRecords() const { return mROFRecords; }

  void clear();

 protected:
#if defined(MID_RAW_VECTORS)
  std::vector<std::unique_ptr<LinkDecoder>> mLinkDecoders{}; /// GBT decoders
#else
  std::unordered_map<uint16_t, std::unique_ptr<LinkDecoder>> mLinkDecoders{}; /// GBT decoders
#endif

 private:
  std::vector<ROBoard> mData{};         /// Vector of output data
  std::vector<ROFRecord> mROFRecords{}; /// List of ROF records
};

std::unique_ptr<Decoder> createDecoder(const o2::header::RDHAny& rdh, bool isDebugMode, const ElectronicsDelay& electronicsDelay, const CrateMasks& crateMasks, const FEEIdConfig& feeIdConfig);
std::unique_ptr<Decoder> createDecoder(const o2::header::RDHAny& rdh, bool isDebugMode, const char* electronicsDelayFile = "", const char* crateMasksFile = "", const char* feeIdConfigFile = "");

} // namespace mid
} // namespace o2

#endif /* O2_MID_DECODER_H */
