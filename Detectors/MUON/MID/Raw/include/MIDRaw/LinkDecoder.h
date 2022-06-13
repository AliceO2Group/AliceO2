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

/// \file   MIDRaw/LinkDecoder.h
/// \brief  Class interface for the MID link decoder
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   07 November 2020
#ifndef O2_MID_LINKDECODER_H
#define O2_MID_LINKDECODER_H

#include <cstdint>
#include <vector>
#include <functional>
#include <gsl/gsl>
#include "DetectorsRaw/RDHUtils.h"
#include "Headers/RAWDataHeader.h"
#include "DataFormatsMID/ROBoard.h"
#include "DataFormatsMID/ROFRecord.h"
#include "MIDRaw/ElectronicsDelay.h"
#include "MIDRaw/FEEIdConfig.h"

namespace o2
{
namespace mid
{

class LinkDecoder
{
 public:
  LinkDecoder(std::function<void(gsl::span<const uint8_t>, uint32_t orbit, uint32_t trigger, std::vector<ROBoard>& data, std::vector<ROFRecord>& rofs)> decode) : mDecode(decode) {}
  void process(gsl::span<const uint8_t> payload, uint32_t orbit, uint32_t trigger, std::vector<ROBoard>& data, std::vector<ROFRecord>& rofs);

  template <class RDH>
  void process(gsl::span<const uint8_t> payload, const RDH& rdh, std::vector<ROBoard>& data, std::vector<ROFRecord>& rofs)
  {
    process(payload, o2::raw::RDHUtils::getHeartBeatOrbit(rdh), o2::raw::RDHUtils::getTriggerType(rdh), data, rofs);
  }

 protected:
  std::function<void(gsl::span<const uint8_t>, uint32_t orbit, uint32_t trigger, std::vector<ROBoard>& data, std::vector<ROFRecord>& rofs)> mDecode{nullptr};
};

std::unique_ptr<LinkDecoder> createGBTDecoder(const o2::header::RDHAny& rdh, uint16_t feeId, bool isDebugMode, uint8_t mask, const ElectronicsDelay& electronicsDelay);

std::unique_ptr<LinkDecoder> createLinkDecoder(const o2::header::RDHAny& rdh, uint16_t feeId, bool isDebugMode, uint8_t mask, const ElectronicsDelay& electronicsDelay, const FEEIdConfig& feeIdConfig);
std::unique_ptr<LinkDecoder> createLinkDecoder(uint16_t feeId, bool isBare = false, bool isDebugMode = false, uint8_t mask = 0xFF, const ElectronicsDelay& electronicsDelay = ElectronicsDelay(), const FEEIdConfig& feeIdConfig = FEEIdConfig());

} // namespace mid
} // namespace o2

#endif /* O2_MID_LINKDECODER_H */
