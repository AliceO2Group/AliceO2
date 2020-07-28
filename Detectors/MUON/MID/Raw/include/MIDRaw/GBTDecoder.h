// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MIDRaw/GBTDecoder.h
/// \brief  Class interface for the MID GBT decoder
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   07 November 2020
#ifndef O2_MID_GBTDECODER_H
#define O2_MID_GBTDECODER_H

#include <cstdint>
#include <vector>
#include <gsl/gsl>
#include "DetectorsRaw/RDHUtils.h"
#include "Headers/RAWDataHeader.h"
#include "DataFormatsMID/ROFRecord.h"
#include "MIDRaw/ElectronicsDelay.h"
#include "DataFormatsMID/ROBoard.h"

namespace o2
{
namespace mid
{

class GBTDecoder
{
 public:
  GBTDecoder(std::function<void(gsl::span<const uint8_t>, uint32_t orbit, std::vector<ROBoard>& data, std::vector<ROFRecord>& rofs)> decode) : mDecode(decode) {}
  void process(gsl::span<const uint8_t> payload, uint32_t orbit, std::vector<ROBoard>& data, std::vector<ROFRecord>& rofs);

  template <class RDH>
  void process(gsl::span<const uint8_t> payload, const RDH& rdh, std::vector<ROBoard>& data, std::vector<ROFRecord>& rofs)
  {
    process(payload, o2::raw::RDHUtils::getHeartBeatOrbit(rdh), data, rofs);
  }

 protected:
  std::function<void(gsl::span<const uint8_t>, uint32_t orbit, std::vector<ROBoard>& data, std::vector<ROFRecord>& rofs)> mDecode{nullptr};
};

std::unique_ptr<GBTDecoder> createGBTDecoder(const o2::header::RDHAny& rdh, uint16_t feeId, bool isDebugMode, uint8_t mask, const ElectronicsDelay& electronicsDelay);
std::unique_ptr<GBTDecoder> createGBTDecoder(uint16_t feeId, bool isBare = false, bool isDebugMode = false, uint8_t mask = 0xFF, const ElectronicsDelay& electronicsDelay = ElectronicsDelay());

} // namespace mid
} // namespace o2

#endif /* O2_MID_GBTDECODER_H */
