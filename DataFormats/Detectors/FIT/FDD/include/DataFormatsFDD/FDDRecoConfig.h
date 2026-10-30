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

#ifndef ALICEO2_FDD_RECO_PARAM
#define ALICEO2_FDD_RECO_PARAM

#include "CommonUtils/ConfigurableParamHelper.h"
#include "DataFormatsFDD/ChannelData.h"

namespace o2::fdd
{
struct FDDRecoConfig : o2::conf::ConfigurableParamHelper<FDDRecoConfig> {
  int16_t ChannelAmplitudeUpper = 4096;
  int16_t ChannelAmplitudeLower = 0;
  int16_t ChannelTimeLower = -2050;
  int16_t ChannelTimeUpper = 2050;

  uint8_t ChannelValidPmInputFlagMask = 0;
  uint8_t ChannelValidPmInputFlags = 0;
  bool isChannelDataGood(uint8_t flags) const
  {
    return (flags & ChannelValidPmInputFlagMask) == ChannelValidPmInputFlags;
  }

  uint8_t TriggerBitsValidationMask = 0;
  uint8_t TriggerBitsValid = 0;
  bool areTriggerBitsGood(uint8_t flags) const
  {
    return (flags & TriggerBitsValidationMask) == TriggerBitsValid;
  }

  int16_t CollisionTimeAmpltudeUpper = 4096;
  int16_t CollisionTimeAmplitudeLower = 0;
  int16_t CollisionTimeTimeLower = -500;
  int16_t CollisionTimeTimeUpper = 500;

  uint8_t CollisionTimeValidPmInputFlagMask = static_cast<uint8_t>(~(1u << ChannelData::kNumberADC));
  uint8_t CollisionTimeValidPmInputFlags = static_cast<uint8_t>((1u << ChannelData::kIsCFDinADCgate) | (1u << ChannelData::kIsEventInTVDC));
  bool isChannelDataGoodForCollisionTime(uint8_t flags) const
  {
    return (flags & CollisionTimeValidPmInputFlagMask) == CollisionTimeValidPmInputFlags;
  }

  int16_t SwTriggerTimeLowThreshold = -153;
  int16_t SwTriggerTimeHighThreshold = 153;
  int16_t SwTriggerAmplitudeLowThreshold = 10;
  int16_t SwTriggerAmplitudeHighThreshold = 4096;
  float SwTriggerTimeDiffMin = -1.3;
  float SwTriggerTimeDiffMax = 1.3;
  bool SwTriggerEnabled = false;

  O2ParamDef(FDDRecoConfig, "FDDRecoConfig");
  ClassDefNV(FDDRecoConfig, 2);
};
} // namespace o2::fdd
#endif