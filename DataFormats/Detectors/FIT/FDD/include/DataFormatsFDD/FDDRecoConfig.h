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
  float AmplitudeCutOnCollisionTimeWeights = 3;
  float AmplitudeCollisionTimeCutSideA = 1.0;
  float AmplitudeCollisionTimeCutSideC = 0.0;
  
  float SwTriggerTimeLowThreshold= -1.25;
  float SwTriggerTimeHighThreshold= 1.25;
  float SwTriggerAmplitudeLowThreshold = 0.0;
  float SwTriggerAmplitudeHighThreshold = 4096.0;
  bool SwTriggerEnabled = false;

  uint8_t ValidPmInputFlagMask = static_cast<uint8_t>(~(1u << ChannelData::kNumberADC));
  uint8_t ValidPmInputFlags = static_cast<uint8_t>((1u << ChannelData::kIsCFDinADCgate) | (1u << ChannelData::kIsEventInTVDC));
  bool areChannelDataFlagsGood(uint8_t flags) const
  {
    return (flags & mValidPmInputFlagMask) == mValidPmInputFlags;
  }

  O2ParamDef(FDDRecoConfig, "FDDRecoConfig");
  ClassDefNV(FDDRecoConfig, 1);
};
} // namespace o2::fdd
#endif