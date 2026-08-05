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

#ifndef ALICEO2_FV0_DIGIT_FILTER_PARAM
#define ALICEO2_FV0_DIGIT_FILTER_PARAM

#include "CommonUtils/ConfigurableParamHelper.h"
#include "DataFormatsFV0/ChannelData.h"

namespace o2::fv0
{
struct FV0RecoConfig : o2::conf::ConfigurableParamHelper<FV0RecoConfig> {
  double AmplitudeLowerThreshold = 24;     // only channels with amplitude higher will participate in calibration and collision time
  double AmplitudeThreholdForMeanTime = 5; // Charge threshold, only above which the time is taken into account in calculating the mean time of all qualifying channels
  double TimeUpperThershold = 1000.0;      // only channels with time below will participate in calibration and collision time
  uint8_t mValidPmInputFlagMask = ~(1u << ChannelData::kNumberADC);
  uint8_t mValidPmInputFlags = static_cast<uint8_t>((1u << ChannelData::kIsCFDinADCgate) | (1u << ChannelData::kIsEventInTVDC));

  bool areChannelDataFlagsGood(uint8_t flags) const
  {
    return (flags & mValidPmInputFlagMask) == mValidPmInputFlags;
  }
  O2ParamDef(FV0RecoConfig, "FV0RecoConfig");
};

} // namespace o2::fv0

#endif