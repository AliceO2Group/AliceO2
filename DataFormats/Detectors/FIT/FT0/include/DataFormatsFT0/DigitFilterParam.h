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

/// \file DigitFilterParam.h
/// \brief Configurable digit filtering

#ifndef ALICEO2_DIGIT_FILTER_PARAM
#define ALICEO2_DIGIT_FILTER_PARAM

#include "CommonUtils/ConfigurableParamHelper.h"
#include "DataFormatsFT0/Digit.h"
#include "DataFormatsFT0/ChannelData.h"

namespace o2::ft0
{
struct DigitFilterParam : o2::conf::ConfigurableParamHelper<DigitFilterParam> {
  int16_t mAmpThreshold = 10;
  int16_t mTimeWindow = 153;
  uint8_t mPMbitsGood = (1 << ChannelData::EEventDataBit::kIsCFDinADCgate) | (1 << ChannelData::EEventDataBit::kIsEventInTVDC);
  uint8_t mPMbitsBad = (1 << ChannelData::EEventDataBit::kIsDoubleEvent) | (1 << ChannelData::EEventDataBit::kIsTimeInfoNOTvalid) | (1 << ChannelData::EEventDataBit::kIsTimeInfoLate) | (1 << ChannelData::EEventDataBit::kIsAmpHigh) | (1 << ChannelData::EEventDataBit::kIsTimeInfoLost);
  uint8_t mPMbitsToCheck = mPMbitsGood | mPMbitsBad;
  uint8_t mTrgBitsGood = (1 << Triggers::bitVertex) | (1 << Triggers::bitDataIsValid);
  uint8_t mTrgBitsBad = (1 << Triggers::bitOutputsAreBlocked);
  uint8_t mTrgBitsToCheck = mTrgBitsGood | mTrgBitsBad;

  O2ParamDef(DigitFilterParam, "FT0DigitFilterParam");
};
} // namespace o2::ft0
#endif