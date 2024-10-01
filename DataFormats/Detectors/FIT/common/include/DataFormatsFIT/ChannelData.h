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

/// \file ChannelData.h
/// \brief Class represents enity with info per channel in given event
/// \author Artur Furs afurs@cern.ch

#ifndef _FT0_CHANNELDATA_H_
#define _FT0_CHANNELDATA_H_

#include <Rtypes.h>
#include <map>
#include <string>
#include "DetectorsCommonDataFormats/DetID.h"

namespace o2
{
namespace fit
{

template <o2::detectors::DetID::ID DetID>
struct ChannelData {
  static constexpr o2::detectors::DetID sDetID = o2::detectors::DetID(DetID);
  static constexpr uint8_t sDUMMY_CHANNEL_ID = 0xff;
  static constexpr uint8_t sDUMMY_PM_WORD = 0xff;
  static constexpr int16_t sDUMMY_TIME = -5000;
  static constexpr int16_t sDUMMY_AMP = -5000;
  uint8_t mChannelID = sDUMMY_CHANNEL_ID; // channel id
  uint8_t mWordPM = sDUMMY_PM_WORD;       // PM word, based on EBitsPM
  int16_t mTime = sDUMMY_TIME;            // time in TDC units
  int16_t mAmp = sDUMMY_AMP;              // amplitude in ADC units
  enum EBitsPM {
    kNumberADC,
    kIsDoubleEvent,
    kIsTimeInfoNotValid,
    kIsCFDinADCgate,
    kIsTimeInfoLate,
    kIsAmpNotValid,
    kIsVertexEvent,
    kIsTimeInfoLost
  };
  static const inline std::map<unsigned int, std::string> sMapBitsPM = {
    {EBitsPM::kNumberADC, "NumberADC"},
    {EBitsPM::kIsDoubleEvent, "IsDoubleEvent"},
    {EBitsPM::kIsTimeInfoNotValid, "IsTimeInfoNotValid"},
    {EBitsPM::kIsCFDinADCgate, "IsCFDinADCgate"},
    {EBitsPM::kIsTimeInfoLate, "IsTimeInfoLate"},
    {EBitsPM::kIsAmpNotValid, "IsAmpNotValid"},
    {EBitsPM::kIsVertexEvent, "IsVertexEvent"},
    {EBitsPM::kIsTimeInfoLost, "IsTimeInfoLost"}};
  ChannelData() = default;
  ChannelData(uint8_t channelID, uint8_t wordPM, int16_t time, int16_t amp) : mChannelID(channelID), mWordPM(wordPM), mTime(time), mAmp(amp)
  {
  }
  //  void print() const;
  bool operator<=>(ChannelData const& other) const = default;
  bool operator==(ChannelData const& other) const = default;
  ClassDefNV(ChannelData, 1);
};
} // namespace fit
} // namespace o2
#endif
