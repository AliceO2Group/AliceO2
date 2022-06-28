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
/// \brief Container class to store time and charge values of single FV0 channel
/// \author maciej.slupecki@cern.ch

#ifndef _FV0_CHANNEL_DATA_H_
#define _FV0_CHANNEL_DATA_H_

#include <Rtypes.h>
#include <tuple>
namespace o2
{
namespace fv0
{

struct ChannelData {
  static constexpr char sChannelNameDPL[] = "DIGITSCH";
  static constexpr char sDigitName[] = "ChannelData";
  static constexpr char sDigitBranchName[] = "FV0DigitCh";
  static constexpr uint8_t DUMMY_CHANNEL_ID = 0xff;
  static constexpr uint8_t DUMMY_CHAIN_QTC = 0xff;
  static constexpr int16_t DUMMY_CFD_TIME = -5000;
  static constexpr int16_t DUMMY_QTC_AMPL = -5000;
  uint8_t ChId = DUMMY_CHANNEL_ID;    // channel Id
  uint8_t ChainQTC = DUMMY_CHAIN_QTC; // QTC chain
  int16_t CFDTime = DUMMY_CFD_TIME;   // time in #CFD channels, 0 at the LHC clk center
  int16_t QTCAmpl = DUMMY_QTC_AMPL;   // Amplitude #channels
  enum EEventDataBit { kNumberADC,
                       kIsDoubleEvent,
                       kIsTimeInfoNOTvalid,
                       kIsCFDinADCgate,
                       kIsTimeInfoLate,
                       kIsAmpHigh,
                       kIsEventInTVDC,
                       kIsTimeInfoLost
  };

  ChannelData() = default;
  ChannelData(uint8_t iPmt, int time, int charge, uint8_t chainQTC)
  {
    ChId = iPmt;
    CFDTime = time;
    QTCAmpl = charge;
    ChainQTC = chainQTC;
  }
  void setFlag(uint8_t flag)
  {
    ChainQTC = flag;
  }
  void setFlag(EEventDataBit bitFlag, bool value)
  {
    ChainQTC |= (value << bitFlag);
  }
  static void setFlag(EEventDataBit bitFlag, uint8_t& chainQTC) { chainQTC |= (1 << bitFlag); }
  static void clearFlag(EEventDataBit bitFlag, uint8_t& chainQTC) { chainQTC &= ~(1 << bitFlag); }
  bool getFlag(EEventDataBit bitFlag) const { return bool(ChainQTC & (1 << bitFlag)); }
  void print() const;
  void printLog() const;
  [[nodiscard]] uint8_t getChannelID() const { return ChId; }
  [[nodiscard]] uint16_t getTime() const { return CFDTime; }
  [[nodiscard]] uint16_t getAmp() const { return QTCAmpl; }

  bool operator==(ChannelData const& other) const
  {
    return std::tie(ChId, CFDTime, QTCAmpl, ChainQTC) == std::tie(other.ChId, other.CFDTime, other.QTCAmpl, other.ChainQTC);
  }
  ClassDefNV(ChannelData, 3);
};
} // namespace fv0
} // namespace o2

#endif
