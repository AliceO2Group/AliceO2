// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file ChannelData.h
/// \brief Class to describe fired and  stored channels for the BC and to refer to channel data
/// \author Alla.Maevskaya@cern.ch

#ifndef _FT0_CHANNELDATA_H_
#define _FT0_CHANNELDATA_H_

#include <Rtypes.h>
#include <tuple>
namespace o2
{
namespace ft0
{
struct ChannelData {
  static constexpr char sChannelNameDPL[] = "DIGITSCH";
  static constexpr char sDigitName[] = "ChannelData";
  static constexpr char sDigitBranchName[] = "FT0DIGITSBCH";
  static constexpr uint8_t DUMMY_CHANNEL_ID = 0xff;
  static constexpr uint8_t DUMMY_CHAIN_QTC = 0xff;
  static constexpr int16_t DUMMY_CFD_TIME = -5000;
  static constexpr int16_t DUMMY_QTC_AMPL = -5000;
  uint8_t ChId = DUMMY_CHANNEL_ID;    //channel Id
  uint8_t ChainQTC = DUMMY_CHAIN_QTC; //QTC chain
  int16_t CFDTime = DUMMY_CFD_TIME;   //time in #CFD channels, 0 at the LHC clk center
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
  bool getFlag(EEventDataBit bitFlag) const { return bool(ChainQTC & (1 << bitFlag)); }
  void print() const;
  void printLog() const;
  [[nodiscard]] uint8_t getChannelID() const { return ChId; }
  [[nodiscard]] uint16_t getTime() const { return CFDTime; }
  [[nodiscard]] uint16_t getAmp() const { return QTCAmpl; }

  bool operator==(ChannelData const& other) const
  {
    return std::tie(ChId, CFDTime, QTCAmpl) == std::tie(other.ChId, other.CFDTime, other.QTCAmpl);
  }
  ClassDefNV(ChannelData, 4);
};
} // namespace ft0
} // namespace o2
#endif
