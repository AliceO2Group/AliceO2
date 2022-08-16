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

#ifndef _FDD_CHANNEL_DATA_H_
#define _FDD_CHANNEL_DATA_H_

#include <Framework/Logger.h>
#include <array>
#include <Rtypes.h>
#include <tuple>
/// \file ChannelData.h
/// \brief Container class to store values of single FDD channel
/// \author micha.broz@cern.ch

namespace o2
{
namespace fdd
{

struct ChannelData {
  static constexpr char sChannelNameDPL[] = "DIGITSCH";
  static constexpr char sDigitName[] = "ChannelData";
  static constexpr char sDigitBranchName[] = "FDDDigitCh";
  static constexpr uint8_t DUMMY_CHANNEL_ID = 0xff;
  static constexpr uint8_t DUMMY_CHAIN_QTC = 0;
  static constexpr int16_t DUMMY_CFD_TIME = -5000;
  static constexpr int16_t DUMMY_QTC_AMPL = -5000;
  uint8_t mPMNumber = DUMMY_CHANNEL_ID; // PhotoMultiplier number (0 to 16)
  int16_t mTime = DUMMY_CFD_TIME;       // Time of Flight
  int16_t mChargeADC = DUMMY_QTC_AMPL;  // ADC sample
  uint8_t mFEEBits = DUMMY_CHAIN_QTC;   // Bit information from FEE
  /*  enum Flags { Integrator = 0x1 << 0,
DoubleEvent = 0x1 << 1,
Event1TimeLost = 0x1 << 2,
Event2TimeLost = 0x1 << 3,
AdcInGate = 0x1 << 4,
TimeTooLate = 0x1 << 5,
AmpTooHigh = 0x1 << 6,
EventInTrigger = 0x1 << 7,
TimeLost = 0x1 << 8 };*/
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
  ChannelData(uint8_t channel, int time, int adc, uint8_t bits) : mPMNumber(channel), mTime(time), mChargeADC(adc), mFEEBits(bits) {}
  uint8_t getChannelID() const { return mPMNumber; }
  static void setFlag(EEventDataBit bitFlag, uint8_t& mFEEBits) { mFEEBits |= (1 << bitFlag); }
  static void clearFlag(EEventDataBit bitFlag, uint8_t& mFEEBits) { mFEEBits &= ~(1 << bitFlag); }
  bool getFlag(EEventDataBit bitFlag) const { return bool(mFEEBits & (1 << bitFlag)); }
  void print() const;
  bool operator==(ChannelData const& other) const
  {
    return std::tie(mPMNumber, mTime, mChargeADC) == std::tie(other.mPMNumber, other.mTime, other.mChargeADC);
  }
  void printLog() const
  {
    LOG(info) << "ChId: " << static_cast<uint16_t>(mPMNumber) << " |  FEE bits:" << static_cast<uint16_t>(mFEEBits) << " | Time: " << mTime << " | Charge: " << mChargeADC;
  }
  ClassDefNV(ChannelData, 4);
};
} // namespace fdd
} // namespace o2

#endif
