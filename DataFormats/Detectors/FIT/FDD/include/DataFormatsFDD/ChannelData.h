// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
  uint8_t mPMNumber = -1;     // PhotoMultiplier number (0 to 16)
  int16_t mTime = -1024;      // Time of Flight
  int16_t mChargeADC = -1024; // ADC sample
  uint8_t mFEEBits = 0;       //Bit information from FEE
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
  void print() const;
  bool operator==(ChannelData const& other) const
  {
    return std::tie(mPMNumber, mTime, mChargeADC) == std::tie(other.mPMNumber, other.mTime, other.mChargeADC);
  }
  void printLog() const
  {
    LOG(INFO) << "ChId: " << static_cast<uint16_t>(mPMNumber) << " |  FEE bits:" << static_cast<uint16_t>(mFEEBits) << " | Time: " << mTime << " | Charge: " << mChargeADC;
  }
  ClassDefNV(ChannelData, 3);
};
} // namespace fdd
} // namespace o2

#endif
