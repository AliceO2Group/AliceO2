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

#include <array>
#include <Rtypes.h>

/// \file ChannelData.h
/// \brief Container class to store values of single FDD channel
/// \author micha.broz@cern.ch

namespace o2
{
namespace fdd
{

struct ChannelData {

  int mPMNumber = -1;       // PhotoMultiplier number (0 to 16)
  float mTime = -1024;      // Time of Flight
  short mChargeADC = -1024; // ADC sample
  short mFEEBits = 0;       //Bit information from FEE
  enum Flags { Integrator = 0x1 << 0,
               DoubleEvent = 0x1 << 1,
               Event1TimeLost = 0x1 << 2,
               Event2TimeLost = 0x1 << 3,
               AdcInGate = 0x1 << 4,
               TimeTooLate = 0x1 << 5,
               AmpTooHigh = 0x1 << 6,
               EventInTrigger = 0x1 << 7,
               TimeLost = 0x1 << 8 };

  ChannelData() = default;
  ChannelData(int channel, float time, short adc, short bits) : mPMNumber(channel), mTime(time), mChargeADC(adc), mFEEBits(bits) {}

  void print() const;

  ClassDefNV(ChannelData, 2);
};
} // namespace fdd
} // namespace o2

#endif
