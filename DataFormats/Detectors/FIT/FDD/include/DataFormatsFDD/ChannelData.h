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
/// \author ruben.shahoyan@cern.ch

namespace o2
{
namespace fdd
{

struct ChannelData {

  Int_t mPMNumber;    // PhotoMultiplier number (0 to 16)
  Float_t mTime;      // Time of Flight
  Short_t mChargeADC; // ADC sample
  //Bit information from FEE
  Bool_t mIntegrator;
  Bool_t mDoubleEvent;
  Bool_t mEvent1TimeLost;
  Bool_t mEvent2TimeLost;
  Bool_t mAdcInGate;
  Bool_t mTimeTooLate;
  Bool_t mAmpTooHigh;
  Bool_t mEventInTrigger;
  Bool_t mTimeLost;

  ChannelData() = default;
  ChannelData(Int_t channel, Float_t time, Short_t adc, Bool_t integrator, Bool_t doubleEvent, Bool_t event1TimeLost, Bool_t event2TimeLost, Bool_t adcInGate, Bool_t timeTooLate, Bool_t ampTooHigh, Bool_t eventInTrigger, Bool_t timeLost)
  {
    mPMNumber = channel;
    mTime = time;
    mChargeADC = adc;
    mIntegrator = integrator;
    mDoubleEvent = doubleEvent;
    mEvent1TimeLost = event1TimeLost;
    mEvent2TimeLost = event2TimeLost;
    mAdcInGate = adcInGate;
    mTimeTooLate = timeTooLate;
    mAmpTooHigh = ampTooHigh;
    mEventInTrigger = eventInTrigger;
    mTimeLost = timeLost;
  }

  void print() const;

  ClassDefNV(ChannelData, 1);
};
} // namespace fdd
} // namespace o2

#endif
