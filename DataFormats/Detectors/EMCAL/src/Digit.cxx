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

#include "DataFormatsEMCAL/Digit.h"
#include <iostream>

using namespace o2::emcal;

Digit::Digit(Short_t tower, Double_t amplitudeGeV, Double_t time)
  : DigitBase(time), mTower(tower), mAmplitudeGeV(amplitudeGeV)
{
}

Digit::Digit(Short_t tower, uint16_t noiseLG, uint16_t noiseHG, Double_t time)
  : DigitBase(time), mNoiseLG(noiseLG), mNoiseHG(noiseHG), mTower(tower)
{
}

Digit& Digit::operator+=(const Digit& other)
{
  if (canAdd(other)) {
    mAmplitudeGeV += other.mAmplitudeGeV;
    mNoiseHG += other.mNoiseHG;
    mNoiseLG += other.mNoiseLG;
  }
  return *this;
}

void Digit::setAmplitudeADC(Short_t amplitude, ChannelType_t ctype)
{

  // truncate energy in case dynamic range is saturated
  if (amplitude >= constants::MAX_RANGE_ADC) {
    amplitude = constants::MAX_RANGE_ADC;
  }

  switch (ctype) {
    case ChannelType_t::HIGH_GAIN: {
      mAmplitudeGeV = amplitude * constants::EMCAL_ADCENERGY;
      break;
    };
    case ChannelType_t::LOW_GAIN: {
      mAmplitudeGeV = amplitude * (constants::EMCAL_ADCENERGY * constants::EMCAL_HGLGFACTOR);
      break;
    };
    case ChannelType_t::TRU: {
      mAmplitudeGeV = amplitude * constants::EMCAL_TRU_ADCENERGY;
      break;
    };

    default:
      // can only be LEDMon which is not simulated
      mAmplitudeGeV = 0.;
      break;
  };
}

Int_t Digit::getAmplitudeADC(ChannelType_t ctype) const
{

  switch (ctype) {
    case ChannelType_t::HIGH_GAIN: {
      int ampADC = std::floor(mAmplitudeGeV / constants::EMCAL_ADCENERGY);
      // truncate energy in case dynamic range is saturated
      if (ampADC >= constants::MAX_RANGE_ADC) {
        return constants::MAX_RANGE_ADC;
      }
      return ampADC + mNoiseHG;
    };
    case ChannelType_t::LOW_GAIN: {
      int ampADC = std::floor(mAmplitudeGeV / (constants::EMCAL_ADCENERGY * constants::EMCAL_HGLGFACTOR));
      // truncate energy in case dynamic range is saturated
      if (ampADC >= constants::MAX_RANGE_ADC) {
        return constants::MAX_RANGE_ADC;
      }
      return ampADC + mNoiseLG;
    };
    case ChannelType_t::TRU: {
      int ampADC = std::floor(mAmplitudeGeV / constants::EMCAL_TRU_ADCENERGY);
      // truncate energy in case dynamic range is saturated
      if (ampADC >= constants::MAX_RANGE_ADC) {
        return constants::MAX_RANGE_ADC;
      }
      return ampADC;
    };

    default:
      // can only be LEDMon which is not simulated
      return 0;
  };
}

Double_t Digit::getAmplitude() const
{
  double noise = 0;

  switch (getType()) {
    case ChannelType_t::HIGH_GAIN: {
      noise = mNoiseHG * constants::EMCAL_ADCENERGY;
      return mAmplitudeGeV + noise;
    };
    case ChannelType_t::LOW_GAIN: {
      noise = mNoiseLG * (constants::EMCAL_ADCENERGY * constants::EMCAL_HGLGFACTOR);
      return mAmplitudeGeV + noise;
    };
    case ChannelType_t::TRU: {
      noise = mNoiseHG * constants::EMCAL_TRU_ADCENERGY;
      return mAmplitudeGeV + noise;
    };

    default:
      // can only be LEDMon which is not simulated
      return 0;
  };
}

ChannelType_t Digit::getType() const
{

  if (mIsTRU) {
    return ChannelType_t::TRU;
  }
  constexpr double ENERGYHGLGTRANISITION = (constants::EMCAL_HGLGTRANSITION * constants::EMCAL_ADCENERGY);

  if (mAmplitudeGeV < ENERGYHGLGTRANISITION) {
    return ChannelType_t::HIGH_GAIN;
  }

  return ChannelType_t::LOW_GAIN;
}

void Digit::PrintStream(std::ostream& stream) const
{
  stream << "EMCAL Digit: Tower " << mTower << ", Time " << getTimeStamp() << ", Amplitude " << getAmplitude() << " GeV, Type " << channelTypeToString(getType());
}

std::ostream& operator<<(std::ostream& stream, const Digit& digi)
{
  digi.PrintStream(stream);
  return stream;
}
