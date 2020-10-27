// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "DataFormatsEMCAL/Digit.h"
#include <iostream>

using namespace o2::emcal;

Digit::Digit(Short_t tower, Double_t amplitudeGeV, Double_t time, ChannelType_t ctype)
  : DigitBase(time), mTower(tower), mChannelType(ctype)
{
  setAmplitude(amplitudeGeV);
}

Digit& Digit::operator+=(const Digit& other)
{
  if (canAdd(other)) {
    Int_t a = getAmplitudeADC() + other.getAmplitudeADC();
    Short_t s = (Short_t)(a);

    if (a < -0x8000) {
      setAmplitudeADC(-0x8000);
    } else if (a <= constants::EMCAL_HGLGTRANSITION) {
      setAmplitudeADC(s);
    } else if (a <= 0x7fff * constants::EMCAL_HGLGFACTOR) {
      setAmplitudeADC(s / constants::EMCAL_HGLGFACTOR, ChannelType_t::LOW_GAIN);
    } else {
      setAmplitudeADC(0x7fff, ChannelType_t::LOW_GAIN);
    }
  }
  // Does nothing if the digits are in different towers or have incompatible times.
  return *this;
}

void Digit::setAmplitudeADC(Short_t amplitude, ChannelType_t ctype)
{
  if (ctype == ChannelType_t::LOW_GAIN) {
    if (amplitude > constants::EMCAL_HGLGTRANSITION / constants::EMCAL_HGLGFACTOR) {
      mAmplitude = amplitude;
      mChannelType = ChannelType_t::LOW_GAIN;
    } else if (amplitude < -0x8000 / constants::EMCAL_HGLGFACTOR) {
      mAmplitude = -0x8000;
      mChannelType = ChannelType_t::HIGH_GAIN;
    } else {
      mAmplitude = amplitude * constants::EMCAL_HGLGFACTOR;
      mChannelType = ChannelType_t::HIGH_GAIN;
    }
  } else {
    if (amplitude > constants::EMCAL_HGLGTRANSITION) {
      mAmplitude = amplitude / constants::EMCAL_HGLGFACTOR;
      mChannelType = ChannelType_t::LOW_GAIN;
    } else {
      mAmplitude = amplitude;
      mChannelType = ctype;
    }
  }
}

Int_t Digit::getAmplitudeADC(ChannelType_t ctype) const
{
  if (ctype == ChannelType_t::LOW_GAIN) {
    // return in units of Low-Gain ADC counts
    if (mChannelType == ChannelType_t::LOW_GAIN) {
      return (Int_t)(mAmplitude);
    } else {
      return mAmplitude / constants::EMCAL_HGLGFACTOR;
    }
  }

  // return in units of High-Gain ADC counts (default)
  if (mChannelType == ChannelType_t::LOW_GAIN) {
    return mAmplitude * constants::EMCAL_HGLGFACTOR;
  }

  return (Int_t)(mAmplitude);
}

void Digit::setAmplitude(Double_t amplitude)
{
  if (amplitude < -0x8000 * constants::EMCAL_ADCENERGY) {
    setAmplitudeADC(-0x8000, ChannelType_t::HIGH_GAIN);
  } else if (amplitude <= constants::EMCAL_HGLGTRANSITION * constants::EMCAL_ADCENERGY) {
    Short_t a = (Short_t)(std::floor(amplitude / constants::EMCAL_ADCENERGY));
    setAmplitudeADC(a, ChannelType_t::HIGH_GAIN);
  } else if (amplitude <= 0x7fff * constants::EMCAL_ADCENERGY * constants::EMCAL_HGLGFACTOR) {
    Short_t a = (Short_t)(std::floor(amplitude / (constants::EMCAL_ADCENERGY * constants::EMCAL_HGLGFACTOR)));
    setAmplitudeADC(a, ChannelType_t::LOW_GAIN);
  } else {
    setAmplitudeADC(0x7fff, ChannelType_t::LOW_GAIN);
  }
}

void Digit::PrintStream(std::ostream& stream) const
{
  stream << "EMCAL Digit: Tower " << mTower << ", Time " << getTimeStamp() << ", Amplitude " << getAmplitude() << " GeV, Type " << mChannelType;
}

std::ostream& operator<<(std::ostream& stream, const Digit& digi)
{
  digi.PrintStream(stream);
  return stream;
}
