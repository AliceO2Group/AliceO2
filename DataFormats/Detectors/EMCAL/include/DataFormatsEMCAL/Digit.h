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

#ifndef ALICEO2_EMCAL_DIGIT_H_
#define ALICEO2_EMCAL_DIGIT_H_

#include <iosfwd>
#include <cmath>
#include "Rtypes.h"
#include "CommonDataFormat/TimeStamp.h"
#include "DataFormatsEMCAL/Constants.h"

#include <boost/serialization/base_object.hpp> // for base_object

namespace o2
{

namespace emcal
{
using DigitBase = o2::dataformats::TimeStamp<double>;

/// \class Digit
/// \brief EMCAL digit implementation
/// \ingroup EMCALDataFormat
class Digit : public DigitBase
{
 public:
  Digit() = default;

  Digit(Short_t tower, Double_t amplitudeGeV, Double_t time);
  Digit(Short_t tower, uint16_t noiseLG, uint16_t noiseHG, double time);
  ~Digit() = default; // override

  bool operator<(const Digit& other) const { return getTimeStamp() < other.getTimeStamp(); }
  bool operator>(const Digit& other) const { return getTimeStamp() > other.getTimeStamp(); }
  bool operator==(const Digit& other) const { return getTimeStamp() == other.getTimeStamp(); }

  bool canAdd(const Digit other)
  {
    return (mTower == other.getTower() && std::abs(getTimeStamp() - other.getTimeStamp()) < constants::EMCAL_TIMESAMPLE);
  }

  Digit& operator+=(const Digit& other);              // Adds amplitude of other digits to this digit.
  friend Digit operator+(Digit lhs, const Digit& rhs) // Adds amplitudes of two digits.
  {
    lhs += rhs;
    return lhs;
  }

  void setTower(Short_t tower) { mTower = tower; }
  Short_t getTower() const { return mTower; }

  void setAmplitude(Double_t amplitude) { mAmplitudeGeV = amplitude; }
  Double_t getAmplitude() const;

  void setEnergy(Double_t energy) { mAmplitudeGeV = energy; }
  Double_t getEnergy() const { return mAmplitudeGeV; }

  void setAmplitudeADC(Short_t amplitude, ChannelType_t ctype = ChannelType_t::HIGH_GAIN);
  Int_t getAmplitudeADC(ChannelType_t ctype) const;
  Int_t getAmplitudeADC() const { return getAmplitudeADC(getType()); };

  void setType(ChannelType_t ctype) {}
  ChannelType_t getType() const;

  void setHighGain() {}
  Bool_t getHighGain() const { return (getType() == ChannelType_t::HIGH_GAIN); };

  void setLowGain() {}
  Bool_t getLowGain() const { return (getType() == ChannelType_t::LOW_GAIN); };

  void setTRU() { mIsTRU = true; }
  Bool_t getTRU() const { return mIsTRU; }

  void setLEDMon() {}
  Bool_t getLEDMon() const { return false; }

  void PrintStream(std::ostream& stream) const;

  void setNoiseLG(uint16_t noise) { mNoiseLG = noise; }
  uint16_t getNoiseLG() const { return mNoiseLG; }

  void setNoiseHG(uint16_t noise) { mNoiseHG = noise; }
  uint16_t getNoiseHG() const { return mNoiseHG; }

  void setNoiseTRU(uint16_t noise) { mNoiseHG = noise; }
  uint16_t getNoiseTRU() const { return mNoiseHG; }

 private:
  friend class boost::serialization::access;

  double mAmplitudeGeV = 0.; ///< Amplitude (GeV)
  Short_t mTower = -1;       ///< Tower index (absolute cell ID)
  bool mIsTRU = false;       ///< TRU flag
  uint16_t mNoiseLG = 0;     ///< Noise of the low gain digits
  uint16_t mNoiseHG = 0;     ///< Noise of the high gain digits or TRU digits (can never be at the same time)

  ClassDefNV(Digit, 2);
};

std::ostream& operator<<(std::ostream& stream, const Digit& dig);
} // namespace emcal
} // namespace o2
#endif
