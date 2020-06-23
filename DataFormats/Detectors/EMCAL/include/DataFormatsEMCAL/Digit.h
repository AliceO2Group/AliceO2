// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

  Digit(Short_t tower, Double_t amplitudeGeV, Double_t time, ChannelType_t ctype = ChannelType_t::HIGH_GAIN);
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

  void setAmplitude(Double_t amplitude); // GeV
  Double_t getAmplitude() const { return getAmplitudeADC() * constants::EMCAL_ADCENERGY; }

  void setEnergy(Double_t energy) { setAmplitude(energy); }
  Double_t getEnergy() const { return getAmplitude(); }

  void setAmplitudeADC(Short_t amplitude, ChannelType_t ctype = ChannelType_t::HIGH_GAIN);
  Int_t getAmplitudeADC(ChannelType_t ctype = ChannelType_t::HIGH_GAIN) const;

  void setType(ChannelType_t ctype) { mChannelType = ctype; }
  ChannelType_t getType() const { return mChannelType; }

  void setHighGain() { mChannelType = ChannelType_t::HIGH_GAIN; }
  Bool_t getHighGain() const { return mChannelType == ChannelType_t::HIGH_GAIN; }

  void setLowGain() { mChannelType = ChannelType_t::LOW_GAIN; }
  Bool_t getLowGain() const { return mChannelType == ChannelType_t::LOW_GAIN; }

  void setTRU() { mChannelType = ChannelType_t::TRU; }
  Bool_t getTRU() const { return mChannelType == ChannelType_t::TRU; }

  void setLEDMon() { mChannelType = ChannelType_t::LEDMON; }
  Bool_t getLEDMon() const { return mChannelType == ChannelType_t::LEDMON; }

  void PrintStream(std::ostream& stream) const;

 private:
  friend class boost::serialization::access;

  Short_t mAmplitude = 0;                                ///< Amplitude (ADC counts)
  Short_t mTower = -1;                                   ///< Tower index (absolute cell ID)
  ChannelType_t mChannelType = ChannelType_t::HIGH_GAIN; ///< Channel type (high gain, low gain, TRU, LEDMON)

  ClassDefNV(Digit, 2);
};

std::ostream& operator<<(std::ostream& stream, const Digit& dig);
} // namespace emcal
} // namespace o2
#endif
