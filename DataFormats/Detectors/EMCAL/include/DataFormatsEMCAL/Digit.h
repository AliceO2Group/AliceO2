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
#include <bitset>
#include "Rtypes.h"
#include "CommonDataFormat/TimeStamp.h"
#include "DataFormatsEMCAL/Constants.h"
#include "DataFormatsEMCAL/TriggerRecord.h"

#include <boost/serialization/base_object.hpp> // for base_object

namespace o2
{

namespace emcal
{
using DigitBase = o2::dataformats::TimeStamp<double>;

enum Triggers {
  kMB,  // Minimum bias
  kEL0, // EMCAL L0 trigger
  kDL0, // DCAL L0 trigger
  kEG1, // EMCAL gamma L1 trigger
  kDG1, // DCAL gamma L1 trigger
  kEG2, // EMCAL gamma L1 trigger
  kDG2, // DCAL gamma L1 trigger
  kEJ1, // EMCAL jet L1 trigger
  kDJ1, // DCAL jet L1 trigger
  kEJ2, // EMCAL jet L1 trigger
  kDJ2  // DCAL jet L1 trigger
};

struct DetTrigInput {
  static constexpr char sChannelNameDPL[] = "TRIGGERINPUT";
  static constexpr char sDigitName[] = "DetTrigInput";
  static constexpr char sDigitBranchName[] = "EMCTRIGGERINPUT";
  o2::InteractionRecord mIntRecord{}; // bc/orbit of the intpur
  std::bitset<5> mInputs{};           // pattern of inputs.
  DetTrigInput() = default;
  DetTrigInput(const o2::InteractionRecord& iRec, Bool_t isMB, Bool_t isEL0, Bool_t isDL0, Bool_t isEG1, Bool_t isDG1, Bool_t isEG2, Bool_t isDG2, Bool_t isEJ1, Bool_t isDJ1, Bool_t isEJ2, Bool_t isDJ2)
    : mIntRecord(iRec),
      mInputs((isMB << o2::emcal::Triggers::kMB) |
              (isEL0 << o2::emcal::Triggers::kEL0) |
              (isDL0 << o2::emcal::Triggers::kDL0) |
              (isEG1 << o2::emcal::Triggers::kEG1) |
              (isDG1 << o2::emcal::Triggers::kDG1) |
              (isEG2 << o2::emcal::Triggers::kEG2) |
              (isDG2 << o2::emcal::Triggers::kDG2) |
              (isEJ1 << o2::emcal::Triggers::kEJ1) |
              (isDJ1 << o2::emcal::Triggers::kDJ1) |
              (isEJ2 << o2::emcal::Triggers::kEJ2) |
              (isDJ2 << o2::emcal::Triggers::kDJ2))
  {
  }
  ClassDefNV(DetTrigInput, 1);
};

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

  ClassDefNV(Digit, 3);
};

std::ostream& operator<<(std::ostream& stream, const Digit& dig);
} // namespace emcal
} // namespace o2
#endif
