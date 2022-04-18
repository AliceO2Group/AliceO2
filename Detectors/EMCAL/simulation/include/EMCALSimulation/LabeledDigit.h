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

#ifndef ALICEO2_EMCAL_LABELEDDIGIT_H_
#define ALICEO2_EMCAL_LABELEDDIGIT_H_

#include <iosfwd>
#include <cmath>
#include "Rtypes.h"
#include "CommonDataFormat/TimeStamp.h"
#include "DataFormatsEMCAL/Constants.h"
#include "DataFormatsEMCAL/Digit.h"
#include "DataFormatsEMCAL/MCLabel.h"

#include <boost/serialization/base_object.hpp> // for base_object

namespace o2
{

namespace emcal
{
/// \class LabeledDigit
/// \brief EMCAL labeled digit implementation
/// \ingroup EMCALsimulation

class LabeledDigit
{
 public:
  LabeledDigit() = default;

  LabeledDigit(Digit digit, o2::emcal::MCLabel label);
  LabeledDigit(Short_t tower, Double_t amplitudeGeV, Double_t time, o2::emcal::MCLabel label);
  LabeledDigit(Short_t tower, uint16_t noiseLG, uint16_t noiseHG, Double_t time, o2::emcal::MCLabel label);

  ~LabeledDigit() = default; // override

  void setDigit(Digit d) { mDigit = d; }
  Digit getDigit() const { return mDigit; }

  void addLabel(o2::emcal::MCLabel l) { mLabels.push_back(l); }
  Int_t getNumberOfLabels() const { return mLabels.size(); }
  std::vector<o2::emcal::MCLabel> getLabels() const { return mLabels; }

  bool operator<(const LabeledDigit& other) const { return getTimeStamp() < other.getTimeStamp(); }
  bool operator>(const LabeledDigit& other) const { return getTimeStamp() > other.getTimeStamp(); }
  bool operator==(const LabeledDigit& other) const { return (getTimeStamp() == other.getTimeStamp()); }

  bool canAdd(const LabeledDigit other)
  {
    return (getTower() == other.getTower() && std::abs(getTimeStamp() - other.getTimeStamp()) < constants::EMCAL_TIMESAMPLE);
  }

  LabeledDigit& operator+=(const LabeledDigit& other);                     // Adds energy of other digit to this digit, combines lists of labels
  friend LabeledDigit operator+(LabeledDigit lhs, const LabeledDigit& rhs) // Adds energy of two digits, combines lists of labels
  {
    lhs += rhs;
    return lhs;
  }

  void setTimeStamp(Double_t time) { mDigit.setTimeStamp(time); }
  Double_t getTimeStamp() const { return mDigit.getTimeStamp(); }

  void setTower(Short_t tower) { mDigit.setTower(tower); }
  Short_t getTower() const { return mDigit.getTower(); }

  void setAmplitude(Double_t amplitude) { mDigit.setAmplitude(amplitude); } // GeV
  Double_t getAmplitude() const { return mDigit.getAmplitude(); }

  void setEnergy(Double_t energy) { setAmplitude(energy); }
  Double_t getEnergy() { return getAmplitude(); }

  void setAmplitudeADC(Short_t amplitude, ChannelType_t ctype = ChannelType_t::HIGH_GAIN) { mDigit.setAmplitudeADC(amplitude, ctype); }
  Int_t getAmplitudeADC(ChannelType_t ctype = ChannelType_t::HIGH_GAIN) const { return mDigit.getAmplitudeADC(ctype); }

  void setType(ChannelType_t ctype) { mDigit.setType(ctype); }
  ChannelType_t getType() const { return mDigit.getType(); }

  void setHighGain() { mDigit.setHighGain(); }
  Bool_t getHighGain() const { return mDigit.getHighGain(); }

  void setLowGain() { mDigit.setLowGain(); }
  Bool_t getLowGain() const { return mDigit.getLowGain(); }

  void setTRU() { mDigit.setTRU(); }
  Bool_t getTRU() const { return mDigit.getTRU(); }

  void setLEDMon() { mDigit.setLEDMon(); }
  Bool_t getLEDMon() const { return mDigit.getLEDMon(); }

  void PrintStream(std::ostream& stream) const;

 private:
  friend class boost::serialization::access;

  Digit mDigit;                            ///< Digit
  std::vector<o2::emcal::MCLabel> mLabels; ///< Labels

  ClassDefNV(LabeledDigit, 1);
};

std::ostream& operator<<(std::ostream& stream, const LabeledDigit& dig);
} // namespace emcal
} // namespace o2
#endif
