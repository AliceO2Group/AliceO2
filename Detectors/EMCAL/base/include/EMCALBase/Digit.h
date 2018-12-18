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
#include "EMCALBase/Constants.h"

#include <boost/serialization/base_object.hpp> // for base_object

namespace o2
{

namespace EMCAL
{
/// \class Digit
/// \brief EMCAL digit implementation

using DigitBase = o2::dataformats::TimeStamp<double>;
class Digit : public DigitBase
{
 public:
  Digit() = default;

  Digit(Short_t tower, Double_t amplitude, Double_t time, Int_t label = -1);
  ~Digit() = default; // override

  bool operator<(const Digit& other) const { return getTimeStamp() < other.getTimeStamp(); }
  bool operator>(const Digit& other) const { return getTimeStamp() > other.getTimeStamp(); }
  bool operator==(const Digit& other) const { return getTimeStamp() == other.getTimeStamp(); }

  bool canAdd(const Digit other)
  {
    return (mTower == other.GetTower() && std::abs(getTimeStamp() - other.getTimeStamp()) <= constants::EMCAL_TIMESAMPLE);
  }

  Digit& operator+=(const Digit& other);              // Adds energy of other digits to this digit.
  friend Digit operator+(Digit lhs, const Digit& rhs) // Adds energy of two digits.
  {
    lhs += rhs;
    return lhs;
  }

  Short_t GetTower() const { return mTower; }
  void SetTower(Short_t tower) { mTower = tower; }

  Double_t GetAmplitude() const { return mAmplitude; }
  void SetAmplitude(Double_t amplitude) { mAmplitude = amplitude; }

  Int_t GetLabel() const { return mLabel; }
  void SetLabel(Int_t label) { mLabel = label; }

  void PrintStream(std::ostream& stream) const;

 private:
  friend class boost::serialization::access;

  Double_t mAmplitude; ///< Amplitude
  Int_t mLabel;        ///< Index of the corresponding entry/entries in the MC label array
  Short_t mTower;      ///< Tower index (absolute cell ID)

  ClassDefNV(Digit, 1);
};

std::ostream& operator<<(std::ostream& stream, const Digit& dig);
} // namespace EMCAL
} // namespace o2
#endif
