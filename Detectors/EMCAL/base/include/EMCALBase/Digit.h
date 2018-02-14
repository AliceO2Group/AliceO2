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

  Digit(Int_t tower, Double_t amplitude, Double_t time);
  ~Digit() = default; // override

  bool operator<(const Digit& other) const { return getTimeStamp() < other.getTimeStamp(); }
  bool operator>(const Digit& other) const { return getTimeStamp() > other.getTimeStamp(); }

  bool canAdd(const Digit other)
  {
    return (mTower == other.GetTower() && fabs(getTimeStamp() - other.getTimeStamp()) <= 25);
  }

  Digit& operator+=(const Digit& other);              // Adds energy of other digits to this digit.
  friend Digit operator+(Digit lhs, const Digit& rhs) // Adds energy of two digits.
  {
    lhs += rhs;
    return lhs;
  }

  Int_t GetTower() const { return mTower; }
  void SetTower(Int_t tower) { mTower = tower; }

  Double_t GetAmplitude() const { return mAmplitude; }
  void SetAmplitude(Double_t amplitude) { mAmplitude = amplitude; }

  void PrintStream(std::ostream& stream) const;

 private:
  friend class boost::serialization::access;

  Int_t mTower;        ///< Tower index (absolute cell ID)
  Double_t mAmplitude; ///< Amplitude

  ClassDefNV(Digit, 1);
};

std::ostream& operator<<(std::ostream& stream, const Digit& dig);
} // namespace EMCAL
} // namespace o2
#endif
