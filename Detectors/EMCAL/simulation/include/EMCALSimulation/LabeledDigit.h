// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
#include "EMCALSimulation/MCLabel.h"

#include <boost/serialization/base_object.hpp> // for base_object

namespace o2
{

namespace emcal
{
/// \class LabeledDigit
/// \brief EMCAL labeled digit implementation

class LabeledDigit
{
 public:
  LabeledDigit() = default;

  LabeledDigit(Digit digit, o2::emcal::MCLabel label);
  LabeledDigit(Short_t tower, Double_t energy, Double_t time, o2::emcal::MCLabel label);
  ~LabeledDigit() = default; // override

  void setDigit(Digit d) { mDigit = d; }
  Digit getDigit() const { return mDigit; }

  void addLabel(o2::emcal::MCLabel l) { mLabels.push_back(l); }
  Int_t getNumberOfLabels() const { return mLabels.size(); }
  std::vector<o2::emcal::MCLabel> getLabels() const { return mLabels; }

  bool operator<(const LabeledDigit& other) const { return getTimeStamp() < other.getTimeStamp(); }
  bool operator>(const LabeledDigit& other) const { return getTimeStamp() > other.getTimeStamp(); }
  bool operator==(const LabeledDigit& other) const { return getTimeStamp() == other.getTimeStamp(); }

  bool canAdd(const LabeledDigit other)
  {
    return (getTower() == other.getTower() && std::abs(getTimeStamp() - other.getTimeStamp()) <= constants::EMCAL_TIMESAMPLE);
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

  void setEnergy(Double_t energy) { mDigit.setEnergy(energy); }
  Double_t getEnergy() const { return mDigit.getEnergy(); }

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
