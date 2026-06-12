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

///
/// \file Digit.h
/// \brief Definition of IOTOF digit class
/// \author Nicolò Jacazio, Università del Piemonte Orientale (IT)
/// \since 2026-03-17
///

#ifndef ALICEO2_IOTOF_DIGIT_H
#define ALICEO2_IOTOF_DIGIT_H

#include "SimulationDataFormat/MCCompLabel.h"
#include "DataFormatsITSMFT/Digit.h"

namespace o2::iotof
{
class Digit : public o2::itsmft::Digit
{
 public:
  ~Digit() = default;
  Digit(UShort_t chipindex = 0, UShort_t row = 0, UShort_t col = 0, Int_t charge = 0, double time = 0.)
    : o2::itsmft::Digit(chipindex, row, col, charge), mTime(time) {};

  // Setters
  void setTime(double time) { mTime = time; }

  // Getters
  double getTime() const { return mTime; }

  static UInt_t getOrderingKey(UShort_t chipindex, UShort_t row, UShort_t col)
  {
    return (static_cast<UInt_t>(chipindex) << 16) | (static_cast<UInt_t>(row) << 8) | static_cast<UInt_t>(col);
  }

 private:
  double mTime = 0.; ///< Measured time (ns)
  ClassDefNV(Digit, 1);
};

// McLabelRef is used to store the MC label of the hit contributing to a digit, and eventually link to extra contributions to the same pixel
struct McLabelRef {
  o2::MCCompLabel mLabel; ///< hit label
  int mNext = -1;         ///< eventual next contribution to the same pixel
  McLabelRef(o2::MCCompLabel label = 0, int next = -1) : mLabel(label), mNext(next) {}

  ClassDefNV(McLabelRef, 1);
};

class LabeledDigit : public Digit
{
 public:
  LabeledDigit(UShort_t chipindex = 0, UShort_t row = 0, UShort_t col = 0, Int_t charge = 0, double time = 0.,
               o2::MCCompLabel label = 0)
    : Digit(chipindex, row, col, charge, time), mLabel(label) {}

  void setLabel(McLabelRef label) { mLabel = label; }
  McLabelRef getLabel() const { return mLabel; }

 private:
  McLabelRef mLabel; ///< label of the hit contributing to the digit, and eventually reference to extra contributions to the same pixel
  ClassDefNV(LabeledDigit, 1);
};

} // namespace o2::iotof
#endif // ALICEO2_IOTOF_DIGIT_H
