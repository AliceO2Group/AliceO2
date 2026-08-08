// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_DATAFORMATSTRKFT3_DIGIT_H
#define ALICEO2_DATAFORMATSTRKFT3_DIGIT_H

#include "Rtypes.h"
#include <climits>
#include <iosfwd>

namespace o2::trkft3
{

class Digit
{
 public:
  Digit(UShort_t chipindex = 0, UShort_t row = 0, UShort_t col = 0, Int_t charge = 0);
  ~Digit() = default;

  UShort_t getChipIndex() const { return mChipIndex; }
  UShort_t getColumn() const { return mCol; }
  UShort_t getRow() const { return mRow; }
  Int_t getCharge() const { return mCharge; }

  void setChipIndex(UShort_t index) { mChipIndex = index; }
  void setPixelIndex(UShort_t row, UShort_t col)
  {
    mRow = row;
    mCol = col;
  }
  void setCharge(Int_t charge) { mCharge = charge < USHRT_MAX ? charge : USHRT_MAX; }
  void addCharge(int charge) { setCharge(charge + int(mCharge)); }

  std::ostream& print(std::ostream& output) const;
  friend std::ostream& operator<<(std::ostream& output, const Digit& digi)
  {
    digi.print(output);
    return output;
  }

 private:
  UShort_t mChipIndex = 0;
  UShort_t mRow = 0;
  UShort_t mCol = 0;
  UShort_t mCharge = 0;

  ClassDefNV(Digit, 1);
};

} // namespace o2::trkft3

#endif
