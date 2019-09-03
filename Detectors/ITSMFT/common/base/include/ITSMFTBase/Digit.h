// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Digit.h
/// \brief Definition of the ITSMFT digit
#ifndef ALICEO2_ITSMFT_DIGIT_H
#define ALICEO2_ITSMFT_DIGIT_H

#include "Rtypes.h" // for Double_t, ULong_t, etc
#include <climits>

namespace o2
{

namespace itsmft
{
/// \class Digit
/// \brief Digit class for the ITS
///

class Digit
{

 public:
  /// Constructor, initializing values for position, charge and readout frame
  Digit(UShort_t chipindex = 0, UInt_t frame = 0, UShort_t row = 0, UShort_t col = 0, Int_t charge = 0);

  /// Destructor
  ~Digit() = default;

  /// Get the index of the chip
  UShort_t getChipIndex() const { return mChipIndex; }

  /// Get the column of the pixel within the chip
  UShort_t getColumn() const { return mCol; }

  /// Get the row of the pixel within the chip
  UShort_t getRow() const { return mRow; }

  /// Get the accumulated charged of the digit
  Int_t getCharge() const { return mCharge; }

  /// Set the index of the chip
  void setChipIndex(UShort_t index) { mChipIndex = index; }

  /// Set the index of the pixel within the chip
  void setPixelIndex(UShort_t row, UShort_t col)
  {
    mRow = row;
    mCol = col;
  }

  /// Set the charge of the digit
  void setCharge(Int_t charge) { mCharge = charge < USHRT_MAX ? charge : USHRT_MAX; }

  /// Add charge to the digit, registering the label if provided
  void addCharge(int charge) { setCharge(charge + int(mCharge)); }

  /// Get the RO frame ID
  UInt_t getROFrame() const { return mROFrame; }

  /// Set RO frame ID
  void setROFrame(UInt_t v) { mROFrame = v; }

  /// Print function: Print basic digit information on the  output stream
  std::ostream& print(std::ostream& output) const;

  /// Streaming operator for the digit class
  /// Using streaming functionality defined in function Print
  friend std::ostream& operator<<(std::ostream& output, const Digit& digi)
  {
    digi.print(output);
    return output;
  }

 private:
  UShort_t mChipIndex = 0; ///< Chip index
  UShort_t mRow = 0;       ///< Pixel index in X
  UShort_t mCol = 0;       ///< Pixel index in Z
  UShort_t mCharge = 0.f;  ///< Accumulated N electrons
  UInt_t mROFrame = 0;     ///< readout frame ID

  ClassDefNV(Digit, 1);
};
} // namespace itsmft
} // namespace o2

#endif /* ALICEO2_ITSMFT_DIGIT_H */
