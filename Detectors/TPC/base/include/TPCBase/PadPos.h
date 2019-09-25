// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// @file   PadPos.h
/// @author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de
///

/// @brief  Pad position (pad an row)
///
/// This class represents a pad position (pad and row)
/// withing e.g. a sector, ROC or CRU
///
/// origin: TPC
/// @author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de

#ifndef AliceO2_TPC_PadPos_H
#define AliceO2_TPC_PadPos_H

namespace o2
{
namespace tpc
{
class PadPos
{
 public:
  /// default constructor
  PadPos() = default;

  /// constructor
  /// @param [in] row pad row
  /// @param [in] pad pad in a row
  PadPos(const unsigned char row, const unsigned char pad) : mRow(row), mPad(pad) {}

  /// numeric row number
  /// @return numeric row number
  unsigned char getRow() const { return mRow; }

  /// numeric pad number
  /// @return numeric pad number
  unsigned char getPad() const { return mPad; }

  /// setter for row number
  /// @param [in] row row number
  void setRow(const unsigned char row) { mRow = row; }

  /// setter for pad number
  /// @param [in] pad pad number
  void setPad(const unsigned char pad) { mPad = pad; }

  /// add row offset
  /// @param [in] rowOffset row offset to add
  void addRowOffset(const unsigned char rowOffset) { mRow += rowOffset; }

  /// setter for row and pad number
  /// @param [in] row row number
  /// @param [in] pad pad number
  void set(const unsigned char row, const unsigned char pad)
  {
    mRow = row;
    mPad = pad;
  }

  /// check if is valid
  /// @return pad valid
  bool isValid() const { return !(mRow == 255 && mPad == 255); }

  /// equal operator
  bool operator==(const PadPos& other) const { return (mRow == other.mRow) && (mPad == other.mPad); }

  /// unequal operator
  bool operator!=(const PadPos& other) const { return (mRow != other.mRow) || (mPad != other.mPad); }

  /// smaller operator
  bool operator<(const PadPos& other) const
  {
    if (mRow < other.mRow) {
      return true;
    }
    if (mRow == other.mRow && mPad < other.mPad) {
      return true;
    }
    return false;
  }

 private:
  unsigned char mRow{0}; ///< row number
  unsigned char mPad{0}; ///< pad number in row
};
} // namespace tpc
} // namespace o2

#endif
