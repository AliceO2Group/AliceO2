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

/// \file CompCluster.h
/// \brief Definition of the ITSMFT compact cluster
#ifndef ALICEO2_ITSMFT_COMPCLUSTER_H
#define ALICEO2_ITSMFT_COMPCLUSTER_H

#include <Rtypes.h>
#include <iosfwd>

namespace o2
{
namespace itsmft
{

/// This is a version of the ALPIDE cluster represented by the pattern ID and the address of the
/// top-left (min row,col) pixel of the topololy bounding box

class CompCluster
{
 public:
  static constexpr int NBitsRow = 9;     // number of bits for the row
  static constexpr int NBitsCol = 10;    // number of bits for the column
  static constexpr int NBitsPattID = 11; // number of bits for the pattern ID
 private:
  static constexpr UInt_t RowMask = (0x1 << NBitsRow) - 1;
  static constexpr UInt_t ColMask = (0x1 << NBitsCol) - 1;
  static constexpr UInt_t PattIDMask = (0x1 << NBitsPattID) - 1;
  static constexpr UInt_t FlagBit = 0x1 << (NBitsRow + NBitsCol + NBitsPattID);
  //
  ///< compactified data: bits [0:8] - row, [9-18] - col, [19-30] - pattern ID, bit 31 - special flag
  UInt_t mData;

  void sanityCheck();

 public:
  static constexpr unsigned short InvalidPatternID = (0x1 << NBitsPattID) - 1; // All 11 bits of pattern ID are 1
  CompCluster(UShort_t row = 0, UShort_t col = 0, UShort_t patt = 0)
  {
    set(row, col, patt);
  }

  void set(UShort_t row, UShort_t col, UShort_t patt)
  {
    mData = (row & RowMask) | ((col & ColMask) << NBitsRow) | ((patt & PattIDMask) << (NBitsRow + NBitsCol));
  }

  UShort_t getRow() const { return mData & RowMask; }
  UShort_t getCol() const { return (mData >> NBitsRow) & ColMask; }
  UShort_t getPatternID() const { return (mData >> (NBitsRow + NBitsCol)) & PattIDMask; }
  bool getFlag() const { return (mData & FlagBit) == FlagBit; }

  void setRow(UShort_t r)
  {
    mData &= ~RowMask;
    mData |= (r & RowMask);
  }
  void setCol(UShort_t c)
  {
    mData &= ~(ColMask << NBitsRow);
    mData |= (c & ColMask) << NBitsRow;
  };
  void setPatternID(UShort_t p)
  {
    mData &= ~(PattIDMask << (NBitsRow + NBitsCol));
    mData |= (p & PattIDMask) << (NBitsRow + NBitsCol);
  }
  void setFlag(bool v)
  {
    mData &= ~FlagBit;
    if (v) {
      mData |= FlagBit;
    }
  }

  void print() const;

  ClassDefNV(CompCluster, 2);
};

/// Extension of the compact cluster, augmented by the chipID
/// This is a TEMPORARY class, until we converge to more economical container
class CompClusterExt : public CompCluster
{
 private:
  UShort_t mChipID;  ///< chip id

 public:
  CompClusterExt(UShort_t row = 0, UShort_t col = 0, UShort_t patt = 0, UShort_t chipID = 0) : CompCluster(row, col, patt), mChipID(chipID)
  {
  }

  void set(UShort_t row, UShort_t col, UShort_t patt, UShort_t chipID)
  {
    CompCluster::set(row, col, patt);
    mChipID = chipID;
  }

  UShort_t getChipID() const { return mChipID; }
  UShort_t getSensorID() const { return mChipID; } // to have the same signature as BaseCluster

  void setChipID(UShort_t c) { mChipID = c; }

  void print() const;

  ClassDefNV(CompClusterExt, 1);
};

} // namespace itsmft
} // namespace o2

std::ostream& operator<<(std::ostream& stream, const o2::itsmft::CompCluster& cl);
std::ostream& operator<<(std::ostream& stream, const o2::itsmft::CompClusterExt& cl);

#endif /* ALICEO2_ITSMFT_COMPACTCLUSTER_H */
