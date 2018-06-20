// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file CompCluster.h
/// \brief Definition of the ITSMFT compact cluster
#ifndef ALICEO2_ITSMFT_COMPCLUSTER_H
#define ALICEO2_ITSMFT_COMPCLUSTER_H

#include <Rtypes.h>

namespace o2
{
namespace ITSMFT
{

/// This is a version of the ALPIDE cluster represented by the pattern ID and the address of the
/// top-left (min row,col) pixel of the topololy bounding box

class CompCluster
{
 private:
  static constexpr int NBitsRow = 9;     // number of bits for the row
  static constexpr int NBitsCol = 10;    // number of bits for the column
  static constexpr int NBitsPattID = 11; // number of bits for the pattern ID
  static constexpr UInt_t RowMask = (0x1 << NBitsRow) - 1;
  static constexpr UInt_t ColMask = (0x1 << NBitsCol) - 1;
  static constexpr UInt_t PattIDMask = (0x1 << NBitsPattID) - 1;
  static constexpr UInt_t FlagBit = 0x1 << (NBitsRow + NBitsCol + NBitsPattID);
  //
  ///< compactified data: bits [0:8] - row, [9-18] - col, [19-30] - pattern ID, bit 31 - special flag
  UInt_t mData;

  void sanityCheck();

 public:
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

  ClassDefNV(CompCluster, 1)
};

/// Extension of the compact cluster, augmented by the chipID and ROFrame
/// This is a TEMPORARY class, until we converge to more economical container
class CompClusterExt : public CompCluster
{
 private:
  UShort_t mChipID;  ///< chip id
  UShort_t mROFrame; ///< readout frame

 public:
  CompClusterExt(UShort_t row = 0, UShort_t col = 0, UShort_t patt = 0, UShort_t chipID = 0, UShort_t rof = 0) : CompCluster(row, col, patt), mChipID(chipID), mROFrame(rof)
  {
  }

  void set(UShort_t row, UShort_t col, UShort_t patt, UShort_t chipID, UShort_t rof)
  {
    CompCluster::set(row, col, patt);
    mChipID = chipID;
    mROFrame = rof;
  }

  UShort_t getChipID() const { return mChipID; }
  UShort_t getROFrame() const { return mROFrame; }

  void setChipID(UShort_t c) { mChipID = c; }
  void setROFrame(UShort_t f) { mROFrame = f; }

  void print() const;

  ClassDefNV(CompClusterExt, 1)
};

} // namespace ITSMFT
} // namespace o2

std::ostream& operator<<(std::ostream& stream, const o2::ITSMFT::CompCluster& cl);
std::ostream& operator<<(std::ostream& stream, const o2::ITSMFT::CompClusterExt& cl);

#endif /* ALICEO2_ITSMFT_COMPACTCLUSTER_H */
