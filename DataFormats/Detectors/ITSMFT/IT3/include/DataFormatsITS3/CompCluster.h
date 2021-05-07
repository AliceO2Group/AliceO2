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
/// \brief Definition of the ITS3 compact cluster
#ifndef ALICEO2_ITS3_COMPCLUSTER_H
#define ALICEO2_ITS3_COMPCLUSTER_H

#include <Rtypes.h>

namespace o2
{
namespace its3
{

/// This is a version of the ALPIDE cluster represented by the pattern ID and the address of the
/// top-left (min row,col) pixel of the topololy bounding box

class CompCluster
{
 public:
  static constexpr int NBitsPattID = 11; // number of bits for the pattern ID
 private:
  static constexpr UShort_t PattIDMask = (0x1 << NBitsPattID) - 1;
  static constexpr UShort_t FlagBit = 0x1 << (NBitsPattID);
  //
  ///< compactified data: bits [0:8] - row, [9-18] - col, [19-30] - pattern ID, bit 31 - special flag
  UShort_t mRow;
  UShort_t mCol;
  UShort_t mPat;

 public:
  static constexpr unsigned short InvalidPatternID = (0x1 << NBitsPattID) - 1; // All 11 bits of pattern ID are 1
  CompCluster(UShort_t row = 0, UShort_t col = 0, UShort_t patt = 0)
  {
    set(row, col, patt);
  }

  void set(UShort_t row, UShort_t col, UShort_t patt)
  {
    setPatternID(patt);
    mRow = row;
    mCol = col;
  }

  UShort_t getRow() const { return mRow; }
  UShort_t getCol() const { return mCol; }
  UShort_t getPatternID() const { return (mPat & PattIDMask); }
  bool getFlag() const { return (mPat & FlagBit) == FlagBit; }

  void setRow(UShort_t r)
  {
    mRow = r;
  }
  void setCol(UShort_t c)
  {
    mCol = c;
  };
  void setPatternID(UShort_t p)
  {
    mPat &= ~(PattIDMask);
    mPat |= p;
  }
  void setFlag(bool v)
  {
    mPat &= ~FlagBit;
    if (v) {
      mPat |= FlagBit;
    }
  }

  void print() const;

  ClassDefNV(CompCluster, 1);
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

} // namespace its3
} // namespace o2

std::ostream& operator<<(std::ostream& stream, const o2::its3::CompCluster& cl);
std::ostream& operator<<(std::ostream& stream, const o2::its3::CompClusterExt& cl);

#endif /* ALICEO2_ITS3_COMPACTCLUSTER_H */
