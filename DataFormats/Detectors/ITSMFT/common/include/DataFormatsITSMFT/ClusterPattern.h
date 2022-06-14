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

/// \file ClusterTopology.h
/// \brief Definition of the ClusterTopology class.
///
/// \author Luca Barioglio, University and INFN of Torino
///
/// Short ClusterPattern descritpion
///
/// This class contains the information of the pattern of a cluster of pixels (bitmask), together with the number of
/// rows and columns of the vinding box.
///

#ifndef ALICEO2_ITSMFT_CLUSTERPATTERN_H
#define ALICEO2_ITSMFT_CLUSTERPATTERN_H
#include <Rtypes.h>
#include <array>
#include <iosfwd>
#include <gsl/gsl>
#include "DataFormatsITSMFT/CompCluster.h"

namespace o2
{
namespace itsmft
{
class ClusterTopology;
class TopologyDictionary;
class BuildTopologyDictionary;

class ClusterPattern
{
 public:
  static constexpr uint8_t MaxRowSpan = 128;
  static constexpr uint8_t MaxColSpan = 128;
  static constexpr int MaxPatternBits = MaxRowSpan * MaxColSpan;
  static constexpr int MaxPatternBytes = MaxPatternBits / 8;
  static constexpr int SpanMask = 0x7fff;
  static constexpr int TruncateMask = 0x8000;

  /// Default constructor
  ClusterPattern();
  /// Standard constructor
  ClusterPattern(int nRow, int nCol, const unsigned char patt[MaxPatternBytes]);

  template <class iterator>
  void acquirePattern(iterator& pattIt)
  {
    mBitmap[0] = *pattIt++;
    mBitmap[1] = *pattIt++;
    int nbits = mBitmap[0] * mBitmap[1];
    int nBytes = nbits / 8;
    if (((nbits) % 8) != 0) {
      nBytes++;
    }
    memcpy(&mBitmap[2], &(*pattIt), nBytes);
    pattIt += nBytes;
  }

  template <class iterator>
  static void skipPattern(iterator& pattIt)
  {
    unsigned char b0 = *pattIt++, b1 = *pattIt++;
    int nbits = b0 * b1;
    pattIt += nbits / 8 + (((nbits) % 8) != 0);
  }

  /// Constructor from cluster patterns
  template <class iterator>
  ClusterPattern(iterator& pattIt)
  {
    acquirePattern(pattIt);
  }

  /// Maximum number of bytes for the cluster puttern + 2 bytes respectively for the number of rows and columns of the bounding box
  static constexpr int kExtendedPatternBytes = MaxPatternBytes + 2;
  /// Returns the pattern
  const std::array<unsigned char, kExtendedPatternBytes>& getPattern() const { return mBitmap; }
  /// Returns a specific byte of the pattern
  unsigned char getByte(int n) const;
  /// Returns the number of rows
  int getRowSpan() const { return (int)mBitmap[0]; }
  /// Returns the number of columns
  int getColumnSpan() const { return (int)mBitmap[1]; }
  /// Returns the number of bytes used for the pattern
  int getUsedBytes() const;
  /// Returns the number of fired pixels
  int getNPixels() const;
  /// Prints the pattern
  friend std::ostream& operator<<(std::ostream& os, const ClusterPattern& top);
  /// Sets the pattern
  void setPattern(int nRow, int nCol, const unsigned char patt[MaxPatternBytes]);
  /// Sets the whole bitmask: the number of rows, the number of columns and the pattern
  void setPattern(const unsigned char bitmask[kExtendedPatternBytes]);
  /// Static: Compute pattern's COG position. Returns the number of fired pixels
  static int getCOG(int rowSpan, int colSpan, const unsigned char patt[MaxPatternBytes], float& xCOG, float& zCOG);
  /// Compute pattern's COG position. Returns the number of fired pixels
  int getCOG(float& xCOG, float& zCOG) const { return ClusterPattern::getCOG(getRowSpan(), getColumnSpan(), mBitmap.data() + 2, xCOG, zCOG); }

  bool isSet(int row, int col) const
  {
    const auto bmap = mBitmap.data() + 2;
    int pos = row * getColumnSpan() + col;
    return pos < getColumnSpan() * getRowSpan() && (bmap[pos >> 3] & (0x1 << (7 - (pos % 8))));
  }

  void resetPixel(int row, int col)
  {
    const auto bmap = mBitmap.data() + 2;
    int pos = row * getColumnSpan() + col;
    if (pos < getColumnSpan() * getRowSpan()) {
      bmap[pos >> 3] &= 0xff & ~(0x1 << (7 - (pos % 8)));
    }
  }

  void setPixel(int row, int col)
  {
    const auto bmap = mBitmap.data() + 2;
    int pos = row * getColumnSpan() + col;
    if (pos < getColumnSpan() * getRowSpan()) {
      bmap[pos >> 3] |= 0x1 << (7 - (pos % 8));
    }
  }

  template <typename Processor>
  void process(Processor procRowCol);

  friend ClusterTopology;
  friend TopologyDictionary;
  friend BuildTopologyDictionary;

 private:
  /// Pattern:
  ///
  /// - 1st byte: number of rows
  /// - 2nd byte: number of columns
  /// - remainig bytes : pixels of the cluster, where 1 is a fired pixel and 0
  /// is a non-fired pixel. The number of bytes used for the pixels depends on
  /// the size of the bounding box

  std::array<unsigned char, kExtendedPatternBytes> mBitmap; ///< Cluster pattern: 1 is a fired pixel and 0 is a non-fired pixel

  ClassDefNV(ClusterPattern, 1);
};

template <typename Processor>
void ClusterPattern::process(Processor procRowCol)
{
  auto cspan = getColumnSpan(), rspan = getRowSpan();
  uint32_t nBits = cspan * rspan;
  uint32_t nBytes = (nBits >> 3) + (nBits % 8 != 0);
  const auto bmap = mBitmap.data() + 2;
  uint16_t ic = 0, ir = 0;
  for (unsigned int i = 0; i < nBytes; i++) {
    int s = 128; // 0b10000000
    while (s > 0) {
      if ((bmap[i] & s) != 0) {
        procRowCol(ir, ic);
      }
      ic++;
      s >>= 1;
      if (uint32_t(ir + 1) * ic == nBits) {
        break;
      }
      if (ic == cspan) {
        ic = 0;
        ir++;
      }
    }
    if (uint32_t(ir + 1) * ic == nBits) {
      break;
    }
  }
}

} // namespace itsmft
} // namespace o2
#endif /* ALICEO2_ITS_CLUSTERPATTERN_H */
