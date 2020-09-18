// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file ClusterTopology.cxx
/// \brief Implementation of the ClusterPattern class.
///
/// \author Luca Barioglio, University and INFN of Torino
#include "DataFormatsITSMFT/ClusterTopology.h"
#include "Framework/Logger.h"
#include <iostream>

ClassImp(o2::itsmft::ClusterPattern);

namespace o2
{
namespace itsmft
{
ClusterPattern::ClusterPattern() : mBitmap{0} {}

ClusterPattern::ClusterPattern(int nRow, int nCol, const unsigned char patt[MaxPatternBytes])
{
  setPattern(nRow, nCol, patt);
}

unsigned char ClusterPattern::getByte(int n) const
{
  if (n < 0 || n > MaxPatternBytes + 1) {
    LOG(ERROR) << "Invalid element of the pattern";
    return -1;
  } else {
    return mBitmap[n];
  }
}

int ClusterPattern::getUsedBytes() const
{
  int nBits = (int)mBitmap[0] * (int)mBitmap[1];
  int nBytes = nBits / 8;
  if (nBits % 8 != 0)
    nBytes++;
  return nBytes;
}

void ClusterPattern::setPattern(int nRow, int nCol, const unsigned char patt[MaxPatternBytes])
{
  mBitmap[0] = (unsigned char)nRow;
  mBitmap[1] = (unsigned char)nCol;
  int nBytes = nRow * nCol / 8;
  if (((nRow * nCol) % 8) != 0)
    nBytes++;
  memcpy(&mBitmap[2], patt, nBytes);
}

void ClusterPattern::setPattern(const unsigned char patt[ClusterPattern::kExtendedPatternBytes])
{
  memcpy(&mBitmap[0], patt, ClusterPattern::kExtendedPatternBytes);
}

std::ostream& operator<<(std::ostream& os, const ClusterPattern& pattern)
{
  os << "rowSpan: " << pattern.getRowSpan() << " columnSpan: " << pattern.getColumnSpan()
     << " #bytes: " << pattern.getUsedBytes() << std::endl;
  unsigned char tempChar = 0;
  int s = 0;
  int ic = 0;
  for (unsigned int i = 2; i < pattern.getUsedBytes() + 2; i++) {
    tempChar = pattern.mBitmap[i];
    s = 128; // 0b10000000
    while (s > 0) {
      if (ic % pattern.getColumnSpan() == 0) {
        os << "|";
      }
      ic++;
      if ((tempChar & s) != 0) {
        os << '+';
      } else {
        os << ' ';
      }
      s /= 2;
      if (ic % pattern.getColumnSpan() == 0) {
        os << "|" << std::endl;
      }
      if (ic == (pattern.getRowSpan() * pattern.getColumnSpan())) {
        break;
      }
    }
    if (ic == (pattern.getRowSpan() * pattern.getColumnSpan())) {
      break;
    }
  }
  os << std::endl;
  return os;
}

int ClusterPattern::getCOG(int rowSpan, int colSpan, const unsigned char patt[MaxPatternBytes], float& xCOG, float& zCOG)
{
  int tempxCOG = 0, tempzCOG = 0, tempFiredPixels = 0, ic = 0, ir = 0;
  int nBits = rowSpan * colSpan;
  int nBytes = nBits / 8;
  if (nBits % 8 != 0) {
    nBytes++;
  }
  for (unsigned int i = 0; i < nBytes; i++) {
    unsigned char tempChar = patt[i];
    int s = 128; // 0b10000000
    while (s > 0) {
      if ((tempChar & s) != 0) {
        tempFiredPixels++;
        tempxCOG += ir;
        tempzCOG += ic;
      }
      ic++;
      s /= 2;
      if ((ir + 1) * ic == nBits) {
        break;
      }
      if (ic == colSpan) {
        ic = 0;
        ir++;
      }
    }
    if ((ir + 1) * ic == nBits) {
      break;
    }
  }
  xCOG = float(tempxCOG) / tempFiredPixels;
  zCOG = float(tempzCOG) / tempFiredPixels;

  return tempFiredPixels;
}

int ClusterPattern::getCOG(float& xCOG, float& zCOG) const
{
  auto patt = getPattern();
  return ClusterPattern::getCOG(getRowSpan(), getColumnSpan(), &patt[2], xCOG, zCOG);
}

} // namespace itsmft
} // namespace o2
