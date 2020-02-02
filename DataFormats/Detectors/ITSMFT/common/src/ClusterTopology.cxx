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
/// \brief Implementation of the ClusterTopology class.
///
/// \author Luca Barioglio, University and INFN of Torino

#include "DataFormatsITSMFT/ClusterTopology.h"
#include "Framework/Logger.h"
#include <iostream>
#include <TMath.h>

ClassImp(o2::itsmft::ClusterTopology);

namespace o2
{
namespace itsmft
{
ClusterTopology::ClusterTopology() : mPattern{}, mHash{0} {}

ClusterTopology::ClusterTopology(int nRow, int nCol, const unsigned char patt[Cluster::kMaxPatternBytes]) : mHash{0}
{
  setPattern(nRow, nCol, patt);
}

void ClusterTopology::setPattern(int nRow, int nCol, const unsigned char patt[Cluster::kMaxPatternBytes])
{
  mPattern.setPattern(nRow, nCol, patt);
  mHash = getCompleteHash(*this);
}

unsigned int ClusterTopology::hashFunction(const void* key, int len)
{
  //
  // Developed from https://github.com/rurban/smhasher , function MurMur2
  //
  // 'm' and 'r' are mixing constants generated offline.
  const unsigned int m = 0x5bd1e995;
  const int r = 24;
  // Initialize the hash
  unsigned int h = len ^ 0xdeadbeef;
  // Mix 4 bytes at a time into the hash
  const unsigned char* data = (const unsigned char*)key;
  // int recIndex=0;
  while (len >= 4) {
    unsigned int k = *(unsigned int*)data;
    k *= m;
    k ^= k >> r;
    k *= m;
    h *= m;
    h ^= k;
    data += 4;
    len -= 4;
  }
  // Handle the last few bytes of the input array
  switch (len) {
    case 3:
      h ^= data[2] << 16;
    case 2:
      h ^= data[1] << 8;
    case 1:
      h ^= data[0];
      h *= m;
  };
  // Do a few final mixes of the hash to ensure the last few
  // bytes are well-incorporated.
  h ^= h >> 13;
  h *= m;
  h ^= h >> 15;
  return h;
}

unsigned long ClusterTopology::getCompleteHash(int nRow, int nCol,
                                               const unsigned char patt[Cluster::kMaxPatternBytes])
{
  unsigned char extended_pattern[ClusterPattern::kExtendedPatternBytes] = {0};
  extended_pattern[0] = (unsigned char)nRow;
  extended_pattern[1] = (unsigned char)nCol;
  int nBits = nRow * nCol;
  int nBytes = nBits / 8;
  if (nBits % 8 != 0)
    nBytes++;
  memcpy(&extended_pattern[2], patt, nBytes);

  unsigned long partialHash = (unsigned long)hashFunction(extended_pattern, nBytes);
  // The first four bytes are directly taken from partialHash
  unsigned long completeHash = partialHash << 32;
  // The last four bytes of the hash are the first 32 pixels of the topology.
  // The bits reserved for the pattern that are not used are set to 0.
  if (nBytes == 1) {
    completeHash += ((((unsigned long)extended_pattern[2]) << 24));
  } else if (nBytes == 2) {
    completeHash += ((((unsigned long)extended_pattern[2]) << 24) + (((unsigned long)extended_pattern[3]) << 16));
  } else if (nBytes == 3) {
    completeHash += ((((unsigned long)extended_pattern[2]) << 24) + (((unsigned long)extended_pattern[3]) << 16) +
                     (((unsigned long)extended_pattern[4]) << 8));
  } else if (nBytes >= 4) {
    completeHash += ((((unsigned long)extended_pattern[2]) << 24) + (((unsigned long)extended_pattern[3]) << 16) +
                     (((unsigned long)extended_pattern[4]) << 8) + ((unsigned long)extended_pattern[5]));
  } else {
    LOG(ERROR) << "No fired pixels in small topology";
    throw std::runtime_error("No fired pixels in small topology");
  }
  return completeHash;
}

unsigned long ClusterTopology::getCompleteHash(const ClusterTopology& topology)
{
  auto patt = topology.getPattern();
  int nBytesUsed = topology.getUsedBytes();
  unsigned long partialHash = (unsigned long)hashFunction(patt.data(), nBytesUsed);
  // The first four bytes are directly taken from partialHash
  unsigned long completeHash = partialHash << 32;
  // The last four bytes of the hash are the first 32 pixels of the topology.
  // The bits reserved for the pattern that are not used are set to 0.
  if (nBytesUsed == 1) {
    completeHash += ((((unsigned long)patt[2]) << 24));
  } else if (nBytesUsed == 2) {
    completeHash += ((((unsigned long)patt[2]) << 24) + (((unsigned long)patt[3]) << 16));
  } else if (nBytesUsed == 3) {
    completeHash +=
      ((((unsigned long)patt[2]) << 24) + (((unsigned long)patt[3]) << 16) + (((unsigned long)patt[4]) << 8));
  } else if (nBytesUsed >= 4) {
    completeHash += ((((unsigned long)patt[2]) << 24) + (((unsigned long)patt[3]) << 16) +
                     (((unsigned long)patt[4]) << 8) + ((unsigned long)patt[5]));
  } else {
    LOG(ERROR) << "No fired pixels in small topology";
    throw std::runtime_error("No fired pixels in small topology");
  }
  return completeHash;
}

std::ostream& operator<<(std::ostream& os, const ClusterTopology& topology)
{
  os << topology.mPattern << std::endl;
  return os;
}

void ClusterTopology::getCOGshift(int nRows, int nCols, const unsigned char patt[Cluster::kMaxPatternBytes], int& rowShift, int& colShift)
{
  int tempxCOG = 0, tempzCOG = 0, tempFiredPixels = 0, s = 0, ic = 0, ir = 0;
  int nBits = nRows * nCols;
  int nBytes = nBits / 8;
  if (nBits % 8 != 0) {
    nBytes++;
  }
  for (unsigned int i = 0; i < nBytes; i++) {
    unsigned char tempChar = patt[i];
    s = 128; // 0b10000000
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
      if (ic == nCols) {
        ic = 0;
        ir++;
      }
    }
    if ((ir + 1) * ic == nBits) {
      break;
    }
  }
  rowShift = TMath::Nint(float(tempxCOG) / tempFiredPixels);
  colShift = TMath::Nint(float(tempzCOG) / tempFiredPixels);
}

} // namespace itsmft
} // namespace o2
