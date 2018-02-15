// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file ClusterTopology.h
/// \brief Definition of the ClusterTopology class.
///
/// \author Luca Barioglio, University and INFN of Torino
///
/// Short ClusterTopology descritpion
///
/// This class is used to store the information concerning the shape of a cluster (topology),
/// such as the size of the bounding box of the cluster (number of rows/columns) and the
/// bit-mask corresponding to the pixels of the cluster.
///

#ifndef ALICEO2_ITSMFT_CLUSTERTOPOLOGY_H
#define ALICEO2_ITSMFT_CLUSTERTOPOLOGY_H
#include <Rtypes.h>
#include <array>
#include <iostream>
#include <string>
#include "DataFormatsITSMFT/Cluster.h"

namespace o2
{
namespace ITSMFT
{
class ClusterTopology
{
 public:
  /// Default constructor
  ClusterTopology();
  /// Standard constructor
  ClusterTopology(int nRow, int nCol, const unsigned char patt[Cluster::kMaxPatternBytes]);

  /// Returns the pattern
  const std::array<unsigned char, Cluster::kMaxPatternBytes + 2>& getPattern() const { return mPattern; }
  /// Returns the number of rows
  int getRowSpan() const { return (int)mPattern[0]; }
  /// Returns the number of columns
  int getColumnSpan() const { return (int)mPattern[1]; }
  /// Returns the number of used bytes
  int getUsedBytes() const { return mNbytes; }
  /// Returns the hashcode
  unsigned long getHash() const { return mHash; }
  /// Prints the topology
  friend std::ostream& operator<<(std::ostream& os, const ClusterTopology& top);
  /// MurMur2 hash fucntion
  static unsigned int hashFunction(const void* key, int len);
  /// Compute the complete hash as defined for mHash
  static unsigned long getCompleteHash(int nRow, int nCol, const unsigned char patt[Cluster::kMaxPatternBytes],
                                       int nBytesUsed);
  static unsigned long getCompleteHash(const ClusterTopology& topology);
  /// Sets the pattern
  void setPattern(int nRow, int nCol, const unsigned char patt[Cluster::kMaxPatternBytes]);

 private:
  /// Pattern:
  ///
  /// - 1st byte: number of rows
  /// - 2nd byte: number of columns
  /// - remainig bytes : pixels of the cluster, where 1 is a fired pixel and 0
  /// is a non-fired pixel. The number of pixels used for the pixel depends on
  /// the size of the bounding box
  std::array<unsigned char, Cluster::kMaxPatternBytes + 2>
    mPattern;  ///< Cluster pattern: 1 is a fired pixel and 0 is a non-fired pixel
  int mNbytes; ///< Number of bytes that are effectively used in mPattern
  /// Hashcode computed from the pattern
  ///
  /// The first four bytes are computed with MurMur2 hash-function. The remaining
  /// four bytes are the first 32 pixels of the pattern. If the number of pixles
  /// is less than 32, the remaining bits are set to 0.
  unsigned long mHash;

  ClassDefNV(ClusterTopology, 1);
};
} // namespace ITSMFT
} // namespace o2

#endif /* ALICEO2_ITS_CLUSTERTOPOLOGY_H */
