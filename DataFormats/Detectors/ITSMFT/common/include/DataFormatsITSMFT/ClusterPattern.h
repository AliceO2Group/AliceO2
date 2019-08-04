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
#include "DataFormatsITSMFT/Cluster.h"

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
  /// Default constructor
  ClusterPattern();
  /// Standard constructor
  ClusterPattern(int nRow, int nCol, const unsigned char patt[Cluster::kMaxPatternBytes]);
  /// Maximum number of bytes for the cluster puttern + 2 bytes respectively for the number of rows and columns of the bounding box
  static constexpr int kExtendedPatternBytes = Cluster::kMaxPatternBytes + 2;
  /// Returns the pattern
  std::array<unsigned char, kExtendedPatternBytes> getPattern() const { return mBitmap; }
  /// Returns a specific byte of the pattern
  unsigned char getByte(int n) const;
  /// Returns the number of rows
  int getRowSpan() const { return (int)mBitmap[0]; }
  /// Returns the number of columns
  int getColumnSpan() const { return (int)mBitmap[1]; }
  /// Returns the number of bytes used for the pattern
  int getUsedBytes() const;
  /// Prints the pattern
  friend std::ostream& operator<<(std::ostream& os, const ClusterPattern& top);
  /// Sets the pattern
  void setPattern(int nRow, int nCol, const unsigned char patt[Cluster::kMaxPatternBytes]);
  /// Sets the whole bitmask: the number of rows, the number of columns and the pattern
  void setPattern(const unsigned char bitmask[kExtendedPatternBytes]);

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
} // namespace itsmft
} // namespace o2
#endif /* ALICEO2_ITS_CLUSTERPATTERN_H */
