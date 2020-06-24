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
#include "DataFormatsITSMFT/ClusterPattern.h"
#include <map>
#include <iosfwd>

namespace o2
{
namespace itsmft
{
class ClusterTopology
{
 public:
  /// Default constructor
  ClusterTopology();
  /// Standard constructor
  ClusterTopology(int nRow, int nCol, const unsigned char patt[ClusterPattern::MaxPatternBytes]);
  /// Constructor
  ClusterTopology(const ClusterPattern& patt);

  /// Returns a specific byte of the pattern
  unsigned char getByte(int n) const { return mPattern.getByte(n); }
  /// Returns the pattern
  std::array<unsigned char, ClusterPattern::kExtendedPatternBytes> getPattern() const { return mPattern.getPattern(); }
  ClusterPattern getClusterPattern() const { return mPattern; }
  /// Returns the number of rows
  int getRowSpan() const { return mPattern.getRowSpan(); }
  /// Returns the number of columns
  int getColumnSpan() const { return mPattern.getColumnSpan(); }
  /// Returns the number of used bytes
  int getUsedBytes() const { return mPattern.getUsedBytes(); }
  /// Returns the hashcode
  unsigned long getHash() const { return mHash; }
  /// Prints the topology
  friend std::ostream& operator<<(std::ostream& os, const ClusterTopology& top);
  /// Prints to the stdout
  void print() const;
  /// MurMur2 hash fucntion
  static unsigned int hashFunction(const void* key, int len);
  /// Compute the complete hash as defined for mHash
  static unsigned long getCompleteHash(int nRow, int nCol, const unsigned char patt[ClusterPattern::MaxPatternBytes]);
  static unsigned long getCompleteHash(const ClusterTopology& topology);
  /// Sets the pattern
  void setPattern(int nRow, int nCol, const unsigned char patt[ClusterPattern::MaxPatternBytes]);
  /// Sets the pattern
  void setPattern(const ClusterPattern& patt);

  ///Helper function useful for analyses with topologies stored on a separate branch
  static void makeRareTopologyMap(const std::vector<ClusterTopology>& vec, std::map<int, ClusterPattern>& map)
  {
    for (const auto& topo : vec) {
      auto key = topo.getHash();
      map[key] = topo.getClusterPattern();
    }
  }

 private:
  void setHash(unsigned long hash) { mHash = hash; }
  ClusterPattern mPattern; ///< Pattern of pixels
  /// Hashcode computed from the pattern
  ///
  /// The first four bytes are computed with MurMur2 hash-function. The remaining
  /// four bytes are the first 32 pixels of the pattern. If the number of pixles
  /// is less than 32, the remaining bits are set to 0.
  unsigned long mHash;

  ClassDefNV(ClusterTopology, 2);
};
} // namespace itsmft
} // namespace o2

#endif /* ALICEO2_ITS_CLUSTERTOPOLOGY_H */
