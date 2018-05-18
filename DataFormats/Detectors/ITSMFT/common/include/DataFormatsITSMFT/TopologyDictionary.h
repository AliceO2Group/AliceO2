// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file TopologyDictionary.h
/// \brief Definition of the ClusterTopology class.
///
/// \author Luca Barioglio, University and INFN of Torino
///
/// Short TopologyDictionary descritpion
///
/// The entries of the dictionaries are the cluster topologies, with all the indormation
/// which is common to the clusters with the same topology:
/// - number of rows
/// - number of columns
/// - position of the Centre Of Gravity (COG) wrt the bottom left corner pixel of the bounding box
/// - error associated to the position of the hit point
/// Rare topologies, i.e. with a frequency below a threshold defined a priori, have not their own entries
/// in the dictionaries, but are grouped together with topologies with similar dimensions.
///

#ifndef ALICEO2_ITSMFT_TOPOLOGYDICTIONARY_H
#define ALICEO2_ITSMFT_TOPOLOGYDICTIONARY_H
#include <Rtypes.h>
#include <fstream>
#include <iosfwd>
#include <string>
#include <unordered_map>
#include <vector>

namespace o2
{
namespace ITSMFT
{
class BuildTopologyDictionary;
class LookUp;
class TopologyFastSimulation;

/// Structure containing the most relevant pieces of information of a topology
struct GroupStruct {
  unsigned long mHash; ///< Hashcode
  float mErrX;         ///< Error associated to the hit point in the x direction.
  float mErrZ;         ///< Error associated to the hit point in the z direction.
  float mXCOG;         ///< x position of te COG wrt the boottom left corner of the bounding box
  float mZCOG;         ///< z position of te COG wrt the boottom left corner of the bounding box
  int mNpixels;        ///< Number of fired pixels
  double mFrequency;   ///< Frequency of the topology
};

class TopologyDictionary
{
 public:
  /// constexpr for the definition of the groups of rare topologies
  static constexpr int NumberOfRowClasses = 7; ///< Number of row classes for the groups of rare topologies
  static constexpr int NumberOfColClasses = 7; ///< Number of column classes for the groups of rare topologies
  static constexpr int RowClassSpan = 5;       ///< Row span of the classes of rare topologies
  static constexpr int ColClassSpan = 5;       ///< Column span of the classes of rare topologies
  static constexpr int MaxRowSpan = 32;        ///< Maximum row span
  static constexpr int MaxColSpan = 32;        ///< Maximum column span
  /// Prints the dictionary
  friend std::ostream& operator<<(std::ostream& os, const TopologyDictionary& dictionary);
  /// Prints the dictionary in a binary file
  void WriteBinaryFile(std::string outputFile);
  /// Reads the dictionary from a file
  void ReadFile(std::string fileName);
  /// Reads the dictionary from a binary file
  void ReadBinaryFile(std::string fileName);
  friend BuildTopologyDictionary;
  friend LookUp;
  friend TopologyFastSimulation;

 private:
  std::unordered_map<unsigned long, int> mFinalMap; ///< Map of pair <hash, position in mVectorOfGroupIDs>
  std::vector<GroupStruct> mVectorOfGroupIDs;       ///< Vector of topologies and groups

  ClassDefNV(TopologyDictionary, 1);
};
} // namespace ITSMFT
} // namespace o2

#endif
