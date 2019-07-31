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
/// The entries of the dictionaries are the cluster topologies, with all the information
/// which is common to the clusters with the same topology:
/// - number of rows
/// - number of columns
/// - pixel bitmask
/// - position of the Centre Of Gravity (COG) wrt the bottom left corner pixel of the bounding box
/// - error associated to the position of the hit point
/// Rare topologies, i.e. with a frequency below a threshold defined a priori, have not their own entries
/// in the dictionaries, but are grouped together with topologies with similar dimensions.
/// For the groups of rare topollogies a dummy bitmask is used.

#ifndef ALICEO2_ITSMFT_TOPOLOGYDICTIONARY_H
#define ALICEO2_ITSMFT_TOPOLOGYDICTIONARY_H
#include "DataFormatsITSMFT/ClusterPattern.h"
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>
#include "MathUtils/Cartesian3D.h"
#include "DataFormatsITSMFT/CompCluster.h"

namespace o2
{
namespace itsmft
{
class BuildTopologyDictionary;
class LookUp;
class TopologyFastSimulation;

/// Structure containing the most relevant pieces of information of a topology
struct GroupStruct {
  unsigned long mHash;     ///< Hashcode
  float mErrX;             ///< Error associated to the hit point in the x direction.
  float mErrZ;             ///< Error associated to the hit point in the z direction.
  float mXCOG;             ///< x position of te COG wrt the boottom left corner of the bounding box
  float mZCOG;             ///< z position of te COG wrt the boottom left corner of the bounding box
  int mNpixels;            ///< Number of fired pixels
  ClusterPattern mPattern; ///< Bitmask of pixels. For groups the biggest bounding box for the group is taken, with all
                           ///the bits set to 1.
  double mFrequency;       ///< Frequency of the topology
  bool mIsGroup;           ///< false: common topology; true: group of rare topologies
  ClassDefNV(GroupStruct, 2);
};

class TopologyDictionary
{
 public:
  /// Default constructor
  TopologyDictionary();
  /// Constructor
  TopologyDictionary(std::string fileName);

  /// constexpr for the definition of the groups of rare topologies.
  /// The attritbution of the group ID is stringly dependent on the following parameters: it must be a power of 2.
  static constexpr int RowClassSpan = 4;                                                 ///< Row span of the classes of rare topologies
  static constexpr int ColClassSpan = 4;                                                 ///< Column span of the classes of rare topologies
  static constexpr int MinimumClassArea = RowClassSpan * ColClassSpan;                   ///< Area of the smallest class of rare topologies (used as reference)
  static constexpr int MaxNumberOfClasses = Cluster::kMaxPatternBits / MinimumClassArea; ///< Maximum number of row/column classes for the groups of rare topologies
  static constexpr int NumberOfRareGroups = MaxNumberOfClasses * MaxNumberOfClasses;     ///< Number of entries corresponding to groups of rare topologies (those whos matrix exceed the max number of bytes are empty).
  /// Prints the dictionary
  friend std::ostream& operator<<(std::ostream& os, const TopologyDictionary& dictionary);
  /// Prints the dictionary in a binary file
  void WriteBinaryFile(std::string outputFile);
  /// Reads the dictionary from a binary file
  int ReadBinaryFile(std::string fileName);
  /// Returns the x position of the COG for the n_th element
  float GetXcog(int n) const;
  /// Returns the error on the x position of the COG for the n_th element
  float GetErrX(int n) const;
  /// Returns the z position of the COG for the n_th element
  float GetZcog(int n) const;
  /// Returns the error on the z position of the COG for the n_th element
  float GetErrZ(int n) const;
  /// Returns the hash of the n_th element
  unsigned long GetHash(int n) const;
  /// Returns the number of fired pixels of the n_th element
  int GetNpixels(int n) const;
  /// Returns true if the element corresponds to a group of rare topologies
  inline bool IsGroup(int n) const
  {
    if (n >= (int)mVectorOfGroupIDs.size()) {
      LOG(ERROR) << "Index out of bounds";
      return false;
    } else
      return mVectorOfGroupIDs[n].mIsGroup;
  }

  /// Returns the pattern of the topology
  ClusterPattern
    GetPattern(int n) const;
  /// Returns the frequency of the n_th element;
  double GetFrequency(int n) const;
  /// Returns the number of elements in the dicionary;
  int GetSize() const { return (int)mVectorOfGroupIDs.size(); }
  ///Returns the local position of a compact cluster
  Point3D<float> getClusterCoordinates(const CompCluster& cl) const;

  friend BuildTopologyDictionary;
  friend LookUp;
  friend TopologyFastSimulation;

 private:
  std::unordered_map<unsigned long, int> mFinalMap; ///< Map of pair <hash, position in mVectorOfGroupIDs>
  int mSmallTopologiesLUT[8 * 255];                 ///< Look-Up Table for the topologies with 1-byte linearised matrix
  std::vector<GroupStruct> mVectorOfGroupIDs;       ///< Vector of topologies and groups

  ClassDefNV(TopologyDictionary, 3);
}; // namespace itsmft
} // namespace itsmft
} // namespace o2

#endif
