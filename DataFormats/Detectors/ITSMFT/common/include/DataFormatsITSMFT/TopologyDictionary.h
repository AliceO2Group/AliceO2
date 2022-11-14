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
#include "Framework/Logger.h"
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>
#include "MathUtils/Cartesian.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "TH1F.h"

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
  float mErr2X;            ///< Squared Error associated to the hit point in the x direction.
  float mErr2Z;            ///< Squared Error associated to the hit point in the z direction.
  float mXCOG;             ///< x position of te COG wrt the boottom left corner of the bounding box
  float mZCOG;             ///< z position of te COG wrt the boottom left corner of the bounding box
  int mNpixels;            ///< Number of fired pixels
  ClusterPattern mPattern; ///< Bitmask of pixels. For groups the biggest bounding box for the group is taken, with all
                           /// the bits set to 1.
  double mFrequency;       ///< Frequency of the topology
  bool mIsGroup;           ///< false: common topology; true: group of rare topologies
  ClassDefNV(GroupStruct, 3);
};

class TopologyDictionary
{
 public:
  /// Default constructor
  TopologyDictionary();
  /// Constructor
  TopologyDictionary(const std::string& fileName);
  TopologyDictionary& operator=(const TopologyDictionary& dict) = default;
  /// constexpr for the definition of the groups of rare topologies.
  /// The attritbution of the group ID is stringly dependent on the following parameters: it must be a power of 2.
  static constexpr int RowClassSpan = 4;                                                            ///< Row span of the classes of rare topologies
  static constexpr int ColClassSpan = 4;                                                            ///< Column span of the classes of rare topologies
  static constexpr int MaxNumberOfRowClasses = 1 + (ClusterPattern::MaxRowSpan - 1) / RowClassSpan; ///< Maximum number of row classes for the groups of rare topologies
  static constexpr int MaxNumberOfColClasses = 1 + (ClusterPattern::MaxColSpan - 1) / ColClassSpan; ///< Maximum number of col classes for the groups of rare topologies
  static constexpr int NumberOfRareGroups = MaxNumberOfRowClasses * MaxNumberOfColClasses;          ///< Number of entries corresponding to groups of rare topologies (those whos matrix exceed the max number of bytes are empty).
  /// Prints the dictionary
  friend std::ostream& operator<<(std::ostream& os, const TopologyDictionary& dictionary);
  /// Prints the dictionary in a binary file
  void writeBinaryFile(const std::string& outputFile);
  /// Reads the dictionary from a binary file
  int readBinaryFile(const std::string& fileName);

  int readFromFile(const std::string& fileName);

  /// Returns the x position of the COG for the n_th element
  inline float getXCOG(int n) const
  {
    assert(n >= 0 || n < (int)mVectorOfIDs.size());
    return mVectorOfIDs[n].mXCOG;
  }
  /// Returns the error on the x position of the COG for the n_th element
  inline float getErrX(int n) const
  {
    assert(n >= 0 || n < (int)mVectorOfIDs.size());
    return mVectorOfIDs[n].mErrX;
  }
  /// Returns the z position of the COG for the n_th element
  inline float getZCOG(int n) const
  {
    assert(n >= 0 || n < (int)mVectorOfIDs.size());
    return mVectorOfIDs[n].mZCOG;
  }
  /// Returns the error on the z position of the COG for the n_th element
  inline float getErrZ(int n) const
  {
    assert(n >= 0 || n < (int)mVectorOfIDs.size());
    return mVectorOfIDs[n].mErrZ;
  }
  /// Returns the error^2 on the x position of the COG for the n_th element
  inline float getErr2X(int n) const
  {
    assert(n >= 0 || n < (int)mVectorOfIDs.size());
    return mVectorOfIDs[n].mErr2X;
  }
  /// Returns the error^2 on the z position of the COG for the n_th element
  inline float getErr2Z(int n) const
  {
    assert(n >= 0 || n < (int)mVectorOfIDs.size());
    return mVectorOfIDs[n].mErr2Z;
  }
  /// Returns the hash of the n_th element
  inline unsigned long getHash(int n) const
  {
    assert(n >= 0 || n < (int)mVectorOfIDs.size());
    return mVectorOfIDs[n].mHash;
  }
  /// Returns the number of fired pixels of the n_th element
  inline int getNpixels(int n) const
  {
    assert(n >= 0 || n < (int)mVectorOfIDs.size());
    return mVectorOfIDs[n].mNpixels;
  }
  /// Returns the frequency of the n_th element;
  inline double getFrequency(int n) const
  {
    assert(n >= 0 || n < (int)mVectorOfIDs.size());
    return mVectorOfIDs[n].mFrequency;
  }
  /// Returns true if the element corresponds to a group of rare topologies
  inline bool isGroup(int n) const
  {
    assert(n >= 0 || n < (int)mVectorOfIDs.size());
    return mVectorOfIDs[n].mIsGroup;
  }
  /// Returns the pattern of the topology
  inline const ClusterPattern& getPattern(int n) const
  {
    assert(n >= 0 || n < (int)mVectorOfIDs.size());
    return mVectorOfIDs[n].mPattern;
  }

  /// Fills a hostogram with the distribution of the IDs
  static void getTopologyDistribution(const TopologyDictionary& dict, TH1F*& histo, const char* histName);
  /// Returns the number of elements in the dicionary;
  int getSize() const { return (int)mVectorOfIDs.size(); }
  /// Returns the local position of a compact cluster

  // array version of getClusterCoordinates
  template <typename T = float>
  std::array<T, 3> getClusterCoordinatesA(const CompCluster& cl) const;
  /// Returns the local position of a compact cluster
  template <typename T = float>
  static std::array<T, 3> getClusterCoordinatesA(const CompCluster& cl, const ClusterPattern& patt, bool isGroup = true);

  template <typename T = float>
  math_utils::Point3D<T> getClusterCoordinates(const CompCluster& cl) const;
  /// Returns the local position of a compact cluster
  template <typename T = float>
  static math_utils::Point3D<T> getClusterCoordinates(const CompCluster& cl, const ClusterPattern& patt, bool isGroup = true);

  static TopologyDictionary* loadFrom(const std::string& fileName = "", const std::string& objName = "ccdb_object");

  friend BuildTopologyDictionary;
  friend LookUp;
  friend TopologyFastSimulation;

 private:
  static constexpr int STopoSize = 8 * 255 + 1;
  std::unordered_map<unsigned long, int> mCommonMap; ///< Map of pair <hash, position in mVectorOfIDs>
  std::unordered_map<int, int> mGroupMap;            ///< Map of pair <groudID, position in mVectorOfIDs>
  int mSmallTopologiesLUT[STopoSize];                ///< Look-Up Table for the topologies with 1-byte linearised matrix
  std::vector<GroupStruct> mVectorOfIDs;             ///< Vector of topologies and groups

  ClassDefNV(TopologyDictionary, 4);
}; // namespace itsmft
} // namespace itsmft
} // namespace o2

#endif
