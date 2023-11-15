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
/// \brief Definition of the TopologyDictionary class for ITS3

#ifndef ALICEO2_ITS3_TOPOLOGYDICTIONARY_H
#define ALICEO2_ITS3_TOPOLOGYDICTIONARY_H

#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "DataFormatsITSMFT/ClusterPattern.h"
#include "DataFormatsITS3/CompCluster.h"

namespace o2
{
namespace its3
{

class BuildTopologyDictionary;
class LookUp;

class TopologyDictionary
{
 public:
  /// Default constructor
  TopologyDictionary();
  /// Constructor
  TopologyDictionary(const std::string& fileName);
  TopologyDictionary& operator=(const its3::TopologyDictionary& dict) = default;

  /// constexpr for the definition of the groups of rare topologies.
  /// The attritbution of the group ID is stringly dependent on the following parameters: it must be a power of 2.
  static constexpr int RowClassSpan = 4;                                                                    ///< Row span of the classes of rare topologies
  static constexpr int ColClassSpan = 4;                                                                    ///< Column span of the classes of rare topologies
  static constexpr int MaxNumberOfRowClasses = 1 + (itsmft::ClusterPattern::MaxRowSpan - 1) / RowClassSpan; ///< Maximum number of row classes for the groups of rare topologies
  static constexpr int MaxNumberOfColClasses = 1 + (itsmft::ClusterPattern::MaxColSpan - 1) / ColClassSpan; ///< Maximum number of col classes for the groups of rare topologies
  static constexpr int NumberOfRareGroups = MaxNumberOfRowClasses * MaxNumberOfColClasses;                  ///< Number of entries corresponding to groups of rare topologies (those whos matrix exceed the max number of bytes are empty).
  /// Prints the dictionary
  friend std::ostream& operator<<(std::ostream& os, const its3::TopologyDictionary& dictionary);
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
  inline const itsmft::ClusterPattern& getPattern(int n) const
  {
    assert(n >= 0 || n < (int)mVectorOfIDs.size());
    return mVectorOfIDs[n].mPattern;
  }

  /// Fills a hostogram with the distribution of the IDs
  static void getTopologyDistribution(const its3::TopologyDictionary& dict, TH1F*& histo, const char* histName);
  /// Returns the number of elements in the dicionary;
  int getSize() const { return (int)mVectorOfIDs.size(); }
  /// Returns the local position of a compact cluster

  // array version of getClusterCoordinates
  template <typename T = float>
  std::array<T, 3> getClusterCoordinatesA(const its3::CompClusterExt& cl, int nChipsITS3 = 6) const;
  /// Returns the local position of a compact cluster
  template <typename T = float>
  static std::array<T, 3> getClusterCoordinatesA(const its3::CompClusterExt& cl, const itsmft::ClusterPattern& patt, bool isGroup = true, int nChipsITS3 = 6);
  /// Returns the local position of a compact cluster
  math_utils::Point3D<float> getClusterCoordinates(const its3::CompClusterExt& cl, int nChipsITS3 = 6) const;
  /// Returns the local position of a compact cluster
  static math_utils::Point3D<float> getClusterCoordinates(const its3::CompClusterExt& cl, const itsmft::ClusterPattern& patt, bool isGroup = true, int nChipsITS3 = 6);

  static TopologyDictionary* loadFrom(const std::string& fileName = "", const std::string& objName = "ccdb_object");

  friend its3::BuildTopologyDictionary;
  friend its3::LookUp;

 private:
  static constexpr int STopoSize = 8 * 255 + 1;
  std::unordered_map<unsigned long, int> mCommonMap; ///< Map of pair <hash, position in mVectorOfIDs>
  std::unordered_map<int, int> mGroupMap;            ///< Map of pair <groudID, position in mVectorOfIDs>
  int mSmallTopologiesLUT[STopoSize];                ///< Look-Up Table for the topologies with 1-byte linearised matrix
  std::vector<itsmft::GroupStruct> mVectorOfIDs;     ///< Vector of topologies and groups

  ClassDefNV(TopologyDictionary, 2);
};
} // namespace its3
} // namespace o2

#endif
