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

/// \file BuildTopologyDictionary.h
/// \brief Definition of the BuildTopologyDictionary class.
///
/// \author Luca Barioglio, University and INFN of Torino
///
/// Short BuildTopologyDictionary descritpion
///
/// This class is used to build the dictionary of topologies, storing the information
/// concerning topologies with their own entry and groups of rare topologies
///

#ifndef ALICEO2_ITSMFT_BUILDTOPOLOGYDICTIONARY_H
#define ALICEO2_ITSMFT_BUILDTOPOLOGYDICTIONARY_H
#include <algorithm>
#include <map>
#include "ITSMFTBase/SegmentationAlpide.h"
#include "DataFormatsITSMFT/ClusterTopology.h"
#include "DataFormatsITSMFT/TopologyDictionary.h"

namespace o2
{
namespace itsmft
{
struct TopologyInfo {
  int mSizeX = 0;
  int mSizeZ = 0;
  float mCOGx = 0.f;
  float mCOGz = 0.f;
  float mXmean = 0.f;
  float mXsigma2 = 0.f;
  float mZmean = 0.f;
  float mZsigma2 = 0.f;
  int mNpixels = 0;
  ClusterPattern mPattern; ///< Bitmask of pixels. For groups the biggest bounding box for the group is taken, with all
                           ///the bits set to 1.
};

// transient structure to accumulate topology statistics
struct TopoStat {
  ClusterTopology topology;
  unsigned long countsTotal = 0;    // counts for this topology
  unsigned long countsWithBias = 0; // counts with assigned dX,dY provided
  TopoStat() = default;
};

class BuildTopologyDictionary
{
 public:
  BuildTopologyDictionary();
  static constexpr float IgnoreVal = 999.;
  void accountTopology(const ClusterTopology& cluster, float dX = IgnoreVal, float dZ = IgnoreVal);
  void setNCommon(unsigned int nCommon); // set number of common topologies
  void setThreshold(double thr);
  void setThresholdCumulative(double cumulative); // Considering the integral
  void groupRareTopologies();
  friend std::ostream& operator<<(std::ostream& os, const BuildTopologyDictionary& BD);
  void printDictionary(const std::string& fname);
  void printDictionaryBinary(const std::string& fname);
  void saveDictionaryRoot(const std::string& fname);

  int getTotClusters() const { return mTotClusters; }
  int getNotInGroups() const { return mNCommonTopologies; }
  TopologyDictionary getDictionary() const { return mDictionary; }

 protected:
  TopologyDictionary mDictionary;                                          ///< Dictionary of topologies
  std::map<unsigned long, TopoStat> mTopologyMap;                          //! Temporary map of type <hash,TopStat>
  std::vector<std::pair<unsigned long, unsigned long>> mTopologyFrequency; //! <freq,hash>, needed to define threshold
  int mTotClusters;
  int mNCommonTopologies;
  double mFrequencyThreshold;

  std::unordered_map<long unsigned, TopologyInfo> mMapInfo;

  ClassDefNV(BuildTopologyDictionary, 4);
};
} // namespace itsmft
} // namespace o2
#endif
