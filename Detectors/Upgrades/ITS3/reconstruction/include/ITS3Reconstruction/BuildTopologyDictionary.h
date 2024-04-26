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
/// \brief Definition of the BuildTopologyDictionary class for ITS3

#ifndef ALICEO2_ITS3_BUILDTOPOLOGYDICTIONARY_H
#define ALICEO2_ITS3_BUILDTOPOLOGYDICTIONARY_H

#include "ITSMFTReconstruction/BuildTopologyDictionary.h"
#include "DataFormatsITSMFT/ClusterTopology.h"
#include "ITS3Reconstruction/TopologyDictionary.h"

namespace o2::its3
{

class BuildTopologyDictionary
{
 public:
  static constexpr float IgnoreVal = 999.;
  void accountTopology(const itsmft::ClusterTopology& cluster, float dX = IgnoreVal, float dZ = IgnoreVal);
  void setNCommon(unsigned int nCommon); // set number of common topologies
  void setThreshold(double thr);
  void setThresholdCumulative(double cumulative); // Considering the integral
  void groupRareTopologies();
  friend std::ostream& operator<<(std::ostream& os, const BuildTopologyDictionary& BD);
  void printDictionary(const std::string& fname);
  void printDictionaryBinary(const std::string& fname);
  void saveDictionaryRoot(const std::string& fname);

  unsigned int getTotClusters() const { return mTotClusters; }
  unsigned int getNotInGroups() const { return mNCommonTopologies; }
  TopologyDictionary getDictionary() const { return mDictionary; }

 private:
  TopologyDictionary mDictionary;                                          ///< Dictionary of topologies
  std::map<unsigned long, itsmft::TopoStat> mTopologyMap;                  //! Temporary map of type <hash, TopStat>
  std::vector<std::pair<unsigned long, unsigned long>> mTopologyFrequency; //! <freq,hash>, needed to define threshold
  unsigned int mTotClusters{0};
  unsigned int mNCommonTopologies{0};
  double mFrequencyThreshold{0.};

  std::unordered_map<long unsigned, itsmft::TopologyInfo> mMapInfo;

  ClassDefNV(BuildTopologyDictionary, 3);
};
} // namespace o2::its3

#endif
