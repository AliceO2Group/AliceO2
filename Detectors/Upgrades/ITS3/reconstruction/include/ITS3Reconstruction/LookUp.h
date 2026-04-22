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

/// \file LookUp.h
/// \brief Definition of the LookUp class for its3.
///
/// Short LookUp description
///
/// This class is for the association of the cluster topology with the corresponding
/// entry in the dictionary - its3 implementation
///

#ifndef ALICEO2_ITS3_LOOKUP_H
#define ALICEO2_ITS3_LOOKUP_H

#include "ITS3Base/SpecsV2.h"
#include "ITS3Reconstruction/TopologyDictionary.h"

namespace o2::its3
{
class LookUp
{
 public:
  LookUp() = default;
  LookUp(std::string fileName);
  static int groupFinder(int nRow, int nCol);
  int findGroupID(int nRow, int nCol, bool IB, const unsigned char patt[itsmft::ClusterPattern::MaxPatternBytes]) const;
  int findGroupID(int nRow, int nCol, uint16_t chipID, const unsigned char patt[itsmft::ClusterPattern::MaxPatternBytes]) const
  {
    return findGroupID(nRow, nCol, constants::detID::isDetITS3(chipID), patt);
  }
  int getTopologiesOverThreshold(bool IB) const { return (IB) ? mTopologiesOverThresholdIB : mTopologiesOverThresholdOB; }
  void loadDictionary(std::string fileName);
  void setDictionary(const TopologyDictionary* dict);
  auto getDictionary() const { return mDictionary; }
  bool isGroup(int id, bool IB) const { return mDictionary.isGroup(id, IB); }
  bool isGroup(int id, uint16_t chipID) const { return isGroup(id, constants::detID::isDetITS3(chipID)); }
  int size(bool IB) const { return mDictionary.getSize(IB); }
  int size(uint16_t chipID) const { return size(constants::detID::isDetITS3(chipID)); }
  auto getPattern(int id, bool IB) const { return mDictionary.getPattern(id, IB); }

 private:
  TopologyDictionary mDictionary;
  int mTopologiesOverThresholdIB{0};
  int mTopologiesOverThresholdOB{0};

  ClassDefNV(LookUp, 3);
};
} // namespace o2::its3

#endif
