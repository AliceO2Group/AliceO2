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
/// Short LookUp descritpion
///
/// This class is for the association of the cluster topology with the corresponding
/// entry in the dictionary - its3 implementation
///

#ifndef ALICEO2_ITS3_LOOKUP_H
#define ALICEO2_ITS3_LOOKUP_H
#include <array>
#include "DataFormatsITSMFT/ClusterTopology.h"
#include "ITS3Reconstruction/TopologyDictionary.h"

namespace o2
{
namespace its3
{
class LookUp
{
 public:
  LookUp();
  LookUp(std::string fileName);
  static int groupFinder(int nRow, int nCol);
  int findGroupID(int nRow, int nCol, const unsigned char patt[itsmft::ClusterPattern::MaxPatternBytes]) const;
  int getTopologiesOverThreshold() const { return mTopologiesOverThreshold; }
  void loadDictionary(std::string fileName);
  void setDictionary(const its3::TopologyDictionary* dict);
  bool isGroup(int id) const { return mDictionary.isGroup(id); }
  int size() const { return mDictionary.getSize(); }
  auto getPattern(int id) const { return mDictionary.getPattern(id); }
  auto getDictionaty() const { return mDictionary; }

 private:
  its3::TopologyDictionary mDictionary;
  int mTopologiesOverThreshold;

  ClassDefNV(LookUp, 1);
};
} // namespace its3
} // namespace o2

#endif
