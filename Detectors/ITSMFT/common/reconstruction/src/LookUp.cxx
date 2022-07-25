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

/// \file LookUp.cxx
/// \brief Implementation of the LookUp class.
///
/// \author Luca Barioglio, University and INFN of Torino

#include "ITSMFTReconstruction/LookUp.h"
#include "DataFormatsITSMFT/CompCluster.h"

ClassImp(o2::itsmft::LookUp);

using std::array;

namespace o2
{
namespace itsmft
{

LookUp::LookUp() : mDictionary{}, mTopologiesOverThreshold{0} {}

LookUp::LookUp(std::string fileName)
{
  loadDictionary(fileName);
}

void LookUp::loadDictionary(std::string fileName)
{
  mDictionary.readFromFile(fileName);
  mTopologiesOverThreshold = mDictionary.mCommonMap.size();
}

void LookUp::setDictionary(const TopologyDictionary* dict)
{
  if (dict) {
    mDictionary = *dict;
  }
  mTopologiesOverThreshold = mDictionary.mCommonMap.size();
}

int LookUp::groupFinder(int nRow, int nCol)
{
  int row_index = nRow / TopologyDictionary::RowClassSpan;
  if (nRow % TopologyDictionary::RowClassSpan == 0) {
    row_index--;
  }
  int col_index = nCol / TopologyDictionary::ColClassSpan;
  if (nCol % TopologyDictionary::RowClassSpan == 0) {
    col_index--;
  }
  int grNum = -1;
  if (row_index > TopologyDictionary::MaxNumberOfRowClasses || col_index > TopologyDictionary::MaxNumberOfColClasses) {
    grNum = TopologyDictionary::NumberOfRareGroups - 1;
  } else {
    grNum = row_index * TopologyDictionary::MaxNumberOfColClasses + col_index;
  }
  return grNum;
}

int LookUp::findGroupID(int nRow, int nCol, const unsigned char patt[ClusterPattern::MaxPatternBytes]) const
{
  int nBits = nRow * nCol;
  if (nBits < 9) { // Small unique topology
    int ID = mDictionary.mSmallTopologiesLUT[(nCol - 1) * 255 + (int)patt[0]];
    if (ID >= 0) {
      return ID;
    }
  } else { // Big unique topology
    unsigned long hash = ClusterTopology::getCompleteHash(nRow, nCol, patt);
    auto ret = mDictionary.mCommonMap.find(hash);
    if (ret != mDictionary.mCommonMap.end()) {
      return ret->second;
    }
  }
  if (!mDictionary.mGroupMap.empty()) { // rare valid topology group
    int index = groupFinder(nRow, nCol);
    auto res = mDictionary.mGroupMap.find(index);
    return res == mDictionary.mGroupMap.end() ? CompCluster::InvalidPatternID : res->second;
  }
  return CompCluster::InvalidPatternID;
}

} // namespace itsmft
} // namespace o2
