// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file LookUp.cxx
/// \brief Implementation of the LookUp class.
///
/// \author Luca Barioglio, University and INFN of Torino

#include "EndCapsReconstruction/LookUp.h"
#include "DataFormatsEndCaps/TopologyDictionary.h"

ClassImp(o2::endcaps::LookUp);

using std::array;

namespace o2
{
namespace endcaps
{

LookUp::LookUp() : mDictionary{}, mTopologiesOverThreshold{0} {}

LookUp::LookUp(std::string fileName)
{
  loadDictionary(fileName);
}

void LookUp::loadDictionary(std::string fileName)
{
  mDictionary.readBinaryFile(fileName);
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
  if (row_index > TopologyDictionary::MaxNumberOfClasses || col_index > TopologyDictionary::MaxNumberOfClasses) {
    return TopologyDictionary::NumberOfRareGroups - 1;
  } else {
    return row_index * TopologyDictionary::MaxNumberOfClasses + col_index;
  }
}

int LookUp::findGroupID(int nRow, int nCol, const unsigned char patt[o2::itsmft::Cluster::kMaxPatternBytes])
{
  int nBits = nRow * nCol;
  // Small topology
  if (nBits < 9) {
    int ID = mDictionary.mSmallTopologiesLUT[(nCol - 1) * 255 + (int)patt[0]];
    if (ID >= 0)
      return ID;
    else { //small rare topology (inside groups)
      int index = groupFinder(nRow, nCol);
      return mDictionary.mGroupMap[index];
    }
  }
  // Big topology
  unsigned long hash = o2::itsmft::ClusterTopology::getCompleteHash(nRow, nCol, patt);
  auto ret = mDictionary.mCommonMap.find(hash);
  if (ret != mDictionary.mCommonMap.end())
    return ret->second;
  else { // Big rare topology (inside groups)
    int index = groupFinder(nRow, nCol);
    return mDictionary.mGroupMap[index];
  }
}

bool LookUp::isGroup(int id) const
{
  return mDictionary.isGroup(id);
}

} // namespace endcaps
} // namespace o2
