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

#include "ITSMFTReconstruction/LookUp.h"

ClassImp(o2::ITSMFT::LookUp)

  using std::array;

namespace o2
{
namespace ITSMFT
{
LookUp::LookUp(std::string fileName)
{
  mDictionary.ReadBinaryFile(fileName);
  mTopologiesOverThreshold = mDictionary.mFinalMap.size();
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

int LookUp::findGroupID(int nRow, int nCol, const unsigned char patt[Cluster::kMaxPatternBytes])
{
  int nBits = nRow * nCol;
  // Small topology
  if (nBits < 9) {
    int ID = mDictionary.mSmallTopologiesLUT[(nRow - 1) * 255 + (int)patt[0]];
    if (ID >= 0)
      return ID;
    else { //small rare topology (inside groups)
      int index = groupFinder(nRow, nCol);
      return (mTopologiesOverThreshold + index);
    }
  }
  // Big topology
  unsigned long hash = ClusterTopology::getCompleteHash(nRow, nCol, patt);
  auto ret = mDictionary.mFinalMap.find(hash);
  if (ret != mDictionary.mFinalMap.end())
    return ret->second;
  else { // Big rare topology (inside groups)
    int index = groupFinder(nRow, nCol);
    return (mTopologiesOverThreshold + index);
  }
}
} // namespace ITSMFT
} // namespace o2
