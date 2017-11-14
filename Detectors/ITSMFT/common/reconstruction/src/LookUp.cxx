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

LookUp::LookUp(std::string fileName){
  mDictionary.ReadBinaryFile(fileName);
  mTopologiesOverThreshold = mDictionary.mFinalMap.size();
}

int LookUp::findGroupID(int nRow, int nCol, const unsigned char patt[Cluster::kMaxPatternBytes], int nBytesUsed){
  unsigned long hash = ClusterTopology::getCompleteHash(nRow,nCol,patt,nBytesUsed);
  auto ret = mDictionary.mFinalMap.find(hash);
  if(ret!=mDictionary.mFinalMap.end()) return ret->second;
  else{
    int index = (nRow/TopologyDictionary::RowClassSpan)*TopologyDictionary::NumberOfRowClasses + nCol/TopologyDictionary::ColClassSpan;
    if(index>=TopologyDictionary::NumberOfRowClasses*TopologyDictionary::NumberOfColClasses) index = TopologyDictionary::NumberOfRowClasses*TopologyDictionary::NumberOfColClasses;
    return (mTopologiesOverThreshold+index);
  }
}
}
}
