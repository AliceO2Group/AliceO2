// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file LookUp.h
/// \brief Definition of the LookUp class.
///
/// \author Luca Barioglio, University and INFN of Torino
///
/// Short LookUp descritpion
///
/// This class is for the association of the cluster topology with the corresponding
/// entry in the dictionary
///

#ifndef ALICEO2_ITSMFT_LOOKUP_H
#define ALICEO2_ITSMFT_LOOKUP_H
#include <array>
#include "DataFormatsITSMFT/ClusterTopology.h"
#include "DataFormatsITSMFT/TopologyDictionary.h"

namespace o2
{
namespace itsmft
{
class LookUp
{
 public:
  LookUp();
  LookUp(std::string fileName);
  static int groupFinder(int nRow, int nCol);
  int findGroupID(int nRow, int nCol, const unsigned char patt[Cluster::kMaxPatternBytes]);
  int getTopologiesOverThreshold() { return mTopologiesOverThreshold; }
  void loadDictionary(std::string fileName);
  bool IsGroup(int id) const;

 private:
  TopologyDictionary mDictionary;
  int mTopologiesOverThreshold;

  ClassDefNV(LookUp, 2);
};
} // namespace itsmft
} // namespace o2

#endif
