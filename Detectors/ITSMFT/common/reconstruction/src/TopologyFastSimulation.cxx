// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file TopologyFastSimulation.cxx
/// \brief Implementation of the TopologyFastSimulation class.
///
/// \author Luca Barioglio, University and INFN of Torino

#include "ITSMFTReconstruction/TopologyFastSimulation.h"
#include <algorithm>

ClassImp(o2::itsmft::TopologyFastSimulation);

namespace o2
{
  namespace itsmft
  {
  TopologyFastSimulation::TopologyFastSimulation(std::string fileName, unsigned seed)
  {
    mDictionary.ReadBinaryFile(fileName);
    mGenerator = std::mt19937(seed);
    mDistribution = std::uniform_real_distribution<double>(0.0, 1.0);
  }

  int TopologyFastSimulation::getRandom()
  {
    double rnd = mDistribution(mGenerator);
    auto ind = std::upper_bound(mDictionary.mVectorOfGroupIDs.begin(), mDictionary.mVectorOfGroupIDs.end(), rnd,
                                [](const double& comp1, const GroupStruct& comp2) { return comp1 < comp2.mFrequency; });
    return std::distance(mDictionary.mVectorOfGroupIDs.begin(), ind);
  }
  } // namespace itsmft
}
