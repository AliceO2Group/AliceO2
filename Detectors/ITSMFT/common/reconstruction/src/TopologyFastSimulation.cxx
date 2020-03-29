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
  mDictionary.readBinaryFile(fileName);
  double tot_freq = 0.;
  int dictSize = mDictionary.getSize();
  mFreqArray.reserve(dictSize);
  for (int iKey = 0; iKey < dictSize; iKey++) {
    tot_freq += mDictionary.getFrequency(iKey);
    mFreqArray[iKey] = tot_freq;
  }
  mGenerator = std::mt19937(seed);
  mDistribution = std::uniform_real_distribution<double>(0.0, 1.0);
}

int TopologyFastSimulation::getRandom()
{
  double rnd = mDistribution(mGenerator);
  auto ind = std::upper_bound(mFreqArray.begin(), mFreqArray.end(), rnd,
                              [](const double& comp1, const double& comp2) { return comp1 < comp2; });
  return std::distance(mFreqArray.begin(), ind);
}
} // namespace itsmft
} // namespace o2
