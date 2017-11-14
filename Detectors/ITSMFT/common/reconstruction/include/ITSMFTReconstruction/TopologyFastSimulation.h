// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file TopologyFastSimulation.h
/// \brief Definition of the TopologyFastSimulation class.
///
/// \author Luca Barioglio, University and INFN of Torino
///
/// Short TopologyFastSimulation descritpion
///
/// This class is used for the generation of a distribution of topologies according to
/// to the frequencies of the entries in the dictionary
///


#ifndef ALICEO2_ITSMFT_TOPOLOGYFASTSIMULATION_H
#define ALICEO2_ITSMFT_TOPOLOGYFASTSIMULATION_H
#include "ITSMFTReconstruction/TopologyDictionary.h"
#include <random>

namespace o2
{
namespace ITSMFT
{
class TopologyFastSimulation{

  public:
    TopologyFastSimulation(std::string fileName, unsigned seed=0xdeadbeef);
    int getRandom();

  private:
    TopologyDictionary mDictionary;
    std::mt19937 mGenerator;
    std::uniform_real_distribution<double> mDistribution;

  ClassDefNV(TopologyFastSimulation,1);
};
}
}

#endif
