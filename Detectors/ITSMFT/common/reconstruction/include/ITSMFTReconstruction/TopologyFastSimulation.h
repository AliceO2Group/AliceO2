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
#include <random>
#include "DataFormatsITSMFT/TopologyDictionary.h"

namespace o2
{
namespace itsmft
{
class TopologyFastSimulation
{
 public:
  TopologyFastSimulation(std::string fileName, unsigned seed = 0xdeadbeef);
  int getRandom();

 private:
  TopologyDictionary mDictionary;
  std::vector<double> mFreqArray;
  std::mt19937 mGenerator;
  std::uniform_real_distribution<double> mDistribution;

  ClassDefNV(TopologyFastSimulation, 1);
};
} // namespace itsmft
} // namespace o2

#endif
