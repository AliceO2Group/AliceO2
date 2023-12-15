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

/// \file ClusterLabel.cxx

#include "DataFormatsEMCAL/ClusterLabel.h"

using namespace o2::emcal;

//_______________________________________________________________________
void ClusterLabel::clear()
{
  mClusterLabels.clear();
}

//_______________________________________________________________________
void ClusterLabel::addValue(int label, float energyFraction)
{
  auto it = std::find_if(mClusterLabels.begin(), mClusterLabels.end(),
                         [label](const labelWithE& lWE) { return lWE.label == label; });

  if (it != mClusterLabels.end()) {
    // label already exists, accumulate energy fraction
    it->energyFraction += energyFraction;
  } else {
    // label does not exist, add new energy fraction
    mClusterLabels.emplace_back(label, energyFraction);
  }
}

//_______________________________________________________________________
void ClusterLabel::normalize(float factor)
{
  for (auto& clusterlabel : mClusterLabels) {
    clusterlabel.energyFraction = clusterlabel.energyFraction / factor;
  }
}

//_______________________________________________________________________
std::vector<int32_t> ClusterLabel::getLabels()
{
  std::vector<int32_t> vLabels;
  vLabels.reserve(mClusterLabels.size());
  for (auto& clusterlabel : mClusterLabels) {
    vLabels.push_back(clusterlabel.label);
  }
  return vLabels;
}

//_______________________________________________________________________
std::vector<float> ClusterLabel::getEnergyFractions()
{
  std::vector<float> vEnergyFractions;
  vEnergyFractions.reserve(mClusterLabels.size());
  for (auto& clusterlabel : mClusterLabels) {
    vEnergyFractions.push_back(clusterlabel.energyFraction);
  }
  return vEnergyFractions;
}

//_______________________________________________________________________
void ClusterLabel::orderLabels()
{
  // Sort the pairs based on values in descending order
  std::sort(mClusterLabels.begin(), mClusterLabels.end(),
            [](const labelWithE& a, const labelWithE& b) { return a.label >= b.label; });
}
