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

/// \file CellLabel.cxx

#include "DataFormatsEMCAL/CellLabel.h"
#include "fairlogger/Logger.h"
#include <cstddef>
#include <cstdint>
#include <gsl/span>
#include <vector>
#include <utility>

using namespace o2::emcal;

CellLabel::CellLabel(std::vector<int> labels, std::vector<float> amplitudeFractions) : mLabels(std::move(labels)), mAmplitudeFraction(std::move(amplitudeFractions))
{
  if (labels.size() != amplitudeFractions.size()) {
    LOG(error) << "Size of labels " << labels.size() << " does not match size of amplitude fraction " << amplitudeFractions.size() << " !";
  }
}

CellLabel::CellLabel(gsl::span<const int> labels, gsl::span<const float> amplitudeFractions) : mLabels(labels.begin(), labels.end()), mAmplitudeFraction(amplitudeFractions.begin(), amplitudeFractions.end())
{
  if (labels.size() != amplitudeFractions.size()) {
    LOG(error) << "Size of labels " << labels.size() << " does not match size of amplitude fraction " << amplitudeFractions.size() << " !";
  }
}

int32_t CellLabel::GetLeadingMCLabel() const
{
  size_t maxIndex = 0;
  float maxFraction = mAmplitudeFraction[0];

  for (size_t i = 1; i < mAmplitudeFraction.size(); ++i) {
    if (mAmplitudeFraction[i] > maxFraction) {
      maxFraction = mAmplitudeFraction[i];
      maxIndex = i;
    }
  }
  return mLabels[maxIndex];
}
