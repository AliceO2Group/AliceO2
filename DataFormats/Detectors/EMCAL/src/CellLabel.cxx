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

using namespace o2::emcal;

CellLabel::CellLabel(const gsl::span<const int> labels, const gsl::span<const float> amplitudeFractions) : mLabels(labels), mAmplitudeFraction(amplitudeFractions)
{
  if (labels.size() != amplitudeFractions.size()) {
    LOG(error) << "Size of labels " << labels.size() << " does not match size of amplitude fraction " << amplitudeFractions.size() << " !";
  }
}
