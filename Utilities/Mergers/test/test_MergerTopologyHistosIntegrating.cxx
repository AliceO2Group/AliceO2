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

/// \file    test_MergerTopologyHistosIntegrating.cxx
/// \author  Michal Tichak
///
/// \brief   Test which creates DPL workflow for integrating (delta) merging of histograms

#include "histosTopologyCommon.h"

o2::framework::WorkflowSpec defineDataProcessing(o2::framework::ConfigContext const&)
{
  constexpr size_t producersCount = 2;

  constexpr size_t binsCount = 10;
  constexpr double min = 0;
  constexpr double max = 10;

  o2::framework::WorkflowSpec specs;

  o2::framework::HistosMergerTestGenerator generator(std::array<float, 12>{0., 0., 1., 1., 0., 0., 2., 0., 0., 0., 0., 0.}, binsCount, min, max);

  const auto mergersInputs = generator.generateHistoProducers(specs, producersCount);

  generator.generateMergers(specs, mergersInputs, o2::mergers::InputObjectsTimespan::LastDifference);

  generator.generateChecker(specs);

  return specs;
}
