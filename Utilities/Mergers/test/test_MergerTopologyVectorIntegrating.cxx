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

/// \file    test_MergerTopologyVectorIntegrating.cxx
/// \author  Michal Tichak
///
/// \brief   Test which creates DPL workflow for integrating (delta) merging of vectors of TObjects

#include "vectorTopologyCommon.h"

using namespace o2::framework;
using namespace o2::mergers;

WorkflowSpec defineDataProcessing(ConfigContext const&)
{

  constexpr size_t producersCount = 2;
  constexpr size_t histoBinsCount = 10;
  constexpr double histoMin = 0.;
  constexpr double histoMax = 10.;

  WorkflowSpec specs;

  VectorMergerTestGenerator<12> generator({{0., 0., 1., 1., 0., 0., 2., 0., 0., 0., 0., 0.}, {0., 0., 1., 1., 0., 0., 2., 0., 0., 0., 0., 0.}}, histoBinsCount, histoMin, histoMax);

  const auto mergerInputs = generator.generateHistoProducers(specs, producersCount);

  generator.generateMergers(specs, mergerInputs, InputObjectsTimespan::LastDifference);

  generator.generateChecker(specs);

  return specs;
}
