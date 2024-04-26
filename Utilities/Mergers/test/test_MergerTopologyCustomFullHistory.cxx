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

/// \file    test_MergerTopologyCustomFullHistory.cxx
/// \author  Michal Tichak
///
/// \brief Test which creates DPL workflow for full history merging of merger interfaces

#include "customTopologyCommon.h"

using namespace o2::framework;
using namespace o2::mergers;

WorkflowSpec defineDataProcessing(ConfigContext const&)
{

  constexpr size_t producersCount = 2;

  WorkflowSpec specs;

  CustomMergerTestGenerator generator(producersCount);

  const auto mergerInputs = generator.generateHistoProducers(specs, producersCount);

  generator.generateMergers(specs, mergerInputs, InputObjectsTimespan::FullHistory);

  generator.generateCheckerFullHistory(specs);

  return specs;
}
