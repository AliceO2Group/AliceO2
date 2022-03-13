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

// // we need to add workflow options before including Framework/runDataProcessing
// void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
// {
//   // option allowing to set parameters
// }

// ------------------------------------------------------------------

#include <fmt/format.h>

#include "Framework/ConfigParamSpec.h"

#include "DCStestWorkflow/DCSDataReplaySpec.h"
#include "TPCdcs/DCSDPHints.h"

using namespace o2::framework;

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  std::vector<ConfigParamSpec> options{
    {"max-sectors", VariantType::Int, 0, {"max sector number to use for HV sensors, 0-17"}},
  };

  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"

o2::framework::WorkflowSpec defineDataProcessing(ConfigContext const& config)
{
  const auto maxSectors = std::min(config.options().get<int>("max-sectors"), 17);

  auto dphints = o2::tpc::dcs::getTPCDCSDPHints(maxSectors);

  WorkflowSpec specs;
  specs.emplace_back(o2::dcs::test::getDCSDataReplaySpec(dphints, "TPC"));
  return specs;
}
