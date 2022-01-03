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

#include <vector>
#include "Framework/ConfigParamSpec.h"
#include "MCHWorkflow/TrackWriterSpec.h"

using namespace o2::framework;

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  workflowOptions.emplace_back("enable-mc", VariantType::Bool, false, ConfigParamSpec::HelpString{"Propagate MC info"});
  workflowOptions.emplace_back("digits", VariantType::Bool, false, ConfigParamSpec::HelpString{"Write associated digits"});
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(const ConfigContext& config)
{
  bool useMC = config.options().get<bool>("enable-mc");
  bool digits = config.options().get<bool>("digits");
  return WorkflowSpec{o2::mch::getTrackWriterSpec(useMC, "mch-track-writer", "mchtracks.root", digits)};
}
