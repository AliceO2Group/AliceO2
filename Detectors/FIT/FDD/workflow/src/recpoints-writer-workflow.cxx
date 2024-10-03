// Copyright 2019-2024 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file  recpoints-writer-workflow.cxx
/// \brief FDD RecPoints writer workflow
///
/// \author Andreas Molander andreas.molander@cern.ch

#include "FDDWorkflow/RecPointWriterSpec.h"

#include "CommonUtils/ConfigurableParam.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/Variant.h"

#include <vector>

using namespace o2::framework;

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  std::vector<ConfigParamSpec> options{
    {"disable-mc", VariantType::Bool, false, {"disable MC propagation even if available"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings"}}};
  workflowOptions.insert(workflowOptions.end(), options.begin(), options.end());
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(const ConfigContext& ctx)
{
  o2::conf::ConfigurableParam::updateFromString(ctx.options().get<std::string>("configKeyValues"));
  bool disableMC = ctx.options().get<bool>("disable-mc");

  WorkflowSpec specs;
  DataProcessorSpec producer = o2::fdd::getFDDRecPointWriterSpec(!disableMC);
  specs.push_back(producer);
  return specs;
}
