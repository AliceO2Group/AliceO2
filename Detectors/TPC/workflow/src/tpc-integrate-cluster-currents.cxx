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

/// \file   tpc-integrate-cluster-currents.cxx
/// \author Matthias Kleiner, mkleiner@ikf.uni-frankfurt.de

#include "TPCWorkflow/TPCIntegrateClusterSpec.h"
#include "TPCWorkflow/TPCIntegrateClusterWriterSpec.h"
#include "CommonUtils/ConfigurableParam.h"
#include "TPCReaderWorkflow/TPCSectorCompletionPolicy.h"
#include "Framework/ConfigParamSpec.h"

using namespace o2::framework;

// customize the completion policy
void customize(std::vector<o2::framework::CompletionPolicy>& policies)
{
  policies.push_back(o2::tpc::TPCSectorCompletionPolicy("TPCIntegrateClusters",
                                                        o2::tpc::TPCSectorCompletionPolicy::Config::RequireAll,
                                                        InputSpec{"cluster", ConcreteDataTypeMatcher{"TPC", "CLUSTERNATIVE"}})());
}

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  std::vector<ConfigParamSpec> options{
    ConfigParamSpec{"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings"}},
    {"disable-root-output", VariantType::Bool, false, {"disable root-files output writers"}}};

  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& config)
{
  WorkflowSpec workflow;
  o2::conf::ConfigurableParam::updateFromString(config.options().get<std::string>("configKeyValues"));
  const bool disableWriter = config.options().get<bool>("disable-root-output");
  workflow.emplace_back(o2::tpc::getTPCIntegrateClusterSpec(disableWriter));
  if (!disableWriter) {
    workflow.emplace_back(o2::tpc::getTPCIntegrateClusterWriterSpec());
  }
  return workflow;
}
