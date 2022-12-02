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

#include <vector>
#include <string>
#include "Framework/WorkflowSpec.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/CompletionPolicy.h"
#include "CommonUtils/ConfigurableParam.h"
#include "TPCWorkflow/TPCIntegrateClusterCurrent.h"
#include "TPCReaderWorkflow/TPCSectorCompletionPolicy.h"

using namespace o2::framework;

// customize the completion policy
void customize(std::vector<o2::framework::CompletionPolicy>& policies)
{
  policies.push_back(o2::tpc::TPCSectorCompletionPolicy("calib-tpc-gainmap-tracks",
                                                        o2::tpc::TPCSectorCompletionPolicy::Config::RequireAll,
                                                        InputSpec{"cluster", ConcreteDataTypeMatcher{"TPC", "CLUSTERNATIVE"}})());
}

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  std::vector<ConfigParamSpec> options{
    {"configFile", VariantType::String, "", {"configuration file for configurable parameters"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings"}}};

  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& config)
{
  using namespace o2::tpc;

  // set up configuration
  o2::conf::ConfigurableParam::updateFromFile(config.options().get<std::string>("configFile"));
  o2::conf::ConfigurableParam::updateFromString(config.options().get<std::string>("configKeyValues"));
  o2::conf::ConfigurableParam::writeINI("o2tpcintegrateclusters_configuration.ini");

  WorkflowSpec workflow{getTPCIntegrateClustersSpec()};
  return workflow;
}
