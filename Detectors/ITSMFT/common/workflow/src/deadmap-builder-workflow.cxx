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

#include "ITSMFTWorkflow/DeadMapBuilderSpec.h"
#include "CommonUtils/ConfigurableParam.h"
#include "Framework/ConfigParamSpec.h"

using namespace o2::framework;

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  std::vector<ConfigParamSpec> options{
    ConfigParamSpec{"runmft", VariantType::Bool, false, {"Expect MFT data"}},
    ConfigParamSpec{"source", VariantType::String, "chipsstatus", {"Loop over: digits, clusters or chipsstatus"}},
    ConfigParamSpec{"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings"}}};

  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"
#include "Framework/Logger.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  LOG(info) << "Initializing O2 ITSMFT Dead Map Builder";

  WorkflowSpec wf;

  bool doMFT = configcontext.options().get<bool>("runmft");
  std::string datasource = configcontext.options().get<std::string>("source");
  std::string detector = doMFT ? "MFT" : "ITS";
  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));
  LOG(info) << "Building " << detector << " deadmaps from collection of:  " << datasource;
  wf.emplace_back(o2::itsmft::getITSMFTDeadMapBuilderSpec(datasource, doMFT));

  return wf;
}
