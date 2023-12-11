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

#include "ITSWorkflow/DeadMapBuilderSpec.h"
#include "CommonUtils/ConfigurableParam.h"
#include "ITStracking/TrackingConfigParam.h"
#include "ITStracking/Configuration.h"
#include "Framework/ConfigParamSpec.h"

using namespace o2::framework;

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  std::vector<ConfigParamSpec> options{
    ConfigParamSpec{"chip-mod-selector", VariantType::Int, 0, {"Integer to be used with chip-mod-base for parallel chip access: if(chipID %% chipModSel != chipModBase), chip id skipped"}},
    ConfigParamSpec{"chip-mod-base", VariantType::Int, 1, {"Integer to be used with chip-mod-selector chip access: if(chipID %% chipModSel != chipModBase), chip id skipped"}},
    ConfigParamSpec{"source", VariantType::String, "chipsstatus", {"Loop over: digits, clusters or chipsstatus"}}};

  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"
#include "Framework/Logger.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  LOG(info) << "Initializing O2 ITS Dead Map Builder";

  WorkflowSpec wf;
  o2::its::ITSDMInpConf inpConf;
  inpConf.chipModSel = configcontext.options().get<int>("chip-mod-selector");
  inpConf.chipModBase = configcontext.options().get<int>("chip-mod-base");

  std::string datasource = configcontext.options().get<std::string>("source");

  LOG(info) << "Building deadmaps from collection of:  " << datasource;

  wf.emplace_back(o2::its::getITSDeadMapBuilderSpec(inpConf, datasource));

  return wf;
}
