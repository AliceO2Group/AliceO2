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

/// \file   tpc-scaler.cxx
/// \author Matthias Kleiner, mkleiner@ikf.uni-frankfurt.de

#include "TPCWorkflow/TPCScalerSpec.h"
#include "CommonUtils/ConfigurableParam.h"
#include "Framework/ConfigParamSpec.h"

using namespace o2::framework;

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  std::vector<ConfigParamSpec> options{
    ConfigParamSpec{"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings"}},
    {"enable-M-shape-correction", VariantType::Bool, false, {"Enable M-shape distortion correction"}},
    {"disable-IDC-scalers", VariantType::Bool, false, {"Disable TPC scalers for space-charge distortion fluctuation correction"}}};

  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& config)
{
  WorkflowSpec workflow;
  o2::conf::ConfigurableParam::updateFromString(config.options().get<std::string>("configKeyValues"));
  const auto enableMShape = config.options().get<bool>("enable-M-shape-correction");
  const auto enableIDCs = !config.options().get<bool>("disable-IDC-scalers");
  workflow.emplace_back(o2::tpc::getTPCScalerSpec(enableIDCs, enableMShape));
  return workflow;
}
