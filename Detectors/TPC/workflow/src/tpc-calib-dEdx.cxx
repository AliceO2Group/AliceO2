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

#include "Framework/ConfigParamSpec.h"
#include "CommonUtils/ConfigurableParam.h"
#include "TPCWorkflow/CalibdEdxSpec.h"

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  std::vector<ConfigParamSpec> options{
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings (e.g.: 'TPCCalibPedestal.FirstTimeBin=10;...')"}},
    {"configFile", VariantType::String, "", {"configuration file for configurable parameters"}},
  };

  std::swap(workflowOptions, options);
}
#include "Framework/runDataProcessing.h"

using namespace o2::framework;

WorkflowSpec defineDataProcessing(ConfigContext const& config)
{
  // set up configuration
  o2::conf::ConfigurableParam::updateFromFile(config.options().get<std::string>("configFile"));
  o2::conf::ConfigurableParam::updateFromString(config.options().get<std::string>("configKeyValues"));
  o2::conf::ConfigurableParam::writeINI("o2tpccalibration_configuration.ini");

  using namespace o2::tpc;
  return WorkflowSpec{getCalibdEdxSpec()};
}
