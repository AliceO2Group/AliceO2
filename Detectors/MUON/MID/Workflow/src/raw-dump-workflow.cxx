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

/// \file   MID/Workflow/src/raw-to-digits-workflow.cxx
/// \brief  MID raw to digits workflow
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   22 September 2020

#include <string>
#include <vector>
#include "Framework/Variant.h"
#include "Framework/ConfigParamSpec.h"
#include "CommonUtils/ConfigurableParam.h"
#include "MIDWorkflow/RawDumpSpec.h"

using namespace o2::framework;

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  std::vector<ConfigParamSpec> options{
    {"feeId-config-file", VariantType::String, "", {"Filename with crate FEE ID correspondence"}},
    {"crate-masks-file", VariantType::String, "", {"Filename with crate masks"}},
    {"electronics-delay-file", VariantType::String, "", {"Filename with electronics delay"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings"}},
    {"ignore-dist-stf", VariantType::Bool, false, {"do not subscribe to FLP/DISTSUBTIMEFRAME/0 message (no lost TF recovery)"}}};
  workflowOptions.insert(workflowOptions.end(), options.begin(), options.end());
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  o2::conf::ConfigurableParam::updateFromString(cfgc.options().get<std::string>("configKeyValues"));
  auto feeIdConfigFilename = cfgc.options().get<std::string>("feeId-config-file");
  o2::mid::FEEIdConfig feeIdConfig;
  if (!feeIdConfigFilename.empty()) {
    feeIdConfig = o2::mid::FEEIdConfig(feeIdConfigFilename.c_str());
  }
  auto crateMasksFilename = cfgc.options().get<std::string>("crate-masks-file");
  o2::mid::CrateMasks crateMasks;
  if (!crateMasksFilename.empty()) {
    crateMasks = o2::mid::CrateMasks(crateMasksFilename.c_str());
  }
  auto electronicsDelayFilename = cfgc.options().get<std::string>("electronics-delay-file");
  o2::mid::ElectronicsDelay electronicsDelay;
  if (!electronicsDelayFilename.empty()) {
    electronicsDelay = o2::mid::readElectronicsDelay(electronicsDelayFilename.c_str());
  }

  auto askDISTSTF = !cfgc.options().get<bool>("ignore-dist-stf");

  o2::framework::WorkflowSpec specs;
  specs.emplace_back(o2::mid::getRawDumpSpec(true, feeIdConfig, crateMasks, electronicsDelay, askDISTSTF));
  return specs;
}
