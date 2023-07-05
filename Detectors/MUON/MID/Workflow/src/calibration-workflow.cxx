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

/// \file   MID/Workflow/src/calibration-workflow.cxx
/// \brief  MID noise calibration workflow
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   22 February 2022

#include <string>
#include <vector>
#include "Framework/Variant.h"
#include "Framework/ConfigParamSpec.h"

using namespace o2::framework;

void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  std::vector<ConfigParamSpec> options{
    {"feeId-config-file", VariantType::String, "", {"Filename with crate FEE ID correspondence"}},
    {"crate-masks-file", VariantType::String, "", {"Filename with crate masks"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings"}}};
  workflowOptions.insert(workflowOptions.end(), options.begin(), options.end());
}

#include "Framework/runDataProcessing.h"
#include "CommonUtils/ConfigurableParam.h"
#include "MIDWorkflow/CalibDataProcessorSpec.h"
#include "MIDWorkflow/ChannelCalibratorSpec.h"
#include "MIDRaw/CrateMasks.h"
#include "MIDRaw/FEEIdConfig.h"

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
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
  o2::conf::ConfigurableParam::updateFromString(cfgc.options().get<std::string>("configKeyValues"));
  WorkflowSpec specs;
  specs.emplace_back(o2::mid::getCalibDataProcessorSpec(feeIdConfig, crateMasks));
  specs.emplace_back(o2::mid::getChannelCalibratorSpec(feeIdConfig, crateMasks));

  return specs;
}
