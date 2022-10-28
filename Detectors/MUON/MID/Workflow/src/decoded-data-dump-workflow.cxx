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

/// \file   MID/Workflow/src/raw-dump-workflow.cxx
/// \brief  MID raw dump workflow
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   17 February 2022

#include <string>
#include <vector>
#include "Framework/Variant.h"
#include "Framework/ConfigParamSpec.h"
#include "MIDRaw/CrateMasks.h"
#include "MIDRaw/ElectronicsDelay.h"
#include "MIDRaw/FEEIdConfig.h"
#include "MIDWorkflow/RawDumpSpec.h"
#include "MIDWorkflow/RawDecoderSpec.h"

using namespace o2::framework;

// add workflow options, note that customization needs to be declared before
// including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  std::vector<ConfigParamSpec>
    options{
      {"feeId-config-file", VariantType::String, "", {"Filename with crate FEE ID correspondence"}},
      {"crate-masks-file", VariantType::String, "", {"Filename with crate masks"}},
      {"electronics-delay-file", VariantType::String, "", {"Filename with electronics delay"}}};
  workflowOptions.insert(workflowOptions.end(), options.begin(), options.end());
}

#include "Framework/runDataProcessing.h"

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
  auto electronicsDelayFilename = cfgc.options().get<std::string>("electronics-delay-file");
  o2::mid::ElectronicsDelay electronicsDelay;
  if (!electronicsDelayFilename.empty()) {
    electronicsDelay = o2::mid::readElectronicsDelay(electronicsDelayFilename.c_str());
  }

  WorkflowSpec specs;
  specs.emplace_back(o2::mid::getRawDecoderSpec(true, feeIdConfig, crateMasks, electronicsDelay, false));
  specs.emplace_back(o2::mid::getRawDumpSpec());
  return specs;
}
