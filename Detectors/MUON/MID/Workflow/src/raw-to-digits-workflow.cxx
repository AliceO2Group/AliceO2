// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
#include "MIDWorkflow/RawDecoderSpec.h"
#include "MIDWorkflow/RawAggregatorSpec.h"

using namespace o2::framework;

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  std::vector<ConfigParamSpec>
    options{
      {"feeId-config-file", VariantType::String, "", {"Filename with crate FEE ID correspondence"}},
      {"crate-masks-file", VariantType::String, "", {"Filename with crate masks"}},
      {"electronics-delay-file", VariantType::String, "", {"Filename with electronics delay"}},
      {"bare", VariantType::Bool, false, {"Is bare decoder"}},
      {"decode-only", o2::framework::VariantType::Bool, false, {"Output decoded boards instead of digits"}}};
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

  bool isBare = cfgc.options().get<bool>("bare");
  bool decodeOnly = cfgc.options().get<bool>("decode-only");

  o2::framework::WorkflowSpec specs;
  specs.emplace_back(o2::mid::getRawDecoderSpec(isBare, false, feeIdConfig, crateMasks, electronicsDelay));
  if (!decodeOnly) {
    specs.emplace_back(o2::mid::getRawAggregatorSpec());
  }
  return specs;
}
