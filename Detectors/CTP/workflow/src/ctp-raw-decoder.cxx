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

/// @file   ctp-reco-workflow.cxx
/// @author RL from CPV example
/// @brief  Basic DPL workflow for CTP reconstruction starting from digits
#include "Framework/WorkflowSpec.h"
#include "Framework/ConfigParamSpec.h"
#include "CommonUtils/ConfigurableParam.h"
#include "Framework/CallbacksPolicy.h"

#include <string>
#include <stdexcept>
#include <unordered_map>

// add workflow options, note that customization needs to be declared before
// including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  std::vector<o2::framework::ConfigParamSpec> options{
    {"ignore-dist-stf", o2::framework::VariantType::Bool, false, {"do not subscribe to FLP/DISTSUBTIMEFRAME/0 message (no lost TF recovery)"}},
    {"no-lumi", o2::framework::VariantType::Bool, false, {"do not produce luminosity output"}},
    {"no-digits", o2::framework::VariantType::Bool, false, {"do not produce digits output"}},
    {"disable-root-output", o2::framework::VariantType::Bool, false, {"disable root-files output writer"}},
    {"configKeyValues", o2::framework::VariantType::String, "", {"Semicolon separated key=value strings ..."}}};
  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h" // the main driver
#include "CTPWorkflow/RawDecoderSpec.h"
#include "CTPWorkflowIO/DigitWriterSpec.h"

/// The workflow executable for the stand alone CTP reconstruction workflow
/// - digit and lumi reader
/// This function hooks up the the workflow specifications into the DPL driver.
o2::framework::WorkflowSpec defineDataProcessing(o2::framework::ConfigContext const& cfgc)
{
  o2::framework::WorkflowSpec specs;
  o2::conf::ConfigurableParam::updateFromString(cfgc.options().get<std::string>("configKeyValues"));

  specs.emplace_back(o2::ctp::reco_workflow::getRawDecoderSpec(!cfgc.options().get<bool>("ignore-dist-stf"),
                                                               !cfgc.options().get<bool>("no-digits"),
                                                               !cfgc.options().get<bool>("no-lumi")));
  if (!cfgc.options().get<bool>("disable-root-output")) {
    specs.emplace_back(o2::ctp::getDigitWriterSpec(!cfgc.options().get<bool>("no-lumi")));
  }
  return specs;
}
