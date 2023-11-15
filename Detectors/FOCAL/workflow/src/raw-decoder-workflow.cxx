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

#include <string>
#include <vector>
#include "Framework/Variant.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/CallbacksPolicy.h"
#include "DetectorsRaw/HBFUtilsInitializer.h"
#include "FOCALWorkflow/RawDecoderSpec.h"
#include "CommonUtils/ConfigurableParam.h"

using namespace o2::framework;
using namespace o2::focal;

void customize(std::vector<o2::framework::CallbacksPolicy>& policies)
{
  o2::raw::HBFUtilsInitializer::addNewTimeSliceCallback(policies);
}

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  std::vector<ConfigParamSpec> options{
    {"output-subspec", VariantType::UInt32, 0U, {"Subspecification for output objects"}},
    {"askdistsft", VariantType::Bool, false, {"Subscribe to FLP/DISTSUBTIMEFRAME"}},
    {"no-pads", VariantType::Bool, false, {"Disable handling of pad data"}},
    {"no-pixels", VariantType::Bool, false, {"Disable handling of pixel data"}},
    {"debugmode", VariantType::Bool, false, {"Run dedicated debug code"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings"}}};
  o2::raw::HBFUtilsInitializer::addConfigOption(options);
  workflowOptions.insert(workflowOptions.end(), options.begin(), options.end());
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{

  bool disablePads = cfgc.options().get<bool>("no-pads"),
       disablePixels = cfgc.options().get<bool>("no-pixels"),
       askdiststf = cfgc.options().get<bool>("askdistsft"),
       debugmode = cfgc.options().get<bool>("debugmode");
  int outputSubspec = cfgc.options().get<int>("output-subspec");

  o2::conf::ConfigurableParam::updateFromString(cfgc.options().get<std::string>("configKeyValues"));

  WorkflowSpec specs;
  specs.emplace_back(o2::focal::reco_workflow::getRawDecoderSpec(askdiststf, outputSubspec, !disablePads, !disablePixels, debugmode));

  // configure dpl timer to inject correct firstTForbit: start from the 1st orbit of TF containing 1st sampled orbit
  o2::raw::HBFUtilsInitializer hbfIni(cfgc, specs);
  return specs;
}