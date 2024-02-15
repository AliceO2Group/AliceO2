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

#include "CommonUtils/ConfigurableParam.h"
#include "Framework/ConfigParamSpec.h"
#include "TRDWorkflowIO/TRDDigitReaderSpec.h"
#include "DetectorsRaw/HBFUtilsInitializer.h"

using namespace o2::framework;

// ------------------------------------------------------------------

void customize(std::vector<o2::framework::CallbacksPolicy>& policies)
{
  o2::raw::HBFUtilsInitializer::addNewTimeSliceCallback(policies);
}

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  std::vector<o2::framework::ConfigParamSpec> options{
    {"disable-mc", VariantType::Bool, true, {"disable MC propagation"}},
    {"disable-trigrec", VariantType::Bool, false, {"disable trigger record reading when these come from tracklet reader"}},
    {"digit-subspec", VariantType::Int, 1, {"allow overwriting default output data subspec"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings ..."}}};
  o2::raw::HBFUtilsInitializer::addConfigOption(options);
  std::swap(workflowOptions, options);
}

// ------------------------------------------------------------------

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  // Update the (declared) parameters if changed from the command line
  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));

  auto useMC = !configcontext.options().get<bool>("disable-mc");
  auto sendTriggerRecords = !configcontext.options().get<bool>("disable-trigrec");
  auto dataSubspec = configcontext.options().get<int>("digit-subspec");
  WorkflowSpec specs;
  specs.emplace_back(o2::trd::getTRDDigitReaderSpec(useMC, sendTriggerRecords, dataSubspec));
  o2::raw::HBFUtilsInitializer hbfIni(configcontext, specs);
  return specs;
}
