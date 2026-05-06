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
#include "Framework/ConfigParamRegistry.h"
#include "Framework/CompletionPolicy.h"
#include "Framework/CompletionPolicyHelpers.h"

using namespace o2::framework;

void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  std::vector<ConfigParamSpec> options;
  options.push_back(ConfigParamSpec{"dataspec", VariantType::String, "tst:TST/A", {"selection string for the data to be proxied"}});
  options.push_back(ConfigParamSpec{"triggerspec", VariantType::String, "", {"selection string for the trigger input (must be also in dataspec if non-empty)"}});
  options.push_back(ConfigParamSpec{"configKeyValues", VariantType::String, "", {"semicolon separated key=value strings"}});
  std::swap(workflowOptions, options);
}

void customize(std::vector<CompletionPolicy>& policies)
{
  policies.push_back({CompletionPolicyHelpers::consumeWhenPastOldestPossibleTimeframe("raw-tf-dump", [](auto const&) -> bool { return true; })});
  // policies.push_back({CompletionPolicyHelpers::consumeWhenAllOrdered("raw-tf-dump", [](auto const&) -> bool { return true; })}); // RSTOREM
}

// ------------------------------------------------------------------

#include "Framework/runDataProcessing.h"
#include "RawTFDumpSpec.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));
  auto inpconfig = configcontext.options().get<std::string>("dataspec");
  auto trigger = configcontext.options().get<std::string>("triggerspec");
  WorkflowSpec specs{o2::rawdd::getRawTFDumpSpec(inpconfig, trigger)};
  return specs;
}
