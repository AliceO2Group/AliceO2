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
#include "Framework/CompletionPolicyHelpers.h"

using namespace o2::framework;

// This program is o2-zdc-digits-writer and it is used to convert CTF data to digits
// to allow manual inspection
// In order to convert raw data to digits you should use o2-zdc-raw2digits

// ------------------------------------------------------------------
// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  std::vector<ConfigParamSpec> options{
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings ..."}}};
  std::swap(workflowOptions, options);
}

void customize(std::vector<o2::framework::CompletionPolicy>& policies)
{
  // ordered policies for the writers
  policies.push_back(CompletionPolicyHelpers::consumeWhenAllOrdered(".*(?:ZDC|zdc).*[W,w]riter.*"));
}

// ------------------------------------------------------------------
#include "Framework/runDataProcessing.h"
#include "Framework/WorkflowSpec.h"
#include "ZDCWorkflow/ZDCDigitWriterDPLSpec.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));
  WorkflowSpec specs;
  specs.emplace_back(o2::zdc::getZDCDigitWriterDPLSpec(false, false));
  return std::move(specs);
}
