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

// ------------------------------------------------------------------
// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  std::vector<ConfigParamSpec> options{
    {"use-process", VariantType::Bool, false, {"enable processor for data taking/dumping"}},
    {"dump-blocks-process", VariantType::Bool, false, {"enable dumping of event blocks at processor side"}},
    {"dump-blocks-reader", VariantType::Bool, false, {"enable dumping of event blocks at reader side"}},
    {"disable-root-output", VariantType::Bool, false, {"disable root-files output writers"}},
    {"ignore-dist-stf", VariantType::Bool, false, {"do not subscribe to FLP/DISTSUBTIMEFRAME/0 message (no lost TF recovery)"}},
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
#include "ZDCWorkflow/ZDCDataReaderDPLSpec.h"
#include "ZDCWorkflow/ZDCDigitWriterDPLSpec.h"
#include "ZDCRaw/RawReaderZDC.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  auto useProcessor = configcontext.options().get<bool>("use-process");
  auto dumpProcessor = configcontext.options().get<bool>("dump-blocks-process");
  auto dumpReader = configcontext.options().get<bool>("dump-blocks-reader");
  auto disableRootOut = configcontext.options().get<bool>("disable-root-output");
  auto askSTFDist = true;
  auto notaskSTFDist = configcontext.options().get<bool>("ignore-dist-stf");
  if (notaskSTFDist) {
    LOG(info) << "Not subscribing to FLP/DISTSUBTIMEFRAME/0 message (no lost TF recovery)";
    askSTFDist = false;
  }

  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));
  WorkflowSpec specs;
  specs.emplace_back(o2::zdc::getZDCDataReaderDPLSpec(o2::zdc::RawReaderZDC{dumpReader}, askSTFDist));
  //  if (useProcess) {
  //    specs.emplace_back(o2::zdc::getZDCDataProcessDPLSpec(dumpProcessor));
  //  }
  if (!disableRootOut) {
    specs.emplace_back(o2::zdc::getZDCDigitWriterDPLSpec(false, false));
  }
  return std::move(specs);
}
