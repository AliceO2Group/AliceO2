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

#include "ZDCWorkflow/ZDCRecoWriterDPLSpec.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/CompletionPolicyHelpers.h"

using namespace o2::framework;

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  workflowOptions.push_back(ConfigParamSpec{"disable-mc", o2::framework::VariantType::Bool, false, {"disable MC propagation even if available"}});
  workflowOptions.push_back(ConfigParamSpec{"output", o2::framework::VariantType::String, "zdcreco.root", {"output file"}});
}

void customize(std::vector<o2::framework::CompletionPolicy>& policies)
{
  // ordered policies for the writers
  policies.push_back(CompletionPolicyHelpers::consumeWhenAllOrdered(".*(?:ZDC|zdc).*[W,w]riter.*"));
}

#include "Framework/runDataProcessing.h"
#include "Framework/Logger.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  auto useMC = !configcontext.options().get<bool>("disable-mc");
  if (useMC) {
    LOG(warning) << "ZDC reconstruction does not support MC labels at the moment";
  }
  auto output = configcontext.options().get<std::string>("output");
  WorkflowSpec specs{o2::zdc::getZDCRecoWriterDPLSpec(output)};
  return std::move(specs);
}
