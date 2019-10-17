// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/WorkflowSpec.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/ControlService.h"
#include "Framework/Logger.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/ExternalFairMQDeviceProxy.h"
#include <vector>

using namespace o2::framework;

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  workflowOptions.push_back(
    ConfigParamSpec{
      "dataspec", VariantType::String, "A:FLP/RAWDATA;B:FLP/DISTSUBTIMEFRAME/0", {"selection string for the data to be proxied"}});

  workflowOptions.push_back(
    ConfigParamSpec{
      "throwOnUnmatched", VariantType::Bool, false, {"throw if unmatched input data is found"}});
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& config)
{
  std::string outputconfig = config.options().get<std::string>("dataspec");
  bool throwOnUnmatched = config.options().get<bool>("throwOnUnmatched");
  std::vector<InputSpec> matchers = select(outputconfig.c_str());
  Outputs readoutProxyOutput;
  for (auto const& matcher : matchers) {
    readoutProxyOutput.emplace_back(DataSpecUtils::asOutputSpec(matcher));
  }

  // we use the same specs as filters in the dpl adaptor
  auto filterSpecs = readoutProxyOutput;
  DataProcessorSpec readoutProxy = specifyExternalFairMQDeviceProxy(
    "readout-proxy",
    std::move(readoutProxyOutput),
    "type=pair,method=connect,address=ipc:///tmp/readout-pipe-0,rateLogging=1,transport=shmem",
    dplModelAdaptor(filterSpecs, throwOnUnmatched));

  WorkflowSpec workflow;
  workflow.emplace_back(readoutProxy);
  return workflow;
}
