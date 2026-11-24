// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

// Workflow plugin built in a DSO, loaded by test_WorkflowPlugins
// together with test_WorkflowPluginA.
#include "Framework/ConfigParamSpec.h"
#include <vector>

void customize(std::vector<o2::framework::ConfigParamSpec>& options)
{
  options.push_back({"shared", o2::framework::VariantType::Int, 1, {"option declared by both plugins"}});
  options.push_back({"b-only", o2::framework::VariantType::String, "b", {"option declared only by plugin B"}});
}

#include "Framework/ControlService.h"
#include "Framework/runDataProcessing.h"

using namespace o2::framework;

WorkflowSpec defineDataProcessing(ConfigContext const& context)
{
  auto shared = context.options().get<int>("shared");
  auto expected = context.options().get<int>("a-only") + 1;
  return {DataProcessorSpec{
    .name = "plugin-consumer",
    .inputs = {InputSpec{"value", "TST", "PLUGIN", 0}},
    .algorithm = AlgorithmSpec{adaptStateless([shared, expected](InputRecord& inputs, ControlService& control) {
      auto value = inputs.get<int>("value");
      // The test passes --shared 3 and --a-only 2: both plugins see the same value of the
      // shared option and plugin B sees the options of plugin A, since they are parsed once.
      if (value != shared || shared != expected) {
        LOGP(fatal, "Got {} from plugin A, {} in plugin B, {} expected", value, shared, expected);
      }
      LOGP(info, "Got {} from plugin A, as expected", value);
      control.readyToQuit(QuitRequest::All);
    })}}};
}
