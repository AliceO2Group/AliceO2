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
// together with test_WorkflowPluginB.
#include "Framework/ConfigParamSpec.h"
#include <vector>

void customize(std::vector<o2::framework::ConfigParamSpec>& options)
{
  options.push_back({"shared", o2::framework::VariantType::Int, 1, {"option declared by both plugins"}});
  options.push_back({"a-only", o2::framework::VariantType::Int, 2, {"option declared only by plugin A"}});
}

#include "Framework/ControlService.h"
#include "Framework/runDataProcessing.h"

using namespace o2::framework;

WorkflowSpec defineDataProcessing(ConfigContext const& context)
{
  auto shared = context.options().get<int>("shared");
  return {DataProcessorSpec{
    .name = "plugin-producer",
    .outputs = {OutputSpec{{"value"}, "TST", "PLUGIN", 0}},
    .algorithm = AlgorithmSpec{adaptStateless([shared](DataAllocator& outputs, ControlService& control) {
      outputs.make<int>(OutputRef{"value"}) = shared;
      control.endOfStream();
      control.readyToQuit(QuitRequest::Me);
    })}}};
}
