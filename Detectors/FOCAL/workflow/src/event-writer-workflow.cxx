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
#include "Framework/CompletionPolicyHelpers.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "CommonUtils/ConfigurableParam.h"
#include "DataFormatsFOCAL/Event.h"
#include "DataFormatsFOCAL/PixelHit.h"
#include "DataFormatsFOCAL/PixelChipRecord.h"
#include "DataFormatsFOCAL/TriggerRecord.h"

using namespace o2::framework;
using namespace o2::focal;

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  std::vector<ConfigParamSpec> options{{"event-writer-name", VariantType::String, "focal-event-writer", {"Workflow name"}},
                                       {"subspec", VariantType::UInt32, 0U, {"Input subspecification"}},
                                       {"no-subspec", VariantType::Bool, false, {"No subspecification (for output from raw STFs)"}},
                                       {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings"}}};
  workflowOptions.insert(workflowOptions.end(), options.begin(), options.end());
}

void customize(std::vector<o2::framework::CompletionPolicy>& policies)
{
  // ordered policies for the writers
  policies.push_back(CompletionPolicyHelpers::consumeWhenAllOrdered(".*(?:FOC|foc).*[W,w]riter.*"));
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  auto subspec = cfgc.options().get<uint32_t>("subspec");
  auto workflowname = cfgc.options().get<std::string>("event-writer-name");
  o2::conf::ConfigurableParam::updateFromString(cfgc.options().get<std::string>("configKeyValues"));

  WorkflowSpec specs;
  if (cfgc.options().get<bool>("no-subspec")) {
    // No subspecification defined (i.e. output from the raw-tf-reader workflow): Use concrete data type matcher
    specs.emplace_back(MakeRootTreeWriterSpec(workflowname.data(), "focalevents.root", "o2sim",
                                              MakeRootTreeWriterSpec::BranchDefinition<std::vector<PadLayerEvent>>{InputSpec{"dataapd", o2::framework::ConcreteDataTypeMatcher{"FOC", "PADLAYERS"}}, "FOCALPadLayer", "pad-branch-name"},
                                              MakeRootTreeWriterSpec::BranchDefinition<std::vector<PixelHit>>{InputSpec{"datapixelhit", o2::framework::ConcreteDataTypeMatcher{"FOC", "PIXELHITS"}}, "FOCALPixelHit", "pixel-hit-branch-name"},
                                              MakeRootTreeWriterSpec::BranchDefinition<std::vector<PixelChipRecord>>{InputSpec{"datapixelchip", o2::framework::ConcreteDataTypeMatcher{"FOC", "PIXELCHIPS"}}, "FOCALPixelChip", "pixel-chip-branch-name"},
                                              MakeRootTreeWriterSpec::BranchDefinition<std::vector<TriggerRecord>>{InputSpec{"datatrigger", o2::framework::ConcreteDataTypeMatcher{"FOC", "TRIGGERS"}}, "FOCALTrigger", "trigger-branch-name"})());
  } else {
    // Subspecification specified: Use full specification
    specs.emplace_back(MakeRootTreeWriterSpec(workflowname.data(), "focalevents.root", "o2sim",
                                              MakeRootTreeWriterSpec::BranchDefinition<std::vector<PadLayerEvent>>{InputSpec{"dataapd", "FOC", "PADLAYERS", subspec}, "FOCALPadLayer", "pad-branch-name"},
                                              MakeRootTreeWriterSpec::BranchDefinition<std::vector<PixelHit>>{InputSpec{"datapixelhit", "FOC", "PIXELHITS", subspec}, "FOCALPixelHit", "pixel-hit-branch-name"},
                                              MakeRootTreeWriterSpec::BranchDefinition<std::vector<PixelChipRecord>>{InputSpec{"datapixelchip", "FOC", "PIXELCHIPS", subspec}, "FOCALPixelChip", "pixel-chip-branch-name"},
                                              MakeRootTreeWriterSpec::BranchDefinition<std::vector<TriggerRecord>>{InputSpec{"datatrigger", "FOC", "TRIGGERS", subspec}, "FOCALTrigger", "trigger-branch-name"})());
  }
  return specs;
}
