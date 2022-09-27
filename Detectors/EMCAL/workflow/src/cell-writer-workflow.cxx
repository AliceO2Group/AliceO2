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
#include "DataFormatsEMCAL/MCLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "CommonUtils/ConfigurableParam.h"
#include "DataFormatsEMCAL/Cell.h"
#include "DataFormatsEMCAL/TriggerRecord.h"

using namespace o2::framework;
using namespace o2::emcal;

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  std::vector<ConfigParamSpec> options{{"disable-mc", VariantType::Bool, false, {"Do not propagate MC labels"}},
                                       {"subspec", VariantType::UInt32, 0U, {"Input subspecification"}},
                                       {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings"}}};
  workflowOptions.insert(workflowOptions.end(), options.begin(), options.end());
}

void customize(std::vector<o2::framework::CompletionPolicy>& policies)
{
  // ordered policies for the writers
  policies.push_back(CompletionPolicyHelpers::consumeWhenAllOrdered(".*(?:EMC|emc).*[W,w]riter.*"));
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  bool disableMC = cfgc.options().get<bool>("disable-mc");
  auto subspec = cfgc.options().get<uint32_t>("subspec");
  o2::conf::ConfigurableParam::updateFromString(cfgc.options().get<std::string>("configKeyValues"));

  WorkflowSpec specs;
  specs.emplace_back(MakeRootTreeWriterSpec("emcal-cells-writer", "emccells.root", "o2sim",
                                            MakeRootTreeWriterSpec::BranchDefinition<std::vector<Cell>>{InputSpec{"data", "EMC", "CELLS", subspec}, "EMCALCell", "cell-branch-name"},
                                            MakeRootTreeWriterSpec::BranchDefinition<std::vector<TriggerRecord>>{InputSpec{"trigger", "EMC", "CELLSTRGR", subspec}, "EMCALCellTRGR", "celltrigger-branch-name"},
                                            MakeRootTreeWriterSpec::BranchDefinition<o2::dataformats::MCTruthContainer<MCLabel>>{InputSpec{"mc", "EMC", "CELLSMCTR", subspec}, "EMCALCellMCTruth", "cellmc-branch-name", disableMC ? 0 : 1})());
  return specs;
}
