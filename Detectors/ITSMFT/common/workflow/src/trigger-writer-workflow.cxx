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

/// @brief  Processor spec for a ROOT file writer for ITSMFT digits

#include "ITSMFTWorkflow/DigitWriterSpec.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include <DataFormatsITSMFT/PhysTrigger.h>
#include "Headers/DataHeader.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "CommonUtils/ConfigurableParam.h"
#include <vector>
#include <string>
#include <algorithm>

using namespace o2::framework;
using SubSpecificationType = o2::framework::DataAllocator::SubSpecificationType;
using DetID = o2::detectors::DetID;

namespace o2
{
namespace itsmft
{

template <typename T>
using BranchDefinition = MakeRootTreeWriterSpec::BranchDefinition<T>;

/// create the processor spec
/// describing a processor receiving physics triggers words from ITS/MFT decoder and writing them to file
DataProcessorSpec getPhyTrigWriterSpec(DetID detId)
{
  std::string detStr = DetID::getName(detId);
  std::string detStrL = "o2_";
  detStrL += detStr;
  std::transform(detStrL.begin(), detStrL.end(), detStrL.begin(), ::tolower);
  auto logger = [](std::vector<o2::itsmft::PhysTrigger> const& inp) {
    LOG(info) << "Received " << inp.size() << " triggers";
  };

  return MakeRootTreeWriterSpec((detStr + "phytrigwriter").c_str(),
                                (detStrL + "phy-triggers.root").c_str(),
                                MakeRootTreeWriterSpec::TreeAttributes{"o2sim", "Physics triggers tree"},
                                BranchDefinition<std::vector<o2::itsmft::PhysTrigger>>{InputSpec{"trig", detId == DetID::ITS ? "ITS" : "MFT", "PHYSTRIG", 0}, (detStr + "Trig").c_str()})();
}

} // end namespace itsmft
} // end namespace o2

#include "Framework/ConfigParamSpec.h"
#include "Framework/CompletionPolicyHelpers.h"

// ------------------------------------------------------------------
void customize(std::vector<o2::framework::CompletionPolicy>& policies)
{
  // ordered policies for the writers
  policies.push_back(CompletionPolicyHelpers::consumeWhenAllOrdered(".*phytrigwriter.*"));
}

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  std::vector<ConfigParamSpec> options{
    ConfigParamSpec{"runmft", VariantType::Bool, false, {"expect MFT data"}},
    ConfigParamSpec{"configKeyValues", VariantType::String, "", {"semicolon separated key=value strings"}}};

  std::swap(workflowOptions, options);
}

// ------------------------------------------------------------------

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  WorkflowSpec wf;
  o2::conf::ConfigurableParam::updateFromString(cfgc.options().get<std::string>("configKeyValues"));
  wf.emplace_back(o2::itsmft::getPhyTrigWriterSpec(cfgc.options().get<bool>("runmft") ? DetID::MFT : DetID::ITS));
  return wf;
}
