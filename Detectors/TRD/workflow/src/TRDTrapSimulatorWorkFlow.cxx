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

#include "DetectorsBase/Propagator.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/CompletionPolicy.h"
#include "Framework/DeviceSpec.h"
#include "Algorithm/RangeTokenizer.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "CommonUtils/ConfigurableParam.h"
#include "Framework/CompletionPolicyHelpers.h"

// for TRD
#include "TRDWorkflow/TRDTrapSimulatorSpec.h"
#include "TRDWorkflowIO/TRDTrackletWriterSpec.h"
#include "TRDWorkflowIO/TRDDigitReaderSpec.h"

#include "DataFormatsParameters/GRPObject.h"

#include <string>

using namespace o2::framework;

// ------------------------------------------------------------------
void customize(std::vector<o2::framework::CompletionPolicy>& policies)
{
  // ordered policies for the writers
  policies.push_back(CompletionPolicyHelpers::consumeWhenAllOrdered(".*(?:TRD|trd).*[W,w]riter.*"));
}

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowoptions)
{
  workflowoptions.push_back(ConfigParamSpec{"disable-mc", o2::framework::VariantType::Bool, false, {"Disable MC labels"}});
  workflowoptions.push_back(ConfigParamSpec{"disable-root-input", o2::framework::VariantType::Bool, false, {"Disable root-files input readers"}});
  workflowoptions.push_back(ConfigParamSpec{"disable-root-output", o2::framework::VariantType::Bool, false, {"Disable root-files output writers"}});
  workflowoptions.push_back(ConfigParamSpec{"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings (e.g.: 'TRDSimParams.digithreads=4;...')"}});
}

#include "Framework/runDataProcessing.h"

/// This function is required to be implemented to define the workflow
/// specifications
WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  using namespace o2::conf;
  ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));
  WorkflowSpec specs;
  auto useMC = !configcontext.options().get<bool>("disable-mc");
  auto disableRootInput = configcontext.options().get<bool>("disable-root-input");
  auto disableRootOutput = configcontext.options().get<bool>("disable-root-output");
  if (!disableRootInput) {
    specs.emplace_back(o2::trd::getTRDDigitReaderSpec(useMC));
  }
  specs.emplace_back(o2::trd::getTRDTrapSimulatorSpec(useMC, 1));
  if (!disableRootOutput) {
    specs.emplace_back(o2::trd::getTRDTrackletWriterSpec(useMC));
  }
  return specs;
}
