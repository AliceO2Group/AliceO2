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

#include "Algorithm/RangeTokenizer.h"
#include "CommonUtils/ConfigurableParam.h"
#include "DetectorsRaw/HBFUtilsInitializer.h"
#include "Framework/CallbacksPolicy.h"
#include "Framework/CompletionPolicyHelpers.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/WorkflowSpec.h"

#include "EMCALWorkflow/EventBuilderSpec.h"
#include "EMCALWorkflow/RawToCellConverterSpec.h"

void customize(std::vector<o2::framework::CallbacksPolicy>& policies)
{
  o2::raw::HBFUtilsInitializer::addNewTimeSliceCallback(policies);
}

// add workflow options, note that customization needs to be declared before
// including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  std::vector<o2::framework::ConfigParamSpec> options{
    {"configKeyValues", o2::framework::VariantType::String, "", {"Semicolon separated key=value strings"}},
    {"disable-decoding-errors", o2::framework::VariantType::Bool, false, {"disable propagating decoding errors"}},
    {"ignore-dist-stf", o2::framework::VariantType::Bool, false, {"do not subscribe to FLP/DISTSUBTIMEFRAME/0 message (no lost TF recovery)"}}};
  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h" // the main driver

o2::framework::WorkflowSpec defineDataProcessing(o2::framework::ConfigContext const& cfgc)
{
  o2::conf::ConfigurableParam::updateFromString(cfgc.options().get<std::string>("configKeyValues"));

  o2::framework::WorkflowSpec wf;

  bool askdiststf = !cfgc.options().get<bool>("ignore-dist-stf");
  bool disableDecodingErrors = cfgc.options().get<bool>("disable-decoding-errors");

  // Run 2 raw to cell converters, each one mimicing a single FLP (0 - EMCAL, 1 - DCAL)
  wf.emplace_back(o2::emcal::reco_workflow::getRawToCellConverterSpecFLP(askdiststf, disableDecodingErrors, 146));
  wf.emplace_back(o2::emcal::reco_workflow::getRawToCellConverterSpecFLP(askdiststf, disableDecodingErrors, 147));

  // Run event builder reading cells from the two converters and combining to single events
  wf.emplace_back(o2::emcal::reco_workflow::getEventBuilderSpec());

  // configure dpl timer to inject correct firstTFOrbit: start from the 1st orbit of TF containing 1st sampled orbit
  o2::raw::HBFUtilsInitializer hbfIni(cfgc, wf);

  return std::move(wf);
}