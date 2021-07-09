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

#include "TRDWorkflow/TRDTrackletTransformerSpec.h"
#include "TRDWorkflowIO/TRDCalibratedTrackletWriterSpec.h"
#include "TRDWorkflowIO/TRDTrackletReaderSpec.h"
#include "GlobalTrackingWorkflowHelpers/InputHelper.h"

#include "CommonUtils/ConfigurableParam.h"
#include "Framework/CompletionPolicy.h"

using namespace o2::framework;

void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  std::vector<o2::framework::ConfigParamSpec> options{
    {"disable-root-input", o2::framework::VariantType::Bool, false, {"disable root-files input reader"}},
    {"disable-root-output", o2::framework::VariantType::Bool, false, {"disable root-files output writer"}},
    {"filter-trigrec", o2::framework::VariantType::Bool, false, {"ignore interaction records without ITS data"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings ..."}}};

  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));

  auto trigRecFilterActive = configcontext.options().get<bool>("filter-trigrec");

  WorkflowSpec spec;

  if (!configcontext.options().get<bool>("disable-root-input")) {
    // cannot use InputHelper here, since we have to create the calibrated tracklets first
    spec.emplace_back(o2::trd::getTRDTrackletReaderSpec(false, false));
  }

  if (trigRecFilterActive) {
    o2::globaltracking::InputHelper::addInputSpecsIRFramesITS(configcontext, spec);
  }

  spec.emplace_back(o2::trd::getTRDTrackletTransformerSpec(trigRecFilterActive));

  if (!configcontext.options().get<bool>("disable-root-output")) {
    spec.emplace_back(o2::trd::getTRDCalibratedTrackletWriterSpec());
  }

  return spec;
}
