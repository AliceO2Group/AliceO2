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

/// \file   MID/Workflow/src/tracks-reader-workflow.cxx
/// \brief  DPL workflow to send MID tracks read from a root file
/// \author Philippe Pillot, Subatech

#include <vector>
#include "Framework/ConfigParamSpec.h"
#include "Framework/CallbacksPolicy.h"
#include "MIDWorkflow/TrackReaderSpec.h"
#include "CommonUtils/ConfigurableParam.h"
#include "DetectorsRaw/HBFUtilsInitializer.h"

using namespace o2::framework;

void customize(std::vector<o2::framework::CallbacksPolicy>& policies)
{
  o2::raw::HBFUtilsInitializer::addNewTimeSliceCallback(policies);
}

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  std::vector<ConfigParamSpec> options{{"disable-mc", VariantType::Bool, false, ConfigParamSpec::HelpString{"Disable MC info"}},
                                       {"configKeyValues", VariantType::String, "", ConfigParamSpec::HelpString{"Semicolon separated key=value strings ..."}}};
  o2::raw::HBFUtilsInitializer::addConfigOption(options);
  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(const ConfigContext& config)
{
  bool useMC = !config.options().get<bool>("disable-mc");
  WorkflowSpec specs{o2::mid::getTrackReaderSpec(useMC)};
  // configure dpl timer to inject correct firstTForbit: start from the 1st orbit of TF containing 1st sampled orbit
  o2::raw::HBFUtilsInitializer hbfIni(config, specs);
  return specs;
}
