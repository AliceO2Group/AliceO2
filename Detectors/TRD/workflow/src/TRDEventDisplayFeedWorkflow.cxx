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

#include "TRDWorkflowIO/TRDDigitReaderSpec.h"
#include "TRDWorkflowIO/TRDTrackReaderSpec.h"
#include "TRDWorkflow/TRDEventDisplayFeedSpec.h"
#include "TRDWorkflowIO/TRDTrackletReaderSpec.h"

#include "CommonUtils/ConfigurableParam.h"

using namespace o2::framework;

void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  std::vector<o2::framework::ConfigParamSpec> options{
    {"disable-root-input", o2::framework::VariantType::Bool, false, {"disable root-files input reader"}},
    {"nEventsMax", o2::framework::VariantType::Int, 5, {"Number of events to display"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings ..."}}};

  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));

  WorkflowSpec spec;

  bool useMC = true;
  if (!configcontext.options().get<bool>("disable-root-input")) {
    spec.emplace_back(o2::trd::getTRDGlobalTrackReaderSpec(useMC));
    spec.emplace_back(o2::trd::getTRDTrackletReaderSpec(useMC, false));
    spec.emplace_back(o2::trd::getTRDDigitReaderSpec(useMC));
  }

  int nEventsMax = configcontext.options().get<int>("nEventsMax");
  spec.emplace_back(o2::trd::getTRDEventDisplayFeedSpec(nEventsMax));

  return spec;
}
