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

#include "ClusterTransformerSpec.h"
#include "CommonUtils/ConfigurableParam.h"
#include "DetectorsRaw/HBFUtilsInitializer.h"
#include "DigitFilteringSpec.h"
#include "DigitReaderSpec.h"
#include "Framework/CallbacksPolicy.h"
#include "Framework/ConfigContext.h"
#include "Framework/Logger.h"
#include "Framework/Variant.h"
#include "Framework/WorkflowSpec.h"
#include "MCHWorkflow/ClusterFinderOriginalSpec.h"
#include "MCHWorkflow/PreClusterFinderSpec.h"
#include "MCHWorkflow/TrackWriterSpec.h"
#include "TimeClusterFinderSpec.h"
#include "TrackFinderSpec.h"
#include "TrackFitterSpec.h"
#include "TrackMCLabelFinderSpec.h"

using o2::framework::ConfigContext;
using o2::framework::ConfigParamSpec;
using o2::framework::VariantType;
using o2::framework::WorkflowSpec;

void customize(std::vector<o2::framework::CallbacksPolicy>& policies)
{
  o2::raw::HBFUtilsInitializer::addNewTimeSliceCallback(policies);
}

void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  std::vector<ConfigParamSpec> options{
    {"disable-root-input", o2::framework::VariantType::Bool, false, {"disable root-files input reader"}},
    {"disable-root-output", o2::framework::VariantType::Bool, false, {"do not write output root files"}},
    {"disable-mc", o2::framework::VariantType::Bool, false, {"disable MC propagation even if available"}},
    {"digits", VariantType::Bool, false, {"Write digits associated to tracks"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings"}}};
  o2::raw::HBFUtilsInitializer::addConfigOption(options);
  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  WorkflowSpec specs;

  auto disableRootOutput = configcontext.options().get<bool>("disable-root-output");
  auto disableRootInput = configcontext.options().get<bool>("disable-root-input");
  auto digits = configcontext.options().get<bool>("digits");
  auto useMC = !configcontext.options().get<bool>("disable-mc");

  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));

  if (!disableRootInput) {
    specs.emplace_back(o2::mch::getDigitReaderSpec(useMC, "mch-sim-digit-reader"));
  }

  specs.emplace_back(o2::mch::getDigitFilteringSpec(useMC, "mch-digit-filtering",
                                                    "DIGITS", "F-DIGITS",
                                                    "DIGITROFS", "F-DIGITROFS",
                                                    "DIGITLABELS", "F-DIGITLABELS"));

  if (!useMC) {
    specs.emplace_back(o2::mch::getTimeClusterFinderSpec("mch-time-cluster-finder",
                                                         "F-DIGITS",
                                                         "F-DIGITROFS",
                                                         "TC-F-DIGITROFS"));
  }

  specs.emplace_back(o2::mch::getPreClusterFinderSpec("mch-precluster-finder",
                                                      "F-DIGITS",
                                                      useMC ? "F-DIGITROFS" : "TC-F-DIGITROFS"));

  specs.emplace_back(o2::mch::getClusterFinderOriginalSpec("mch-cluster-finder"));
  specs.emplace_back(o2::mch::getClusterTransformerSpec());
  specs.emplace_back(o2::mch::getTrackFinderSpec("mch-track-finder", digits));
  if (useMC) {
    specs.emplace_back(o2::mch::getTrackMCLabelFinderSpec("mch-track-mc-label-finder",
                                                          "F-DIGITROFS",
                                                          "F-DIGITLABELS"));
  }

  if (!disableRootOutput) {
    specs.emplace_back(o2::mch::getTrackWriterSpec(useMC, "mch-track-writer", "mchtracks.root", digits));
  }

  // configure dpl timer to inject correct firstTFOrbit: start from the 1st orbit of TF containing 1st sampled orbit
  o2::raw::HBFUtilsInitializer hbfIni(configcontext, specs);

  // write the configuration used for the reco workflow
  o2::conf::ConfigurableParam::writeINI("o2mchrecoflow_configuration.ini");

  return specs;
}
