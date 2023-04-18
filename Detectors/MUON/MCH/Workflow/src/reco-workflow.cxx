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

#include "ClusterFinderGEMSpec.h"
#include "CommonUtils/ConfigurableParam.h"
#include "DetectorsRaw/HBFUtilsInitializer.h"
#include "ErrorMergerSpec.h"
#include "EventFinderSpec.h"
#include "Framework/CallbacksPolicy.h"
#include "Framework/CompletionPolicyHelpers.h"
#include "Framework/ConfigContext.h"
#include "Framework/Logger.h"
#include "Framework/Variant.h"
#include "Framework/WorkflowSpec.h"
#include "MCHClustering/ClusterizerParam.h"
#include "MCHDigitFiltering/DigitFilteringSpec.h"
#include "MCHGeometryTransformer/ClusterTransformerSpec.h"
#include "MCHIO/ClusterWriterSpec.h"
#include "MCHIO/DigitReaderSpec.h"
#include "MCHIO/TrackWriterSpec.h"
#include "MCHPreClustering/PreClusterFinderSpec.h"
#include "MCHStatus/StatusMapCreatorParam.h"
#include "MCHStatus/StatusMapCreatorSpec.h"
#include "MCHTimeClustering/TimeClusterFinderSpec.h"
#include "MCHTracking/TrackFinderSpec.h"
#include "MCHWorkflow/ClusterFinderOriginalSpec.h"
#include "MCHWorkflow/ErrorWriterSpec.h"
#include "TrackMCLabelFinderSpec.h"

using namespace o2::framework;

void customize(std::vector<o2::framework::CallbacksPolicy>& policies)
{
  o2::raw::HBFUtilsInitializer::addNewTimeSliceCallback(policies);
}

void customize(std::vector<o2::framework::CompletionPolicy>& policies)
{
  // ordered policies for the writers
  policies.push_back(CompletionPolicyHelpers::consumeWhenAllOrdered(".*(?:MCH|mch).*[W,w]riter.*"));
}

void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  std::vector<ConfigParamSpec> options{
    {"disable-root-input", o2::framework::VariantType::Bool, false, {"disable root-files input reader"}},
    {"disable-root-output", o2::framework::VariantType::Bool, false, {"do not write output root files"}},
    {"disable-mc", o2::framework::VariantType::Bool, false, {"disable MC propagation even if available"}},
    {"enable-clusters-root-output", o2::framework::VariantType::Bool, false, {"write output root file containing all clusters"}},
    {"disable-clustering", o2::framework::VariantType::Bool, false, {"disable clustering (and tracking) steps (for debug)"}},
    {"disable-tracking", o2::framework::VariantType::Bool, false, {"disable tracking step (for debug)"}},
    {"digits", VariantType::Bool, false, {"Write digits associated to tracks"}},
    {"triggered", VariantType::Bool, false, {"use MID to trigger the MCH reconstruction"}},
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
  auto triggered = configcontext.options().get<bool>("triggered");
  auto useMC = !configcontext.options().get<bool>("disable-mc");
  auto disableClustering = configcontext.options().get<bool>("disable-clustering");
  auto disableTracking = disableClustering || configcontext.options().get<bool>("disable-tracking");
  auto disableAllClustersRootOutput = !configcontext.options().get<bool>("enable-clusters-root-output");

  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));

  if (!disableRootInput) {
    specs.emplace_back(o2::mch::getDigitReaderSpec(useMC, "mch-digit-reader"));
  }

  if (o2::mch::StatusMapCreatorParam::Instance().isActive()) {
    specs.emplace_back(o2::mch::getStatusMapCreatorSpec("mch-statusmap-creator"));
  }

  specs.emplace_back(o2::mch::getDigitFilteringSpec(useMC, "mch-digit-filtering",
                                                    "DIGITS", "F-DIGITS",
                                                    "DIGITROFS", "F-DIGITROFS",
                                                    "DIGITLABELS", "F-DIGITLABELS"));

  if (triggered) {
    specs.emplace_back(o2::mch::getEventFinderSpec(useMC, "mch-event-finder",
                                                   "F-DIGITS", "E-F-DIGITS",
                                                   "F-DIGITROFS", "E-F-DIGITROFS",
                                                   "F-DIGITLABELS", "E-F-DIGITLABELS"));
  } else {
    specs.emplace_back(o2::mch::getTimeClusterFinderSpec("mch-time-cluster-finder",
                                                         "F-DIGITS",
                                                         "F-DIGITROFS",
                                                         "TC-F-DIGITROFS"));
  }

  specs.emplace_back(o2::mch::getPreClusterFinderSpec("mch-precluster-finder",
                                                      triggered ? "E-F-DIGITS" : "F-DIGITS",
                                                      triggered ? "E-F-DIGITROFS" : "TC-F-DIGITROFS"));

  if (!disableClustering) {
    if (o2::mch::ClusterizerParam::Instance().legacy) {
      specs.emplace_back(o2::mch::getClusterFinderOriginalSpec("mch-cluster-finder"));
    } else {
      specs.emplace_back(o2::mch::getClusterFinderGEMSpec("mch-cluster-finder"));
    }
    specs.emplace_back(o2::mch::getClusterTransformerSpec("mch-cluster-transformer", false));
    if (!disableRootOutput && !disableAllClustersRootOutput) {
      specs.emplace_back(o2::mch::getClusterWriterSpec(false, "mch-global-cluster-writer", true, true));
    }
    if (!disableTracking) {
      specs.emplace_back(o2::mch::getTrackFinderSpec("mch-track-finder", true, digits, false, false));
      if (useMC) {
        specs.emplace_back(o2::mch::getTrackMCLabelFinderSpec("mch-track-mc-label-finder",
                                                              triggered ? "E-F-DIGITROFS" : "TC-F-DIGITROFS",
                                                              triggered ? "E-F-DIGITLABELS" : "F-DIGITLABELS"));
      }

      if (!disableRootOutput) {
        specs.emplace_back(o2::mch::getTrackWriterSpec(useMC, "mch-track-writer", "mchtracks.root", digits));
      }
    }
  }

  specs.emplace_back(o2::mch::getErrorMergerSpec("mch-error-merger", true, !disableClustering,
                                                 !(disableClustering || disableTracking)));
  if (!disableRootOutput) {
    specs.emplace_back(o2::mch::getErrorWriterSpec("mch-error-writer"));
  }

  // configure dpl timer to inject correct firstTForbit: start from the 1st orbit of TF containing 1st sampled orbit
  o2::raw::HBFUtilsInitializer hbfIni(configcontext, specs);

  // write the configuration used for the reco workflow
  o2::conf::ConfigurableParam::writeINI("o2mchrecoflow_configuration.ini");

  return specs;
}
