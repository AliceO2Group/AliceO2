// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   tpc-reco-workflow.cxx
/// @author Matthias Richter
/// @since  2018-03-15
/// @brief  Basic DPL workflow for TPC reconstruction starting from digits

#include "Framework/WorkflowSpec.h"
#include "Framework/DeviceSpec.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/CompletionPolicyHelpers.h"
#include "Framework/DispatchPolicy.h"
#include "Framework/PartRef.h"
#include "Framework/ConcreteDataMatcher.h"
#include "TPCWorkflow/RecoWorkflow.h"
#include "TPCWorkflow/TPCSectorCompletionPolicy.h"
#include "DataFormatsTPC/TPCSectorHeader.h"
#include "Algorithm/RangeTokenizer.h"
#include "CommonUtils/ConfigurableParam.h"

#include <string>
#include <stdexcept>
#include <unordered_map>
#include <regex>

// we need a global variable to propagate the type the message dispatching of the
// publisher will trigger on. This is dependent on the input type
o2::framework::Output gDispatchTrigger{"", ""};

// Global variable used to transport data to the completion policy
o2::tpc::reco_workflow::CompletionPolicyData gPolicyData;

// add workflow options, note that customization needs to be declared before
// including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  using namespace o2::framework;

  std::vector<ConfigParamSpec> options{
    {"input-type", VariantType::String, "digits", {"digitizer, digits, zsraw, clustershw, clustersnative, compressed-clusters, compressed-clusters-ctf"}},
    {"output-type", VariantType::String, "tracks", {"digits, zsraw, clustershw, clustersnative, tracks, compressed-clusters, encoded-clusters, disable-writer, send-clusters-per-sector, qa, no-shared-cluster-map"}},
    {"no-ca-clusterer", VariantType::Bool, false, {"Use HardwareClusterer instead of clusterer of GPUCATracking"}},
    {"disable-mc", VariantType::Bool, false, {"disable sending of MC information"}},
    //{"tpc-sectors", VariantType::String, "0-35", {"TPC sector range, e.g. 5-7,8,9"}},
    {"tpc-lanes", VariantType::Int, 1, {"number of parallel lanes up to the tracker"}},
    {"dispatching-mode", VariantType::String, "prompt", {"determines when to dispatch: prompt, complete"}},
    {"no-tpc-zs-on-the-fly", VariantType::Bool, false, {"Do not use TPC zero suppression on the fly"}},
    {"zs-threshold", VariantType::Float, 2.0f, {"zero suppression threshold"}},
    {"zs-10bit", VariantType::Bool, false, {"use 10 bit ADCs for TPC zero suppression, default = 12 bit ADC"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings (e.g.: 'TPCHwClusterer.peakChargeThreshold=4;...')"}},
    {"configFile", VariantType::String, "", {"configuration file for configurable parameters"}}};

  std::swap(workflowOptions, options);
}

// customize dispatch policy, dispatch immediately what is ready
void customize(std::vector<o2::framework::DispatchPolicy>& policies)
{
  using DispatchOp = o2::framework::DispatchPolicy::DispatchOp;
  // we customize all devices to dispatch data immediately
  auto readerMatcher = [](auto const& spec) {
    return std::regex_match(spec.name.begin(), spec.name.end(), std::regex(".*-reader"));
  };
  auto triggerMatcher = [](auto const& query) {
    // a bit of a hack but we want this to be configurable from the command line,
    // however DispatchPolicy is inserted before all other setup. Triggering depending
    // on the global variable set from the command line option. If scheduled messages
    // are not triggered they are sent out at the end of the computation
    return gDispatchTrigger.origin == query.origin && gDispatchTrigger.description == query.description;
  };
  policies.push_back({"prompt-for-reader", readerMatcher, DispatchOp::WhenReady, triggerMatcher});
}

// customize clusterers and cluster decoders to process immediately what comes in
void customize(std::vector<o2::framework::CompletionPolicy>& policies)
{
  // we customize the pipeline processors to consume data as it comes
  using CompletionPolicy = o2::framework::CompletionPolicy;
  using CompletionPolicyHelpers = o2::framework::CompletionPolicyHelpers;
  policies.push_back(CompletionPolicyHelpers::defineByName("tpc-cluster-decoder.*", CompletionPolicy::CompletionOp::Consume));
  policies.push_back(CompletionPolicyHelpers::defineByName("tpc-clusterer.*", CompletionPolicy::CompletionOp::Consume));
  // the custom completion policy for the tracker
  policies.push_back(o2::tpc::TPCSectorCompletionPolicy("tpc-tracker.*", o2::tpc::TPCSectorCompletionPolicy::Config::RequireAll, &gPolicyData)());
}

#include "Framework/runDataProcessing.h" // the main driver

using namespace o2::framework;

/// The workflow executable for the stand alone TPC reconstruction workflow
/// The basic workflow for TPC reconstruction is defined in RecoWorkflow.cxx
/// and contains the following default processors
/// - digit reader
/// - clusterer
/// - ClusterHardware Decoder
/// - CA tracker
///
/// The default workflow can be customized by specifying input and output types
/// e.g. digits, clustershw, tracks.
///
/// MC info is processed by default, disabled by using command line option `--disable-mc`
///
/// This function hooks up the the workflow specifications into the DPL driver.
WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  //auto tpcSectors = o2::RangeTokenizer::tokenize<int>(cfgc.options().get<std::string>("tpc-sectors"));
  std::vector<int> tpcSectors(36);
  std::iota(tpcSectors.begin(), tpcSectors.end(), 0);
  // the lane configuration defines the subspecification ids to be distributed among the lanes.
  std::vector<int> laneConfiguration = tpcSectors; // Currently just a copy of the tpcSectors, why?
  auto nLanes = cfgc.options().get<int>("tpc-lanes");
  auto inputType = cfgc.options().get<std::string>("input-type");

  // depending on whether to dispatch early (prompt) and on the input type, we
  // set the matcher. Note that this has to be in accordance with the OutputSpecs
  // configured for the PublisherSpec
  auto dispmode = cfgc.options().get<std::string>("dispatching-mode");
  if (dispmode == "complete") {
    // nothing to do we leave the matcher empty which will suppress the dispatch
    // trigger and all messages will be sent out together at end of computation
  } else if (inputType == "digits") {
    gDispatchTrigger = o2::framework::Output{"TPC", "DIGITS"};
  } else if (inputType == "clustershw") {
    gDispatchTrigger = o2::framework::Output{"TPC", "CLUSTERHW"};
  } else if (inputType == "clustersnative") {
    gDispatchTrigger = o2::framework::Output{"TPC", "CLUSTERNATIVE"};
  } else if (inputType == "zsraw") {
    gDispatchTrigger = o2::framework::Output{"TPC", "RAWDATA"};
  }
  // set up configuration
  o2::conf::ConfigurableParam::updateFromFile(cfgc.options().get<std::string>("configFile"));
  o2::conf::ConfigurableParam::updateFromString(cfgc.options().get<std::string>("configKeyValues"));
  o2::conf::ConfigurableParam::writeINI("o2tpcrecoworkflow_configuration.ini");

  bool doMC = not cfgc.options().get<bool>("disable-mc");
  return o2::tpc::reco_workflow::getWorkflow(&gPolicyData,                                      //
                                             tpcSectors,                                        // sector configuration
                                             laneConfiguration,                                 // lane configuration
                                             doMC,                                              //
                                             nLanes,                                            //
                                             inputType,                                         //
                                             cfgc.options().get<std::string>("output-type"),    //
                                             !cfgc.options().get<bool>("no-ca-clusterer"),      //
                                             !cfgc.options().get<bool>("no-tpc-zs-on-the-fly"), //
                                             cfgc.options().get<bool>("zs-10bit"),              //
                                             cfgc.options().get<float>("zs-threshold")          //
  );
}
