// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   TrackReaderWorkflow.cxx

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

#include "TPCWorkflow/TrackReaderSpec.h"

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
    {"input-type", VariantType::String, "tracks", {"tracks"}},
    {"dispatching-mode", VariantType::String, "prompt", {"determines when to dispatch: prompt, complete"}},
    {"disable-mc", VariantType::Bool, false, {"disable sending of MC information"}}};

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

#include "Framework/runDataProcessing.h" // the main driver

using namespace o2::framework;

/// MC info is processed by default, disabled by using command line option `--disable-mc`
///
/// This function hooks up the the workflow specifications into the DPL driver.
WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  WorkflowSpec specs;

  auto inputType = cfgc.options().get<std::string>("input-type");
  auto dispmode = cfgc.options().get<std::string>("dispatching-mode");
  if (dispmode == "complete") {
    // nothing to do we leave the matcher empty which will suppress the dispatch
    // trigger and all messages will be sent out together at end of computation
  } else if (inputType == "tracks") {
    gDispatchTrigger = o2::framework::Output{"TPC", "TRACKS"};
  }

  bool doMC = not cfgc.options().get<bool>("disable-mc");

  specs.push_back(o2::tpc::getTPCTrackReaderSpec(doMC));

  return std::move(specs);
}
