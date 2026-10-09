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

#define BOOST_TEST_MODULE MFTCARecoWorkflow
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <string_view>
#include <stdexcept>

#include "MFTWorkflow/CARecoWorkflow.h"
#include "Framework/ConfigContext.h"
#include "Framework/ConfigParamStore.h"
#include "Framework/ParamRetriever.h"
#include "Framework/ServiceRegistry.h"
#include "CommonUtils/ConfigurableParam.h"

namespace
{
bool hasDevice(const o2::framework::WorkflowSpec& workflow, std::string_view name)
{
  return std::any_of(workflow.begin(), workflow.end(), [name](const auto& spec) { return spec.name == name; });
}
} // namespace

BOOST_AUTO_TEST_CASE(DefaultWorkflowIsMonolithic)
{
  o2::mft::ca::WorkflowOptionInput input;
  input.useMC = false;
  const auto workflow = o2::mft::ca_reco_workflow::getWorkflow(o2::mft::ca::resolveWorkflowOptions(input, {}));

  BOOST_CHECK(hasDevice(workflow, "mft-digit-reader"));
  BOOST_CHECK(hasDevice(workflow, "mft-clusterer"));
  BOOST_CHECK(hasDevice(workflow, "mft-cluster-writer"));
  BOOST_CHECK(hasDevice(workflow, "mft-ca-tracker"));
  BOOST_CHECK(hasDevice(workflow, "mft-track-writer"));
  BOOST_CHECK(!hasDevice(workflow, "mft-tracker"));
}

BOOST_AUTO_TEST_CASE(UpstreamClustersCanRunTrackerOnly)
{
  o2::mft::ca::WorkflowOptionInput input;
  input.useMC = false;
  input.upstreamClusters = true;
  input.disableRootOutput = true;
  const auto workflow = o2::mft::ca_reco_workflow::getWorkflow(o2::mft::ca::resolveWorkflowOptions(input, {}));

  BOOST_REQUIRE_EQUAL(workflow.size(), 1);
  BOOST_CHECK_EQUAL(workflow.front().name, "mft-ca-tracker");
}

BOOST_AUTO_TEST_CASE(ParameterAliasesHaveIdenticalPrecedenceForBothEntryPoints)
{
  using namespace o2::mft::ca;
  WorkflowOptionInput input;
  input.mode = o2::itsmft::TrackingMode::Sync;
  input.nThreads = 4;
  const TrackerOptionAliases aliases{1, 1, true};
  const auto reco = resolveWorkflowOptions(input, aliases);
  input.kind = WorkflowKind::TrackerOnly;
  const auto standalone = resolveWorkflowOptions(input, aliases);
  BOOST_CHECK(reco.tracker.mode == o2::itsmft::TrackingMode::Async);
  BOOST_CHECK(reco.tracker.mode == standalone.tracker.mode);
  BOOST_CHECK_EQUAL(reco.tracker.nThreads, 1);
  BOOST_CHECK_EQUAL(reco.tracker.nThreads, standalone.tracker.nThreads);
  BOOST_CHECK(reco.tracker.filterIRFrames == standalone.tracker.filterIRFrames);
  BOOST_REQUIRE_EQUAL(reco.diagnostics.size(), 2u);
  BOOST_CHECK(reco.diagnostics[0].find("MFTCATrackerParam.trackingMode") != std::string::npos);
  BOOST_CHECK(reco.diagnostics[0].find("--tracking-mode") != std::string::npos);
  BOOST_CHECK(reco.diagnostics[1].find("MFTCATrackerParam.nThreads") != std::string::npos);
  BOOST_CHECK(reco.diagnostics[1].find("--nThreads") != std::string::npos);
}

BOOST_AUTO_TEST_CASE(InputAndIRRoutingMatrixMatchesGraphSubscriptions)
{
  using namespace o2::mft::ca;
  for (int stage = 0; stage < 3; ++stage) {
    for (const bool subscribe : {false, true}) {
      for (const bool filter : {false, true}) {
        WorkflowOptionInput input;
        input.useMC = false;
        input.upstreamDigits = stage == 1;
        input.upstreamClusters = stage == 2;
        input.useIRFrames = subscribe;
        const auto resolved = resolveWorkflowOptions(input, {-1, 1, filter});
        const auto workflow = o2::mft::ca_reco_workflow::getWorkflow(resolved);
        BOOST_CHECK_EQUAL(hasDevice(workflow, "mft-digit-reader"), stage == 0);
        BOOST_CHECK_EQUAL(hasDevice(workflow, "mft-clusterer"), stage != 2);
        BOOST_CHECK_EQUAL(hasDevice(workflow, "its-irframe-reader"), stage == 0 && (subscribe || filter));
        const auto tracker = std::find_if(workflow.begin(), workflow.end(), [](const auto& spec) { return spec.name == "mft-ca-tracker"; });
        BOOST_REQUIRE(tracker != workflow.end());
        const bool consumesIR = std::any_of(tracker->inputs.begin(), tracker->inputs.end(), [](const auto& spec) { return spec.binding == "IRFramesITS"; });
        BOOST_CHECK_EQUAL(consumesIR, subscribe || filter);
        BOOST_CHECK_EQUAL(resolved.tracker.filterIRFrames, filter);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(OutputFlagsRetainTheirWriterPolicy)
{
  using namespace o2::mft::ca;
  for (const bool disable : {false, true}) {
    for (const bool rofs : {false, true}) {
      WorkflowOptionInput input;
      input.useMC = false;
      input.disableRootOutput = disable;
      input.clusterROFsOnly = rofs;
      const auto workflow = o2::mft::ca_reco_workflow::getWorkflow(resolveWorkflowOptions(input, {}));
      BOOST_CHECK_EQUAL(hasDevice(workflow, "mft-cluster-writer"), !disable || rofs);
      BOOST_CHECK_EQUAL(hasDevice(workflow, "mft-track-writer"), !disable);
    }
  }
}

BOOST_AUTO_TEST_CASE(DisabledAndInactiveTrackingHaveDistinctGraphs)
{
  using namespace o2::mft::ca;
  WorkflowOptionInput input;
  input.useMC = false;
  input.mode = o2::itsmft::TrackingMode::Off;
  auto workflow = o2::mft::ca_reco_workflow::getWorkflow(resolveWorkflowOptions(input, {}));
  BOOST_CHECK(hasDevice(workflow, "mft-ca-tracker"));
  BOOST_CHECK(hasDevice(workflow, "mft-track-writer"));
  input.runTracking = false;
  input.useIRFrames = true;
  workflow = o2::mft::ca_reco_workflow::getWorkflow(resolveWorkflowOptions(input, {}));
  BOOST_CHECK(!hasDevice(workflow, "mft-ca-tracker"));
  BOOST_CHECK(!hasDevice(workflow, "mft-track-writer"));
  BOOST_CHECK(!hasDevice(workflow, "its-irframe-reader"));
  input.assessment = true;
  BOOST_CHECK_THROW(resolveWorkflowOptions(input, {}), std::invalid_argument);
  input.assessment = false;
  input.tracksToRecords = true;
  BOOST_CHECK_THROW(resolveWorkflowOptions(input, {}), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(InvalidAliasesAndConflictingInputStagesFailBeforeGraphConstruction)
{
  using namespace o2::mft::ca;
  WorkflowOptionInput input;
  BOOST_CHECK_THROW(resolveWorkflowOptions(input, {99, 1, false}), std::invalid_argument);
  BOOST_CHECK_THROW(resolveWorkflowOptions(input, {-1, 0, false}), std::invalid_argument);
  input.nThreads = 0;
  BOOST_CHECK_THROW(resolveWorkflowOptions(input, {}), std::invalid_argument);
  input.nThreads = 1;
  input.upstreamDigits = input.upstreamClusters = true;
  BOOST_CHECK_THROW(resolveWorkflowOptions(input, {}), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(DriverBoundaryAppliesThreadAndModeAliasesBeforeDeviceConstruction)
{
  using namespace o2::framework;
  using namespace o2::mft::ca;
  std::vector<ConfigParamSpec> specs{
    {"nThreads", VariantType::Int, 4, {"threads"}},
    {"tracking-mode", VariantType::String, "sync", {"mode"}},
    {"configKeyValues", VariantType::String, "MFTCATrackerParam.nThreads=1;MFTCATrackerParam.trackingMode=1", {"parameters"}}};
  for (const auto* key : {"disable-mc", "disable-root-output", "use-geom", "use-full-geometry", "use-irframes",
                          "digits-from-upstream", "clusters-from-upstream", "cluster-rof-branch-only", "disable-tracking",
                          "run-assessment", "disable-process-gen", "run-tracks2records", "enable-mft-staggering"}) {
    specs.push_back({key, VariantType::Bool, false, {key}});
  }
  auto store = std::make_unique<ConfigParamStore>(specs, std::vector<std::unique_ptr<ParamRetriever>>{});
  store->preload();
  store->activate();
  ConfigParamRegistry registry{std::move(store)};
  ServiceRegistry services;
  ConfigContext context{registry, ServiceRegistryRef{services}, 0, nullptr};
  const auto reco = readWorkflowOptions(context, WorkflowKind::Reconstruction);
  const auto standalone = readWorkflowOptions(context, WorkflowKind::TrackerOnly);
  BOOST_CHECK_EQUAL(reco.tracker.nThreads, 1);
  BOOST_CHECK_EQUAL(reco.tracker.nThreads, standalone.tracker.nThreads);
  BOOST_CHECK(reco.tracker.mode == o2::itsmft::TrackingMode::Async);
  BOOST_CHECK(reco.tracker.mode == standalone.tracker.mode);
  registry.override("configKeyValues", std::string{"MFTCATrackerParam.trackingMode=-1"});
  BOOST_CHECK_EQUAL(readWorkflowOptions(context, WorkflowKind::TrackerOnly).tracker.nThreads, 4);
  o2::conf::ConfigurableParam::setValue<int>("MFTCATrackerParam", "nThreads", 1);
}
