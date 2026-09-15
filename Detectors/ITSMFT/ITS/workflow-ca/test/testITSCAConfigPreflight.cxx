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

// Driver configuration validation before constructing any DPL device.

#define BOOST_TEST_MODULE ITSMFT ITSCAConfigPreflight
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <array>
#include <stdexcept>

#include <boost/property_tree/ptree.hpp>
#include <fairlogger/Logger.h>

#include "CommonUtils/ConfigurableParam.h"
#include "ITSCAWorkflow/ConfigPreflight.h"
#include "Framework/ConfigContext.h"
#include "Framework/ConfigParamStore.h"
#include "Framework/ParamRetriever.h"
#include "Framework/ServiceRegistry.h"
#include "ITSMFTTracking/TrackingConfigParam.h"

using namespace o2::its::ca;

namespace
{
struct FatalToExceptionFixture {
  FatalToExceptionFixture()
  {
    fair::Logger::OnFatal([]() { throw std::runtime_error("fatal"); });
  }
};
} // namespace

// --- applyConfigKeyValuesOrFatal(): preflight runs before the update -------

BOOST_FIXTURE_TEST_CASE(LegacyNamespaceIsRejectedBeforeAnyUpdate, FatalToExceptionFixture)
{
  // Sentinel: if the rejection did not actually run before
  // ConfigurableParam::updateFromString(), this legacy-namespace string
  // would still throw from updateFromString() itself (unknown param), so
  // this alone would not distinguish "preflight fired first" from "update
  // itself fatal'd" -- the meaningful assertion is in the next test, which
  // confirms the dedicated param was NOT mutated by the rejected string.
  BOOST_CHECK_THROW(applyConfigKeyValuesOrFatal("ITSCATrackerParam.trackFollowerTop=1"), std::runtime_error);
}

BOOST_FIXTURE_TEST_CASE(RejectedStringNeverReachesConfigurableParamUpdate, FatalToExceptionFixture)
{
  // A malicious/confused string mixing a real dedicated-namespace override
  // with an offending legacy one must not have its dedicated part applied
  // either -- the whole string is rejected pre-update, atomically.
  o2::conf::ConfigurableParam::setValue<bool>("ITSCommonCATrackerParam", "useDiamond", false);
  BOOST_CHECK_THROW(
    applyConfigKeyValuesOrFatal("ITSCommonCATrackerParam.useDiamond=true;ITSCATrackerParam.trackFollowerTop=1"),
    std::runtime_error);
  BOOST_CHECK_EQUAL(o2::itsmft::ITSCommonCATrackerParam::Instance().useDiamond, false);
}

BOOST_FIXTURE_TEST_CASE(DedicatedNamespaceIsAcceptedAndApplied, FatalToExceptionFixture)
{
  o2::conf::ConfigurableParam::setValue<bool>("ITSCommonCATrackerParam", "useDiamond", false);
  BOOST_CHECK_NO_THROW(applyConfigKeyValuesOrFatal("ITSCommonCATrackerParam.useDiamond=true"));
  BOOST_CHECK_EQUAL(o2::itsmft::ITSCommonCATrackerParam::Instance().useDiamond, true);
  o2::conf::ConfigurableParam::setValue<bool>("ITSCommonCATrackerParam", "useDiamond", false);
}

BOOST_FIXTURE_TEST_CASE(EmptyStringIsAcceptedAndApplied, FatalToExceptionFixture)
{
  BOOST_CHECK_NO_THROW(applyConfigKeyValuesOrFatal(""));
}

BOOST_FIXTURE_TEST_CASE(LegacyNamespaceWithoutFieldIsRejected, FatalToExceptionFixture)
{
  BOOST_CHECK_THROW(applyConfigKeyValuesOrFatal("ITSCATrackerParam=1"), std::runtime_error);
}

BOOST_FIXTURE_TEST_CASE(MixedInputRejectsLegacyNamespaceInEitherPosition, FatalToExceptionFixture)
{
  for (const auto* config : {"ITSCATrackerParam.trackFollowerTop=1;ITSCommonCATrackerParam.useDiamond=true",
                             "ITSCommonCATrackerParam.useDiamond=true;ITSCATrackerParam.trackFollowerTop=1"}) {
    o2::conf::ConfigurableParam::setValue<bool>("ITSCommonCATrackerParam", "useDiamond", false);
    BOOST_CHECK_THROW(applyConfigKeyValuesOrFatal(config), std::runtime_error);
    BOOST_CHECK_EQUAL(o2::itsmft::ITSCommonCATrackerParam::Instance().useDiamond, false);
  }
}

BOOST_FIXTURE_TEST_CASE(OuterWhitespaceAndInternalKeyWhitespaceKeepNamespace, FatalToExceptionFixture)
{
  o2::conf::ConfigurableParam::setValue<bool>("ITSCommonCATrackerParam", "useDiamond", false);
  BOOST_CHECK_THROW(
    applyConfigKeyValuesOrFatal("  ITSCommonCATrackerParam.useDiamond=true ; ITSCATrackerParam.trackFollowerTop=1  "),
    std::runtime_error);
  BOOST_CHECK_EQUAL(o2::itsmft::ITSCommonCATrackerParam::Instance().useDiamond, false);

  BOOST_CHECK_THROW(applyConfigKeyValuesOrFatal("ITSCATrackerParam.trackFollowerTop = 1"), std::runtime_error);
}

BOOST_FIXTURE_TEST_CASE(EmptyEntriesAndUnrelatedNamespacesAreAccepted, FatalToExceptionFixture)
{
  o2::conf::ConfigurableParam::setValue<bool>("ITSCommonCATrackerParam", "dropTFUponFailure", false);
  o2::conf::ConfigurableParam::setValue<int>("ITSVertexerParam", "nIterations", 1);
  BOOST_CHECK_NO_THROW(applyConfigKeyValuesOrFatal(
    ";;ITSCommonCATrackerParam.dropTFUponFailure=true;;;ITSVertexerParam.nIterations=2;;"));
  BOOST_CHECK_EQUAL(o2::itsmft::ITSCommonCATrackerParam::Instance().dropTFUponFailure, true);
  BOOST_CHECK_EQUAL(o2::its::VertexerParamConfig::Instance().nIterations, 2);
  o2::conf::ConfigurableParam::setValue<bool>("ITSCommonCATrackerParam", "dropTFUponFailure", false);
  o2::conf::ConfigurableParam::setValue<int>("ITSVertexerParam", "nIterations", 1);
}

BOOST_FIXTURE_TEST_CASE(MalformedTokensRemainConfiguratorErrors, FatalToExceptionFixture)
{
  for (const auto* config : {"ITSCATrackerParamNoEquals", "=ITSCATrackerParam.x", "ITSCATrackerParam.x="}) {
    BOOST_CHECK_THROW(applyConfigKeyValuesOrFatal(config), std::runtime_error);
  }
}

BOOST_FIXTURE_TEST_CASE(RepeatedAcceptedAndRejectedCallsRemainDeterministic, FatalToExceptionFixture)
{
  for (int i = 0; i < 5; ++i) {
    o2::conf::ConfigurableParam::setValue<bool>("ITSCommonCATrackerParam", "useDiamond", false);
    BOOST_CHECK_THROW(applyConfigKeyValuesOrFatal("ITSCATrackerParam.trackFollowerTop=1"), std::runtime_error);
    BOOST_CHECK_NO_THROW(applyConfigKeyValuesOrFatal("ITSCommonCATrackerParam.useDiamond=true"));
    BOOST_CHECK_EQUAL(o2::itsmft::ITSCommonCATrackerParam::Instance().useDiamond, true);
  }
  o2::conf::ConfigurableParam::setValue<bool>("ITSCommonCATrackerParam", "useDiamond", false);
}

// --- requireSupportedTrackingModeOrFatal(): Sync and Async are accepted ---

BOOST_FIXTURE_TEST_CASE(SupportedModesAreAccepted, FatalToExceptionFixture)
{
  BOOST_CHECK_NO_THROW(requireSupportedTrackingModeOrFatal(o2::itsmft::TrackingMode::Sync));
  BOOST_CHECK_NO_THROW(requireSupportedTrackingModeOrFatal(o2::itsmft::TrackingMode::Async));
}

BOOST_FIXTURE_TEST_CASE(UnsupportedModesFailClosed, FatalToExceptionFixture)
{
  const std::array<o2::itsmft::TrackingMode::Type, 3> rejected{
    o2::itsmft::TrackingMode::Off, o2::itsmft::TrackingMode::Unset, o2::itsmft::TrackingMode::Cosmics};
  for (const auto mode : rejected) {
    BOOST_CHECK_THROW(requireSupportedTrackingModeOrFatal(mode), std::runtime_error);
  }
}

BOOST_AUTO_TEST_CASE(VertexSelectionIsExplicitAndLegacyAliasesMustAgree)
{
  BOOST_CHECK_THROW(resolveVertexSource("", false, false), std::invalid_argument);
  BOOST_CHECK_THROW(resolveVertexSource("", true, true), std::invalid_argument);
  BOOST_CHECK_THROW(resolveVertexSource("truth", true, false), std::invalid_argument);
  BOOST_CHECK_THROW(resolveVertexSource("diamond", false, true), std::invalid_argument);
  BOOST_CHECK_THROW(resolveVertexSource("unknown", false, false), std::invalid_argument);
  BOOST_CHECK(resolveVertexSource("", true, false) == VertexSource::Diamond);
  BOOST_CHECK(resolveVertexSource("", false, true) == VertexSource::Truth);
  BOOST_CHECK(resolveVertexSource("diamond", false, false) == VertexSource::Diamond);
  BOOST_CHECK(resolveVertexSource("truth", false, false) == VertexSource::Truth);
  BOOST_CHECK(resolveVertexSource("diamond", true, false) == VertexSource::Diamond);
  BOOST_CHECK(resolveVertexSource("truth", false, true) == VertexSource::Truth);
}

BOOST_AUTO_TEST_CASE(DriverResolvesTruthContextIndependentlyOfMCLabels)
{
  using namespace o2::framework;
  std::vector<ConfigParamSpec> specs{
    {"configKeyValues", VariantType::String, "ITSCommonCATrackerParam.useDiamond=false;ITSVertexerParam.useTruthSeeding=false", {"parameters"}},
    {"tracking-mode", VariantType::String, "async", {"mode"}},
    {"vertex-source", VariantType::String, "truth", {"vertices"}},
    {"truth-context", VariantType::String, "custom-context.root", {"context"}},
    {"disable-mc", VariantType::Bool, true, {"MC labels"}},
    {"disable-root-output", VariantType::Bool, false, {"output"}},
    {"use-geom", VariantType::Bool, false, {"geometry"}},
    {"use-full-geometry", VariantType::Bool, true, {"geometry alias"}}};
  auto store = std::make_unique<ConfigParamStore>(specs, std::vector<std::unique_ptr<ParamRetriever>>{});
  store->preload();
  store->activate();
  ConfigParamRegistry registry{std::move(store)};
  ServiceRegistry services;
  ConfigContext context{registry, ServiceRegistryRef{services}, 0, nullptr};
  const auto resolved = readWorkflowOptions(context);
  BOOST_CHECK(resolved.vertexSource == VertexSource::Truth);
  BOOST_CHECK(!resolved.useMC);
  BOOST_CHECK(resolved.useFullGeometry);
  BOOST_CHECK_EQUAL(resolved.truthContext, "custom-context.root");
  registry.override("truth-context", std::string{});
  BOOST_CHECK_THROW(readWorkflowOptions(context), std::invalid_argument);
  registry.override("vertex-source", std::string{"diamond"});
  BOOST_CHECK(readWorkflowOptions(context).vertexSource == VertexSource::Diamond);
  registry.override("configKeyValues", std::string{"ITSCommonCATrackerParam.nThreads=0"});
  BOOST_CHECK_THROW(readWorkflowOptions(context), std::invalid_argument);
  o2::conf::ConfigurableParam::setValue<int>("ITSCommonCATrackerParam", "nThreads", 1);
}
