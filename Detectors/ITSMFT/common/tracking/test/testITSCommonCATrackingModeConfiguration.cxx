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

// Workflow-onboarding Slice 1: focused tests for the dedicated
// ITSCommonCATrackerParam configuration type (TrackingConfigParam.h) and the
// real ITS Sync and Async branches of TrackingMode::getTrackingParameters()
// (Configuration.cxx). No workflow spec exists yet -- these tests call the
// common-tracking library directly, the same way
// the workflow loading tests already document that the
// ITS branch of getTrackingParameters() used to unconditionally
// LOGP(fatal, ...) regardless of mode; that fatal is now real per-mode
// behaviour instead, exercised here.

#define BOOST_TEST_MODULE ITSMFT ITSCommonCATrackingModeConfiguration
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include "TrackingParameterTestSupport.h"
#include <boost/test/unit_test.hpp>

#include <boost/property_tree/ptree.hpp>
#include <fairlogger/Logger.h>
#include <stdexcept>

#include "DetectorsCommonDataFormats/DetID.h"
#include "DetectorsBase/Propagator.h"
#include "ITSMFTTracking/Configuration.h"
#include "ITSMFTTracking/TrackingConfigParam.h"
#include "ITSMFTTracking/ITSTrackingConfigParam.h"
#include "ITStracking/Configuration.h"

using namespace o2::itsmft;
using namespace o2::itsmft::tracking;

namespace
{
struct MagneticFieldFixture {
  MagneticFieldFixture() { o2::base::Propagator::initFieldFromGRP(0.f, 0.f, true, false); }
};
} // namespace

BOOST_TEST_GLOBAL_FIXTURE(MagneticFieldFixture);

// --- Dedicated name is distinct from every other registered CA param name --

BOOST_AUTO_TEST_CASE(DedicatedNameIsDistinctFromLegacyAndMFTNames)
{
  const auto& itsCommonCA = ITSCommonCATrackerParam::Instance();
  const auto& itsLegacy = o2::its::TrackerParamConfig::Instance();
  const auto& mftCommonCA = TrackerParamConfig<o2::detectors::DetID::MFT>::Instance();

  BOOST_CHECK_EQUAL(itsCommonCA.getName(), "ITSCommonCATrackerParam");
  BOOST_CHECK_EQUAL(itsLegacy.getName(), "ITSCATrackerParam");
  BOOST_CHECK_EQUAL(mftCommonCA.getName(), "MFTCATrackerParam");

  BOOST_CHECK(itsCommonCA.getName() != itsLegacy.getName());
  BOOST_CHECK(itsCommonCA.getName() != mftCommonCA.getName());
  BOOST_CHECK(itsLegacy.getName() != mftCommonCA.getName());
}

// --- ITSCommonCATrackerParam defaults match the documented Sync baseline ---

BOOST_AUTO_TEST_CASE(DedicatedDefaultsMatchDocumentedSyncBaseline)
{
  const auto& tc = ITSCommonCATrackerParam::Instance();
  BOOST_CHECK_EQUAL(tc.dropTFUponFailure, false);
  BOOST_CHECK_EQUAL(tc.printMemory, false);
  BOOST_CHECK_EQUAL(tc.maxMemory, std::numeric_limits<size_t>::max());
  BOOST_CHECK_EQUAL(tc.saveTimeBenchmarks, false);
  BOOST_CHECK_EQUAL(tc.useDiamond, false);
  BOOST_CHECK_EQUAL(tc.diamondPos[0], 0.f);
  BOOST_CHECK_EQUAL(tc.diamondPos[1], 0.f);
  BOOST_CHECK_EQUAL(tc.diamondPos[2], 0.f);
  BOOST_CHECK_EQUAL(tc.pvRes, -1.f);
  BOOST_CHECK_EQUAL(tc.nThreads, 1);
}

// --- ITS Sync construction is valid, one-iteration, with expected values ---

BOOST_AUTO_TEST_CASE(ITSSyncTrackingParametersAreValidOneIteration)
{
  const auto trackParams = o2::itsmft::tracking::test::referenceTrackingParameters(o2::detectors::DetID::ITS, TrackingMode::Sync);

  BOOST_REQUIRE_EQUAL(trackParams.size(), 1u);
  const auto& p = trackParams[0];

  BOOST_CHECK_EQUAL(p.NLayers, tracking::ITSNLayers);
  BOOST_CHECK_EQUAL(p.MinTrackLength, tracking::kCAMinTrackLength);
  BOOST_CHECK_EQUAL(p.MinPt.size(), static_cast<size_t>(tracking::ITSNLayers -
                                                        tracking::kCAMinTrackLength + 1));
  BOOST_CHECK_EQUAL(p.StartLayerMask.count(), tracking::ITSNLayers); // default mask: all 7 barrel layers active

  // Administrative fields wired straight from the dedicated config's defaults.
  BOOST_CHECK_EQUAL(p.DropTFUponFailure, false);
  BOOST_CHECK_EQUAL(p.MaxMemory, std::numeric_limits<size_t>::max());
  BOOST_CHECK_EQUAL(p.UseDiamond, false);

  // resetDetectorDefaults(..., DetID::ITS) supplies real barrel geometry
  // defaults (TrackingParameters' own struct defaults); confirm they were
  // not clobbered.
  BOOST_CHECK_EQUAL(p.LayerRadii.size(), static_cast<size_t>(tracking::ITSNLayers));
  BOOST_CHECK_EQUAL(p.LayerZ.size(), static_cast<size_t>(tracking::ITSNLayers));
}

BOOST_AUTO_TEST_CASE(ITSSyncTrackingParametersAreDeterministic)
{
  const auto a = o2::itsmft::tracking::test::referenceTrackingParameters(o2::detectors::DetID::ITS, TrackingMode::Sync);
  const auto b = o2::itsmft::tracking::test::referenceTrackingParameters(o2::detectors::DetID::ITS, TrackingMode::Sync);
  BOOST_REQUIRE_EQUAL(a.size(), b.size());
  BOOST_CHECK_EQUAL(a[0].MinTrackLength, b[0].MinTrackLength);
  BOOST_CHECK_EQUAL(a[0].NLayers, b[0].NLayers);
  BOOST_CHECK(a[0].LayerRadii == b[0].LayerRadii);
}

BOOST_AUTO_TEST_CASE(ITSAsyncMatchesLegacySelectionParameters)
{
  const auto common = o2::itsmft::tracking::test::referenceTrackingParameters(o2::detectors::DetID::ITS, TrackingMode::Async);
  const auto legacy = o2::its::TrackingMode::getTrackingParameters(o2::its::TrackingMode::Async);

  BOOST_REQUIRE_EQUAL(common.size(), 3u);
  BOOST_REQUIRE_EQUAL(common.size(), legacy.size());
  for (size_t iteration = 0; iteration < common.size(); ++iteration) {
    const auto& commonIteration = common[iteration];
    const auto& legacyIteration = legacy[iteration];
    BOOST_CHECK_EQUAL(commonIteration.ColBins, legacyIteration.ZBins);
    BOOST_CHECK_EQUAL(commonIteration.RowBins, legacyIteration.PhiBins);
    BOOST_CHECK_EQUAL(commonIteration.MinTrackLength, legacyIteration.MinTrackLength);
    BOOST_CHECK_EQUAL(commonIteration.TrackletMinPt, legacyIteration.TrackletMinPt);
    BOOST_CHECK_EQUAL(commonIteration.StartLayerMask.value(), legacyIteration.StartLayerMask.value());
    BOOST_REQUIRE_EQUAL(commonIteration.MinPt.size(), legacyIteration.MinPt.size());
    for (size_t length = 0; length < commonIteration.MinPt.size(); ++length) {
      BOOST_CHECK_EQUAL(commonIteration.MinPt[length], legacyIteration.MinPt[length]);
    }
  }

  // These are currently intentional algorithm limitations, not selection
  // mismatches: common CA has no CellDeltaTanLambdaSigma analogue and its ITS
  // cylindrical surfaces do not yet support the legacy material LUT.
  BOOST_CHECK(legacy.front().CorrType == o2::base::PropagatorImpl<float>::MatCorrType::USEMatCorrLUT);
  BOOST_CHECK(common.front().CorrType == o2::base::PropagatorImpl<float>::MatCorrType::USEMatCorrNONE);
}

// --- Every unsupported TrackingMode fails closed, none silently mapped -----
//
// LOGP(fatal, ...) normally terminates the process (FairLogger default). A
// process-local OnFatal handler converts it into a catchable exception so
// this remains a normal, non-crashing ctest case. Each ITSMFT test source
// file builds its own executable (o2_add_test == one binary per SOURCES
// file), so this handler cannot leak into unrelated test binaries.

namespace
{
struct FatalToExceptionFixture {
  FatalToExceptionFixture()
  {
    fair::Logger::OnFatal([]() { throw std::runtime_error("fatal"); });
  }
};
} // namespace

BOOST_FIXTURE_TEST_CASE(EveryUnsupportedITSModeFailsClosed, FatalToExceptionFixture)
{
  const std::array<TrackingMode::Type, 3> unsupported{
    TrackingMode::Off, TrackingMode::Unset, TrackingMode::Cosmics};

  for (const auto mode : unsupported) {
    BOOST_CHECK_THROW(o2::itsmft::tracking::test::referenceTrackingParameters(o2::detectors::DetID::ITS, mode), std::runtime_error);
  }
}

BOOST_FIXTURE_TEST_CASE(SyncStillSucceedsAfterFatalHandlerInstalled, FatalToExceptionFixture)
{
  // The OnFatal fixture above must not turn the supported paths into false
  // failures: Sync and Async should still construct normally.
  BOOST_CHECK_NO_THROW(o2::itsmft::tracking::test::referenceTrackingParameters(o2::detectors::DetID::ITS, TrackingMode::Sync));
  BOOST_CHECK_NO_THROW(o2::itsmft::tracking::test::referenceTrackingParameters(o2::detectors::DetID::ITS, TrackingMode::Async));
}

// Sync/Async/Cosmics require a configured magnetic-field singleton. The
// detector defaults and the early-return Off path can be tested directly.

BOOST_AUTO_TEST_CASE(MFTDefaultsUseTheCommonFourHitSelection)
{
  TrackingParameters params;
  resetDetectorDefaults(params, o2::detectors::DetID::MFT);

  BOOST_CHECK_EQUAL(TrackerParamConfig<o2::detectors::DetID::MFT>::MinTrackLength, 4);
  BOOST_CHECK_EQUAL(params.MinPt.size(), static_cast<size_t>(tracking::MFTNLayers - 4 + 1));
  BOOST_CHECK_EQUAL(params.ColBins, 64);
  BOOST_CHECK_EQUAL(params.RowBins, 128);
}

BOOST_FIXTURE_TEST_CASE(MFTOffStillReturnsEmptyNotFatal, FatalToExceptionFixture)
{
  BOOST_CHECK_NO_THROW({
    const auto trackParams = o2::itsmft::tracking::test::referenceTrackingParameters(o2::detectors::DetID::MFT, TrackingMode::Off);
    BOOST_CHECK(trackParams.empty());
  });
}

// --- workflow-onboarding Slice 2: diamondPos/pvRes are wired through -------
//
// Mutates the global ITSCommonCATrackerParam singleton via
// ConfigurableParam::setValue -- deliberately placed last in this
// translation unit so no other test observes the mutated state.

BOOST_AUTO_TEST_CASE(DiamondPosAndPVresAreWiredIntoITSSyncTrackingParameters)
{
  o2::conf::ConfigurableParam::setValue<float>("ITSCommonCATrackerParam", "diamondPos[0]", 1.5f);
  o2::conf::ConfigurableParam::setValue<float>("ITSCommonCATrackerParam", "diamondPos[1]", -2.5f);
  o2::conf::ConfigurableParam::setValue<float>("ITSCommonCATrackerParam", "diamondPos[2]", 3.5f);
  o2::conf::ConfigurableParam::setValue<float>("ITSCommonCATrackerParam", "pvRes", 0.25f);
  o2::conf::ConfigurableParam::setValue<bool>("ITSCommonCATrackerParam", "useDiamond", true);

  const auto trackParams = o2::itsmft::tracking::test::referenceTrackingParameters(o2::detectors::DetID::ITS, TrackingMode::Sync);
  BOOST_REQUIRE_EQUAL(trackParams.size(), 1u);
  const auto& p = trackParams[0];

  BOOST_CHECK_EQUAL(p.UseDiamond, true);
  BOOST_CHECK_EQUAL(p.Diamond[0], 1.5f);
  BOOST_CHECK_EQUAL(p.Diamond[1], -2.5f);
  BOOST_CHECK_EQUAL(p.Diamond[2], 3.5f);
  BOOST_CHECK_EQUAL(p.PVres, 0.25f);
}
