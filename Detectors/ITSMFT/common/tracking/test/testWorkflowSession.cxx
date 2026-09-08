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

#define BOOST_TEST_MODULE ITSMFT workflow session
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <boost/mpl/list.hpp>
#include <type_traits>
#include <any>
#include <algorithm>
#include <array>
#include <vector>
#include <map>
#include "ITSMFTTracking/WorkflowSession.h"
#include "ITSMFTTracking/ITSMFTDetectorDefinitions.h"
#include "TrackingParameterTestSupport.h"

using namespace o2::itsmft;
using namespace o2::itsmft::tracking;
using LayerCounts = boost::mpl::list<std::integral_constant<int, ITSNLayers>, std::integral_constant<int, MFTNLayers>>;
namespace
{
struct FieldFixture {
  FieldFixture() { o2::base::Propagator::initFieldFromGRP(0.f, 0.f, true, false); }
};
BOOST_GLOBAL_FIXTURE(FieldFixture);

template <int N>
struct Rig {
  static constexpr auto Detector = N == ITSNLayers ? o2::detectors::DetID::ITS : o2::detectors::DetID::MFT;
  WorkflowSession session{N == ITSNLayers ? "ITS" : "MFT", N};
  Tracker tracker;
  TrackerTraits traits;
  std::shared_ptr<tbb::task_arena> arena;
  TopologyDictionary dictionary;
  std::array<LayerId, N> mapping{};
  std::vector<ROFRecord> rofs{{{100, 5}, 0, 0, 0}};
  std::vector<CompClusterExt> clusters;
  struct Decoder : ClusterDecoder {
    ClusterDecodeResult decode(const CompClusterExt&, BoundedPatternCursor&, const TopologyDictionary*, uint32_t, bool) const override
    {
      ClusterDecodeResult result;
      const float radius = N == ITSNLayers ? kITSStaticSurfaceCatalog[0].referenceCoordinate : 3.f;
      const float z = N == ITSNLayers ? 0.f : kMFTStaticSurfaceCatalog[0].referenceCoordinate;
      result.decoded.global = {radius, 0.f, z};
      result.decoded.cylinderFrame = {radius, 0.f, z, 0.f};
      result.decoded.rowColumnCovariance = {1.e-4f, 0.f, 1.e-4f};
      result.decoded.layer = 0;
      return result;
    }
  } decoder;

  explicit Rig(bool drop = false, size_t memory = std::numeric_limits<size_t>::max())
  {
    TrackingParameters parameters;
    resetDetectorDefaults(parameters, Detector);
    parameters.UseDiamond = true;
    auto plan = test::makeTrackingPlan(parameters);
    plan.execution = {memory, drop};
    SurfaceCatalogView catalog = N == ITSNLayers ? SurfaceCatalogView{kITSStaticSurfaceCatalog.data(), ITSNLayers}
                                                 : SurfaceCatalogView{kMFTStaticSurfaceCatalog.data(), MFTNLayers};
    TrackerInitialization init{catalog, {}, std::move(plan), std::make_shared<BoundedMemoryResource>()};
    BOOST_REQUIRE(tracker.initialize(session.frame, init).ok());
    traits.setNThreads(1, arena);
    for (int layer = 0; layer < N; ++layer) {
      mapping[layer] = LayerId{static_cast<uint16_t>(layer)};
    }
    configure();
  }
  void configure()
  {
    std::vector<o2::its::LayerTiming> timings(N);
    std::fill(timings.begin(), timings.end(), o2::its::LayerTiming{.mNROFsTF = 1, .mROFLength = 40});
    session.configureTiming(timings, [](int) { return true; });
  }
  ClusterSourceInput source()
  {
    ClusterSourceInput input;
    input.detector = Detector;
    input.id = ClusterSourceId{0};
    input.rofs = rofs;
    input.clusters = clusters;
    input.dictionary = &dictionary;
    input.decoder = &decoder;
    input.layerToSurface = mapping;
    return input;
  }
  void checkClean()
  {
    BOOST_CHECK_EQUAL(session.frame.getTotalMeasurements(), 0u);
    BOOST_CHECK(session.externalIndices.empty());
    BOOST_CHECK(session.clusterSizes.empty());
    BOOST_CHECK(!session.publicationClock);
    BOOST_CHECK_EQUAL(session.frame.getROFViews().overlap.mLayerCount, 0);
  }
};
} // namespace

BOOST_AUTO_TEST_CASE_TEMPLATE(SuccessAndValidEmptyInputCompleteBeforeCleanup, Count, LayerCounts)
{
  for (bool withCluster : {false, true}) {
    Rig<Count::value> rig;
    if (withCluster) {
      rig.clusters.emplace_back(0, 0, CompCluster::InvalidPatternID, 0);
      rig.rofs[0].setNEntries(1);
    }
    int loaded = 0, completed = 0;
    {
      auto cleanup = rig.session.cleanupOnExit();
      const auto outcome = rig.session.process(rig.tracker, rig.traits, rig.source(), [&](const o2::InteractionRecord& origin) {
          ++loaded;
          BOOST_CHECK(origin == rig.rofs.front().getBCData());
          BOOST_CHECK_EQUAL(rig.session.frame.getTotalMeasurements(), withCluster ? 1u : 0u);
          BOOST_CHECK_EQUAL(rig.session.frame.getROFViews().overlap.mLayerCount, Count::value); }, [&](const TrackingResult& result) {
          ++completed;
          BOOST_CHECK(result.outcome == TrackingOutcome::Success);
          BOOST_REQUIRE_EQUAL(result.acceptedTrackCounts.size(), 1u);
          BOOST_CHECK_EQUAL(result.acceptedTrackCounts[0], 0u); });
      BOOST_CHECK(decideCATrackerPublicationAction(true, outcome) == CATrackerPublicationAction::PublishActiveResult);
      rig.session.publicationClock.emplace(rig.session.overlap.getView().getClockLayer());
      BOOST_CHECK(rig.session.publicationClock);
    }
    BOOST_CHECK_EQUAL(loaded, 1);
    BOOST_CHECK_EQUAL(completed, 1);
    rig.checkClean();
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(MalformedInputDropsOnlyUnderTheConfiguredPolicy, Count, LayerCounts)
{
  for (bool drop : {false, true}) {
    Rig<Count::value> rig{drop};
    rig.rofs[0].setNEntries(1); // Claims a missing cluster: recoverable InvalidROFRange.
    int completed = 0;
    const auto run = [&] {
      auto cleanup = rig.session.cleanupOnExit();
      const auto result = rig.session.process(rig.tracker, rig.traits, rig.source(), [](const o2::InteractionRecord&) {}, [&](const TrackingResult&) { ++completed; });
      BOOST_CHECK(decideCATrackerPublicationAction(true, result) == CATrackerPublicationAction::SkipDroppedTimeFrame);
      cleanup.frameAlreadyReset();
    };
    if (drop) {
      BOOST_CHECK_NO_THROW(run());
    } else {
      BOOST_CHECK_THROW(run(), RecoverableLoadFailure);
    }
    BOOST_CHECK_EQUAL(completed, 0);
    rig.checkClean();
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(StructuralLoadingAndPublicationExceptionsAlwaysCleanUp, Count, LayerCounts)
{
  for (bool drop : {false, true}) {
    Rig<Count::value> rig{drop};
    auto source = rig.source();
    source.dictionary = nullptr;
    const auto run = [&] {
      auto cleanup = rig.session.cleanupOnExit();
      rig.session.process(rig.tracker, rig.traits, source, [](const o2::InteractionRecord&) {}, [](const TrackingResult&) {});
    };
    BOOST_CHECK_THROW(run(), TimeFrameLoadException);
    rig.checkClean();
    rig.configure();
    const auto publish = [&] {
      auto cleanup = rig.session.cleanupOnExit();
      rig.session.process(rig.tracker, rig.traits, rig.source(), [](const o2::InteractionRecord&) {}, [](const TrackingResult&) { throw std::runtime_error{"publication failed"}; });
    };
    BOOST_CHECK_THROW(publish(), std::runtime_error);
    rig.checkClean();
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(ResourceExceptionsInPostLoadHookFollowLoadingPolicy, Count, LayerCounts)
{
  for (bool drop : {false, true}) {
    for (bool bounded : {false, true}) {
      Rig<Count::value> rig{drop};
      const auto run = [&] {
        auto cleanup = rig.session.cleanupOnExit();
        const auto outcome = rig.session.process(rig.tracker, rig.traits, rig.source(), [&](const o2::InteractionRecord&) {
          if (bounded) { throw BoundedMemoryResource::MemoryLimitExceeded{2, 1, 1}; }
          throw std::bad_alloc{}; }, [](const TrackingResult&) { BOOST_FAIL("must not track after failed loading"); });
        BOOST_CHECK(outcome == TrackingOutcome::RecoverableDropped);
        cleanup.frameAlreadyReset();
      };
      if (drop) {
        BOOST_CHECK_NO_THROW(run());
      } else {
        BOOST_CHECK_THROW(run(), std::bad_alloc);
      }
      rig.checkClean();
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(TrackingResourceFailureSkipsCompletionAndPublication, Count, LayerCounts)
{
  Rig<Count::value> rig{true, 1};
  auto cleanup = rig.session.cleanupOnExit();
  const auto outcome = rig.session.process(rig.tracker, rig.traits, rig.source(), [](const o2::InteractionRecord&) {}, [](const TrackingResult&) { BOOST_FAIL("must not complete a dropped TF"); });
  BOOST_CHECK(outcome == TrackingOutcome::RecoverableDropped);
  BOOST_CHECK(rig.session.frame.getGenericTracks().empty());
  cleanup.frameAlreadyReset();
}

BOOST_AUTO_TEST_CASE_TEMPLATE(TimingViewsBelongToTheSessionAndFilteringSurvivesMoves, Count, LayerCounts)
{
  Rig<Count::value> rig;
  {
    auto cleanup = rig.session.cleanupOnExit();
    std::vector<o2::its::LayerTiming> timings(Count::value);
    std::fill(timings.begin(), timings.end(), o2::its::LayerTiming{.mNROFsTF = 3, .mROFLength = 40});
    rig.session.configureTiming(timings, [](int rof) { return rof != 1; });
    std::fill(timings.begin(), timings.end(), o2::its::LayerTiming{}); // No view may refer to the caller's timing storage.
    const auto views = rig.session.frame.getROFViews();
    BOOST_CHECK_EQUAL(views.overlap.getLayer(0).mROFLength, 40u);
    for (int layer = 0; layer < Count::value; ++layer) {
      BOOST_CHECK(views.mask.isROFEnabled(layer, 0));
      BOOST_CHECK(!views.mask.isROFEnabled(layer, 1));
      BOOST_CHECK(views.mask.isROFEnabled(layer, 2));
    }
    std::fill(timings.begin(), timings.end(), o2::its::LayerTiming{.mNROFsTF = 3, .mROFLength = 40});
    timings[1].mROFLength = 41;
    BOOST_CHECK_THROW(rig.session.configureTiming(timings, [](int) { return true; }), TimeFrameLoadException);
    BOOST_CHECK_EQUAL(rig.session.frame.getROFViews().overlap.getLayer(1).mROFLength, 40u);
  }
  rig.checkClean();
}

BOOST_AUTO_TEST_CASE_TEMPLATE(InactivePublicationRetainsTheEchoedEmptyContract, Count, LayerCounts)
{
  for (auto outcome : {TrackingOutcome::Success, TrackingOutcome::RecoverableDropped, TrackingOutcome::Structural}) {
    BOOST_CHECK(decideCATrackerPublicationAction(false, outcome) == CATrackerPublicationAction::PublishInactiveEmpty);
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(UnclassifiedExceptionsDoNotBecomeRecoverableDrops, Count, LayerCounts)
{
  for (bool drop : {false, true}) {
    for (bool standard : {false, true}) {
      Rig<Count::value> rig{drop};
      const auto run = [&] {
        auto cleanup = rig.session.cleanupOnExit();
        rig.session.process(rig.tracker, rig.traits, rig.source(), [&](const o2::InteractionRecord&) {
          if (standard) { throw std::logic_error{"unexpected loading failure"}; }
          throw 7; }, [](const TrackingResult&) { BOOST_FAIL("must not complete after an exception"); });
      };
      if (standard) {
        BOOST_CHECK_THROW(run(), std::logic_error);
      } else {
        BOOST_CHECK_THROW(run(), int);
      }
      rig.checkClean();
    }
  }
}

namespace
{
struct TestOutputAllocator {
  std::map<int, std::any> values;
  template <typename Vector, typename Iterator>
  Vector& make(int output, Iterator first, Iterator last)
  {
    values[output] = Vector(first, last);
    return std::any_cast<Vector&>(values.at(output));
  }
};
} // namespace
BOOST_AUTO_TEST_CASE_TEMPLATE(PublishedCommonColumnsOwnTheirStorageAfterSessionCleanup, Count, LayerCounts)
{
  using Staged = std::conditional_t<Count::value == ITSNLayers, ITSGenericTrackOutput, MFTGenericTrackOutput>;
  Rig<Count::value> rig;
  TestOutputAllocator outputs;
  {
    auto cleanup = rig.session.cleanupOnExit();
    Staged staged;
    staged.tracks.resize(1);
    staged.trackROFs.emplace_back(o2::InteractionRecord{123, 45}, 0, 0, 1);
    staged.clusterIndices = {17, 23};
    copyTrackingOutputColumns(outputs, 0, 1, 2, staged);
    staged.clusterIndices[0] = 99;
    staged.trackROFs[0].setNEntries(0);
    staged.tracks.clear();
  }
  rig.checkClean();
  const auto& rofs = std::any_cast<const std::vector<ROFRecord>&>(outputs.values.at(0));
  BOOST_REQUIRE_EQUAL(rofs.size(), 1u);
  BOOST_CHECK_EQUAL(rofs[0].getNEntries(), 1);
  BOOST_CHECK((rofs[0].getBCData() == o2::InteractionRecord{123, 45}));
  BOOST_CHECK_EQUAL(std::any_cast<const decltype(Staged{}.tracks)&>(outputs.values.at(1)).size(), 1u);
  const auto& indices = std::any_cast<const std::vector<int>&>(outputs.values.at(2));
  const std::vector<int> expected{17, 23};
  BOOST_CHECK_EQUAL_COLLECTIONS(indices.begin(), indices.end(), expected.begin(), expected.end());
}

BOOST_AUTO_TEST_CASE_TEMPLATE(DetectorTimingConstructionRetainsValidationAndUnits, Count, LayerCounts)
{
  struct AlpideTiming {
    int length = 40;
    int getROFLengthInBC(int) const { return length; }
    int getROFDelayInBC(int) const { return 3; }
    int getROFBiasInBC(int) const { return 4; }
  } alpide;
  Rig<Count::value> rig;
  const std::vector<uint32_t> timeErrors(Count::value, 5);
  const auto timings = rig.session.layerTimings(alpide, 2, timeErrors);
  for (const auto& timing : timings) {
    BOOST_CHECK_EQUAL(timing.mROFLength, 40u);
    BOOST_CHECK_EQUAL(timing.mROFDelay, 3u);
    BOOST_CHECK_EQUAL(timing.mROFBias, 4u);
    BOOST_CHECK_EQUAL(timing.mROFAddTimeErr, 5u);
    BOOST_CHECK_EQUAL(timing.mNROFsTF, 178u);
  }
  BOOST_CHECK_EXCEPTION(rig.session.layerTimings(alpide, 0, timeErrors), TimeFrameLoadException,
                        [](const TimeFrameLoadException& error) { return error.reason() == TimeFrameLoadFailureReason::ZeroROFCount; });
  BOOST_CHECK_THROW(rig.session.layerTimings(alpide, 2, std::vector<uint32_t>(Count::value - 1)), TimeFrameLoadException);
  alpide.length = 0;
  BOOST_CHECK_EXCEPTION(rig.session.layerTimings(alpide, 2, timeErrors), TimeFrameLoadException,
                        [](const TimeFrameLoadException& error) { return error.reason() == TimeFrameLoadFailureReason::NonUniformROFTiming; });
}

BOOST_AUTO_TEST_CASE_TEMPLATE(UnchangedTimingReusesStorageButRefreshesEventData, Count, LayerCounts)
{
  WorkflowSession session{"test", Count::value};
  std::vector<o2::its::LayerTiming> timings(Count::value);
  std::fill(timings.begin(), timings.end(), o2::its::LayerTiming{.mNROFsTF = 3, .mROFLength = 40});
  session.configureTiming(timings, [](int rof) { return rof == 0; });
  o2::its::Vertex vertex;
  vertex.getTimeStamp().setTimeStamp(20);
  vertex.getTimeStamp().setTimeStampError(5);
  session.vertices.update(&vertex, 1);
  const auto overlapStorage = session.overlap.getView().mFlatTable;
  const auto vertexStorage = session.vertices.getView().mFlatTable;
  const auto maskStorage = session.mask.getView().mFlatMask;
  BOOST_REQUIRE_EQUAL(session.vertices.getView().getVertices(0, 0).getEntries(), 1u);
  session.publicationClock.emplace(session.overlap.getView().getClockLayer());
  session.reset();
  session.invalidatePublication();
  BOOST_CHECK_EQUAL(session.frame.getROFViews().overlap.mLayerCount, 0);

  int calls = 0;
  session.configureTiming(timings, [&](int rof) { ++calls; return rof == 2; });
  std::fill(timings.begin(), timings.end(), o2::its::LayerTiming{}); // The key and table definitions own their timing values.
  BOOST_CHECK_EQUAL(calls, 3);
  BOOST_CHECK(session.overlap.getView().mFlatTable == overlapStorage);
  BOOST_CHECK(session.vertices.getView().mFlatTable == vertexStorage);
  BOOST_CHECK(session.mask.getView().mFlatMask == maskStorage);
  BOOST_CHECK(!session.publicationClock);
  for (int layer = 0; layer < Count::value; ++layer) {
    for (int rof = 0; rof < 3; ++rof) {
      const auto range = session.vertices.getView().getVertices(layer, rof);
      BOOST_CHECK_EQUAL(range.getFirstEntry(), 0u);
      BOOST_CHECK_EQUAL(range.getEntries(), 0u);
      BOOST_CHECK_EQUAL(session.frame.getROFViews().mask.isROFEnabled(layer, rof), rof == 2);
    }
  }
  // New truth contents can be bound after a cache hit, then cleared again.
  vertex.getTimeStamp().setTimeStamp(100);
  session.vertices.update(&vertex, 1);
  BOOST_CHECK_EQUAL(session.vertices.getView().getVertices(0, 2).getEntries(), 1u);
  std::fill(timings.begin(), timings.end(), o2::its::LayerTiming{.mNROFsTF = 3, .mROFLength = 40});
  session.configureTiming(timings, [](int) { return false; });
  BOOST_CHECK_EQUAL(session.vertices.getView().getVertices(0, 2).getEntries(), 0u);
  BOOST_CHECK(!session.frame.getROFViews().mask.isROFEnabled(0, 2));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(EveryTimingFieldAndLayerExtentInvalidateTheCache, Count, LayerCounts)
{
  using Timing = o2::its::LayerTiming;
  constexpr std::array fields{&Timing::mNROFsTF, &Timing::mROFLength, &Timing::mROFDelay,
                              &Timing::mROFBias, &Timing::mROFAddTimeErr};
  std::vector<Timing> baseline(Count::value);
  std::fill(baseline.begin(), baseline.end(), Timing{.mNROFsTF = 3, .mROFLength = 40});
  WorkflowSession session{"test", Count::value};
  const auto accept = [](int rof) { return rof < 3 && rof != 1; };
  const auto compareWithFresh = [&](const auto& timings) {
    session.configureTiming(timings, accept);
    WorkflowSession fresh{"oracle", Count::value};
    fresh.configureTiming(timings, accept);
    const auto actual = session.overlap.getView();
    const auto expected = fresh.overlap.getView();
    for (int layer = 0; layer < Count::value; ++layer) {
      for (auto field : fields) {
        BOOST_CHECK_EQUAL(actual.getLayer(layer).*field, timings[layer].*field);
        BOOST_CHECK_EQUAL(session.vertices.getView().getLayer(layer).*field, timings[layer].*field);
      }
      for (uint32_t rof = 0; rof < timings[layer].mNROFsTF; ++rof) {
        BOOST_CHECK_EQUAL(session.mask.getView().isROFEnabled(layer, rof), fresh.mask.getView().isROFEnabled(layer, rof));
        BOOST_CHECK_EQUAL(session.vertices.getView().getVertices(layer, rof).getEntries(), 0u);
        for (int to = 0; to < Count::value; ++to) {
          if (layer == to) {
            continue;
          }
          BOOST_CHECK_EQUAL(actual.getOverlap(layer, to, rof).getFirstEntry(), expected.getOverlap(layer, to, rof).getFirstEntry());
          BOOST_CHECK_EQUAL(actual.getOverlap(layer, to, rof).getEntries(), expected.getOverlap(layer, to, rof).getEntries());
        }
      }
    }
  };
  for (auto field : fields) {
    compareWithFresh(baseline);
    auto changed = baseline;
    for (auto& timing : changed) {
      timing.*field += 1;
    }
    compareWithFresh(changed);
    compareWithFresh(changed);  // Reuse must match the fresh oracle as well.
    compareWithFresh(baseline); // Includes shrinking the TF again.
    if (field != &Timing::mNROFsTF) {
      for (int layer = 0; layer < Count::value; ++layer) {
        auto nonuniform = baseline;
        nonuniform[layer].*field += 1;
        BOOST_CHECK_THROW(session.configureTiming(nonuniform, accept), TimeFrameLoadException);
      }
    }
  }
  // Uniformity constrains the four BC fields, but each layer has its own extent.
  for (int layer = 0; layer < Count::value; ++layer) {
    auto changed = baseline;
    changed[layer].mNROFsTF += 1;
    compareWithFresh(changed);
    compareWithFresh(baseline);
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(FilterFailureLeavesNoEventViewsAndDoesNotPoisonTimingReuse, Count, LayerCounts)
{
  WorkflowSession session{"test", Count::value};
  std::vector<o2::its::LayerTiming> timings(Count::value);
  std::fill(timings.begin(), timings.end(), o2::its::LayerTiming{.mNROFsTF = 3, .mROFLength = 40});
  session.configureTiming(timings, [](int) { return true; });
  for (bool changeTiming : {false, true}) {
    if (changeTiming) {
      for (auto& timing : timings) {
        timing.mROFLength += 1;
      }
    }
    session.publicationClock.emplace(session.overlap.getView().getClockLayer());
    BOOST_CHECK_THROW(session.configureTiming(timings, [](int rof) {
      if (rof == 1) {
        throw std::runtime_error{"filter failed"};
      }
      return true;
    }),
                      std::runtime_error);
    BOOST_CHECK_EQUAL(session.frame.getROFViews().overlap.mLayerCount, 0);
    BOOST_CHECK(!session.publicationClock);
    const auto storage = session.overlap.getView().mFlatTable;
    session.configureTiming(timings, [](int rof) { return rof == 2; });
    BOOST_CHECK(session.overlap.getView().mFlatTable == storage);
    for (int layer = 0; layer < Count::value; ++layer) {
      BOOST_CHECK(!session.frame.getROFViews().mask.isROFEnabled(layer, 0));
      BOOST_CHECK(!session.frame.getROFViews().mask.isROFEnabled(layer, 1));
      BOOST_CHECK(session.frame.getROFViews().mask.isROFEnabled(layer, 2));
      BOOST_CHECK_EQUAL(session.vertices.getView().getVertices(layer, 0).getEntries(), 0u);
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(InvalidTimingLayerCountPreservesCachedConfiguration, Count, LayerCounts)
{
  WorkflowSession session{"test", Count::value};
  std::vector<o2::its::LayerTiming> timings(Count::value, {.mNROFsTF = 3, .mROFLength = 40});
  const auto accept = [](int) { return true; };
  session.configureTiming(timings, accept);
  const auto cached = session.overlap.getView().mFlatTable;
  for (auto count : {0, Count::value - 1, Count::value + 1}) {
    auto invalid = timings;
    invalid.resize(count, timings.front());
    BOOST_CHECK_THROW(session.configureTiming(invalid, accept), TimeFrameLoadException);
    BOOST_CHECK(session.frame.getROFViews().overlap.mFlatTable == cached);
  }
  session.configureTiming(timings, accept);
  BOOST_CHECK(session.overlap.getView().mFlatTable == cached);
}
