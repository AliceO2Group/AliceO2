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

#define BOOST_TEST_MODULE ITSMFT CombinedTrackingComposition
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "TrackingParameterTestSupport.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <boost/test/unit_test.hpp>

#include <oneapi/tbb/task_arena.h>

#include <TGeoGlobalMagField.h>
#include "Field/MagneticField.h"

#include "CommonDataFormat/InteractionRecord.h"
#include "CombinedTrackingTestSupport.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "ITSMFTTracking/Tracker.h"
#include "ITSMFTTracking/Configuration.h"
#include "ITSMFTTracking/detail/TimeFrameScratch.h"
#include "ITSMFTTracking/IOUtils.h"
#include "ITSMFTTracking/ITSMFTDetectorDefinitions.h"
#include "ITSMFTTracking/SurfaceDescriptor.h"
#include "ITSMFTTracking/ClusterDecoding.h"
#include "ITSMFTTracking/TimeFrame.h"
#include "ITSMFTTracking/TrackerTraits.h"
#include "ITSMFTTracking/TrackingConfigParam.h"
#include "ITSMFTTracking/GenericTrackOutputAdapter.h"
#include "ITSMFTTracking/Constants.h"
#include "ReconstructionDataFormats/Track.h"

using namespace o2::itsmft;
using namespace o2::itsmft::tracking;

namespace
{

struct GenericTrackPublicationExport {
  o2::detectors::DetID::ID detector{};
  ClusterSourceId source{};
  ClockTimingPublicationView clock;
  gsl::span<const LayerId> layerMapping;
};

constexpr float Bz = 0.5f;
constexpr std::array<unsigned char, 3> OnePixelPattern{1, 1, 0x80};

const TopologyDictionary& dict()
{
  static const TopologyDictionary d;
  return d;
}

void ensureTrivialMagneticFieldIsSet()
{
  static const bool done = [] {
    TGeoGlobalMagField::Instance()->SetField(new o2::field::MagneticField());
    TGeoGlobalMagField::Instance()->Lock();
    return true;
  }();
  (void)done;
}

std::vector<LayerId> ordered(uint16_t first, uint16_t count)
{
  std::vector<LayerId> result;
  result.reserve(count);
  for (uint16_t i = 0; i < count; ++i) {
    result.push_back(LayerId{static_cast<uint16_t>(first + i)});
  }
  return result;
}

class PrescribedDecoder final : public ClusterDecoder
{
 public:
  PrescribedDecoder(o2::detectors::DetID::ID detector, SurfaceKind kind, std::vector<DecodedCluster> clusters)
    : mDetector{detector}, mKind{kind}, mClusters{std::move(clusters)}
  {
  }

  o2::itsmft::tracking::ClusterDecodeResult decode(
    const CompClusterExt& cluster,
    BoundedPatternCursor& patterns,
    const TopologyDictionary* dictionary,
    uint32_t externalIndex,
    bool) const final
  {
    const auto clusterData = o2::itsmft::ioutils::extractClusterDataBounded(cluster, patterns, dictionary);
    if (!clusterData.ok()) {
      o2::itsmft::tracking::ClusterDecodeResult result;
      result.error = clusterData.error;
      return result;
    }

    o2::itsmft::tracking::ClusterDecodeResult result;
    if (externalIndex >= mClusters.size()) {
      return result;
    }
    auto decoded = mClusters[externalIndex];
    decoded.shape = clusterData.shape;
    result.decoded = decoded;
    return result;
  }

 private:
  o2::detectors::DetID::ID mDetector;
  SurfaceKind mKind;
  std::vector<DecodedCluster> mClusters;
};

DecodedCluster diskCluster(float x, float y, float z, int layer)
{
  DecodedCluster cluster{};
  cluster.global = {x, y, z};
  cluster.rowColumnCovariance = {1.e-2f, 0.f, 1.e-2f};
  cluster.layer = layer;
  return cluster;
}

DecodedCluster cylinderCluster(float radius, float phi, float tanLambda, int layer)
{
  DecodedCluster cluster{};
  cluster.global = {radius * std::cos(phi), radius * std::sin(phi), radius * tanLambda};
  cluster.cylinderFrame = {cluster.global.x, cluster.global.y, cluster.global.z, 0.f};
  cluster.rowColumnCovariance = {1.e-2f, 0.f, 1.e-2f};
  cluster.layer = layer;
  return cluster;
}

/// A genuine, low-but-nonzero-curvature helical ITS barrel trajectory,
/// sampled at each nominal layer radius via the same standard O2 barrel-
/// propagation utility production ITS/TPC-matching code already uses
/// (o2::track::TrackPar::getXatLabR() to find the local x where the helix
/// crosses a given lab radius, then getXYZGloAt() to read the global point
/// there -- both const, no incremental state mutation between layers).
///
/// A perfectly collinear ("infinite pT" / zero-curvature) triple does not
/// define the linearized triplet factor used by cell construction. This
/// helix construction therefore supplies a deliberately non-degenerate ITS
/// road fixture.
std::vector<DecodedCluster> buildItsHelixChainClusters(const std::vector<float>& radii, float bz, float pt, float phi0, float tanl)
{
  const float px = pt * std::cos(phi0);
  const float py = pt * std::sin(phi0);
  const float pz = pt * tanl;
  o2::track::TrackPar seed(std::array<float, 3>{0.f, 0.f, 0.f}, std::array<float, 3>{px, py, pz}, 1, true);

  std::vector<DecodedCluster> clusters;
  clusters.reserve(radii.size());
  for (size_t layer = 0; layer < radii.size(); ++layer) {
    float xAtR = 0.f;
    if (!seed.getXatLabR(radii[layer], xAtR, bz, o2::track::DirType::DirOutward)) {
      return {};
    }
    bool ok = false;
    const auto point = seed.getXYZGloAt(xAtR, bz, ok);
    if (!ok) {
      return {};
    }
    DecodedCluster cluster{};
    cluster.global = {static_cast<float>(point.X()), static_cast<float>(point.Y()), static_cast<float>(point.Z())};
    cluster.cylinderFrame = {cluster.global.x, cluster.global.y, cluster.global.z, 0.f};
    cluster.rowColumnCovariance = {1.e-2f, 0.f, 1.e-2f};
    cluster.layer = static_cast<int>(layer);
    clusters.push_back(cluster);
  }
  return clusters;
}

TrackingParameters makeItsParams()
{
  TrackingParameters p;
  resetDetectorDefaults(p, o2::detectors::DetID::ITS);
  // Tracklet formation needs a primary vertex to seed the search window
  // (TrackerTraits.cxx's forTracklets()): with UseDiamond=false (ITS's own
  // default) that must come from TimeFrame::getPrimaryVertices(), which
  // these focused fixtures never populate. UseDiamond=true instead uses the
  // fixed Diamond{0,0,0} vertex every synthetic radial chain below is built
  // through, with no TimeFrame vertex needed.
  p.UseDiamond = true;
  return p;
}

TrackingParameters makeMftParams()
{
  TrackingParameters p;
  resetDetectorDefaults(p, o2::detectors::DetID::MFT);
  p.UseDiamond = true;
  p.CreateArtefactLabels = false;
  return p;
}

/// Encodes `decoded` as compact/pattern input and returns a
/// ClusterSourceInput referencing `decoder`/`compactOut`/`patternsOut`/
/// `rofsOut` (kept alive by the caller for the lifetime of every process()
/// call that uses it).
ClusterSourceInput makeSource(ClusterSourceId id, o2::detectors::DetID::ID det, const std::vector<LayerId>& surfaces,
                              const PrescribedDecoder& decoder, std::vector<CompClusterExt>& compactOut,
                              std::vector<unsigned char>& patternsOut, std::vector<ROFRecord>& rofsOut,
                              const std::vector<DecodedCluster>& decoded)
{
  compactOut.reserve(decoded.size());
  patternsOut.reserve(decoded.size() * OnePixelPattern.size());
  for (const auto& cluster : decoded) {
    compactOut.emplace_back(0, 0, CompCluster::InvalidPatternID, cluster.layer);
    patternsOut.insert(patternsOut.end(), OnePixelPattern.begin(), OnePixelPattern.end());
  }
  rofsOut = {ROFRecord{{100, 5}, 0, 0, static_cast<int>(compactOut.size())}};

  ClusterSourceInput source{};
  source.id = id;
  source.detector = det;
  source.clusters = compactOut;
  source.patterns = patternsOut;
  source.rofs = rofsOut;
  source.dictionary = &dict();
  source.layerToSurface = surfaces;
  source.timing = ROFTimingConfig{40, 0, 0, 0};
  source.decoder = &decoder;
  return source;
}

/// A source that is valid (dense-empty ROF, zero clusters) but describes no
/// hits at all -- the composition's own required "the other detector may be
/// empty" shape, matching the standalone workflow's zero-cluster path.
ClusterSourceInput makeEmptySource(ClusterSourceId id, o2::detectors::DetID::ID det, const std::vector<LayerId>& surfaces,
                                   const PrescribedDecoder& decoder)
{
  ClusterSourceInput source{};
  source.id = id;
  source.detector = det;
  source.dictionary = &dict();
  source.layerToSurface = surfaces;
  source.timing = ROFTimingConfig{40, 0, 0, 0};
  source.decoder = &decoder;
  return source;
}

/// Independent, non-combined, single-detector reference run: the same shape
/// the standalone path already uses -- global
/// LayerIds equal compact scratch slots, with the same plan-driven binding
/// model as the combined path. Used as the "reproduce the standalone oracle
/// count" reference for the combined composition.
template <o2::detectors::DetID::ID DetId, int NLayers>
struct StandaloneRun {
  TimeFrame frame;
  std::vector<TrackingParameters> params;
  std::shared_ptr<BoundedMemoryResource> pool = std::make_shared<BoundedMemoryResource>();
  Tracker tracker;
  TrackerTraits traits;
  std::shared_ptr<tbb::task_arena> arena;
  TimeFrameScratch* scratch = nullptr;
  std::vector<SurfaceDescriptor> catalog;
  TrackingResult result;

  StandaloneRun(o2::detectors::DetID::ID det, SurfaceKind kind,
                const TrackingParameters& singleParams, const std::vector<DecodedCluster>& decoded,
                int rofLength = 40, LayerMask holeLayers = {})
    : params{singleParams}
  {
    const auto orderedSurfaces = ordered(0, NLayers);
    catalog.reserve(NLayers);
    for (uint16_t i = 0; i < NLayers; ++i) {
      SurfaceDescriptor surface{i, static_cast<uint8_t>(det), kind};
      surface.chartRange = kind == SurfaceKind::Disk ? SurfaceChartRange{kMFTLookupRMin[i], kMFTLookupRMax[i]} : SurfaceChartRange{-20.f, 20.f};
      surface.referenceCoordinate = kind == SurfaceKind::Cylinder
                                      ? singleParams.LayerRadii[i]
                                      : kMFTStaticSurfaceCatalog[i].referenceCoordinate;
      const float xOverX0 = det == o2::detectors::DetID::MFT ? kNominalMFTLayerX0[i] : kNominalITSLayerX0[i];
      surface.material.xOverX0 = xOverX0;
      surface.material.arealDensityGPerCm2 = xOverX0 * o2::its::constants::Radl * o2::its::constants::Rho;
      catalog.push_back(surface);
    }
    const SurfaceCatalogView catalogView{catalog.data(), static_cast<uint32_t>(catalog.size())};
    TrackerInitialization configuration;
    configuration.catalog = catalogView;
    configuration.memoryPool = pool;
    configuration.layout = makeDetectorLayout(holeLayers);
    configuration.plan = o2::itsmft::tracking::test::makeTrackingPlan(singleParams);
    const auto configured = tracker.initialize(frame, configuration);
    BOOST_REQUIRE(configured.ok());
    scratch = &frame.getScratch();
    traits.setNThreads(1, arena);
    frame.setBz(Bz);

    std::vector<CompClusterExt> compact;
    std::vector<unsigned char> patterns;
    for (const auto& cluster : decoded) {
      compact.emplace_back(0, 0, CompCluster::InvalidPatternID, cluster.layer);
      patterns.insert(patterns.end(), OnePixelPattern.begin(), OnePixelPattern.end());
    }
    const std::vector<ROFRecord> rofs{ROFRecord{{100, 5}, 0, 0, static_cast<int>(compact.size())}};
    PrescribedDecoder decoder{det, kind, decoded};
    const auto layerMapping = ordered(0, NLayers);
    const auto load = loadTimeFrameSource(frame, decoder, o2::InteractionRecord{50, 5}, ROFTimingConfig{rofLength, 0, 0, 0},
                                          compact, patterns, rofs, &dict(), nullptr, det,
                                          gsl::span<const LayerId>{layerMapping},
                                          frame.getLayout().getSurfaceCatalog());
    BOOST_REQUIRE(load.ok());

    o2::its::LayerTiming layerTiming{};
    layerTiming.mNROFsTF = 1;
    layerTiming.mROFLength = rofLength;
    o2::its::ROFOverlapTable<NLayers> rofTable;
    for (int layer = 0; layer < NLayers; ++layer) {
      rofTable.defineLayer(layer, layerTiming);
    }
    rofTable.init();
    o2::its::ROFVertexLookupTable<NLayers> vtxTable;
    for (int layer = 0; layer < NLayers; ++layer) {
      vtxTable.defineLayer(layer, layerTiming);
    }
    vtxTable.init();
    o2::its::ROFMaskTable<NLayers> mask{rofTable};
    mask.resetMask();
    for (int layer = 0; layer < NLayers; ++layer) {
      mask.setROFsEnabled(layer, 0, 1, 1);
    }
    frame.setROFViews(RuntimeROFViews{rofTable.getView(), vtxTable.getView(), mask.getView(), {}});
    const auto tracking = tracker.run(frame, traits);
    result.outcome = tracking.outcome;
  }
};

/// Test-only reproduction of the whole-event load/track/publish composition
/// the combined DPL task's own trackFrame() applies -- not a shipped
/// coordinator class (M3 deleted the last one of those), just this file's
/// own driver so these tests can exercise the workflow-owned application plan
/// plus Tracker + loadTimeFrameSources() together the same way the
/// DPL task does, without a DPL ProcessingContext.
struct CombinedTrackingComposer {
  struct Result {
    TrackingOutcome outcome{TrackingOutcome::Structural};
    size_t nITSTracks{0};
    size_t nMFTTracks{0};
  };

  test::CombinedTrackingPlan plan;
  TimeFrame* frame = nullptr;
  std::optional<ClockTimingPublicationView> itsClock;
  std::optional<ClockTimingPublicationView> mftClock;
  bool publicationValid = false;

  CombinedTrackingComposer(std::vector<TrackingParameters> itsParams, std::vector<TrackingParameters> mftParams)
    : plan(std::move(itsParams), std::move(mftParams))
  {
  }

  void adoptFrame(TimeFrame& f)
  {
    frame = &f;
    plan.adoptFrame(f);
  }
  void setBz(float bz) { plan.setBz(bz); }
  void setNThreads(int n) { plan.setNThreads(n); }

  void clearPublicationSidecars() noexcept
  {
    plan.clearPublicationSidecars();
  }
  void invalidatePublication() noexcept
  {
    itsClock.reset();
    mftClock.reset();
    publicationValid = false;
  }
  void markPublicationValid() noexcept
  {
    itsClock.emplace(plan.getITSROFViews().overlap.getClockLayer());
    mftClock.emplace(plan.getMFTROFViews().overlap.getClockLayer());
    publicationValid = true;
  }
  std::optional<GenericTrackPublicationExport> getITSPublicationExport() const
  {
    if (!publicationValid || !itsClock) {
      return std::nullopt;
    }
    return GenericTrackPublicationExport{o2::detectors::DetID::ITS, ClusterSourceId{0}, *itsClock,
                                         plan.getITSLayerMapping()};
  }
  std::optional<GenericTrackPublicationExport> getMFTPublicationExport() const
  {
    if (!publicationValid || !mftClock) {
      return std::nullopt;
    }
    return GenericTrackPublicationExport{o2::detectors::DetID::MFT, ClusterSourceId{1}, *mftClock,
                                         plan.getMFTLayerMapping()};
  }

  Result process(const ClusterSourceInput& itsSource, const ClusterSourceInput& mftSource, const o2::InteractionRecord& origin)
  {
    invalidatePublication();
    clearPublicationSidecars();

    plan.configureRofTables(itsSource, mftSource);
    auto itsInput = itsSource;
    auto mftInput = mftSource;
    itsInput.rofViews = plan.getITSROFViews();
    mftInput.rofViews = plan.getMFTROFViews();
    LoadSourcesResult loadResult;
    if (const auto rejected = plan.validateSources(itsSource, mftSource)) {
      loadResult = *rejected;
    } else {
      const std::array<ClusterSourceInput, 2> sources{itsInput, mftInput};
      loadResult = loadTimeFrameSources(*frame, gsl::span<const ClusterSourceInput>{sources}, plan.catalogView(), origin);
    }
    if (!loadResult.ok()) {
      const bool errorIsRecoverable = isRecoverableLoadError(loadResult.error, loadResult.timingDetail);
      const auto dropAllowed = plan.dropTFUponFailureFor(loadResult.source);
      const bool sourceRecognized = dropAllowed.has_value();
      const auto outcome = errorIsRecoverable && sourceRecognized && dropAllowed.value_or(false)
                             ? TrackingOutcome::RecoverableDropped
                             : TrackingOutcome::Structural;
      plan.clearPublicationSidecars();
      frame->resetTimeFrame();
      invalidatePublication();
      return {outcome, 0, 0};
    }

    try {
      const auto itsResult = plan.runITS();
      if (itsResult.outcome != TrackingOutcome::Success) {
        plan.clearPublicationSidecars();
        frame->resetTimeFrame();
        invalidatePublication();
        return {itsResult.outcome, 0, 0};
      }
      const auto mftResult = plan.runMFT();
      if (mftResult.outcome != TrackingOutcome::Success) {
        plan.clearPublicationSidecars();
        frame->resetTimeFrame();
        invalidatePublication();
        return {mftResult.outcome, 0, 0};
      }
    } catch (const std::exception&) {
      plan.clearPublicationSidecars();
      frame->resetTimeFrame();
      invalidatePublication();
      return {TrackingOutcome::Structural, 0, 0};
    }

    markPublicationValid();
    const auto countFor = [this](int first) {
      return static_cast<size_t>(std::count_if(this->frame->getGenericTracks().begin(), this->frame->getGenericTracks().end(),
                                               [first](const auto& track) { return track.hitLayers.has(first); }));
    };
    return {TrackingOutcome::Success, countFor(0), countFor(ITSNLayers)};
  }

  const TimeFrameScratch& getITSScratch() const noexcept { return plan.getITSScratch(); }
  const TimeFrameScratch& getMFTScratch() const noexcept { return plan.getMFTScratch(); }
  gsl::span<const uint8_t> getITSSharedClusterFlags() const noexcept { return plan.getITSSharedClusterFlags(); }
  gsl::span<const LayerId> getITSLayerMapping() const noexcept { return plan.getITSLayerMapping(); }
  gsl::span<const LayerId> getMFTLayerMapping() const noexcept { return plan.getMFTLayerMapping(); }
};

CombinedTrackingComposer makeComposer(const TrackingParameters& itsParams, const TrackingParameters& mftParams)
{
  return CombinedTrackingComposer{std::vector<TrackingParameters>{itsParams}, std::vector<TrackingParameters>{mftParams}};
}

template <o2::detectors::DetID::ID DetId, int NLayers>
void checkMinimumHitLayers(SurfaceKind kind, TrackingParameters params, std::vector<DecodedCluster> clusters)
{
  ensureTrivialMagneticFieldIsSet();
  BOOST_REQUIRE_EQUAL(clusters.size(), static_cast<size_t>(NLayers));
  const LayerMask allowedHoles{1u << 3};
  params.MaxHoles = 1;
  params.MinTrackLength = NLayers - 1;

  // Exercise both an internal hole (span exceeds hit count) and a missing
  // endpoint (no internal hole, but MaxHoles must not lower the minimum).
  for (const int missingLayer : {3, NLayers - 1}) {
    BOOST_TEST_CONTEXT("missing layer " << missingLayer)
    {
      auto incomplete = clusters;
      incomplete.erase(incomplete.begin() + missingLayer);
      StandaloneRun<DetId, NLayers> accepted{DetId, kind, params, incomplete, 40, allowedHoles};
      BOOST_REQUIRE(accepted.result.outcome == TrackingOutcome::Success);
      BOOST_REQUIRE_EQUAL(accepted.frame.getGenericTracks().size(), 1u);
      BOOST_CHECK_EQUAL(accepted.frame.getGenericTracks().front().hitLayers.count(), NLayers - 1);
      BOOST_CHECK(!accepted.frame.getGenericTracks().front().hitLayers.has(missingLayer));

      auto stricter = params;
      stricter.MinTrackLength = NLayers;
      StandaloneRun<DetId, NLayers> rejected{DetId, kind, stricter, incomplete, 40, allowedHoles};
      BOOST_REQUIRE(rejected.result.outcome == TrackingOutcome::Success);
      BOOST_CHECK(rejected.frame.getGenericTracks().empty());
    }
  }

  // A skipped non-seeding surface is not a hole; it still cannot contribute
  // a hit toward MinTrackLength.
  clusters.erase(clusters.begin() + 3);
  params.MaxHoles = 0;
  params.SeedingLayers = LayerMask::span(0, NLayers - 1) & ~allowedHoles;
  StandaloneRun<DetId, NLayers> sparseAccepted{DetId, kind, params, clusters};
  BOOST_REQUIRE(sparseAccepted.result.outcome == TrackingOutcome::Success);
  BOOST_REQUIRE_EQUAL(sparseAccepted.frame.getGenericTracks().size(), 1u);
  BOOST_CHECK_EQUAL(sparseAccepted.frame.getGenericTracks().front().hitLayers.count(), NLayers - 1);
  params.MinTrackLength = NLayers;
  StandaloneRun<DetId, NLayers> sparseRejected{DetId, kind, params, clusters};
  BOOST_REQUIRE(sparseRejected.result.outcome == TrackingOutcome::Success);
  BOOST_CHECK(sparseRejected.frame.getGenericTracks().empty());
}

} // namespace

BOOST_AUTO_TEST_CASE(CylinderRoadMinimumCountsHitLayers)
{
  const auto params = makeItsParams();
  checkMinimumHitLayers<o2::detectors::DetID::ITS, ITSNLayers>(
    SurfaceKind::Cylinder, params, buildItsHelixChainClusters(params.LayerRadii, Bz, 1.f, 0.4f, 0.3f));
}

BOOST_AUTO_TEST_CASE(CombinedLoadingBackfillsOneGlobalWorkspace)
{
  // TrackerTraits::findRoads() unconditionally touches the global
  // o2::base::Propagator singleton on first use, regardless of whether any
  // road is actually found -- required before any clustersToTracks() call.
  ensureTrivialMagneticFieldIsSet();
  const auto itsSurfaces = ordered(0, ITSNLayers);
  const auto mftSurfaces = ordered(ITSNLayers, MFTNLayers);
  const auto itsClusters = std::vector<DecodedCluster>{cylinderCluster(3.f, 0.2f, 0.1f, 0), cylinderCluster(4.f, 0.2f, 0.1f, 1)};
  const auto mftClusters = std::vector<DecodedCluster>{diskCluster(1.f, 0.5f, kMFTStaticSurfaceCatalog[0].referenceCoordinate, 0), diskCluster(1.f, 0.5f, kMFTStaticSurfaceCatalog[1].referenceCoordinate, 1)};

  PrescribedDecoder itsDecoder{o2::detectors::DetID::ITS, SurfaceKind::Cylinder, itsClusters};
  PrescribedDecoder mftDecoder{o2::detectors::DetID::MFT, SurfaceKind::Disk, mftClusters};
  std::vector<CompClusterExt> itsCompact, mftCompact;
  std::vector<unsigned char> itsPatterns, mftPatterns;
  std::vector<ROFRecord> itsRofs, mftRofs;
  const auto itsSource = makeSource(ClusterSourceId{0}, o2::detectors::DetID::ITS, itsSurfaces, itsDecoder, itsCompact, itsPatterns, itsRofs, itsClusters);
  const auto mftSource = makeSource(ClusterSourceId{1}, o2::detectors::DetID::MFT, mftSurfaces, mftDecoder, mftCompact, mftPatterns, mftRofs, mftClusters);

  auto itsParams = makeItsParams();
  auto mftParams = makeMftParams();
  itsParams.MinTrackLength = 4;
  mftParams.MinTrackLength = 5;
  auto composer = makeComposer(itsParams, mftParams);
  TimeFrame frame;
  composer.adoptFrame(frame);
  composer.setBz(Bz);
  composer.setNThreads(1);

  constexpr uint32_t allCombinedSurfaces = (uint32_t{1} << (ITSNLayers + MFTNLayers)) - 1u;
  BOOST_REQUIRE_EQUAL(composer.plan.itsTracker().getIterationConfigurations().size(), 1u);
  const auto& combined = composer.plan.itsTracker().getIterationConfigurations()[0].parameters;
  const auto& detector = composer.plan.itsTracker().getDetectorConfiguration();
  BOOST_CHECK_EQUAL(combined.NLayers, ITSNLayers + MFTNLayers);
  BOOST_CHECK_EQUAL(combined.StartLayerMask.value(), allCombinedSurfaces);
  BOOST_CHECK(combined.PassFlags == itsParams.PassFlags);
  BOOST_REQUIRE(detector.indexTableConfigs.size() > 0);
  BOOST_CHECK_EQUAL(detector.indexTableConfigs[0].getNcolBins(), itsParams.ColBins);
  BOOST_CHECK_EQUAL(detector.indexTableConfigs[0].getNrowBins(), itsParams.RowBins);
  BOOST_CHECK_EQUAL(combined.UseDiamond, itsParams.UseDiamond);
  BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(combined.Diamond), std::end(combined.Diamond),
                                std::begin(itsParams.Diamond), std::end(itsParams.Diamond));
  BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(combined.DiamondCov), std::end(combined.DiamondCov),
                                std::begin(itsParams.DiamondCov), std::end(itsParams.DiamondCov));
  BOOST_CHECK_EQUAL(combined.MinTrackLength, itsParams.MinTrackLength);
  BOOST_CHECK_EQUAL(combined.MaxHoles, itsParams.MaxHoles);
  BOOST_CHECK_EQUAL(combined.NSigmaCut, itsParams.NSigmaCut);
  BOOST_CHECK_EQUAL(combined.PVres, itsParams.PVres);
  BOOST_CHECK_EQUAL(combined.TrackletMinPt, itsParams.TrackletMinPt);
  BOOST_CHECK(combined.CorrType == itsParams.CorrType);
  BOOST_CHECK_EQUAL(combined.MaxChi2ClusterAttachment, itsParams.MaxChi2ClusterAttachment);
  BOOST_CHECK_EQUAL(combined.MaxChi2NDF, itsParams.MaxChi2NDF);
  BOOST_CHECK_EQUAL(combined.ReseedIfShorter, itsParams.ReseedIfShorter);
  BOOST_CHECK_EQUAL_COLLECTIONS(combined.MinPt.begin(), combined.MinPt.end(), itsParams.MinPt.begin(), itsParams.MinPt.end());
  BOOST_CHECK_EQUAL(combined.RepeatRefitOut, itsParams.RepeatRefitOut);
  BOOST_CHECK_EQUAL(combined.ShiftRefToCluster, itsParams.ShiftRefToCluster);
  BOOST_CHECK_EQUAL(combined.PerPrimaryVertexProcessing, itsParams.PerPrimaryVertexProcessing);
  BOOST_CHECK_EQUAL(combined.AllowSharingFirstCluster, itsParams.AllowSharingFirstCluster);
  BOOST_CHECK_EQUAL(combined.SharedClusterMaxDeltaPhi, itsParams.SharedClusterMaxDeltaPhi);
  BOOST_CHECK_EQUAL(combined.SharedClusterMaxDeltaEta, itsParams.SharedClusterMaxDeltaEta);
  BOOST_CHECK_EQUAL(combined.SharedClusterOppositeSign, itsParams.SharedClusterOppositeSign);
  BOOST_CHECK_EQUAL(combined.SharedMaxClusters, itsParams.SharedMaxClusters);

  const auto checkConcatenated = [](const auto& actual, const auto& itsValues, const auto& mftValues) {
    BOOST_REQUIRE_EQUAL(actual.size(), itsValues.size() + mftValues.size());
    BOOST_CHECK_EQUAL_COLLECTIONS(actual.begin(), actual.begin() + itsValues.size(), itsValues.begin(), itsValues.end());
    BOOST_CHECK_EQUAL_COLLECTIONS(actual.begin() + itsValues.size(), actual.end(), mftValues.begin(), mftValues.end());
  };
  checkConcatenated(detector.addTimeError, itsParams.AddTimeError, mftParams.AddTimeError);
  checkConcatenated(detector.layerRadii, itsParams.LayerRadii, mftParams.LayerRadii);
  checkConcatenated(detector.layerResolution, itsParams.LayerResolution, mftParams.LayerResolution);
  checkConcatenated(detector.systError2Row, itsParams.SystError2Row, mftParams.SystError2Row);
  checkConcatenated(detector.systError2Col, itsParams.SystError2Col, mftParams.SystError2Col);
  const auto catalog = frame.getLayout().getSurfaceCatalog();
  BOOST_REQUIRE_EQUAL(catalog.nSurfaces, ITSNLayers + MFTNLayers);
  for (uint32_t layer = 0; layer < catalog.nSurfaces; ++layer) {
    const auto expected = layer < ITSNLayers ? kITSStaticSurfaceCatalog[layer].material.xOverX0 : kMFTStaticSurfaceCatalog[layer - ITSNLayers].material.xOverX0;
    BOOST_CHECK_EQUAL(catalog.surfaces[layer].material.xOverX0, expected);
  }

  const auto result = composer.process(itsSource, mftSource, o2::InteractionRecord{50, 5});
  BOOST_REQUIRE(result.outcome == TrackingOutcome::Success);

  // The time frame owns two lookup records independently of the tracker cache.
  BOOST_CHECK_EQUAL(&frame.getIndexTableUtils(0), &frame.getIndexTableUtils(ITSNLayers - 1));
  BOOST_CHECK_EQUAL(&frame.getIndexTableUtils(ITSNLayers), &frame.getIndexTableUtils(ITSNLayers + MFTNLayers - 1));
  BOOST_CHECK(&frame.getIndexTableUtils(0) != &detector.indexTableConfigs[0]);
  BOOST_CHECK(frame.getIndexTableUtils(0).getCoordType() == IndexTableCoordType::PhiZ);
  BOOST_CHECK(frame.getIndexTableUtils(ITSNLayers).getCoordType() == IndexTableCoordType::PhiR);

  const auto topology = composer.plan.itsTracker().getIterationConfigurations()[0].getTopologyView(frame.getLayout().getSurfaceCatalog());
  BOOST_CHECK_EQUAL(topology.seedingLayers.value(), allCombinedSurfaces);
  BOOST_REQUIRE_EQUAL(topology.nEdges, static_cast<uint32_t>(ITSNLayers + MFTNLayers - 2));
  for (uint16_t edgeId = 0; edgeId < topology.nEdges; ++edgeId) {
    const auto& edge = topology.getEdge(EdgeId{edgeId});
    const bool fromITS = edge.from.value() < ITSNLayers;
    const bool toITS = edge.to.value() < ITSNLayers;
    BOOST_CHECK_EQUAL(fromITS, toITS);
    BOOST_CHECK(!(edge.from == LayerId{ITSNLayers - 1} && edge.to == LayerId{ITSNLayers}));
  }

  BOOST_CHECK_EQUAL(composer.frame->getTotalClusters(),
                    static_cast<int>(itsClusters.size() + mftClusters.size()));
  BOOST_CHECK_EQUAL(&composer.getITSScratch(), &composer.getMFTScratch());
  // The one workspace keeps source-local ROF numbering per global surface.
  BOOST_CHECK_EQUAL(composer.frame->getNrof(0), 1);
  BOOST_CHECK_EQUAL(composer.frame->getNrof(ITSNLayers), 1);
}

BOOST_AUTO_TEST_CASE(LoadFailureResetsWholeCombinedTFExactlyOnceAndInvalidatesPublication)
{
  ensureTrivialMagneticFieldIsSet();
  const auto itsSurfaces = ordered(0, ITSNLayers);
  const auto mftSurfaces = ordered(ITSNLayers, MFTNLayers);
  const auto itsClusters = std::vector<DecodedCluster>{cylinderCluster(3.f, 0.2f, 0.1f, 0), cylinderCluster(4.f, 0.2f, 0.1f, 1)};
  const auto mftClusters = std::vector<DecodedCluster>{diskCluster(1.f, 0.5f, kMFTStaticSurfaceCatalog[0].referenceCoordinate, 0), diskCluster(1.f, 0.5f, kMFTStaticSurfaceCatalog[1].referenceCoordinate, 1)};

  PrescribedDecoder itsDecoder{o2::detectors::DetID::ITS, SurfaceKind::Cylinder, itsClusters};
  PrescribedDecoder mftDecoder{o2::detectors::DetID::MFT, SurfaceKind::Disk, mftClusters};
  std::vector<CompClusterExt> itsCompact, mftCompact;
  std::vector<unsigned char> itsPatterns, mftPatterns;
  std::vector<ROFRecord> itsRofs, mftRofs;
  const auto itsSource = makeSource(ClusterSourceId{0}, o2::detectors::DetID::ITS, itsSurfaces, itsDecoder, itsCompact, itsPatterns, itsRofs, itsClusters);
  auto mftSource = makeSource(ClusterSourceId{1}, o2::detectors::DetID::MFT, mftSurfaces, mftDecoder, mftCompact, mftPatterns, mftRofs, mftClusters);

  auto composer = makeComposer(makeItsParams(), makeMftParams());
  TimeFrame frame;
  composer.adoptFrame(frame);
  composer.setBz(Bz);
  composer.setNThreads(1);

  // First pass genuinely succeeds, so there is real state (scratches,
  // GenericTracks, publication exports) for the second, failing pass to
  // actually have to clear.
  const auto first = composer.process(itsSource, mftSource, o2::InteractionRecord{50, 5});
  BOOST_REQUIRE(first.outcome == TrackingOutcome::Success);
  BOOST_REQUIRE(composer.getITSPublicationExport().has_value());
  BOOST_REQUIRE(composer.getMFTPublicationExport().has_value());

  // Malformed MFT ROF partition (a gap before the second cluster): a
  // structural load failure loadTimeFrameSources() must
  // reject before touching either scratch or the shared TimeFrame.
  std::vector<ROFRecord> malformedMftRofs{ROFRecord{{100, 5}, 0, 0, 1}, ROFRecord{{140, 5}, 0, 2, 1}};
  mftSource.rofs = malformedMftRofs;

  const auto second = composer.process(itsSource, mftSource, o2::InteractionRecord{50, 5});
  // MFT's own DropTFUponFailure defaults false (makeMftParams() never sets
  // it), so this recoverable InvalidROFRange load error is still classified
  // Structural.
  BOOST_CHECK(second.outcome == TrackingOutcome::Structural);
  BOOST_CHECK_EQUAL(second.nITSTracks, 0u);
  BOOST_CHECK_EQUAL(second.nMFTTracks, 0u);

  BOOST_CHECK_EQUAL(composer.frame->getTotalClusters(), 0);
  BOOST_CHECK_EQUAL(composer.frame->getTotalClusters(), 0);
  BOOST_CHECK(frame.getGenericTracks().empty());
  BOOST_CHECK(frame.getTrackClusterIndices().empty());
  BOOST_CHECK(!composer.getITSPublicationExport().has_value());
  BOOST_CHECK(!composer.getMFTPublicationExport().has_value());
}

BOOST_AUTO_TEST_CASE(CombinedTrackingResourceFailureUsesSharedPolicyAndResetsWorkspace)
{
  ensureTrivialMagneticFieldIsSet();
  const auto itsSurfaces = ordered(0, ITSNLayers);
  const auto mftSurfaces = ordered(ITSNLayers, MFTNLayers);
  const auto itsClusters = std::vector<DecodedCluster>{cylinderCluster(3.f, 0.2f, 0.1f, 0), cylinderCluster(4.f, 0.2f, 0.1f, 1)};
  const auto mftClusters = std::vector<DecodedCluster>{diskCluster(1.f, 0.5f, kMFTStaticSurfaceCatalog[0].referenceCoordinate, 0), diskCluster(1.f, 0.5f, kMFTStaticSurfaceCatalog[1].referenceCoordinate, 1)};

  PrescribedDecoder itsDecoder{o2::detectors::DetID::ITS, SurfaceKind::Cylinder, itsClusters};
  PrescribedDecoder mftDecoder{o2::detectors::DetID::MFT, SurfaceKind::Disk, mftClusters};
  std::vector<CompClusterExt> itsCompact, mftCompact;
  std::vector<unsigned char> itsPatterns, mftPatterns;
  std::vector<ROFRecord> itsRofs, mftRofs;
  const auto itsSource = makeSource(ClusterSourceId{0}, o2::detectors::DetID::ITS, itsSurfaces, itsDecoder, itsCompact, itsPatterns, itsRofs, itsClusters);
  const auto mftSource = makeSource(ClusterSourceId{1}, o2::detectors::DetID::MFT, mftSurfaces, mftDecoder, mftCompact, mftPatterns, mftRofs, mftClusters);

  // One run has one resource budget and one failure policy. The combined
  // scalar baseline is ITS, so exhausting that budget drops and resets the
  // one frame-owned workspace atomically.
  auto itsParams = makeItsParams();
  itsParams.MaxMemory = 1;
  itsParams.DropTFUponFailure = true;

  auto composer = makeComposer(itsParams, makeMftParams());
  TimeFrame frame;
  composer.adoptFrame(frame);
  composer.setBz(Bz);
  composer.setNThreads(1);

  const auto result = composer.process(itsSource, mftSource, o2::InteractionRecord{50, 5});
  BOOST_CHECK(result.outcome == TrackingOutcome::RecoverableDropped);
  BOOST_CHECK_EQUAL(composer.frame->getTotalClusters(), 0);
  BOOST_CHECK_EQUAL(&composer.getITSScratch(), &composer.getMFTScratch());
  BOOST_CHECK(frame.getGenericTracks().empty());
  BOOST_CHECK(!composer.getITSPublicationExport().has_value());
  BOOST_CHECK(!composer.getMFTPublicationExport().has_value());
}

namespace
{

/// A minimal, always-valid ITS+MFT source pair sharing the two-cluster
/// fixture already used by CombinedLoadingBackfillsIndependentCompactScratches.
struct MinimalFixture {
  std::vector<LayerId> itsSurfaces = ordered(0, ITSNLayers);
  std::vector<LayerId> mftSurfaces = ordered(ITSNLayers, MFTNLayers);
  std::vector<DecodedCluster> itsClusters{cylinderCluster(3.f, 0.2f, 0.1f, 0), cylinderCluster(4.f, 0.2f, 0.1f, 1)};
  std::vector<DecodedCluster> mftClusters{diskCluster(1.f, 0.5f, kMFTStaticSurfaceCatalog[0].referenceCoordinate, 0), diskCluster(1.f, 0.5f, kMFTStaticSurfaceCatalog[1].referenceCoordinate, 1)};
  PrescribedDecoder itsDecoder{o2::detectors::DetID::ITS, SurfaceKind::Cylinder, itsClusters};
  PrescribedDecoder mftDecoder{o2::detectors::DetID::MFT, SurfaceKind::Disk, mftClusters};
  std::vector<CompClusterExt> itsCompact, mftCompact;
  std::vector<unsigned char> itsPatterns, mftPatterns;
  std::vector<ROFRecord> itsRofs, mftRofs;
  ClusterSourceInput itsSource;
  ClusterSourceInput mftSource;

  MinimalFixture()
  {
    itsSource = makeSource(ClusterSourceId{0}, o2::detectors::DetID::ITS, itsSurfaces, itsDecoder, itsCompact, itsPatterns, itsRofs, itsClusters);
    mftSource = makeSource(ClusterSourceId{1}, o2::detectors::DetID::MFT, mftSurfaces, mftDecoder, mftCompact, mftPatterns, mftRofs, mftClusters);
  }
};

/// A malformed (gap-before-second-cluster) ROF partition for one detector's
/// source, reproducing MultiSourceLoadError::InvalidROFRange -- a
/// *recoverable* per-TF data error under isRecoverableLoadError()
/// (TimeFrameLoadFailure.cxx) -- without touching the other detector's
/// (still valid) source.
void makeRofGap(std::vector<ROFRecord>& rofs)
{
  rofs = {ROFRecord{{100, 5}, 0, 0, 1}, ROFRecord{{140, 5}, 0, 2, 1}};
}

} // namespace

BOOST_AUTO_TEST_CASE(RecoverableITSLoadFailureIsDroppedOnlyWhenITSDropTFAllows)
{
  ensureTrivialMagneticFieldIsSet();

  for (const bool itsDropTF : {true, false}) {
    MinimalFixture fixture;
    makeRofGap(fixture.itsRofs);
    fixture.itsSource.rofs = fixture.itsRofs;

    auto itsParams = makeItsParams();
    itsParams.DropTFUponFailure = itsDropTF;
    auto composer = makeComposer(itsParams, makeMftParams());
    TimeFrame frame;
    composer.adoptFrame(frame);
    composer.setBz(Bz);
    composer.setNThreads(1);

    const auto result = composer.process(fixture.itsSource, fixture.mftSource, o2::InteractionRecord{50, 5});
    const auto expected = itsDropTF ? TrackingOutcome::RecoverableDropped : TrackingOutcome::Structural;
    BOOST_CHECK_MESSAGE(result.outcome == expected, "ITS DropTFUponFailure=" << itsDropTF);
    // Every non-success path still performs exactly one whole reset:
    // both scratches, the shared TimeFrame's GenericTracks, and both
    // publication exports are empty/invalid regardless of classification.
    BOOST_CHECK_EQUAL(composer.frame->getTotalClusters(), 0);
    BOOST_CHECK_EQUAL(composer.frame->getTotalClusters(), 0);
    BOOST_CHECK(frame.getGenericTracks().empty());
    BOOST_CHECK(!composer.getITSPublicationExport().has_value());
    BOOST_CHECK(!composer.getMFTPublicationExport().has_value());
  }
}

BOOST_AUTO_TEST_CASE(RecoverableMFTLoadFailureUsesSharedCombinedDropPolicy)
{
  ensureTrivialMagneticFieldIsSet();

  for (const bool combinedDropTF : {true, false}) {
    MinimalFixture fixture;
    makeRofGap(fixture.mftRofs);
    fixture.mftSource.rofs = fixture.mftRofs;

    auto itsParams = makeItsParams();
    itsParams.DropTFUponFailure = combinedDropTF;
    auto mftParams = makeMftParams();
    mftParams.DropTFUponFailure = !combinedDropTF;
    auto composer = makeComposer(itsParams, mftParams);
    TimeFrame frame;
    composer.adoptFrame(frame);
    composer.setBz(Bz);
    composer.setNThreads(1);

    const auto result = composer.process(fixture.itsSource, fixture.mftSource, o2::InteractionRecord{50, 5});
    const auto expected = combinedDropTF ? TrackingOutcome::RecoverableDropped : TrackingOutcome::Structural;
    BOOST_CHECK_MESSAGE(result.outcome == expected, "combined DropTFUponFailure=" << combinedDropTF);
    BOOST_CHECK_EQUAL(composer.frame->getTotalClusters(), 0);
    BOOST_CHECK_EQUAL(composer.frame->getTotalClusters(), 0);
    BOOST_CHECK(frame.getGenericTracks().empty());
    BOOST_CHECK(!composer.getITSPublicationExport().has_value());
    BOOST_CHECK(!composer.getMFTPublicationExport().has_value());
  }
}

BOOST_AUTO_TEST_CASE(StructuralLoadErrorIsAlwaysStructuralRegardlessOfDropTF)
{
  ensureTrivialMagneticFieldIsSet();

  // A missing dictionary is MultiSourceLoadError::MissingDictionary, never
  // recoverable under isRecoverableLoadError() -- DropTFUponFailure=true
  // must not turn this into a dropped TF.
  MinimalFixture fixture;
  fixture.itsSource.dictionary = nullptr;

  auto itsParams = makeItsParams();
  itsParams.DropTFUponFailure = true;
  auto composer = makeComposer(itsParams, makeMftParams());
  TimeFrame frame;
  composer.adoptFrame(frame);
  composer.setBz(Bz);
  composer.setNThreads(1);

  const auto result = composer.process(fixture.itsSource, fixture.mftSource, o2::InteractionRecord{50, 5});
  BOOST_CHECK(result.outcome == TrackingOutcome::Structural);
  BOOST_CHECK(frame.getGenericTracks().empty());
  BOOST_CHECK(!composer.getITSPublicationExport().has_value());
}

BOOST_AUTO_TEST_CASE(UnrecognizedLoadSourceIsAlwaysStructural)
{
  ensureTrivialMagneticFieldIsSet();

  // validateSources() rejects any id other than its own fixed ITS=0/MFT=1
  // contract as MultiSourceLoadError::UnsupportedDetector before ever
  // calling loadSources() -- LoadSourcesResult::source then carries the
  // caller's own (unrecognized) id verbatim. Even if a future loader
  // variant ever reported a recoverable error against such an id, this
  // boundary must still classify Structural: an unidentifiable source is
  // never eligible for recoverable/DropTFUponFailure treatment.
  MinimalFixture fixture;
  fixture.itsSource.id = ClusterSourceId{5};

  auto composer = makeComposer(makeItsParams(), makeMftParams());
  TimeFrame frame;
  composer.adoptFrame(frame);
  composer.setBz(Bz);
  composer.setNThreads(1);

  const auto result = composer.process(fixture.itsSource, fixture.mftSource, o2::InteractionRecord{50, 5});
  BOOST_CHECK(result.outcome == TrackingOutcome::Structural);
  BOOST_CHECK(frame.getGenericTracks().empty());
  BOOST_CHECK(!composer.getITSPublicationExport().has_value());
  BOOST_CHECK(!composer.getMFTPublicationExport().has_value());
}

BOOST_AUTO_TEST_CASE(StructuralTrackingExceptionIsClassifiedStructuralAfterWholeReset)
{
  ensureTrivialMagneticFieldIsSet();

  // MaxMemory=1 with the shared DropTFUponFailure left false makes the one
  // tracker propagate the resource exception to the composition boundary.
  MinimalFixture fixture;
  auto itsParams = makeItsParams();
  itsParams.MaxMemory = 1;

  auto composer = makeComposer(itsParams, makeMftParams());
  TimeFrame frame;
  composer.adoptFrame(frame);
  composer.setBz(Bz);
  composer.setNThreads(1);

  const auto result = composer.process(fixture.itsSource, fixture.mftSource, o2::InteractionRecord{50, 5});
  BOOST_CHECK(result.outcome == TrackingOutcome::Structural);
  BOOST_CHECK_EQUAL(composer.frame->getTotalClusters(), 0);
  BOOST_CHECK_EQUAL(composer.frame->getTotalClusters(), 0);
  BOOST_CHECK(frame.getGenericTracks().empty());
  BOOST_CHECK(!composer.getITSPublicationExport().has_value());
  BOOST_CHECK(!composer.getMFTPublicationExport().has_value());
}

BOOST_AUTO_TEST_CASE(OrderedSurfaceGettersAreAlwaysValidUnlikePublicationExports)
{
  auto composer = makeComposer(makeItsParams(), makeMftParams());
  TimeFrame frame;
  composer.adoptFrame(frame);

  // Configuration is installed before ordered-surface access; publication
  // exports remain unavailable until an event is processed.
  const auto itsSurfacesBefore = composer.getITSLayerMapping();
  const auto mftSurfacesBefore = composer.getMFTLayerMapping();
  BOOST_REQUIRE_EQUAL(itsSurfacesBefore.size(), static_cast<size_t>(ITSNLayers));
  BOOST_REQUIRE_EQUAL(mftSurfacesBefore.size(), static_cast<size_t>(MFTNLayers));
  BOOST_CHECK(itsSurfacesBefore[0] == LayerId{0});
  BOOST_CHECK(mftSurfacesBefore[0] == LayerId{ITSNLayers});
  BOOST_CHECK(!composer.getITSPublicationExport().has_value());
  BOOST_CHECK(!composer.getMFTPublicationExport().has_value());

  // Still identical after a failure (which invalidates the publication
  // exports but must never move the fixed catalog-offset spans).
  ensureTrivialMagneticFieldIsSet();
  MinimalFixture fixture;
  makeRofGap(fixture.mftRofs);
  fixture.mftSource.rofs = fixture.mftRofs;
  composer.setBz(Bz);
  composer.setNThreads(1);
  const auto failed = composer.process(fixture.itsSource, fixture.mftSource, o2::InteractionRecord{50, 5});
  BOOST_REQUIRE(failed.outcome != TrackingOutcome::Success);
  BOOST_CHECK(composer.getITSLayerMapping().data() == itsSurfacesBefore.data());
  BOOST_CHECK(composer.getMFTLayerMapping().data() == mftSurfacesBefore.data());
}

BOOST_AUTO_TEST_CASE(AtomicLoadFailureInvokesEngineResetOnlyAndLeavesNoPublicationState)
{
  // A load failure must reach the single frame reset directly --
  // Tracker::run() (and therefore either leg's kernel sequence) must never run on a
  // partially/never-loaded event. Externally this means: zero tracks
  // reported, cleared scratch and ITS shared-cluster flags, and both
  // publication exports invalidated.
  ensureTrivialMagneticFieldIsSet();
  MinimalFixture fixture;
  makeRofGap(fixture.itsRofs);
  fixture.itsSource.rofs = fixture.itsRofs;

  auto composer = makeComposer(makeItsParams(), makeMftParams());
  TimeFrame frame;
  composer.adoptFrame(frame);
  composer.setBz(Bz);
  composer.setNThreads(1);

  const auto result = composer.process(fixture.itsSource, fixture.mftSource, o2::InteractionRecord{50, 5});
  BOOST_REQUIRE(result.outcome != TrackingOutcome::Success);
  BOOST_CHECK_EQUAL(result.nITSTracks, 0u);
  BOOST_CHECK_EQUAL(result.nMFTTracks, 0u);

  BOOST_CHECK_EQUAL(composer.frame->getTotalClusters(), 0);
  BOOST_CHECK_EQUAL(composer.frame->getTotalClusters(), 0);
  BOOST_CHECK(frame.getGenericTracks().empty());
  BOOST_CHECK(frame.getTrackClusterIndices().empty());
  BOOST_CHECK(composer.getITSSharedClusterFlags().empty());
  BOOST_CHECK(!composer.getITSPublicationExport().has_value());
  BOOST_CHECK(!composer.getMFTPublicationExport().has_value());
}

BOOST_AUTO_TEST_CASE(DetectorConfigurationIsSharedAcrossPassesAndOwnsCatalogMaterial)
{
  auto init = test::makeCombinedConfiguration(makeItsParams(), makeMftParams());
  std::vector<SurfaceDescriptor> catalog(init.catalog.surfaces, init.catalog.surfaces + init.catalog.nSurfaces);
  catalog[0].material = {0.123f, 0.456f};
  init.catalog = {catalog.data(), static_cast<uint32_t>(catalog.size())};
  init.plan.detector.LayerRadii[0] = 2.7f; // Deliberate lookup approximation, distinct from the surface.
  init.plan.execution = {123456789, true};
  init.plan.iterations.resize(3, init.plan.iterations.front());
  init.plan.iterations[1].TrackletMinPt = 0.2f;
  init.plan.iterations[2].TrackletMinPt = 0.1f;
  TimeFrame frame;
  Tracker tracker;
  BOOST_REQUIRE(tracker.initialize(frame, init).ok());
  init.plan.detector.LayerRadii[0] = 99.f;
  catalog[0].material = {};
  const auto ownedCatalog = frame.getLayout().getSurfaceCatalog();
  BOOST_CHECK_EQUAL(ownedCatalog.surfaces[0].material.xOverX0, 0.123f);
  BOOST_CHECK_EQUAL(ownedCatalog.surfaces[0].material.arealDensityGPerCm2, 0.456f);
  BOOST_CHECK_EQUAL(tracker.getDetectorConfiguration().layerRadii[0], 2.7f);
  BOOST_CHECK(ownedCatalog.surfaces[0].referenceCoordinate != tracker.getDetectorConfiguration().layerRadii[0]);
  BOOST_CHECK_EQUAL(tracker.getExecutionPolicy().MaxMemory, 123456789u);
  BOOST_CHECK(tracker.getExecutionPolicy().DropTFUponFailure);
  BOOST_REQUIRE_EQUAL(tracker.getIterationConfigurations().size(), 3u);
  BOOST_CHECK_EQUAL(tracker.getIterationConfigurations()[1].parameters.TrackletMinPt, 0.2f);
  BOOST_CHECK_EQUAL(tracker.getIterationConfigurations()[2].parameters.TrackletMinPt, 0.1f);

  const auto& cache = tracker.getDetectorConfiguration().indexTableConfigs;
  BOOST_REQUIRE_EQUAL(cache.configurationCount(), 2u);
  BOOST_CHECK_EQUAL(&cache[0], &cache[ITSNLayers - 1]);
  BOOST_CHECK_EQUAL(&cache[ITSNLayers], &cache[ITSNLayers + MFTNLayers - 1]);
  BOOST_CHECK(&cache[0] != &cache[ITSNLayers]);
  BOOST_CHECK(cache[0].getCoordType() == IndexTableCoordType::PhiZ);
  BOOST_CHECK(cache[ITSNLayers].getCoordType() == IndexTableCoordType::PhiR);
  auto copy = cache;
  BOOST_CHECK(&copy[0] != &cache[0]);
  BOOST_CHECK_EQUAL(&copy[0], &copy[1]);
  BOOST_CHECK_EQUAL(copy[ITSNLayers].getNcolBins(), cache[ITSNLayers].getNcolBins());
}

BOOST_AUTO_TEST_CASE(SingleKindIndexCacheUsesOneConfigurationAndRejectsInvalidCatalogs)
{
  for (const auto catalog : {SurfaceCatalogView{kITSStaticSurfaceCatalog.data(), ITSNLayers},
                             SurfaceCatalogView{kMFTStaticSurfaceCatalog.data(), MFTNLayers}}) {
    IndexTableConfigurationSet cache;
    BOOST_REQUIRE(cache.reset(catalog));
    BOOST_CHECK_EQUAL(cache.size(), catalog.nSurfaces);
    BOOST_CHECK_EQUAL(cache.configurationCount(), 1u);
    BOOST_CHECK_EQUAL(&cache[0], &cache[catalog.nSurfaces - 1]);
    BOOST_CHECK(!cache.reset({nullptr, 1}));
    BOOST_CHECK_EQUAL(cache.size(), 0u);
    BOOST_CHECK_EQUAL(cache.configurationCount(), 0u);
  }
  auto invalid = kITSStaticSurfaceCatalog[0];
  invalid.kind = static_cast<SurfaceKind>(255);
  IndexTableConfigurationSet cache;
  BOOST_CHECK(!cache.reset({&invalid, 1}));
  BOOST_CHECK_EQUAL(cache.size(), 0u);
  BOOST_CHECK(!cache.reset({&invalid, MaxLayoutSurfaces + 1}));
}

BOOST_AUTO_TEST_CASE(DenseTraversalIdsKeepTheirTypesAndRejectOutOfRangeSlots)
{
  auto init = test::makeCombinedConfiguration(makeItsParams(), makeMftParams());
  TimeFrame frame;
  Tracker tracker;
  BOOST_REQUIRE(tracker.initialize(frame, init).ok());
  const auto& configuration = tracker.getIterationConfigurations().front();
  static_assert(std::is_same_v<decltype(configuration.edgeIds()[0]), EdgeId>);
  static_assert(std::is_same_v<decltype(configuration.cellIds()[0]), CellPathId>);
  for (const auto id : configuration.edgeIds()) {
    BOOST_REQUIRE(configuration.getEdgeSlot(id));
    BOOST_CHECK_EQUAL(*configuration.getEdgeSlot(id), id.value());
  }
  for (const auto id : configuration.cellIds()) {
    BOOST_REQUIRE(configuration.getCellSlot(id));
    BOOST_CHECK_EQUAL(*configuration.getCellSlot(id), id.value());
  }
  BOOST_CHECK(!configuration.getEdgeSlot(EdgeId{}));
  BOOST_CHECK(!configuration.getCellSlot(CellPathId{}));
  BOOST_CHECK(!configuration.getEdgeSlot(EdgeId{static_cast<uint16_t>(configuration.topology.edges.size())}));
  BOOST_CHECK(!configuration.getCellSlot(CellPathId{static_cast<uint16_t>(configuration.topology.paths.size())}));
  const IterationConfiguration empty;
  BOOST_CHECK(empty.edgeIds().empty());
  BOOST_CHECK(empty.cellIds().empty());
  BOOST_CHECK(!empty.getEdgeSlot(EdgeId{0}));
  BOOST_CHECK(!empty.getCellSlot(CellPathId{0}));
}
