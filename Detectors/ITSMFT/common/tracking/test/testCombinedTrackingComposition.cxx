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
#include "ITSMFTTracking/detail/ITSSharedClusterCompatibility.h"
#include "ITSMFTTracking/detail/TimeFrameScratch.h"
#include "ITSMFTTracking/detail/MFTFwdTrackHelpers.h"
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

/// Same chained-projection construction as
/// testComputeLayerTrackletsOrchestration.cxx's buildMftChainClusters() /
/// the former traversal-binding orchestration test's identically-named helper:
/// each hop's target is a genuine geometric match via
/// detail::mftTrackletProject, so every adjacent pair in the chain produces
/// a real tracklet, and a full-length chain reaches acceptance.
std::vector<DecodedCluster> buildMftChainClusters(const TrackingParameters& params, float bz, int nHops)
{
  std::vector<DecodedCluster> clusters;
  // Keep the synthetic trajectory on the descriptor-owned MFT radial chart.
  // The former (1, 0.5) seed was inside the legacy square LUT but below the
  // physical inner radius of every MFT disk.
  float x = 3.f, y = 1.5f;
  float z = detail::mftLayerZ(0);
  clusters.push_back(diskCluster(x, y, z, 0));
  for (int hop = 0; hop < nHops; ++hop) {
    const float nextZ = detail::mftLayerZ(hop + 1);
    float targetX = 0.f, targetY = 0.f;
    detail::mftTrackletProject(x, y, z, params.Diamond[0], params.Diamond[1], params.Diamond[2],
                               hop, hop + 1, bz, params.TrackletMinPt, targetX, targetY);
    clusters.push_back(diskCluster(targetX, targetY, nextZ, hop + 1));
    x = targetX;
    y = targetY;
    z = nextZ;
  }
  return clusters;
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
  // through, with no TimeFrame vertex needed -- the same knob
  // buildMftChainClusters()'s MFT fixtures already rely on.
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
                int rofLength = 40)
    : params{singleParams}
  {
    const auto orderedSurfaces = ordered(0, NLayers);
    catalog.reserve(NLayers);
    for (uint16_t i = 0; i < NLayers; ++i) {
      SurfaceDescriptor surface{i, static_cast<uint8_t>(det), kind};
      surface.chartRange = kind == SurfaceKind::Disk ? SurfaceChartRange{kMFTLookupRMin[i], kMFTLookupRMax[i]} : SurfaceChartRange{-20.f, 20.f};
      surface.referenceCoordinate = kind == SurfaceKind::Cylinder
                                      ? singleParams.LayerRadii[i]
                                      : detail::mftLayerZ(i);
      const float xOverX0 = det == o2::detectors::DetID::MFT ? kNominalMFTLayerX0[i] : kNominalITSLayerX0[i];
      surface.material.xOverX0 = xOverX0;
      surface.material.arealDensityGPerCm2 = xOverX0 * o2::its::constants::Radl * o2::its::constants::Rho;
      catalog.push_back(surface);
    }
    const SurfaceCatalogView catalogView{catalog.data(), static_cast<uint32_t>(catalog.size())};
    TrackerInitialization configuration;
    configuration.catalog = catalogView;
    configuration.memoryPool = pool;
    configuration.layout = makeDetectorLayout();
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
  const ITSSharedClusterCompatibility& getITSSharedClusterCompatibility() const noexcept { return plan.getITSSharedClusterCompatibility(); }
  gsl::span<const LayerId> getITSLayerMapping() const noexcept { return plan.getITSLayerMapping(); }
  gsl::span<const LayerId> getMFTLayerMapping() const noexcept { return plan.getMFTLayerMapping(); }
};

CombinedTrackingComposer makeComposer(const TrackingParameters& itsParams, const TrackingParameters& mftParams)
{
  return CombinedTrackingComposer{std::vector<TrackingParameters>{itsParams}, std::vector<TrackingParameters>{mftParams}};
}

} // namespace

BOOST_AUTO_TEST_CASE(CombinedLoadingBackfillsOneGlobalWorkspace)
{
  // TrackerTraits::findRoads() unconditionally touches the global
  // o2::base::Propagator singleton on first use, regardless of whether any
  // road is actually found -- required before any clustersToTracks() call.
  ensureTrivialMagneticFieldIsSet();
  const auto itsSurfaces = ordered(0, ITSNLayers);
  const auto mftSurfaces = ordered(ITSNLayers, MFTNLayers);
  const auto itsClusters = std::vector<DecodedCluster>{cylinderCluster(3.f, 0.2f, 0.1f, 0), cylinderCluster(4.f, 0.2f, 0.1f, 1)};
  const auto mftClusters = std::vector<DecodedCluster>{diskCluster(1.f, 0.5f, detail::mftLayerZ(0), 0), diskCluster(1.f, 0.5f, detail::mftLayerZ(1), 1)};

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

BOOST_AUTO_TEST_CASE(MftGlobalIdsWorkEndToEndThroughRefitUnderCombinedPolicy)
{
  ensureTrivialMagneticFieldIsSet();
  const auto itsSurfaces = ordered(0, ITSNLayers);
  const auto mftSurfaces = ordered(ITSNLayers, MFTNLayers);

  const auto mftParams = makeMftParams();
  const auto mftClusters = buildMftChainClusters(mftParams, Bz, MFTNLayers - 1);
  BOOST_REQUIRE_EQUAL(mftClusters.size(), static_cast<size_t>(MFTNLayers));

  PrescribedDecoder itsDecoder{o2::detectors::DetID::ITS, SurfaceKind::Cylinder, {}};
  PrescribedDecoder mftDecoder{o2::detectors::DetID::MFT, SurfaceKind::Disk, mftClusters};
  std::vector<CompClusterExt> mftCompact;
  std::vector<unsigned char> mftPatterns;
  std::vector<ROFRecord> mftRofs;
  const auto itsSource = makeEmptySource(ClusterSourceId{0}, o2::detectors::DetID::ITS, itsSurfaces, itsDecoder);
  const auto mftSource = makeSource(ClusterSourceId{1}, o2::detectors::DetID::MFT, mftSurfaces, mftDecoder, mftCompact, mftPatterns, mftRofs, mftClusters);

  auto composer = makeComposer(makeItsParams(), mftParams);
  TimeFrame frame;
  composer.adoptFrame(frame);
  composer.setBz(Bz);
  composer.setNThreads(1);

  const auto result = composer.process(itsSource, mftSource, o2::InteractionRecord{50, 5});
  BOOST_REQUIRE(result.outcome == TrackingOutcome::Success);
  BOOST_CHECK_EQUAL(result.nITSTracks, 0u);
  // Global MFT LayerIds 7..16 plus source 1 work end to end through the
  // disk leaves and refit while the one combined selection policy is active.
  BOOST_CHECK_GT(result.nMFTTracks, 0u);
}

BOOST_AUTO_TEST_CASE(CombinedComponentsUseOwnROFTimingInOneCombinedPass)
{
  ensureTrivialMagneticFieldIsSet();
  const auto itsSurfaces = ordered(0, ITSNLayers);
  const auto mftSurfaces = ordered(ITSNLayers, MFTNLayers);

  const auto itsParams = makeItsParams();
  const auto mftParams = makeMftParams();
  const auto itsClusters = buildItsHelixChainClusters(itsParams.LayerRadii, Bz, 1.f, 0.4f, 0.3f);
  BOOST_REQUIRE_EQUAL(itsClusters.size(), static_cast<size_t>(ITSNLayers));
  const auto mftClusters = buildMftChainClusters(mftParams, Bz, MFTNLayers - 1);
  BOOST_REQUIRE_EQUAL(mftClusters.size(), static_cast<size_t>(MFTNLayers));

  StandaloneRun<o2::detectors::DetID::ITS, ITSNLayers> standaloneIts{o2::detectors::DetID::ITS, SurfaceKind::Cylinder, itsParams, itsClusters};
  BOOST_REQUIRE(standaloneIts.result.outcome == TrackingOutcome::Success);
  // A genuine full 7-layer road (MinTrackLength=7, MaxHoles=0): the helix
  // fixture above is a real, non-degenerate curved trajectory, so this is a
  // nonzero accepted-track oracle, not a 0==0 parity check.
  BOOST_REQUIRE_GT(standaloneIts.frame.getGenericTracks().size(), 0u);
  StandaloneRun<o2::detectors::DetID::MFT, MFTNLayers> standaloneMft{o2::detectors::DetID::MFT, SurfaceKind::Disk, mftParams, mftClusters, 80};
  BOOST_REQUIRE(standaloneMft.result.outcome == TrackingOutcome::Success);
  BOOST_REQUIRE_GT(standaloneMft.frame.getGenericTracks().size(), 0u);

  PrescribedDecoder itsDecoder{o2::detectors::DetID::ITS, SurfaceKind::Cylinder, itsClusters};
  PrescribedDecoder mftDecoder{o2::detectors::DetID::MFT, SurfaceKind::Disk, mftClusters};
  std::vector<CompClusterExt> itsCompact, mftCompact;
  std::vector<unsigned char> itsPatterns, mftPatterns;
  std::vector<ROFRecord> itsRofs, mftRofs;
  const auto itsSource = makeSource(ClusterSourceId{0}, o2::detectors::DetID::ITS, itsSurfaces, itsDecoder, itsCompact, itsPatterns, itsRofs, itsClusters);
  auto mftSource = makeSource(ClusterSourceId{1}, o2::detectors::DetID::MFT, mftSurfaces, mftDecoder, mftCompact, mftPatterns, mftRofs, mftClusters);
  // Make the disconnected MFT component's ROF longer than ITS. Reusing the
  // first/global ITS view would incorrectly clamp MFT to an ITS half-ROF;
  // the per-surface timing lookup must reproduce standalone MFT instead.
  mftSource.timing.rofLength = 80;

  auto composer = makeComposer(itsParams, mftParams);
  TimeFrame frame;
  composer.adoptFrame(frame);
  composer.setBz(Bz);
  composer.setNThreads(1);

  const auto result = composer.process(itsSource, mftSource, o2::InteractionRecord{50, 5});
  BOOST_REQUIRE(result.outcome == TrackingOutcome::Success);

  // This full-chain fixture survives both standalone and combined selection,
  // allowing timestamp behavior to be compared on nonzero tracks without
  // making general standalone/combined population parity a requirement.
  BOOST_CHECK_GT(result.nITSTracks, 0u);
  BOOST_CHECK_GT(result.nMFTTracks, 0u);
  BOOST_CHECK_EQUAL(result.nITSTracks, standaloneIts.frame.getGenericTracks().size());
  BOOST_CHECK_EQUAL(result.nMFTTracks, standaloneMft.frame.getGenericTracks().size());

  // The one workspace contains the disjoint components' compact buffers in
  // graph order. Their populated cell count is therefore the sum of the two
  // standalone component counts; tracklets have already been consumed.
  BOOST_CHECK_EQUAL(composer.getITSScratch().getNumberOfTracklets(),
                    standaloneIts.scratch->getNumberOfTracklets() + standaloneMft.scratch->getNumberOfTracklets());
  BOOST_CHECK_EQUAL(composer.getITSScratch().getNumberOfCells(),
                    standaloneIts.scratch->getNumberOfCells() + standaloneMft.scratch->getNumberOfCells());
  BOOST_CHECK_EQUAL(&composer.getITSScratch(), &composer.getMFTScratch());
  BOOST_CHECK_GT(composer.getITSScratch().getNumberOfCells(), 0u);

  // GenericTrack global references resolve correctly and ordering is ITS
  // then MFT: every accepted track's hitLayers mask stays within exactly
  // one detector's own global range, and every ITS-range entry precedes
  // every MFT-range entry (shared TimeFrame, append-only, ITS run first).
  const auto itsMask = LayerMask{uint32_t{(1u << ITSNLayers) - 1u}};
  const auto mftMask = LayerMask{static_cast<uint32_t>(((1u << MFTNLayers) - 1u) << ITSNLayers)};
  const auto& commonTracks = frame.getGenericTracks();
  BOOST_REQUIRE_EQUAL(commonTracks.size(), result.nITSTracks + result.nMFTTracks);
  for (size_t i = 0; i < result.nITSTracks; ++i) {
    BOOST_CHECK_EQUAL(commonTracks[i].timestamp.begin, standaloneIts.frame.getGenericTracks()[i].timestamp.begin);
    BOOST_CHECK_EQUAL(commonTracks[i].timestamp.end, standaloneIts.frame.getGenericTracks()[i].timestamp.end);
  }
  for (size_t i = 0; i < result.nMFTTracks; ++i) {
    const auto& combinedTrack = commonTracks[result.nITSTracks + i];
    const auto& standaloneTrack = standaloneMft.frame.getGenericTracks()[i];
    BOOST_CHECK_EQUAL(combinedTrack.timestamp.begin, standaloneTrack.timestamp.begin);
    BOOST_CHECK_EQUAL(combinedTrack.timestamp.end, standaloneTrack.timestamp.end);
    BOOST_CHECK_GT(combinedTrack.timestamp.end - combinedTrack.timestamp.begin,
                   commonTracks.front().timestamp.end - commonTracks.front().timestamp.begin);
  }
  bool seenMft = false;
  size_t nextReference = 0;
  for (size_t i = 0; i < commonTracks.size(); ++i) {
    const auto& track = commonTracks[i];
    BOOST_CHECK_EQUAL(track.firstClusterRef, nextReference);
    BOOST_CHECK_LT(track.firstClusterRef, track.clusterRefEnd);
    BOOST_CHECK(isValidTrackRange(track, static_cast<uint32_t>(frame.getTrackClusterIndices().size())));
    BOOST_REQUIRE(track.hitLayers.isSubsetOf(itsMask) || track.hitLayers.isSubsetOf(mftMask));
    const bool isMft = track.hitLayers.isSubsetOf(mftMask) && !track.hitLayers.empty();
    if (isMft) {
      seenMft = true;
    } else {
      BOOST_CHECK_MESSAGE(!seenMft, "ITS GenericTrack at index " << i << " appeared after an MFT one");
    }
    for (uint32_t ref = track.firstClusterRef; ref < track.clusterRefEnd; ++ref) {
      const auto& reference = frame.getTrackClusterIndices()[ref];
      const auto globals = frame.getGlobalMeasurements(reference.layer);
      BOOST_CHECK(std::any_of(globals.begin(), globals.end(), [&](const auto& measurement) {
        return measurement.clusterId == reference.clusterId;
      }));
      BOOST_CHECK(isMft ? mftMask.has(reference.layer.value()) : itsMask.has(reference.layer.value()));
    }
    nextReference = track.clusterRefEnd;
  }
  BOOST_CHECK_EQUAL(nextReference, frame.getTrackClusterIndices().size());
  BOOST_CHECK_EQUAL(seenMft, result.nMFTTracks > 0);

  const auto& itsCompatibility = composer.getITSSharedClusterCompatibility().entries();
  BOOST_REQUIRE_EQUAL(itsCompatibility.size(), result.nITSTracks);
  for (size_t i = 0; i < itsCompatibility.size(); ++i) {
    BOOST_CHECK_EQUAL(itsCompatibility[i].genericTrackIndex, i);
  }

  // Publication exports are valid after success, source-qualified, and
  // carry each detector's own ordered-surface span.
  const auto itsExport = composer.getITSPublicationExport();
  const auto mftExport = composer.getMFTPublicationExport();
  BOOST_REQUIRE(itsExport.has_value());
  BOOST_REQUIRE(mftExport.has_value());
  BOOST_CHECK(itsExport->detector == o2::detectors::DetID::ITS);
  BOOST_CHECK(itsExport->source == ClusterSourceId{0});
  BOOST_CHECK_EQUAL(itsExport->layerMapping.size(), static_cast<size_t>(ITSNLayers));
  BOOST_CHECK(itsExport->layerMapping[0] == LayerId{0});
  BOOST_CHECK(mftExport->detector == o2::detectors::DetID::MFT);
  BOOST_CHECK(mftExport->source == ClusterSourceId{1});
  BOOST_CHECK_EQUAL(mftExport->layerMapping.size(), static_cast<size_t>(MFTNLayers));
  BOOST_CHECK(mftExport->layerMapping[0] == LayerId{ITSNLayers});
}

BOOST_AUTO_TEST_CASE(LoadFailureResetsWholeCombinedTFExactlyOnceAndInvalidatesPublication)
{
  ensureTrivialMagneticFieldIsSet();
  const auto itsSurfaces = ordered(0, ITSNLayers);
  const auto mftSurfaces = ordered(ITSNLayers, MFTNLayers);
  const auto itsClusters = std::vector<DecodedCluster>{cylinderCluster(3.f, 0.2f, 0.1f, 0), cylinderCluster(4.f, 0.2f, 0.1f, 1)};
  const auto mftClusters = std::vector<DecodedCluster>{diskCluster(1.f, 0.5f, detail::mftLayerZ(0), 0), diskCluster(1.f, 0.5f, detail::mftLayerZ(1), 1)};

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
  const auto mftClusters = std::vector<DecodedCluster>{diskCluster(1.f, 0.5f, detail::mftLayerZ(0), 0), diskCluster(1.f, 0.5f, detail::mftLayerZ(1), 1)};

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
  std::vector<DecodedCluster> mftClusters{diskCluster(1.f, 0.5f, detail::mftLayerZ(0), 0), diskCluster(1.f, 0.5f, detail::mftLayerZ(1), 1)};
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

BOOST_AUTO_TEST_CASE(SequentialSuccessfulTFsReplaceStateWithoutStaleAccumulation)
{
  ensureTrivialMagneticFieldIsSet();
  const auto itsSurfaces = ordered(0, ITSNLayers);
  const auto mftSurfaces = ordered(ITSNLayers, MFTNLayers);
  const auto itsParams = makeItsParams();
  const auto mftParams = makeMftParams();
  // A genuine nonzero-track fixture (same construction as
  // ITSAndMFTAcceptedResultsReproduceStandaloneCountsInOneCombinedPass): if
  // GenericTrack/TrackClusterIndices storage ever accumulated across TFs
  // instead of being replaced, the second TF's count below would silently
  // double rather than reproduce the same per-TF value.
  const auto itsClusters = buildItsHelixChainClusters(itsParams.LayerRadii, Bz, 1.f, 0.4f, 0.3f);
  BOOST_REQUIRE_EQUAL(itsClusters.size(), static_cast<size_t>(ITSNLayers));
  const auto mftClusters = buildMftChainClusters(mftParams, Bz, MFTNLayers - 1);
  BOOST_REQUIRE_EQUAL(mftClusters.size(), static_cast<size_t>(MFTNLayers));

  PrescribedDecoder itsDecoder{o2::detectors::DetID::ITS, SurfaceKind::Cylinder, itsClusters};
  PrescribedDecoder mftDecoder{o2::detectors::DetID::MFT, SurfaceKind::Disk, mftClusters};
  std::vector<CompClusterExt> itsCompact, mftCompact;
  std::vector<unsigned char> itsPatterns, mftPatterns;
  std::vector<ROFRecord> itsRofs, mftRofs;
  const auto itsSource = makeSource(ClusterSourceId{0}, o2::detectors::DetID::ITS, itsSurfaces, itsDecoder, itsCompact, itsPatterns, itsRofs, itsClusters);
  const auto mftSource = makeSource(ClusterSourceId{1}, o2::detectors::DetID::MFT, mftSurfaces, mftDecoder, mftCompact, mftPatterns, mftRofs, mftClusters);

  auto composer = makeComposer(itsParams, mftParams);
  TimeFrame frame;
  composer.adoptFrame(frame);
  composer.setBz(Bz);
  composer.setNThreads(1);

  const auto firstResult = composer.process(itsSource, mftSource, o2::InteractionRecord{50, 5});
  BOOST_REQUIRE(firstResult.outcome == TrackingOutcome::Success);
  BOOST_REQUIRE_GT(firstResult.nITSTracks + firstResult.nMFTTracks, 0u);
  const auto firstGenericTrackCount = frame.getGenericTracks().size();
  BOOST_REQUIRE_EQUAL(firstGenericTrackCount, firstResult.nITSTracks + firstResult.nMFTTracks);

  // No explicit reset between successful TFs: loadTimeFrameSources()
  // load()'s frame commit atomically replaces the
  // normalized frame and clears mGenericTracks/mTrackClusterIndices in the
  // same commit (TimeFrame.h), so the second process() call alone -- on the
  // identical fixture again -- must reproduce the same per-TF count, not
  // the first TF's count plus the second's.
  const auto secondResult = composer.process(itsSource, mftSource, o2::InteractionRecord{60, 6});
  BOOST_REQUIRE(secondResult.outcome == TrackingOutcome::Success);

  BOOST_CHECK_EQUAL(secondResult.nITSTracks, firstResult.nITSTracks);
  BOOST_CHECK_EQUAL(secondResult.nMFTTracks, firstResult.nMFTTracks);
  BOOST_CHECK_EQUAL(frame.getGenericTracks().size(), firstGenericTrackCount);
  BOOST_CHECK_EQUAL(composer.frame->getTotalClusters(),
                    static_cast<int>(itsClusters.size() + mftClusters.size()));
  BOOST_CHECK_EQUAL(&composer.getITSScratch(), &composer.getMFTScratch());
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

BOOST_AUTO_TEST_CASE(CompatibilitySidecarGettersReflectSealAndReset)
{
  ensureTrivialMagneticFieldIsSet();
  MinimalFixture fixture;
  auto composer = makeComposer(makeItsParams(), makeMftParams());
  TimeFrame frame;
  composer.adoptFrame(frame);
  composer.setBz(Bz);
  composer.setNThreads(1);

  // Not yet sealed before any process() call.
  BOOST_CHECK(!composer.getITSSharedClusterCompatibility().isSealed());

  const auto result = composer.process(fixture.itsSource, fixture.mftSource, o2::InteractionRecord{50, 5});
  BOOST_REQUIRE(result.outcome == TrackingOutcome::Success);
  // A successful run always seals the ITS sidecar (Tracker<ITSNLayers>::
  // clustersToTracks() -> markTracks() -> sealFromMarkedTracks()), which is
  // exactly what stageITSGenericTrackOutput() requires
  // (GenericTrackOutputAdapter.h).
  BOOST_CHECK(composer.getITSSharedClusterCompatibility().isSealed());

  // A whole reset clears both sidecars back to their pre-process() state.
  makeRofGap(fixture.mftRofs);
  fixture.mftSource.rofs = fixture.mftRofs;
  const auto failed = composer.process(fixture.itsSource, fixture.mftSource, o2::InteractionRecord{60, 6});
  BOOST_REQUIRE(failed.outcome != TrackingOutcome::Success);
  BOOST_CHECK(!composer.getITSSharedClusterCompatibility().isSealed());
  BOOST_CHECK(composer.getITSSharedClusterCompatibility().entries().empty());
}

BOOST_AUTO_TEST_CASE(ExplicitScheduleDrivesITSThenMFTThroughTheDelegatedEngine)
{
  // Same construction as
  // ITSAndMFTAcceptedResultsReproduceStandaloneCountsInOneCombinedPass,
  // narrowed to the one claim this test adds: process()'s ITS-then-MFT
  // GenericTrack ordering and per-detector publication exports are produced
  // by the explicit [ITS, MFT] Tracker invocation order
  // (the workflow-owned explicit schedule), not a hand-unrolled pair of
  // clustersToTracks() calls.
  ensureTrivialMagneticFieldIsSet();
  const auto itsSurfaces = ordered(0, ITSNLayers);
  const auto mftSurfaces = ordered(ITSNLayers, MFTNLayers);
  const auto itsParams = makeItsParams();
  const auto mftParams = makeMftParams();
  const auto itsClusters = buildItsHelixChainClusters(itsParams.LayerRadii, Bz, 1.f, 0.4f, 0.3f);
  BOOST_REQUIRE_EQUAL(itsClusters.size(), static_cast<size_t>(ITSNLayers));
  const auto mftClusters = buildMftChainClusters(mftParams, Bz, MFTNLayers - 1);
  BOOST_REQUIRE_EQUAL(mftClusters.size(), static_cast<size_t>(MFTNLayers));

  PrescribedDecoder itsDecoder{o2::detectors::DetID::ITS, SurfaceKind::Cylinder, itsClusters};
  PrescribedDecoder mftDecoder{o2::detectors::DetID::MFT, SurfaceKind::Disk, mftClusters};
  std::vector<CompClusterExt> itsCompact, mftCompact;
  std::vector<unsigned char> itsPatterns, mftPatterns;
  std::vector<ROFRecord> itsRofs, mftRofs;
  const auto itsSource = makeSource(ClusterSourceId{0}, o2::detectors::DetID::ITS, itsSurfaces, itsDecoder, itsCompact, itsPatterns, itsRofs, itsClusters);
  const auto mftSource = makeSource(ClusterSourceId{1}, o2::detectors::DetID::MFT, mftSurfaces, mftDecoder, mftCompact, mftPatterns, mftRofs, mftClusters);

  auto composer = makeComposer(itsParams, mftParams);
  TimeFrame frame;
  composer.adoptFrame(frame);
  composer.setBz(Bz);
  composer.setNThreads(1);

  const auto result = composer.process(itsSource, mftSource, o2::InteractionRecord{50, 5});
  BOOST_REQUIRE(result.outcome == TrackingOutcome::Success);
  BOOST_REQUIRE_GT(result.nITSTracks, 0u);
  BOOST_REQUIRE_GT(result.nMFTTracks, 0u);

  // GenericTrack ordering: every ITS-range entry precedes every MFT-range
  // entry -- the observable footprint of the engine having run track() in
  // schedule order [ITS, MFT], not some other order.
  const auto itsMask = LayerMask{uint32_t{(1u << ITSNLayers) - 1u}};
  const auto mftMask = LayerMask{static_cast<uint32_t>(((1u << MFTNLayers) - 1u) << ITSNLayers)};
  const auto& commonTracks = frame.getGenericTracks();
  BOOST_REQUIRE_EQUAL(commonTracks.size(), result.nITSTracks + result.nMFTTracks);
  bool seenMft = false;
  for (const auto& track : commonTracks) {
    const bool isMft = track.hitLayers.isSubsetOf(mftMask) && !track.hitLayers.empty();
    if (isMft) {
      seenMft = true;
    } else {
      BOOST_CHECK(track.hitLayers.isSubsetOf(itsMask));
      BOOST_CHECK_MESSAGE(!seenMft, "an ITS GenericTrack appeared after an MFT one: schedule order was not ITS-then-MFT");
    }
  }
  BOOST_CHECK(seenMft);

  // Per-detector publication exports still resolve correctly through the
  // participant-owned scratch/plan the composition reads from.
  const auto itsExport = composer.getITSPublicationExport();
  const auto mftExport = composer.getMFTPublicationExport();
  BOOST_REQUIRE(itsExport.has_value());
  BOOST_REQUIRE(mftExport.has_value());
  BOOST_CHECK(itsExport->detector == o2::detectors::DetID::ITS);
  BOOST_CHECK(itsExport->source == ClusterSourceId{0});
  BOOST_CHECK_EQUAL(itsExport->layerMapping.size(), static_cast<size_t>(ITSNLayers));
  BOOST_CHECK(mftExport->detector == o2::detectors::DetID::MFT);
  BOOST_CHECK(mftExport->source == ClusterSourceId{1});
  BOOST_CHECK_EQUAL(mftExport->layerMapping.size(), static_cast<size_t>(MFTNLayers));
}

BOOST_AUTO_TEST_CASE(AtomicLoadFailureInvokesEngineResetOnlyAndLeavesNoParticipantOrSidecarState)
{
  // A load failure must reach the single frame reset directly --
  // Tracker::run() (and therefore either leg's kernel sequence) must never run on a
  // partially/never-loaded event. Externally this means: zero tracks
  // reported, both legs' scratches and both detector compatibility
  // sidecars back to their pre-process() empty/unsealed state (never
  // populated, since track() never ran), and both publication exports
  // invalidated.
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
  // Neither sidecar was ever sealed/populated by this process() call --
  // proof that track() (and therefore the engine's executeEvent()) was
  // never reached on this partially loaded event.
  BOOST_CHECK(!composer.getITSSharedClusterCompatibility().isSealed());
  BOOST_CHECK(composer.getITSSharedClusterCompatibility().entries().empty());
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
