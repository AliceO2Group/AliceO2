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

///
/// \file TrackerACTS.cxx
/// \brief TRK tracker using ACTS seeding and track finding
/// \author Nicolò Jacazio, Università del Piemonte Orientale (IT)
/// \since 2026-04-01
///

#include "ALICE3GlobalReconstruction/TrackerACTS.h"

#include "ACTSInterface/TrackingGeometryManager.h"
#include "MathUtils/Utils.h"
#include "TRKBase/GeometryTGeo.h"

#include <Acts/Definitions/Direction.hpp>
#include <Acts/Definitions/TrackParametrization.hpp>
#include <Acts/EventData/BoundTrackParameters.hpp>
#include <Acts/EventData/ParticleHypothesis.hpp>
#include <Acts/EventData/SeedContainer.hpp>
#include <Acts/EventData/SourceLink.hpp>
#include <Acts/EventData/SpacePointContainer.hpp>
#include <Acts/EventData/TrackContainer.hpp>
#include <Acts/EventData/VectorMultiTrajectory.hpp>
#include <Acts/EventData/VectorTrackContainer.hpp>
#include <Acts/MagneticField/ConstantBField.hpp>
#include <Acts/MagneticField/MagneticFieldContext.hpp>
#include <Acts/Propagator/DirectNavigator.hpp>
#include <Acts/Propagator/EigenStepper.hpp>
#include <Acts/Propagator/Propagator.hpp>
#include <Acts/Seeding/BroadTripletSeedFilter.hpp>
#include <Acts/Seeding/CylindricalSpacePointGrid.hpp>
#include <Acts/Seeding/DoubletSeedFinder.hpp>
#include <Acts/Seeding/EstimateTrackParamsFromSeed.hpp>
#include <Acts/Seeding/TripletSeedFinder.hpp>
#include <Acts/Seeding/TripletSeeder.hpp>
#include <Acts/Surfaces/Surface.hpp>
#include <Acts/TrackFitting/GainMatrixSmoother.hpp>
#include <Acts/TrackFitting/GainMatrixUpdater.hpp>
#include <Acts/TrackFitting/KalmanFitter.hpp>
#include <Acts/Utilities/CalibrationContext.hpp>
#include <Acts/Utilities/GridBinFinder.hpp>

#include "ACTSInterface/TrackingGeometryManager.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace o2::trk
{

template <int nLayers>
TrackerACTS<nLayers>::TrackerACTS()
{
  // Initialize space points storage per layer
  mSpacePointsPerLayer.resize(nLayers);
  mHistSpacePoints = new TH2F("hSpacePoints", "Space points; x (cm); y (cm)", 200, -100, 100, 200, -100, 100);

  // Initialize ACTS Tracking geometry
}

template <int nLayers>
void TrackerACTS<nLayers>::adoptTimeFrame(o2::its::TimeFrame<nLayers>& tf)
{
  mTimeFrame = &tf;
}

template <int nLayers>
void TrackerACTS<nLayers>::buildSpacePoints(int rof)
{
  mSpacePoints.clear();
  for (auto& layerSPs : mSpacePointsPerLayer) {
    layerSPs.clear();
  }

  if (!mSensorIDs) {
    LOG(error) << "Cannot build ACTS space points: Sensor IDs not adopted";
    return;
  }

  auto& actsGeometryManager = o2::acts::TrackingGeometryManager::instance();
  auto* trkGeometry = o2::trk::GeometryTGeo::Instance();

  trkGeometry->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::L2G));

  const auto& surfaceIndex = actsGeometryManager.getIndex(*trkGeometry, 1.e-3, [trkGeometry](int sensorID) {
    return std::string(trkGeometry->getMatrixPath(sensorID).Data());
  });

  int nMapped = 0;
  int nMissing = 0;

  for (int layer = 0; layer < nLayers; ++layer) {
    const auto clusters = mTimeFrame->getUnsortedClustersOnLayer(rof, layer);
    const int firstCluster = mTimeFrame->getSortedStartIndex(rof, layer);

    LOG(debug) << "ACTSTracker: got " << clusters.size() << " clusters on layer " << layer;

    for (size_t iCluster = 0; iCluster < clusters.size(); ++iCluster) {
      const auto& cluster = clusters[iCluster];
      const int clusterIndex = firstCluster + static_cast<int>(iCluster);

      if (static_cast<size_t>(clusterIndex) >= (*mSensorIDs)[layer].size()) {
        LOG(error) << "Missing sensor ID for layer " << layer << " cluster " << clusterIndex;
        ++nMissing;
        continue;
      }

      const int sensorID = (*mSensorIDs)[layer][clusterIndex];
      const auto* surface = surfaceIndex.getSurface(sensorID);
      if (!surface) {
        LOG(error) << "No ACTS surface for sensor " << sensorID << " on layer " << layer;
        ++nMissing;
        continue;
      }

      SpacePoint sp;
      sp.x = cluster.xCoordinate * Acts::UnitConstants::cm;
      sp.y = cluster.yCoordinate * Acts::UnitConstants::cm;
      sp.z = cluster.zCoordinate * Acts::UnitConstants::cm;

      if (mHistSpacePoints) {
        mHistSpacePoints->Fill(sp.x / Acts::UnitConstants::cm, sp.y / Acts::UnitConstants::cm);
      }

      sp.layer = layer;
      sp.clusterId = clusterIndex;
      sp.rof = rof;
      sp.sensorId = sensorID;
      sp.surface = surface;
      sp.varianceR = 0.01f;
      sp.varianceZ = 0.01f;

      mSpacePoints.push_back(sp);
      ++nMapped;
    }
  }

  for (auto& sp : mSpacePoints) {
    if (sp.layer >= 0 && sp.layer < nLayers) {
      mSpacePointsPerLayer[sp.layer].push_back(&sp);
    }
  }

  LOG(info) << "ACTS surface mapping: " << nMapped << " clusters mapped, " << nMissing << " clusters missing";
}

template <int nLayers>
void TrackerACTS<nLayers>::createSeeds()
{
  mSeeds.clear();
  if (mSpacePoints.empty()) {
    LOG(info) << "No space points available for seeding";
    return;
  }

  const float bFieldInZ = mBz * Acts::UnitConstants::T;

  Acts::CylindricalSpacePointGrid::Config gridConfig;
  gridConfig.minPt = mConfig.minPt;
  gridConfig.rMin = 0.f;
  gridConfig.rMax = 100.f * Acts::UnitConstants::cm;
  gridConfig.zMin = mConfig.zMin;
  gridConfig.zMax = mConfig.zMax;
  gridConfig.deltaRMax = std::max(mConfig.deltaRMaxBottom, mConfig.deltaRMaxTop);
  gridConfig.cotThetaMax = mConfig.cotThetaMax;
  gridConfig.impactMax = mConfig.maxImpactParameter;
  gridConfig.bFieldInZ = bFieldInZ;
  gridConfig.bottomBinFinder = Acts::GridBinFinder<3ul>{1, std::vector<std::pair<int, int>>{}, 0};
  gridConfig.topBinFinder = Acts::GridBinFinder<3ul>{1, std::vector<std::pair<int, int>>{}, 0};

  Acts::CylindricalSpacePointGrid grid{gridConfig};
  for (Acts::SpacePointIndex i = 0; i < mSpacePoints.size(); ++i) {
    const auto& sp = mSpacePoints[i];
    grid.insert(i, sp.phi(), sp.z, sp.r());
  }

  // The doublet finder expects each bin to be sorted in increasing radius.
  for (size_t i = 0; i < grid.numberOfBins(); ++i) {
    auto& bin = grid.at(i);
    std::sort(bin.begin(), bin.end(), [this](Acts::SpacePointIndex a, Acts::SpacePointIndex b) {
      return mSpacePoints[a].r() < mSpacePoints[b].r();
    });
  }

  // ACTS 47 uses a non-templated, column-based space-point container.
  Acts::SpacePointContainer spacePoints{
    Acts::SpacePointColumns::CopiedFromIndex |
    Acts::SpacePointColumns::PackedXY |
    Acts::SpacePointColumns::PackedZR |
    Acts::SpacePointColumns::VarianceR |
    Acts::SpacePointColumns::VarianceZ};
  spacePoints.reserve(grid.numberOfSpacePoints());

  std::vector<Acts::SpacePointIndexRange> gridRanges;
  gridRanges.reserve(grid.numberOfBins());
  for (size_t i = 0; i < grid.numberOfBins(); ++i) {
    const auto first = static_cast<Acts::SpacePointIndex>(spacePoints.size());
    for (Acts::SpacePointIndex spIndex : grid.at(i)) {
      const auto& sp = mSpacePoints[spIndex];
      auto proxy = spacePoints.createSpacePoint();
      proxy.copiedFromIndex() = spIndex;
      proxy.xy() = std::array<float, 2>{sp.x, sp.y};
      proxy.zr() = std::array<float, 2>{sp.z, sp.r()};
      proxy.varianceR() = sp.varianceR;
      proxy.varianceZ() = sp.varianceZ;
    }
    const auto last = static_cast<Acts::SpacePointIndex>(spacePoints.size());
    gridRanges.emplace_back(first, last);
  }

  Acts::DoubletSeedFinder::Config bottomConfig;
  bottomConfig.spacePointsSortedByRadius = true;
  bottomConfig.candidateDirection = Acts::Direction::Backward();
  bottomConfig.deltaRMin = mConfig.deltaRMinBottom;
  bottomConfig.deltaRMax = mConfig.deltaRMaxBottom;
  bottomConfig.collisionRegionMin = mConfig.collisionRegionMin;
  bottomConfig.collisionRegionMax = mConfig.collisionRegionMax;
  bottomConfig.cotThetaMax = mConfig.cotThetaMax;
  bottomConfig.impactMax = mConfig.maxImpactParameter;
  bottomConfig.minPt = mConfig.minPt;
  auto bottomFinder = Acts::DoubletSeedFinder::create(
    Acts::DoubletSeedFinder::DerivedConfig{bottomConfig, bFieldInZ});

  Acts::DoubletSeedFinder::Config topConfig = bottomConfig;
  topConfig.candidateDirection = Acts::Direction::Forward();
  topConfig.deltaRMin = mConfig.deltaRMinTop;
  topConfig.deltaRMax = mConfig.deltaRMaxTop;
  auto topFinder = Acts::DoubletSeedFinder::create(
    Acts::DoubletSeedFinder::DerivedConfig{topConfig, bFieldInZ});

  Acts::TripletSeedFinder::Config tripletConfig;
  tripletConfig.minPt = mConfig.minPt;
  tripletConfig.impactMax = mConfig.maxImpactParameter;
  tripletConfig.sigmaScattering = 5.f;
  tripletConfig.radLengthPerSeed = 0.05f;
  auto tripletFinder = Acts::TripletSeedFinder::create(
    Acts::TripletSeedFinder::DerivedConfig{tripletConfig, bFieldInZ});

  Acts::BroadTripletSeedFilter::Config filterConfig;
  filterConfig.maxSeedsPerSpM = static_cast<unsigned int>(mConfig.maxSeedsPerMiddleSP);
  filterConfig.deltaRMin = std::min(mConfig.deltaRMinBottom, mConfig.deltaRMinTop);
  Acts::BroadTripletSeedFilter::State filterState;
  Acts::BroadTripletSeedFilter::Cache filterCache;
  Acts::BroadTripletSeedFilter filter{filterConfig, filterState, filterCache, Acts::getDummyLogger()};

  Acts::TripletSeeder seeder;
  Acts::TripletSeeder::Cache seedingCache;
  Acts::SeedContainer seeds;
  seeds.assignSpacePointContainer(spacePoints);

  std::vector<Acts::SpacePointContainer::ConstRange> bottomRanges;
  std::vector<Acts::SpacePointContainer::ConstRange> topRanges;
  for (const auto [bottom, middle, top] : grid.binnedGroup()) {
    bottomRanges.clear();
    for (const auto bin : bottom) {
      bottomRanges.push_back(spacePoints.range(gridRanges.at(bin)).asConst());
    }
    topRanges.clear();
    for (const auto bin : top) {
      topRanges.push_back(spacePoints.range(gridRanges.at(bin)).asConst());
    }
    auto middleRange = spacePoints.range(gridRanges.at(middle)).asConst();
    if (middleRange.empty()) {
      continue;
    }

    seeder.createSeedsFromGroups(seedingCache, *bottomFinder, *topFinder,
                                 *tripletFinder, filter, spacePoints,
                                 bottomRanges, middleRange, topRanges,
                                 {gridConfig.rMin, gridConfig.rMax}, seeds);
  }

  // Restore the existing SeedACTS representation using original SP indices.
  mSeeds.reserve(seeds.size());
  for (const auto seed : seeds) {
    const auto indices = seed.spacePointIndices();
    SeedACTS output;
    output.bottom = &mSpacePoints[spacePoints.at(indices[0]).copiedFromIndex()];
    output.middle = &mSpacePoints[spacePoints.at(indices[1]).copiedFromIndex()];
    output.top = &mSpacePoints[spacePoints.at(indices[2]).copiedFromIndex()];
    output.quality = seed.quality();
    mSeeds.push_back(output);
  }

  LOGF(info, "Created %zu seeds from %zu space points", mSeeds.size(), mSpacePoints.size());
}

template <int nLayers>
void TrackerACTS<nLayers>::findTracks()
{
  if (mSeeds.empty()) {
    return;
  }

  auto& actsGeometryManager = o2::acts::TrackingGeometryManager::instance();
  const auto& geometryContext = actsGeometryManager.getNominalContext();

  Acts::MagneticFieldContext magneticFieldContext;
  Acts::CalibrationContext calibrationContext;

  // Initial implementation with a constant magnetic field
  auto magneticField = std::make_shared<Acts::ConstantBField>(
    Acts::Vector3{0., 0., mBz * Acts::UnitConstants::T});

  using Propagator = Acts::Propagator<Acts::EigenStepper<>, Acts::DirectNavigator>;
  using Trajectory = Acts::VectorMultiTrajectory;
  using TrackContainer = Acts::TrackContainer<Acts::VectorTrackContainer, Trajectory>;

  Propagator propagator{
    Acts::EigenStepper<>{magneticField},
    Acts::DirectNavigator{}};

  Acts::KalmanFitter<Propagator, Trajectory> fitter{std::move(propagator)};

  struct HitMeasurement {
    const Acts::Surface* surface{nullptr};
    Acts::Vector2 local = Acts::Vector2::Zero();
    Acts::SquareMatrix<2> covariance = Acts::SquareMatrix<2>::Zero();
  };

  struct HitSourceLink {
    const HitMeasurement* hit{nullptr};
  };

  struct SurfaceAccessor {
    const Acts::Surface* getSurface(const Acts::SourceLink& sourceLink) const
    {
      return sourceLink.get<HitSourceLink>().hit->surface;
    }
  };

  struct MeasurementCalibrator {
    void calibrate(const Acts::GeometryContext&,
                   const Acts::CalibrationContext&,
                   const Acts::SourceLink& sourceLink,
                   Trajectory::TrackStateProxy trackState) const
    {
      const auto& hit = *sourceLink.get<HitSourceLink>().hit;

      trackState.setUncalibratedSourceLink(Acts::SourceLink{sourceLink});
      trackState.allocateCalibrated(2);
      trackState.calibrated<2>() = hit.local;
      trackState.calibratedCovariance<2>() = hit.covariance;

      trackState.setProjectorSubspaceIndices(
        std::array<std::uint8_t, 2>{
          static_cast<std::uint8_t>(Acts::eBoundLoc0),
          static_cast<std::uint8_t>(Acts::eBoundLoc1)});
    }
  };

  SurfaceAccessor surfaceAccessor;
  MeasurementCalibrator calibrator;
  Acts::GainMatrixUpdater updater;
  Acts::GainMatrixSmoother smoother;

  Acts::KalmanFitterExtensions<Trajectory> extensions;
  extensions.surfaceAccessor.connect<&SurfaceAccessor::getSurface>(&surfaceAccessor);
  extensions.calibrator.connect<&MeasurementCalibrator::calibrate>(&calibrator);
  extensions.updater.connect<&Acts::GainMatrixUpdater::operator()<Trajectory>>(&updater);
  extensions.smoother.connect<&Acts::GainMatrixSmoother::operator()<Trajectory>>(&smoother);

  Acts::PropagatorPlainOptions propagatorOptions{
    geometryContext, magneticFieldContext};

  Acts::KalmanFitterOptions<Trajectory> fitterOptions{
    geometryContext,
    magneticFieldContext,
    calibrationContext,
    extensions,
    propagatorOptions,
    nullptr,
    false,
    false};

  // Mutable TrackContainer uses reference-holder backends, not rvalue shared_ptrs.
  Acts::VectorTrackContainer trackBackend;
  Trajectory trajectoryBackend;
  TrackContainer tracks{trackBackend, trajectoryBackend};

  int nFitted = 0;
  int nFailed = 0;
  int nInvalid = 0;

  for (const auto& seed : mSeeds) {
    const std::array<const SpacePoint*, 3> spacePoints{
      seed.bottom, seed.middle, seed.top};

    if (!spacePoints[0] || !spacePoints[1] || !spacePoints[2]) {
      ++nInvalid;
      continue;
    }

    if (!spacePoints[0]->surface || !spacePoints[1]->surface ||
        !spacePoints[2]->surface) {
      ++nInvalid;
      continue;
    }

    // The direct fitter expects at most one measurement per surface.
    if (spacePoints[0]->surface == spacePoints[1]->surface ||
        spacePoints[0]->surface == spacePoints[2]->surface ||
        spacePoints[1]->surface == spacePoints[2]->surface) {
      ++nInvalid;
      continue;
    }

    Acts::Vector3 pos0{spacePoints[0]->x, spacePoints[0]->y, spacePoints[0]->z};
    Acts::Vector3 pos1{spacePoints[1]->x, spacePoints[1]->y, spacePoints[1]->z};
    Acts::Vector3 pos2{spacePoints[2]->x, spacePoints[2]->y, spacePoints[2]->z};

    Acts::FreeVector freeParams;
    try {
      freeParams = Acts::estimateTrackParamsFromSeed(
        pos0, 0., pos1, pos2,
        Acts::Vector3{0., 0., mBz * Acts::UnitConstants::T});
    } catch (const std::exception& e) {
      LOG(debug) << "ACTS seed parameter estimation failed: " << e.what();
      ++nInvalid;
      continue;
    }

    const double qOverP = freeParams[Acts::eFreeQOverP];
    if (!std::isfinite(qOverP) || std::abs(qOverP) < 1.e-12) {
      ++nInvalid;
      continue;
    }

    Acts::Vector3 direction{
      freeParams[Acts::eFreeDir0],
      freeParams[Acts::eFreeDir1],
      freeParams[Acts::eFreeDir2]};

    if (!direction.allFinite() || direction.norm() < 1.e-12) {
      ++nInvalid;
      continue;
    }
    direction.normalize();

    std::array<HitMeasurement, 3> measurements;
    std::vector<Acts::SourceLink> sourceLinks;
    std::vector<const Acts::Surface*> surfaces;
    sourceLinks.reserve(3);
    surfaces.reserve(3);

    bool validMeasurements = true;
    for (int i = 0; i < 3; ++i) {
      const auto* sp = spacePoints[i];
      const auto* surface = sp->surface;
      Acts::Vector3 position{sp->x, sp->y, sp->z};

      auto localResult = surface->globalToLocal(
        geometryContext, position, direction,
        1. * Acts::UnitConstants::mm);

      if (!localResult.ok()) {
        validMeasurements = false;
        break;
      }

      auto& measurement = measurements[i];
      measurement.surface = surface;
      measurement.local = localResult.value();

      const double sigma = 100. * Acts::UnitConstants::um;
      measurement.covariance.setZero();
      measurement.covariance(0, 0) = sigma * sigma;
      measurement.covariance(1, 1) = sigma * sigma;

      sourceLinks.emplace_back(HitSourceLink{&measurement});
      surfaces.push_back(surface);
    }

    if (!validMeasurements) {
      ++nInvalid;
      continue;
    }

    Acts::Vector4 position4;
    position4 << pos0.x(), pos0.y(), pos0.z(), 0.;

    Acts::BoundMatrix covariance = Acts::BoundMatrix::Zero();
    covariance(Acts::eBoundLoc0, Acts::eBoundLoc0) =
      std::pow(1. * Acts::UnitConstants::mm, 2);
    covariance(Acts::eBoundLoc1, Acts::eBoundLoc1) =
      std::pow(1. * Acts::UnitConstants::mm, 2);
    covariance(Acts::eBoundPhi, Acts::eBoundPhi) = 0.1 * 0.1;
    covariance(Acts::eBoundTheta, Acts::eBoundTheta) = 0.1 * 0.1;
    covariance(Acts::eBoundQOverP, Acts::eBoundQOverP) =
      std::pow(0.5 * qOverP, 2);
    covariance(Acts::eBoundTime, Acts::eBoundTime) =
      std::pow(1. * Acts::UnitConstants::ns, 2);

    auto initialParameters = Acts::BoundTrackParameters::createCurvilinear(
      position4, direction, qOverP, covariance,
      Acts::ParticleHypothesis::pion());

    auto result = fitter.fit(
      sourceLinks.begin(), sourceLinks.end(), initialParameters,
      fitterOptions, surfaces, tracks);

    if (!result.ok()) {
      LOG(debug) << "ACTS Kalman fit failed: " << result.error().message();
      ++nFailed;
      continue;
    }

    const auto track = result.value();
    if (track.nMeasurements() < 3) {
      ++nFailed;
      continue;
    }

    ++nFitted;
    LOG(debug) << "ACTS fitted seed: "
               << "nMeasurements=" << track.nMeasurements()
               << " chi2=" << track.chi2()
               << " nDoF=" << track.nDoF();
  }

  LOG(info) << "ACTS fitting: " << nFitted << " fitted, "
            << nFailed << " failed, "
            << nInvalid << " invalid from "
            << mSeeds.size() << " seeds";
}

template <int nLayers>
void TrackerACTS<nLayers>::computeTracksMClabels()
{

  return; // For now we skip MC labeling, will be implemented in the next iterations once we have track candidates to label

  if (!mTimeFrame || !mTimeFrame->hasMCinformation()) {

    return;
  }

  // MC labeling using majority voting on cluster labels

  for (int iROF = 0; iROF < mTimeFrame->getNrof(0); ++iROF) {

    for (auto& track : mTimeFrame->getTracks()) {
      std::vector<std::pair<MCCompLabel, size_t>> labelCounts;

      for (int iCluster = 0; iCluster < o2::its::TrackITSExt::MaxClusters; ++iCluster) {
        const int clusterIdx = track.getClusterIndex(iCluster);
        if (clusterIdx == o2::its::constants::UnusedIndex) {
          continue;
        }

        auto clusterLabels = mTimeFrame->getClusterLabels(iCluster, clusterIdx);
        for (const auto& label : clusterLabels) {
          auto it = std::find_if(labelCounts.begin(), labelCounts.end(),
                                 [&label](const auto& p) { return p.first == label; });
          if (it != labelCounts.end()) {
            ++(it->second);
          } else {
            labelCounts.emplace_back(label, 1);
          }
        }
      }

      if (!labelCounts.empty()) {
        // Find label with most occurrences
        auto maxIt = std::max_element(labelCounts.begin(), labelCounts.end(),
                                      [](const auto& a, const auto& b) { return a.second < b.second; });

        MCCompLabel trackLabel = maxIt->first;
        if (maxIt->second < static_cast<size_t>(track.getNumberOfClusters())) {
          trackLabel.setFakeFlag();
        }
        mTimeFrame->getTracksLabel().emplace_back(trackLabel);
      }
    }
  }
}

template <int nLayers>
void TrackerACTS<nLayers>::clustersToTracks()
{
  if (!mTimeFrame) {
    LOG(error) << "Cannot run TrackerACTS: No TimeFrame adopted";
    return;
  }

  double totalTime = 0.;
  LOG(info) << "==== TRK ACTS Tracking ====";

  // Initialize ACTS tracking geometry
  auto& actsGeometryManager = o2::acts::TrackingGeometryManager::instance();
  const auto& trackingGeometry = actsGeometryManager.get();
  (void)trackingGeometry;

  auto* trkGeometry = o2::trk::GeometryTGeo::Instance();
  trkGeometry->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::L2G));
  const auto& surfaceIndex = actsGeometryManager.getIndex(*trkGeometry, 1.e-3, [trkGeometry](int sensorID) {
    return std::string(trkGeometry->getMatrixPath(sensorID).Data());
  });

  LOG(info) << "ACTS tracking geometry initialized with " << surfaceIndex.size()
            << " mapped sensors on " << surfaceIndex.getNSurfaces() << " surfaces";

  for (int iROF = 0; iROF < mTimeFrame->getNrof(0); ++iROF) {
    LOG(info) << "Processing ROF " << iROF;

    mCurState = SpacePointBuilding;
    totalTime += evaluateTask([this, iROF]() { buildSpacePoints(iROF); }, StateNames[mCurState]);

    mCurState = Seeding;
    totalTime += evaluateTask([this]() { createSeeds(); }, StateNames[mCurState]);

    mCurState = TrackFinding;
    totalTime += evaluateTask([this]() { findTracks(); }, StateNames[mCurState]);
  }

  if (mTimeFrame->hasMCinformation()) {
    computeTracksMClabels();
  }

  LOG(info) << "=== TimeFrame " << mTimeFrameCounter << " completed in: " << totalTime << " ms ===";
  ++mTimeFrameCounter;
  mTotalTime += totalTime;
}

template <int nLayers>
void TrackerACTS<nLayers>::printSummary() const
{

  float avgTF = mTimeFrameCounter > 0 ? static_cast<float>(mTotalTime) / mTimeFrameCounter : 0.f;

  LOGP(info, "TrackerACTS summary: Processed {} TFs in TOT={:.2f} ms, AVG/TF={:.2f} ms",

       mTimeFrameCounter, mTotalTime, avgTF);
}

// Explicit template instantiations
template class TrackerACTS<11>;
} // namespace o2::trk
