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

#include <Acts/EventData/Seed.hpp>
#include <Acts/EventData/SpacePointContainer.hpp>
#include <Acts/Seeding/BinnedGroup.hpp>
#include <Acts/Seeding/SeedFilter.hpp>
#include <Acts/Seeding/SeedFilterConfig.hpp>
#include <Acts/Seeding/SeedFinder.hpp>
#include <Acts/Seeding/SeedFinderConfig.hpp>
#include <Acts/Seeding/detail/CylindricalSpacePointGrid.hpp>
#include <Acts/Utilities/GridBinFinder.hpp>
#include <Acts/Utilities/RangeXD.hpp>

namespace o2::trk
{

template <int nLayers>
TrackerACTS<nLayers>::TrackerACTS()
{
  // Initialize space points storage per layer
  mSpacePointsPerLayer.resize(nLayers);
  mHistSpacePoints = new TH2F("hSpacePoints", "Space points; x (cm); y (cm)", 200, -100, 100, 200, -100, 100);
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

  // Get clusters from the TimeFrame and convert to space points
  for (int layer = 0; layer < nLayers; ++layer) {
    // For now we take unsorted clusters, as soon as the cluster trackin is in place we can piggy back on it and switch to the clusters
    auto clusters = mTimeFrame->getUnsortedClusters()[layer];
    // Resize the clusters to the first 100 clusters for testing
    // clusters = clusters.subspan(0, std::min<size_t>(clusters.size(), 100));
    LOG(debug) << "ACTSTracker: got " << clusters.size() << " clusters";

    for (size_t iCluster = 0; iCluster < clusters.size(); ++iCluster) {
      const auto& cluster = clusters[iCluster];

      SpacePoint sp;
      // Check that these are in global coordinates
      sp.x = cluster.xCoordinate * Acts::UnitConstants::cm;
      sp.y = cluster.yCoordinate * Acts::UnitConstants::cm;
      sp.z = cluster.zCoordinate * Acts::UnitConstants::cm;

      if (mHistSpacePoints) {
        mHistSpacePoints->Fill(sp.x / Acts::UnitConstants::cm, sp.y / Acts::UnitConstants::cm);
      }
      sp.layer = layer;
      sp.clusterId = static_cast<int>(iCluster);
      sp.rof = rof;

      // Position uncertainties (could be refined based on cluster properties)
      sp.varianceR = 0.01f; // ~100 um resolution squared
      sp.varianceZ = 0.01f;

      mSpacePoints.push_back(sp);
    }
  }

  // Build per-layer pointers for seeding
  for (auto& sp : mSpacePoints) {
    if (sp.layer >= 0 && sp.layer < nLayers) {
      mSpacePointsPerLayer[sp.layer].push_back(&sp);
    }
  }
}

template <int nLayers>
void TrackerACTS<nLayers>::createSeeds()
{
  if (mSpacePoints.empty()) {
    LOGF(info, "No space points available for seeding");
    return;
  }
  mSeeds.clear();

  // Backend adaptor that exposes mSpacePoints to Acts::SpacePointContainer
  struct SpacePointBackend {
    using ValueType = SpacePoint;
    explicit SpacePointBackend(const std::vector<SpacePoint>& sps) : m_sps{&sps} {}
    std::size_t size_impl() const { return m_sps->size(); }
    float x_impl(std::size_t i) const { return (*m_sps)[i].x; }
    float y_impl(std::size_t i) const { return (*m_sps)[i].y; }
    float z_impl(std::size_t i) const { return (*m_sps)[i].z; }
    float varianceR_impl(std::size_t i) const { return (*m_sps)[i].varianceR; }
    float varianceZ_impl(std::size_t i) const { return (*m_sps)[i].varianceZ; }
    const SpacePoint& get_impl(std::size_t i) const { return (*m_sps)[i]; }
    std::any component_impl(Acts::HashedString /*key*/, std::size_t /*i*/) const
    {
      LOG(fatal) << "No additional components available for space points";
      throw std::runtime_error("SpacePointBackend: no strip component available");
    }
    const std::vector<SpacePoint>* m_sps;
  };

  // Wrap mSpacePoints in an Acts space-point container
  SpacePointBackend backend{mSpacePoints};

  // Configure the ACTS space point container
  Acts::SpacePointContainerConfig spContainerConfig;
  Acts::SpacePointContainerOptions spContainerOpts;
  spContainerOpts.beamPos = {0.f, 0.f};
  Acts::SpacePointContainer<SpacePointBackend, Acts::detail::RefHolder> spContainer{spContainerConfig, spContainerOpts, backend};

  // Configure the ACTS seed finder
  const unsigned int maxSeeds = static_cast<unsigned int>(mConfig.maxSeedsPerMiddleSP);
  Acts::SeedFilterConfig filterConfig;
  filterConfig.maxSeedsPerSpM = maxSeeds;

  // ACTS requires minPt / bFieldInZ >= rMax / 2 (minHelixRadius >= rMax/2).
  // Cap rMax so that the constraint is satisfied for the configured minPt and field.
  const float bFieldInZ = mBz * Acts::UnitConstants::T;
  const float safeRMax = 1.8f * mConfig.minPt / bFieldInZ; // 10% margin below the hard limit

  using SPProxy = typename Acts::SpacePointContainer<SpacePointBackend, Acts::detail::RefHolder>::SpacePointProxyType;
  Acts::SeedFinderConfig<SPProxy> finderConfig;
  finderConfig.rMin = 0.f;
  finderConfig.rMax = 100.f * Acts::UnitConstants::cm;
  finderConfig.zMin = mConfig.zMin;
  finderConfig.zMax = mConfig.zMax;
  finderConfig.deltaRMin = std::min(mConfig.deltaRMinBottom, mConfig.deltaRMinTop);
  finderConfig.deltaRMax = std::max(mConfig.deltaRMaxBottom, mConfig.deltaRMaxTop);
  finderConfig.deltaRMinBottomSP = mConfig.deltaRMinBottom;
  finderConfig.deltaRMaxBottomSP = mConfig.deltaRMaxBottom;
  finderConfig.deltaRMinTopSP = mConfig.deltaRMinTop;
  finderConfig.deltaRMaxTopSP = mConfig.deltaRMaxTop;
  finderConfig.collisionRegionMin = mConfig.collisionRegionMin;
  finderConfig.collisionRegionMax = mConfig.collisionRegionMax;
  finderConfig.cotThetaMax = mConfig.cotThetaMax;
  finderConfig.minPt = mConfig.minPt;
  finderConfig.impactMax = mConfig.maxImpactParameter;
  finderConfig.maxSeedsPerSpM = maxSeeds;
  finderConfig.sigmaScattering = 5.f;
  finderConfig.radLengthPerSeed = 0.05f;
  finderConfig.seedFilter = std::make_shared<Acts::SeedFilter<SPProxy>>(filterConfig);
  finderConfig = finderConfig.calculateDerivedQuantities();
  Acts::SeedFinder<SPProxy, Acts::CylindricalSpacePointGrid<SPProxy>> seedFinder{finderConfig,
                                                                                 Acts::getDefaultLogger("Finder", Acts::Logging::Level::VERBOSE)};

  // Configure and create the cylindrical space-point grid
  Acts::CylindricalSpacePointGridConfig gridConfig;
  gridConfig.minPt = finderConfig.minPt;
  gridConfig.rMin = finderConfig.rMin;
  gridConfig.rMax = finderConfig.rMax;
  gridConfig.zMin = finderConfig.zMin;
  gridConfig.zMax = finderConfig.zMax;
  gridConfig.deltaRMax = finderConfig.deltaRMax;
  gridConfig.cotThetaMax = finderConfig.cotThetaMax;
  gridConfig.impactMax = finderConfig.impactMax;

  Acts::CylindricalSpacePointGridOptions gridOpts;
  gridOpts.bFieldInZ = bFieldInZ;

  Acts::SeedFinderOptions finderOpts;
  finderOpts.beamPos = spContainerOpts.beamPos;
  finderOpts.bFieldInZ = gridOpts.bFieldInZ;
  try {
    finderOpts = finderOpts.calculateDerivedQuantities(finderConfig);
  } catch (const std::exception& e) {
    LOG(fatal) << "Error in seed finder configuration: " << e.what();
    return;
  }

  Acts::CylindricalSpacePointGrid<SPProxy> grid = Acts::CylindricalSpacePointGridCreator::createGrid<SPProxy>(gridConfig, gridOpts);
  try {
    Acts::CylindricalSpacePointGridCreator::fillGrid(finderConfig, finderOpts, grid,
                                                     spContainer.begin(), spContainer.end());
  } catch (const std::exception& e) {
    LOG(fatal) << "Error during grid creation/filling: " << e.what();
    return;
  }
  LOG(debug) << "Grid created with " << grid.dimensions();

  // Build the binned group and iterate over triplet combinations
  Acts::GridBinFinder<3ul> bottomBinFinder{1, std::vector<std::pair<int, int>>{}, 0};
  Acts::GridBinFinder<3ul> topBinFinder{1, std::vector<std::pair<int, int>>{}, 0};
  Acts::CylindricalBinnedGroup<SPProxy> spGroup{std::move(grid), bottomBinFinder, topBinFinder};

  std::vector<std::vector<Acts::Seed<SPProxy>>> seedsPerGroup;
  typename Acts::SeedFinder<SPProxy, Acts::CylindricalSpacePointGrid<SPProxy>>::SeedingState seedingState;
  seedingState.spacePointMutableData.resize(spContainer.size());
  const Acts::Range1D<float> rMiddleSPRange;
  for (auto [bottom, middle, top] : spGroup) {
    auto& v = seedsPerGroup.emplace_back();
    try {
      seedFinder.createSeedsForGroup(finderOpts, seedingState, spGroup.grid(), v, bottom, middle, top, rMiddleSPRange);
    } catch (const std::exception& e) {
      LOG(fatal) << "Error during seed finding for a group: " << e.what();
      return;
    }
  }
  LOG(debug) << "Seed finding completed, found " << seedsPerGroup.size() << " groups with seeds";

  // Convert Acts seeds to the internal SeedACTS representation
  for (const auto& groupSeeds : seedsPerGroup) {
    for (const auto& actsSeed : groupSeeds) {
      SeedACTS seed;
      seed.bottom = &actsSeed.sp()[0]->externalSpacePoint();
      seed.middle = &actsSeed.sp()[1]->externalSpacePoint();
      seed.top = &actsSeed.sp()[2]->externalSpacePoint();
      seed.quality = actsSeed.seedQuality();
      mSeeds.push_back(seed);
    }
  }

  LOGF(info, "Created %zu seeds from %zu space points", mSeeds.size(), mSpacePoints.size());
}

template <int nLayers>
bool TrackerACTS<nLayers>::estimateTrackParams(const SeedACTS& seed, o2::its::TrackITSExt& track) const
{
  return true;
}

template <int nLayers>
void TrackerACTS<nLayers>::findTracks()
{
}

template <int nLayers>
void TrackerACTS<nLayers>::computeTracksMClabels()
{
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
  LOG(info) << "Processing " << mTimeFrame->getNrof() << " ROFs with B = " << mBz << " T";

  // Process each ROF
  for (int iROF = 0; iROF < mTimeFrame->getNrof(); ++iROF) {
    LOG(info) << "Processing ROF " << iROF;
    // Build space points
    mCurState = SpacePointBuilding;
    totalTime += evaluateTask([this, iROF]() { buildSpacePoints(iROF); },
                              StateNames[mCurState]);

    // Run seeding
    mCurState = Seeding;
    totalTime += evaluateTask([this]() { createSeeds(); },
                              StateNames[mCurState]);

    // Find tracks
    mCurState = TrackFinding;
    totalTime += evaluateTask([this]() { findTracks(); },
                              StateNames[mCurState]);
  }

  // MC labeling
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
