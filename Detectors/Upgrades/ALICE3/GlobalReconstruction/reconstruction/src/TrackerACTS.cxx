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
/// \details
/// The tracker processes one readout frame (ROF) at a time. For each ROF it
/// maps clusters to ACTS surfaces, creates ACTS space-point triplet seeds,
/// fits each seed with the ACTS Kalman fitter, and publishes accepted tracks
/// in the O2 TimeFrame. Optional ROOT and JSON output records intermediate
/// information for diagnostics; neither output is required for reconstruction.
///
/// ACTS geometry is supplied by the process-wide
/// o2::acts::TrackingGeometryManager. Before its first build, the workflow
/// must make the aligned O2 TGeo geometry available and install the
/// detector-specific ACTS geometry builder. The manager builds the geometry
/// lazily and keeps it immutable; surface pointers stored in space points are
/// non-owning references to that manager-owned geometry.
///
/// Cluster coordinates from the TimeFrame are in cm and are converted to
/// ACTS mm when space points are created. ACTS contexts, local coordinates,
/// field values, and fit parameters must therefore use ACTS units. Conversion
/// back to the O2 track representation is performed only when a track is
/// accepted.
///

#include "ALICE3GlobalReconstruction/TrackerACTS.h"

#include "ACTSInterface/MagneticFieldAdapter.h"
#include "ACTSInterface/TrackingGeometryManager.h"
#include "MathUtils/Utils.h"
#include "TRKBase/GeometryTGeo.h"

#include <Acts/Geometry/TrackingGeometry.hpp>
#include <Acts/Geometry/TrackingVolume.hpp>
#include <Acts/Geometry/VolumeBounds.hpp>
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

#include <TFile.h>
#include <TTree.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <optional>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace o2::trk
{

namespace
{
/// Convert an ACTS bound state into O2's local track parameterization.
///
/// O2 stores (x, alpha; y, z, sin(phi-alpha), tan(lambda), q/pT), whereas
/// ACTS stores surface-local coordinates and bound direction/momentum
/// parameters. The covariance is propagated through the same conversion with
/// a central-difference Jacobian. Positions are converted from ACTS mm to O2
/// cm and q/p is converted to q/pT. ACTS has a time coordinate but
/// TrackParCov does not, so the time row/column is not included in the output.
bool convertTrackParameters(const Acts::BoundTrackParameters& actsTrack,
                            const Acts::GeometryContext& gctx,
                            o2::track::TrackParCov& output)
{
  const auto& state = actsTrack.parameters();
  if (!state.allFinite() || !actsTrack.covariance().has_value() || !actsTrack.covariance()->allFinite()) {
    return false;
  }

  const auto position = actsTrack.position(gctx);
  if (!position.allFinite()) {
    return false;
  }
  const double alpha = std::atan2(position.y(), position.x());
  const double ca = std::cos(alpha);
  const double sa = std::sin(alpha);

  auto toO2 = [&](const Acts::BoundVector& p, Eigen::Matrix<double, 5, 1>& dest) -> bool {
    const double theta = p[Acts::eBoundTheta];
    const double sinTheta = std::sin(theta);
    if (!std::isfinite(sinTheta) || sinTheta <= 1.e-6) {
      return false;
    }
    const auto direction = Acts::makeDirectionFromPhiTheta(p[Acts::eBoundPhi], theta);
    const Acts::Vector2 local = p.head<2>();
    const auto xyz = actsTrack.referenceSurface().localToGlobal(gctx, local, direction);
    if (!xyz.allFinite()) {
      return false;
    }

    const double snp = std::sin(p[Acts::eBoundPhi] - alpha);
    if (!std::isfinite(snp) || std::abs(snp) >= 0.999999) {
      return false;
    }

    dest << (-xyz.x() * sa + xyz.y() * ca) / Acts::UnitConstants::cm,
      xyz.z() / Acts::UnitConstants::cm,
      snp,
      std::cos(theta) / sinTheta,
      p[Acts::eBoundQOverP] * Acts::UnitConstants::GeV / sinTheta;
    return dest.allFinite();
  };

  Eigen::Matrix<double, 5, 1> pars;
  if (!toO2(state, pars)) {
    return false;
  }

  const double x = (position.x() * ca + position.y() * sa) / Acts::UnitConstants::cm;
  if (!std::isfinite(x)) {
    return false;
  }

  Eigen::Matrix<double, 5, Acts::eBoundSize> jacobian;
  jacobian.setZero();
  for (int i = 0; i < Acts::eBoundSize; ++i) {
    if (i == Acts::eBoundTime) {
      continue; // O2 TrackParCov has no time coordinate
    }
    const double step = (i == Acts::eBoundLoc0 || i == Acts::eBoundLoc1)
                          ? 1.e-3 * Acts::UnitConstants::mm
                        : (i == Acts::eBoundQOverP)
                          ? std::max(std::abs(state[i]) * 1.e-4, 1.e-6 / Acts::UnitConstants::GeV)
                          : 1.e-5;
    Acts::BoundVector plus = state;
    Acts::BoundVector minus = state;
    plus[i] += step;
    minus[i] -= step;
    Eigen::Matrix<double, 5, 1> fPlus, fMinus;
    if (!toO2(plus, fPlus) || !toO2(minus, fMinus)) {
      return false;
    }
    jacobian.col(i) = (fPlus - fMinus) / (2. * step);
  }

  const Eigen::Matrix<double, 5, 5> covariance = jacobian * (*actsTrack.covariance()) * jacobian.transpose();
  if (!covariance.allFinite()) {
    return false;
  }

  o2::track::TrackParCov::params_t o2Params{};
  o2::track::TrackParCov::covMat_t o2Cov{};
  for (int i = 0; i < 5; ++i) {
    if (!(covariance(i, i) > 0.) || !std::isfinite(pars[i])) {
      return false;
    }
    o2Params[i] = static_cast<float>(pars[i]);
    for (int j = 0; j <= i; ++j) {
      o2Cov[o2::track::CovarMap[i][j]] = static_cast<float>(0.5 * (covariance(i, j) + covariance(j, i)));
    }
  }
  const int charge = state[Acts::eBoundQOverP] >= 0. ? 1 : -1;
  output = o2::track::TrackParCov{static_cast<float>(x), static_cast<float>(alpha),
                                  o2Params, o2Cov, charge, o2::track::PID::Pion};
  return output.isValid();
}
} // namespace
template <int nLayers>
TrackerACTS<nLayers>::TrackerACTS()
{
  // Initialize space points storage per layer
  mSpacePointsPerLayer.resize(nLayers);
  // The debug file and histogram are intentionally not allocated here.
}

template <int nLayers>
TrackerACTS<nLayers>::~TrackerACTS()
{
  if (mDebugFile) {
    mDebugFile->cd();
    if (mHistSpacePoints) {
      mHistSpacePoints->Write("", TObject::kOverwrite);
    }
    mDebugFile->Write("", TObject::kOverwrite);
    mDebugFile->Close();
  }
}

template <int nLayers>
bool TrackerACTS<nLayers>::isDebugTimeFrame() const
{
  // A negative limit means unlimited debug output; zero disables it.
  return mDebugFile && (mConfig.debugMaxTimeFrames < 0 ||
                        mTimeFrameCounter < static_cast<unsigned int>(mConfig.debugMaxTimeFrames));
}

template <int nLayers>
void TrackerACTS<nLayers>::initDebugOutput()
{
  if (!mConfig.writeDebugTrees || mDebugFile || mConfig.debugMaxTimeFrames == 0) {
    return;
  }
  mDebugFile.reset(TFile::Open(mConfig.debugFileName.c_str(), "RECREATE"));
  if (!mDebugFile || mDebugFile->IsZombie()) {
    LOG(error) << "Cannot create ACTS debug ROOT file " << mConfig.debugFileName;
    mDebugFile.reset();
    return;
  }
  mDebugFile->cd();
  // The tree branches point into these member records, which are refilled for
  // each entry. The file owns the trees; the tracker owns the histogram.
  mClusterTree = new TTree("clusters", "ACTS input clusters and surface mapping; positions in cm; variances in ACTS units");
  mClusterTree->Branch("tf", &mDebugCluster.tf, "tf/I");
  mClusterTree->Branch("rof", &mDebugCluster.rof, "rof/I");
  mClusterTree->Branch("layer", &mDebugCluster.layer, "layer/I");
  mClusterTree->Branch("clusterId", &mDebugCluster.clusterId, "clusterId/I");
  mClusterTree->Branch("externalId", &mDebugCluster.externalId, "externalId/I");
  mClusterTree->Branch("sensorId", &mDebugCluster.sensorId, "sensorId/I");
  mClusterTree->Branch("mappingStatus", &mDebugCluster.mappingStatus, "mappingStatus/I");
  mClusterTree->Branch("geometryId", &mDebugCluster.geometryId, "geometryId/l");
  mClusterTree->Branch("x", &mDebugCluster.x, "x/F");
  mClusterTree->Branch("y", &mDebugCluster.y, "y/F");
  mClusterTree->Branch("z", &mDebugCluster.z, "z/F");
  mClusterTree->Branch("r", &mDebugCluster.r, "r/F");
  mClusterTree->Branch("phi", &mDebugCluster.phi, "phi/F");
  mClusterTree->Branch("varianceR", &mDebugCluster.varianceR, "varianceR/F");
  mClusterTree->Branch("varianceZ", &mDebugCluster.varianceZ, "varianceZ/F");

  mSeedTree = new TTree("seeds", "ACTS output seed triplets; positions in cm; bottom/middle/top order");
  mSeedTree->Branch("tf", &mDebugSeed.tf, "tf/I");
  mSeedTree->Branch("rof", &mDebugSeed.rof, "rof/I");
  mSeedTree->Branch("seedId", &mDebugSeed.seedId, "seedId/I");
  mSeedTree->Branch("quality", &mDebugSeed.quality, "quality/F");
  mSeedTree->Branch("layer", mDebugSeed.layer, "layer[3]/I");
  mSeedTree->Branch("clusterId", mDebugSeed.clusterId, "clusterId[3]/I");
  mSeedTree->Branch("sensorId", mDebugSeed.sensorId, "sensorId[3]/I");
  mSeedTree->Branch("geometryId", mDebugSeed.geometryId, "geometryId[3]/l");
  mSeedTree->Branch("x", mDebugSeed.x, "x[3]/F");
  mSeedTree->Branch("y", mDebugSeed.y, "y[3]/F");
  mSeedTree->Branch("z", mDebugSeed.z, "z[3]/F");
  mSeedTree->Branch("r", mDebugSeed.r, "r[3]/F");
  mSeedTree->Branch("phi", mDebugSeed.phi, "phi[3]/F");

  mROFTree = new TTree("rofSummary", "ACTS reconstruction counts per time frame and ROF");
  mROFTree->Branch("tf", &mDebugROF.tf, "tf/I");
  mROFTree->Branch("rof", &mDebugROF.rof, "rof/I");
  mROFTree->Branch("nSpacePoints", &mDebugROF.nSpacePoints, "nSpacePoints/I");
  mROFTree->Branch("nSeeds", &mDebugROF.nSeeds, "nSeeds/I");
  mROFTree->Branch("nAcceptedTracks", &mDebugROF.nAcceptedTracks, "nAcceptedTracks/I");

  mHistSpacePoints = std::make_unique<TH2F>("hSpacePoints", "Space points; x (cm); y (cm)",
                                            200, -100, 100, 200, -100, 100);
  mHistSpacePoints->SetDirectory(nullptr); // Owned by TrackerACTS, not by TFile.
  LOG(info) << "ACTS debug trees enabled: " << mConfig.debugFileName;
}

template <int nLayers>
void TrackerACTS<nLayers>::dumpGeometryOnce()
{
  if (!mConfig.dumpTrackingGeometry || mGeometryDumped) {
    return;
  }
  mGeometryDumped = true; // Even on I/O failure, do not retry on every TF.
  auto& manager = o2::acts::TrackingGeometryManager::instance();
  // get() builds the shared geometry on first access if necessary. Its
  // nominal context is appropriate because O2 alignment was already applied
  // to the TGeo matrices from which this ACTS geometry was built.
  const auto& geometry = manager.get();
  const auto& gctx = manager.getNominalContext();
  auto* trkGeometry = o2::trk::GeometryTGeo::Instance();
  // Build/reuse the sensor-to-surface map. The matrix cache is populated in
  // clustersToTracks(); the path provider makes matching use the TGeo sensor
  // path as an identity instead of relying only on transform proximity.
  const auto& surfaceIndex = manager.getIndex(*trkGeometry, 1.e-3, [trkGeometry](int sensorID) {
    return std::string(trkGeometry->getMatrixPath(sensorID).Data());
  });

  std::ofstream out(mConfig.geometryDumpFileName);
  if (!out) {
    LOG(error) << "Cannot open ACTS geometry dump: " << mConfig.geometryDumpFileName;
    return;
  }
  out << "{\n  \"description\": \"ACTS volumes and sensitive surfaces snapshot, not a complete ACTS geometry serialization\",\n"
         "  \"lengthUnits\": \"mm (angles in rad)\",\n  \"volumes\": [\n";
  bool first = true;
  int nVolumes = 0;
  // ACTS visitor callbacks receive non-owning pointers. The snapshot records
  // global transforms in the nominal context and basic bounds/material
  // metadata; it is intended for inspection, not for rebuilding the geometry.
  geometry.visitVolumes([&](const Acts::TrackingVolume* volume) {
    if (volume == nullptr) {
      return;
    }
    const auto& transform = volume->localToGlobalTransform(gctx);
    nlohmann::json item;
    item["geometryId"] = volume->geometryId().value();
    item["motherGeometryId"] = volume->motherVolume() ? volume->motherVolume()->geometryId().value() : 0;
    item["name"] = volume->volumeName();
    item["boundsType"] = static_cast<int>(volume->volumeBounds().type());
    item["boundsValues"] = volume->volumeBounds().values();
    item["hasMaterial"] = volume->volumeMaterial() != nullptr;
    item["translation"] = {transform.translation().x(), transform.translation().y(), transform.translation().z()};
    item["rotation"] = nlohmann::json::array();
    for (int row = 0; row < 3; ++row) {
      item["rotation"].push_back({transform.rotation()(row, 0), transform.rotation()(row, 1), transform.rotation()(row, 2)});
    }
    if (!first) {
      out << ",\n";
    }
    out << item.dump();
    first = false;
    ++nVolumes;
  });
  out << "\n  ],\n  \"surfaces\": [\n";
  first = true;
  int nSurfaces = 0;
  // The one-argument TrackingGeometry visitor visits sensitive surfaces.
  // Associate each geometry identifier with the O2 sensor IDs known to the
  // detector-specific index; a surface can have no matching sensor ID.
  geometry.visitSurfaces([&](const Acts::Surface* surface) {
    if (surface == nullptr) {
      return;
    }
    const auto& transform = surface->localToGlobalTransform(gctx);
    nlohmann::json item;
    item["geometryId"] = surface->geometryId().value();
    item["surfaceType"] = static_cast<int>(surface->type());
    item["boundsType"] = static_cast<int>(surface->bounds().type());
    item["boundsValues"] = surface->bounds().values();
    item["hasMaterial"] = surface->hasMaterial();
    item["translation"] = {transform.translation().x(), transform.translation().y(), transform.translation().z()};
    item["rotation"] = nlohmann::json::array();
    for (int row = 0; row < 3; ++row) {
      item["rotation"].push_back({transform.rotation()(row, 0), transform.rotation()(row, 1), transform.rotation()(row, 2)});
    }
    item["sensorIds"] = surfaceIndex.getSensorIDs(surface->geometryId());
    if (!first) {
      out << ",\n";
    }
    out << item.dump();
    first = false;
    ++nSurfaces;
  });
  out << "\n  ]\n}\n";
  if (!out.good()) {
    LOG(error) << "Error while writing ACTS geometry dump " << mConfig.geometryDumpFileName;
    return;
  }
  LOG(info) << "ACTS tracking geometry snapshot: " << nVolumes << " volumes and "
            << nSurfaces << " surfaces written to " << mConfig.geometryDumpFileName;
}

template <int nLayers>
void TrackerACTS<nLayers>::fillDebugSeeds(int rof)
{
  if (!isDebugTimeFrame()) {
    return;
  }
  for (size_t iSeed = 0; iSeed < mSeeds.size(); ++iSeed) {
    const auto& seed = mSeeds[iSeed];
    mDebugSeed = {};
    mDebugSeed.tf = static_cast<int>(mTimeFrameCounter);
    mDebugSeed.rof = rof;
    mDebugSeed.seedId = static_cast<int>(iSeed);
    mDebugSeed.quality = seed.quality;
    const std::array<const SpacePoint*, 3> points{seed.bottom, seed.middle, seed.top};
    // Keep the output order consistent with the seeder and seed-triplet
    // convention: innermost (bottom), middle, then outermost (top).
    for (int i = 0; i < 3; ++i) {
      const auto& sp = *points[i];
      mDebugSeed.layer[i] = sp.layer;
      mDebugSeed.clusterId[i] = sp.clusterId;
      mDebugSeed.sensorId[i] = sp.sensorId;
      mDebugSeed.geometryId[i] = sp.surface ? sp.surface->geometryId().value() : 0;
      mDebugSeed.x[i] = sp.x / Acts::UnitConstants::cm;
      mDebugSeed.y[i] = sp.y / Acts::UnitConstants::cm;
      mDebugSeed.z[i] = sp.z / Acts::UnitConstants::cm;
      mDebugSeed.r[i] = sp.r() / Acts::UnitConstants::cm;
      mDebugSeed.phi[i] = sp.phi();
    }
    mSeedTree->Fill();
  }
}

template <int nLayers>
void TrackerACTS<nLayers>::adoptTimeFrame(o2::its::TimeFrame<nLayers>& tf)
{
  // Borrow the TimeFrame; its lifetime must cover all calls that use it.
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

  // Surface matching needs global-to-local transforms from O2's geometry
  // helper. Build its L2G cache before requesting the manager's detector index.
  trkGeometry->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::L2G));

  // mSensorIDs is parallel to each layer's cluster ordering. Its sensor ID
  // selects the corresponding ACTS surface; missing IDs or surfaces cannot be
  // used by seeding/fitting and are skipped below.
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

      const bool hasSensorID = clusterIndex >= 0 && static_cast<size_t>(clusterIndex) < (*mSensorIDs)[layer].size();
      const int sensorID = hasSensorID ? (*mSensorIDs)[layer][clusterIndex] : -1;
      const auto* surface = hasSensorID ? surfaceIndex.getSurface(sensorID) : nullptr;
      if (isDebugTimeFrame()) {
        mDebugCluster = {};
        mDebugCluster.tf = static_cast<int>(mTimeFrameCounter);
        mDebugCluster.rof = rof;
        mDebugCluster.layer = layer;
        mDebugCluster.clusterId = clusterIndex;
        mDebugCluster.externalId = mTimeFrame->getClusterExternalIndex(layer, clusterIndex);
        mDebugCluster.sensorId = sensorID;
        // 0: sensor ID missing; 1: ACTS surface missing; 2: mapped.
        mDebugCluster.mappingStatus = !hasSensorID ? 0 : !surface ? 1
                                                                  : 2;
        mDebugCluster.geometryId = surface ? surface->geometryId().value() : 0;
        mDebugCluster.x = cluster.xCoordinate;
        mDebugCluster.y = cluster.yCoordinate;
        mDebugCluster.z = cluster.zCoordinate;
        mDebugCluster.r = std::hypot(mDebugCluster.x, mDebugCluster.y);
        mDebugCluster.phi = std::atan2(mDebugCluster.y, mDebugCluster.x);
        mDebugCluster.varianceR = 0.01f;
        mDebugCluster.varianceZ = 0.01f;
        mClusterTree->Fill();
      }
      if (!hasSensorID) {
        LOG(error) << "Missing sensor ID for layer " << layer << " cluster " << clusterIndex;
        ++nMissing;
        continue;
      }

      if (!surface) {
        LOG(error) << "No ACTS surface for sensor " << sensorID << " on layer " << layer;
        ++nMissing;
        continue;
      }

      SpacePoint sp;
      // SpacePoint and all subsequent ACTS calculations use mm, while the
      // TimeFrame cluster coordinates are stored in cm.
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

  o2::acts::MagneticFieldAdapter magneticField;
  Acts::MagneticFieldContext magneticFieldContext;
  auto magneticFieldCache = magneticField.makeCache(magneticFieldContext);
  const auto fieldAtOrigin = magneticField.getField(Acts::Vector3::Zero(), magneticFieldCache);
  if (!fieldAtOrigin.ok()) {
    throw std::runtime_error("Failed to query the O2 magnetic field for ACTS seeding");
  }
  const float bFieldInZ = fieldAtOrigin.value().z();

  // ACTS seeding consumes cylindrical (phi, z, r) bins in ACTS units. The
  // bins are populated from the owned space-point vector, whose indices are
  // retained so ACTS seeds can be translated back to SpacePoint pointers.
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
  Acts::SpacePointContainer spacePoints{Acts::SpacePointColumns::CopiedFromIndex |
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
  auto bottomFinder = Acts::DoubletSeedFinder::create(Acts::DoubletSeedFinder::DerivedConfig{bottomConfig, bFieldInZ});

  Acts::DoubletSeedFinder::Config topConfig = bottomConfig;
  topConfig.candidateDirection = Acts::Direction::Forward();
  topConfig.deltaRMin = mConfig.deltaRMinTop;
  topConfig.deltaRMax = mConfig.deltaRMaxTop;
  auto topFinder = Acts::DoubletSeedFinder::create(Acts::DoubletSeedFinder::DerivedConfig{topConfig, bFieldInZ});

  Acts::TripletSeedFinder::Config tripletConfig;
  tripletConfig.minPt = mConfig.minPt;
  tripletConfig.impactMax = mConfig.maxImpactParameter;
  tripletConfig.sigmaScattering = 5.f;
  tripletConfig.radLengthPerSeed = 0.05f;
  auto tripletFinder = Acts::TripletSeedFinder::create(Acts::TripletSeedFinder::DerivedConfig{tripletConfig, bFieldInZ});

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
  fillDebugSeeds(mSpacePoints.front().rof);

  LOGF(debug, "Created %zu seeds from %zu space points", mSeeds.size(), mSpacePoints.size());
}

template <int nLayers>
void TrackerACTS<nLayers>::findTracks()
{
  if (mSeeds.empty()) {
    return;
  }

  auto& actsGeometryManager = o2::acts::TrackingGeometryManager::instance();
  // The same nominal context used to construct/map the geometry must be used
  // for surface transforms and propagation.
  const auto& geometryContext = actsGeometryManager.getNominalContext();

  Acts::MagneticFieldContext magneticFieldContext;
  Acts::CalibrationContext calibrationContext;

  auto magneticField = std::make_shared<o2::acts::MagneticFieldAdapter>();
  auto magneticFieldCache = magneticField->makeCache(magneticFieldContext);

  using Propagator = Acts::Propagator<Acts::EigenStepper<>, Acts::DirectNavigator>;
  using Trajectory = Acts::VectorMultiTrajectory;
  using TrackContainer = Acts::TrackContainer<Acts::VectorTrackContainer, Trajectory>;

  Propagator propagator{Acts::EigenStepper<>{magneticField}, Acts::DirectNavigator{}};

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

      // The hit was converted to this ACTS surface's local coordinates before
      // fitting. Calibrate only the two measured local position coordinates.
      trackState.setUncalibratedSourceLink(Acts::SourceLink{sourceLink});
      trackState.allocateCalibrated(2);
      trackState.calibrated<2>() = hit.local;
      trackState.calibratedCovariance<2>() = hit.covariance;

      trackState.setProjectorSubspaceIndices(std::array<std::uint8_t, 2>{static_cast<std::uint8_t>(Acts::eBoundLoc0),
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

  Acts::PropagatorPlainOptions propagatorOptions{geometryContext, magneticFieldContext};

  Acts::KalmanFitterOptions<Trajectory> fitterOptions{geometryContext,
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

  struct FittedCandidate {
    Acts::BoundTrackParameters inner;
    Acts::BoundTrackParameters outer;
    SeedACTS seed;
    float chi2;
  };
  std::vector<FittedCandidate> candidates;
  int nFitted = 0;
  int nFailed = 0;
  int nInvalid = 0;

  for (const auto& seed : mSeeds) {
    const std::array<const SpacePoint*, 3> spacePoints{seed.bottom, seed.middle, seed.top};

    if (!spacePoints[0] || !spacePoints[1] || !spacePoints[2]) {
      ++nInvalid;
      continue;
    }

    if (!spacePoints[0]->surface || !spacePoints[1]->surface || !spacePoints[2]->surface) {
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
      const auto localField = magneticField->getField(pos1, magneticFieldCache);
      if (!localField.ok()) {
        LOG(error) << "Failed to query the O2 magnetic field for ACTS seed parameters";
        ++nInvalid;
        continue;
      }
      freeParams = Acts::estimateTrackParamsFromSeed(pos0, 0., pos1, pos2, localField.value());
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

    Acts::Vector3 direction{freeParams[Acts::eFreeDir0],
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

      // Convert the cluster's global position to the coordinates expected by
      // a measurement on this ACTS surface.
      auto localResult = surface->globalToLocal(geometryContext, position, direction, 1. * Acts::UnitConstants::mm);

      if (!localResult.ok()) {
        validMeasurements = false;
        break;
      }

      auto& measurement = measurements[i];
      measurement.surface = surface;
      measurement.local = localResult.value();

      const double sigma = 100. * Acts::UnitConstants::um;
      // This initial fitter uses a common 100 um diagonal resolution for all
      // three measurements; detector-specific covariances are not yet passed.
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
    auto square = [](const double x) { return x * x; };
    covariance(Acts::eBoundLoc0, Acts::eBoundLoc0) = square(1. * Acts::UnitConstants::mm);
    covariance(Acts::eBoundLoc1, Acts::eBoundLoc1) = square(1. * Acts::UnitConstants::mm);
    covariance(Acts::eBoundPhi, Acts::eBoundPhi) = 0.1 * 0.1;
    covariance(Acts::eBoundTheta, Acts::eBoundTheta) = 0.1 * 0.1;
    covariance(Acts::eBoundQOverP, Acts::eBoundQOverP) = square(0.5 * qOverP);
    covariance(Acts::eBoundTime, Acts::eBoundTime) = square(1. * Acts::UnitConstants::ns);

    auto initialParameters = Acts::BoundTrackParameters::createCurvilinear(position4, direction, qOverP, covariance,
                                                                           Acts::ParticleHypothesis::pion());

    auto result = fitter.fit(sourceLinks.begin(), sourceLinks.end(), initialParameters,
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

    const auto innerState = track.innermostTrackState();
    if (!innerState.has_value()) {
      ++nFailed;
      continue;
    }
    const auto outerState = track.outermostTrackState();
    if (!innerState->hasSmoothed() || !outerState.hasSmoothed() || !std::isfinite(track.chi2()) || track.chi2() < 0.f) {
      ++nFailed;
      continue;
    }

    // Explicitly use the smoothed fit, rather than the original seed estimate.
    typename Trajectory::ConstTrackStateProxy innerProxy{*innerState};
    typename Trajectory::ConstTrackStateProxy outerProxy{outerState};
    candidates.push_back({track.createParametersFromState(innerProxy),
                          track.createParametersFromState(outerProxy),
                          seed, static_cast<float>(track.chi2())});
    ++nFitted;
    LOG(debug) << "ACTS fitted seed: "
               << "nMeasurements=" << track.nMeasurements()
               << " chi2=" << track.chi2()
               << " nDoF=" << track.nDoF();
  }

  // Accept lower-chi2 candidates first: multiple seeds can share clusters.
  std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
    return a.chi2 < b.chi2;
  });
  int nAccepted = 0;
  for (const auto& candidate : candidates) {
    nAccepted += acceptTrack(candidate.inner, candidate.outer, candidate.seed, candidate.chi2);
  }
  LOG(debug) << "ACTS fitting: " << nFitted << " fitted, "
             << nAccepted << " accepted, " << nFailed << " failed, "
             << nInvalid << " invalid from " << mSeeds.size() << " seeds";
}

template <int nLayers>
bool TrackerACTS<nLayers>::acceptTrack(const Acts::BoundTrackParameters& inner,
                                       const Acts::BoundTrackParameters& outer,
                                       const SeedACTS& seed, float chi2)
{
  if (!mTimeFrame || !std::isfinite(chi2) || chi2 < 0.f) {
    return false;
  }

  const std::array<const SpacePoint*, 3> hits{seed.bottom, seed.middle, seed.top};
  std::array<int, o2::its::TrackITSExt::MaxClusters> clusterIndices;
  clusterIndices.fill(-1);
  std::array<int, nLayers> internalIndices;
  internalIndices.fill(-1);
  uint32_t pattern = 0;
  std::array<uint64_t, 3> keys{};

  // TrackITSExt has one cluster slot per detector layer. Retain the internal
  // TimeFrame cluster index for MC lookup, but publish its external index in
  // the O2 track.
  for (int i = 0; i < 3; ++i) {
    const auto* hit = hits[i];
    if (!hit || hit->layer < 0 || hit->layer >= nLayers || hit->clusterId < 0 ||
        hit->rof < 0 || hit->rof >= mTimeFrame->getNrof(hit->layer)) {
      return false;
    }
    const uint32_t mask = 1u << hit->layer;
    if (pattern & mask) { // the TrackITSExt index array has one slot per layer
      return false;
    }
    pattern |= mask;
    keys[i] = (uint64_t(static_cast<uint32_t>(hit->layer)) << 32) | static_cast<uint32_t>(hit->clusterId);
    if (mAcceptedClusters.contains(keys[i])) {
      return false;
    }
    // Keep the internal index for later MC lookup; publish the external one.
    internalIndices[hit->layer] = hit->clusterId;
    clusterIndices[hit->layer] = mTimeFrame->getClusterExternalIndex(hit->layer, hit->clusterId);
    if (clusterIndices[hit->layer] < 0) {
      return false;
    }
  }

  o2::track::TrackParCov innerO2, outerO2;
  // Convert the smoothed ACTS parameters at both ends of the fitted
  // trajectory. Conversion uses the nominal geometry context so that bound
  // local parameters are interpreted on their associated surfaces.
  const auto& gctx = o2::acts::TrackingGeometryManager::instance().getNominalContext();
  if (!convertTrackParameters(inner, gctx, innerO2) || !convertTrackParameters(outer, gctx, outerO2)) {
    return false;
  }

  // Require the three source clusters to have compatible readout timing. Use
  // the nominal overlap when possible, otherwise the widened overlap, exactly
  // as in ITS track acceptance.
  const auto& timing = mTimeFrame->getROFOverlapTableView();
  auto nominal = timing.getLayer(hits[0]->layer).getROFTimeBounds(hits[0]->rof);
  auto expanded = timing.getLayer(hits[0]->layer).getROFTimeBounds(hits[0]->rof, true);
  bool nominalCompatible = true;
  for (int i = 1; i < 3; ++i) {
    const auto other = timing.getLayer(hits[i]->layer).getROFTimeBounds(hits[i]->rof);
    const auto otherExpanded = timing.getLayer(hits[i]->layer).getROFTimeBounds(hits[i]->rof, true);
    if (nominalCompatible) {
      if (nominal.isCompatible(other)) {
        nominal += other;
      } else {
        nominalCompatible = false;
      }
    }
    if (!expanded.isCompatible(otherExpanded)) {
      return false;
    }
    expanded += otherExpanded;
  }

  o2::its::TrackITSExt o2Track{std::move(innerO2), static_cast<short>(hits.size()), chi2, std::move(outerO2), clusterIndices};
  o2Track.setPattern(pattern);
  o2Track.getTimeStamp() = (nominalCompatible ? nominal : expanded).makeSymmetrical();
  const float maxHalfROF = timing.getClockLayer().mROFLength * 0.5f;
  if (o2Track.getTimeStamp().getTimeStampError() > maxHalfROF) {
    o2Track.getTimeStamp().setTimeStampError(maxHalfROF);
  }

  // Labels are computed once, after all ROFs, from the internal indices.
  mTimeFrame->getTracks().emplace_back(std::move(o2Track));
  mAcceptedTrackClusterIndices.push_back(internalIndices);
  for (const auto key : keys) {
    mAcceptedClusters.insert(key);
  }
  return true;
}

template <int nLayers>
void TrackerACTS<nLayers>::computeTracksMClabels()
{
  if (!mTimeFrame || !mTimeFrame->hasMCinformation()) {
    return;
  }

  auto& tracks = mTimeFrame->getTracks();
  auto& labels = mTimeFrame->getTracksLabel();

  if (tracks.size() != mAcceptedTrackClusterIndices.size()) {
    LOGP(fatal, "ACTS MC labeling: {} tracks but {} internal cluster reference sets", tracks.size(), mAcceptedTrackClusterIndices.size());
    return;
  }

  // There must be exactly one label for each accepted track, in track order.
  labels.clear();
  labels.reserve(tracks.size());

  struct LabelCount {
    o2::MCCompLabel label;
    int count;
    int lastHit;
  };

  for (size_t iTrack = 0; iTrack < tracks.size(); ++iTrack) {
    auto& track = tracks[iTrack];
    const auto& indices = mAcceptedTrackClusterIndices[iTrack];
    std::vector<LabelCount> counts;
    int nHits = 0;

    for (int layer = 0; layer < nLayers; ++layer) {
      if (indices[layer] < 0) {
        continue;
      }
      ++nHits;
      for (const auto& candidate : mTimeFrame->getClusterLabels(layer, indices[layer])) {
        if (!candidate.isValid()) {
          continue;
        }

        auto it = std::find_if(counts.begin(), counts.end(), [&candidate](const auto& entry) {
          return entry.label == candidate;
        });
        if (it == counts.end()) {
          counts.push_back({candidate, 1, nHits});
        } else if (it->lastHit != nHits) {
          // Count a given truth label at most once per attached cluster.
          ++it->count;
          it->lastHit = nHits;
        }
      }
    }
    // Choose the truth label occurring in the largest number of attached
    // clusters. Mark it fake unless it is represented on every hit.
    o2::MCCompLabel winner;
    uint32_t pattern = track.getPattern() & 0x0000ffffu;
    if (counts.empty()) {
      winner.setFakeFlag();
    } else {
      const auto best = std::max_element(counts.begin(), counts.end(), [](const auto& a, const auto& b) {
        return a.count < b.count;
      });
      winner = best->label;
      if (best->count != nHits) {
        winner.setFakeFlag();
      }
      // Preserve the standard TrackITS pattern convention: bits [16, 31]
      // indicate clusters compatible with the representative MC label.
      for (int layer = 0; layer < nLayers; ++layer) {
        if (indices[layer] < 0) {
          continue;
        }
        for (const auto& candidate : mTimeFrame->getClusterLabels(layer, indices[layer])) {
          if (candidate.isValid() && candidate == best->label) {
            pattern |= (1u << (16 + layer));
            break;
          }
        }
      }
    }
    track.setPattern(pattern);
    labels.emplace_back(winner);
  }
  LOGP(info, "ACTS MC labeling: {} tracks, {} labels", tracks.size(), labels.size());
}

template <int nLayers>
void TrackerACTS<nLayers>::clustersToTracks()
{
  if (!mTimeFrame) {
    LOG(error) << "Cannot run TrackerACTS: No TimeFrame adopted";
    return;
  }

  double totalTime = 0.;
  mAcceptedClusters.clear();
  mAcceptedTrackClusterIndices.clear();
  LOG(info) << "==== TRK ACTS Tracking ====";

  // Trigger (or retrieve) the process-wide immutable ACTS geometry. A builder
  // must already have been installed and the aligned O2 geometry must be
  // available before this first access.
  auto& actsGeometryManager = o2::acts::TrackingGeometryManager::instance();
  const auto& trackingGeometry = actsGeometryManager.get();

  // Keep the geometry alive and make the explicit initialization step clear;
  // detailed lookups are done through the detector-specific surface index.
  (void)trackingGeometry;

  auto* trkGeometry = o2::trk::GeometryTGeo::Instance();
  // The index uses this L2G cache and an exact TGeo path provider to map each
  // TRK sensor ID to the ACTS surface built for that sensor. It is cached by
  // TrackingGeometryManager and reused by the ROF processing below.
  trkGeometry->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::L2G));
  const auto& surfaceIndex = actsGeometryManager.getIndex(*trkGeometry, 1.e-3, [trkGeometry](int sensorID) {
    return std::string(trkGeometry->getMatrixPath(sensorID).Data());
  });

  LOG(info) << "ACTS tracking geometry initialized with " << surfaceIndex.size()
            << " mapped sensors on " << surfaceIndex.getNSurfaces() << " surfaces";
  try {
    dumpGeometryOnce();
  } catch (const std::exception& e) {
    LOG(error) << "ACTS geometry debug dump failed: " << e.what();
  }
  if (mConfig.writeDebugTrees && mTimeFrameCounter == 0) {
    initDebugOutput();
  }

  // Each ROF is processed independently for space-point creation, seeding,
  // and fitting. Accepted tracks accumulate in the TimeFrame across ROFs.
  for (int iROF = 0; iROF < mTimeFrame->getNrof(0); ++iROF) {
    LOG(info) << "Processing ROF " << iROF;

    mCurState = SpacePointBuilding;
    totalTime += evaluateTask([this, iROF]() { buildSpacePoints(iROF); }, StateNames[mCurState]);

    mCurState = Seeding;
    totalTime += evaluateTask([this]() { createSeeds(); }, StateNames[mCurState]);

    const auto nTracksBefore = mTimeFrame->getTracks().size();
    mCurState = TrackFinding;
    totalTime += evaluateTask([this]() { findTracks(); }, StateNames[mCurState]);
    if (isDebugTimeFrame()) {
      mDebugROF.tf = static_cast<int>(mTimeFrameCounter);
      mDebugROF.rof = iROF;
      mDebugROF.nSpacePoints = static_cast<int>(mSpacePoints.size());
      mDebugROF.nSeeds = static_cast<int>(mSeeds.size());
      mDebugROF.nAcceptedTracks = static_cast<int>(mTimeFrame->getTracks().size() - nTracksBefore);
      mROFTree->Fill();
    }
  }

  // Run MC labeling once, after every accepted track has been published.
  if (mTimeFrame->hasMCinformation()) {
    computeTracksMClabels();
  }

  LOG(info) << "=== TimeFrame " << mTimeFrameCounter << " completed in: " << totalTime << " ms ===";
  if (isDebugTimeFrame()) {
    mDebugFile->cd();
    if (mHistSpacePoints) {
      mHistSpacePoints->Write("", TObject::kOverwrite);
    }
    mDebugFile->Write("", TObject::kOverwrite);
    mDebugFile->Flush();
  }
  ++mTimeFrameCounter;
  mTotalTime += totalTime;
}

template <int nLayers>
void TrackerACTS<nLayers>::printSummary() const
{
  float avgTF = mTimeFrameCounter > 0 ? static_cast<float>(mTotalTime) / mTimeFrameCounter : 0.f;
  LOGP(info, "TrackerACTS summary: Processed {} TFs in TOT={:.2f} ms, AVG/TF={:.2f} ms", mTimeFrameCounter, mTotalTime, avgTF);
}

// Explicit template instantiations
template class TrackerACTS<11>;
} // namespace o2::trk
