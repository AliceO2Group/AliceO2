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
/// \file TrackerACTS.h
/// \brief TRK tracker using ACTS seeding algorithm
/// \author Nicolò Jacazio, Università del Piemonte Orientale (IT)
/// \since 2026-04-01
///

#ifndef ALICE3_GLOBALRECONSTRUCTION_INCLUDE_TRACKERACTS_H_
#define ALICE3_GLOBALRECONSTRUCTION_INCLUDE_TRACKERACTS_H_

#include <Acts/Definitions/Units.hpp>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <memory>
#include <string>
#include <string_view>
#include <Framework/Logger.h>
#include <nlohmann/json.hpp>
#include <array>
#include <vector>
#include <unordered_set>
#include "DataFormatsITS/TrackITS.h"
#include <ITStracking/TimeFrame.h>
#include <TH2F.h>

class TFile;
class TTree;
namespace Acts
{
class Surface;
class BoundTrackParameters;
} // namespace Acts
namespace o2::trk
{

/// Configuration for ACTS seeding, track finding, and optional diagnostic output.
///
/// Length and momentum-valued cuts use ACTS units (mm and GeV, respectively).
struct TrackerACTSConfig {

  TrackerACTSConfig()
  {
    std::ifstream input(configName);
    if (!input) {
      return;
    }
    LOG(info) << "Loading ACTS tracker configuration from " << configName;
    nlohmann::json config;
    try {
      input >> config;
    } catch (const nlohmann::json::parse_error& e) {
      LOG(error) << "Failed to parse ACTS tracker configuration file " << configName << ": " << e.what();
      return;
    }
    if (!config.is_object()) {
      LOG(error) << "ACTS tracker configuration file " << configName << " must contain a JSON object";
      return;
    }

    const auto load = [&config](const char* key, auto& value) {
      if (!config.contains(key)) {
        return;
      }
      try {
        config.at(key).get_to(value);
      } catch (const nlohmann::json::exception& e) {
        LOG(error) << "Failed to load ACTS tracker configuration parameter " << key << ": " << e.what();
      }
    };

    load("minPt", minPt);
    load("maxImpactParameter", maxImpactParameter);
    load("cotThetaMax", cotThetaMax);
    load("deltaRMinBottom", deltaRMinBottom);
    load("deltaRMaxBottom", deltaRMaxBottom);
    load("deltaRMinTop", deltaRMinTop);
    load("deltaRMaxTop", deltaRMaxTop);
    load("zMin", zMin);
    load("zMax", zMax);
    load("collisionRegionMin", collisionRegionMin);
    load("collisionRegionMax", collisionRegionMax);
    load("maxSeedsPerMiddleSP", maxSeedsPerMiddleSP);
    load("deltaPhiMax", deltaPhiMax);
    load("writeDebugTrees", writeDebugTrees);
    load("dumpTrackingGeometry", dumpTrackingGeometry);
    load("debugMaxTimeFrames", debugMaxTimeFrames);
    load("debugFileName", debugFileName);
    load("geometryDumpFileName", geometryDumpFileName);
  }

  ~TrackerACTSConfig()
  {
    const nlohmann::json config = {
      {"minPt", minPt},
      {"maxImpactParameter", maxImpactParameter},
      {"cotThetaMax", cotThetaMax},
      {"deltaRMinBottom", deltaRMinBottom},
      {"deltaRMaxBottom", deltaRMaxBottom},
      {"deltaRMinTop", deltaRMinTop},
      {"deltaRMaxTop", deltaRMaxTop},
      {"zMin", zMin},
      {"zMax", zMax},
      {"collisionRegionMin", collisionRegionMin},
      {"collisionRegionMax", collisionRegionMax},
      {"maxSeedsPerMiddleSP", maxSeedsPerMiddleSP},
      {"deltaPhiMax", deltaPhiMax},
      {"writeDebugTrees", writeDebugTrees},
      {"dumpTrackingGeometry", dumpTrackingGeometry},
      {"debugMaxTimeFrames", debugMaxTimeFrames},
      {"debugFileName", debugFileName},
      {"geometryDumpFileName", geometryDumpFileName},
    };
    std::ofstream output(configName);
    if (!output) {
      LOG(error) << "Failed to open ACTS tracker configuration file " << configName;
      return;
    }

    output << config.dump(2) << '\n';
    if (!output) {
      LOG(error) << "Failed to write ACTS tracker configuration file " << configName;
    }
    LOG(info) << "Saved ACTS tracker configuration to " << configName;
  }

  /// Minimum transverse momentum accepted by the seeding algorithms.
  float minPt = 0.4f * Acts::UnitConstants::GeV; ///< Minimum pT for seeds
  /// Maximum transverse distance of closest approach used in seeding.
  float maxImpactParameter = 10.f * Acts::UnitConstants::mm;
  /// Maximum absolute cot(theta); the default corresponds approximately to |eta| = 4.
  float cotThetaMax = std::sinh(4.0f);

  /// Minimum and maximum radial separation for bottom-to-middle space-point pairs.
  float deltaRMinBottom = 5.f * Acts::UnitConstants::mm;   ///< Min deltaR for bottom-middle
  float deltaRMaxBottom = 200.f * Acts::UnitConstants::mm; ///< Max deltaR for bottom-middle
  /// Minimum and maximum radial separation for middle-to-top space-point pairs.
  float deltaRMinTop = 5.f * Acts::UnitConstants::mm;   ///< Min deltaR for middle-top
  float deltaRMaxTop = 200.f * Acts::UnitConstants::mm; ///< Max deltaR for middle-top

  /// Longitudinal range used to select space points for seeding.
  float zMin = -3000.f * Acts::UnitConstants::mm;
  float zMax = 3000.f * Acts::UnitConstants::mm;

  /// Longitudinal range of the collision region used when constructing seeds.
  float collisionRegionMin = -150.f * Acts::UnitConstants::mm;
  float collisionRegionMax = 150.f * Acts::UnitConstants::mm;

  /// Maximum number of seeds retained for each middle space point.
  float maxSeedsPerMiddleSP = 2;
  /// Reserved maximum azimuthal angle difference for doublets (radians); currently unused.
  float deltaPhiMax = 0.1f;

  /// Enable diagnostic ROOT trees and histograms; disabled by default.
  bool writeDebugTrees = false;
  /// Write a JSON snapshot of the tracking geometry on the first use.
  bool dumpTrackingGeometry = false;
  /// Number of time frames with debug output; 0 disables it and -1 means all.
  int debugMaxTimeFrames = 1;
  /// Destination ROOT file for diagnostic output.
  std::string debugFileName = "TrackerACTS_debug.root";
  /// Destination JSON file for the tracking-geometry snapshot.
  std::string geometryDumpFileName = "TrackerACTS_geometry.json";

  static constexpr const char* configName = "TrackerACTSConfig.json";
};

/// Cluster-derived space point passed to the ACTS seeding algorithms.
struct SpacePoint {
  /// Cartesian position in the global tracking frame, in mm.
  float x{0.f};
  float y{0.f};
  float z{0.f};
  /// Detector layer and cluster indices identifying the source measurement.
  int layer{-1};
  int clusterId{-1};
  /// Readout-frame and sensor indices identifying the source measurement.
  int rof{-1};
  int sensorId{-1};
  /// Geometry surface associated with this measurement; not owned by the space point.
  const Acts::Surface* surface{nullptr};

  /// Transverse radius in mm.
  float r() const { return std::hypot(x, y); }
  /// Radius accessor required by the ACTS cylindrical-grid element concept.
  float radius() const { return r(); }
  /// Azimuthal angle in radians.
  float phi() const { return std::atan2(y, x); }

  /// Estimated radial and longitudinal position variances, in mm^2.
  float varianceR{0.01f};
  float varianceZ{0.01f};
};

/// Three-space-point seed ordered from the innermost to outermost measurement.
struct SeedACTS {
  const SpacePoint* bottom{nullptr};
  const SpacePoint* middle{nullptr};
  const SpacePoint* top{nullptr};
  /// ACTS seed quality value.
  float quality{0.f};
};

/// TRK tracker using ACTS algorithms for seeding and track finding.
///
/// The caller supplies a TimeFrame and, optionally, sensor-ID mappings before
/// calling clustersToTracks(). The adopted data is referenced rather than
/// copied, so it must remain alive while the tracker processes it.
template <int nLayers>
class TrackerACTS
{
 public:
  TrackerACTS();
  ~TrackerACTS();

  /// Set the TimeFrame whose clusters will be reconstructed.
  /// The TimeFrame is borrowed and must outlive its use by this tracker.
  void adoptTimeFrame(o2::its::TimeFrame<nLayers>& tf);
  /// Set per-layer sensor-ID mappings used to associate clusters with geometry.
  /// The mapping array is borrowed and must remain alive while it is used.
  void adoptSensorIDs(const std::array<std::vector<int>, nLayers>& sensorIDs)
  {
    mSensorIDs = &sensorIDs;
  }

  /// Build space points, find tracks, and publish accepted tracks in the TimeFrame.
  void clustersToTracks();

  /// Replace the current tracker configuration.
  void setConfig(const TrackerACTSConfig& cfg) { mConfig = cfg; }
  /// Access the tracker configuration for modification.
  TrackerACTSConfig& getConfig() { return mConfig; }
  /// Read-only access to the tracker configuration.
  const TrackerACTSConfig& getConfig() const { return mConfig; }

  /// Print the accumulated tracking timing and processing summary.
  void printSummary() const;

 private:
  // Debug resources are allocated only if requested and closed at destruction.
  std::unique_ptr<TFile> mDebugFile;
  TTree* mClusterTree = nullptr; // Owned by mDebugFile
  TTree* mSeedTree = nullptr;    // Owned by mDebugFile
  TTree* mROFTree = nullptr;     // Owned by mDebugFile
  std::unique_ptr<TH2F> mHistSpacePoints;
  bool mGeometryDumped = false;

  struct DebugClusterRecord {
    int tf = 0, rof = 0, layer = 0, clusterId = -1;
    int externalId = -1, sensorId = -1, mappingStatus = 0;
    unsigned long long geometryId = 0;
    float x = 0.f, y = 0.f, z = 0.f;
    float r = 0.f, phi = 0.f;
    float varianceR = 0.f, varianceZ = 0.f;
  } mDebugCluster;

  struct DebugSeedRecord {
    int tf = 0, rof = 0, seedId = 0;
    float quality = 0.f;
    int layer[3] = {}, clusterId[3] = {}, sensorId[3] = {};
    unsigned long long geometryId[3] = {};
    float x[3] = {}, y[3] = {}, z[3] = {}, r[3] = {}, phi[3] = {};
  } mDebugSeed;

  struct DebugROFRecord {
    int tf = 0, rof = 0, nSpacePoints = 0, nSeeds = 0;
    int nAcceptedTracks = 0;
  } mDebugROF;

  void initDebugOutput();
  void dumpGeometryOnce();
  bool isDebugTimeFrame() const;
  void fillDebugSeeds(int rof);

  /// Build space points from clusters in the TimeFrame
  void buildSpacePoints(int rof);

  /// Create seeds (triplets) from space points using ACTS SeedFinder
  void createSeeds();

  /// Run track finding from seeds
  void findTracks();

  /// Convert fitted ACTS states to a TrackITSExt and publish it in the TimeFrame.
  bool acceptTrack(const Acts::BoundTrackParameters& inner,
                   const Acts::BoundTrackParameters& outer,
                   const SeedACTS& seed, float chi2);
  /// Assign one MC label per accepted track, after all ROFs have been processed.
  void computeTracksMClabels();

  /// Helper: time a task
  template <typename Func>
  float evaluateTask(Func&& task, std::string_view taskName);

  // Configuration
  TrackerACTSConfig mConfig;

  // TimeFrame data
  o2::its::TimeFrame<nLayers>* mTimeFrame = nullptr;

  // Space points built from clusters
  std::vector<SpacePoint> mSpacePoints;
  std::vector<std::vector<const SpacePoint*>> mSpacePointsPerLayer;

  // Seeds
  std::vector<SeedACTS> mSeeds;
  // Avoid accepting seeds which reuse a cluster already assigned to a track.
  std::unordered_set<uint64_t> mAcceptedClusters;
  // Layer-local indices in mUnsortedClusters, in the same order as getTracks().
  // Keep them separate from the external indices stored in TrackITSExt.
  std::vector<std::array<int, nLayers>> mAcceptedTrackClusterIndices;

  // Tracking state
  unsigned int mTimeFrameCounter{0};
  double mTotalTime{0.};
  const std::array<std::vector<int>, nLayers>* mSensorIDs = nullptr;
  // Tracking states for logging
  enum State {
    SpacePointBuilding = 0,
    Seeding,
    TrackFinding,
    NStates,
  };
  State mCurState{SpacePointBuilding};
  static constexpr std::array<const char*, NStates> StateNames{
    "Space point building",
    "ACTS seeding",
    "Track finding"};
};

template <int nLayers>
template <typename Func>
float TrackerACTS<nLayers>::evaluateTask(Func&& task, std::string_view taskName)
{
  LOG(debug) << " + Starting " << taskName;
  const auto start = std::chrono::high_resolution_clock::now();
  task();
  const auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> diff{end - start};

  LOG(debug) << " - " << taskName << " completed in: " << std::fixed << std::setprecision(2) << diff.count() << " ms";
  return static_cast<float>(diff.count());
}

} // namespace o2::trk

#endif /* ALICE3_GLOBALRECONSTRUCTION_INCLUDE_TRACKERACTS_H_ */
