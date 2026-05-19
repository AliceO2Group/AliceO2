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

#include "Acts/Definitions/Units.hpp"
#include "Framework/Logger.h"

#include "ITStracking/TimeFrame.h"
#include "TH2F.h"

namespace o2::trk
{

/// Configuration for the ACTS-based tracker
struct TrackerACTSConfig {
  // Seeding parameters
  float minPt = 0.4f * Acts::UnitConstants::GeV;             ///< Minimum pT for seeds
  float maxImpactParameter = 10.f * Acts::UnitConstants::mm; ///< Maximum impact parameter
  float cotThetaMax = std::sinh(4.0f);                       ///< Maximum cot(theta), corresponds to eta ~4

  // Delta R cuts for doublet/triplet formation
  float deltaRMinBottom = 5.f * Acts::UnitConstants::mm;   ///< Min deltaR for bottom-middle
  float deltaRMaxBottom = 200.f * Acts::UnitConstants::mm; ///< Max deltaR for bottom-middle
  float deltaRMinTop = 5.f * Acts::UnitConstants::mm;      ///< Min deltaR for middle-top
  float deltaRMaxTop = 200.f * Acts::UnitConstants::mm;    ///< Max deltaR for middle-top

  // Z cuts
  float zMin = -3000.f * Acts::UnitConstants::mm;
  float zMax = 3000.f * Acts::UnitConstants::mm;

  // Collision region
  float collisionRegionMin = -150.f * Acts::UnitConstants::mm;
  float collisionRegionMax = 150.f * Acts::UnitConstants::mm;

  // Quality cuts
  float maxSeedsPerMiddleSP = 2;
  float deltaPhiMax = 0.1f; ///< Maximum phi difference for doublets
};

/// Space point representation for tracking
struct SpacePoint {
  float x{0.f};
  float y{0.f};
  float z{0.f};
  int layer{-1};
  int clusterId{-1};
  int rof{-1};

  // Derived quantities
  float r() const { return std::hypot(x, y); }
  float radius() const { return r(); } // required by Acts::CylindricalGridElement concept
  float phi() const { return std::atan2(y, x); }

  // Variance estimates (can be refined based on cluster properties)
  float varianceR{0.01f}; // ~100 um resolution squared
  float varianceZ{0.01f};
};

/// Seed (triplet of space points)
struct SeedACTS {
  const SpacePoint* bottom{nullptr};
  const SpacePoint* middle{nullptr};
  const SpacePoint* top{nullptr};
  float quality{0.f};
};

/// TRK Tracker using ACTS algorithms for seeding and track finding
template <int nLayers>
class TrackerACTS
{
 public:
  TrackerACTS();
  ~TrackerACTS()
  {
    if (mHistSpacePoints) {
      mHistSpacePoints->SaveAs("/tmp/mHistSpacePoints.C");
      delete mHistSpacePoints;
      mHistSpacePoints = nullptr;
    }
  }

  /// Adopt a TimeFrame for processing
  void adoptTimeFrame(o2::its::TimeFrame<nLayers>& tf);

  /// Main tracking entry point: convert clusters to tracks
  void clustersToTracks();

  /// Configuration
  void setConfig(const TrackerACTSConfig& cfg) { mConfig = cfg; }
  TrackerACTSConfig& getConfig() { return mConfig; }
  const TrackerACTSConfig& getConfig() const { return mConfig; }

  /// Set the magnetic field strength
  void setBz(float bz) { mBz = bz; }

  /// Get the magnetic field strength
  float getBz() const { return mBz; }

  /// Print tracking summary
  void printSummary() const;

 private:
  TH2F* mHistSpacePoints = nullptr;

  /// Build space points from clusters in the TimeFrame
  void buildSpacePoints(int rof);

  /// Create seeds (triplets) from space points using ACTS SeedFinder
  void createSeeds();

  /// Estimate track parameters from a seed using ACTS
  bool estimateTrackParams(const SeedACTS& seed, o2::its::TrackITSExt& track) const;

  /// Run track finding from seeds
  void findTracks();

  /// Assign MC labels to tracks
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

  // Tracking state
  float mBz{0.5f}; ///< Magnetic field in Tesla
  unsigned int mTimeFrameCounter{0};
  double mTotalTime{0.};

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
