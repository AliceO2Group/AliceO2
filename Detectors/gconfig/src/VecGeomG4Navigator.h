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

#ifndef O2_SIMSETUP_VECGEOMG4NAVIGATOR_H_
#define O2_SIMSETUP_VECGEOMG4NAVIGATOR_H_

#include "VecGeomG4NavigatorBase.h"

namespace o2::simsetup
{

/// A Geant4 tracking navigator that answers every navigation query from VecGeom while keeping the
/// Geant4 navigation history, the touchable the scoring code reads, in step with the VecGeom state.
/// It works as G4VecGeomNav's
/// TG4VecGeomNavigator does:
///
/// - ComputeStep leaves the current volume alone. It records whether the step ends on a boundary
///   and, for a daughter hit, the state that enters it.
/// - The locate on a boundary relocates: into the recorded daughter, or out of the current volume,
///   which then stays blocked for the next step. The point is first pushed across the face by a
///   small depth, and afterwards leaves every volume it is flush with and heading out of.
/// - Safety is zero only at the boundary point itself.
///
/// Unlike TG4VecGeomNavigator, the VecGeom geometry is converted from TGeo, not from Geant4, so one
/// VecGeom placement can stand for a chain of g4root volumes (VecGeomG4Map).
class VecGeomG4Navigator : public VecGeomG4NavigatorBase
{
 public:
  /// \param pushDepth how far past a face (cm, measured across it) a boundary point is pushed before
  /// it is located. \param zeroSafety answers zero to every safety query.
  VecGeomG4Navigator(VecGeomG4Map const& map, double pushDepth, bool zeroSafety);
  ~VecGeomG4Navigator() override;

  G4double ComputeStep(const G4ThreeVector& globalPoint, const G4ThreeVector& direction,
                       const G4double proposedStepLength, G4double& newSafety) override;

  G4VPhysicalVolume* ResetHierarchyAndLocate(const G4ThreeVector& point, const G4ThreeVector& direction,
                                             const G4TouchableHistory& history) override;

  G4VPhysicalVolume* LocateGlobalPointAndSetup(const G4ThreeVector& point, const G4ThreeVector* direction = nullptr,
                                               const G4bool relativeSearch = true,
                                               const G4bool ignoreDirection = true) override;

  void LocateGlobalPointWithinVolume(const G4ThreeVector& position) override;

  G4double ComputeSafety(const G4ThreeVector& globalPoint, const G4double proposedMaxLength = DBL_MAX,
                         const G4bool keepState = true) override;

  // Both point out of the volume left; the local one is in the frame of the final volume.
  G4ThreeVector GetLocalExitNormal(G4bool* valid) override;
  G4ThreeVector GetGlobalExitNormal(const G4ThreeVector& point, G4bool* valid) override;

 private:
  /// Rewrites the Geant4 history from mCurState, unless it already stands for it.
  G4VPhysicalVolume* updateG4History();
  void locateFromWorld(const V3& point);
  /// Go up past every volume the point is on the surface of and heading out of, then down again.
  void leaveFlushVolumes(const V3& point, const V3& dir, int minLevel, vecgeom::VPlacedVolume const* avoid);
  /// Sets fEnteredDaughter and fExitedMother from the paths before (mReloScratch) and after a crossing.
  void updateCrossingFlags(bool entering);
  void clearLastExited() { mCurState.SetLastExited(mEmptyState.GetLastExitedState()); }
  /// How far (cm) a boundary point is pushed along the direction before it is located.
  double boundaryPush(const V3& point, const V3& dir) const;
  /// The normal of the surface the last geometry-limited ComputeStep ended on, global, unit length.
  bool computeExitNormal(const G4ThreeVector& point, V3& globalNormal) const;

  double mPushDepth = 1.e-9; ///< cm
  bool mZeroSafety = false;

  vecgeom::NavigationState mCurState;     ///< the volume the track is in; changed by the locates only
  vecgeom::NavigationState mNextState;    ///< the state entering the daughter the last ComputeStep hit
  vecgeom::NavigationState mStepState;    ///< scratch: a trial step
  vecgeom::NavigationState mReloScratch;  ///< scratch: the state before a crossing
  vecgeom::NavigationState mPathScratch;  ///< scratch: path comparisons
  vecgeom::NavigationState mHistoryState; ///< the state fHistory was built from
  vecgeom::NavigationState mEmptyState;   ///< permanently empty; its last-exited entry clears others
  vecgeom::NavigationState mNormalState;  ///< the volume whose surface the last boundary step ended on
  bool mHistoryValid = false;

  bool mWouldEnter = false; ///< the last ComputeStep ends by entering a daughter
  bool mWouldExit = false;  ///< the last ComputeStep ends by leaving the current volume
  G4ThreeVector mNextPoint{-1e8, -1e8, -1e8};
  G4ThreeVector mLastLocatedPoint{-1e8, -1e8, -1e8};
  bool mLocatedOnBoundary = false;
  G4ThreeVector mSafetyOrig{-1e8, -1e8, -1e8}; ///< the last point a safety was computed for
  double mLastSafety = 0.;                     ///< mm
  bool mNormalEnter = false;
  bool mNormalValid = false;
  G4ThreeVector mNormalPoint{-1e8, -1e8, -1e8};

  /// Geant4's thresholds for a track that is not moving.
  static constexpr int kActionThresholdNoZeroSteps = 10;
  static constexpr int kAbandonThresholdNoZeroSteps = 25;
  int mNzeroSteps = 0;

  // What the navigator had to work around, reported at the end.
  long mZeroStepCount = 0;
  long mStuckPushCount = 0;
  long mAbandonCount = 0;
  long mNegativeSafetyCount = 0;
  long mUnmappableHistoryCount = 0;
  long mRelocatedResumeCount = 0;
  long mNoNormalCount = 0;
};

} // namespace o2::simsetup

#endif
