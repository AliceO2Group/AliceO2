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

class VecGeomParallelWorld;

/// A Geant4 tracking navigator that answers every navigation query from VecGeom while keeping the
/// Geant4 navigation history, the touchable the scoring code reads, in step with the VecGeom state.
/// Selected with G4.vecgeomNavigator=kRelocating, the default. It works as G4VecGeomNav's
/// TG4VecGeomNavigator does:
///
/// - ComputeStep leaves the current volume alone. It records whether the step ends on a boundary
///   and, for a daughter hit, the state that enters it.
/// - The locate on a boundary relocates: into the recorded daughter, or out of the current volume.
/// - With the lean locate (G4.vecgeomLeanLocate, the default) the point is pushed 10 nm along the step,
///   which takes it past every coincident face. The geometry setup then steps every mother with daughters
///   with the BVH navigator (GeometryManager::setVecGeomBvhForSmallMothers), whose daughter query refuses
///   at zero distance a daughter the ray is leaving; that is the block of the volume just left.
/// - With the previous locate, kept as a fallback, the point is pushed across the face by a small depth
///   and afterwards leaves every volume it is flush with and heading out of. As in Geant4, the volume
///   left is blocked only in the first ComputeStep after the exit, and only while the direction points
///   away from it.
/// - Safety is zero only at the boundary point itself. A bounded query is answered from the last safety
///   computed in the same volume when that covers the bound.
///
/// Unlike TG4VecGeomNavigator, the VecGeom geometry is converted from TGeo, not from Geant4, so one
/// VecGeom placement can stand for a chain of g4root volumes (VecGeomG4Map).
class VecGeomG4Navigator : public VecGeomG4NavigatorBase
{
 public:
  /// \param pushDepth how far past a face (cm, measured across it) a boundary point is pushed before
  /// it is located. \param zeroSafety answers zero to every safety query.
  /// \param parallelWorld the prioritised placements, or null if there are none.
  /// \param leanLocate locates a crossing as VecGeom relocates: 10 nm along the step, no face normal, no
  /// flush-volume loop; it needs GeometryManager::setVecGeomBvhForSmallMothers(true).
  VecGeomG4Navigator(VecGeomG4Map const& map, double pushDepth, bool zeroSafety,
                     VecGeomParallelWorld const* parallelWorld, bool leanLocate);
  ~VecGeomG4Navigator() override;

  bool leanLocate() const { return mLeanLocate; }

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
  void locateFromWorld(const V3& point);
  /// Go up past every volume the point is on the surface of and heading out of, then down again.
  void leaveFlushVolumes(const V3& point, const V3& dir, int minLevel, vecgeom::NavigationState const* avoid);
  /// Gives a parallel world node holding the point priority over the state the main hierarchy located.
  /// \param exiting the locate follows a step out of a volume, whose path is in mReloScratch.
  /// \return whether the state was moved into a node.
  bool applyParallelWorld(const V3& point, bool exiting);
  /// Locates \a point in the daughters of the top volume of \a state, which holds it.
  static void descendInto(vecgeom::NavigationState& state, const V3& point);
  /// Sets fEnteredDaughter and fExitedMother from the paths before (mReloScratch) and after a crossing.
  void updateCrossingFlags(bool entering);
  void clearLastExited()
  {
    mCurState.SetLastExited(mEmptyState.GetLastExitedState());
    mExitBlockPending = false;
  }
  /// Records the top of \a st as the volume just left, blocked in the next ComputeStep by Geant4's rule.
  void setExited(vecgeom::NavigationState const& st, bool normalFromPush)
  {
    mExitedState = st;
    mExitBlockPending = true;
    mExitNormalFromPush = normalFromPush && mPushNormalValid;
  }
  /// How far (cm) a boundary point is pushed along the direction before it is located. Keeps the
  /// global normal of the face in mPushNormal, for the exit block.
  double boundaryPush(const V3& point, const V3& dir);
  /// The normal of the surface the last geometry-limited ComputeStep ended on, global, unit length.
  bool computeExitNormal(const G4ThreeVector& point, V3& globalNormal) const;

  double mPushDepth = 1.e-9; ///< cm
  bool mZeroSafety = false;
  bool mLeanLocate = false;
  /// The push of the lean locate, VecGeom's own after a boundary (VNavigator::MovePointAfterBoundary), cm.
  static constexpr double kLeanPush = 1.e-6;
  VecGeomParallelWorld const* mParallelWorld = nullptr;
  bool mInParallelNode = false; ///< the track is in a parallel world node; changed by the locates only

  vecgeom::NavigationState mCurState;    ///< the volume the track is in; changed by the locates only
  vecgeom::NavigationState mNextState;   ///< the state entering the daughter the last ComputeStep hit
  vecgeom::NavigationState mStepState;   ///< scratch: a trial step
  vecgeom::NavigationState mReloScratch; ///< scratch: the state before a crossing
  vecgeom::NavigationState mPathScratch; ///< scratch: path comparisons
  vecgeom::NavigationState mEmptyState;  ///< permanently empty; its last-exited entry clears others
  vecgeom::NavigationState mNormalState; ///< the volume whose surface the last boundary step ended on
  vecgeom::NavigationState mExitedState; ///< the volume the last crossing left
  bool mExitBlockPending = false;        ///< the next ComputeStep is the first after that exit
  bool mExitNormalFromPush = false;      ///< mPushNormal is the outward normal of the volume left
  V3 mPushNormal;                        ///< global normal of the face the last push crossed
  bool mPushNormalValid = false;

  bool mWouldEnter = false; ///< the last ComputeStep ends by entering a daughter
  bool mWouldExit = false;  ///< the last ComputeStep ends by leaving the current volume
  G4ThreeVector mNextPoint{-1e8, -1e8, -1e8};
  vecgeom::Vector3D<double> mNextDir{0., 0., 1.}; ///< direction of the step that found the boundary
  G4ThreeVector mLastLocatedPoint{-1e8, -1e8, -1e8};
  bool mLocatedOnBoundary = false;
  G4ThreeVector mSafetyOrig{-1e8, -1e8, -1e8}; ///< the last point a safety was computed for
  double mLastSafety = 0.;                     ///< mm
  double mLastSafetyLimit = kInfinity;         ///< mm; mLastSafety is exact below it, a lower bound above
  bool mNormalEnter = false;
  bool mNormalValid = false;
  G4ThreeVector mNormalPoint{-1e8, -1e8, -1e8};

  /// Geant4's thresholds for a track that is not moving.
  static constexpr int kActionThresholdNoZeroSteps = 10;
  static constexpr int kAbandonThresholdNoZeroSteps = 25;
  int mNzeroSteps = 0;
  bool mZeroLinearSeen = false; ///< a step along the located momentum occurred in the current run of zero steps
  /// Restarts of a run of zero steps after a push, at most this many until the next finite step.
  static constexpr int kMaxPushRestarts = 3;
  int mPushRestarts = 0;
  bool mPushPending = false; ///< the last zero step was lengthened
  G4ThreeVector mPushOrigin; ///< where that step started
  double mPushLength = 0.;   ///< mm
  static constexpr int kRecordZeroRun = kActionThresholdNoZeroSteps - 2;
  /// The volumes the current run of zero steps alternates between, from its kRecordZeroRun-th step.
  vecgeom::NavigationState mZeroRun[2];
  int mZeroRunSize = 0;
  bool inZeroRun(vecgeom::NavigationState const& st) const
  {
    return (mZeroRunSize > 0 && mZeroRun[0].HasSamePathAsOther(st)) ||
           (mZeroRunSize > 1 && mZeroRun[1].HasSamePathAsOther(st));
  }
  void resetZeroSteps()
  {
    mNzeroSteps = 0;
    mZeroLinearSeen = false;
    mPushRestarts = 0;
    mPushPending = false;
    mZeroRunSize = 0;
  }
  G4ThreeVector mLastLocatedDir; ///< the direction the last locate was given

  // What the navigator had to work around, reported at the end.
  long mZeroStepCount = 0;
  long mStuckPushCount = 0;
  long mAbandonCount = 0;
  long mNegativeSafetyCount = 0;
  long mUnmappableHistoryCount = 0;
  long mRelocatedResumeCount = 0;
  long mNoNormalCount = 0;
  long mParallelEntries = 0;   ///< steps that ended on a parallel world node
  long mParallelRelocated = 0; ///< locates the parallel world overrode
  long mPushRestartCount = 0;  ///< runs of zero steps restarted after a push
};

} // namespace o2::simsetup

#endif
