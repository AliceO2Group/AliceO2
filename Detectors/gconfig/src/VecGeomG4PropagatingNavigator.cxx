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

#include "VecGeomG4PropagatingNavigator.h"

#include "G4TouchableHistory.hh"
#include "G4VPhysicalVolume.hh"

#include <VecGeom/base/Transformation3D.h>
#include <VecGeom/management/GeoManager.h>
#include <VecGeom/navigation/GlobalLocator.h>
#include <VecGeom/navigation/VNavigator.h>
#include <VecGeom/navigation/VSafetyEstimator.h>
#include <VecGeom/volumes/LogicalVolume.h>
#include <VecGeom/volumes/PlacedVolume.h>

#include <fairlogger/Logger.h>

#include <cmath>

namespace
{
/// Geant4 abandons a track after ~50 zero steps, so a stalled step is nudged forward by this much
/// (in Geant4 units) rather than returned as zero.
const double kNudge = 1.e-3;
} // namespace

namespace o2::simsetup
{

VecGeomG4PropagatingNavigator::VecGeomG4PropagatingNavigator(VecGeomG4Map const& map, bool zeroSafety)
  : VecGeomG4NavigatorBase(map), mZeroSafety(zeroSafety)
{
  mEmptyState.Clear();
}

VecGeomG4PropagatingNavigator::~VecGeomG4PropagatingNavigator()
{
  LOG(info) << "VecGeom navigation: " << mNudgedSteps << " stalled steps nudged forward, " << mGlobalRelocates
            << " relocations restarted from the world";
}

G4double VecGeomG4PropagatingNavigator::ComputeStep(const G4ThreeVector& globalPoint, const G4ThreeVector& direction,
                                                    const G4double proposedStepLength, G4double& newSafety)
{
  newSafety = 0.;
  mLastDirection = direction;
  mHaveNextState = false;

  auto const* top = mCurState.Top();
  if (top == nullptr) { // the track is outside the world
    mWouldEnter = mWouldExit = false;
    return kInfinity;
  }
  auto const* navigator = top->GetLogicalVolume()->GetNavigator();

  double limit = proposedStepLength * kG4ToVG;
  if (!(limit < vecgeom::kInfLength)) {
    limit = vecgeom::kInfLength;
  }

  // VecGeom's own combined entry point, rather than a step followed by a relocation of our own: it
  // is the one that knows how to descend through an assembly, whose placed volume can never be the
  // answer because it has no DistanceToOut.
  // The adopted state marks the volume the crossing left. As in Geant4, it is blocked in the first
  // step after the exit only, and only while the direction points away from it; a track turning
  // back into it must see its boundary.
  const bool block = mExitBlockPending && globalPoint.diff2(mLocatedPoint) < 1.e-20 &&
                     directionLeaves(mPrevState, toVG(globalPoint), toDir(direction));
  mExitBlockPending = false;
  if (!block) {
    mCurState.SetLastExited(mEmptyState.GetLastExitedState());
  }

  double vgSafety = 0.;
  const double vgStep = navigator->ComputeStepAndSafetyAndPropagatedState(
    toVG(globalPoint), toDir(direction), limit, mCurState, mNextState, !mOnBoundary && !mZeroSafety, vgSafety);
  mOnBoundary = false;
  newSafety = (vgSafety > 0. && !mZeroSafety) ? vgSafety * kVGToG4 : 0.;

  G4double step = vgStep * kVGToG4;
  if (mNextState.IsOnBoundary()) {
    // Entering a daughter deepens the state, possibly by more than one level when an assembly
    // stands in between; leaving the current volume does not.
    mWouldEnter = mNextState.GetCurrentLevel() > mCurState.GetCurrentLevel();
    mWouldExit = !mWouldEnter;
    mNextPoint = globalPoint + step * direction;
    mHaveNextState = true;
  } else {
    mWouldEnter = mWouldExit = false;
    step = kInfinity;
  }

  if (vgStep < 0.) {
    // A negative distance means the state and the point disagree. Nudge forward and relocate from
    // the world on the next call rather than propagating the inconsistency.
    mForceReInit = true;
    mHaveNextState = false;
    ++mNudgedSteps;
    mZeroSteps = 0;
    return kNudge;
  }
  if (step < 1.e-10) {
    if (++mZeroSteps > 4) {
      mForceReInit = true;
      mHaveNextState = false;
      ++mNudgedSteps;
      return kNudge;
    }
  } else {
    mForceReInit = false;
    mZeroSteps = 0;
  }
  return step;
}

G4VPhysicalVolume* VecGeomG4PropagatingNavigator::ResetHierarchyAndLocate(const G4ThreeVector&, const G4ThreeVector&,
                                                                          const G4TouchableHistory& history)
{
  // Geant4 hands back a touchable it saved earlier, e.g. when resuming a track whose secondaries
  // were followed first. The VecGeom state is rebuilt from it.
  fEnteredDaughter = false;
  fExitedMother = false;
  mWouldEnter = false;
  mWouldExit = false;
  mOnBoundary = false;
  mHaveNextState = false;
  fHistory = *history.GetHistory();
  if (!stateFromHistory(mCurState) && fHistory.GetVolume(0) != nullptr) {
    LOG(fatal) << "Geant4 handed back a touchable that matches no VecGeom path";
  }
  mPrevState = mCurState;
  return fHistory.GetTopVolume();
}

G4VPhysicalVolume* VecGeomG4PropagatingNavigator::LocateGlobalPointAndSetup(const G4ThreeVector& point,
                                                                            const G4ThreeVector*,
                                                                            const G4bool relativeSearch, const G4bool)
{
  bool onBoundary = fWasLimitedByGeometry;
  if (mHaveNextState && point.diff2(mNextPoint) < 1.e-16) {
    onBoundary = true;
  }

  mPrevState = mCurState;
  mLocatedPoint = point;
  mExitBlockPending = false;

  if (!mForceReInit && relativeSearch && onBoundary && mHaveNextState) {
    // The state on the far side of the boundary was already worked out, and relocated, by the step
    // that found it. Adopting it is cheaper than locating again.
    mCurState = mNextState;
    mExitBlockPending = mWouldExit;
  } else if (mForceReInit || !relativeSearch || onBoundary) {
    mCurState.Clear();
    vecgeom::GlobalLocator::LocateGlobalPoint(vecgeom::GeoManager::Instance().GetWorld(), toVG(point), mCurState, true);
    mForceReInit = false;
    ++mGlobalRelocates;
  }
  // Otherwise the point only moved inside the volume the state already names.

  auto* target = historyFromState(mCurState);
  mCrossed = onBoundary;
  if (onBoundary) {
    fExitedMother = mWouldExit;
    fEnteredDaughter = mWouldEnter;
    mOnBoundary = true;
  }
  mHaveNextState = false;
  return target;
}

void VecGeomG4PropagatingNavigator::LocateGlobalPointWithinVolume(const G4ThreeVector&)
{
  // The track moved inside the volume it is already in, so only the boundary flags change.
  mWouldEnter = false;
  mWouldExit = false;
  mOnBoundary = false;
  mCrossed = false;
  mHaveNextState = false;
  fEnteredDaughter = false;
  fExitedMother = false;
}

G4double VecGeomG4PropagatingNavigator::ComputeSafety(const G4ThreeVector& globalPoint, const G4double, const G4bool)
{
  if (mZeroSafety || mOnBoundary || mCrossed || fEnteredDaughter || fExitedMother || mWouldEnter || mWouldExit) {
    return 0.;
  }
  auto const* top = mCurState.Top();
  if (top == nullptr) {
    return 0.;
  }
  const double safety = top->GetLogicalVolume()->GetSafetyEstimator()->ComputeSafety(toVG(globalPoint), mCurState);
  return (safety > 0.) ? safety * kVGToG4 : 0.;
}

G4ThreeVector VecGeomG4PropagatingNavigator::GetGlobalExitNormal(const G4ThreeVector& point, G4bool* valid)
{
  // The surface just crossed belongs to the volume that was left when the step exited a mother, and
  // to the volume that was entered when it entered a daughter.
  auto const& state = mWouldExit ? mPrevState : mCurState;
  auto const* volume = state.Top();
  if (volume == nullptr) {
    *valid = false;
    return G4ThreeVector(0., 0., 1.);
  }

  vecgeom::Transformation3D m;
  state.TopMatrix(m);
  V3 normal;
  volume->Normal(m.Transform(toVG(point)), normal);
  V3 global = m.InverseTransformDirection(normal);

  // Oriented along the direction of motion, as TGeo's FindNormalFast does.
  const V3 dir = toDir(mLastDirection);
  if (global.Dot(dir) < 0.) {
    global = -global;
  }
  // VecGeom's Normal() also answers whether the point was on the surface; Geant4 hands back points a
  // few nanometres off the face, so only a degenerate vector is refused.
  const double mag2 = global.Mag2();
  *valid = std::isfinite(mag2) && mag2 > 0.25;
  return G4ThreeVector(global[0], global[1], global[2]);
}

G4ThreeVector VecGeomG4PropagatingNavigator::GetLocalExitNormal(G4bool* valid)
{
  // By convention the local normal is expressed in the frame of the final volume.
  const G4ThreeVector global = GetGlobalExitNormal(mNextPoint, valid);
  vecgeom::Transformation3D m;
  mCurState.TopMatrix(m);
  const V3 local = m.TransformDirection(toDir(global));
  return G4ThreeVector(local[0], local[1], local[2]);
}

} // namespace o2::simsetup
