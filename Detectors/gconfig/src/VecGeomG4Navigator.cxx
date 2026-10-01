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

#include "VecGeomG4Navigator.h"

#include "G4Exception.hh"
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

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

namespace
{
const G4ThreeVector kNoPoint(-1e8, -1e8, -1e8);

bool samePoint(const G4ThreeVector& a, const G4ThreeVector& b) { return a.diff2(b) < 1e-20; }

vecgeom::VPlacedVolume const* topOf(vecgeom::NavigationState const& st)
{
  return st.IsOutside() ? nullptr : st.Top();
}
} // namespace

namespace o2::simsetup
{

VecGeomG4Navigator::VecGeomG4Navigator(VecGeomG4Map const& map, double pushDepth, bool zeroSafety)
  : VecGeomG4NavigatorBase(map), mPushDepth(pushDepth), mZeroSafety(zeroSafety)
{
  mEmptyState.Clear();
}

VecGeomG4Navigator::~VecGeomG4Navigator()
{
  LOG(info) << "VecGeom navigation: zero steps " << mZeroStepCount << ", stuck pushes " << mStuckPushCount
            << ", abandoned " << mAbandonCount << ", negative safeties " << mNegativeSafetyCount
            << ", unmappable touchables " << mUnmappableHistoryCount << ", relocated resumes "
            << mRelocatedResumeCount << ", missing exit normals " << mNoNormalCount;
}

G4VPhysicalVolume* VecGeomG4Navigator::updateG4History()
{
  // The history is a function of the VecGeom state alone, so it is rebuilt only when that changed.
  if (mHistoryValid && mCurState.HasSamePathAsOther(mHistoryState)) {
    return fHistory.GetTopVolume();
  }
  mHistoryState = mCurState;
  mHistoryValid = true;
  return historyFromState(mCurState);
}

void VecGeomG4Navigator::locateFromWorld(const V3& point)
{
  mCurState.Clear();
  vecgeom::GlobalLocator::LocateGlobalPoint(vecgeom::GeoManager::Instance().GetWorld(), point, mCurState, true);
}

/// The push is set across the face, as a depth, so that a grazing track is moved off the face as
/// surely as one at normal incidence, and it is kept small: TOF has layers 2.4e-8 cm apart, and a
/// fixed push along the direction steps over them. It never goes below a thousand times the rounding
/// of the largest coordinate, which matters far from the origin.
double VecGeomG4Navigator::boundaryPush(const V3& point, const V3& dir)
{
  constexpr double kMaxPush = 1.e-4; // cm along the direction
  const double big = std::max({std::abs(point.x()), std::abs(point.y()), std::abs(point.z())});
  const double rounding = 1.e3 * big * std::numeric_limits<double>::epsilon();

  // The face just crossed: the entered daughter's, or the current volume's own.
  vecgeom::NavigationState const& st = mWouldEnter ? mNextState : mCurState;
  double cosn = 1.;
  mPushNormalValid = false;
  if (!st.IsOutside() && st.Top() != nullptr) {
    vecgeom::Transformation3D m;
    st.TopMatrix(m);
    V3 n;
    st.Top()->GetUnplacedVolume()->Normal(m.Transform(point), n);
    const double c = std::abs(n.Dot(m.TransformDirection(dir)));
    if (c > 0. && n.Mag2() > 0.5) {
      cosn = c;
    }
    if (n.Mag2() > 0.5) {
      mPushNormal = m.InverseTransformDirection(n);
      mPushNormalValid = true;
    }
  }
  return std::max(rounding, std::min(mPushDepth / cosn, kMaxPush));
}

G4double VecGeomG4Navigator::ComputeStep(const G4ThreeVector& globalPoint, const G4ThreeVector& direction,
                                         const G4double proposedStepLength, G4double& newSafety)
{
  newSafety = 0.;
  mWouldEnter = false;
  mWouldExit = false;

  auto const* top = topOf(mCurState);
  if (top == nullptr) { // the track is outside the world
    return kInfinity;
  }
  auto const* navigator = top->GetLogicalVolume()->GetNavigator();

  // On the point a boundary locate left the track on, the safety is zero; a point seen before
  // reuses its safety. Otherwise the navigator computes it with the step.
  bool calcSafety = !mZeroSafety && !(mLocatedOnBoundary && samePoint(globalPoint, mLastLocatedPoint));
  if (calcSafety && samePoint(globalPoint, mSafetyOrig)) {
    calcSafety = false;
    newSafety = mLastSafety;
  }

  // The step is computed on a copy. The state stays as the locate left it for every call until the
  // next locate, which is what the field propagator relies on when it calls this from trial points
  // along the curve. The volume the last crossing left is blocked in the first call only, and only
  // while the direction points away from it; a track turning back into it must see its boundary.
  mStepState = mCurState;
  if (mExitBlockPending) {
    const bool block = samePoint(globalPoint, mLastLocatedPoint) &&
                       (mExitNormalFromPush ? directionLeaves(mPushNormal, toDir(direction))
                                            : directionLeaves(mExitedState, toVG(globalPoint), toDir(direction)));
    if (!block) {
      mStepState.SetLastExited(mEmptyState.GetLastExitedState());
    }
    clearLastExited();
  }
  const double limit = std::min(proposedStepLength * kG4ToVG, static_cast<double>(vecgeom::kInfLength));
  double safety = 0.;
  double vgStep = navigator->ComputeStepAndSafety(toVG(globalPoint), toDir(direction), limit, mStepState, calcSafety,
                                                  safety, true);
  if (calcSafety) {
    if (safety < 0.) {
      ++mNegativeSafetyCount;
      safety = 0.;
    }
    newSafety = safety * kVGToG4;
    mSafetyOrig = globalPoint;
    mLastSafety = newSafety;
  }
  const bool boundaryLimited = vgStep < limit;
  G4double step = std::max(vgStep, 0.) * kVGToG4;
  const bool entering = mStepState.GetCurrentLevel() > mCurState.GetCurrentLevel();

  // A track that is not moving, handled as G4Navigator does: after ten zero steps the step is
  // lengthened by 100 kCarTolerance, after twenty-five the event is aborted.
  if (step < 0.05 * kCarTolerance) {
    ++mZeroStepCount;
    if (++mNzeroSteps >= kActionThresholdNoZeroSteps) {
      ++mStuckPushCount;
      step += 100. * kCarTolerance;
      if (mNzeroSteps >= kAbandonThresholdNoZeroSteps) {
        ++mAbandonCount;
        std::ostringstream msg;
        msg << "Track stuck or not moving: " << mNzeroSteps << " zero steps in " << top->GetLabel() << " at ("
            << globalPoint.x() << ", " << globalPoint.y() << ", " << globalPoint.z() << ") mm. Event aborted, as "
            << "G4Navigator does.";
        mNzeroSteps = 0;
        G4Exception("VecGeomG4Navigator::ComputeStep()", "GeomNav0003", EventMustBeAborted, msg.str().c_str());
      }
    }
  } else {
    mNzeroSteps = 0;
  }

  if (boundaryLimited) {
    mWouldEnter = entering;
    mWouldExit = !entering;
    mNextPoint = globalPoint + step * direction;
    if (entering) {
      mNextState = mStepState;
    }
    // The surface this step ends on, for the exit normal: the entered daughter's or our own.
    mNormalState = entering ? mNextState : mCurState;
    mNormalEnter = entering;
    mNormalPoint = mNextPoint;
    mNormalValid = true;
  } else {
    step = kInfinity;
    mNormalValid = false;
  }
  return step;
}

/// Geant4's rule for a point on several coincident faces: go up past every volume the point is on
/// the surface of and heading out of, then look down again from there, never back into the volume
/// just left, which stays blocked for the next step. Bounded, because popping and descending can
/// meet another flush face. It never pops to or above minLevel: a daughter the step entered stays
/// entered, because the step decided that with the chord direction, and the direction a locate gets
/// under a field is the momentum. Nor does it descend back into `avoid`, the volume the crossing
/// exited: on a face a helix touches tangentially chord and momentum disagree about the side.
void VecGeomG4Navigator::leaveFlushVolumes(const V3& point, const V3& dir, int minLevel,
                                           vecgeom::VPlacedVolume const* avoid)
{
  for (int round = 0; round < 4; ++round) {
    vecgeom::VPlacedVolume const* left = nullptr;
    while (topOf(mCurState) != nullptr && static_cast<int>(mCurState.GetCurrentLevel()) > minLevel) {
      vecgeom::Transformation3D m;
      mCurState.TopMatrix(m);
      if (mCurState.Top()->GetUnplacedVolume()->DistanceToOut(m.Transform(point), m.TransformDirection(dir)) > 0.) {
        break;
      }
      left = mCurState.Top();
      mCurState.SetLastExited();
      setExited(mCurState, false);
      if (mCurState.GetCurrentLevel() <= 1) {
        mCurState.Clear(); // nothing to travel in even in the world: the track left it
        return;
      }
      mCurState.Pop();
    }
    if (left == nullptr) {
      return;
    }
    auto const* mother = mCurState.Top();
    vecgeom::Transformation3D m;
    mCurState.TopMatrix(m);
    const auto level = mCurState.GetCurrentLevel();
    const auto blocked = mCurState.GetLastExitedState();
    mCurState.Pop();
    vecgeom::GlobalLocator::LocateGlobalPointExclVolume(mother, left, m.Transform(point), mCurState, false);
    mCurState.SetLastExited(blocked);
    if (mCurState.GetCurrentLevel() == level) {
      return; // no daughter holds the point
    }
    if (avoid != nullptr && mCurState.GetCurrentLevel() > level) {
      // Undo a descent into the volume the crossing exited, at whatever depth the search put it.
      mPathScratch = mCurState;
      while (mPathScratch.GetCurrentLevel() > level && mPathScratch.Top() != avoid) {
        mPathScratch.Pop();
      }
      if (mPathScratch.GetCurrentLevel() > level) {
        while (mCurState.GetCurrentLevel() > level) {
          mCurState.Pop();
        }
        return;
      }
    }
  }
}

/// Geant4's two flags describe the transition, not a change of depth: a track leaving one volume
/// straight into a touching sibling has both exited and entered. Everything below the common prefix
/// of the old path (mReloScratch) was left, everything below it on the new path entered.
void VecGeomG4Navigator::updateCrossingFlags(bool entering)
{
  const int preLevel = mReloScratch.IsOutside() ? 0 : static_cast<int>(mReloScratch.GetCurrentLevel());
  const int curLevel = mCurState.IsOutside() ? 0 : static_cast<int>(mCurState.GetCurrentLevel());
  mPathScratch = mReloScratch;
  mStepState = mCurState;
  int la = preLevel, lb = curLevel;
  while (la > lb) {
    mPathScratch.Pop();
    --la;
  }
  while (lb > la) {
    mStepState.Pop();
    --lb;
  }
  while (la > 0 && !mPathScratch.HasSamePathAsOther(mStepState)) {
    mPathScratch.Pop();
    mStepState.Pop();
    --la;
  }
  fExitedMother = !entering || la < preLevel;
  fEnteredDaughter = entering || la < curLevel;
}

G4VPhysicalVolume* VecGeomG4Navigator::LocateGlobalPointAndSetup(const G4ThreeVector& point,
                                                                 const G4ThreeVector* direction,
                                                                 const G4bool relativeSearch, const G4bool)
{
  // A boundary is being crossed when Geant4 says the last step was limited by the geometry (or the
  // point is where the last ComputeStep put the boundary) and that step found one.
  const bool onBoundary = relativeSearch && (fWasLimitedByGeometry || samePoint(point, mNextPoint));
  const bool crossing = onBoundary && (mWouldEnter || mWouldExit) && direction != nullptr;
  const bool entering = crossing && mWouldEnter;
  const V3 p = toVG(point);
  const V3 dir = direction != nullptr ? toDir(*direction) : V3(0., 0., 0.);
  const double push = (onBoundary && direction != nullptr) ? boundaryPush(p, dir) : 0.;
  const V3 q = p + push * dir;

  fWasLimitedByGeometry = false;
  fEnteredDaughter = false;
  fExitedMother = false;
  mLocatedOnBoundary = false;
  mLastLocatedPoint = point;
  mSafetyOrig = kNoPoint;
  clearLastExited();

  if (!relativeSearch) {
    mNzeroSteps = 0; // a new track: nothing of the previous one applies
    locateFromWorld(p);
  } else if (topOf(mCurState) == nullptr) {
    locateFromWorld(p);
  } else if (entering) {
    // Into the daughter the step hit, then down inside it.
    mReloScratch = mCurState;
    mCurState = mNextState;
    auto const* daughter = mCurState.Top();
    mCurState.Pop();
    vecgeom::Transformation3D m;
    mCurState.TopMatrix(m);
    vecgeom::GlobalLocator::LocateGlobalPoint(daughter, daughter->GetTransformation()->Transform(m.Transform(q)),
                                              mCurState, false);
    mLocatedOnBoundary = true;
  } else if (crossing) {
    // Out of the current volume: up until the point is contained, then down, never back into the
    // volume just left; that volume is blocked at zero distance in the next ComputeStep while the
    // direction points away from it, as G4Navigator's fBlockedPhysicalVolume.
    mReloScratch = mCurState;
    if (mCurState.GetCurrentLevel() <= 1) {
      mCurState.Clear(); // left the world
    } else {
      vecgeom::Transformation3D m;
      mCurState.TopMatrix(m);
      vecgeom::GlobalLocator::RelocatePointFromPathForceDifferent(m.Transform(q), mCurState);
      mReloScratch.SetLastExited();
      mCurState.SetLastExited(mReloScratch.GetLastExitedState());
      setExited(mReloScratch, true);
    }
    mLocatedOnBoundary = true;
  } else {
    // Anywhere else: from the current path, up until contained, then down.
    vecgeom::Transformation3D m;
    mCurState.TopMatrix(m);
    vecgeom::GlobalLocator::RelocatePointFromPath(m.Transform(q), mCurState);
    if (topOf(mCurState) == nullptr) {
      locateFromWorld(p);
    }
    mLocatedOnBoundary = onBoundary;
  }
  if (crossing) {
    leaveFlushVolumes(q, dir, entering ? static_cast<int>(mNextState.GetCurrentLevel()) : 0,
                      entering ? nullptr : topOf(mReloScratch));
    updateCrossingFlags(entering);
  }
  mWouldEnter = false;
  mWouldExit = false;
  return updateG4History();
}

G4VPhysicalVolume* VecGeomG4Navigator::ResetHierarchyAndLocate(const G4ThreeVector& point, const G4ThreeVector&,
                                                               const G4TouchableHistory& history)
{
  // A track resumes from a stored touchable, usually a secondary starting where its parent's step
  // ended. The touchable names its volume; it is kept unless it does not hold the point.
  fWasLimitedByGeometry = false;
  fEnteredDaughter = false;
  fExitedMother = false;
  mWouldEnter = false;
  mWouldExit = false;
  mLocatedOnBoundary = false;
  mLastLocatedPoint = point;
  mSafetyOrig = kNoPoint;
  mNzeroSteps = 0;
  fHistory = *history.GetHistory();
  const V3 p = toVG(point);
  bool kept = false;
  if (stateFromHistory(mCurState)) {
    vecgeom::Transformation3D m;
    mCurState.TopMatrix(m);
    if (mCurState.Top()->GetUnplacedVolume()->Contains(m.Transform(p))) {
      kept = true;
    } else {
      ++mRelocatedResumeCount;
      vecgeom::GlobalLocator::RelocatePointFromPath(m.Transform(p), mCurState);
      if (topOf(mCurState) == nullptr) {
        locateFromWorld(p);
      }
    }
  } else {
    if (++mUnmappableHistoryCount <= 10) {
      LOG(warning) << "VecGeom navigation: a touchable matches no VecGeom path; locating from the world";
    }
    locateFromWorld(p);
  }
  clearLastExited();
  if (kept) {
    // The touchable's history is the path of the state: keep it rather than rebuild it.
    mHistoryState = mCurState;
    mHistoryValid = true;
    return fHistory.GetTopVolume();
  }
  mHistoryValid = false;
  return updateG4History();
}

void VecGeomG4Navigator::LocateGlobalPointWithinVolume(const G4ThreeVector& position)
{
  // The caller guarantees the point is in the current volume, so the state stays; only what
  // described the last crossing is dropped, as in G4Navigator.
  mLastLocatedPoint = position;
  mLocatedOnBoundary = false;
  mWouldEnter = false;
  mWouldExit = false;
  fEnteredDaughter = false;
  fExitedMother = false;
  clearLastExited();
}

G4double VecGeomG4Navigator::ComputeSafety(const G4ThreeVector& globalPoint, const G4double, const G4bool)
{
  if (mZeroSafety) {
    return 0.;
  }
  if (mLocatedOnBoundary && samePoint(globalPoint, mLastLocatedPoint)) {
    return 0.;
  }
  if ((mWouldEnter || mWouldExit) && samePoint(globalPoint, mNextPoint)) {
    return 0.;
  }
  if (samePoint(globalPoint, mSafetyOrig)) {
    return mLastSafety;
  }
  auto const* top = topOf(mCurState);
  if (top == nullptr) {
    return 0.;
  }
  auto const* estimator = top->GetLogicalVolume()->GetSafetyEstimator();
  if (estimator == nullptr) {
    return 0.;
  }
  double safety = estimator->ComputeSafety(toVG(globalPoint), mCurState);
  if (safety < 0.) {
    ++mNegativeSafetyCount;
    safety = 0.;
  }
  mSafetyOrig = globalPoint;
  mLastSafety = safety * kVGToG4;
  return mLastSafety;
}

/// The normal of the surface the last ComputeStep ended on, in global coordinates, with Geant4's
/// convention: out of the volume being left, or into the daughter being entered. The field
/// propagator's intersection locator compares it with the momentum to see whether a curved track
/// turns back through the surface, so the sign is the geometric one, not the direction of travel.
/// VecGeom's Normal() also reports whether the point was on the surface; Geant4 hands back points a
/// few nanometres off the face, so only a degenerate vector is refused.
bool VecGeomG4Navigator::computeExitNormal(const G4ThreeVector& point, V3& globalNormal) const
{
  if (!mNormalValid || topOf(mNormalState) == nullptr) {
    return false;
  }
  vecgeom::Transformation3D m;
  mNormalState.TopMatrix(m);
  V3 localNormal(0., 0., 0.);
  mNormalState.Top()->GetUnplacedVolume()->Normal(m.Transform(toVG(point)), localNormal);
  globalNormal = m.InverseTransformDirection(localNormal);
  const double mag = globalNormal.Mag();
  if (!(mag > 0.5)) {
    return false;
  }
  globalNormal /= mag;
  if (mNormalEnter) {
    globalNormal = -globalNormal;
  }
  return true;
}

G4ThreeVector VecGeomG4Navigator::GetLocalExitNormal(G4bool* valid)
{
  V3 n;
  if (!computeExitNormal(mNormalPoint, n) || topOf(mCurState) == nullptr) {
    ++mNoNormalCount;
    if (valid != nullptr) {
      *valid = false;
    }
    return G4ThreeVector();
  }
  vecgeom::Transformation3D m;
  mCurState.TopMatrix(m);
  const auto local = m.TransformDirection(n);
  if (valid != nullptr) {
    *valid = true;
  }
  return G4ThreeVector(local[0], local[1], local[2]);
}

G4ThreeVector VecGeomG4Navigator::GetGlobalExitNormal(const G4ThreeVector& point, G4bool* valid)
{
  V3 n;
  if (!computeExitNormal(point, n)) {
    ++mNoNormalCount;
    if (valid != nullptr) {
      *valid = false;
    }
    return G4ThreeVector();
  }
  if (valid != nullptr) {
    *valid = true;
  }
  return G4ThreeVector(n[0], n[1], n[2]);
}

} // namespace o2::simsetup
