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

#ifndef O2_SIMSETUP_VECGEOMG4PROPAGATINGNAVIGATOR_H_
#define O2_SIMSETUP_VECGEOMG4PROPAGATINGNAVIGATOR_H_

#include "VecGeomG4NavigatorBase.h"

namespace o2::simsetup
{

/// A lighter VecGeom navigator for Geant4, selected with G4.vecgeomNavigator=kPropagated. ComputeStep
/// uses VecGeom's combined step-and-relocate, and a boundary locate adopts the state that step
/// propagated instead of locating again. Safety is zero for the whole step after a crossing, a stuck
/// step is nudged forward, and the exit normal follows the direction of motion. It does less work per
/// step and per crossing than VecGeomG4Navigator.
class VecGeomG4PropagatingNavigator : public VecGeomG4NavigatorBase
{
 public:
  VecGeomG4PropagatingNavigator(VecGeomG4Map const& map, bool zeroSafety);
  ~VecGeomG4PropagatingNavigator() override;

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

  G4ThreeVector GetLocalExitNormal(G4bool* valid) override;
  G4ThreeVector GetGlobalExitNormal(const G4ThreeVector& point, G4bool* valid) override;

 private:
  vecgeom::NavigationState mCurState;  ///< where the track is now
  vecgeom::NavigationState mNextState; ///< where the last computed step would put it
  vecgeom::NavigationState mPrevState; ///< where it was before the last boundary crossing
  vecgeom::NavigationState mEmptyState; ///< permanently empty; its last-exited entry clears others

  bool mZeroSafety = false;
  bool mHaveNextState = false; ///< mNextState holds the result of a geometry-limited step

  G4ThreeVector mNextPoint{-1e8, -1e8, -1e8}; ///< where the last computed step ends
  G4ThreeVector mLastDirection{0, 0, 1};      ///< direction of the last computed step

  bool mWouldEnter = false;  ///< the last step ends by entering a daughter
  bool mWouldExit = false;   ///< the last step ends by leaving the current volume
  bool mOnBoundary = false;  ///< the current point sits on a boundary
  bool mForceReInit = false; ///< next locate must start from the world, the state is suspect
  bool mCrossed = false;     ///< the last locate acted on a boundary crossing
  bool mExitBlockPending = false;                ///< the next step is the first after leaving mPrevState's volume
  G4ThreeVector mLocatedPoint{-1e8, -1e8, -1e8}; ///< where the last locate put the track

  int mZeroSteps = 0;
  long mNudgedSteps = 0;
  long mGlobalRelocates = 0;
};

} // namespace o2::simsetup

#endif
