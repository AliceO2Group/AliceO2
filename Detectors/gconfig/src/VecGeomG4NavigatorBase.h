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

#ifndef O2_SIMSETUP_VECGEOMG4NAVIGATORBASE_H_
#define O2_SIMSETUP_VECGEOMG4NAVIGATORBASE_H_

#include "VecGeomG4Map.h"

#include "G4Navigator.hh"
#include "G4SystemOfUnits.hh"
#include "G4ThreeVector.hh"

#include <VecGeom/base/Transformation3D.h>
#include <VecGeom/base/Vector3D.h>
#include <VecGeom/navigation/NavigationState.h>
#include <VecGeom/volumes/PlacedVolume.h>

namespace o2::simsetup
{

/// What the two VecGeom navigators share: the correspondence between a VecGeom navigation state and
/// the Geant4 navigation history, which is the touchable the scoring code reads, and the unit
/// conversion. Geant4 works in millimetres, the VecGeom geometry is converted from TGeo in
/// centimetres; every point crossing this boundary is scaled, directions are not.
class VecGeomG4NavigatorBase : public G4Navigator
{
 protected:
  using V3 = vecgeom::Vector3D<double>;
  static constexpr double kG4ToVG = 1. / CLHEP::cm;
  static constexpr double kVGToG4 = CLHEP::cm;

  explicit VecGeomG4NavigatorBase(VecGeomG4Map const& map) : mMap(map) {}

  static V3 toVG(const G4ThreeVector& p) { return {p.x() * kG4ToVG, p.y() * kG4ToVG, p.z() * kG4ToVG}; }
  static V3 toDir(const G4ThreeVector& d) { return {d.x(), d.y(), d.z()}; }

  /// Rewrites fHistory to stand for \a state, keeping the levels it already has right. Returns the
  /// top volume, or null if the state is outside the world.
  G4VPhysicalVolume* historyFromState(vecgeom::NavigationState const& state);

  /// Builds \a state from fHistory, extending past the Geant4 levels assembly flattening dissolved.
  /// False if a level matches no VecGeom placement; \a state is then empty.
  bool stateFromHistory(vecgeom::NavigationState& state) const;

  /// The safety (mm) at \a point in the top volume of \a state, exact below \a limit (mm) and a valid
  /// lower bound above it: daughters beyond the limit are bounded by their boxes, as in G4VoxelSafety.
  static double boundedSafety(vecgeom::NavigationState const& state, const G4ThreeVector& point, double limit);

  /// Geant4's rule for the volume a track just left (G4NormalNavigation, G4VoxelNavigation): it is
  /// blocked only while the direction points away from it, along its outward normal \a n (global).
  [[gnu::always_inline]] static bool directionLeaves(const V3& n, const V3& dir)
  {
    constexpr double kMinExitingNormalCosine = 1e-3; // as in G4NormalNavigation
    return n.Dot(dir) >= kMinExitingNormalCosine;
  }

  /// The same, with the normal taken from \a exited, which has the volume left on top.
  [[gnu::always_inline]] static bool directionLeaves(vecgeom::NavigationState const& exited, const V3& point,
                                                     const V3& dir)
  {
    if (exited.IsOutside() || exited.Top() == nullptr) {
      return false;
    }
    vecgeom::Transformation3D m;
    exited.TopMatrix(m);
    V3 n;
    exited.Top()->GetUnplacedVolume()->Normal(m.Transform(point), n);
    if (!(n.Mag2() > 0.5)) {
      return false; // no valid exit normal: Geant4 does not block either
    }
    return directionLeaves(m.InverseTransformDirection(n), dir);
  }

  VecGeomG4Map const& mMap;
};

} // namespace o2::simsetup

#endif
