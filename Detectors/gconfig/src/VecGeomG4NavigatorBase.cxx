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

#include "VecGeomG4NavigatorBase.h"

#include "G4VPhysicalVolume.hh"

#include <VecGeom/management/BVHManager.h>
#include <VecGeom/navigation/BVHNavigator.h>
#include <VecGeom/navigation/BVHSafetyEstimator.h>
#include <VecGeom/navigation/VSafetyEstimator.h>
#include <VecGeom/volumes/LogicalVolume.h>
#include <VecGeom/volumes/PlacedVolume.h>

#include <fairlogger/Logger.h>

#include <algorithm>

namespace
{
/// Longest run of Geant4 levels one flattened VecGeom placement can stand for.
constexpr std::size_t kMaxChain = 16;
/// Deepest Geant4 touchable the ALICE geometry can produce, with headroom.
constexpr int kMaxDepth = 64;
} // namespace

namespace o2::simsetup
{

double VecGeomG4NavigatorBase::boundedSafety(vecgeom::NavigationState const& state, const G4ThreeVector& point,
                                             double limit)
{
  auto const* pvol = state.Top();
  auto const* lvol = pvol->GetLogicalVolume();
  auto const* estimator = lvol->GetSafetyEstimator();
  if (estimator == nullptr) {
    return 0.;
  }
  if (limit < kInfinity && estimator == vecgeom::BVHSafetyEstimator::Instance() && lvol->GetDaughters().size() > 0) {
    // The BVH estimator's own computation, with the search limited.
    vecgeom::Transformation3D m;
    state.TopMatrix(m);
    const V3 local = m.Transform(toVG(point));
    double safety = pvol->SafetyToOut(local);
    if (safety > 0.) {
      safety = vecgeom::BVHNavigator::ComputeBVHSafety<vecgeom::BVHSafetyEstimator>(
        *vecgeom::BVHManager::GetBVH(lvol), local, safety, std::min(safety, limit * kG4ToVG));
    }
    return safety * kVGToG4;
  }
  return estimator->ComputeSafety(toVG(point), state) * kVGToG4;
}

G4VPhysicalVolume* VecGeomG4NavigatorBase::historyFromState(vecgeom::NavigationState const& state)
{
  // Collect the Geant4 volumes the state stands for, then keep the levels the history already has
  // right and rebuild only from the first difference down: NewLevel composes a transform per level.
  G4VPhysicalVolume* want[kMaxDepth];
  int n = 0;
  if (!state.IsOutside()) {
    const int levels = state.GetCurrentLevel();
    for (int l = 0; l < levels && n < kMaxDepth; ++l) {
      auto const* placed = state.At(l);
      if (placed == nullptr) {
        break;
      }
      unsigned size = 0;
      auto* const* chain = mMap.chain(placed->id(), size);
      if (size == 0) {
        LOG(fatal) << "VecGeom placement " << placed->GetLabel() << " (id " << placed->id() << ") at level " << l
                   << " has no Geant4 counterpart";
      }
      for (unsigned c = 0; c < size && n < kMaxDepth; ++c) {
        want[n++] = chain[c];
      }
    }
  }

  if (n == 0) {
    // Outside the world. A null first entry is how G4NavigationHistory says so.
    fHistory.Reset();
    fHistory.SetFirstEntry(nullptr);
    return nullptr;
  }

  const int depth = static_cast<int>(fHistory.GetDepth());
  int common = 0;
  while (common <= depth && common < n && fHistory.GetVolume(common) == want[common]) {
    ++common;
  }
  if (common == 0) {
    fHistory.Reset();
    fHistory.SetFirstEntry(want[0]);
    common = 1;
  } else if (depth >= common) {
    fHistory.BackLevel(depth - common + 1);
  }
  for (int k = common; k < n; ++k) {
    // The copy number is what TG4StepManager::CurrentVolID and CurrentVolOffID report, so it has to
    // be filled exactly as the TGeo navigator fills it.
    fHistory.NewLevel(want[k], kNormal, want[k]->GetCopyNo());
  }
  return fHistory.GetTopVolume();
}

bool VecGeomG4NavigatorBase::stateFromHistory(vecgeom::NavigationState& state) const
{
  // A run of levels that ends in a Geant4 volume several placements share is settled by comparing
  // the run against the placements' recorded chains, which is exact.
  state.Clear();
  if (mMap.world() == nullptr || fHistory.GetVolume(0) == nullptr) {
    return false;
  }
  state.Push(mMap.world());
  const std::size_t depth = fHistory.GetDepth();
  std::size_t l = mMap.chainSize(mMap.world()->id());
  while (l <= depth) {
    G4VPhysicalVolume* run[kMaxChain];
    vecgeom::VPlacedVolume const* found = nullptr;
    std::size_t n = 0;
    for (std::size_t e = l; e <= depth && n < kMaxChain; ++e, ++n) {
      run[n] = fHistory.GetVolume(e);
      auto const* candidate = mMap.toVecGeom(run[n]->GetInstanceID());
      if (candidate == nullptr) {
        continue; // an assembly level; the chain reaches further down
      }
      if (candidate == VecGeomG4Map::ambiguous()) {
        for (auto const* c : mMap.candidates(run[n]->GetInstanceID())) {
          if (mMap.chainMatches(c, run, n + 1)) {
            found = c;
            break;
          }
        }
      } else if (mMap.chainMatches(candidate, run, n + 1)) {
        found = candidate;
      }
      if (found != nullptr) {
        break;
      }
    }
    if (found == nullptr) {
      state.Clear();
      return false;
    }
    state.Push(found);
    l += mMap.chainSize(found->id());
  }
  return true;
}

} // namespace o2::simsetup
