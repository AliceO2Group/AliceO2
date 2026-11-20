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
  G4VPhysicalVolume* want[kMaxHistoryDepth];
  int n = 0;
  if (!state.IsOutside()) {
    const int levels = state.GetCurrentLevel();
    for (int l = 0; l < levels && n < kMaxHistoryDepth; ++l) {
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
      for (unsigned c = 0; c < size && n < kMaxHistoryDepth; ++c) {
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

G4VPhysicalVolume* VecGeomG4NavigatorBase::updateHistory(vecgeom::NavigationState const& state)
{
  if (mHistoryValid && state.HasSamePathAsOther(mHistoryState)) {
    return fHistory.GetTopVolume();
  }
  if (mHistoryExact && updateHistoryByOneLevel(state)) {
    ++mHistoryStepCount;
    return fHistory.GetTopVolume();
  }
  ++mHistoryRebuildCount;
  mHistoryState = state;
  mHistoryValid = true;
  auto* top = historyFromState(state);
  // A history at the depth limit may have been cut short.
  mHistoryExact = static_cast<int>(fHistory.GetDepth()) + 1 < kMaxHistoryDepth;
  return top;
}

/// The levels and transforms are those historyFromState makes: it keeps the common Geant4 levels and calls
/// NewLevel for the rest, as here.
bool VecGeomG4NavigatorBase::updateHistoryByOneLevel(vecgeom::NavigationState const& state)
{
  if (state.IsOutside() || mHistoryState.IsOutside()) {
    return false;
  }
  const int cur = state.GetCurrentLevel();
  const int old = mHistoryState.GetCurrentLevel();
  const int levels = static_cast<int>(fHistory.GetDepth()) + 1;
  unsigned left = 0, size = 0;
  G4VPhysicalVolume* const* chain = nullptr;
  if (cur == old + 1) { // into a daughter
    mHistoryChild = state;
    mHistoryChild.Pop();
    if (!mHistoryChild.HasSamePathAsOther(mHistoryState)) {
      return false;
    }
    chain = mMap.chain(state.Top()->id(), size);
  } else if (cur + 1 == old) { // out to the mother
    mHistoryParent = mHistoryState;
    mHistoryParent.Pop();
    if (!mHistoryParent.HasSamePathAsOther(state)) {
      return false;
    }
    mMap.chain(mHistoryState.Top()->id(), left);
  } else if (cur == old && cur > 1) { // into a sibling
    mHistoryChild = state;
    mHistoryChild.Pop();
    mHistoryParent = mHistoryState;
    mHistoryParent.Pop();
    if (!mHistoryChild.HasSamePathAsOther(mHistoryParent)) {
      return false;
    }
    mMap.chain(mHistoryState.Top()->id(), left);
    chain = mMap.chain(state.Top()->id(), size);
  } else {
    return false;
  }
  if ((chain != nullptr && size == 0) || (cur <= old && left == 0) || static_cast<int>(left) >= levels ||
      levels - static_cast<int>(left) + static_cast<int>(size) > kMaxHistoryDepth) {
    return false;
  }
  if (left > 0) {
    fHistory.BackLevel(left);
  }
  for (unsigned c = 0; c < size; ++c) {
    // The copy number is filled exactly as historyFromState fills it.
    fHistory.NewLevel(chain[c], kNormal, chain[c]->GetCopyNo());
  }
  mHistoryState = state;
  return true;
}

bool VecGeomG4NavigatorBase::stateFromHistory(vecgeom::NavigationState& state) const
{
  return mMap.stateFromPath([this](std::size_t level) { return fHistory.GetVolume(level); }, fHistory.GetDepth(), state);
}

} // namespace o2::simsetup
