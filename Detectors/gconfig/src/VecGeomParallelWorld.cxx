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

#include "VecGeomParallelWorld.h"

#include "TG4RootDetectorConstruction.h"

#include <VecGeom/volumes/LogicalVolume.h>
#include <VecGeom/volumes/PlacedVolume.h>

#include <TGeoManager.h>
#include <TGeoMatrix.h>
#include <TGeoParallelWorld.h>
#include <TGeoPhysicalNode.h>
#include <TObjArray.h>

#include <fairlogger/Logger.h>

#include <algorithm>

namespace
{
/// TGeoParallelWorld has no accessor for its nodes; a derived class may name the protected member.
struct ParallelWorldAccess : TGeoParallelWorld {
  static TObjArray* physicalNodes(TGeoParallelWorld* pw) { return pw->*(&ParallelWorldAccess::fPhysical); }
};

constexpr std::uint64_t kHashSeed = 1469598103934665603ULL;
std::uint64_t hashStep(std::uint64_t h, int id) { return (h ^ static_cast<std::uint64_t>(id)) * 1099511628211ULL; }
} // namespace

namespace o2::simsetup
{

bool VecGeomParallelWorld::build(TG4RootDetectorConstruction const& dc, VecGeomG4Map const& map)
{
  auto* pw = gGeoManager != nullptr ? gGeoManager->GetParallelWorld() : nullptr;
  if (pw == nullptr || !pw->IsClosed()) {
    return false;
  }
  auto* physical = ParallelWorldAccess::physicalNodes(pw);
  const int n = physical != nullptr ? physical->GetEntriesFast() : 0;
  if (n == 0) {
    return false;
  }
  std::vector<vecgeom::NavigationState> states(n);
  double worstOffset = 0.;
  for (int i = 0; i < n; ++i) {
    auto* pnode = static_cast<TGeoPhysicalNode*>(physical->UncheckedAt(i));
    auto volumeAt = [&](std::size_t level) {
      return level == 0 ? dc.GetTopPV() : dc.GetG4VPhysicalVolume(pnode->GetNode(static_cast<int>(level)));
    };
    if (!map.stateFromPath(volumeAt, static_cast<std::size_t>(pnode->GetLevel()), states[i])) {
      LOG(fatal) << "Parallel world node " << pnode->GetName() << " matches no VecGeom path";
    }
    // The VecGeom placement has to sit where TGeo has the node, alignment and rotation included.
    vecgeom::Transformation3D m;
    states[i].TopMatrix(m);
    for (auto const& local : {V3(0., 0., 0.), V3(1., 0., 0.), V3(0., 1., 0.), V3(0., 0., 1.)}) {
      const double in[3] = {local[0], local[1], local[2]};
      double master[3];
      pnode->GetMatrix()->LocalToMaster(in, master);
      worstOffset = std::max(worstOffset, (m.InverseTransform(local) - V3(master[0], master[1], master[2])).Mag());
    }
  }
  if (worstOffset > 1.e-6) {
    LOG(fatal) << "Parallel world nodes are up to " << worstOffset << " cm away from their VecGeom placements";
  }
  build(std::move(states));
  return true;
}

void VecGeomParallelWorld::build(std::vector<vecgeom::NavigationState> states)
{
  mStates = std::move(states);
  const int n = size();
  mToLocal.resize(n);
  mSolid.resize(n);
  mLevel.resize(n);
  mIsTop.clear();
  mByPath.clear();
  std::vector<V3> boxes(2 * static_cast<std::size_t>(n));
  mMin.Set(vecgeom::kInfLength);
  mMax.Set(-vecgeom::kInfLength);
  mMinLevel = std::numeric_limits<int>::max();
  mMaxLevel = 0;
  for (int i = 0; i < n; ++i) {
    auto const& state = mStates[i];
    state.TopMatrix(mToLocal[i]);
    mSolid[i] = state.Top()->GetUnplacedVolume();
    mLevel[i] = state.GetCurrentLevel();
    if (mLevel[i] < 2) {
      LOG(fatal) << "The world volume cannot be a parallel world node";
    }
    mMinLevel = std::min(mMinLevel, mLevel[i]);
    mMaxLevel = std::max(mMaxLevel, mLevel[i]);

    V3 lo, hi;
    mSolid[i]->Extent(lo, hi);
    V3 bmin(vecgeom::kInfLength), bmax(-vecgeom::kInfLength);
    for (int c = 0; c < 8; ++c) {
      const V3 corner =
        mToLocal[i].InverseTransform(V3((c & 1) ? hi[0] : lo[0], (c & 2) ? hi[1] : lo[1], (c & 4) ? hi[2] : lo[2]));
      for (int a = 0; a < 3; ++a) {
        bmin[a] = std::min(bmin[a], corner[a]);
        bmax[a] = std::max(bmax[a], corner[a]);
      }
    }
    boxes[2 * i] = bmin;
    boxes[2 * i + 1] = bmax;
    for (int a = 0; a < 3; ++a) {
      mMin[a] = std::min(mMin[a], bmin[a]);
      mMax[a] = std::max(mMax[a], bmax[a]);
    }

    std::uint64_t h = kHashSeed;
    for (int l = 0; l < mLevel[i]; ++l) {
      h = hashStep(h, state.At(l)->id());
    }
    mByPath.emplace(h, i);
    const auto id = static_cast<std::size_t>(state.Top()->id());
    if (id >= mIsTop.size()) {
      mIsTop.resize(id + 1, 0);
    }
    mIsTop[id] = 1;
  }
  constexpr double kPad = 1.e-3; // cm
  mMin -= V3(kPad, kPad, kPad);
  mMax += V3(kPad, kPad, kPad);
  mBVH = std::make_unique<BVH>(0, boxes.data(), n);
  LOG(info) << "VecGeom parallel world: " << n << " nodes at levels " << mMinLevel << " to " << mMaxLevel << ", "
            << mBVH->GetNNodes() << " BVH nodes";
}

bool VecGeomParallelWorld::isInAnyNode(vecgeom::NavigationState const& st) const
{
  if (st.IsOutside()) {
    return false;
  }
  const int levels = std::min(static_cast<int>(st.GetCurrentLevel()), mMaxLevel);
  std::uint64_t h = kHashSeed;
  for (int l = 0; l < levels; ++l) {
    const auto id = static_cast<std::size_t>(st.At(l)->id());
    h = hashStep(h, static_cast<int>(id));
    if (l + 1 < mMinLevel || id >= mIsTop.size() || mIsTop[id] == 0) {
      continue;
    }
    // The hash only selects candidates; the path comparison decides.
    const auto range = mByPath.equal_range(h);
    for (auto it = range.first; it != range.second; ++it) {
      if (mLevel[it->second] == l + 1 && isInNode(it->second, st)) {
        return true;
      }
    }
  }
  return false;
}

double VecGeomParallelWorld::safetyToIn(const V3& p, double limit) const
{
  auto const& bvh = *mBVH;
  const F3 pf(static_cast<float>(p[0]), static_cast<float>(p[1]), static_cast<float>(p[2]));
  double best = limit;
  // Node distances come squared and rounded down; a node at or beyond the best safety cannot lower it.
  auto beyond = [&](unsigned int node) {
    return BVH::FinalizeSafetySqr(bvh.NodeSafetySqr(node, pf)) >= best * best;
  };
  if (best <= 0. || beyond(0)) {
    return std::max(best, 0.);
  }
  traverse(
    [&](unsigned int left, unsigned int right, bool& goLeft, bool& goRight, bool&) {
      goLeft = !beyond(left);
      goRight = !beyond(right);
    },
    [&](int i) {
      const double s = mSolid[i]->SafetyToIn(mToLocal[i].Transform(p));
      if (s < best) {
        best = std::max(s, 0.);
      }
      return best == 0.;
    });
  return best;
}

} // namespace o2::simsetup
