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

#ifndef O2_SIMSETUP_VECGEOMPARALLELWORLD_H_
#define O2_SIMSETUP_VECGEOMPARALLELWORLD_H_

#include "VecGeomG4Map.h"

#include <VecGeom/base/IndexedBVH.h>
#include <VecGeom/base/Transformation3D.h>
#include <VecGeom/base/Vector3D.h>
#include <VecGeom/navigation/NavigationState.h>
#include <VecGeom/volumes/UnplacedVolume.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <unordered_map>
#include <vector>

class TG4RootDetectorConstruction;

namespace o2::simsetup
{

/// The VecGeom counterpart of the TGeoParallelWorld: the prioritised placements of the main
/// geometry, each with its navigation state, global transformation and solid, and a BVH over their
/// boxes in the world frame. A point inside one of them is in that placement, whatever the main
/// hierarchy says, which is what keeps misaligned and hence overlapping volumes navigable. The
/// rules are those of TGeo: among several nodes holding a point the one added first wins, and a
/// track inside a node sees no other node until it leaves. Immutable once built; every query is
/// const and keeps no state, so one instance serves all navigators. Lengths are in cm.
class VecGeomParallelWorld
{
 public:
  using V3 = vecgeom::Vector3D<double>;

  /// Reads the nodes of gGeoManager's parallel world. False if there is none or it is empty.
  bool build(TG4RootDetectorConstruction const& dc, VecGeomG4Map const& map);

  /// Takes the nodes as navigation states, in the order of their priority.
  void build(std::vector<vecgeom::NavigationState> states);

  int size() const { return static_cast<int>(mStates.size()); }

  /// The state with node \a i on top.
  vecgeom::NavigationState const& state(int i) const { return mStates[i]; }

  /// Whether \a st is node \a i or inside it.
  bool isInNode(int i, vecgeom::NavigationState const& st) const
  {
    const int level = mLevel[i];
    if (st.IsOutside() || static_cast<int>(st.GetCurrentLevel()) < level) {
      return false;
    }
    for (int l = level - 1; l > 0; --l) {
      if (st.At(l) != mStates[i].At(l)) {
        return false;
      }
    }
    return true;
  }

  /// Whether \a st is in any node.
  bool isInAnyNode(vecgeom::NavigationState const& st) const;

  /// Whether \a p is in the box around all nodes; outside it no node holds the point.
  bool nearNodes(const V3& p) const
  {
    return p[0] >= mMin[0] && p[0] <= mMax[0] && p[1] >= mMin[1] && p[1] <= mMax[1] && p[2] >= mMin[2] &&
           p[2] <= mMax[2];
  }

  /// The node with the smallest index that holds \a p and that \a skip does not refuse, or -1.
  template <typename Skip>
  int findNode(const V3& p, Skip&& skip) const;

  /// The first node a ray enters before \a step, which is lowered to the entry distance; -1 if none.
  /// Nodes \a skip refuses and nodes the point is inside are not seen.
  template <typename Skip>
  int distanceToIn(const V3& p, const V3& dir, double& step, Skip&& skip) const;

  /// The distance to the nearest node if below \a limit, else \a limit. Never negative.
  double safetyToIn(const V3& p, double limit) const;

 private:
  using BVH = vecgeom::IndexedBVH<float>;
  using F3 = vecgeom::Vector3D<float>;

  /// Walks the BVH. \a children says which of two child nodes to visit and which first; \a leaf gets
  /// each node index of a visited leaf and stops the walk by returning true.
  template <typename Children, typename Leaf>
  void traverse(Children&& children, Leaf&& leaf) const;

  std::vector<vecgeom::NavigationState> mStates;
  std::vector<vecgeom::Transformation3D> mToLocal;
  std::vector<vecgeom::VUnplacedVolume const*> mSolid;
  std::vector<int> mLevel;                             ///< levels of the node's state
  std::vector<char> mIsTop;                            ///< by placed-volume id: some node has it on top
  std::unordered_multimap<std::uint64_t, int> mByPath; ///< path hash to node
  std::unique_ptr<BVH> mBVH;
  V3 mMin, mMax; ///< the box around all nodes, padded
  int mMinLevel = 0;
  int mMaxLevel = 0;
};

template <typename Children, typename Leaf>
void VecGeomParallelWorld::traverse(Children&& children, Leaf&& leaf) const
{
  auto const* nodes = mBVH->GetNodes();
  auto const* primId = mBVH->GetPrimId();
  unsigned int stack[BVH::kTraversalStackSize];
  int stackSize = 0;
  unsigned int top = 0;
  for (;;) {
    bool dead = false;
    while (!nodes[top].IsLeaf()) {
      const unsigned int left = nodes[top].first;
      const unsigned int right = left + 1;
      bool goLeft = false, goRight = false, leftFirst = true;
      children(left, right, goLeft, goRight, leftFirst);
      if (goLeft && goRight) {
        stack[stackSize++] = leftFirst ? right : left;
        top = leftFirst ? left : right;
      } else if (goLeft) {
        top = left;
      } else if (goRight) {
        top = right;
      } else {
        dead = true;
        break;
      }
    }
    if (!dead) {
      for (int k = 0; k < nodes[top].count; ++k) {
        if (leaf(primId[nodes[top].first + k])) {
          return;
        }
      }
    }
    if (stackSize == 0) {
      return;
    }
    top = stack[--stackSize];
  }
}

template <typename Skip>
int VecGeomParallelWorld::findNode(const V3& p, Skip&& skip) const
{
  auto const& bvh = *mBVH;
  const F3 pf(static_cast<float>(p[0]), static_cast<float>(p[1]), static_cast<float>(p[2]));
  if (!bvh.NodeContains(0, pf)) {
    return -1;
  }
  // No early exit: nodes overlap and nest, and the smallest index has to win as in TGeo.
  int best = size();
  traverse(
    [&](unsigned int left, unsigned int right, bool& goLeft, bool& goRight, bool&) {
      goLeft = bvh.NodeContains(left, pf);
      goRight = bvh.NodeContains(right, pf);
    },
    [&](int i) {
      if (i < best && bvh.BoxContains(i, pf) && !skip(i) && mSolid[i]->Contains(mToLocal[i].Transform(p))) {
        best = i;
      }
      return false;
    });
  return best < size() ? best : -1;
}

template <typename Skip>
int VecGeomParallelWorld::distanceToIn(const V3& p, const V3& dir, double& step, Skip&& skip) const
{
  auto const& bvh = *mBVH;
  const F3 pf(static_cast<float>(p[0]), static_cast<float>(p[1]), static_cast<float>(p[2]));
  const F3 inv(1.f / vecgeom::NonZero(static_cast<float>(dir[0])), 1.f / vecgeom::NonZero(static_cast<float>(dir[1])),
               1.f / vecgeom::NonZero(static_cast<float>(dir[2])));
  const unsigned int octant = (inv[0] < 0.f ? 1u : 0u) | (inv[1] < 0.f ? 2u : 0u) | (inv[2] < 0.f ? 4u : 0u);
  // The bound in the precision of the boxes, rounded up so that it never cuts off the exact answer.
  auto roundUp = [](double x) {
    return x < 1.e30 ? std::nextafter(static_cast<float>(x), std::numeric_limits<float>::infinity()) : 1.e30f;
  };
  float bound = roundUp(step);
  float approach = 0.f;
  int hit = -1;
  if (!bvh.IntersectNode(0, pf, inv, octant, bound, approach)) {
    return hit;
  }
  traverse(
    [&](unsigned int left, unsigned int right, bool& goLeft, bool& goRight, bool& leftFirst) {
      float approachLeft = 0.f, approachRight = 0.f;
      goLeft = bvh.IntersectNode(left, pf, inv, octant, bound, approachLeft);
      goRight = bvh.IntersectNode(right, pf, inv, octant, bound, approachRight);
      leftFirst = approachLeft <= approachRight;
    },
    [&](int i) {
      if (!bvh.IntersectBox(i, pf, inv, bound, approach) || skip(i)) {
        return false;
      }
      // A negative distance says the point is inside the node, which is then not seen.
      const double d = mSolid[i]->DistanceToIn(mToLocal[i].Transform(p), mToLocal[i].TransformDirection(dir), step);
      if (d >= 0. && d < step) {
        step = d;
        hit = i;
        bound = roundUp(step);
      }
      return false;
    });
  return hit;
}

} // namespace o2::simsetup

#endif
