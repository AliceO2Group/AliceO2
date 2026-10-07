// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
/// \author Sandro Wenzel <sandro.wenzel@cern.ch>
/// \since 2026-08

#include "CADSupport/O2BVHAssembly.h"

#include "TGeoBBox.h"
#include "TGeoManager.h"
#include "TGeoMatrix.h"
#include "TGeoNode.h"
#include "TGeoVolume.h"

// the same third-party BVH2 entry point O2Tessellated and O2BVHSurfaceSolid use
#include "bvh2_third_party.h"
#include "bvh2_extra_kernels.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

using namespace o2::cad;
ClassImp(O2BVHAssembly);

namespace
{
// float BVH types, following the O2Tessellated::BuildBVH pattern
using BVHScalar = float;
using BVHBBox = bvh::v2::BBox<BVHScalar, 3>;
using BVHVec3 = bvh::v2::Vec<BVHScalar, 3>;
using BVHNode = bvh::v2::Node<BVHScalar, 3>;
using BVH = bvh::v2::Bvh<BVHNode>;
using BVHRay = bvh::v2::Ray<BVHScalar, 3>;

/// Widening of every daughter box before the outward float rounding; the value O2BVHSurfaceSolid uses.
constexpr double kBoxTolerance = 1.e-3;

/// Round a double outward into float, away from the interval the box encloses.
inline float roundOutward(double value, bool up)
{
  return std::nextafterf(static_cast<float>(value),
                         up ? std::numeric_limits<float>::infinity() : -std::numeric_limits<float>::infinity());
}

/// A float ray bound that is never *shorter* than the double distance it stands for.
inline float truncateRoundUp(double value)
{
  const float rounded = static_cast<float>(value);
  return rounded < value ? std::nextafterf(rounded, std::numeric_limits<float>::infinity()) : rounded;
}

/// Squared distance from \a point to a node box, in double and shrunk by a relative guard; scale it by kSafetyBoundShare.
inline double boxDistanceSq(const BVHBBox& box, const double* point)
{
  double distanceSq = 0.;
  for (int dimension = 0; dimension < 3; ++dimension) {
    const double lower = static_cast<double>(box.min[dimension]);
    const double upper = static_cast<double>(box.max[dimension]);
    const double coordinate = point[dimension];
    if (coordinate < lower) {
      const double gap = lower - coordinate;
      distanceSq += gap * gap;
    } else if (coordinate > upper) {
      const double gap = coordinate - upper;
      distanceSq += gap * gap;
    }
  }
  return distanceSq * (1. - 1.e-12);
}

/// Share of a node's squared box distance that bounds a box daughter's axis-max Safety from below: max d_i >= |d| / sqrt(3).
/// A sharp daughter (a thin tube segment, an Arb8) can answer less, so the result stays a sound safety but may differ from Safety_Loop.
constexpr double kSafetyBoundShare = 1. / 3.;

inline bool boxContains(const BVHBBox& box, const double* point)
{
  return point[0] >= static_cast<double>(box.min[0]) && point[0] <= static_cast<double>(box.max[0]) &&
         point[1] >= static_cast<double>(box.min[1]) && point[1] <= static_cast<double>(box.max[1]) &&
         point[2] >= static_cast<double>(box.min[2]) && point[2] <= static_cast<double>(box.max[2]);
}

/// One entry of the nearest-daughter traversal stack.
struct SafetyEntry {
  double distanceSq;
  size_t node;
};

/// Capacity of the call-stack traversal stack; a local one, because assembly queries nest.
constexpr unsigned kSmallStackCapacity = 64;

/// Run \a traverse with a fixed-size stack when a tree of \a treeDepth levels fits it, else a growing one.
template <typename T, typename Traverse>
auto withTraversalStack(int treeDepth, Traverse&& traverse)
{
  if (treeDepth + 2 <= static_cast<int>(kSmallStackCapacity)) {
    bvh::v2::SmallStack<T, kSmallStackCapacity> stack;
    return traverse(stack);
  }
  bvh::v2::GrowingStack<T> stack;
  return traverse(stack);
}
} // namespace

O2BVHAssembly::O2BVHAssembly() : TGeoShapeAssembly() {}

O2BVHAssembly::O2BVHAssembly(TGeoVolumeAssembly* volume) : TGeoShapeAssembly(volume)
{
  if (volume != nullptr) {
    BuildBVH();
  }
}

O2BVHAssembly::~O2BVHAssembly()
{
  delete static_cast<BVH*>(fBVH);
  fBVH = nullptr;
}

size_t O2BVHAssembly::GetBVHMemory() const
{
  const auto* bvh = static_cast<const BVH*>(fBVH);
  if (bvh == nullptr) {
    return 0;
  }
  return bvh->nodes.size() * sizeof(BVHNode) + bvh->prim_ids.size() * sizeof(size_t);
}

////////////////////////////////////////////////////////////////////////////////
/// BuildBVH -- one primitive per daughter: its box in the assembly frame, widened and rounded outward in float.

void O2BVHAssembly::BuildBVH()
{
  delete static_cast<BVH*>(fBVH);
  fBVH = nullptr;
  fNbuilt = -1;
  fTreeDepth = 0;
  if (fVolume == nullptr) {
    return;
  }
  ComputeBBox();
  const int nDaughters = fVolume->GetNdaughters();
  fNbuilt = nDaughters;
  if (nDaughters == 0) {
    return;
  }

  std::vector<BVHBBox> boxes;
  std::vector<BVHVec3> centers;
  boxes.reserve(nDaughters);
  centers.reserve(nDaughters);

  double corners[24];
  double master[3];
  for (int index = 0; index < nDaughters; ++index) {
    TGeoNode* node = fVolume->GetNode(index);
    TGeoShape* shape = node->GetVolume()->GetShape();
    // an assembly daughter, or one whose box was never computed, has to produce it first --
    // the same guard TGeoShapeAssembly::RecomputeBoxLast uses
    if (node->GetVolume()->IsAssembly() || TGeoShape::IsSameWithinTolerance(((TGeoBBox*)shape)->GetDX(), 0.)) {
      shape->ComputeBBox();
    }
    ((TGeoBBox*)shape)->SetBoxPoints(corners);
    double lower[3] = {TGeoShape::Big(), TGeoShape::Big(), TGeoShape::Big()};
    double upper[3] = {-TGeoShape::Big(), -TGeoShape::Big(), -TGeoShape::Big()};
    for (int corner = 0; corner < 8; ++corner) {
      node->LocalToMaster(&corners[3 * corner], master);
      for (int dimension = 0; dimension < 3; ++dimension) {
        lower[dimension] = std::min(lower[dimension], master[dimension]);
        upper[dimension] = std::max(upper[dimension], master[dimension]);
      }
    }
    BVHBBox box;
    for (int dimension = 0; dimension < 3; ++dimension) {
      box.min[dimension] = roundOutward(lower[dimension] - kBoxTolerance, false);
      box.max[dimension] = roundOutward(upper[dimension] + kBoxTolerance, true);
    }
    boxes.push_back(box);
    centers.emplace_back(box.get_center());
  }

  typename bvh::v2::DefaultBuilder<BVHNode>::Config config;
  config.quality = bvh::v2::DefaultBuilder<BVHNode>::Quality::High;
  // One daughter per leaf: bvh2 enters a leaf without a box test, and a daughter query costs far more than one.
  config.max_leaf_size = 1;
  auto* built = new BVH(bvh::v2::DefaultBuilder<BVHNode>::build(boxes, centers, config));
  fBVH = static_cast<void*>(built);

  // tree depth, which decides whether a traversal fits the fixed-size stack
  std::vector<std::pair<size_t, int>> pending{{0, 1}};
  while (!pending.empty()) {
    const auto [index, level] = pending.back();
    pending.pop_back();
    fTreeDepth = std::max(fTreeDepth, level);
    const auto& node = built->nodes[index];
    if (!node.is_leaf()) {
      const size_t firstChild = node.index.first_id();
      for (size_t child : {firstChild, firstChild + 1}) {
        if (child < built->nodes.size()) {
          pending.push_back({child, level + 1});
        }
      }
    }
  }
}

void O2BVHAssembly::EnsureBuilt() const
{
  const int nDaughters = fVolume != nullptr ? fVolume->GetNdaughters() : 0;
  if (fNbuilt == nDaughters && (fBVH != nullptr || nDaughters == 0)) {
    return;
  }
  const_cast<O2BVHAssembly*>(this)->BuildBVH();
}

////////////////////////////////////////////////////////////////////////////////
/// Contains

Bool_t O2BVHAssembly::Contains(const Double_t* point) const
{
  EnsureBuilt();
  if (!fBBoxOK) {
    const_cast<O2BVHAssembly*>(this)->ComputeBBox();
  }
  if (!TGeoBBox::Contains(point)) {
    return kFALSE;
  }
  const auto* bvh = static_cast<const BVH*>(fBVH);
  if (bvh == nullptr) {
    return kFALSE;
  }

  int best = -1;
  withTraversalStack<size_t>(fTreeDepth, [&](auto& stack) {
    double local[3];
    stack.push(0); // the bvh2 root node
    while (!stack.is_empty()) {
      const auto& node = bvh->nodes[stack.pop()];
      if (!boxContains(node.get_bbox(), point)) {
        continue;
      }
      if (node.is_leaf()) {
        const auto beginPrimitive = node.index.first_id();
        const auto endPrimitive = beginPrimitive + node.index.prim_count();
        for (auto primitive = beginPrimitive; primitive < endPrimitive; ++primitive) {
          const int daughter = static_cast<int>(bvh->prim_ids[primitive]);
          // the loop twin takes the lowest-indexed daughter that contains the point, so a candidate
          // that cannot beat the standing answer need not be resolved at all
          if (best >= 0 && daughter > best) {
            continue;
          }
          TGeoNode* geoNode = fVolume->GetNode(daughter);
          geoNode->MasterToLocal(point, local);
          if (geoNode->GetVolume()->GetShape()->Contains(local)) {
            best = daughter;
          }
        }
      } else {
        const auto firstChild = node.index.first_id();
        for (size_t child : {firstChild, firstChild + 1}) {
          if (child < bvh->nodes.size()) {
            stack.push(child);
          }
        }
      }
    }
  });

  if (best < 0) {
    return kFALSE;
  }
  // this is how the daughter identity reaches TGeoNavigator, and through it the hit
  fVolume->SetCurrentNodeIndex(best);
  fVolume->SetNextNodeIndex(best);
  return kTRUE;
}

Bool_t O2BVHAssembly::Contains_Loop(const Double_t* point) const
{
  if (!fBBoxOK) {
    const_cast<O2BVHAssembly*>(this)->ComputeBBox();
  }
  if (!TGeoBBox::Contains(point)) {
    return kFALSE;
  }
  double local[3];
  const int nDaughters = fVolume != nullptr ? fVolume->GetNdaughters() : 0;
  for (int index = 0; index < nDaughters; ++index) {
    TGeoNode* geoNode = fVolume->GetNode(index);
    geoNode->MasterToLocal(point, local);
    if (geoNode->GetVolume()->GetShape()->Contains(local)) {
      fVolume->SetCurrentNodeIndex(index);
      fVolume->SetNextNodeIndex(index);
      return kTRUE;
    }
  }
  return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// DistFromOutside -- daughters are queried with the fixed query bound, so the answer is visit-order independent.

Double_t O2BVHAssembly::DistFromOutside(const Double_t* point, const Double_t* dir, Int_t iact, Double_t step,
                                        Double_t* safe) const
{
  EnsureBuilt();
  if (!fBBoxOK) {
    const_cast<O2BVHAssembly*>(this)->ComputeBBox();
  }
  if (iact < 3 && safe != nullptr) {
    *safe = Safety(point, kFALSE);
    if (iact == 0) {
      return TGeoShape::Big();
    }
    if (iact == 1 && step <= *safe) {
      return TGeoShape::Big();
    }
  }
  const auto* bvh = static_cast<const BVH*>(fBVH);
  if (bvh == nullptr) {
    return TGeoShape::Big();
  }

  double best = TGeoShape::Big();
  int bestIndex = -1;
  BVHRay ray(BVHVec3(static_cast<float>(point[0]), static_cast<float>(point[1]), static_cast<float>(point[2])),
             BVHVec3(static_cast<float>(dir[0]), static_cast<float>(dir[1]), static_cast<float>(dir[2])), 0.f,
             truncateRoundUp(step + kBoxTolerance));
  static constexpr bool useRobustTraversal = true;
  auto* volume = fVolume;
  withTraversalStack<BVH::Index>(fTreeDepth, [&](auto& stack) {
    bvh->intersect<false, useRobustTraversal>(
      ray, bvh->get_root().index, stack, [&](size_t beginPrimitive, size_t endPrimitive) {
        double local[3];
        double localDir[3];
        for (size_t primitive = beginPrimitive; primitive < endPrimitive; ++primitive) {
          const int daughter = static_cast<int>(bvh->prim_ids[primitive]);
          TGeoNode* geoNode = volume->GetNode(daughter);
          geoNode->MasterToLocal(point, local);
          geoNode->MasterToLocalVect(dir, localDir);
          const double distance = geoNode->GetVolume()->GetShape()->DistFromOutside(local, localDir, 3, step);
          if (distance < best) {
            best = distance;
            bestIndex = daughter;
          } else if (distance == best && daughter < bestIndex) {
            bestIndex = daughter;
          }
        }
        // A daughter whose box the ray only meets beyond best + kBoxTolerance cannot cross nearer,
        // and cannot tie either: its true crossing is at least its box entry distance.
        if (bestIndex >= 0) {
          ray.tmax = std::min(ray.tmax, truncateRoundUp(best + kBoxTolerance));
        }
        return false; // keep traversing
      });
  });

  if (bestIndex < 0 || best >= step) {
    return TGeoShape::Big();
  }
  volume->SetNextNodeIndex(bestIndex);
  return best;
}

Double_t O2BVHAssembly::DistFromOutside_Loop(const Double_t* point, const Double_t* dir, Double_t step) const
{
  if (!fBBoxOK) {
    const_cast<O2BVHAssembly*>(this)->ComputeBBox();
  }
  double best = TGeoShape::Big();
  int bestIndex = -1;
  double local[3];
  double localDir[3];
  const int nDaughters = fVolume != nullptr ? fVolume->GetNdaughters() : 0;
  for (int index = 0; index < nDaughters; ++index) {
    TGeoNode* geoNode = fVolume->GetNode(index);
    geoNode->MasterToLocal(point, local);
    geoNode->MasterToLocalVect(dir, localDir);
    const double distance = geoNode->GetVolume()->GetShape()->DistFromOutside(local, localDir, 3, step);
    if (distance < best) {
      best = distance;
      bestIndex = index;
    }
  }
  if (bestIndex < 0 || best >= step) {
    return TGeoShape::Big();
  }
  fVolume->SetNextNodeIndex(bestIndex);
  return best;
}

////////////////////////////////////////////////////////////////////////////////
/// Safety -- from inside as ROOT; from outside a nearest-daughter descent of the BVH.

Double_t O2BVHAssembly::Safety(const Double_t* point, Bool_t in) const
{
  if (in) {
    return TGeoShapeAssembly::Safety(point, in);
  }
  EnsureBuilt();
  if (!fBBoxOK) {
    const_cast<O2BVHAssembly*>(this)->ComputeBBox();
  }
  const auto* bvh = static_cast<const BVH*>(fBVH);
  if (bvh == nullptr) {
    return TGeoShape::Big();
  }

  return withTraversalStack<SafetyEntry>(fTreeDepth, [&](auto& stack) {
    double best = TGeoShape::Big();
    stack.push({boxDistanceSq(bvh->nodes[0].get_bbox(), point), size_t(0)});
    while (!stack.is_empty()) {
      const SafetyEntry entry = stack.pop();
      if (entry.distanceSq * kSafetyBoundShare >= best * best) {
        continue;
      }
      const auto& node = bvh->nodes[entry.node];
      if (node.is_leaf()) {
        const auto beginPrimitive = node.index.first_id();
        const auto endPrimitive = beginPrimitive + node.index.prim_count();
        for (auto primitive = beginPrimitive; primitive < endPrimitive; ++primitive) {
          const int daughter = static_cast<int>(bvh->prim_ids[primitive]);
          const double safety = fVolume->GetNode(daughter)->Safety(point, kFALSE);
          if (safety <= 0.) {
            return 0.;
          }
          best = std::min(best, safety);
        }
      } else {
        const auto firstChild = node.index.first_id();
        const size_t children[2] = {firstChild, firstChild + 1};
        double distancesSq[2] = {TGeoShape::Big(), TGeoShape::Big()};
        for (int side = 0; side < 2; ++side) {
          if (children[side] < bvh->nodes.size()) {
            distancesSq[side] = boxDistanceSq(bvh->nodes[children[side]].get_bbox(), point);
          }
        }
        // push the farther child first so the nearer one is popped, and prunes, first
        const bool leftIsFarther = distancesSq[0] >= distancesSq[1];
        const int order[2] = {leftIsFarther ? 0 : 1, leftIsFarther ? 1 : 0};
        for (int side = 0; side < 2; ++side) {
          const int which = order[side];
          if (children[which] < bvh->nodes.size() && distancesSq[which] * kSafetyBoundShare < best * best) {
            stack.push({distancesSq[which], children[which]});
          }
        }
      }
    }
    return best;
  });
}

Double_t O2BVHAssembly::Safety_Loop(const Double_t* point, Bool_t in) const
{
  if (in) {
    return TGeoShapeAssembly::Safety(point, in);
  }
  double best = TGeoShape::Big();
  const int nDaughters = fVolume != nullptr ? fVolume->GetNdaughters() : 0;
  for (int index = 0; index < nDaughters; ++index) {
    const double safety = fVolume->GetNode(index)->Safety(point, kFALSE);
    if (safety <= 0.) {
      return 0.;
    }
    best = std::min(best, safety);
  }
  return best;
}

////////////////////////////////////////////////////////////////////////////////
/// MakeBVHAssembly

O2BVHAssembly* O2BVHAssembly::MakeBVHAssembly(TGeoVolumeAssembly* volume)
{
  if (volume == nullptr) {
    return nullptr;
  }
  auto* shape = new O2BVHAssembly(volume);
  volume->SetShape(shape);
  return shape;
}
