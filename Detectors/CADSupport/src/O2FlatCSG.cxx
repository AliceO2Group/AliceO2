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

#include "CADSupport/O2FlatCSG.h"

#include "BoundedSurface.h"

// the same third-party BVH2 entry point O2Tessellated, O2BVHSurfaceSolid and O2BVHAssembly use
#include "bvh2_third_party.h"
#include "bvh2_extra_kernels.h"

#include "TGeoShape.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <vector>

ClassImp(o2::cad::O2FlatCSG);

namespace o2
{
namespace cad
{

namespace
{
/// The most roots one cell can contribute to one ray: four per torus halfspace.
constexpr int kMaxRootsPerHalfspace = 4;

/// Per-path cap on SplitBox's aspect-ratio-equalising splits; bounds recursion on pathological cells.
constexpr int kMaxCubifySplits = 10;

/// An upper bound on the `[enter, exit]` pairs one cell produces along a ray.
int maxPairsForCell(int halfspaceCount)
{
  return 2 + kMaxRootsPerHalfspace * halfspaceCount;
}

// float BVH types: the BVH only nominates boxes, and roundOutward makes each node box a superset of its boxes.
using BVHScalar = float;
using BVHBBox = bvh::v2::BBox<BVHScalar, 3>;
using BVHVec3 = bvh::v2::Vec<BVHScalar, 3>;
using BVHNode = bvh::v2::Node<BVHScalar, 3>;
using BVH = bvh::v2::Bvh<BVHNode>;

/// Per-thread count of DistFromInside queries redone without pruning.
thread_local long long gUnprunedRetryCount = 0;

/// Round a double outward into float, away from the interval the box encloses.
inline float roundOutward(double value, bool up)
{
  return std::nextafterf(static_cast<float>(value), up ? std::numeric_limits<float>::infinity()
                                                       : -std::numeric_limits<float>::infinity());
}

/// Clip [tlo, thi] to the box's slab; false when nothing survives.
/// Divides by dir (no reciprocal) so a box face on the cell's own plane gives HalfspaceRoots' t exactly.
bool slabWindow(const double* boxMin, const double* boxMax, const double* origin, const double* dir,
                double& tlo, double& thi)
{
  for (int index = 0; index < 3; ++index) {
    if (std::abs(dir[index]) < 1.e-300) {
      // parallel to this pair of faces: the ray is either inside the slab for every t or outside
      // it for every t
      if (origin[index] < boxMin[index] || origin[index] > boxMax[index]) {
        return false;
      }
      continue;
    }
    double low = (boxMin[index] - origin[index]) / dir[index];
    double high = (boxMax[index] - origin[index]) / dir[index];
    if (low > high) {
      std::swap(low, high);
    }
    tlo = std::max(tlo, low);
    thi = std::min(thi, high);
    if (tlo > thi) {
      return false;
    }
  }
  return true;
}

/// The same clip against a BVH node's (float, outward-rounded) box.
inline bool nodeWindow(const BVHBBox& box, const double* origin, const double* dir, double& tlo,
                       double& thi)
{
  const double lo[3] = {box.min[0], box.min[1], box.min[2]};
  const double hi[3] = {box.max[0], box.max[1], box.max[2]};
  return slabWindow(lo, hi, origin, dir, tlo, thi);
}

/// Whether \a point is in the box's own double bounds, closed on every face.
inline bool boxHoldsPoint(const FlatCSGBox& box, const double* point)
{
  return point[0] >= box.min[0] && point[0] <= box.max[0] && point[1] >= box.min[1] &&
         point[1] <= box.max[1] && point[2] >= box.min[2] && point[2] <= box.max[2];
}

/// Unnormalised gradient of `sign * f` at \a point: `2(Ax + b)` for a quadric, the gradient of the signed distance for a torus.
void halfspaceGradient(const FlatCSGHalfspace& halfspace, const double* point, double grad[3])
{
  if (halfspace.kind == FlatCSGHalfspace::kTorus) {
    const double* c = halfspace.c;
    const double axis[3] = {c[3], c[4], c[5]};
    const double major = c[6];
    const double offset[3] = {point[0] - c[0], point[1] - c[1], point[2] - c[2]};
    const double along = offset[0] * axis[0] + offset[1] * axis[1] + offset[2] * axis[2];
    double radial[3];
    for (int index = 0; index < 3; ++index) {
      radial[index] = offset[index] - along * axis[index];
    }
    const double rho = std::sqrt(radial[0] * radial[0] + radial[1] * radial[1] + radial[2] * radial[2]);
    const double u = rho - major;
    const double s = std::hypot(u, along);
    if (s < 1.e-300 || rho < 1.e-300) {
      // degenerate: on the revolution axis or the kissing point; leave it zero for the caller's fallback
      grad[0] = grad[1] = grad[2] = 0.;
      return;
    }
    const double du = u / s;
    const double dv = along / s;
    for (int index = 0; index < 3; ++index) {
      grad[index] = halfspace.sign * (du * (radial[index] / rho) + dv * axis[index]);
    }
    return;
  }
  const double* c = halfspace.c;
  const double a[3][3] = {{c[0], c[1], c[2]}, {c[1], c[3], c[4]}, {c[2], c[4], c[5]}};
  const double b[3] = {c[6], c[7], c[8]};
  for (int row = 0; row < 3; ++row) {
    double value = b[row];
    for (int column = 0; column < 3; ++column) {
      value += a[row][column] * point[column];
    }
    grad[row] = halfspace.sign * 2. * value;
  }
}

/// Hand every leaf box whose node box the ray meets within `[0, cap]` to \a visit.
/// \a tmax is re-read at every node test, so the visitor may lower it; with \a nearFirst the nearer child
/// is visited first, and a non-null \a culled collects the nearest entry the lowered bound skipped.
template <typename Visit>
void traverseRay(const BVH& bvh, const double* origin, const double* dir, double cap, const double& tmax,
                 bool nearFirst, double* culled, Visit&& visit)
{
  struct Entry {
    size_t node;
    double tlo; ///< where the ray enters the node box
  };
  // thread_local rather than a member or a fresh vector per call: TGeo shares one shape object
  // across every navigator under TGeoManager::SetMaxThreads, and this is not re-entered
  thread_local std::vector<Entry> stack;
  stack.clear();
  const auto entersWithin = [&](size_t index, double& tlo) {
    tlo = 0.;
    double thi = cap;
    return nodeWindow(bvh.nodes[index].get_bbox(), origin, dir, tlo, thi);
  };
  // a skipped node's own entry is a lower bound on every piece under it
  const auto skip = [&](double tlo) {
    if (culled != nullptr && tlo < *culled) {
      *culled = tlo;
    }
  };
  double rootTlo = 0.;
  if (entersWithin(0, rootTlo)) {
    stack.push_back({0, rootTlo}); // the bvh2 root node
  }
  while (!stack.empty()) {
    const Entry entry = stack.back();
    stack.pop_back();
    if (entry.tlo > tmax) {
      skip(entry.tlo); // the visitor lowered tmax past this node
      continue;
    }
    const auto& node = bvh.nodes[entry.node];
    if (node.is_leaf()) {
      const auto beginPrimitive = node.index.first_id();
      const auto endPrimitive = beginPrimitive + node.index.prim_count();
      for (auto primitive = beginPrimitive; primitive < endPrimitive; ++primitive) {
        visit(static_cast<int>(bvh.prim_ids[primitive]));
      }
    } else {
      const auto firstChild = node.index.first_id();
      Entry children[2];
      int count = 0;
      for (size_t child : {firstChild, firstChild + 1}) {
        double tlo = 0.;
        if (child < bvh.nodes.size() && entersWithin(child, tlo)) {
          if (tlo > tmax) {
            skip(tlo);
          } else {
            children[count++] = {child, tlo};
          }
        }
      }
      // LIFO: the farther child is pushed first
      if (nearFirst && count == 2 && children[0].tlo < children[1].tlo) {
        std::swap(children[0], children[1]);
      }
      for (int index = 0; index < count; ++index) {
        stack.push_back(children[index]);
      }
    }
  }
}
/// Hand every leaf box whose node box holds \a point to \a visit, in traversal order, until \a visit
/// returns true; returns whether it did.
template <typename Visit>
bool traversePoint(const BVH& bvh, const double* point, Visit&& visit)
{
  const BVHVec3 query(static_cast<float>(point[0]), static_cast<float>(point[1]),
                      static_cast<float>(point[2]));
  thread_local std::vector<size_t> stack;
  stack.clear();
  stack.push_back(0); // the bvh2 root node
  while (!stack.empty()) {
    const size_t current = stack.back();
    stack.pop_back();
    const auto& node = bvh.nodes[current];
    if (!bvh::v2::extra::contains(node.get_bbox(), query)) {
      continue;
    }
    if (node.is_leaf()) {
      const auto beginPrimitive = node.index.first_id();
      const auto endPrimitive = beginPrimitive + node.index.prim_count();
      for (auto primitive = beginPrimitive; primitive < endPrimitive; ++primitive) {
        if (visit(static_cast<int>(bvh.prim_ids[primitive]))) {
          return true;
        }
      }
    } else {
      const auto firstChild = node.index.first_id();
      for (size_t child : {firstChild, firstChild + 1}) {
        if (child < bvh.nodes.size()) {
          stack.push_back(child);
        }
      }
    }
  }
  return false;
}

/// Squared distance from \a point to the box's own double bounds; 0 inside.
inline double boxDistanceSquared(const FlatCSGBox& box, const double* point)
{
  double squared = 0.;
  for (int index = 0; index < 3; ++index) {
    const double value = point[index];
    if (value < box.min[index]) {
      squared += (box.min[index] - value) * (box.min[index] - value);
    } else if (value > box.max[index]) {
      squared += (value - box.max[index]) * (value - box.max[index]);
    }
  }
  return squared;
}

/// Distance from \a point, inside the box, to the box's nearest face.
inline double distanceToFaces(const FlatCSGBox& box, const double* point)
{
  double toFace = TGeoShape::Big();
  for (int index = 0; index < 3; ++index) {
    toFace = std::min(toFace, std::min(point[index] - box.min[index], box.max[index] - point[index]));
  }
  return toFace;
}
} // namespace

O2FlatCSG::O2FlatCSG() : TGeoBBox(0., 0., 0.) {}

O2FlatCSG::O2FlatCSG(const char* name) : TGeoBBox(name, 0., 0., 0.) {}

O2FlatCSG::~O2FlatCSG()
{
  delete static_cast<BVH*>(fBVH);
  fBVH = nullptr;
}

size_t O2FlatCSG::GetBVHMemory() const
{
  const auto* bvh = static_cast<const BVH*>(fBVH);
  if (bvh == nullptr) {
    return 0;
  }
  return bvh->nodes.size() * sizeof(BVHNode) + bvh->prim_ids.size() * sizeof(size_t);
}

int O2FlatCSG::AddQuadric(double sign, const double coeff[10])
{
  FlatCSGHalfspace halfspace;
  halfspace.kind = FlatCSGHalfspace::kQuadric;
  halfspace.sign = sign < 0. ? -1. : 1.;
  for (int index = 0; index < 10; ++index) {
    halfspace.c[index] = coeff[index];
  }
  fHalfspaces.push_back(halfspace);
  return static_cast<int>(fHalfspaces.size()) - 1;
}

int O2FlatCSG::AddTorus(double sign, const double* centre, const double* axis, double major,
                        double minor)
{
  FlatCSGHalfspace halfspace;
  halfspace.kind = FlatCSGHalfspace::kTorus;
  halfspace.sign = sign < 0. ? -1. : 1.;
  // normalise the axis once here; a zero axis is a caller bug and asserts
  const double axisNorm = std::sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]);
  assert(axisNorm > 0. && "O2FlatCSG::AddTorus: axis must not be the zero vector");
  for (int index = 0; index < 3; ++index) {
    halfspace.c[index] = centre[index];
    halfspace.c[3 + index] = axis[index] / axisNorm;
  }
  halfspace.c[6] = major;
  halfspace.c[7] = minor;
  fHalfspaces.push_back(halfspace);
  return static_cast<int>(fHalfspaces.size()) - 1;
}

int O2FlatCSG::AddCell(int first, int count, double volume)
{
  FlatCSGCell cell;
  cell.first = first;
  cell.count = count;
  cell.volume = volume;
  fCells.push_back(cell);
  return static_cast<int>(fCells.size()) - 1;
}

void O2FlatCSG::EnsureCellBBoxStorage()
{
  if (static_cast<int>(fCellBBoxSet.size()) < GetNcells()) {
    fCellLo.resize(3 * GetNcells(), 0.);
    fCellHi.resize(3 * GetNcells(), 0.);
    fCellBBoxSet.resize(GetNcells(), false);
  }
}

void O2FlatCSG::SetCellBBox(int cell, const double* lo, const double* hi)
{
  if (cell < 0 || cell >= GetNcells()) {
    // a cell index before its AddCell would write past the end of fCellLo/fCellHi
    Error("SetCellBBox", "Shape %s: cell %d is out of range (%d cell(s) so far); ignoring",
          GetName(), cell, GetNcells());
    return;
  }
  EnsureCellBBoxStorage();
  for (int index = 0; index < 3; ++index) {
    fCellLo[3 * cell + index] = lo[index];
    fCellHi[3 * cell + index] = hi[index];
  }
  fCellBBoxSet[cell] = true;
}

void O2FlatCSG::GetCellBBox(int cell, double* lo, double* hi) const
{
  const bool set = cell >= 0 && cell < GetNcells() && static_cast<size_t>(cell) < fCellBBoxSet.size() &&
                   fCellBBoxSet[cell];
  for (int index = 0; index < 3; ++index) {
    lo[index] = set ? fCellLo[3 * cell + index] : 0.;
    hi[index] = set ? fCellHi[3 * cell + index] : 0.;
  }
}

double O2FlatCSG::EvalHalfspace(const FlatCSGHalfspace& halfspace, const double* point)
{
  if (halfspace.kind == FlatCSGHalfspace::kTorus) {
    const double* c = halfspace.c;
    const double offset[3] = {point[0] - c[0], point[1] - c[1], point[2] - c[2]};
    const double along = offset[0] * c[3] + offset[1] * c[4] + offset[2] * c[5];
    const double radial[3] = {offset[0] - along * c[3], offset[1] - along * c[4],
                              offset[2] - along * c[5]};
    const double rho = std::sqrt(radial[0] * radial[0] + radial[1] * radial[1] +
                                 radial[2] * radial[2]);
    // the exact signed distance, which is 1-Lipschitz
    return halfspace.sign * (std::hypot(rho - c[6], along) - c[7]);
  }
  const double* c = halfspace.c;
  const double x = point[0];
  const double y = point[1];
  const double z = point[2];
  const double quadratic = c[0] * x * x + c[3] * y * y + c[5] * z * z +
                           2. * (c[1] * x * y + c[2] * x * z + c[4] * y * z);
  const double linear = 2. * (c[6] * x + c[7] * y + c[8] * z);
  return halfspace.sign * (quadratic + linear + c[9]);
}

void O2FlatCSG::HalfspaceRange(const FlatCSGHalfspace& halfspace, const double* lo,
                               const double* hi, double& rangeLo, double& rangeHi)
{
  // preconditions (see the header): non-negative half-extents and finite bounds
  assert(std::isfinite(lo[0]) && std::isfinite(lo[1]) && std::isfinite(lo[2]) &&
         std::isfinite(hi[0]) && std::isfinite(hi[1]) && std::isfinite(hi[2]) &&
         lo[0] <= hi[0] && lo[1] <= hi[1] && lo[2] <= hi[2] &&
         "O2FlatCSG::HalfspaceRange: lo/hi must be finite and lo[i] <= hi[i] on every axis");

  double centre[3];
  double half[3];
  for (int index = 0; index < 3; ++index) {
    centre[index] = 0.5 * (lo[index] + hi[index]);
    half[index] = 0.5 * (hi[index] - lo[index]);
  }
  const double middle = EvalHalfspace(halfspace, centre);

  // Pad by 64 eps times the summed term magnitudes, not |middle|, which cancels on a straddling box.
  constexpr double kPadFactor = 64. * std::numeric_limits<double>::epsilon();

  double halfWidth;
  double mag;
  if (halfspace.kind == FlatCSGHalfspace::kTorus) {
    // the torus's signed distance is 1-Lipschitz, so over the box it deviates by at most |h|
    halfWidth = std::sqrt(half[0] * half[0] + half[1] * half[1] + half[2] * half[2]);
    const double* c = halfspace.c;
    const double offset[3] = {centre[0] - c[0], centre[1] - c[1], centre[2] - c[2]};
    const double along = offset[0] * c[3] + offset[1] * c[4] + offset[2] * c[5];
    const double radial[3] = {offset[0] - along * c[3], offset[1] - along * c[4],
                              offset[2] - along * c[5]};
    const double rho = std::sqrt(radial[0] * radial[0] + radial[1] * radial[1] +
                                 radial[2] * radial[2]);
    // mag needs no term for the centre's scale: near the core circle offset is exact by Sterbenz's lemma
    mag = rho + std::abs(c[6]) + std::abs(along) + std::abs(c[7]);
  } else {
    const double* c = halfspace.c;
    const double a[3][3] = {{c[0], c[1], c[2]}, {c[1], c[3], c[4]}, {c[2], c[4], c[5]}};
    const double b[3] = {c[6], c[7], c[8]};
    double slack = 0.;
    mag = std::abs(c[9]);
    for (int row = 0; row < 3; ++row) {
      double gradient = b[row];
      mag += 2. * std::abs(b[row] * centre[row]);
      for (int column = 0; column < 3; ++column) {
        gradient += a[row][column] * centre[column];
        // sum |A_ij| h_i h_j over-estimates the cross-term deviation only for non-negative half-extents
        slack += std::abs(a[row][column]) * half[row] * half[column];
        mag += std::abs(a[row][column] * centre[row] * centre[column]);
      }
      slack += 2. * std::abs(gradient) * half[row];
    }
    // |sign| == 1, so the unsigned slack bounds the signed deviation too
    halfWidth = slack;
  }
  // widen by the pad: the drop tests treat the bound as exact and nActive == 0 is trusted
  halfWidth += kPadFactor * mag;
  rangeLo = middle - halfWidth;
  rangeHi = middle + halfWidth;
}

bool O2FlatCSG::CellContains(int index, const double* point) const
{
  const FlatCSGCell& cell = fCells[index];
  for (int offset = 0; offset < cell.count; ++offset) {
    if (EvalHalfspace(fHalfspaces[cell.first + offset], point) > 0.) {
      return false;
    }
  }
  return true;
}

void O2FlatCSG::SplitBox(int cell, const double* lo, const double* hi,
                         const std::vector<int>& active, int depth, double minSize,
                         int cubifyBudget)
{
  std::vector<int> stillActive;
  stillActive.reserve(active.size());
  for (int halfspace : active) {
    double rangeLo = 0.;
    double rangeHi = 0.;
    HalfspaceRange(fHalfspaces[halfspace], lo, hi, rangeLo, rangeHi);
    if (rangeLo > 0.) {
      return; // the box is wholly outside this halfspace, hence wholly outside the cell
    }
    if (rangeHi > 0.) {
      stillActive.push_back(halfspace); // undecided; it stays
    }
    // rangeHi <= 0: the halfspace holds everywhere in the box, so it is dropped
  }

  double longest = 0.;
  double shortest = TGeoShape::Big();
  int axis = 0;
  for (int index = 0; index < 3; ++index) {
    const double extent = hi[index] - lo[index];
    if (extent > longest) {
      longest = extent;
      axis = index;
    }
    shortest = std::min(shortest, extent);
  }
  // a split out of a far-from-cubic box draws on cubifyBudget, not on depth
  // `shortest` is floored at minSize so a flat cell does not burn the whole cubifyBudget
  const bool farFromCubic = longest > 2. * std::max(shortest, minSize);
  const bool keep = stillActive.empty() || depth <= 0 || longest <= minSize ||
                    (farFromCubic && cubifyBudget <= 0);
  if (keep) {
    FlatCSGBox box;
    for (int index = 0; index < 3; ++index) {
      box.min[index] = lo[index];
      box.max[index] = hi[index];
    }
    box.cell = cell;
    box.firstActive = static_cast<int>(fActive.size());
    box.nActive = static_cast<int>(stillActive.size());
    fActive.insert(fActive.end(), stillActive.begin(), stillActive.end());
    fBoxes.push_back(box);
    return;
  }

  const int childDepth = farFromCubic ? depth : depth - 1;
  const int childCubifyBudget = farFromCubic ? cubifyBudget - 1 : cubifyBudget;
  const double middle = 0.5 * (lo[axis] + hi[axis]);
  double childLo[3] = {lo[0], lo[1], lo[2]};
  double childHi[3] = {hi[0], hi[1], hi[2]};
  childHi[axis] = middle;
  SplitBox(cell, childLo, childHi, stillActive, childDepth, minSize, childCubifyBudget);
  childHi[axis] = hi[axis];
  childLo[axis] = middle;
  SplitBox(cell, childLo, childHi, stillActive, childDepth, minSize, childCubifyBudget);
}

void O2FlatCSG::CloseShape()
{
  fBoxes.clear();
  fActive.clear();
  fClosed = false;
  // dropped before the validation below can return: a BVH left over from an earlier CloseShape
  // would describe boxes that no longer exist, and the queries key off `fBVH != nullptr`
  delete static_cast<BVH*>(fBVH);
  fBVH = nullptr;

  EnsureCellBBoxStorage();
  // Refuse the whole shape when a cell's bbox is missing, inverted or non-finite: a cell without a box would vanish.
  bool anyProblem = false;
  for (int cell = 0; cell < GetNcells(); ++cell) {
    if (!fCellBBoxSet[cell]) {
      Error("CloseShape",
            "Shape %s cell %d has no bounding box (SetCellBBox was never called for it); it would "
            "silently vanish from the solid. Not building any boxes -- IsClosed() stays false.",
            GetName(), cell);
      anyProblem = true;
      continue;
    }
    for (int index = 0; index < 3; ++index) {
      const double loValue = fCellLo[3 * cell + index];
      const double hiValue = fCellHi[3 * cell + index];
      if (!std::isfinite(loValue) || !std::isfinite(hiValue)) {
        Error("CloseShape",
              "Shape %s cell %d has a non-finite bounding box on axis %d (lo %g, hi %g). Not "
              "building any boxes -- IsClosed() stays false.",
              GetName(), cell, index, loValue, hiValue);
        anyProblem = true;
        continue;
      }
      if (hiValue < loValue) {
        Error("CloseShape",
              "Shape %s cell %d has an inverted bounding box on axis %d (lo %g > hi %g); "
              "SetCellBBox's arguments look swapped. Not building any boxes -- IsClosed() stays "
              "false.",
              GetName(), cell, index, loValue, hiValue);
        anyProblem = true;
      }
    }
  }
  if (anyProblem) {
    return;
  }

  double partLo[3] = {TGeoShape::Big(), TGeoShape::Big(), TGeoShape::Big()};
  double partHi[3] = {-TGeoShape::Big(), -TGeoShape::Big(), -TGeoShape::Big()};
  for (int cell = 0; cell < GetNcells(); ++cell) {
    for (int index = 0; index < 3; ++index) {
      partLo[index] = std::min(partLo[index], fCellLo[3 * cell + index]);
      partHi[index] = std::max(partHi[index], fCellHi[3 * cell + index]);
    }
  }
  const double diagonal = std::sqrt((partHi[0] - partLo[0]) * (partHi[0] - partLo[0]) +
                                    (partHi[1] - partLo[1]) * (partHi[1] - partLo[1]) +
                                    (partHi[2] - partLo[2]) * (partHi[2] - partLo[2]));
  const double minSize = fMinBoxFraction * diagonal;

#ifndef NDEBUG
  // A cell must lie inside its bbox; one that spills out makes the accelerated queries and the twins disagree.
  {
    const double reach = 1.e-6 * (diagonal > 0. ? diagonal : 1.);
    for (int cell = 0; cell < GetNcells(); ++cell) {
      const double* cellLo = &fCellLo[3 * cell];
      const double* cellHi = &fCellHi[3 * cell];
      for (int axis = 0; axis < 3; ++axis) {
        const int first = (axis + 1) % 3;
        const int second = (axis + 2) % 3;
        for (int side = 0; side < 2; ++side) {
          for (int step1 = 0; step1 <= 4; ++step1) {
            for (int step2 = 0; step2 <= 4; ++step2) {
              double probe[3];
              probe[axis] = side == 0 ? cellLo[axis] - reach : cellHi[axis] + reach;
              probe[first] = cellLo[first] + 0.25 * step1 * (cellHi[first] - cellLo[first]);
              probe[second] = cellLo[second] + 0.25 * step2 * (cellHi[second] - cellLo[second]);
              assert(!CellContains(cell, probe) &&
                     "O2FlatCSG::CloseShape: a cell reaches past the bounding box SetCellBBox was "
                     "given, so this shape and its own _Loop twins answer differently out there. "
                     "The converter's box is the CAD piece's own bbox, so the cell is larger than "
                     "the part: close the cell's halfspaces or refuse the part -- do NOT widen "
                     "the box, which would ship the phantom material");
            }
          }
        }
      }
    }
  }
#endif

  for (int cell = 0; cell < GetNcells(); ++cell) {
    std::vector<int> active;
    active.reserve(fCells[cell].count);
    for (int offset = 0; offset < fCells[cell].count; ++offset) {
      active.push_back(fCells[cell].first + offset);
    }
    SplitBox(cell, &fCellLo[3 * cell], &fCellHi[3 * cell], active, fSplitDepth, minSize,
             kMaxCubifySplits);
  }

  if (!fBoxes.empty()) {
    std::vector<BVHBBox> boxes;
    std::vector<BVHVec3> centers;
    boxes.reserve(fBoxes.size());
    centers.reserve(fBoxes.size());
    for (const auto& box : fBoxes) {
      BVHBBox bounds;
      for (int index = 0; index < 3; ++index) {
        // outward, so a float node box is a superset of the double box it stands for and the
        // traversal can only ever nominate too many candidates -- never drop one
        bounds.min[index] = roundOutward(box.min[index], false);
        bounds.max[index] = roundOutward(box.max[index], true);
      }
      boxes.push_back(bounds);
      centers.emplace_back(bounds.get_center());
    }
    typename bvh::v2::DefaultBuilder<BVHNode>::Config config;
    config.quality = bvh::v2::DefaultBuilder<BVHNode>::Quality::High;
    // One box per leaf: bvh2 enters a leaf without a box test, and each box is visited at most once per traversal.
    config.max_leaf_size = 1;
    fBVH = static_cast<void*>(
      new BVH(bvh::v2::DefaultBuilder<BVHNode>::build(boxes, centers, config)));
  }

  fClosed = true;
  ComputeBBox();
}

Bool_t O2FlatCSG::Contains_Loop(const Double_t* point) const
{
  for (int index = 0; index < GetNcells(); ++index) {
    if (CellContains(index, point)) {
      return kTRUE;
    }
  }
  return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Contains -- inside its box a box's active list is the cell, so the point must first be in the box's own bounds.

Bool_t O2FlatCSG::GetPointsOnSegments(Int_t npoints, Double_t* array) const
{
  if (array == nullptr || npoints <= 0 || !fClosed) {
    return kFALSE;
  }
  // the boxes that carry boundary: those with a non-empty active list
  std::vector<int> boundaryBoxes;
  for (int index = 0; index < static_cast<int>(fBoxes.size()); ++index) {
    if (fBoxes[index].nActive > 0) {
      boundaryBoxes.push_back(index);
    }
  }
  if (boundaryBoxes.empty()) {
    return kFALSE;
  }
  // the R2 low-discrepancy pair O2Tessellated uses, mapped to directions on the unit sphere
  constexpr double kAlpha1 = 0.7548776662466927;
  constexpr double kAlpha2 = 0.5698402909980532;
  constexpr double kFlipProbe = 1.e-6; ///< cm either side of a point at which Contains must change
  const double zAxis[3] = {0., 0., 1.};
  std::vector<double> pairs;
  int produced = 0;
  const long long maxAttempts = 64LL * npoints;
  for (long long attempt = 0; attempt < maxAttempts && produced < npoints; ++attempt) {
    const FlatCSGBox& box = fBoxes[boundaryBoxes[attempt % static_cast<long long>(boundaryBoxes.size())]];
    const double u = std::fmod(0.5 + kAlpha1 * static_cast<double>(attempt + 1), 1.);
    const double v = std::fmod(0.5 + kAlpha2 * static_cast<double>(attempt + 1), 1.);
    const double cosTheta = 1. - 2. * u;
    const double sinTheta = std::sqrt(std::max(0., 1. - cosTheta * cosTheta));
    const double phi = o2::cad::surface::kTwoPi * v;
    const double dir[3] = {sinTheta * std::cos(phi), sinTheta * std::sin(phi), cosTheta};
    const double centre[3] = {0.5 * (box.min[0] + box.max[0]), 0.5 * (box.min[1] + box.max[1]),
                              0.5 * (box.min[2] + box.max[2])};
    double tlo = 0.;
    double thi = TGeoShape::Big();
    if (!slabWindow(box.min, box.max, centre, dir, tlo, thi)) {
      continue;
    }
    const int capacity = maxPairsForCell(box.nActive);
    pairs.resize(2 * static_cast<size_t>(capacity));
    const int found = CellIntervals(box.cell, fActive.data() + box.firstActive, box.nActive, centre, dir, tlo, thi,
                                    pairs.data(), capacity);
    // the first crossing of the cell's surface inside the box; a window end is a box face, not surface
    double crossing = -1.;
    for (int pair = 0; pair < found && crossing < 0.; ++pair) {
      if (pairs[2 * pair] > tlo) {
        crossing = pairs[2 * pair];
      } else if (pairs[2 * pair + 1] < thi) {
        crossing = pairs[2 * pair + 1];
      }
    }
    if (crossing < 0.) {
      continue;
    }
    double* slot = &array[3 * static_cast<size_t>(produced)];
    for (int axis = 0; axis < 3; ++axis) {
      slot[axis] = centre[axis] + crossing * dir[axis];
    }
    // a face between two cells is not boundary of the union: keep only points where containment flips
    double normal[3] = {0., 0., 0.};
    ComputeNormal(slot, zAxis, normal);
    double below[3];
    double above[3];
    for (int axis = 0; axis < 3; ++axis) {
      below[axis] = slot[axis] - kFlipProbe * normal[axis];
      above[axis] = slot[axis] + kFlipProbe * normal[axis];
    }
    if (Contains(below) != Contains(above)) {
      ++produced;
    }
  }
  return produced == npoints ? kTRUE : kFALSE;
}

Bool_t O2FlatCSG::Contains(const Double_t* point) const
{
  if (!fClosed || fBVH == nullptr) {
    // no boxes to walk: answer from the twin rather than report no material
    return Contains_Loop(point);
  }
  const bool inside = traversePoint(*static_cast<const BVH*>(fBVH), point, [&](int index) {
    const FlatCSGBox& box = fBoxes[index];
    if (!boxHoldsPoint(box, point)) {
      return false;
    }
    if (box.nActive == 0) {
      return true; // wholly inside its cell: nothing left to test
    }
    bool inCell = true;
    for (int slot = 0; slot < box.nActive && inCell; ++slot) {
      inCell = EvalHalfspace(fHalfspaces[fActive[box.firstActive + slot]], point) <= 0.;
    }
    return inCell;
  });
  return inside ? kTRUE : kFALSE;
}

int O2FlatCSG::HalfspaceRoots(const FlatCSGHalfspace& halfspace, const double* origin,
                              const double* dir, double* roots)
{
  if (halfspace.kind == FlatCSGHalfspace::kTorus) {
    // the quartic derivation below takes the leading coefficient a4 = |dir|^4 to be exactly 1;
    // a non-unit direction silently returns wrong roots instead of failing, so catch it here
    assert(std::abs(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2] - 1.) < 1.e-9 &&
           "O2FlatCSG::HalfspaceRoots: torus branch requires a unit direction");
    const double* c = halfspace.c;
    const double axis[3] = {c[3], c[4], c[5]};
    const double major = c[6];
    const double minor = c[7];
    const double offset[3] = {origin[0] - c[0], origin[1] - c[1], origin[2] - c[2]};
    // components along the axis, and the perpendicular parts
    const double pz = offset[0] * axis[0] + offset[1] * axis[1] + offset[2] * axis[2];
    const double dz = dir[0] * axis[0] + dir[1] * axis[1] + dir[2] * axis[2];
    double pPerp[3];
    double dPerp[3];
    for (int index = 0; index < 3; ++index) {
      pPerp[index] = offset[index] - pz * axis[index];
      dPerp[index] = dir[index] - dz * axis[index];
    }
    const double pp = pPerp[0] * pPerp[0] + pPerp[1] * pPerp[1] + pPerp[2] * pPerp[2];
    const double dd = dPerp[0] * dPerp[0] + dPerp[1] * dPerp[1] + dPerp[2] * dPerp[2];
    const double pd = pPerp[0] * dPerp[0] + pPerp[1] * dPerp[1] + pPerp[2] * dPerp[2];
    // (|X|^2 + R^2 - r^2)^2 - 4 R^2 (X_perp . X_perp) = 0 with X = P + tD, |D| = 1
    const double e = pp + pz * pz + major * major - minor * minor;
    const double f = pd + pz * dz;
    const double a4 = 1.;
    const double a3 = 4. * f;
    const double a2 = 2. * e + 4. * f * f - 4. * major * major * dd;
    const double a1 = 4. * e * f - 8. * major * major * pd;
    const double a0 = e * e - 4. * major * major * pp;
    // solveQuarticReal is scale-normalised, so the torus needs no degeneracy guard
    const auto found = o2::cad::surface::solveQuarticReal(a4, a3, a2, a1, a0);
    int count = 0;
    for (double value : found) {
      if (count < kMaxRootsPerHalfspace) {
        roots[count++] = value;
      }
    }
    return count;
  }
  const double* c = halfspace.c;
  // A d
  const double ad[3] = {c[0] * dir[0] + c[1] * dir[1] + c[2] * dir[2],
                        c[1] * dir[0] + c[3] * dir[1] + c[4] * dir[2],
                        c[2] * dir[0] + c[4] * dir[1] + c[5] * dir[2]};
  // A o + b
  const double aob[3] = {c[0] * origin[0] + c[1] * origin[1] + c[2] * origin[2] + c[6],
                         c[1] * origin[0] + c[3] * origin[1] + c[4] * origin[2] + c[7],
                         c[2] * origin[0] + c[4] * origin[1] + c[5] * origin[2] + c[8]};
  const double alpha = dir[0] * ad[0] + dir[1] * ad[1] + dir[2] * ad[2];
  const double beta = dir[0] * aob[0] + dir[1] * aob[1] + dir[2] * aob[2];
  const double gamma = EvalHalfspace(halfspace, origin) * halfspace.sign; // sign*sign==1: the unsigned Q(o)

  // a plane has alpha exactly 0 and an axis-parallel ray nearly so: both are linear equations
  // 1e-14 is cm-dependent: a root it discards lies at |t| >= ~1e6 cm, outside any ALICE geometry
  const double reference = std::abs(beta) + std::abs(gamma) + 1.e-300;
  if (std::abs(alpha) <= 1.e-14 * reference) {
    if (std::abs(beta) <= 1.e-300) {
      return 0;
    }
    roots[0] = -0.5 * gamma / beta;
    return 1;
  }
  const double disc = beta * beta - alpha * gamma;
  if (disc < 0.) {
    return 0;
  }
  const double root = std::sqrt(disc);
  // the numerically stable pair, so a grazing ray does not lose the near root to cancellation
  const double q = -(beta + (beta >= 0. ? root : -root));
  if (q == 0.) {
    // q == 0 only when beta == gamma == 0: one double root at t = 0, without the 0/0 of the general formula
    roots[0] = 0.;
    return 1;
  }
  roots[0] = q / alpha;
  roots[1] = gamma / q;
  return 2;
}

int O2FlatCSG::CellIntervals(int cell, const int* active, int nActive, const double* origin,
                             const double* dir, double tlo, double thi, double* out,
                             int maxOut) const
{
  const FlatCSGCell& description = fCells[cell];
  const int count = nActive < 0 ? description.count : nActive;
  if (thi <= tlo) {
    return 0;
  }

  // every root of every active halfspace in the window; thread_local, sized from the cell's halfspace count
  thread_local std::vector<double> breakBuffer;
  const std::size_t needed = 2 + static_cast<std::size_t>(kMaxRootsPerHalfspace) * static_cast<std::size_t>(count);
  if (breakBuffer.size() < needed) {
    breakBuffer.resize(needed);
  }
  double* breaks = breakBuffer.data();
  int nBreaks = 0;
  breaks[nBreaks++] = tlo;
  breaks[nBreaks++] = thi;
  for (int slot = 0; slot < count; ++slot) {
    const int index = active != nullptr ? active[slot] : description.first + slot;
    double roots[kMaxRootsPerHalfspace];
    const int found = HalfspaceRoots(fHalfspaces[index], origin, dir, roots);
    for (int root = 0; root < found; ++root) {
      if (roots[root] > tlo && roots[root] < thi) {
        breaks[nBreaks++] = roots[root];
      }
    }
  }
  std::sort(breaks, breaks + nBreaks);

  // classify the midpoint of each sub-interval and merge the runs that are inside
  int pairs = 0;
  bool open = false;
  bool overflow = false;
  for (int index = 0; index + 1 < nBreaks; ++index) {
    const double lo = breaks[index];
    const double hi = breaks[index + 1];
    if (hi <= lo) {
      continue;
    }
    const double middle = 0.5 * (lo + hi);
    double probe[3] = {origin[0] + middle * dir[0], origin[1] + middle * dir[1],
                       origin[2] + middle * dir[2]};
    bool inside = true;
    for (int slot = 0; slot < count && inside; ++slot) {
      const int halfspace = active != nullptr ? active[slot] : description.first + slot;
      inside = EvalHalfspace(fHalfspaces[halfspace], probe) <= 0.;
    }
    if (inside) {
      if (open) {
        out[2 * (pairs - 1) + 1] = hi;
      } else if (pairs < maxOut) {
        out[2 * pairs] = lo;
        out[2 * pairs + 1] = hi;
        ++pairs;
        open = true;
      } else {
        // maxOut was too small for this cell along this ray: fail loudly (a negative count)
        // rather than hand the caller a silently truncated list that reads as a valid answer
        overflow = true;
        open = false;
      }
    } else {
      open = false;
    }
  }
  return overflow ? -1 : pairs;
}

namespace
{
/// Merge `[enter, exit]` pairs in place, joining ones that touch within \a glue.
int mergeIntervals(double* pairs, int count, double glue)
{
  if (count < 2) {
    return count;
  }
  // sort by entry
  for (int outer = 1; outer < count; ++outer) {
    const double lo = pairs[2 * outer];
    const double hi = pairs[2 * outer + 1];
    int inner = outer - 1;
    while (inner >= 0 && pairs[2 * inner] > lo) {
      pairs[2 * (inner + 1)] = pairs[2 * inner];
      pairs[2 * (inner + 1) + 1] = pairs[2 * inner + 1];
      --inner;
    }
    pairs[2 * (inner + 1)] = lo;
    pairs[2 * (inner + 1) + 1] = hi;
  }
  int kept = 1;
  for (int index = 1; index < count; ++index) {
    if (pairs[2 * index] <= pairs[2 * (kept - 1) + 1] + glue) {
      pairs[2 * (kept - 1) + 1] = std::max(pairs[2 * (kept - 1) + 1], pairs[2 * index + 1]);
    } else {
      pairs[2 * kept] = pairs[2 * index];
      pairs[2 * kept + 1] = pairs[2 * index + 1];
      ++kept;
    }
  }
  return kept;
}
} // namespace

Double_t O2FlatCSG::DistFromOutside_Loop(const Double_t* point, const Double_t* dir,
                                         Double_t step) const
{
  // thread_local: see the comment on the scratch-buffer members it replaced in the header
  thread_local std::vector<double> pairBuffer;
  double best = TGeoShape::Big();
  for (int cell = 0; cell < GetNcells(); ++cell) {
    // sized from this cell's own halfspace count, so a busy cell's intervals are never truncated
    const int capacity = maxPairsForCell(fCells[cell].count);
    if (static_cast<int>(pairBuffer.size()) < 2 * capacity) {
      pairBuffer.resize(2 * capacity);
    }
    const int found = CellIntervals(cell, nullptr, -1, point, dir, 0., step,
                                    pairBuffer.data(), capacity);
    // capacity is provably sufficient (maxPairsForCell), so CellIntervals cannot overflow here;
    // a negative found would mean that bound itself is wrong, which is a bug, not live data
    for (int pair = 0; pair < found; ++pair) {
      // a point exactly on the boundary is already inside; only a real entry counts
      if (pairBuffer[2 * pair + 1] > TGeoShape::Tolerance() && pairBuffer[2 * pair] < best) {
        best = std::max(pairBuffer[2 * pair], 0.);
      }
    }
  }
  return best;
}

Double_t O2FlatCSG::DistFromInside_Loop(const Double_t* point, const Double_t* dir,
                                        Double_t step) const
{
  // the union's occupancy; the buffer fits every cell's worst case at once
  thread_local std::vector<double> pairBuffer;
  int totalCapacity = 0;
  for (int cell = 0; cell < GetNcells(); ++cell) {
    totalCapacity += maxPairsForCell(fCells[cell].count);
  }
  if (static_cast<int>(pairBuffer.size()) < 2 * totalCapacity) {
    pairBuffer.resize(2 * totalCapacity);
  }
  int count = 0;
  for (int cell = 0; cell < GetNcells(); ++cell) {
    // not expected to overflow, but a negative count must never reach the pointer arithmetic
    const int found = CellIntervals(cell, nullptr, -1, point, dir, 0., step,
                                    pairBuffer.data() + 2 * count, totalCapacity - count);
    if (found < 0) {
      Error("DistFromInside_Loop",
            "CellIntervals overflowed for cell %d: the maxPairsForCell bound no longer holds",
            cell);
      return TGeoShape::Big();
    }
    count += found;
  }
  count = mergeIntervals(pairBuffer.data(), count, TGeoShape::Tolerance());
  for (int pair = 0; pair < count; ++pair) {
    if (pairBuffer[2 * pair] <= TGeoShape::Tolerance()) {
      return pairBuffer[2 * pair + 1];
    }
  }
  return 0.;
}

////////////////////////////////////////////////////////////////////////////////
/// GatherRayPieces -- each box's window is its own slab intersected with `[0, step]`, never pooled across boxes.

bool O2FlatCSG::GatherRayPieces(const Double_t* point, const Double_t* dir, Double_t step,
                                std::vector<double>& pairs, std::vector<int>& cells, RayBound bound,
                                double& smallestPruned) const
{
  pairs.clear();
  cells.clear();
  smallestPruned = TGeoShape::Big();
  const BVH& bvh = *static_cast<const BVH*>(fBVH);

  // one box's intervals; thread_local for the reason the header's scratch-buffer comment gives
  thread_local std::vector<double> boxPairs;
  bool overflowed = false;
  // the running bound: with kEntry an upper bound on DistFromOutside's answer, with kExit the far
  // end of the interval holding t = 0; a box entered past it cannot change the answer
  double limit = step;
  double reach = -1.; // kExit's chain end, negative until a piece holds t = 0
  // only kExit can prune a box that later turns out to matter, so only it needs the record
  double* culled = bound == RayBound::kExit ? &smallestPruned : nullptr;
  traverseRay(bvh, point, dir, step, limit, bound != RayBound::kNone, culled, [&](int index) {
    const FlatCSGBox& box = fBoxes[index];
    double tlo = 0.;
    double thi = step;
    if (!slabWindow(box.min, box.max, point, dir, tlo, thi) || thi <= tlo) {
      return;
    }
    if (tlo > limit) {
      if (culled != nullptr && tlo < smallestPruned) {
        smallestPruned = tlo;
      }
      return;
    }
    // sized from THIS box's active-list length, which is the count CellIntervals will walk, so
    // the bound it is asked to respect is the one it was given
    const int capacity = maxPairsForCell(box.nActive);
    if (static_cast<int>(boxPairs.size()) < 2 * capacity) {
      boxPairs.resize(2 * capacity);
    }
    // nActive == 0 means the box is wholly inside its cell; CellIntervals then has no halfspace
    // to break on and returns the whole window, which is exactly the right answer
    const int* active = box.nActive > 0 ? fActive.data() + box.firstActive : nullptr;
    const int found = CellIntervals(box.cell, active, box.nActive, point, dir, tlo, thi,
                                    boxPairs.data(), capacity);
    if (found < 0) {
      overflowed = true;
      return;
    }
    for (int pair = 0; pair < found; ++pair) {
      const double enter = boxPairs[2 * pair];
      const double exit = boxPairs[2 * pair + 1];
      pairs.push_back(enter);
      pairs.push_back(exit);
      cells.push_back(box.cell);
      if (bound == RayBound::kEntry && exit > TGeoShape::Tolerance()) {
        limit = std::min(limit, std::max({enter, 0., TGeoShape::Tolerance()}));
      } else if (bound == RayBound::kExit &&
                 (reach < 0. ? enter <= TGeoShape::Tolerance() : enter <= reach + TGeoShape::Tolerance())) {
        // the chain of pieces holding t = 0, joined with DistFromInside's own merge glue
        reach = std::max(reach, exit);
        limit = std::min(step, reach + TGeoShape::Tolerance());
      }
    }
  });
  return !overflowed;
}

////////////////////////////////////////////////////////////////////////////////
/// DistFromOutsideBVH -- the pieces are rejoined per cell, never across cells, as the twin's per-cell intervals.

Double_t O2FlatCSG::DistFromOutsideBVH(const Double_t* point, const Double_t* dir,
                                       Double_t step) const
{
  thread_local std::vector<double> pairs;
  thread_local std::vector<int> cells;
  double smallestPruned = TGeoShape::Big();
  if (!GatherRayPieces(point, dir, step, pairs, cells, RayBound::kEntry, smallestPruned)) {
    Error("DistFromOutside",
          "Shape %s: CellIntervals overflowed a per-box buffer sized from that box's own active "
          "list; the maxPairsForCell bound no longer holds. Answering from the loop twin.",
          GetName());
    return DistFromOutside_Loop(point, dir, step);
  }

  // sort the pieces by (cell, entry) through a permutation, so the run merge below sees each
  // cell's pieces contiguously and in order
  const int count = static_cast<int>(cells.size());
  thread_local std::vector<int> order;
  order.resize(count);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int left, int right) {
    if (cells[left] != cells[right]) {
      return cells[left] < cells[right];
    }
    return pairs[2 * left] < pairs[2 * right];
  });

  double best = TGeoShape::Big();
  int index = 0;
  while (index < count) {
    const int cell = cells[order[index]];
    const double enter = pairs[2 * order[index]];
    double exit = pairs[2 * order[index] + 1];
    ++index;
    // join what is only one interval of this cell, cut into pieces by the boxes that tile it
    while (index < count && cells[order[index]] == cell && pairs[2 * order[index]] <= exit) {
      exit = std::max(exit, pairs[2 * order[index] + 1]);
      ++index;
    }
    // DistFromOutside_Loop's rule, unchanged: a point exactly on the boundary is already inside,
    // so only an interval that really extends past the tolerance counts as an entry
    if (exit > TGeoShape::Tolerance() && enter < best) {
      best = std::max(enter, 0.);
    }
  }
  return best;
}

////////////////////////////////////////////////////////////////////////////////
/// DistFromInsideBVH -- the far end of the union's interval containing t = 0, merged across cells with the twin's glue.

Double_t O2FlatCSG::DistFromInsideBVH(const Double_t* point, const Double_t* dir,
                                      Double_t step) const
{
  thread_local std::vector<double> pairs;
  thread_local std::vector<int> cells;
  for (int attempt = 0; attempt < 2; ++attempt) {
    // the bound grows as pieces merge, so a box skipped against an earlier, smaller one might have
    // mattered after all; the second attempt does not prune and is the definition of the answer
    const RayBound bound = attempt == 0 ? RayBound::kExit : RayBound::kNone;
    double smallestPruned = TGeoShape::Big();
    if (!GatherRayPieces(point, dir, step, pairs, cells, bound, smallestPruned)) {
      Error("DistFromInside",
            "Shape %s: CellIntervals overflowed a per-box buffer sized from that box's own active "
            "list; the maxPairsForCell bound no longer holds. Answering from the loop twin.",
            GetName());
      return DistFromInside_Loop(point, dir, step);
    }
    const int count = mergeIntervals(pairs.data(), static_cast<int>(cells.size()),
                                     TGeoShape::Tolerance());
    double answer = 0.;
    for (int pair = 0; pair < count; ++pair) {
      if (pairs[2 * pair] <= TGeoShape::Tolerance()) {
        answer = pairs[2 * pair + 1];
        break;
      }
    }
    if (attempt == 1 || smallestPruned > answer + TGeoShape::Tolerance()) {
      return answer;
    }
    ++gUnprunedRetryCount;
  }
  return 0.; // unreachable: the second attempt never prunes
}

void O2FlatCSG::ResetUnprunedRetryCounter()
{
  gUnprunedRetryCount = 0;
}

long long O2FlatCSG::GetUnprunedRetryCount()
{
  return gUnprunedRetryCount;
}

Double_t O2FlatCSG::DistFromOutside(const Double_t* point, const Double_t* dir, Int_t iact,
                                    Double_t step, Double_t* safe) const
{
  if (iact < 3 && safe != nullptr) {
    *safe = Safety(point, kFALSE);
    if (iact == 0) {
      return TGeoShape::Big();
    }
    if (iact == 1 && step < *safe) {
      return TGeoShape::Big();
    }
  }
  if (!fClosed || fBVH == nullptr) {
    // no boxes to walk: see the note on Contains. The twin is the definition of the answer, and
    // an empty box array in the accelerated path would silently report empty space.
    return DistFromOutside_Loop(point, dir, step);
  }
  return DistFromOutsideBVH(point, dir, step);
}

Double_t O2FlatCSG::DistFromInside(const Double_t* point, const Double_t* dir, Int_t iact,
                                   Double_t step, Double_t* safe) const
{
  if (iact < 3 && safe != nullptr) {
    *safe = Safety(point, kTRUE);
    if (iact == 0) {
      return TGeoShape::Big();
    }
    if (iact == 1 && step < *safe) {
      return TGeoShape::Big();
    }
  }
  if (!fClosed || fBVH == nullptr) {
    return DistFromInside_Loop(point, dir, step);
  }
  return DistFromInsideBVH(point, dir, step);
}

////////////////////////////////////////////////////////////////////////////////
/// Safety_Loop -- outside the distance to the nearest box; inside the distance to the faces of a wholly-inside box, else 0.

Double_t O2FlatCSG::Safety_Loop(const Double_t* point, Bool_t in) const
{
  if (!in) {
    double best = TGeoShape::Big();
    for (const auto& box : fBoxes) {
      best = std::min(best, boxDistanceSquared(box, point));
    }
    return best >= TGeoShape::Big() ? 0. : std::sqrt(best);
  }

  double best = 0.;
  for (const auto& box : fBoxes) {
    if (boxHoldsPoint(box, point) && box.nActive == 0) {
      best = std::max(best, distanceToFaces(box, point));
    }
  }
  return std::max(best, 0.);
}

////////////////////////////////////////////////////////////////////////////////
/// Safety -- Safety_Loop's computation through the BVH; the pruning never drops the nearest box.

Double_t O2FlatCSG::Safety(const Double_t* point, Bool_t in) const
{
  if (!fClosed || fBVH == nullptr) {
    // no boxes to walk: see the note on Contains -- the twin is the definition of the answer.
    return Safety_Loop(point, in);
  }
  const BVH& bvh = *static_cast<const BVH*>(fBVH);

  if (!in) {
    // node boxes are read back as double and measured against the double point: a float query could prune the nearest box
    using DVec3 = bvh::v2::Vec<double, 3>;
    using DBBox = bvh::v2::BBox<double, 3>;
    const DVec3 dpoint(point[0], point[1], point[2]);
    const auto nodeDistanceSquared = [&bvh, &dpoint](size_t index) {
      const auto& fbox = bvh.nodes[index].get_bbox();
      const DBBox dbox(DVec3(static_cast<double>(fbox.min[0]), static_cast<double>(fbox.min[1]),
                             static_cast<double>(fbox.min[2])),
                       DVec3(static_cast<double>(fbox.max[0]), static_cast<double>(fbox.max[1]),
                             static_cast<double>(fbox.max[2])));
      return bvh::v2::extra::SafetySqToNode(dbox, dpoint);
    };
    struct NodeEntry {
      size_t node;
      double squared; ///< the node box's squared distance, computed once when pushed
    };
    thread_local std::vector<NodeEntry> nearStack;
    nearStack.clear();
    nearStack.push_back({0, nodeDistanceSquared(0)}); // the bvh2 root node
    double best = TGeoShape::Big();
    while (!nearStack.empty()) {
      const NodeEntry entry = nearStack.back();
      nearStack.pop_back();
      const auto& node = bvh.nodes[entry.node];
      if (entry.squared >= best) {
        continue; // this subtree cannot hold anything nearer than what is already found
      }
      if (node.is_leaf()) {
        const auto beginPrimitive = node.index.first_id();
        const auto endPrimitive = beginPrimitive + node.index.prim_count();
        for (auto primitive = beginPrimitive; primitive < endPrimitive; ++primitive) {
          best = std::min(best, boxDistanceSquared(fBoxes[bvh.prim_ids[primitive]], point));
        }
      } else {
        // nearer child first, pruning on the way in; the same min in another order
        const auto firstChild = node.index.first_id();
        size_t children[2] = {firstChild, firstChild + 1};
        double childSquared[2] = {TGeoShape::Big(), TGeoShape::Big()};
        for (int index = 0; index < 2; ++index) {
          if (children[index] < bvh.nodes.size()) {
            childSquared[index] = nodeDistanceSquared(children[index]);
          }
        }
        const int nearer = childSquared[0] <= childSquared[1] ? 0 : 1;
        const int farther = 1 - nearer;
        // LIFO, so the farther child is pushed first and popped last.
        if (children[farther] < bvh.nodes.size() && childSquared[farther] < best) {
          nearStack.push_back({children[farther], childSquared[farther]});
        }
        if (children[nearer] < bvh.nodes.size() && childSquared[nearer] < best) {
          nearStack.push_back({children[nearer], childSquared[nearer]});
        }
      }
    }
    return best >= TGeoShape::Big() ? 0. : std::sqrt(best);
  }

  double best = 0.;
  traversePoint(bvh, point, [&](int index) {
    const FlatCSGBox& box = fBoxes[index];
    if (boxHoldsPoint(box, point) && box.nActive == 0) {
      best = std::max(best, distanceToFaces(box, point));
    }
    return false;
  });
  return std::max(best, 0.);
}

Double_t O2FlatCSG::Capacity() const
{
  // the cells of a decomposition are disjoint by construction, so their own volumes just sum
  return std::accumulate(fCells.begin(), fCells.end(), 0.,
                         [](double sum, const FlatCSGCell& cell) { return sum + cell.volume; });
}

////////////////////////////////////////////////////////////////////////////////
/// ComputeNormal -- the halfspace with the smallest first-order distance |f| / |grad f| among the active list of the box
/// holding \a point, which HalfspaceRange's 64-eps pad makes every halfspace that can be at equality there.

void O2FlatCSG::ComputeNormal(const Double_t* point, const Double_t* dir, Double_t* norm) const
{
  norm[0] = norm[1] = norm[2] = 0.;
  if (fHalfspaces.empty()) {
    return;
  }

  // the candidates: the active list of the box that holds the point; for a box wholly inside its
  // cell, that cell's halfspace run; and when no box holds the point, every halfspace
  const int* activeList = nullptr;
  int rangeFirst = 0;
  int nCandidates = GetNhalfspaces();
  if (fClosed && fBVH != nullptr) {
    traversePoint(*static_cast<const BVH*>(fBVH), point, [&](int index) {
      const FlatCSGBox& box = fBoxes[index];
      if (!boxHoldsPoint(box, point)) {
        return false;
      }
      if (box.nActive > 0) {
        activeList = fActive.data() + box.firstActive;
        nCandidates = box.nActive;
      } else {
        rangeFirst = fCells[box.cell].first;
        nCandidates = fCells[box.cell].count;
      }
      return true; // cells are disjoint; the first box that holds the point is the answer
    });
  }
  const auto indexAt = [&](int slot) { return activeList != nullptr ? activeList[slot] : rangeFirst + slot; };

  int best = -1;
  double bestValue = std::numeric_limits<double>::infinity();
  double bestGrad[3] = {0., 0., 0.};
  for (int slot = 0; slot < nCandidates; ++slot) {
    const int candidate = indexAt(slot);
    const FlatCSGHalfspace& halfspace = fHalfspaces[candidate];
    const double f = EvalHalfspace(halfspace, point);
    double grad[3];
    halfspaceGradient(halfspace, point, grad);
    const double gradLength = std::sqrt(grad[0] * grad[0] + grad[1] * grad[1] + grad[2] * grad[2]);
    if (gradLength < 1.e-300) {
      continue; // degenerate gradient (see halfspaceGradient); this halfspace cannot win
    }
    const double value = std::abs(f) / gradLength; // the first-order distance to this surface
    if (value < bestValue) {
      bestValue = value;
      best = candidate;
      bestGrad[0] = grad[0] / gradLength;
      bestGrad[1] = grad[1] / gradLength;
      bestGrad[2] = grad[2] / gradLength;
    }
  }

  if (best < 0) {
    // every candidate's gradient was degenerate (a torus axis or core circle): fall back to the travel direction
    const double dirLength = std::sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
    if (dirLength > 1.e-300) {
      for (int index = 0; index < 3; ++index) {
        norm[index] = dir[index] / dirLength;
      }
    }
    return;
  }

  for (int index = 0; index < 3; ++index) {
    norm[index] = bestGrad[index];
  }
  const double dot = norm[0] * dir[0] + norm[1] * dir[1] + norm[2] * dir[2];
  if (dot < 0.) {
    for (int index = 0; index < 3; ++index) {
      norm[index] = -norm[index];
    }
  }
}

void O2FlatCSG::ComputeBBox()
{
  // the union of the retained sub-cell boxes, tighter than the union of the cell AABBs
  if (fBoxes.empty()) {
    return;
  }
  double lo[3] = {TGeoShape::Big(), TGeoShape::Big(), TGeoShape::Big()};
  double hi[3] = {-TGeoShape::Big(), -TGeoShape::Big(), -TGeoShape::Big()};
  for (const FlatCSGBox& box : fBoxes) {
    for (int index = 0; index < 3; ++index) {
      lo[index] = std::min(lo[index], box.min[index]);
      hi[index] = std::max(hi[index], box.max[index]);
    }
  }
  for (int index = 0; index < 3; ++index) {
    fOrigin[index] = 0.5 * (lo[index] + hi[index]);
  }
  fDX = 0.5 * (hi[0] - lo[0]);
  fDY = 0.5 * (hi[1] - lo[1]);
  fDZ = 0.5 * (hi[2] - lo[2]);
}

} // namespace cad
} // namespace o2
