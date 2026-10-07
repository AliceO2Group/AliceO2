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
/// \since 2026-07

#include "CADSupport/O2BVHSurfaceSolid.h"

#include "BoundedSurface.h"

// the third-party BVH headers plus extra kernels, shared with O2Tessellated
#include "bvh2_third_party.h"
#include "bvh2_extra_kernels.h"

#include "TBuffer.h"
#include "TBuffer3D.h"
#include "TBuffer3DTypes.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <utility>

using namespace o2::cad;
using namespace o2::cad::surface;
ClassImp(O2BVHSurfaceSolid);

namespace
{
// float BVH types following the O2Tessellated::BuildBVH pattern
using BVHScalar = float;
using BVHBBox = bvh::v2::BBox<BVHScalar, 3>;
using BVHVec3 = bvh::v2::Vec<BVHScalar, 3>;
using BVHNode = bvh::v2::Node<BVHScalar, 3>;
using BVH = bvh::v2::Bvh<BVHNode>;
using BVHRay = bvh::v2::Ray<BVHScalar, 3>;

Vec2 makeVec2(const O2BVHSurfaceSolid::Point2D& point)
{
  return {point[0], point[1]};
}

Vec3 makeVec3(const O2BVHSurfaceSolid::Point3D& point)
{
  return {point[0], point[1], point[2]};
}

Vec3 makeVec3(const Double_t* point)
{
  return {point[0], point[1], point[2]};
}

// The arbitrary skew test direction used for parity-based containment: probes all normals and
// avoids evident symmetries (same as O2Tessellated), normalized so hit distances are lengths.
const Vec3 kContainsTestDirection = normalized({1., 1.41421356237, 1.73205080757});

/// The re-shoot vote's directions: five golden-angle spiral directions, well separated and off every axis and symmetry plane.
const std::array<Vec3, 5>& reshootDirections()
{
  static const std::array<Vec3, 5> directions = [] {
    std::array<Vec3, 5> spiral{};
    for (int index = 0; index < 5; ++index) {
      const double cosTheta = 1. - 2. * (index + 0.5) / 5.;
      const double sinTheta = std::sqrt(1. - cosTheta * cosTheta);
      const double phi = 2.399963229728653 * index; // golden angle
      spiral[index] = normalized({sinTheta * std::cos(phi), sinTheta * std::sin(phi), cosTheta});
    }
    return spiral;
  }();
  return directions;
}

// Ray tmax tightening in the BVH distance queries; see O2BVHSurfaceSolid::SetRayTMaxPruning.
bool gRayTMaxPruning = true;
// Per-thread diagnostic counter of leaf surface patches visited by the BVH distance queries.
thread_local long long gRayCandidateCount = 0;
// ... and by the nearest-patch queries behind Safety and ComputeNormal; see
// O2BVHSurfaceSolid::ResetSafetyCandidateCounter.
thread_local long long gSafetyCandidateCount = 0;
// Deliberately unsound node bound for the nearest-patch traversal; see
// O2BVHSurfaceSolid::SetSafetyBoundUnsoundForTest. Never true outside a test.
bool gSafetyBoundUnsound = false;

// Per-thread backing store of SurfaceVisitMarker, one stamp per surface index plus the epoch the
// live marker stamps with; see the class below.
thread_local std::vector<unsigned long long> gSurfaceVisitStamps;
thread_local unsigned long long gSurfaceVisitEpoch = 0;

/// Per-query dedup of the surfaces a traversal hands on, epoch-stamped over a thread_local array.
/// Traversals never nest, so one stamp array per thread is enough.
class SurfaceVisitMarker
{
 public:
  explicit SurfaceVisitMarker(size_t surfaceCount) : mStamps(gSurfaceVisitStamps), mEpoch(++gSurfaceVisitEpoch)
  {
    if (mStamps.size() < surfaceCount) {
      mStamps.resize(surfaceCount, 0);
    }
  }

  /// True exactly once per surface index and marker lifetime.
  bool firstVisit(size_t index)
  {
    if (mStamps[index] == mEpoch) {
      return false;
    }
    mStamps[index] = mEpoch;
    return true;
  }

 private:
  /// bound once per query rather than looked up per visit, which is the hot path
  std::vector<unsigned long long>& mStamps;
  unsigned long long mEpoch;
};

/// Squared distance from \a point to a node box, shrunk by (1 - 1e-12) so it never exceeds the distance to a patch inside.
/// \a unsoundBound is gSafetyBoundUnsound, read once per query by the caller.
inline double boxDistanceSq(const BVHBBox& box, const Vec3& point, bool unsoundBound)
{
  const double coordinates[3] = {point.xCoord, point.yCoord, point.zCoord};
  double distanceSq = 0.;
  for (int dimension = 0; dimension < 3; ++dimension) {
    const double lower = static_cast<double>(box.min[dimension]);
    const double upper = static_cast<double>(box.max[dimension]);
    const double value = coordinates[dimension];
    if (value < lower) {
      distanceSq += (lower - value) * (lower - value);
    } else if (value > upper) {
      distanceSq += (value - upper) * (value - upper);
    }
  }
  if (unsoundBound) {
    // The negative control: the distance to the box centre bounds nothing.
    double centreDistanceSq = 0.;
    for (int dimension = 0; dimension < 3; ++dimension) {
      const double centre =
        0.5 * (static_cast<double>(box.min[dimension]) + static_cast<double>(box.max[dimension]));
      const double gap = coordinates[dimension] - centre;
      centreDistanceSq += gap * gap;
    }
    return centreDistanceSq;
  }
  return distanceSq * (1. - 1.e-12);
}

/// Convert a double ray bound to float, rounding up so the float bound is never below the double one.
inline BVHScalar truncateRoundUp(double bound)
{
  const double clamped = std::min(bound, static_cast<double>(std::numeric_limits<BVHScalar>::max()));
  const double biased = clamped + std::numeric_limits<BVHScalar>::epsilon() * std::abs(clamped);
  return static_cast<BVHScalar>(biased);
}

/// Lower ray parameter of the distance queries: just behind the origin, so a point on a face sees its t = 0 crossing.
constexpr double kDistanceRayTolerance = -kRayTolerance;

/// Which side of the surface a hit is on for a ray along \a rayDirection; Tangential within kTolerance of tangency.
enum class CrossingSense { Entering,
                           Exiting,
                           Tangential };

/// Twice the half-width of the window sameIntersection() treats as one intersection at \a distance.
inline double clusterMargin(double distance)
{
  return 2. * kIntersectionTolerance * std::max(1., std::abs(distance));
}

inline CrossingSense crossingSense(const RayHit& hit, const Vec3& rayDirection)
{
  const double alignment = dot(hit.normal, rayDirection);
  if (alignment < -kTolerance) {
    return CrossingSense::Entering;
  }
  if (alignment > kTolerance) {
    return CrossingSense::Exiting;
  }
  return CrossingSense::Tangential;
}

/// Sort \a hits and visit their clusters in increasing distance; a cluster with both senses is a graze and reports Tangential.
template <typename ClusterVisitor>
void forEachCrossingCluster(std::vector<RayHit>& hits, const Vec3& rayDirection, ClusterVisitor&& visitor)
{
  std::sort(hits.begin(), hits.end(),
            [](const RayHit& firstHit, const RayHit& secondHit) { return firstHit.distance < secondHit.distance; });

  size_t hitIndex = 0;
  while (hitIndex < hits.size()) {
    bool entering = false;
    bool exiting = false;
    size_t clusterEnd = hitIndex;
    // Compared against the cluster's first member, not its predecessor: chaining would merge thin features at large t.
    while (clusterEnd < hits.size() &&
           (clusterEnd == hitIndex || sameIntersection(hits[clusterEnd].distance, hits[hitIndex].distance))) {
      switch (crossingSense(hits[clusterEnd], rayDirection)) {
        case CrossingSense::Entering:
          entering = true;
          break;
        case CrossingSense::Exiting:
          exiting = true;
          break;
        case CrossingSense::Tangential:
          break;
      }
      ++clusterEnd;
    }
    // both, or neither: nothing was crossed
    const CrossingSense sense = entering == exiting ? CrossingSense::Tangential
                                                    : (entering ? CrossingSense::Entering : CrossingSense::Exiting);
    if (!visitor(hitIndex, clusterEnd, sense)) {
      return;
    }
    hitIndex = clusterEnd;
  }
}

/// Distance to the nearest genuine entering or exiting crossing in \a hits, or Big; \a grazedFirst reports a graze on the way.
template <bool wantEntering>
double nearestCrossingInHits(std::vector<RayHit>& hits, const Vec3& rayDirection, bool& grazedFirst)
{
  constexpr CrossingSense wanted = wantEntering ? CrossingSense::Entering : CrossingSense::Exiting;
  double distance = TGeoShape::Big();
  grazedFirst = false;
  forEachCrossingCluster(hits, rayDirection, [&](size_t firstIndex, size_t, CrossingSense sense) {
    if (sense == CrossingSense::Tangential) {
      grazedFirst = true;
      return true;
    }
    if (sense != wanted) {
      return true;
    }
    // clusters come in increasing distance, so the first match is the answer; a crossing is never negative
    distance = std::max(0., hits[firstIndex].distance);
    return false;
  });
  return distance;
}

/// @name Persistent surface records: translation between the Add*Surface arguments and BVHSurfaceRecord
/// @{

void fillPoint3(double (&target)[3], const O2BVHSurfaceSolid::Point3D& source)
{
  target[0] = source[0];
  target[1] = source[1];
  target[2] = source[2];
}

O2BVHSurfaceSolid::Point3D makePoint3D(const double (&source)[3])
{
  return {source[0], source[1], source[2]};
}

/// The frame-and-scalars part of a record, shared by all six surface families.
BVHSurfaceRecord makeRecord(int kind, const O2BVHSurfaceSolid::Point3D& origin,
                            const O2BVHSurfaceSolid::Point3D& axisA, const O2BVHSurfaceSolid::Point3D& axisB,
                            std::vector<double> scalars, bool innerWall, bool trimmed)
{
  BVHSurfaceRecord record;
  record.kind = kind;
  fillPoint3(record.origin, origin);
  fillPoint3(record.axisA, axisA);
  fillPoint3(record.axisB, axisB);
  record.scalars = std::move(scalars);
  record.innerWall = innerWall;
  record.trimmed = trimmed;
  return record;
}

BVHSurfaceCurveRecord makeCurveRecord(const O2BVHSurfaceSolid::PlanarBoundaryCurve& curve)
{
  BVHSurfaceCurveRecord record;
  record.kind = static_cast<int>(curve.kind);
  record.lineStart[0] = curve.lineStart[0];
  record.lineStart[1] = curve.lineStart[1];
  record.lineEnd[0] = curve.lineEnd[0];
  record.lineEnd[1] = curve.lineEnd[1];
  record.center[0] = curve.center[0];
  record.center[1] = curve.center[1];
  record.radius = curve.radius;
  record.startAngle = curve.startAngle;
  record.endAngle = curve.endAngle;
  record.degree = curve.degree;
  record.poles.reserve(2 * curve.poles.size());
  for (const auto& pole : curve.poles) {
    record.poles.push_back(pole[0]);
    record.poles.push_back(pole[1]);
  }
  record.weights = curve.weights;
  record.knots = curve.knots;
  return record;
}

O2BVHSurfaceSolid::PlanarBoundaryCurve makeBoundaryCurve(const BVHSurfaceCurveRecord& record)
{
  O2BVHSurfaceSolid::PlanarBoundaryCurve curve;
  curve.kind = static_cast<O2BVHSurfaceSolid::PlanarBoundaryCurve::Kind>(record.kind);
  curve.lineStart = {record.lineStart[0], record.lineStart[1]};
  curve.lineEnd = {record.lineEnd[0], record.lineEnd[1]};
  curve.center = {record.center[0], record.center[1]};
  curve.radius = record.radius;
  curve.startAngle = record.startAngle;
  curve.endAngle = record.endAngle;
  curve.degree = record.degree;
  curve.poles.reserve(record.poles.size() / 2);
  for (size_t index = 0; index + 1 < record.poles.size(); index += 2) {
    curve.poles.push_back({record.poles[index], record.poles[index + 1]});
  }
  curve.weights = record.weights;
  curve.knots = record.knots;
  return curve;
}

/// Store an outer wire plus its holes as one flat curve list with per-wire sizes.
void storeCurveWires(BVHSurfaceRecord& record, const std::vector<O2BVHSurfaceSolid::PlanarBoundaryCurve>& outerWire,
                     const std::vector<std::vector<O2BVHSurfaceSolid::PlanarBoundaryCurve>>& innerWires)
{
  record.wireSizes.push_back(static_cast<int>(outerWire.size()));
  for (const auto& curve : outerWire) {
    record.curves.push_back(makeCurveRecord(curve));
  }
  for (const auto& innerWire : innerWires) {
    record.wireSizes.push_back(static_cast<int>(innerWire.size()));
    for (const auto& curve : innerWire) {
      record.curves.push_back(makeCurveRecord(curve));
    }
  }
}

/// The inverse of storeCurveWires. Returns false when the per-wire sizes do not add up to the
/// stored curve count, i.e. when the record is truncated or corrupt.
bool loadCurveWires(const BVHSurfaceRecord& record, std::vector<O2BVHSurfaceSolid::PlanarBoundaryCurve>& outerWire,
                    std::vector<std::vector<O2BVHSurfaceSolid::PlanarBoundaryCurve>>& innerWires)
{
  size_t consumed = 0;
  for (size_t wireIndex = 0; wireIndex < record.wireSizes.size(); ++wireIndex) {
    const int wireSize = record.wireSizes[wireIndex];
    if (wireSize < 0 || consumed + static_cast<size_t>(wireSize) > record.curves.size()) {
      return false;
    }
    auto& wire = wireIndex == 0 ? outerWire : innerWires.emplace_back();
    for (int curveIndex = 0; curveIndex < wireSize; ++curveIndex) {
      wire.push_back(makeBoundaryCurve(record.curves[consumed + curveIndex]));
    }
    consumed += static_cast<size_t>(wireSize);
  }
  return consumed == record.curves.size();
}

/// storeCurveWires/loadCurveWires for the polygon-vertex flavour of a planar surface.
void storePolygonWires(BVHSurfaceRecord& record, const std::vector<O2BVHSurfaceSolid::Point2D>& outerWire,
                       const std::vector<std::vector<O2BVHSurfaceSolid::Point2D>>& innerWires)
{
  const auto append = [&record](const std::vector<O2BVHSurfaceSolid::Point2D>& wire) {
    record.wireSizes.push_back(static_cast<int>(wire.size()));
    for (const auto& vertex : wire) {
      record.polygonPoints.push_back(vertex[0]);
      record.polygonPoints.push_back(vertex[1]);
    }
  };
  append(outerWire);
  for (const auto& innerWire : innerWires) {
    append(innerWire);
  }
}

bool loadPolygonWires(const BVHSurfaceRecord& record, std::vector<O2BVHSurfaceSolid::Point2D>& outerWire,
                      std::vector<std::vector<O2BVHSurfaceSolid::Point2D>>& innerWires)
{
  size_t consumed = 0;
  for (size_t wireIndex = 0; wireIndex < record.wireSizes.size(); ++wireIndex) {
    const int wireSize = record.wireSizes[wireIndex];
    if (wireSize < 0 || 2 * (consumed + static_cast<size_t>(wireSize)) > record.polygonPoints.size()) {
      return false;
    }
    auto& wire = wireIndex == 0 ? outerWire : innerWires.emplace_back();
    for (int vertexIndex = 0; vertexIndex < wireSize; ++vertexIndex) {
      const size_t offset = 2 * (consumed + vertexIndex);
      wire.push_back({record.polygonPoints[offset], record.polygonPoints[offset + 1]});
    }
    consumed += static_cast<size_t>(wireSize);
  }
  return 2 * consumed == record.polygonPoints.size();
}
/// @}

// Ray parity of a full intersection list (sorts in place); a mixed-sense cluster is a graze and counts even.
bool oddCrossingParity(std::vector<RayHit>& hits, const Vec3& rayDirection)
{
  int crossings = 0;
  forEachCrossingCluster(hits, rayDirection, [&](size_t, size_t, CrossingSense sense) {
    if (sense != CrossingSense::Tangential) {
      ++crossings;
    }
    return true;
  });
  return (crossings & 1) != 0;
}

/// A rim's state on the solid's scale; the solid reports the worst over its rims.
O2BVHSurfaceSolid::NavigationReliability rimStateToReliability(RimState state)
{
  using Reliability = O2BVHSurfaceSolid::NavigationReliability;
  switch (state) {
    case RimState::Matched:
      return Reliability::Reliable;
    case RimState::Reversed:
      return Reliability::ReversedFaces;
    case RimState::Boundary:
      return Reliability::OpenSurfaceSet;
    case RimState::NonManifold:
      return Reliability::NonManifold;
  }
  return Reliability::Undetermined;
}
} // namespace

struct O2BVHSurfaceSolid::Impl {
  std::vector<std::unique_ptr<BoundedSurface>> surfaces;
  std::vector<Vec3> displayVertices;
  std::vector<std::array<int, 3>> displayTriangles;
  /// The surface each display triangle came from, parallel to displayTriangles; see GetPointsOnSegments.
  std::vector<int> displayTriangleSurface;
  ClosureReport closure;
  /// closure.rimRecords in the public form, built once by CloseShape so the accessor can hand out
  /// a reference. The two are the same data; only the state enum and the Vec3 differ in type.
  std::vector<RimReport> rimReports;
  bool defined = false;
  std::unique_ptr<BVH> bvh; //!< acceleration structure over the sub-patch cover boxes (built in CloseShape)
  /// The surface of each BVH leaf primitive, in leaf order.
  std::vector<int> leafSurface;
  /// GetNavigationReliability() is Reliable; set by CloseShape.
  bool reliable = false;
  /// A few on-patch display vertices, seeding the nearest-patch traversal's upper bound; see anchorSeedDistanceSq.
  std::vector<Vec3> safetyAnchors;

  /// Build the BVH over the surfaces' cover boxes, widened by kBVHBoxTolerance and rounded outward to float.
  void buildBVH()
  {
    bvh.reset();
    leafSurface.clear();
    if (surfaces.empty()) {
      return;
    }

    std::vector<BVHBBox> primitiveBoxes;
    std::vector<BVHVec3> primitiveCenters;
    std::vector<BoundedSurface::CoverBox> coverBoxes;
    std::vector<int> coverSurface;
    for (size_t surfaceIndex = 0; surfaceIndex < surfaces.size(); ++surfaceIndex) {
      coverBoxes.clear();
      surfaces[surfaceIndex]->appendCoverBoxes(coverBoxes);
      for (const auto& coverBox : coverBoxes) {
        BVHBBox primitiveBox;
        for (int dimension = 0; dimension < 3; ++dimension) {
          primitiveBox.min[dimension] = std::nextafterf(
            static_cast<float>(component(coverBox.first, dimension) - kBVHBoxTolerance),
            -std::numeric_limits<float>::infinity());
          primitiveBox.max[dimension] = std::nextafterf(
            static_cast<float>(component(coverBox.second, dimension) + kBVHBoxTolerance),
            std::numeric_limits<float>::infinity());
        }
        primitiveBoxes.push_back(primitiveBox);
        primitiveCenters.emplace_back(primitiveBox.get_center());
        coverSurface.push_back(static_cast<int>(surfaceIndex));
      }
    }

    typename bvh::v2::DefaultBuilder<BVHNode>::Config config;
    config.quality = bvh::v2::DefaultBuilder<BVHNode>::Quality::High;
    // One cover box per leaf: bvh2 enters a leaf without a box test, and a patch intersection costs far more than one.
    config.max_leaf_size = 1;
    bvh = std::make_unique<BVH>(bvh::v2::DefaultBuilder<BVHNode>::build(primitiveBoxes, primitiveCenters, config));
    leafSurface.resize(bvh->prim_ids.size());
    for (size_t leaf = 0; leaf < bvh->prim_ids.size(); ++leaf) {
      leafSurface[leaf] = coverSurface[bvh->prim_ids[leaf]];
    }
  }

  /// The surface a BVH leaf primitive belongs to.
  size_t surfaceOfPrimitive(size_t primitive) const
  {
    return static_cast<size_t>(leafSurface[primitive]);
  }

  /// Subsample the display vertices, which lie on their patches, as safety anchors.
  void collectSafetyAnchors()
  {
    constexpr size_t kAnchorCount = 24;
    safetyAnchors.clear();
    if (displayVertices.empty()) {
      return;
    }
    const size_t stride = std::max<size_t>(1, displayVertices.size() / kAnchorCount);
    for (size_t index = 0; index < displayVertices.size() && safetyAnchors.size() < kAnchorCount; index += stride) {
      safetyAnchors.push_back(displayVertices[index]);
    }
  }

  /// The squared distance to the nearest safety anchor, inflated by a hair: an upper bound on the exact answer.
  /// It prunes only nodes that cannot win, so the value and index stay the loop's; infinity without anchors.
  double anchorSeedDistanceSq(const Vec3& point) const
  {
    double bestDistanceSq = std::numeric_limits<double>::infinity();
    for (const auto& anchor : safetyAnchors) {
      bestDistanceSq = std::min(bestDistanceSq, normSq(point - anchor));
    }
    if (!std::isfinite(bestDistanceSq)) {
      return bestDistanceSq;
    }
    // the relative term dominates the roundings, the absolute one the anchors' on-patch tolerance; both far below kBVHBoxTolerance
    const double inflated = std::sqrt(bestDistanceSq) * (1. + 1.e-12) + 1.e-10;
    return inflated * inflated;
  }

  /// Visit every surface one of whose cover-box leaves is traversed by the (unbounded) ray,
  /// each exactly once however many of its boxes the ray crosses.
  template <typename SurfaceVisitor>
  void visitRayCandidates(const Vec3& rayOrigin, const Vec3& rayDirection, SurfaceVisitor&& visitor) const
  {
    BVHRay ray(BVHVec3(rayOrigin.xCoord, rayOrigin.yCoord, rayOrigin.zCoord),
               BVHVec3(rayDirection.xCoord, rayDirection.yCoord, rayDirection.zCoord), 0.f,
               std::numeric_limits<BVHScalar>::max());
    static constexpr bool useRobustTraversal = true;
    static thread_local bvh::v2::GrowingStack<BVH::Index> stack;
    stack.clear();
    SurfaceVisitMarker marker(surfaces.size());
    bvh->intersect<false, useRobustTraversal>(ray, bvh->get_root().index, stack,
                                              [&](size_t beginPrimitive, size_t endPrimitive) {
                                                for (size_t primitive = beginPrimitive; primitive < endPrimitive;
                                                     ++primitive) {
                                                  const size_t surfaceIndex = surfaceOfPrimitive(primitive);
                                                  if (marker.firstVisit(surfaceIndex)) {
                                                    visitor(*surfaces[surfaceIndex]);
                                                  }
                                                }
                                                return false; // keep traversing
                                              });
  }

  /// Distance to the nearest entering (\a wantEntering) or exiting crossing within \a stepmax, else Big.
  /// The ray bound shrinks to the best candidate, rounded up past kBVHBoxTolerance, so no nearer hit is cut.
  template <bool wantEntering>
  double nearestCrossing(const Vec3& rayOrigin, const Vec3& rayDirection, double stepmax) const
  {
    static thread_local std::vector<RayHit> collectedHits;
    constexpr CrossingSense wanted = wantEntering ? CrossingSense::Entering : CrossingSense::Exiting;

    // Hits are classified with their neighbours, since a graze crosses nothing. If pruning stopped at a candidate
    // that turns out to be a graze, redo the query without pruning: both passes must return the same number.
    long long candidates = 0;
    for (int attempt = 0; attempt < 2; ++attempt) {
      const bool pruning = gRayTMaxPruning && attempt == 0;
      collectedHits.clear();

      double bestCandidate = TGeoShape::Big();
      BVHRay ray(BVHVec3(rayOrigin.xCoord, rayOrigin.yCoord, rayOrigin.zCoord),
                 BVHVec3(rayDirection.xCoord, rayDirection.yCoord, rayDirection.zCoord), 0.f,
                 truncateRoundUp(stepmax));
      static constexpr bool useRobustTraversal = true;

      static thread_local bvh::v2::GrowingStack<BVH::Index> stack;
      stack.clear();
      SurfaceVisitMarker marker(surfaces.size());
      // ray is captured by reference on purpose: bvh2 takes it as const Ray&, but the object
      // itself is ours and mutable, and the traversal reads tmax afresh at every node test.
      bvh->intersect<false, useRobustTraversal>(
        ray, bvh->get_root().index, stack, [&](size_t beginPrimitive, size_t endPrimitive) {
          for (size_t primitive = beginPrimitive; primitive < endPrimitive; ++primitive) {
            const size_t surfaceIndex = surfaceOfPrimitive(primitive);
            if (!marker.firstVisit(surfaceIndex)) {
              continue;
            }
            const BoundedSurface& surface = *surfaces[surfaceIndex];
            ++candidates;
            // the per-surface bound keeps a margin past the candidate, so its cluster partners are never cut
            const double bound =
              pruning ? std::min(stepmax, bestCandidate + clusterMargin(bestCandidate)) : stepmax;
            const size_t firstNewHit = collectedHits.size();
            surface.appendIntersections(rayOrigin, rayDirection, kDistanceRayTolerance, bound, collectedHits);
            for (size_t hitIndex = firstNewHit; hitIndex < collectedHits.size(); ++hitIndex) {
              const RayHit& hit = collectedHits[hitIndex];
              if (crossingSense(hit, rayDirection) == wanted && hit.distance < bestCandidate) {
                bestCandidate = hit.distance;
              }
            }
          }
          if (pruning && bestCandidate < stepmax) {
            ray.tmax = std::min(ray.tmax, truncateRoundUp(bestCandidate + kBVHBoxTolerance));
          }
          return false; // keep traversing; the shrunk tmax does the pruning
        });

      bool grazedFirst = false;
      const double distance = nearestCrossingInHits<wantEntering>(collectedHits, rayDirection, grazedFirst);
      if (!pruning || !grazedFirst) {
        gRayCandidateCount += candidates;
        return distance;
      }
    }
    gRayCandidateCount += candidates;
    return TGeoShape::Big(); // unreachable: the second attempt never prunes
  }

  /// Same query without the BVH: visit every surface. Oracle and baseline for nearestCrossing.
  template <bool wantEntering>
  double nearestCrossingLoop(const Vec3& rayOrigin, const Vec3& rayDirection, double stepmax) const
  {
    static thread_local std::vector<RayHit> collectedLoopHits;

    // No pruning at all here: this is the oracle the accelerated query is checked against, so it
    // trades the shrinking upper bound for having every hit in hand and needing no retry.
    collectedLoopHits.clear();
    for (const auto& surface : surfaces) {
      surface->appendIntersections(rayOrigin, rayDirection, kDistanceRayTolerance, stepmax, collectedLoopHits);
    }
    bool grazedFirst = false;
    return nearestCrossingInHits<wantEntering>(collectedLoopHits, rayDirection, grazedFirst);
  }

  /// Parity of the ray's crossings with the surface set, through the BVH or the loop; \a ambiguous reports a trim-band tie-break.
  bool parityAlong(const Vec3& point, const Vec3& direction, bool useBVH, bool* ambiguous = nullptr) const
  {
    // reused across calls so containment allocates nothing on the hot path; the capacity is paid
    // once per thread. Distinct from the distance queries' buffers, which are their own.
    static thread_local std::vector<RayHit> parityHits;
    parityHits.clear();
    if (useBVH) {
      visitRayCandidates(point, direction, [&](const BoundedSurface& surface) {
        surface.appendIntersections(point, direction, kRayTolerance, TGeoShape::Big(), parityHits);
      });
    } else {
      for (const auto& surface : surfaces) {
        surface->appendIntersections(point, direction, kRayTolerance, TGeoShape::Big(), parityHits);
      }
    }
    if (ambiguous != nullptr) {
      *ambiguous = std::any_of(parityHits.begin(), parityHits.end(),
                               [](const RayHit& hit) { return hit.onTrimBoundary; });
    }
    return oddCrossingParity(parityHits, direction);
  }

  /// Containment by majority vote over reshootDirections() for a solid that is not a closed 2-manifold; stops at a majority.
  /// \a allTiedOnBoundary reports that no direction's parity rested on the geometry alone.
  bool containsByVote(const Vec3& point, bool useBVH, bool* allTiedOnBoundary = nullptr) const
  {
    constexpr int kMajority = 3; // of the five directions
    int inside = 0;              // shots whose parity rests on no trim-boundary tie-break
    int outside = 0;
    int insideOnBoundary = 0; // and shots that do, counted apart
    int outsideOnBoundary = 0;
    for (const auto& direction : reshootDirections()) {
      bool ambiguous = false;
      const bool answer = parityAlong(point, direction, useBVH, &ambiguous);
      if (ambiguous) {
        answer ? ++insideOnBoundary : ++outsideOnBoundary;
      } else {
        answer ? ++inside : ++outside;
      }
      if (inside >= kMajority || outside >= kMajority) {
        break;
      }
    }
    if (allTiedOnBoundary != nullptr) {
      *allTiedOnBoundary = (inside == outside);
    }
    // Decide among the shots that rest on the geometry unless they tie; a genuine tie counts all five.
    if (inside != outside) {
      return inside > outside;
    }
    return (inside + insideOnBoundary) > (outside + outsideOnBoundary);
  }

  /// Visit every surface whose widened leaf box holds the point, until the visitor returns true.
  template <typename SurfaceVisitor>
  bool visitPointCandidates(const Vec3& point, SurfaceVisitor&& visitor) const
  {
    const BVHVec3 testPoint(point.xCoord, point.yCoord, point.zCoord);
    SurfaceVisitMarker marker(surfaces.size());
    static thread_local std::vector<size_t> nodeStack;
    nodeStack.clear();
    nodeStack.push_back(0); // start from the root node
    while (!nodeStack.empty()) {
      const auto& node = bvh->nodes[nodeStack.back()];
      nodeStack.pop_back();
      if (!bvh::v2::extra::contains(node.get_bbox(), testPoint)) {
        continue;
      }
      if (node.is_leaf()) {
        const auto beginPrimitive = node.index.first_id();
        const auto endPrimitive = beginPrimitive + node.index.prim_count();
        for (auto primitive = beginPrimitive; primitive < endPrimitive; ++primitive) {
          const size_t surfaceIndex = surfaceOfPrimitive(primitive);
          if (marker.firstVisit(surfaceIndex) && visitor(*surfaces[surfaceIndex])) {
            return true;
          }
        }
      } else {
        const auto firstChild = node.index.first_id();
        for (size_t child : {firstChild, firstChild + 1}) {
          if (child < bvh->nodes.size()) {
            nodeStack.push_back(child);
          }
        }
      }
    }
    return false;
  }

  /// The brute-force nearest patch and its index; the lowest index wins an exact tie, which ComputeNormal relies on.
  double nearestPatchDistanceSqLoop(const Vec3& point, size_t* closestIndex) const
  {
    double bestDistanceSq = std::numeric_limits<double>::infinity();
    size_t bestIndex = surfaces.size();
    for (size_t index = 0; index < surfaces.size(); ++index) {
      const double patchDistanceSq = surfaces[index]->distanceSqToPatch(point);
      if (patchDistanceSq < bestDistanceSq) {
        bestDistanceSq = patchDistanceSq;
        bestIndex = index;
      }
    }
    if (closestIndex != nullptr) {
      *closestIndex = bestIndex;
    }
    return bestDistanceSq;
  }

  /// Same answer as nearestPatchDistanceSqLoop through the BVH: an ordered descent with a running best.
  /// Box and patch distances both err downward, so Safety can only be too small; \a TrackIndex keeps ties for ComputeNormal.
  template <bool TrackIndex>
  double nearestPatchDistanceSq(const Vec3& point, size_t* closestIndex) const
  {
    if (bvh == nullptr) {
      return nearestPatchDistanceSqLoop(point, closestIndex);
    }

    struct StackEntry {
      size_t node;
      double lowerBoundSq;
    };
    // reused across calls so the hot path allocates nothing; capacity is paid once per thread
    static thread_local std::vector<StackEntry> nodeStack;
    nodeStack.clear();

    // Seed the running best with the anchor distance; it never displaces the true winner (see anchorSeedDistanceSq).
    double bestDistanceSq = anchorSeedDistanceSq(point);
    size_t bestIndex = surfaces.size();
    SurfaceVisitMarker marker(surfaces.size());
    const bool unsound = gSafetyBoundUnsound;
    long long candidates = 0;

    auto pruned = [](double lowerBoundSq, double bestSoFarSq) {
      return TrackIndex ? lowerBoundSq > bestSoFarSq : lowerBoundSq >= bestSoFarSq;
    };

    nodeStack.push_back({0, boxDistanceSq(bvh->nodes[0].get_bbox(), point, unsound)});
    while (!nodeStack.empty()) {
      const StackEntry entry = nodeStack.back();
      nodeStack.pop_back();
      if (pruned(entry.lowerBoundSq, bestDistanceSq)) {
        continue;
      }
      const auto& node = bvh->nodes[entry.node];
      if (node.is_leaf()) {
        const auto beginPrimitive = node.index.first_id();
        const auto endPrimitive = beginPrimitive + node.index.prim_count();
        for (auto primitive = beginPrimitive; primitive < endPrimitive; ++primitive) {
          const size_t surfaceIndex = surfaceOfPrimitive(primitive);
          if (!marker.firstVisit(surfaceIndex)) {
            continue;
          }
          ++candidates;
          const double patchDistanceSq = surfaces[surfaceIndex]->distanceSqToPatch(point);
          if (patchDistanceSq < bestDistanceSq) {
            bestDistanceSq = patchDistanceSq;
            bestIndex = surfaceIndex;
          } else if (TrackIndex && patchDistanceSq == bestDistanceSq && surfaceIndex < bestIndex) {
            bestIndex = surfaceIndex;
          }
        }
        continue;
      }
      const size_t firstChild = node.index.first_id();
      const size_t secondChild = firstChild + 1;
      if (secondChild >= bvh->nodes.size()) {
        if (firstChild < bvh->nodes.size()) {
          nodeStack.push_back({firstChild, boxDistanceSq(bvh->nodes[firstChild].get_bbox(), point, unsound)});
        }
        continue;
      }
      double nearBound = boxDistanceSq(bvh->nodes[firstChild].get_bbox(), point, unsound);
      double farBound = boxDistanceSq(bvh->nodes[secondChild].get_bbox(), point, unsound);
      size_t nearChild = firstChild;
      size_t farChild = secondChild;
      if (farBound < nearBound) {
        std::swap(nearBound, farBound);
        std::swap(nearChild, farChild);
      }
      // farther child first: the stack is LIFO, so the nearer one is popped -- and tightens the
      // best -- before the farther one is re-tested
      if (!pruned(farBound, bestDistanceSq)) {
        nodeStack.push_back({farChild, farBound});
      }
      if (!pruned(nearBound, bestDistanceSq)) {
        nodeStack.push_back({nearChild, nearBound});
      }
    }

    gSafetyCandidateCount += candidates;
    if (closestIndex != nullptr) {
      *closestIndex = bestIndex;
    }
    return bestDistanceSq;
  }

  /// True, after reporting it for \a method of \a owner, if the shape is defined and takes no more surfaces.
  bool refuseIfDefined(const O2BVHSurfaceSolid& owner, const char* method) const
  {
    if (!defined) {
      return false;
    }
    owner.Error(method, "Shape %s already fully defined. Not adding", owner.GetName());
    return true;
  }

  /// Append a built surface, and to \a records the record that rebuilds it.
  bool commit(std::unique_ptr<BoundedSurface> surface, BVHSurfaceRecord record,
              std::vector<BVHSurfaceRecord>& records)
  {
    records.push_back(std::move(record));
    surfaces.emplace_back(std::move(surface));
    return true;
  }
};

int BVHSurfaceRecord::expectedScalarCount(int recordKind)
{
  switch (recordKind) {
    case PlanarPolygon:
    case CurvedPlanar:
      return 0;
    case Cylindrical: // radius, heightMin, heightMax, phiStart, phiSweep
    case Spherical:   // radius, thetaMin, thetaMax, phiStart, phiSweep
      return 5;
    case Conical:  // radiusAtMin, radiusAtMax, heightMin, heightMax, phiStart, phiSweep
    case Toroidal: // majorRadius, minorRadius, phiStart, phiSweep, tubeStart, tubeSweep
      return 6;
    default:
      return -1;
  }
}

O2BVHSurfaceSolid::O2BVHSurfaceSolid() : TGeoBBox(), fImpl(new Impl)
{
}

O2BVHSurfaceSolid::O2BVHSurfaceSolid(const char* name) : TGeoBBox(name, 0., 0., 0.), fImpl(new Impl)
{
}

O2BVHSurfaceSolid::~O2BVHSurfaceSolid()
{
  delete fImpl;
}

bool O2BVHSurfaceSolid::AddPlanarSurface(const Point3D& origin, const Point3D& axisU, const Point3D& axisV,
                                         const std::vector<Point2D>& outerWire,
                                         const std::vector<std::vector<Point2D>>& innerWires)
{
  if (fImpl->refuseIfDefined(*this, "AddPlanarSurface")) {
    return false;
  }

  std::vector<Vec2> convertedOuterWire;
  convertedOuterWire.reserve(outerWire.size());
  for (const auto& vertex : outerWire) {
    convertedOuterWire.push_back(makeVec2(vertex));
  }

  std::vector<std::vector<Vec2>> convertedInnerWires;
  convertedInnerWires.reserve(innerWires.size());
  for (const auto& innerWire : innerWires) {
    auto& convertedInnerWire = convertedInnerWires.emplace_back();
    convertedInnerWire.reserve(innerWire.size());
    for (const auto& vertex : innerWire) {
      convertedInnerWire.push_back(makeVec2(vertex));
    }
  }

  auto surface = std::make_unique<PlanarBoundedSurface>();
  std::string errorMessage;
  if (!surface->initialize(makeVec3(origin), makeVec3(axisU), makeVec3(axisV), convertedOuterWire, convertedInnerWires,
                           errorMessage)) {
    Error("AddPlanarSurface", "%s", errorMessage.c_str());
    return false;
  }
  if (surface->wasReoriented()) {
    Warning("AddPlanarSurface", "Shape %s: planar surface %d had a wire re-oriented to match its role", GetName(),
            static_cast<int>(fImpl->surfaces.size()));
  }

  auto record = makeRecord(BVHSurfaceRecord::PlanarPolygon, origin, axisU, axisV, {}, false, false);
  storePolygonWires(record, outerWire, innerWires);
  return fImpl->commit(std::move(surface), std::move(record), fRecords);
}

namespace
{
/// Translate a public PlanarBoundaryCurve wire into the internal Curve2D loop.
std::vector<Curve2D> makeCurveWire(const std::vector<O2BVHSurfaceSolid::PlanarBoundaryCurve>& wire)
{
  std::vector<Curve2D> curves;
  curves.reserve(wire.size());
  for (const auto& c : wire) {
    if (c.kind == O2BVHSurfaceSolid::PlanarBoundaryCurve::Arc) {
      curves.push_back(Curve2D::makeArc({c.center[0], c.center[1]}, c.radius, c.startAngle, c.endAngle));
    } else if (c.kind == O2BVHSurfaceSolid::PlanarBoundaryCurve::BSpline) {
      std::vector<Vec2> poles;
      poles.reserve(c.poles.size());
      for (const auto& pole : c.poles) {
        poles.push_back({pole[0], pole[1]});
      }
      curves.push_back(Curve2D::makeBSpline(c.degree, std::move(poles), c.weights, c.knots));
    } else {
      curves.push_back(Curve2D::makeLine({c.lineStart[0], c.lineStart[1]}, {c.lineEnd[0], c.lineEnd[1]}));
    }
  }
  return curves;
}

/// Translate public PlanarBoundaryCurve wires into internal Curve2D loops.
std::vector<std::vector<Curve2D>> makeCurveWires(
  const std::vector<std::vector<O2BVHSurfaceSolid::PlanarBoundaryCurve>>& wires)
{
  std::vector<std::vector<Curve2D>> loops;
  loops.reserve(wires.size());
  for (const auto& wire : wires) {
    loops.push_back(makeCurveWire(wire));
  }
  return loops;
}
} // namespace

bool O2BVHSurfaceSolid::AddCurvedPlanarSurface(const Point3D& origin, const Point3D& axisU, const Point3D& axisV,
                                               const std::vector<PlanarBoundaryCurve>& outerWire,
                                               const std::vector<std::vector<PlanarBoundaryCurve>>& innerWires)
{
  if (fImpl->refuseIfDefined(*this, "AddCurvedPlanarSurface")) {
    return false;
  }

  const std::vector<Curve2D> outerCurves = makeCurveWire(outerWire);
  const std::vector<std::vector<Curve2D>> innerCurves = makeCurveWires(innerWires);

  auto surface = std::make_unique<CurvedPlanarBoundedSurface>();
  std::string errorMessage;
  if (!surface->initialize(makeVec3(origin), makeVec3(axisU), makeVec3(axisV), outerCurves, innerCurves,
                           errorMessage, wireJoinToleranceFor(fModelTolerance))) {
    Error("AddCurvedPlanarSurface", "%s", errorMessage.c_str());
    return false;
  }

  auto record = makeRecord(BVHSurfaceRecord::CurvedPlanar, origin, axisU, axisV, {}, false, false);
  storeCurveWires(record, outerWire, innerWires);
  return fImpl->commit(std::move(surface), std::move(record), fRecords);
}

bool O2BVHSurfaceSolid::AddCylindricalSurface(const Point3D& centerPoint, const Point3D& axis,
                                              const Point3D& referenceAxisU, double radius, double heightMin,
                                              double heightMax, double phiStart, double phiSweep, bool innerWall)
{
  if (fImpl->refuseIfDefined(*this, "AddCylindricalSurface")) {
    return false;
  }

  auto surface = std::make_unique<CylindricalBoundedSurface>();
  std::string errorMessage;
  if (!surface->initialize(makeVec3(centerPoint), makeVec3(axis), makeVec3(referenceAxisU), radius, heightMin,
                           heightMax, phiStart, phiSweep, innerWall, errorMessage)) {
    Error("AddCylindricalSurface", "%s", errorMessage.c_str());
    return false;
  }

  return fImpl->commit(std::move(surface),
                       makeRecord(BVHSurfaceRecord::Cylindrical, centerPoint, axis, referenceAxisU,
                                  {radius, heightMin, heightMax, phiStart, phiSweep}, innerWall, false),
                       fRecords);
}

bool O2BVHSurfaceSolid::AddCylindricalSurface(const Point3D& centerPoint, const Point3D& axis,
                                              const Point3D& referenceAxisU, double radius, double heightMin,
                                              double heightMax, double phiStart, double phiSweep, bool innerWall,
                                              const std::vector<PlanarBoundaryCurve>& outerTrim,
                                              const std::vector<std::vector<PlanarBoundaryCurve>>& innerTrims)
{
  if (fImpl->refuseIfDefined(*this, "AddCylindricalSurface")) {
    return false;
  }

  const std::vector<Curve2D> outerCurves = makeCurveWire(outerTrim);
  const std::vector<std::vector<Curve2D>> innerCurves = makeCurveWires(innerTrims);

  auto surface = std::make_unique<CylindricalBoundedSurface>();
  std::string errorMessage;
  if (!surface->initialize(makeVec3(centerPoint), makeVec3(axis), makeVec3(referenceAxisU), radius, heightMin,
                           heightMax, phiStart, phiSweep, innerWall, outerCurves, innerCurves, errorMessage,
                           wireJoinToleranceFor(fModelTolerance))) {
    Error("AddCylindricalSurface", "%s", errorMessage.c_str());
    return false;
  }

  auto record = makeRecord(BVHSurfaceRecord::Cylindrical, centerPoint, axis, referenceAxisU,
                           {radius, heightMin, heightMax, phiStart, phiSweep}, innerWall, true);
  storeCurveWires(record, outerTrim, innerTrims);
  return fImpl->commit(std::move(surface), std::move(record), fRecords);
}

bool O2BVHSurfaceSolid::AddSphericalSurface(const Point3D& center, const Point3D& polarAxis,
                                            const Point3D& referenceAxisU, double radius, double thetaMin,
                                            double thetaMax, double phiStart, double phiSweep, bool innerWall)
{
  if (fImpl->refuseIfDefined(*this, "AddSphericalSurface")) {
    return false;
  }

  auto surface = std::make_unique<SphericalBoundedSurface>();
  std::string errorMessage;
  if (!surface->initialize(makeVec3(center), makeVec3(polarAxis), makeVec3(referenceAxisU), radius, thetaMin,
                           thetaMax, phiStart, phiSweep, innerWall, errorMessage)) {
    Error("AddSphericalSurface", "%s", errorMessage.c_str());
    return false;
  }

  return fImpl->commit(std::move(surface),
                       makeRecord(BVHSurfaceRecord::Spherical, center, polarAxis, referenceAxisU,
                                  {radius, thetaMin, thetaMax, phiStart, phiSweep}, innerWall, false),
                       fRecords);
}

bool O2BVHSurfaceSolid::AddSphericalSurface(const Point3D& center, const Point3D& polarAxis,
                                            const Point3D& referenceAxisU, double radius, double thetaMin,
                                            double thetaMax, double phiStart, double phiSweep, bool innerWall,
                                            const std::vector<PlanarBoundaryCurve>& outerTrim,
                                            const std::vector<std::vector<PlanarBoundaryCurve>>& innerTrims)
{
  if (fImpl->refuseIfDefined(*this, "AddSphericalSurface")) {
    return false;
  }

  const std::vector<Curve2D> outerCurves = makeCurveWire(outerTrim);
  const std::vector<std::vector<Curve2D>> innerCurves = makeCurveWires(innerTrims);

  auto surface = std::make_unique<SphericalBoundedSurface>();
  std::string errorMessage;
  if (!surface->initialize(makeVec3(center), makeVec3(polarAxis), makeVec3(referenceAxisU), radius, thetaMin,
                           thetaMax, phiStart, phiSweep, innerWall, outerCurves, innerCurves, errorMessage,
                           wireJoinToleranceFor(fModelTolerance))) {
    Error("AddSphericalSurface", "%s", errorMessage.c_str());
    return false;
  }

  auto record = makeRecord(BVHSurfaceRecord::Spherical, center, polarAxis, referenceAxisU,
                           {radius, thetaMin, thetaMax, phiStart, phiSweep}, innerWall, true);
  storeCurveWires(record, outerTrim, innerTrims);
  return fImpl->commit(std::move(surface), std::move(record), fRecords);
}

bool O2BVHSurfaceSolid::AddConicalSurface(const Point3D& centerPoint, const Point3D& axis,
                                          const Point3D& referenceAxisU, double radiusAtMin, double radiusAtMax,
                                          double heightMin, double heightMax, double phiStart, double phiSweep,
                                          bool innerWall)
{
  if (fImpl->refuseIfDefined(*this, "AddConicalSurface")) {
    return false;
  }

  auto surface = std::make_unique<ConicalBoundedSurface>();
  std::string errorMessage;
  if (!surface->initialize(makeVec3(centerPoint), makeVec3(axis), makeVec3(referenceAxisU), radiusAtMin,
                           radiusAtMax, heightMin, heightMax, phiStart, phiSweep, innerWall, errorMessage)) {
    Error("AddConicalSurface", "%s", errorMessage.c_str());
    return false;
  }

  return fImpl->commit(std::move(surface),
                       makeRecord(BVHSurfaceRecord::Conical, centerPoint, axis, referenceAxisU,
                                  {radiusAtMin, radiusAtMax, heightMin, heightMax, phiStart, phiSweep}, innerWall,
                                  false),
                       fRecords);
}

bool O2BVHSurfaceSolid::AddConicalSurface(const Point3D& centerPoint, const Point3D& axis,
                                          const Point3D& referenceAxisU, double radiusAtMin, double radiusAtMax,
                                          double heightMin, double heightMax, double phiStart, double phiSweep,
                                          bool innerWall, const std::vector<PlanarBoundaryCurve>& outerTrim,
                                          const std::vector<std::vector<PlanarBoundaryCurve>>& innerTrims)
{
  if (fImpl->refuseIfDefined(*this, "AddConicalSurface")) {
    return false;
  }

  const std::vector<Curve2D> outerCurves = makeCurveWire(outerTrim);
  const std::vector<std::vector<Curve2D>> innerCurves = makeCurveWires(innerTrims);

  auto surface = std::make_unique<ConicalBoundedSurface>();
  std::string errorMessage;
  if (!surface->initialize(makeVec3(centerPoint), makeVec3(axis), makeVec3(referenceAxisU), radiusAtMin,
                           radiusAtMax, heightMin, heightMax, phiStart, phiSweep, innerWall, outerCurves,
                           innerCurves, errorMessage, wireJoinToleranceFor(fModelTolerance))) {
    Error("AddConicalSurface", "%s", errorMessage.c_str());
    return false;
  }

  auto record = makeRecord(BVHSurfaceRecord::Conical, centerPoint, axis, referenceAxisU,
                           {radiusAtMin, radiusAtMax, heightMin, heightMax, phiStart, phiSweep}, innerWall, true);
  storeCurveWires(record, outerTrim, innerTrims);
  return fImpl->commit(std::move(surface), std::move(record), fRecords);
}

bool O2BVHSurfaceSolid::AddToroidalSurface(const Point3D& centerPoint, const Point3D& axis,
                                           const Point3D& referenceAxisU, double majorRadius, double minorRadius,
                                           double phiStart, double phiSweep, double tubeStart, double tubeSweep,
                                           bool innerWall)
{
  if (fImpl->refuseIfDefined(*this, "AddToroidalSurface")) {
    return false;
  }

  auto surface = std::make_unique<TorusBoundedSurface>();
  std::string errorMessage;
  if (!surface->initialize(makeVec3(centerPoint), makeVec3(axis), makeVec3(referenceAxisU), majorRadius, minorRadius,
                           phiStart, phiSweep, tubeStart, tubeSweep, innerWall, errorMessage)) {
    Error("AddToroidalSurface", "%s", errorMessage.c_str());
    return false;
  }

  return fImpl->commit(std::move(surface),
                       makeRecord(BVHSurfaceRecord::Toroidal, centerPoint, axis, referenceAxisU,
                                  {majorRadius, minorRadius, phiStart, phiSweep, tubeStart, tubeSweep}, innerWall,
                                  false),
                       fRecords);
}

bool O2BVHSurfaceSolid::AddToroidalSurface(const Point3D& centerPoint, const Point3D& axis,
                                           const Point3D& referenceAxisU, double majorRadius, double minorRadius,
                                           double phiStart, double phiSweep, double tubeStart, double tubeSweep,
                                           bool innerWall, const std::vector<PlanarBoundaryCurve>& outerTrim,
                                           const std::vector<std::vector<PlanarBoundaryCurve>>& innerTrims)
{
  if (fImpl->refuseIfDefined(*this, "AddToroidalSurface")) {
    return false;
  }

  const std::vector<Curve2D> outerCurves = makeCurveWire(outerTrim);
  const std::vector<std::vector<Curve2D>> innerCurves = makeCurveWires(innerTrims);

  auto surface = std::make_unique<TorusBoundedSurface>();
  std::string errorMessage;
  if (!surface->initialize(makeVec3(centerPoint), makeVec3(axis), makeVec3(referenceAxisU), majorRadius, minorRadius,
                           phiStart, phiSweep, tubeStart, tubeSweep, innerWall, outerCurves, innerCurves,
                           errorMessage, wireJoinToleranceFor(fModelTolerance))) {
    Error("AddToroidalSurface", "%s", errorMessage.c_str());
    return false;
  }

  auto record = makeRecord(BVHSurfaceRecord::Toroidal, centerPoint, axis, referenceAxisU,
                           {majorRadius, minorRadius, phiStart, phiSweep, tubeStart, tubeSweep}, innerWall, true);
  storeCurveWires(record, outerTrim, innerTrims);
  return fImpl->commit(std::move(surface), std::move(record), fRecords);
}

void O2BVHSurfaceSolid::CloseShape(bool check)
{
  // An empty surface set is unknown, not closed: leave it undefined (Undetermined) and keep the streamed bounding box.
  if (fImpl->surfaces.empty()) {
    Error("CloseShape", "Shape %s has no bounded surfaces; it stays undefined and reports itself not navigable",
          GetName());
    return;
  }

  ComputeBBox();

  // the display mesh feeds the safety anchors, so it is assembled before the BVH machinery
  fImpl->displayVertices.clear();
  fImpl->displayTriangles.clear();
  fImpl->displayTriangleSurface.clear();
  for (size_t surfaceIndex = 0; surfaceIndex < fImpl->surfaces.size(); ++surfaceIndex) {
    fImpl->surfaces[surfaceIndex]->appendDisplayMesh(fImpl->displayVertices, fImpl->displayTriangles);
    // resize() only writes the entries it adds, so each surface stamps exactly its own triangles.
    fImpl->displayTriangleSurface.resize(fImpl->displayTriangles.size(), static_cast<int>(surfaceIndex));
  }
  fImpl->collectSafetyAnchors();
  fImpl->buildBVH();

  fImpl->closure = validateClosure(fImpl->surfaces, fModelTolerance);
  fImpl->rimReports.clear();
  fImpl->rimReports.reserve(fImpl->closure.rimRecords.size());
  for (const RimRecord& record : fImpl->closure.rimRecords) {
    RimReport report;
    report.surface = record.surfaceIndex;
    report.rimOnSurface = record.rimIndexOnSurface;
    report.closed = record.closed;
    report.chords = record.chords;
    report.unmatchedChords = record.unmatchedChords;
    report.length = record.length;
    report.unmatchedLength = record.unmatchedLength;
    report.maxIsolation = record.maxIsolation;
    report.maxIsolationPoint = {record.maxIsolationPoint.xCoord, record.maxIsolationPoint.yCoord,
                                record.maxIsolationPoint.zCoord};
    report.maxIsolationFace = record.maxIsolationFace;
    report.state = rimStateToReliability(record.state);
    fImpl->rimReports.push_back(report);
  }
  fImpl->defined = true;
  fImpl->reliable = GetNavigationReliability() == NavigationReliability::Reliable;

  if (check) {
    const auto& closure = fImpl->closure;
    // State the consequence, not only the counts.
    if (closure.edgeIdentityAvailable && closure.boundaryRims > 0) {
      // counted by edge identity, so it says a face is missing
      Error("CloseShape",
            "Shape %s is NOT a closed surface: %d of its %d source edge(s) have only one face and %d more than two, "
            "leaving %d of %d trim loop(s) open; navigation is unreliable, see GetRimReports().",
            GetName(), closure.edgeBoundaryCount, closure.edgeIncidences, closure.edgeNonManifoldCount,
            closure.boundaryRims, closure.rims);
    } else if (closure.boundaryRims > 0) {
      Error("CloseShape",
            "Shape %s is NOT a closed surface: %d of %d trim loop(s) have no neighbouring face within %g cm, leaving "
            "%g cm of %g cm of boundary open (loneliest chord %g cm); navigation is unreliable, see GetRimReports().",
            GetName(), closure.boundaryRims, closure.rims, closure.rimEpsilon, closure.unmatchedRimLength,
            closure.totalRimLength, closure.maxRimIsolation);
    }
    if (closure.nonManifoldRims > 0) {
      Error("CloseShape",
            "Shape %s is NOT a 2-manifold: %d of %d trim loop(s) run along two or more other faces; navigation is "
            "unreliable, see GetRimReports().",
            GetName(), closure.nonManifoldRims, closure.rims);
    }
    if (!closure.orientationConsistent) {
      Error("CloseShape",
            "Shape %s has %d inconsistently oriented (reversed) trim loop(s); navigation is unreliable, see "
            "GetRimReports().",
            GetName(), closure.reversedRims);
    }
    if (closure.closed && closure.signedVolume < 0.) {
      Warning("CloseShape",
              "Shape %s has inward-pointing surface normals (signed volume %g); navigation expects outward normals",
              GetName(), closure.signedVolume);
    }
  }
}

int O2BVHSurfaceSolid::GetNsurfaces() const
{
  return static_cast<int>(fImpl->surfaces.size());
}

bool O2BVHSurfaceSolid::IsDefined() const
{
  return fImpl->defined;
}

void O2BVHSurfaceSolid::SetModelTolerance(double toleranceCm)
{
  if (!(toleranceCm >= 0.) || !std::isfinite(toleranceCm)) {
    Error("SetModelTolerance", "Shape %s: ignoring a non-finite or negative model tolerance %g; it stays %g",
          GetName(), toleranceCm, fModelTolerance);
    return;
  }
  fModelTolerance = toleranceCm;
}

bool O2BVHSurfaceSolid::HasBVH() const
{
  return fImpl->bvh != nullptr;
}

bool O2BVHSurfaceSolid::GetBVHRootBounds(Point3D& lower, Point3D& upper) const
{
  if (!HasBVH()) {
    return false;
  }
  const auto rootBox = fImpl->bvh->get_root().get_bbox();
  for (int dimension = 0; dimension < 3; ++dimension) {
    lower[dimension] = rootBox.min[dimension];
    upper[dimension] = rootBox.max[dimension];
  }
  return true;
}

int O2BVHSurfaceSolid::CountBVHRayCandidates(const Point3D& point, const Point3D& direction) const
{
  if (!HasBVH()) {
    return -1;
  }
  int candidates = 0;
  fImpl->visitRayCandidates(makeVec3(point), makeVec3(direction), [&](const BoundedSurface&) { ++candidates; });
  return candidates;
}

bool O2BVHSurfaceSolid::IsClosed() const
{
  return fImpl->defined && fImpl->closure.closed;
}

bool O2BVHSurfaceSolid::IsOrientationConsistent() const
{
  return fImpl->defined && fImpl->closure.orientationConsistent;
}

O2BVHSurfaceSolid::NavigationReliability O2BVHSurfaceSolid::GetNavigationReliability() const
{
  if (!fImpl->defined) {
    return NavigationReliability::Undetermined;
  }
  // the worst defect wins; the enum is ordered by severity
  const auto& closure = fImpl->closure;
  // With edge identities their counts are the verdict, read directly so that faces without rims still report.
  if (closure.edgeIdentityAvailable) {
    if (closure.edgeNonManifoldCount > 0) {
      return NavigationReliability::NonManifold;
    }
    if (closure.edgeBoundaryCount > 0) {
      return NavigationReliability::OpenSurfaceSet;
    }
    if (closure.edgeReversedCount > 0) {
      return NavigationReliability::ReversedFaces;
    }
    return NavigationReliability::Reliable;
  }
  if (closure.nonManifoldRims > 0) {
    return NavigationReliability::NonManifold;
  }
  if (closure.boundaryRims > 0) {
    return NavigationReliability::OpenSurfaceSet;
  }
  if (closure.reversedRims > 0) {
    return NavigationReliability::ReversedFaces;
  }
  return NavigationReliability::Reliable;
}

bool O2BVHSurfaceSolid::IsNavigable() const
{
  return GetNavigationReliability() == NavigationReliability::Reliable;
}

const char* O2BVHSurfaceSolid::GetNavigationReliabilityName(NavigationReliability reliability)
{
  switch (reliability) {
    case NavigationReliability::Undetermined:
      return "undetermined";
    case NavigationReliability::Reliable:
      return "reliable";
    case NavigationReliability::ReversedFaces:
      return "reversed-faces";
    case NavigationReliability::OpenSurfaceSet:
      return "open-surface-set";
    case NavigationReliability::NonManifold:
      return "non-manifold";
  }
  return "unknown";
}

int O2BVHSurfaceSolid::GetBoundaryEdgeCount() const
{
  return fImpl->closure.boundaryEdges;
}

int O2BVHSurfaceSolid::GetNonManifoldEdgeCount() const
{
  return fImpl->closure.nonManifoldEdges;
}

int O2BVHSurfaceSolid::GetReversedEdgeCount() const
{
  return fImpl->closure.reversedEdges;
}

double O2BVHSurfaceSolid::GetMaxRimIsolation() const
{
  return fImpl->closure.maxRimIsolation;
}

bool O2BVHSurfaceSolid::SetSurfaceBoundaryEdges(int surfaceIndex, const std::vector<unsigned int>& edgeIds,
                                                const std::vector<unsigned char>& edgeFlags)
{
  if (surfaceIndex < 0 || surfaceIndex >= static_cast<int>(fImpl->surfaces.size()) ||
      surfaceIndex >= static_cast<int>(fRecords.size())) {
    Error("SetSurfaceBoundaryEdges", "Shape %s: surface index %d is out of range (%d surface(s))", GetName(),
          surfaceIndex, GetNsurfaces());
    return false;
  }
  if (edgeIds.size() != edgeFlags.size()) {
    Error("SetSurfaceBoundaryEdges", "Shape %s: surface %d was given %d edge id(s) and %d flag(s)", GetName(),
          surfaceIndex, static_cast<int>(edgeIds.size()), static_cast<int>(edgeFlags.size()));
    return false;
  }
  std::vector<BoundedSurface::BoundaryEdgeRef> refs;
  refs.reserve(edgeIds.size());
  for (size_t index = 0; index < edgeIds.size(); ++index) {
    BoundedSurface::BoundaryEdgeRef ref;
    ref.edgeId = edgeIds[index];
    ref.reversed = (edgeFlags[index] & kEdgeReversed) != 0;
    ref.degenerate = (edgeFlags[index] & kEdgeDegenerate) != 0;
    ref.anchored = (edgeFlags[index] & kEdgeAnchored) != 0;
    refs.push_back(ref);
  }
  fImpl->surfaces[static_cast<size_t>(surfaceIndex)]->setBoundaryEdges(std::move(refs));
  fRecords[static_cast<size_t>(surfaceIndex)].boundaryEdgeIds = edgeIds;
  fRecords[static_cast<size_t>(surfaceIndex)].boundaryEdgeFlags = edgeFlags;
  return true;
}

bool O2BVHSurfaceSolid::HasEdgeIdentity() const
{
  return fImpl->closure.edgeIdentityAvailable;
}

int O2BVHSurfaceSolid::GetSourceEdgeCount() const
{
  return fImpl->closure.edgeIncidences;
}

int O2BVHSurfaceSolid::GetSharedSourceEdgeCount() const
{
  return fImpl->closure.edgeSharedCount;
}

int O2BVHSurfaceSolid::GetBoundarySourceEdgeCount() const
{
  return fImpl->closure.edgeBoundaryCount;
}

int O2BVHSurfaceSolid::GetNonManifoldSourceEdgeCount() const
{
  return fImpl->closure.edgeNonManifoldCount;
}

int O2BVHSurfaceSolid::GetReversedSourceEdgeCount() const
{
  return fImpl->closure.edgeReversedCount;
}

int O2BVHSurfaceSolid::GetDegenerateSourceEdgeCount() const
{
  return fImpl->closure.edgeDegenerateCount;
}

double O2BVHSurfaceSolid::GetMaxSharedEdgeDeviation() const
{
  return fImpl->closure.maxSharedEdgeDeviation;
}

int O2BVHSurfaceSolid::GetMeasuredSharedEdgeCount() const
{
  return fImpl->closure.sharedEdgesMeasured;
}

int O2BVHSurfaceSolid::GetUnmeasuredSharedEdgeCount() const
{
  return fImpl->closure.sharedEdgesUnmeasured;
}

double O2BVHSurfaceSolid::GetRimChordResolution() const
{
  return fImpl->closure.rimChordResolution;
}

double O2BVHSurfaceSolid::GetRimMatchTolerance() const
{
  return fImpl->closure.rimEpsilon;
}

double O2BVHSurfaceSolid::GetTotalRimLength() const
{
  return fImpl->closure.totalRimLength;
}

double O2BVHSurfaceSolid::GetUnmatchedRimLength() const
{
  return fImpl->closure.unmatchedRimLength;
}

int O2BVHSurfaceSolid::GetRimCount() const
{
  return fImpl->closure.rims;
}

int O2BVHSurfaceSolid::GetMatchedRimCount() const
{
  return fImpl->closure.matchedRims;
}

int O2BVHSurfaceSolid::GetBoundaryRimCount() const
{
  return fImpl->closure.boundaryRims;
}

int O2BVHSurfaceSolid::GetNonManifoldRimCount() const
{
  return fImpl->closure.nonManifoldRims;
}

int O2BVHSurfaceSolid::GetReversedRimCount() const
{
  return fImpl->closure.reversedRims;
}

const std::vector<O2BVHSurfaceSolid::RimReport>& O2BVHSurfaceSolid::GetRimReports() const
{
  return fImpl->rimReports;
}

void O2BVHSurfaceSolid::GetSurfaceCapacityContributions(std::vector<double>& contributions) const
{
  contributions.clear();
  contributions.reserve(fImpl->surfaces.size());
  for (const auto& surface : fImpl->surfaces) {
    contributions.push_back(surface == nullptr ? 0. : surface->capacityContribution());
  }
}

void O2BVHSurfaceSolid::ComputeBBox()
{
  if (fImpl->surfaces.empty()) {
    fDX = fDY = fDZ = 0.;
    fOrigin[0] = fOrigin[1] = fOrigin[2] = 0.;
    return;
  }

  Vec3 lowerCorner{TGeoShape::Big(), TGeoShape::Big(), TGeoShape::Big()};
  Vec3 upperCorner{-TGeoShape::Big(), -TGeoShape::Big(), -TGeoShape::Big()};
  for (const auto& surface : fImpl->surfaces) {
    surface->conservativeBounds(lowerCorner, upperCorner);
  }

  for (int dimension = 0; dimension < 3; ++dimension) {
    const double lowerValue = component(lowerCorner, dimension) - kTolerance;
    const double upperValue = component(upperCorner, dimension) + kTolerance;
    fOrigin[dimension] = 0.5 * (lowerValue + upperValue);
    const double halfLength = 0.5 * (upperValue - lowerValue);
    if (dimension == 0) {
      fDX = halfLength;
    } else if (dimension == 1) {
      fDY = halfLength;
    } else {
      fDZ = halfLength;
    }
  }
}

void O2BVHSurfaceSolid::GetMeshNumbers(int& nvert, int& nsegs, int& npols) const
{
  nvert = GetNmeshVertices();
  npols = static_cast<int>(fImpl->displayTriangles.size());
  nsegs = 3 * npols;
}

int O2BVHSurfaceSolid::GetNmeshVertices() const
{
  return static_cast<int>(fImpl->displayVertices.size());
}

namespace
{
/// The deterministic R2 low-discrepancy pair in [0,1)^2, so a shape's sample points depend on the shape alone.
void r2Pair(long long index, double& firstCoordinate, double& secondCoordinate)
{
  constexpr double kAlpha1 = 0.7548776662466927; // 1 / plastic number
  constexpr double kAlpha2 = 0.5698402909980532; // 1 / plastic number^2
  const double shifted = static_cast<double>(index + 1);
  firstCoordinate = std::fmod(0.5 + kAlpha1 * shifted, 1.);
  secondCoordinate = std::fmod(0.5 + kAlpha2 * shifted, 1.);
}
} // namespace

/// Newton on the patch distance along the patch normal; returns whether the point reached the surface.
bool O2BVHSurfaceSolid::ProjectOntoPatch(int surfaceIndex, double* point) const
{
  if (surfaceIndex < 0 || static_cast<size_t>(surfaceIndex) >= fImpl->surfaces.size()) {
    return false;
  }
  const BoundedSurface& surface = *fImpl->surfaces[surfaceIndex];
  constexpr double kToleranceSquared = kSurfacePointTolerance * kSurfacePointTolerance;

  Vec3 current = makeVec3(point);
  double currentDistanceSq = surface.distanceSqToPatch(current);
  for (int iteration = 0; iteration < 8 && currentDistanceSq > kToleranceSquared; ++iteration) {
    const double distance = std::sqrt(currentDistanceSq);
    const Vec3 normal = surface.normalAt(current);
    const Vec3 inward{current.xCoord - distance * normal.xCoord, current.yCoord - distance * normal.yCoord,
                      current.zCoord - distance * normal.zCoord};
    const Vec3 outward{current.xCoord + distance * normal.xCoord, current.yCoord + distance * normal.yCoord,
                       current.zCoord + distance * normal.zCoord};
    const double inwardDistanceSq = surface.distanceSqToPatch(inward);
    const double outwardDistanceSq = surface.distanceSqToPatch(outward);
    const double bestDistanceSq = std::min(inwardDistanceSq, outwardDistanceSq);
    // not converging: the nearest patch point lies on the trim wire
    if (!(bestDistanceSq < currentDistanceSq)) {
      return false;
    }
    current = (inwardDistanceSq < outwardDistanceSq) ? inward : outward;
    currentDistanceSq = bestDistanceSq;
  }

  if (currentDistanceSq > kToleranceSquared) {
    return false;
  }
  point[0] = current.xCoord;
  point[1] = current.yCoord;
  point[2] = current.zCoord;
  return true;
}

Bool_t O2BVHSurfaceSolid::GetPointsOnSegments(Int_t npoints, Double_t* array) const
{
  if (array == nullptr || npoints <= 0 || fImpl->displayVertices.empty()) {
    return kFALSE;
  }
  const int vertexCount = static_cast<int>(fImpl->displayVertices.size());
  // Below the mesh size, decline so ROOT uses SetPoints(), whose vertices all lie on patches.
  if (npoints < vertexCount) {
    return kFALSE;
  }

  auto writeVertex = [&](int slot, const Vec3& vertex) {
    array[3 * slot + 0] = vertex.xCoord;
    array[3 * slot + 1] = vertex.yCoord;
    array[3 * slot + 2] = vertex.zCoord;
  };

  for (int vertexIndex = 0; vertexIndex < vertexCount; ++vertexIndex) {
    writeVertex(vertexIndex, fImpl->displayVertices[vertexIndex]);
  }

  const int extraCount = npoints - vertexCount;
  const int triangleCount = static_cast<int>(fImpl->displayTriangles.size());
  if (extraCount == 0) {
    return kTRUE;
  }
  if (triangleCount == 0 || fImpl->displayTriangleSurface.size() != fImpl->displayTriangles.size()) {
    // No triangles (or a mesh built before the provenance existed): repeat vertices rather than
    // leave the tail of the buffer uninitialised, which the caller would read as coordinates.
    for (int extraIndex = 0; extraIndex < extraCount; ++extraIndex) {
      writeVertex(vertexCount + extraIndex, fImpl->displayVertices[extraIndex % vertexCount]);
    }
    return kTRUE;
  }

  for (int extraIndex = 0; extraIndex < extraCount; ++extraIndex) {
    // Stride over the triangles rather than walking them in order, so a request that cannot cover
    // every triangle still spreads over the whole solid instead of over its first few faces.
    const int triangleIndex =
      static_cast<int>((static_cast<long long>(extraIndex) * triangleCount) / extraCount) % triangleCount;
    const auto& triangle = fImpl->displayTriangles[triangleIndex];
    const Vec3& cornerA = fImpl->displayVertices[triangle[0]];
    const Vec3& cornerB = fImpl->displayVertices[triangle[1]];
    const Vec3& cornerC = fImpl->displayVertices[triangle[2]];

    double firstCoordinate = 0.;
    double secondCoordinate = 0.;
    r2Pair(extraIndex, firstCoordinate, secondCoordinate);
    if (firstCoordinate + secondCoordinate > 1.) {
      firstCoordinate = 1. - firstCoordinate;
      secondCoordinate = 1. - secondCoordinate;
    }
    const double weightA = 1. - firstCoordinate - secondCoordinate;
    double candidate[3] = {weightA * cornerA.xCoord + firstCoordinate * cornerB.xCoord + secondCoordinate * cornerC.xCoord,
                           weightA * cornerA.yCoord + firstCoordinate * cornerB.yCoord + secondCoordinate * cornerC.yCoord,
                           weightA * cornerA.zCoord + firstCoordinate * cornerB.zCoord + secondCoordinate * cornerC.zCoord};

    if (!ProjectOntoPatch(fImpl->displayTriangleSurface[triangleIndex], candidate)) {
      // fall back to a vertex of the sampled triangle, which is on the patch
      candidate[0] = cornerA.xCoord;
      candidate[1] = cornerA.yCoord;
      candidate[2] = cornerA.zCoord;
    }
    array[3 * (vertexCount + extraIndex) + 0] = candidate[0];
    array[3 * (vertexCount + extraIndex) + 1] = candidate[1];
    array[3 * (vertexCount + extraIndex) + 2] = candidate[2];
  }
  return kTRUE;
}

TBuffer3D* O2BVHSurfaceSolid::MakeBuffer3D() const
{
  int nvert = 0;
  int nsegs = 0;
  int npols = 0;
  GetMeshNumbers(nvert, nsegs, npols);
  auto buff = new TBuffer3D(TBuffer3DTypes::kGeneric, nvert, 3 * nvert, nsegs, 3 * nsegs, npols, 6 * npols);
  if (buff != nullptr) {
    SetPoints(buff->fPnts);
    SetSegsAndPols(*buff);
  }
  return buff;
}

void O2BVHSurfaceSolid::Print(Option_t*) const
{
  std::cout << "=== BVH surface solid " << GetName() << " having " << GetNsurfaces() << " bounded surfaces\n";
  const auto reliability = GetNavigationReliability();
  std::cout << "    navigation: " << GetNavigationReliabilityName(reliability);
  if (reliability != NavigationReliability::Reliable && reliability != NavigationReliability::Undetermined) {
    std::cout << " (UNRELIABLE; boundary=" << GetBoundaryEdgeCount() << " non-manifold=" << GetNonManifoldEdgeCount()
              << " reversed=" << GetReversedEdgeCount() << ")";
  }
  std::cout << "\n    model tolerance: ";
  if (fModelTolerance > 0.) {
    std::cout << fModelTolerance << " cm (from the source model)";
  } else {
    std::cout << "not stated";
  }
  // the identity counts first: when present they are the verdict
  if (HasEdgeIdentity()) {
    std::cout << "\n    edge identity: " << GetSourceEdgeCount() << " source edge(s), shared=" << GetSharedSourceEdgeCount()
              << " boundary=" << GetBoundarySourceEdgeCount() << " non-manifold=" << GetNonManifoldSourceEdgeCount()
              << " reversed=" << GetReversedSourceEdgeCount() << " degenerate=" << GetDegenerateSourceEdgeCount()
              << "\n    shared edge deviation: max " << GetMaxSharedEdgeDeviation() << " cm over "
              << GetMeasuredSharedEdgeCount() << " measured edge(s)";
    if (GetUnmeasuredSharedEdgeCount() > 0) {
      std::cout << " (" << GetUnmeasuredSharedEdgeCount() << " not measurable: parametric-rectangle trim)";
    }
  }
  // The isolation, and the resolution that widened the band it was judged in, always together: the
  // number is how alone the loneliest chord is, not how far apart two faces are.
  if (GetRimCount() > 0) {
    std::cout << "\n    rim isolation: max " << GetMaxRimIsolation() << " cm (chord resolution "
              << GetRimChordResolution() << " cm, declared tolerance " << GetRimMatchTolerance() << " cm)"
              << "\n    rims: " << GetRimCount() << " (matched=" << GetMatchedRimCount()
              << " boundary=" << GetBoundaryRimCount() << " non-manifold=" << GetNonManifoldRimCount()
              << " reversed=" << GetReversedRimCount() << "), open " << GetUnmatchedRimLength() << " of "
              << GetTotalRimLength() << " cm";
  }
  std::cout << "\n";
}

void O2BVHSurfaceSolid::SetPoints(double* points) const
{
  int coordinateIndex = 0;
  for (const auto& vertex : fImpl->displayVertices) {
    points[coordinateIndex++] = vertex.xCoord;
    points[coordinateIndex++] = vertex.yCoord;
    points[coordinateIndex++] = vertex.zCoord;
  }
}

void O2BVHSurfaceSolid::SetPoints(Float_t* points) const
{
  int coordinateIndex = 0;
  for (const auto& vertex : fImpl->displayVertices) {
    points[coordinateIndex++] = vertex.xCoord;
    points[coordinateIndex++] = vertex.yCoord;
    points[coordinateIndex++] = vertex.zCoord;
  }
}

void O2BVHSurfaceSolid::SetSegsAndPols(TBuffer3D& buff) const
{
  const int color = GetBasicColor();
  int* segs = buff.fSegs;
  int* pols = buff.fPols;
  int segmentDataIndex = 0;
  int polygonDataIndex = 0;
  int segmentIndex = 0;
  for (const auto& triangle : fImpl->displayTriangles) {
    pols[polygonDataIndex++] = color;
    pols[polygonDataIndex++] = 3;
    for (int triangleEdge = 0; triangleEdge < 3; ++triangleEdge) {
      const int nextTriangleEdge = (triangleEdge + 1) % 3;
      segs[segmentDataIndex++] = color;
      segs[segmentDataIndex++] = triangle[triangleEdge];
      segs[segmentDataIndex++] = triangle[nextTriangleEdge];
      pols[polygonDataIndex + 2 - triangleEdge] = segmentIndex++;
    }
    polygonDataIndex += 3;
  }
}

const TBuffer3D& O2BVHSurfaceSolid::GetBuffer3D(int reqSections, Bool_t localFrame) const
{
  static TBuffer3D buffer(TBuffer3DTypes::kGeneric);

  FillBuffer3D(buffer, reqSections, localFrame);

  int nvert = 0;
  int nsegs = 0;
  int npols = 0;
  GetMeshNumbers(nvert, nsegs, npols);

  if (reqSections & TBuffer3D::kRawSizes) {
    if (buffer.SetRawSizes(nvert, 3 * nvert, nsegs, 3 * nsegs, npols, 6 * npols)) {
      buffer.SetSectionsValid(TBuffer3D::kRawSizes);
    }
  }
  if ((reqSections & TBuffer3D::kRaw) && buffer.SectionsValid(TBuffer3D::kRawSizes)) {
    SetPoints(buffer.fPnts);
    if (!buffer.fLocalFrame) {
      TransformPoints(buffer.fPnts, buffer.NbPnts());
    }
    SetSegsAndPols(buffer);
    buffer.SetSectionsValid(TBuffer3D::kRaw);
  }

  return buffer;
}

bool O2BVHSurfaceSolid::Contains(const Double_t* point) const
{
  if (fImpl->surfaces.empty()) {
    return false;
  }

  if (fImpl->bvh == nullptr) {
    // Before CloseShape there is no BVH and no bounding box, so this fallback must come before the box check.
    return Contains_Loop(point);
  }

  const Vec3 testPoint = makeVec3(point);
  if (std::abs(testPoint.xCoord - fOrigin[0]) > fDX + kTolerance ||
      std::abs(testPoint.yCoord - fOrigin[1]) > fDY + kTolerance ||
      std::abs(testPoint.zCoord - fOrigin[2]) > fDZ + kTolerance) {
    return false;
  }

  // boundary policy: a point within tolerance of any surface patch counts as inside
  if (fImpl->visitPointCandidates(
        testPoint, [&](const BoundedSurface& surface) { return surface.containsPointOnSurface(testPoint); })) {
    return true;
  }

  return containsByParity(point, true);
}

bool O2BVHSurfaceSolid::ContainsAlongDirection(const Double_t* point, const Double_t* direction) const
{
  if (fImpl->surfaces.empty()) {
    return false;
  }
  const Vec3 testPoint = makeVec3(point);
  for (const auto& surface : fImpl->surfaces) {
    if (surface->containsPointOnSurface(testPoint)) {
      return true;
    }
  }
  return fImpl->parityAlong(testPoint, normalized(makeVec3(direction)), fImpl->bvh != nullptr);
}

bool O2BVHSurfaceSolid::Contains_Loop(const Double_t* point) const
{
  if (fImpl->surfaces.empty()) {
    return false;
  }

  const Vec3 testPoint = makeVec3(point);
  for (const auto& surface : fImpl->surfaces) {
    if (surface->containsPointOnSurface(testPoint)) {
      return true;
    }
  }

  return containsByParity(point, false);
}

bool O2BVHSurfaceSolid::containsByParity(const Double_t* point, bool useBVH) const
{
  // Reliable solid: one parity shot, unless it rests on a trim-band tie-break; otherwise a 5-direction vote.
  const Vec3 testPoint = makeVec3(point);
  if (fImpl->reliable) {
    bool ambiguous = false;
    const bool answer = fImpl->parityAlong(testPoint, kContainsTestDirection, useBVH, &ambiguous);
    if (!ambiguous) {
      return answer;
    }
    // This shot crossed a patch within its own trim accuracy, so its parity rests on a tie-break
    // rather than on the geometry. Re-aim: the sliver belongs to the ray, not to the point.
    return fImpl->containsByVote(testPoint, useBVH);
  }
  return fImpl->containsByVote(testPoint, useBVH);
}

void O2BVHSurfaceSolid::DescribeContainsCrossings(const Point3D& point,
                                                  std::vector<ContainsCrossing>& bvhCrossings,
                                                  std::vector<ContainsCrossing>& loopCrossings) const
{
  const Point3D direction{kContainsTestDirection.xCoord, kContainsTestDirection.yCoord,
                          kContainsTestDirection.zCoord};
  DescribeContainsCrossings(point, direction, bvhCrossings, loopCrossings);
}

void O2BVHSurfaceSolid::DescribeContainsCrossings(const Point3D& point, const Point3D& direction,
                                                  std::vector<ContainsCrossing>& bvhCrossings,
                                                  std::vector<ContainsCrossing>& loopCrossings) const
{
  bvhCrossings.clear();
  loopCrossings.clear();
  if (fImpl->surfaces.empty()) {
    return;
  }
  const Vec3 testPoint = makeVec3(point.data());
  const Vec3 testDirection = normalized(makeVec3(direction.data()));

  auto collect = [&](std::vector<RayHit>& hits, std::vector<ContainsCrossing>& out) {
    std::sort(hits.begin(), hits.end(),
              [](const RayHit& first, const RayHit& second) { return first.distance < second.distance; });
    out.reserve(hits.size());
    for (const auto& hit : hits) {
      out.push_back({hit.distance, dot(hit.normal, testDirection), hit.onTrimBoundary});
    }
  };

  std::vector<RayHit> loopHits;
  for (const auto& surface : fImpl->surfaces) {
    surface->appendIntersections(testPoint, testDirection, kRayTolerance, TGeoShape::Big(), loopHits);
  }
  collect(loopHits, loopCrossings);

  if (fImpl->bvh != nullptr) {
    std::vector<RayHit> bvhHits;
    fImpl->visitRayCandidates(testPoint, testDirection, [&](const BoundedSurface& surface) {
      surface.appendIntersections(testPoint, testDirection, kRayTolerance, TGeoShape::Big(), bvhHits);
    });
    collect(bvhHits, bvhCrossings);
  }
}

Double_t O2BVHSurfaceSolid::DistFromOutside(const Double_t* point, const Double_t* dir, Int_t iact, Double_t stepmax,
                                            Double_t* safe) const
{
  if (iact < 3 && safe != nullptr) {
    *safe = Safety(point, kFALSE);
    if (iact == 0) {
      return TGeoShape::Big();
    }
    if (iact == 1 && stepmax < *safe) {
      return TGeoShape::Big();
    }
  }
  if (fImpl->surfaces.empty()) {
    return TGeoShape::Big();
  }
  if (fImpl->bvh == nullptr) {
    // before CloseShape there is no acceleration structure yet; stay usable via the plain loop
    return DistFromOutside_Loop(point, dir, stepmax);
  }

  // cheap reject: a per-axis gap to the bounding box beyond stepmax means no reachable crossing
  const Double_t halfLengths[3] = {fDX, fDY, fDZ};
  for (int dimension = 0; dimension < 3; ++dimension) {
    const Double_t lower = fOrigin[dimension] - halfLengths[dimension];
    const Double_t upper = fOrigin[dimension] + halfLengths[dimension];
    if (lower - point[dimension] > stepmax + kBVHBoxTolerance ||
        point[dimension] - upper > stepmax + kBVHBoxTolerance) {
      return TGeoShape::Big();
    }
  }

  return fImpl->nearestCrossing<true>(makeVec3(point), makeVec3(dir), stepmax);
}

Double_t O2BVHSurfaceSolid::DistFromInside(const Double_t* point, const Double_t* dir, Int_t iact, Double_t stepmax,
                                           Double_t* safe) const
{
  if (iact < 3 && safe != nullptr) {
    *safe = Safety(point, kTRUE);
    if (iact == 0) {
      return TGeoShape::Big();
    }
    if (iact == 1 && stepmax < *safe) {
      return TGeoShape::Big();
    }
  }
  if (fImpl->surfaces.empty()) {
    return TGeoShape::Big();
  }
  if (fImpl->bvh == nullptr) {
    return DistFromInside_Loop(point, dir, stepmax);
  }
  // no bounding-box reject here: the point is inside by contract, so the box is always reachable
  return fImpl->nearestCrossing<false>(makeVec3(point), makeVec3(dir), stepmax);
}

Double_t O2BVHSurfaceSolid::DistFromOutside_Loop(const Double_t* point, const Double_t* dir, Double_t stepmax) const
{
  if (fImpl->surfaces.empty()) {
    return TGeoShape::Big();
  }
  return fImpl->nearestCrossingLoop<true>(makeVec3(point), makeVec3(dir), stepmax);
}

Double_t O2BVHSurfaceSolid::DistFromInside_Loop(const Double_t* point, const Double_t* dir, Double_t stepmax) const
{
  if (fImpl->surfaces.empty()) {
    return TGeoShape::Big();
  }
  return fImpl->nearestCrossingLoop<false>(makeVec3(point), makeVec3(dir), stepmax);
}

void O2BVHSurfaceSolid::SetRayTMaxPruning(bool enable)
{
  gRayTMaxPruning = enable;
}

bool O2BVHSurfaceSolid::GetRayTMaxPruning()
{
  return gRayTMaxPruning;
}

void O2BVHSurfaceSolid::ResetRayCandidateCounter()
{
  gRayCandidateCount = 0;
}

long long O2BVHSurfaceSolid::GetRayCandidateCount()
{
  return gRayCandidateCount;
}

void O2BVHSurfaceSolid::ResetSafetyCandidateCounter()
{
  gSafetyCandidateCount = 0;
}

long long O2BVHSurfaceSolid::GetSafetyCandidateCount()
{
  return gSafetyCandidateCount;
}

void O2BVHSurfaceSolid::SetSafetyBoundUnsoundForTest(bool enable)
{
  gSafetyBoundUnsound = enable;
}

bool O2BVHSurfaceSolid::GetSafetyBoundUnsoundForTest()
{
  return gSafetyBoundUnsound;
}

/// The distance to the nearest patch, rounded down by one ulp so that Safety is never too large.
Double_t O2BVHSurfaceSolid::Safety(const Double_t* point, Bool_t) const
{
  if (fImpl->surfaces.empty()) {
    return TGeoShape::Big();
  }
  const double bestDistanceSq = fImpl->nearestPatchDistanceSq<false>(makeVec3(point), nullptr);
  return std::nextafter(std::sqrt(bestDistanceSq), 0.);
}

Double_t O2BVHSurfaceSolid::Safety_Loop(const Double_t* point, Bool_t) const
{
  if (fImpl->surfaces.empty()) {
    return TGeoShape::Big();
  }
  const double bestDistanceSq = fImpl->nearestPatchDistanceSqLoop(makeVec3(point), nullptr);
  return std::nextafter(std::sqrt(bestDistanceSq), 0.);
}

void O2BVHSurfaceSolid::ComputeNormal(const Double_t* point, const Double_t* dir, Double_t* norm) const
{
  computeNormalFrom(point, dir, norm, false);
}

void O2BVHSurfaceSolid::ComputeNormal_Loop(const Double_t* point, const Double_t* dir, Double_t* norm) const
{
  computeNormalFrom(point, dir, norm, true);
}

void O2BVHSurfaceSolid::computeNormalFrom(const Double_t* point, const Double_t* dir, Double_t* norm,
                                          bool useLoop) const
{
  if (fImpl->surfaces.empty()) {
    norm[0] = 1.;
    norm[1] = 0.;
    norm[2] = 0.;
    return;
  }

  const Vec3 testPoint = makeVec3(point);
  size_t closestIndex = fImpl->surfaces.size();
  if (useLoop) {
    fImpl->nearestPatchDistanceSqLoop(testPoint, &closestIndex);
  } else {
    fImpl->nearestPatchDistanceSq<true>(testPoint, &closestIndex);
  }

  if (closestIndex >= fImpl->surfaces.size()) {
    norm[0] = 1.;
    norm[1] = 0.;
    norm[2] = 0.;
    return;
  }

  Vec3 normal = fImpl->surfaces[closestIndex]->normalAt(testPoint);
  if (dir != nullptr) {
    const Vec3 direction = makeVec3(dir);
    if (dot(normal, direction) < 0.) {
      normal = normal * -1.;
    }
  }
  norm[0] = normal.xCoord;
  norm[1] = normal.yCoord;
  norm[2] = normal.zCoord;
}

Double_t O2BVHSurfaceSolid::Capacity() const
{
  double capacity = 0.;
  for (const auto& surface : fImpl->surfaces) {
    capacity += surface->capacityContribution();
  }
  return std::abs(capacity);
}

bool O2BVHSurfaceSolid::RebuildFromRecords()
{
  // Add*Surface refuses to run on a defined shape and re-appends to fRecords as it replays, so
  // take the records aside and start from a fresh implementation.
  std::vector<BVHSurfaceRecord> records;
  records.swap(fRecords);
  delete fImpl;
  fImpl = new Impl;
  // a solid missing a face is a different solid, so a failed record discards the whole shape
  const auto discard = [this]() {
    fRecords.clear();
    delete fImpl;
    fImpl = new Impl;
    return false;
  };

  if (records.empty()) {
    Error("RebuildFromRecords", "Shape %s carries no surface records, so it stays undefined and not navigable.",
          GetName());
    return false;
  }

  for (size_t recordIndex = 0; recordIndex < records.size(); ++recordIndex) {
    const auto& record = records[recordIndex];
    const int expectedScalars = BVHSurfaceRecord::expectedScalarCount(record.kind);
    if (expectedScalars < 0 || record.scalars.size() != static_cast<size_t>(expectedScalars)) {
      Error("RebuildFromRecords", "Shape %s: surface record %d has kind %d with %d scalar(s), expected %d", GetName(),
            static_cast<int>(recordIndex), record.kind, static_cast<int>(record.scalars.size()), expectedScalars);
      return discard();
    }

    const Point3D origin = makePoint3D(record.origin);
    const Point3D axisA = makePoint3D(record.axisA);
    const Point3D axisB = makePoint3D(record.axisB);
    const auto& s = record.scalars;

    std::vector<PlanarBoundaryCurve> outerWire;
    std::vector<std::vector<PlanarBoundaryCurve>> innerWires;
    std::vector<Point2D> outerPolygon;
    std::vector<std::vector<Point2D>> innerPolygons;
    const bool wiresLoaded = record.kind == BVHSurfaceRecord::PlanarPolygon
                               ? loadPolygonWires(record, outerPolygon, innerPolygons)
                               : loadCurveWires(record, outerWire, innerWires);

    bool added = false;
    if (!wiresLoaded) {
      Error("RebuildFromRecords", "Shape %s: surface record %d has inconsistent wire sizes", GetName(),
            static_cast<int>(recordIndex));
    } else {
      switch (record.kind) {
        case BVHSurfaceRecord::PlanarPolygon:
          added = AddPlanarSurface(origin, axisA, axisB, outerPolygon, innerPolygons);
          break;
        case BVHSurfaceRecord::CurvedPlanar:
          added = AddCurvedPlanarSurface(origin, axisA, axisB, outerWire, innerWires);
          break;
        case BVHSurfaceRecord::Cylindrical:
          added = record.trimmed ? AddCylindricalSurface(origin, axisA, axisB, s[0], s[1], s[2], s[3], s[4],
                                                         record.innerWall, outerWire, innerWires)
                                 : AddCylindricalSurface(origin, axisA, axisB, s[0], s[1], s[2], s[3], s[4],
                                                         record.innerWall);
          break;
        case BVHSurfaceRecord::Spherical:
          added = record.trimmed ? AddSphericalSurface(origin, axisA, axisB, s[0], s[1], s[2], s[3], s[4],
                                                       record.innerWall, outerWire, innerWires)
                                 : AddSphericalSurface(origin, axisA, axisB, s[0], s[1], s[2], s[3], s[4],
                                                       record.innerWall);
          break;
        case BVHSurfaceRecord::Conical:
          added = record.trimmed ? AddConicalSurface(origin, axisA, axisB, s[0], s[1], s[2], s[3], s[4], s[5],
                                                     record.innerWall, outerWire, innerWires)
                                 : AddConicalSurface(origin, axisA, axisB, s[0], s[1], s[2], s[3], s[4], s[5],
                                                     record.innerWall);
          break;
        case BVHSurfaceRecord::Toroidal:
          added = record.trimmed ? AddToroidalSurface(origin, axisA, axisB, s[0], s[1], s[2], s[3], s[4], s[5],
                                                      record.innerWall, outerWire, innerWires)
                                 : AddToroidalSurface(origin, axisA, axisB, s[0], s[1], s[2], s[3], s[4], s[5],
                                                      record.innerWall);
          break;
        default:
          break;
      }
    }

    if (!added) {
      // a solid missing a face is a different solid: discard it rather than return a partial shape
      Error("RebuildFromRecords",
            "Shape %s: surface record %d (kind %d) did not rebuild, so the shape is discarded and stays undefined.",
            GetName(), static_cast<int>(recordIndex), record.kind);
      return discard();
    }

    // the edge identities are part of the record: replay them, or the read-back closure verdict could differ
    if (!record.boundaryEdgeIds.empty()) {
      SetSurfaceBoundaryEdges(static_cast<int>(recordIndex), record.boundaryEdgeIds, record.boundaryEdgeFlags);
    }
  }

  // check == false: replaying a solid must not re-emit the closure diagnostics that were already
  // reported when it was first built. The report itself is recomputed, not trusted.
  CloseShape(false);
  return true;
}

void O2BVHSurfaceSolid::Streamer(TBuffer& buffer)
{
  if (buffer.IsReading()) {
    buffer.ReadClassBuffer(O2BVHSurfaceSolid::Class(), this);
    RebuildFromRecords();
  } else {
    buffer.WriteClassBuffer(O2BVHSurfaceSolid::Class(), this);
  }
}