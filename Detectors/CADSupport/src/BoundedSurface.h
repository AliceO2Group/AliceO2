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

/// \file BoundedSurface.h
/// \brief Private analytic bounded surfaces, trim wires and closure checks behind O2BVHSurfaceSolid.

#ifndef ALICEO2_CADSUPPORT_BOUNDEDSURFACE_H_
#define ALICEO2_CADSUPPORT_BOUNDEDSURFACE_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace o2::cad::surface
{

/// \name Numerical conventions: the tolerances shared by all bounded-surface code
/// @{
inline constexpr double kTolerance = 1.e-9; ///< generic length tolerance
inline constexpr double kToleranceSq = kTolerance * kTolerance;
inline constexpr double kAreaTolerance = 1.e-18;        ///< degenerate (zero) parametric area
inline constexpr double kRayTolerance = 1.e-9;          ///< minimum positive ray parameter t
inline constexpr double kIntersectionTolerance = 1.e-7; ///< clustering of near-equal intersections
inline constexpr double kClosureQuantum = 1.e-7;        ///< vertex quantization for closure matching
/// Wire-closure tolerance, a 3D length in cm through the surface metric: the CAD extractor's endpoint precision.
inline constexpr double kWireJoinTolerance = 1.e-6;
/// The wire-join band for a model with a declared tolerance: that tolerance when looser than kWireJoinTolerance, else the floor.
inline constexpr double wireJoinToleranceFor(double modelTolerance)
{
  return modelTolerance > kWireJoinTolerance ? modelTolerance : kWireJoinTolerance;
}
/// Chord flatness of the adaptive B-spline sampler, in the curve's parametric units; a B-spline trim is this polyline.
inline constexpr double kBSplineFlatness = 1.e-5;
inline constexpr double kBSplineFlatnessSq = kBSplineFlatness * kBSplineFlatness;
/// Rim-matching distance in cm when the model states no tolerance: the extractor precision, as kWireJoinTolerance.
inline constexpr double kRimMatchTolerance = 1.e-6;

/// Widening of the BVH leaf boxes before the outward float rounding; it dominates every navigation length tolerance.
inline constexpr double kBVHBoxTolerance = 1.e-3;
/// Zero threshold of solveQuarticReal's branch tests, in machine epsilons relative to the normalised terms: dimensionless.
inline constexpr double kQuarticEpsilon = 32. * 2.220446049250313e-16;
/// @}

/// A 2D point/vector in a surface's parametric (u, v) domain.
struct Vec2 {
  double uCoord = 0.;
  double vCoord = 0.;
};

/// A 3D point/vector in the solid's local frame.
struct Vec3 {
  double xCoord = 0.;
  double yCoord = 0.;
  double zCoord = 0.;
};

inline Vec3 operator+(const Vec3& firstVector, const Vec3& secondVector)
{
  return {firstVector.xCoord + secondVector.xCoord, firstVector.yCoord + secondVector.yCoord,
          firstVector.zCoord + secondVector.zCoord};
}

inline Vec3 operator-(const Vec3& firstVector, const Vec3& secondVector)
{
  return {firstVector.xCoord - secondVector.xCoord, firstVector.yCoord - secondVector.yCoord,
          firstVector.zCoord - secondVector.zCoord};
}

inline Vec3 operator*(const Vec3& vector, double scale)
{
  return {vector.xCoord * scale, vector.yCoord * scale, vector.zCoord * scale};
}

inline Vec3 operator*(double scale, const Vec3& vector)
{
  return vector * scale;
}

inline Vec2 operator-(const Vec2& firstPoint, const Vec2& secondPoint)
{
  return {firstPoint.uCoord - secondPoint.uCoord, firstPoint.vCoord - secondPoint.vCoord};
}

/// The 3D length squared of parametric displacement \a delta under the first fundamental form (\a gUU, \a gUV, \a gVV).
inline double parametricLengthSq(double gUU, double gUV, double gVV, const Vec2& delta)
{
  return gUU * delta.uCoord * delta.uCoord + 2. * gUV * delta.uCoord * delta.vCoord +
         gVV * delta.vCoord * delta.vCoord;
}

/// How a wire converts a parametric separation into a 3D length: the owning surface's first fundamental form, or the identity.
struct ParametricMetric {
  using Evaluate = void (*)(const void* context, const Vec2& uv, double& gUU, double& gUV, double& gVV);

  Evaluate evaluate = nullptr;
  const void* context = nullptr;

  /// The 3D length squared spanned by the parametric displacement \a delta starting at \a uv.
  double lengthSq(const Vec2& uv, const Vec2& delta) const
  {
    if (evaluate == nullptr) {
      return delta.uCoord * delta.uCoord + delta.vCoord * delta.vCoord;
    }
    double gUU = 1.;
    double gUV = 0.;
    double gVV = 1.;
    evaluate(context, uv, gUU, gUV, gVV);
    return parametricLengthSq(gUU, gUV, gVV, delta);
  }

  /// The 3D distance squared between two nearby parametric points, with the form evaluated at \a from.
  double distanceSq(const Vec2& from, const Vec2& to) const { return lengthSq(from, to - from); }

  /// The largest 3D length a unit parametric displacement spans at \a uv: the square root of the larger eigenvalue.
  double maxScale(const Vec2& uv) const
  {
    if (evaluate == nullptr) {
      return 1.;
    }
    double gUU = 1.;
    double gUV = 0.;
    double gVV = 1.;
    evaluate(context, uv, gUU, gUV, gVV);
    const double trace = gUU + gVV;
    const double determinant = gUU * gVV - gUV * gUV;
    // the eigenvalues of a symmetric 2x2 form, guarded against a slightly negative discriminant
    const double discriminant = std::max(0., trace * trace - 4. * determinant);
    return std::sqrt(std::max(0., 0.5 * (trace + std::sqrt(discriminant))));
  }
};

/// A ParametricMetric that defers to \a surface, which must outlive it. Every use here is a
/// surface building its own wires inside initialize(), so that holds by construction.
template <typename Surface>
inline ParametricMetric parametricMetricOf(const Surface& surface)
{
  return {[](const void* context, const Vec2& uv, double& gUU, double& gUV, double& gVV) {
            static_cast<const Surface*>(context)->parametricMetric(uv, gUU, gUV, gVV);
          },
          &surface};
}

inline double dot(const Vec3& firstVector, const Vec3& secondVector)
{
  return firstVector.xCoord * secondVector.xCoord + firstVector.yCoord * secondVector.yCoord +
         firstVector.zCoord * secondVector.zCoord;
}

inline Vec3 cross(const Vec3& firstVector, const Vec3& secondVector)
{
  return {firstVector.yCoord * secondVector.zCoord - firstVector.zCoord * secondVector.yCoord,
          firstVector.zCoord * secondVector.xCoord - firstVector.xCoord * secondVector.zCoord,
          firstVector.xCoord * secondVector.yCoord - firstVector.yCoord * secondVector.xCoord};
}

inline double normSq(const Vec3& vector)
{
  return dot(vector, vector);
}

inline double norm(const Vec3& vector)
{
  return std::sqrt(normSq(vector));
}

inline Vec3 normalized(const Vec3& vector)
{
  const double vectorNorm = norm(vector);
  if (vectorNorm <= kTolerance) {
    return {};
  }
  return vector * (1. / vectorNorm);
}

inline double component(const Vec3& vector, int dimension)
{
  if (dimension == 0) {
    return vector.xCoord;
  }
  if (dimension == 1) {
    return vector.yCoord;
  }
  return vector.zCoord;
}

inline void assignComponent(Vec3& vector, int dimension, double value)
{
  if (dimension == 0) {
    vector.xCoord = value;
  } else if (dimension == 1) {
    vector.yCoord = value;
  } else {
    vector.zCoord = value;
  }
}

inline bool finite(const Vec2& point)
{
  return std::isfinite(point.uCoord) && std::isfinite(point.vCoord);
}

inline bool finite(const Vec3& point)
{
  return std::isfinite(point.xCoord) && std::isfinite(point.yCoord) && std::isfinite(point.zCoord);
}

inline double distanceSq(const Vec2& firstPoint, const Vec2& secondPoint)
{
  const double deltaU = firstPoint.uCoord - secondPoint.uCoord;
  const double deltaV = firstPoint.vCoord - secondPoint.vCoord;
  return deltaU * deltaU + deltaV * deltaV;
}

inline double distanceSq(const Vec3& firstPoint, const Vec3& secondPoint)
{
  return normSq(firstPoint - secondPoint);
}

inline double cross2D(const Vec2& firstVector, const Vec2& secondVector)
{
  return firstVector.uCoord * secondVector.vCoord - firstVector.vCoord * secondVector.uCoord;
}

inline double pointSegmentDistanceSq(const Vec2& point, const Vec2& segmentStart, const Vec2& segmentEnd)
{
  const Vec2 segmentVector = segmentEnd - segmentStart;
  const double segmentLengthSq = segmentVector.uCoord * segmentVector.uCoord + segmentVector.vCoord * segmentVector.vCoord;
  if (segmentLengthSq <= kToleranceSq) {
    return distanceSq(point, segmentStart);
  }
  const double pointProjection = ((point.uCoord - segmentStart.uCoord) * segmentVector.uCoord +
                                  (point.vCoord - segmentStart.vCoord) * segmentVector.vCoord) /
                                 segmentLengthSq;
  const double clampedProjection = std::max(0., std::min(1., pointProjection));
  const Vec2 closestPoint{segmentStart.uCoord + clampedProjection * segmentVector.uCoord,
                          segmentStart.vCoord + clampedProjection * segmentVector.vCoord};
  return distanceSq(point, closestPoint);
}

inline double pointSegmentDistanceSq(const Vec3& point, const Vec3& segmentStart, const Vec3& segmentEnd)
{
  const Vec3 segmentVector = segmentEnd - segmentStart;
  const double segmentLengthSq = normSq(segmentVector);
  if (segmentLengthSq <= kToleranceSq) {
    return distanceSq(point, segmentStart);
  }
  const double pointProjection = dot(point - segmentStart, segmentVector) / segmentLengthSq;
  const double clampedProjection = std::max(0., std::min(1., pointProjection));
  const Vec3 closestPoint = segmentStart + segmentVector * clampedProjection;
  return distanceSq(point, closestPoint);
}

/// \name First fundamental forms by surface family, shared by the surfaces and the sidecar reader
/// @{

/// Plane: the frame axes carry the domain's units and need be neither unit nor orthogonal, which
/// makes this the only family with a cross term.
inline void planeParametricMetric(const Vec3& axisU, const Vec3& axisV, double& gUU, double& gUV, double& gVV)
{
  gUU = dot(axisU, axisU);
  gUV = dot(axisU, axisV);
  gVV = dot(axisV, axisV);
}

/// Cylinder, (u, v) = (phi[rad], h[cm]).
inline void cylinderParametricMetric(double radius, double& gUU, double& gUV, double& gVV)
{
  gUU = radius * radius;
  gUV = 0.;
  gVV = 1.;
}

/// Cone, (u, v) = (phi[rad], h[cm]). \a radiusAtHeight is r(v), which reaches zero at an apex;
/// a step in h also walks along the slope, hence gVV > 1.
inline void coneParametricMetric(double radiusAtHeight, double slope, double& gUU, double& gUV, double& gVV)
{
  gUU = radiusAtHeight * radiusAtHeight;
  gUV = 0.;
  gVV = 1. + slope * slope;
}

/// Sphere, (u, v) = (phi[rad], theta[rad]). The azimuthal scale is the radius of the parallel at
/// \a theta, so it vanishes at either pole.
inline void sphereParametricMetric(double radius, double theta, double& gUU, double& gUV, double& gVV)
{
  const double parallelRadius = radius * std::sin(theta);
  gUU = parallelRadius * parallelRadius;
  gUV = 0.;
  gVV = radius * radius;
}

/// Torus, (u, v) = (phiRing[rad], phiTube[rad]). The ring scale runs from R - r to R + r.
inline void torusParametricMetric(double majorRadius, double minorRadius, double phiTube, double& gUU, double& gUV,
                                  double& gVV)
{
  const double ringRadius = majorRadius + minorRadius * std::cos(phiTube);
  gUU = ringRadius * ringRadius;
  gUV = 0.;
  gVV = minorRadius * minorRadius;
}
/// @}

inline bool sameIntersection(double firstDistance, double secondDistance)
{
  return std::abs(firstDistance - secondDistance) <=
         kIntersectionTolerance * std::max(1., std::max(std::abs(firstDistance), std::abs(secondDistance)));
}

/// One ray/surface intersection: the ray parameter and the outward normal; a quadric patch can give several per ray.
struct RayHit {
  double distance = 0.;
  Vec3 normal;
  /// The hit lies within the trim's on-boundary band, so its inside/outside side is a tie-break, not data.
  bool onTrimBoundary = false;
};

/// One straight line segment of a polygon wire, in a surface's parametric (u, v) domain.
struct SurfaceEdge {
  Vec2 start;
  Vec2 end;

  Vec2 direction() const { return end - start; }

  double lengthSq() const
  {
    const Vec2 delta = end - start;
    return delta.uCoord * delta.uCoord + delta.vCoord * delta.vCoord;
  }

  bool degenerate() const { return lengthSq() <= kToleranceSq; }

  /// Squared distance from a parametric point to this edge.
  double distanceSq(const Vec2& point) const { return pointSegmentDistanceSq(point, start, end); }

  /// Closest point on this edge to \a point. Returns the projected point and its clamped
  /// parameter \a parameter in [0, 1] (0 at start, 1 at end). Degenerate edges return start.
  Vec2 closestPoint(const Vec2& point, double& parameter) const
  {
    const Vec2 segmentVector = end - start;
    const double segmentLengthSq = segmentVector.uCoord * segmentVector.uCoord +
                                   segmentVector.vCoord * segmentVector.vCoord;
    if (segmentLengthSq <= kToleranceSq) {
      parameter = 0.;
      return start;
    }
    const double projection = ((point.uCoord - start.uCoord) * segmentVector.uCoord +
                               (point.vCoord - start.vCoord) * segmentVector.vCoord) /
                              segmentLengthSq;
    parameter = std::max(0., std::min(1., projection));
    return {start.uCoord + parameter * segmentVector.uCoord, start.vCoord + parameter * segmentVector.vCoord};
  }

  /// Accumulate the edge endpoints into a parametric axis-aligned bounding box.
  void extendBounds(Vec2& lower, Vec2& upper) const
  {
    lower.uCoord = std::min({lower.uCoord, start.uCoord, end.uCoord});
    lower.vCoord = std::min({lower.vCoord, start.vCoord, end.vCoord});
    upper.uCoord = std::max({upper.uCoord, start.uCoord, end.uCoord});
    upper.vCoord = std::max({upper.vCoord, start.vCoord, end.vCoord});
  }
};

/// Classification of a parametric point against a closed wire.
enum class WireClassification { Outside,
                                Boundary,
                                Inside };

/// The role a wire plays for a bounded surface. Outer wires bound the material, inner wires
/// (holes) subtract from it. The role fixes the expected winding relative to the surface normal.
enum class WireRole { Outer,
                      Inner };

/// Outcome of wire construction / validation. Valid and Reversed are both usable results;
/// Reversed additionally signals that the orientation had to be normalized (a logged repair).
enum class WireStatus {
  Valid,            ///< well-formed and already correctly oriented
  Reversed,         ///< well-formed but re-oriented to match its role (simple, logged repair)
  NonFinite,        ///< a vertex/edge contained a non-finite coordinate
  Open,             ///< an explicit edge list did not form a closed loop
  TooFewVertices,   ///< fewer than three distinct vertices after cleanup
  DegenerateVertex, ///< a non-adjacent vertex coincided (self-touching / pinched loop)
  ZeroArea          ///< the loop encloses no area
};

/// Human-readable description of a wire status, for logging.
inline const char* wireStatusMessage(WireStatus status)
{
  switch (status) {
    case WireStatus::Valid:
      return "valid";
    case WireStatus::Reversed:
      return "orientation normalized to match wire role";
    case WireStatus::NonFinite:
      return "wire contains a non-finite vertex";
    case WireStatus::Open:
      return "wire edges do not form a closed loop";
    case WireStatus::TooFewVertices:
      return "wire needs at least three distinct vertices";
    case WireStatus::DegenerateVertex:
      return "wire has a coincident (pinched) vertex";
    case WireStatus::ZeroArea:
      return "wire has zero area";
  }
  return "unknown wire status";
}

/// kTolerance as a parametric separation at \a uv: the floor of every trim's on-boundary band.
inline double trimLengthFloor(const ParametricMetric& metric, const Vec2& uv)
{
  const double scale = metric.maxScale(uv);
  return scale > kTolerance ? kTolerance / scale : 0.;
}

/// One closed, oriented polygon loop in a surface's parametric domain: outer loops wind counter-clockwise, holes clockwise.
struct SurfaceWire {
  std::vector<Vec2> vertices;
  WireRole role = WireRole::Outer;

  /// For each stored segment its input segment, or -1 once a vertex was dropped; sidecar v3 edge identities key on it.
  std::vector<int> sourceEdge;

  int edgeCount() const { return static_cast<int>(vertices.size()); }

  /// The stored segment that came from input segment \a inputIndex, or -1 if there is none.
  int storedIndexOfSource(int inputIndex) const
  {
    for (size_t index = 0; index < sourceEdge.size(); ++index) {
      if (sourceEdge[index] == inputIndex) {
        return static_cast<int>(index);
      }
    }
    return -1;
  }

  SurfaceEdge edge(int index) const
  {
    const int count = edgeCount();
    return {vertices[index % count], vertices[(index + 1) % count]};
  }

  /// Build and validate the wire from an implicitly closed vertex ring; \a metric turns separations into 3D lengths.
  bool initialize(const std::vector<Vec2>& inputVertices, WireRole wireRole, WireStatus& status,
                  const ParametricMetric& metric = {})
  {
    role = wireRole;
    vertices.clear();
    vertices.reserve(inputVertices.size());
    bool droppedAVertex = false;

    for (const auto& vertex : inputVertices) {
      if (!finite(vertex)) {
        status = WireStatus::NonFinite;
        return false;
      }
      if (vertices.empty() || metric.distanceSq(vertices.back(), vertex) > kToleranceSq) {
        vertices.push_back(vertex);
      } else {
        droppedAVertex = true;
      }
    }

    // drop an explicit closing duplicate (first == last)
    if (vertices.size() > 1 && metric.distanceSq(vertices.front(), vertices.back()) <= kToleranceSq) {
      vertices.pop_back();
      droppedAVertex = true;
    }

    if (vertices.size() < 3) {
      status = WireStatus::TooFewVertices;
      return false;
    }

    // reject self-touching loops (non-adjacent coincident vertices)
    for (size_t firstIndex = 0; firstIndex < vertices.size(); ++firstIndex) {
      for (size_t secondIndex = firstIndex + 1; secondIndex < vertices.size(); ++secondIndex) {
        if (metric.distanceSq(vertices[firstIndex], vertices[secondIndex]) <= kToleranceSq) {
          status = WireStatus::DegenerateVertex;
          return false;
        }
      }
    }

    const double area = signedArea();
    if (std::abs(area) <= kAreaTolerance) {
      status = WireStatus::ZeroArea;
      return false;
    }

    // segment i is input segment i unless a vertex was dropped; then it is unknown
    const int storedCount = static_cast<int>(vertices.size());
    sourceEdge.assign(static_cast<size_t>(storedCount), -1);
    if (!droppedAVertex) {
      for (int index = 0; index < storedCount; ++index) {
        sourceEdge[static_cast<size_t>(index)] = index;
      }
    }

    // outer wires must wind CCW (positive area), inner wires CW (negative area)
    const bool wantPositiveArea = (role == WireRole::Outer);
    if ((area > 0.) != wantPositiveArea) {
      std::reverse(vertices.begin(), vertices.end());
      // reversing the ring maps old vertex k to new index n-1-k, so new segment j spans old
      // vertices n-1-j and n-2-j, i.e. it is old segment n-2-j traversed backwards
      std::vector<int> reversedSource(static_cast<size_t>(storedCount), -1);
      for (int index = 0; index < storedCount; ++index) {
        reversedSource[static_cast<size_t>(index)] =
          sourceEdge[static_cast<size_t>((storedCount - 2 - index % storedCount + 2 * storedCount) % storedCount)];
      }
      sourceEdge.swap(reversedSource);
      status = WireStatus::Reversed;
      return true;
    }

    status = WireStatus::Valid;
    return true;
  }

  /// Build and validate the wire from an ordered edge list, joining within \a joinTolerance through \a metric, as CurveWire does.
  bool initializeFromEdges(const std::vector<SurfaceEdge>& edges, WireRole wireRole, WireStatus& status,
                           const ParametricMetric& metric = {}, double joinTolerance = kWireJoinTolerance)
  {
    if (edges.size() < 3) {
      status = WireStatus::TooFewVertices;
      return false;
    }
    for (size_t edgeIndex = 0; edgeIndex < edges.size(); ++edgeIndex) {
      if (!finite(edges[edgeIndex].start) || !finite(edges[edgeIndex].end)) {
        status = WireStatus::NonFinite;
        return false;
      }
      const Vec2& nextStart = edges[(edgeIndex + 1) % edges.size()].start;
      if (metric.distanceSq(edges[edgeIndex].end, nextStart) > joinTolerance * joinTolerance) {
        status = WireStatus::Open;
        return false;
      }
    }

    std::vector<Vec2> ringVertices;
    ringVertices.reserve(edges.size());
    for (const auto& singleEdge : edges) {
      ringVertices.push_back(singleEdge.start);
    }
    return initialize(ringVertices, wireRole, status, metric);
  }

  double signedArea() const
  {
    double area = 0.;
    for (size_t vertexIndex = 0; vertexIndex < vertices.size(); ++vertexIndex) {
      const auto& currentVertex = vertices[vertexIndex];
      const auto& nextVertex = vertices[(vertexIndex + 1) % vertices.size()];
      area += currentVertex.uCoord * nextVertex.vCoord - nextVertex.uCoord * currentVertex.vCoord;
    }
    return 0.5 * area;
  }

  /// Accumulate this wire's vertices into a parametric axis-aligned bounding box. This is
  /// independent of any concrete surface so cylinders, spheres and cones can reuse it.
  void parametricBounds(Vec2& lower, Vec2& upper) const
  {
    for (const auto& vertex : vertices) {
      lower.uCoord = std::min(lower.uCoord, vertex.uCoord);
      lower.vCoord = std::min(lower.vCoord, vertex.vCoord);
      upper.uCoord = std::max(upper.uCoord, vertex.uCoord);
      upper.vCoord = std::max(upper.vCoord, vertex.vCoord);
    }
  }

  /// The de-duplicated vertex ring, closed back to its first vertex.
  std::vector<Vec2> sampledBoundary() const
  {
    std::vector<Vec2> samples;
    if (vertices.empty()) {
      return samples;
    }
    samples.reserve(vertices.size() + 1);
    samples.insert(samples.end(), vertices.begin(), vertices.end());
    samples.push_back(vertices.front());
    return samples;
  }

  /// Classify against the polygon with an on-boundary half-width of \a band, in parametric units.
  WireClassification classify(const Vec2& point, double band) const
  {
    const double bandSq = band * band;
    bool inside = false;
    for (size_t vertexIndex = 0; vertexIndex < vertices.size(); ++vertexIndex) {
      const auto& segmentStart = vertices[vertexIndex];
      const auto& segmentEnd = vertices[(vertexIndex + 1) % vertices.size()];
      if (pointSegmentDistanceSq(point, segmentStart, segmentEnd) <= bandSq) {
        return WireClassification::Boundary;
      }
      const bool crossesScanline = (segmentStart.vCoord > point.vCoord) != (segmentEnd.vCoord > point.vCoord);
      if (crossesScanline) {
        const double intersectionU = segmentStart.uCoord + (point.vCoord - segmentStart.vCoord) *
                                                             (segmentEnd.uCoord - segmentStart.uCoord) /
                                                             (segmentEnd.vCoord - segmentStart.vCoord);
        if (point.uCoord < intersectionU) {
          inside = !inside;
        }
      }
    }
    return inside ? WireClassification::Inside : WireClassification::Outside;
  }

  /// \a metric sizes the band only: a polygon is exact, so its band is the length floor.
  WireClassification classify(const Vec2& point, const ParametricMetric& metric = {}) const
  {
    return classify(point, trimLengthFloor(metric, point));
  }
};

inline bool pointInTriangle(const Vec2& point, const Vec2& firstVertex, const Vec2& secondVertex,
                            const Vec2& thirdVertex)
{
  const double firstCross = cross2D(secondVertex - firstVertex, point - firstVertex);
  const double secondCross = cross2D(thirdVertex - secondVertex, point - secondVertex);
  const double thirdCross = cross2D(firstVertex - thirdVertex, point - thirdVertex);
  return firstCross >= -kTolerance && secondCross >= -kTolerance && thirdCross >= -kTolerance;
}

/// Ear-clipping triangulation of a simple (non-self-intersecting) parametric wire.
inline std::vector<std::array<int, 3>> triangulateSimpleWire(const SurfaceWire& wire)
{
  std::vector<int> remainingIndices;
  remainingIndices.reserve(wire.vertices.size());
  if (wire.signedArea() >= 0.) {
    for (size_t vertexIndex = 0; vertexIndex < wire.vertices.size(); ++vertexIndex) {
      remainingIndices.push_back(static_cast<int>(vertexIndex));
    }
  } else {
    for (size_t reverseIndex = wire.vertices.size(); reverseIndex > 0; --reverseIndex) {
      remainingIndices.push_back(static_cast<int>(reverseIndex - 1));
    }
  }

  std::vector<std::array<int, 3>> triangles;
  size_t guardCounter = 0;
  while (remainingIndices.size() > 3 && guardCounter++ < wire.vertices.size() * wire.vertices.size()) {
    bool clippedEar = false;
    for (size_t indexPosition = 0; indexPosition < remainingIndices.size(); ++indexPosition) {
      const int previousIndex = remainingIndices[(indexPosition + remainingIndices.size() - 1) % remainingIndices.size()];
      const int currentIndex = remainingIndices[indexPosition];
      const int nextIndex = remainingIndices[(indexPosition + 1) % remainingIndices.size()];

      const auto& previousVertex = wire.vertices[previousIndex];
      const auto& currentVertex = wire.vertices[currentIndex];
      const auto& nextVertex = wire.vertices[nextIndex];
      if (cross2D(currentVertex - previousVertex, nextVertex - currentVertex) <= kTolerance) {
        continue;
      }

      bool containsOtherVertex = false;
      for (int candidateIndex : remainingIndices) {
        if (candidateIndex == previousIndex || candidateIndex == currentIndex || candidateIndex == nextIndex) {
          continue;
        }
        if (pointInTriangle(wire.vertices[candidateIndex], previousVertex, currentVertex, nextVertex)) {
          containsOtherVertex = true;
          break;
        }
      }
      if (containsOtherVertex) {
        continue;
      }

      triangles.push_back({previousIndex, currentIndex, nextIndex});
      remainingIndices.erase(remainingIndices.begin() + indexPosition);
      clippedEar = true;
      break;
    }

    if (!clippedEar) {
      break;
    }
  }

  if (remainingIndices.size() == 3) {
    triangles.push_back({remainingIndices[0], remainingIndices[1], remainingIndices[2]});
  }
  return triangles;
}

/// \name Angular constants for parametric arc curves
/// @{
inline constexpr double kPi = 3.14159265358979323846;
inline constexpr double kTwoPi = 2. * kPi;
inline constexpr double kHalfPi = 0.5 * kPi;
/// Chords per full-circle arc for display and rims, shared by all surfaces so shared rims match; divisible by 4.
inline constexpr int kArcSamples = 24;
/// @}

/// Angular tolerance equivalent to a kTolerance arc length at the given radius.
inline double angularTolerance(double radius)
{
  return kTolerance / std::max(radius, kTolerance);
}

/// Widest angular span of one cover box: pi/4, eight boxes per full turn.
inline constexpr double kCoverChunkAngle = kPi / 4.;

/// The number of kCoverChunkAngle chunks covering an angular span: at least one, and never more
/// than a full turn takes, since a sweep may overshoot 2pi by a rounding hair.
inline int coverChunkCount(double span)
{
  constexpr int fullTurnChunks = static_cast<int>(kTwoPi / kCoverChunkAngle); // eight
  return std::max(1, std::min(fullTurnChunks, static_cast<int>(std::ceil(span / kCoverChunkAngle))));
}

/// Exact range of a cos(t) + b sin(t) over [t0, t1], at most a turn: the endpoint values, widened to the amplitude at a crest.
inline void sinusoidRange(double a, double b, double t0, double t1, double& minimum, double& maximum)
{
  const double atStart = a * std::cos(t0) + b * std::sin(t0);
  const double atEnd = a * std::cos(t1) + b * std::sin(t1);
  minimum = std::min(atStart, atEnd);
  maximum = std::max(atStart, atEnd);
  const double amplitude = std::hypot(a, b);
  const double crest = std::atan2(b, a);
  // shifted into [t0, t0 + 2pi), where a span of at most a full turn makes "<= t1" exactly the
  // test for falling inside the interval
  const double crestInRange = crest - kTwoPi * std::floor((crest - t0) / kTwoPi);
  if (crestInRange <= t1) {
    maximum = amplitude;
  }
  const double trough = crest + kPi;
  const double troughInRange = trough - kTwoPi * std::floor((trough - t0) / kTwoPi);
  if (troughInRange <= t1) {
    minimum = -amplitude;
  }
}

/// One end of sinusoidRange, for the doubly swept covers of the sphere and the torus.
/// @{
inline double sinusoidMinimum(double a, double b, double t0, double t1)
{
  double minimum = 0.;
  double maximum = 0.;
  sinusoidRange(a, b, t0, t1, minimum, maximum);
  return minimum;
}

inline double sinusoidMaximum(double a, double b, double t0, double t1)
{
  double minimum = 0.;
  double maximum = 0.;
  sinusoidRange(a, b, t0, t1, minimum, maximum);
  return maximum;
}
/// @}

/// True if \a angle lies within the angular range [start, start + sweep] (sweep in (0, 2pi]),
/// allowing \a tolerance on both ends and treating a >= 2pi sweep as the full circle.
inline bool angleInSweepRange(double angle, double start, double sweep, double tolerance)
{
  if (sweep >= kTwoPi - kTolerance) {
    return true;
  }
  double delta = angle - start;
  delta -= kTwoPi * std::floor(delta / kTwoPi); // wrap into [0, 2pi)
  return delta <= sweep + tolerance || delta >= kTwoPi - tolerance;
}

/// The \a n-point Gauss-Legendre nodes and weights on [-1, 1], by Newton iteration on P_n.
inline void gaussLegendre(int n, std::vector<double>& nodes, std::vector<double>& weights)
{
  nodes.assign(std::max(n, 1), 0.);
  weights.assign(std::max(n, 1), 0.);
  if (n < 1) {
    return;
  }
  for (int i = 0; i < n; ++i) {
    double root = std::cos(kPi * (i + 0.75) / (n + 0.5)); // asymptotic initial guess
    double derivative = 1.;
    for (int iteration = 0; iteration < 100; ++iteration) {
      double previous = 1.;
      double current = root;
      for (int degreeIndex = 2; degreeIndex <= n; ++degreeIndex) {
        const double next = ((2 * degreeIndex - 1) * root * current - (degreeIndex - 1) * previous) / degreeIndex;
        previous = current;
        current = next;
      }
      derivative = n * (root * current - previous) / (root * root - 1.);
      const double delta = current / derivative;
      root -= delta;
      if (std::abs(delta) < 1.e-15) {
        break;
      }
    }
    nodes[i] = root;
    weights[i] = 2. / ((1. - root * root) * derivative * derivative);
  }
}

/// Fill \a roots with the real roots of w^3 + P w + Q = 0 and return their count: Cardano, or the trigonometric form for three.
/// The branch is chosen by the sign of P, not by a tolerance, so every input is covered.
inline int solveDepressedCubic(double coeffP, double coeffQ, std::array<double, 3>& roots)
{
  const double discriminant = coeffQ * coeffQ / 4. + coeffP * coeffP * coeffP / 27.;
  if (!(coeffP < 0.) || discriminant > 0.) {
    const double sqrtDiscriminant = std::sqrt(std::max(0., discriminant));
    roots[0] = std::cbrt(-0.5 * coeffQ + sqrtDiscriminant) + std::cbrt(-0.5 * coeffQ - sqrtDiscriminant);
    return 1;
  }
  // three real roots: coeffP < 0 here, so the trigonometric form is well defined
  const double magnitude = 2. * std::sqrt(-coeffP / 3.);
  const double cosineArgument = std::max(-1., std::min(1., 3. * coeffQ / (coeffP * magnitude)));
  const double baseAngle = std::acos(cosineArgument);
  for (int branch = 0; branch < 3; ++branch) {
    roots[branch] = magnitude * std::cos((baseAngle - kTwoPi * branch) / 3.);
  }
  return 3;
}

/// Which of solveQuarticReal's branches produced its roots, for the tests.
enum class QuarticBranch {
  NotAQuartic, ///< the leading coefficient vanishes; no roots are produced
  Biquadratic, ///< the depressed quartic's odd term is zero, so y^4 + p y^2 + r = 0 is solved directly
  Resolvent    ///< Ferrari's general branch, through the resolvent cubic
};

/// The real roots of a quartic: at most four, held inline.
struct QuarticRoots {
  std::array<double, 4> value{};
  int count = 0;
  void push_back(double root) { value[count++] = root; }
  double* begin() { return value.data(); }
  double* end() { return value.data() + count; }
  const double* begin() const { return value.data(); }
  const double* end() const { return value.data() + count; }
  size_t size() const { return static_cast<size_t>(count); }
  bool empty() const { return count == 0; }
  double operator[](size_t index) const { return value[index]; }
};

/// Real roots of a4 x^4 + a3 x^3 + a2 x^2 + a1 x + a0 = 0 (a4 != 0) by Ferrari's method and Newton polishing; a tangential root is a near-equal pair.
/// The root variable is first rescaled by a power of two, exactly, so all branch tests are dimensionless; \a takenBranch reports the branch.
inline QuarticRoots solveQuarticReal(double a4, double a3, double a2, double a1, double a0,
                                     QuarticBranch* takenBranch = nullptr)
{
  const auto note = [takenBranch](QuarticBranch branch) {
    if (takenBranch) {
      *takenBranch = branch;
    }
  };
  note(QuarticBranch::NotAQuartic);
  QuarticRoots roots;
  // A genuine quartic needs only a non-zero leading coefficient. There is no scale to compare it
  // against -- the normalisation below handles any coefficient ratio -- so the test is exact.
  if (!(std::abs(a4) > 0.)) {
    return roots; // the torus caller guarantees a4 = |dir|^4 > 0
  }
  // monic x^4 + b x^3 + c x^2 + d x + e
  double coeffB = a3 / a4, coeffC = a2 / a4, coeffD = a1 / a4, coeffE = a0 / a4;
  if (!std::isfinite(coeffB) || !std::isfinite(coeffC) || !std::isfinite(coeffD) || !std::isfinite(coeffE)) {
    return roots; // a4 is denormal-small next to the rest, or an input was not finite
  }
  // Cauchy root bound rounded up to a power of two, so x = scale * y is exact; x^4 = 0 keeps scale = 1
  const double rootBound = std::max({std::abs(coeffB), std::sqrt(std::abs(coeffC)),
                                     std::cbrt(std::abs(coeffD)), std::sqrt(std::sqrt(std::abs(coeffE)))});
  int boundExponent = 0;
  std::frexp(rootBound, &boundExponent);
  const double scale = std::ldexp(1., boundExponent);
  coeffB /= scale;
  coeffC /= scale * scale;
  coeffD /= scale * scale * scale;
  coeffE /= scale * scale * scale * scale;

  // depress with y = z - b/4: z^4 + p z^2 + q z + r
  const double termP = coeffC - 3. * coeffB * coeffB / 8.;
  const double termQ = coeffD - coeffB * coeffC / 2. + coeffB * coeffB * coeffB / 8.;
  const double termR =
    coeffE - coeffB * coeffD / 4. + coeffB * coeffB * coeffC / 16. - 3. * coeffB * coeffB * coeffB * coeffB / 256.;
  const double shift = -coeffB / 4.;

  auto addQuadraticRoots = [&](double quadB, double quadC) {
    const double discriminant = quadB * quadB - 4. * quadC;
    if (discriminant < 0.) {
      return; // complex pair
    }
    const double sqrtDiscriminant = std::sqrt(discriminant);
    roots.push_back(shift + 0.5 * (-quadB - sqrtDiscriminant));
    roots.push_back(shift + 0.5 * (-quadB + sqrtDiscriminant));
  };

  auto addBiquadraticRoots = [&]() {
    // biquadratic z^4 + p z^2 + r = 0
    const double discriminant = termP * termP - 4. * termR;
    if (discriminant < 0.) {
      return;
    }
    const double sqrtDiscriminant = std::sqrt(discriminant);
    for (const double zSquared : {0.5 * (-termP + sqrtDiscriminant), 0.5 * (-termP - sqrtDiscriminant)}) {
      if (zSquared >= 0.) {
        const double z = std::sqrt(zSquared);
        roots.push_back(shift + z);
        roots.push_back(shift - z);
      }
    }
  };

  // q is zero to the precision of its terms, which normalisation bounds by 1: kQuarticEpsilon over the whole quartic, not over q's terms
  bool biquadratic = std::abs(termQ) <= kQuarticEpsilon;
  if (!biquadratic) {
    note(QuarticBranch::Resolvent);
    // resolvent cubic m^3 + p m^2 + (p^2/4 - r) m - q^2/8 = 0; its largest real root is > 0
    const double cubicA2 = termP;
    const double cubicA1 = termP * termP / 4. - termR;
    const double cubicA0 = -termQ * termQ / 8.;
    const double cubicP = cubicA1 - cubicA2 * cubicA2 / 3.;
    const double cubicQ = 2. * cubicA2 * cubicA2 * cubicA2 / 27. - cubicA2 * cubicA1 / 3. + cubicA0;
    std::array<double, 3> cubicRoots;
    const int cubicCount = solveDepressedCubic(cubicP, cubicQ, cubicRoots);
    double resolvent = 0.;
    for (int index = 0; index < cubicCount; ++index) {
      resolvent = std::max(resolvent, cubicRoots[index] - cubicA2 / 3.);
    }
    // a resolvent below the resolution of its cubic is noise; then the biquadratic branch is the better-conditioned answer
    const double resolventScale = std::max({std::abs(cubicA2), std::sqrt(std::abs(cubicA1)),
                                            std::cbrt(std::abs(cubicA0))});
    if (resolvent > kQuarticEpsilon * resolventScale) {
      const double sqrtTwoResolvent = std::sqrt(2. * resolvent);
      const double linearTerm = sqrtTwoResolvent * termQ / (4. * resolvent);
      addQuadraticRoots(-sqrtTwoResolvent, termP / 2. + resolvent + linearTerm);
      addQuadraticRoots(sqrtTwoResolvent, termP / 2. + resolvent - linearTerm);
    } else {
      biquadratic = true;
    }
  }
  if (biquadratic) {
    note(QuarticBranch::Biquadratic);
    addBiquadraticRoots();
  }

  // Newton polish against the monic quartic; a step longer than the Cauchy bound 2, or non-finite, is rejected
  auto quartic = [&](double x) { return (((x + coeffB) * x + coeffC) * x + coeffD) * x + coeffE; };
  auto quarticDerivative = [&](double x) { return ((4. * x + 3. * coeffB) * x + 2. * coeffC) * x + coeffD; };
  for (double& root : roots) {
    for (int iteration = 0; iteration < 2; ++iteration) {
      const double step = quartic(root) / quarticDerivative(root);
      if (std::isfinite(step) && std::abs(step) <= 2.) {
        root -= step;
      }
    }
  }
  for (double& root : roots) {
    root *= scale; // exact: scale is a power of two
  }
  return roots;
}

/// Kind of a 2D trimmed boundary curve.
enum class CurveKind { Line,   ///< straight line segment
                       Arc,    ///< circular arc
                       BSpline ///< clamped (rational) B-spline curve
};

/// One trimmed boundary curve in a surface's (u, v) domain: a line segment, a circular arc or a clamped (rational) B-spline.
struct Curve2D {
  CurveKind kind = CurveKind::Line;
  Vec2 lineStart;         ///< line: start point (unused for arcs)
  Vec2 lineEnd;           ///< line: end point (unused for arcs)
  Vec2 center;            ///< arc: circle centre (unused for lines)
  double radius = 0.;     ///< arc: circle radius
  double startAngle = 0.; ///< arc: start angle [rad]
  double endAngle = 0.;   ///< arc: end angle [rad] (sweep = endAngle - startAngle)

  /// \name B-spline data (kind == BSpline): poles, optional weights and a clamped knot vector; the curve parameter runs on [0, 1]. @{
  int degree = 0;
  std::vector<Vec2> poles;
  std::vector<double> weights;
  std::vector<double> knots;
  /// The flattened on-curve polyline, both ends included; CurveWire::initialize fills it and reversing clears it.
  mutable std::vector<Vec2> bsplineCache;
  /// @}

  /// \name Loop-canonical endpoints: the seam vertices the curve's neighbours agree on, substituted at the polyline's ends
  /// @{
  Vec2 canonicalStart;
  Vec2 canonicalEnd;
  bool hasCanonicalEndpoints = false;

  void setCanonicalEndpoints(const Vec2& start, const Vec2& end)
  {
    canonicalStart = start;
    canonicalEnd = end;
    hasCanonicalEndpoints = true;
    bsplineCache.clear(); // the polyline carries them, so it has to be rebuilt
  }

  /// Where this curve begins and ends as far as the loop is concerned: the canonical seam vertex
  /// when a wire has fixed one, and the curve's own endpoint when it stands alone.
  Vec2 loopStart() const { return hasCanonicalEndpoints ? canonicalStart : startPoint(); }
  Vec2 loopEnd() const { return hasCanonicalEndpoints ? canonicalEnd : endPoint(); }
  /// @}

  static Curve2D makeLine(const Vec2& start, const Vec2& end)
  {
    Curve2D curve;
    curve.kind = CurveKind::Line;
    curve.lineStart = start;
    curve.lineEnd = end;
    return curve;
  }

  static Curve2D makeArc(const Vec2& arcCenter, double arcRadius, double arcStartAngle, double arcEndAngle)
  {
    Curve2D curve;
    curve.kind = CurveKind::Arc;
    curve.center = arcCenter;
    curve.radius = arcRadius;
    curve.startAngle = arcStartAngle;
    curve.endAngle = arcEndAngle;
    return curve;
  }

  /// Full circle as one arc curve (counter-clockwise unless \a clockwise is set).
  static Curve2D makeCircle(const Vec2& arcCenter, double arcRadius, bool clockwise = false)
  {
    return makeArc(arcCenter, arcRadius, 0., clockwise ? -kTwoPi : kTwoPi);
  }

  /// Clamped (rational) B-spline curve of degree \a splineDegree. \a splineWeights may be empty
  /// for a non-rational curve; \a splineKnots must be the clamped flat knot vector.
  static Curve2D makeBSpline(int splineDegree, std::vector<Vec2> splinePoles,
                             std::vector<double> splineWeights, std::vector<double> splineKnots)
  {
    Curve2D curve;
    curve.kind = CurveKind::BSpline;
    curve.degree = splineDegree;
    curve.poles = std::move(splinePoles);
    curve.weights = std::move(splineWeights);
    curve.knots = std::move(splineKnots);
    return curve;
  }

  bool isArc() const { return kind == CurveKind::Arc; }
  bool isBSpline() const { return kind == CurveKind::BSpline; }

  double sweep() const { return endAngle - startAngle; }

  /// \name B-spline evaluation helpers (kind == BSpline)
  /// @{
  double bsplineT0() const { return knots[degree]; }
  double bsplineT1() const { return knots[poles.size()]; }

  /// True when the knot vector is clamped, so the curve interpolates its first and last pole.
  bool bsplineIsClamped() const
  {
    const size_t lastKnot = knots.size() - 1;
    for (int offset = 1; offset <= degree; ++offset) {
      if (std::abs(knots[offset] - knots[0]) > kTolerance ||
          std::abs(knots[lastKnot - offset] - knots[lastKnot]) > kTolerance) {
        return false;
      }
    }
    return true;
  }

  /// True if the curve carries non-unit weights (a rational B-spline).
  bool bsplineRational() const
  {
    for (double weight : weights) {
      if (std::abs(weight - 1.) > kTolerance) {
        return true;
      }
    }
    return false;
  }

  /// Knot span index of parameter \a knotValue for the clamped knot vector.
  int bsplineSpan(double knotValue) const
  {
    const int lastPole = static_cast<int>(poles.size()) - 1;
    if (knotValue >= knots[lastPole + 1]) {
      return lastPole;
    }
    if (knotValue <= knots[degree]) {
      return degree;
    }
    int low = degree;
    int high = lastPole + 1;
    int mid = (low + high) / 2;
    while (knotValue < knots[mid] || knotValue >= knots[mid + 1]) {
      if (knotValue < knots[mid]) {
        high = mid;
      } else {
        low = mid;
      }
      mid = (low + high) / 2;
    }
    return mid;
  }

  /// Non-zero degree-p basis functions and first derivatives at \a knotValue in \a span (The NURBS Book, DersBasisFuns).
  void bsplineBasis(int span, double knotValue, std::vector<double>& basis,
                    std::vector<double>& basisDeriv) const
  {
    const int p = degree;
    std::vector<std::vector<double>> ndu(p + 1, std::vector<double>(p + 1, 0.));
    std::vector<double> left(p + 1, 0.);
    std::vector<double> right(p + 1, 0.);
    ndu[0][0] = 1.;
    for (int j = 1; j <= p; ++j) {
      left[j] = knotValue - knots[span + 1 - j];
      right[j] = knots[span + j] - knotValue;
      double saved = 0.;
      for (int r = 0; r < j; ++r) {
        ndu[j][r] = right[r + 1] + left[j - r];
        const double temp = ndu[r][j - 1] / ndu[j][r];
        ndu[r][j] = saved + right[r + 1] * temp;
        saved = left[j - r] * temp;
      }
      ndu[j][j] = saved;
    }
    basis.assign(p + 1, 0.);
    basisDeriv.assign(p + 1, 0.);
    for (int j = 0; j <= p; ++j) {
      basis[j] = ndu[j][p];
    }
    // first derivative (specialization of DersBasisFuns for the k = 1 term)
    for (int r = 0; r <= p; ++r) {
      double d = 0.;
      const int pk = p - 1;
      if (r >= 1) {
        d += (1. / ndu[pk + 1][r - 1]) * ndu[r - 1][pk];
      }
      if (r <= pk) {
        d += (-1. / ndu[pk + 1][r]) * ndu[r][pk];
      }
      basisDeriv[r] = d * p;
    }
  }

  /// Evaluate the (rational) B-spline point \a pointOut and its knot-parameter derivative
  /// \a derivativeOut at knot parameter \a knotValue.
  void bsplineEval(double knotValue, Vec2& pointOut, Vec2& derivativeOut) const
  {
    const int p = degree;
    const int span = bsplineSpan(knotValue);
    std::vector<double> basis;
    std::vector<double> basisDeriv;
    bsplineBasis(span, knotValue, basis, basisDeriv);
    Vec2 weightedSum{0., 0.};
    Vec2 weightedDeriv{0., 0.};
    double weightTotal = 0.;
    double weightDeriv = 0.;
    for (int j = 0; j <= p; ++j) {
      const int idx = span - p + j;
      const double weight = weights.empty() ? 1. : weights[idx];
      weightedSum.uCoord += basis[j] * weight * poles[idx].uCoord;
      weightedSum.vCoord += basis[j] * weight * poles[idx].vCoord;
      weightTotal += basis[j] * weight;
      weightedDeriv.uCoord += basisDeriv[j] * weight * poles[idx].uCoord;
      weightedDeriv.vCoord += basisDeriv[j] * weight * poles[idx].vCoord;
      weightDeriv += basisDeriv[j] * weight;
    }
    const double invWeight = (std::abs(weightTotal) > kTolerance) ? 1. / weightTotal : 0.;
    pointOut = {weightedSum.uCoord * invWeight, weightedSum.vCoord * invWeight};
    derivativeOut = {(weightedDeriv.uCoord * weightTotal - weightedSum.uCoord * weightDeriv) * invWeight * invWeight,
                     (weightedDeriv.vCoord * weightTotal - weightedSum.vCoord * weightDeriv) * invWeight * invWeight};
  }

  /// B-spline point at curve parameter \a parameter in [0, 1].
  Vec2 bsplinePointAt(double parameter) const
  {
    const double knotValue = bsplineT0() + parameter * (bsplineT1() - bsplineT0());
    Vec2 point;
    Vec2 derivative;
    bsplineEval(knotValue, point, derivative);
    return point;
  }

  /// Adaptively sample the B-spline into an on-curve polyline, subdividing until each chord is flat to sqrt(\a flatnessSq).
  void bsplineSampleInto(std::vector<Vec2>& samples, double flatnessSq = kBSplineFlatnessSq,
                         int maxDepth = 16) const
  {
    const double t0 = bsplineT0();
    const double t1 = bsplineT1();
    Vec2 startPointValue;
    Vec2 endPointValue;
    Vec2 unusedDerivative;
    bsplineEval(t0, startPointValue, unusedDerivative);
    bsplineEval(t1, endPointValue, unusedDerivative);
    samples.push_back(startPointValue);
    bsplineSampleRecursive(t0, t1, startPointValue, endPointValue, flatnessSq, maxDepth, samples);
  }

  /// Whether a knot lies strictly inside (\a lowT, \a highT); such an interval is never called flat.
  bool spansInteriorKnot(double lowT, double highT) const
  {
    // a clamped knot vector repeats its ends degree+1 times, so the interior knots are the
    // entries [degree + 1, poles.size()); a single-span (Bezier) curve has none
    const size_t firstInterior = static_cast<size_t>(degree) + 1;
    const size_t endInterior = std::min(poles.size(), knots.size());
    if (firstInterior >= endInterior) {
      return false;
    }
    const auto begin = knots.begin() + static_cast<std::ptrdiff_t>(firstInterior);
    const auto end = knots.begin() + static_cast<std::ptrdiff_t>(endInterior);
    const auto firstAbove = std::upper_bound(begin, end, lowT);
    return firstAbove != end && *firstAbove < highT;
  }

  /// Append the interior knots in (\a from, \a to), in the curve's [0, 1] parameter; none for a line or an arc.
  void appendInteriorKnots(double from, double to, std::vector<double>& breakpoints) const
  {
    if (kind != CurveKind::BSpline) {
      return;
    }
    const double t0 = bsplineT0();
    const double span = bsplineT1() - t0;
    if (!(span > 0.)) {
      return;
    }
    const size_t firstInterior = static_cast<size_t>(degree) + 1;
    const size_t endInterior = std::min(poles.size(), knots.size());
    for (size_t index = firstInterior; index < endInterior; ++index) {
      const double parameter = (knots[index] - t0) / span;
      if (parameter > from && parameter < to) {
        breakpoints.push_back(parameter);
      }
    }
  }

  /// An upper bound on how far u travels along the curve between \a from and \a to.
  double uVariation(double from, double to) const
  {
    if (kind == CurveKind::Line) {
      return std::abs(lineEnd.uCoord - lineStart.uCoord) * std::abs(to - from);
    }
    if (kind == CurveKind::BSpline) {
      // within one knot span the curve lies in the hull of its degree + 1 poles, so their u spread bounds the travel
      const double knotStart = bsplineT0();
      const double knotSpan = bsplineT1() - knotStart;
      const double knotMid = knotStart + 0.5 * (from + to) * knotSpan;
      size_t spanIndex = static_cast<size_t>(degree);
      while (spanIndex + 1 < poles.size() && spanIndex + 1 < knots.size() && knots[spanIndex + 1] <= knotMid) {
        ++spanIndex;
      }
      const size_t firstPole = spanIndex - static_cast<size_t>(degree);
      double lowU = std::numeric_limits<double>::infinity();
      double highU = -std::numeric_limits<double>::infinity();
      for (size_t index = firstPole; index <= spanIndex && index < poles.size(); ++index) {
        lowU = std::min(lowU, poles[index].uCoord);
        highU = std::max(highU, poles[index].uCoord);
      }
      return (highU >= lowU) ? (highU - lowU) : 0.;
    }
    // arc: u(angle) = center.u + radius cos(angle), so u turns exactly at angle = 0 and pi (mod
    // 2 pi). Sum the monotone runs between those turning points and the interval's own ends.
    const double angleFrom = startAngle + from * sweep();
    const double angleTo = startAngle + to * sweep();
    const double low = std::min(angleFrom, angleTo);
    const double high = std::max(angleFrom, angleTo);
    double variation = 0.;
    double previous = low;
    const double firstTurn = std::ceil(low / kPi) * kPi;
    for (double turn = firstTurn; turn < high; turn += kPi) {
      variation += std::abs(radius * (std::cos(turn) - std::cos(previous)));
      previous = turn;
    }
    return variation + std::abs(radius * (std::cos(high) - std::cos(previous)));
  }

  void bsplineSampleRecursive(double t0, double t1, const Vec2& p0, const Vec2& p1, double flatnessSq,
                              int depth, std::vector<Vec2>& samples) const
  {
    const double tMid = 0.5 * (t0 + t1);
    Vec2 midPoint;
    Vec2 unusedDerivative;
    bsplineEval(tMid, midPoint, unusedDerivative);
    // a degenerate (closed) chord must not end the recursion: test the distance to its single point instead
    const bool degenerateChord = surface::distanceSq(p0, p1) <= flatnessSq;
    const auto deviationSq = [&](const Vec2& point) {
      return degenerateChord ? surface::distanceSq(point, p0) : pointSegmentDistanceSq(point, p0, p1);
    };
    // Three interior probes: a single midpoint probe is blind to curves symmetric about their parameter midpoint.
    double flatness = deviationSq(midPoint);
    for (const double fraction : {0.25, 0.75}) {
      Vec2 probePoint;
      bsplineEval(t0 + (t1 - t0) * fraction, probePoint, unusedDerivative);
      flatness = std::max(flatness, deviationSq(probePoint));
    }
    if (depth <= 0 || (flatness <= flatnessSq && !spansInteriorKnot(t0, t1))) {
      samples.push_back(p1);
      return;
    }
    bsplineSampleRecursive(t0, tMid, p0, midPoint, flatnessSq, depth - 1, samples);
    bsplineSampleRecursive(tMid, t1, midPoint, p1, flatnessSq, depth - 1, samples);
  }

  /// The flattened polyline in \a bsplineCache, computed here if the wire has not filled it.
  const std::vector<Vec2>& bsplineSamples() const
  {
    if (bsplineCache.empty()) {
      bsplineSampleInto(bsplineCache);
      // one canonical polyline, with the seam vertices substituted at its ends
      if (hasCanonicalEndpoints && bsplineCache.size() >= 2) {
        bsplineCache.front() = canonicalStart;
        bsplineCache.back() = canonicalEnd;
      }
    }
    return bsplineCache;
  }
  /// @}

  /// Basic structural validity (finite data, positive radius for arcs, well-formed clamped knot
  /// vector for B-splines).
  bool valid() const
  {
    if (kind == CurveKind::Line) {
      return finite(lineStart) && finite(lineEnd);
    }
    if (kind == CurveKind::Arc) {
      return finite(center) && std::isfinite(radius) && radius > kTolerance && std::isfinite(startAngle) &&
             std::isfinite(endAngle);
    }
    // B-spline
    const int nPoles = static_cast<int>(poles.size());
    if (degree < 1 || nPoles < degree + 1) {
      return false;
    }
    if (static_cast<int>(knots.size()) != nPoles + degree + 1) {
      return false;
    }
    if (!weights.empty() && static_cast<int>(weights.size()) != nPoles) {
      return false;
    }
    for (const auto& pole : poles) {
      if (!finite(pole)) {
        return false;
      }
    }
    for (double weight : weights) {
      if (!std::isfinite(weight) || weight <= kTolerance) {
        return false;
      }
    }
    for (size_t index = 1; index < knots.size(); ++index) {
      if (!std::isfinite(knots[index]) || knots[index] < knots[index - 1] - kTolerance) {
        return false;
      }
    }
    return bsplineT1() - bsplineT0() > kTolerance;
  }

  Vec2 pointAtAngle(double angle) const
  {
    return {center.uCoord + radius * std::cos(angle), center.vCoord + radius * std::sin(angle)};
  }

  /// Point at curve parameter \a parameter in [0, 1] (0 at the start, 1 at the end).
  Vec2 pointAt(double parameter) const
  {
    if (kind == CurveKind::Line) {
      return {lineStart.uCoord + parameter * (lineEnd.uCoord - lineStart.uCoord),
              lineStart.vCoord + parameter * (lineEnd.vCoord - lineStart.vCoord)};
    }
    if (kind == CurveKind::BSpline) {
      return bsplinePointAt(parameter);
    }
    return pointAtAngle(startAngle + parameter * sweep());
  }

  Vec2 startPoint() const
  {
    if (kind == CurveKind::Line) {
      return lineStart;
    }
    if (kind == CurveKind::BSpline) {
      // a clamped knot vector interpolates its first pole exactly; anything else has to be evaluated
      return bsplineIsClamped() ? poles.front() : bsplinePointAt(0.);
    }
    return pointAtAngle(startAngle);
  }
  Vec2 endPoint() const
  {
    if (kind == CurveKind::Line) {
      return lineEnd;
    }
    if (kind == CurveKind::BSpline) {
      return bsplineIsClamped() ? poles.back() : bsplinePointAt(1.);
    }
    return pointAtAngle(endAngle);
  }

  /// dC/dt at \a parameter in [0, 1], unnormalised; tangentAt() is it normalised.
  Vec2 derivativeAt(double parameter) const
  {
    if (kind == CurveKind::Line) {
      return {lineEnd.uCoord - lineStart.uCoord, lineEnd.vCoord - lineStart.vCoord};
    }
    if (kind == CurveKind::BSpline) {
      const double span = bsplineT1() - bsplineT0();
      Vec2 point;
      Vec2 derivative;
      bsplineEval(bsplineT0() + parameter * span, point, derivative);
      return {derivative.uCoord * span, derivative.vCoord * span};
    }
    const double angle = startAngle + parameter * sweep();
    return {-radius * std::sin(angle) * sweep(), radius * std::cos(angle) * sweep()};
  }

  /// Unit tangent at parameter \a parameter, pointing in the direction of increasing parameter.
  Vec2 tangentAt(double parameter) const
  {
    if (kind == CurveKind::Line) {
      const Vec2 delta{lineEnd.uCoord - lineStart.uCoord, lineEnd.vCoord - lineStart.vCoord};
      const double length = std::sqrt(delta.uCoord * delta.uCoord + delta.vCoord * delta.vCoord);
      if (length <= kTolerance) {
        return {0., 0.};
      }
      return {delta.uCoord / length, delta.vCoord / length};
    }
    if (kind == CurveKind::BSpline) {
      // dC/dt scaled by the positive constant dt/ds, so the normalized direction is unchanged
      const double knotValue = bsplineT0() + parameter * (bsplineT1() - bsplineT0());
      Vec2 point;
      Vec2 derivative;
      bsplineEval(knotValue, point, derivative);
      const double length = std::sqrt(derivative.uCoord * derivative.uCoord + derivative.vCoord * derivative.vCoord);
      if (length <= kTolerance) {
        return {0., 0.};
      }
      return {derivative.uCoord / length, derivative.vCoord / length};
    }
    const double angle = startAngle + parameter * sweep();
    const double direction = sweep() >= 0. ? 1. : -1.;
    return {-direction * std::sin(angle), direction * std::cos(angle)};
  }

  /// True if \a angle lies within the arc's angular sweep (accounting for direction and wrap).
  bool angleInSweep(double angle) const
  {
    const double totalSweep = sweep();
    const double magnitude = std::abs(totalSweep);
    if (magnitude >= kTwoPi - kTolerance) {
      return true; // full circle
    }
    double delta = (totalSweep >= 0.) ? (angle - startAngle) : (startAngle - angle);
    delta -= kTwoPi * std::floor(delta / kTwoPi); // wrap into [0, 2pi)
    return delta <= magnitude + kTolerance;
  }

  /// Map an angle known to lie within the sweep to a clamped parameter in [0, 1].
  double angleParameter(double angle) const
  {
    const double totalSweep = sweep();
    if (std::abs(totalSweep) <= kTolerance) {
      return 0.;
    }
    double delta = (totalSweep >= 0.) ? (angle - startAngle) : (startAngle - angle);
    delta -= kTwoPi * std::floor(delta / kTwoPi);
    return std::max(0., std::min(1., delta / std::abs(totalSweep)));
  }

  /// Accumulate this curve's exact extent into a parametric axis-aligned bounding box.
  void extendBounds(Vec2& lower, Vec2& upper) const
  {
    auto include = [&](const Vec2& point) {
      lower.uCoord = std::min(lower.uCoord, point.uCoord);
      lower.vCoord = std::min(lower.vCoord, point.vCoord);
      upper.uCoord = std::max(upper.uCoord, point.uCoord);
      upper.vCoord = std::max(upper.vCoord, point.vCoord);
    };
    if (kind == CurveKind::BSpline) {
      // the control-point convex hull contains the curve, so its box is a conservative (exact
      // upper bound) parametric AABB — consistent with the BVH's conservative-box philosophy
      for (const auto& pole : poles) {
        include(pole);
      }
      return;
    }
    includeAnalyticExtremes(include);
  }

  /// As extendBounds, measured on the curve: a B-spline contributes its sampled polyline, not its pole hull.
  void extendTightBounds(Vec2& lower, Vec2& upper) const
  {
    auto include = [&](const Vec2& point) {
      lower.uCoord = std::min(lower.uCoord, point.uCoord);
      lower.vCoord = std::min(lower.vCoord, point.vCoord);
      upper.uCoord = std::max(upper.uCoord, point.uCoord);
      upper.vCoord = std::max(upper.vCoord, point.vCoord);
    };
    if (kind == CurveKind::BSpline) {
      for (const auto& sample : bsplineSamples()) {
        include(sample);
      }
      return;
    }
    includeAnalyticExtremes(include);
  }

  /// Endpoints plus an arc's axis-extreme points inside the sweep: the exact extent of a line or an arc.
  template <typename Include>
  void includeAnalyticExtremes(const Include& include) const
  {
    include(startPoint());
    include(endPoint());
    if (kind == CurveKind::Arc) {
      // include the axis-extreme points (angles 0, pi/2, pi, 3pi/2) that fall within the sweep
      const double cardinalAngles[4] = {0., kHalfPi, kPi, 3. * kHalfPi};
      for (double cardinal : cardinalAngles) {
        if (angleInSweep(cardinal)) {
          include(pointAtAngle(cardinal));
        }
      }
    }
  }

  /// Closest point on the curve to \a point, returning the clamped parameter in \a parameter.
  Vec2 closestPoint(const Vec2& point, double& parameter) const
  {
    if (kind == CurveKind::BSpline) {
      // distance to the cached polyline, accurate to the sampling flatness
      const auto& polyline = bsplineSamples();
      if (polyline.size() < 2) {
        parameter = 0.;
        return startPoint();
      }
      const int segments = static_cast<int>(polyline.size()) - 1;
      double bestDistanceSq = std::numeric_limits<double>::infinity();
      Vec2 bestPoint = polyline.front();
      double bestParameter = 0.;
      for (int index = 0; index < segments; ++index) {
        const Vec2 segmentStart = polyline[index];
        const Vec2 segmentVector = polyline[index + 1] - segmentStart;
        const double segmentLengthSq =
          segmentVector.uCoord * segmentVector.uCoord + segmentVector.vCoord * segmentVector.vCoord;
        double projection = 0.;
        if (segmentLengthSq > kToleranceSq) {
          projection = ((point.uCoord - segmentStart.uCoord) * segmentVector.uCoord +
                        (point.vCoord - segmentStart.vCoord) * segmentVector.vCoord) /
                       segmentLengthSq;
          projection = std::max(0., std::min(1., projection));
        }
        const Vec2 candidate{segmentStart.uCoord + projection * segmentVector.uCoord,
                             segmentStart.vCoord + projection * segmentVector.vCoord};
        const double candidateDistanceSq = surface::distanceSq(point, candidate);
        if (candidateDistanceSq < bestDistanceSq) {
          bestDistanceSq = candidateDistanceSq;
          bestPoint = candidate;
          bestParameter = (index + projection) / segments;
        }
      }
      parameter = bestParameter;
      return bestPoint;
    }
    if (kind == CurveKind::Line) {
      const Vec2 segment{lineEnd.uCoord - lineStart.uCoord, lineEnd.vCoord - lineStart.vCoord};
      const double lengthSq = segment.uCoord * segment.uCoord + segment.vCoord * segment.vCoord;
      if (lengthSq <= kToleranceSq) {
        parameter = 0.;
        return lineStart;
      }
      const double projection = ((point.uCoord - lineStart.uCoord) * segment.uCoord +
                                 (point.vCoord - lineStart.vCoord) * segment.vCoord) /
                                lengthSq;
      parameter = std::max(0., std::min(1., projection));
      return {lineStart.uCoord + parameter * segment.uCoord, lineStart.vCoord + parameter * segment.vCoord};
    }
    // arc: project radially onto the circle, then clamp the angle to the sweep
    const double deltaU = point.uCoord - center.uCoord;
    const double deltaV = point.vCoord - center.vCoord;
    if (deltaU * deltaU + deltaV * deltaV <= kToleranceSq) {
      parameter = 0.; // point at the centre: every arc point is equidistant
      return startPoint();
    }
    const double angle = std::atan2(deltaV, deltaU);
    if (angleInSweep(angle)) {
      parameter = angleParameter(angle);
      return pointAtAngle(angle);
    }
    const Vec2 startCandidate = startPoint();
    const Vec2 endCandidate = endPoint();
    if (surface::distanceSq(point, startCandidate) <= surface::distanceSq(point, endCandidate)) {
      parameter = 0.;
      return startCandidate;
    }
    parameter = 1.;
    return endCandidate;
  }

  /// Squared distance from \a point to the curve.
  double distanceSq(const Vec2& point) const
  {
    double parameter = 0.;
    return surface::distanceSq(point, closestPoint(point, parameter));
  }

  /// Exact contribution of this directed curve to the enclosed signed area,
  /// i.e. (1/2) * integral of (u dv - v du) along the curve (Green's theorem).
  double signedAreaContribution() const
  {
    if (kind == CurveKind::Line) {
      return 0.5 * (lineStart.uCoord * lineEnd.vCoord - lineEnd.uCoord * lineStart.vCoord);
    }
    if (kind == CurveKind::BSpline) {
      // Green's area per knot span by Gauss-Legendre: exact for a non-rational span, approximate for a rational one
      const int p = degree;
      const int order = bsplineRational() ? std::max(2 * p + 2, 8) : (p + 1);
      std::vector<double> nodes;
      std::vector<double> nodeWeights;
      gaussLegendre(order, nodes, nodeWeights);
      double area = 0.;
      const int lastSpan = static_cast<int>(poles.size()) - 1;
      for (int spanIndex = p; spanIndex <= lastSpan; ++spanIndex) {
        const double spanLow = knots[spanIndex];
        const double spanHigh = knots[spanIndex + 1];
        const double halfSpan = 0.5 * (spanHigh - spanLow);
        if (halfSpan <= kTolerance) {
          continue;
        }
        const double spanMid = 0.5 * (spanLow + spanHigh);
        for (int nodeIndex = 0; nodeIndex < order; ++nodeIndex) {
          const double knotValue = spanMid + halfSpan * nodes[nodeIndex];
          Vec2 point;
          Vec2 derivative;
          bsplineEval(knotValue, point, derivative);
          area += 0.5 * (point.uCoord * derivative.vCoord - point.vCoord * derivative.uCoord) *
                  nodeWeights[nodeIndex] * halfSpan;
        }
      }
      return area;
    }
    return 0.5 * (radius * center.uCoord * (std::sin(endAngle) - std::sin(startAngle)) -
                  radius * center.vCoord * (std::cos(endAngle) - std::cos(startAngle)) +
                  radius * radius * (endAngle - startAngle));
  }

  /// How far this curve's representation can sit from the curve, in parametric units: kBSplineFlatness for a B-spline, else 0.
  double representationTolerance() const { return kind == CurveKind::BSpline ? kBSplineFlatness : 0.; }

  /// B-spline only: true if \a point is within sqrt(\a bandSq) of the polyline, else adds its rightward crossings.
  /// One walk of the polyline with the arithmetic of closestPoint and rightwardCrossings.
  bool bsplineBandOrCrossings(const Vec2& point, double bandSq, int& crossings) const
  {
    const auto& polyline = bsplineSamples();
    if (polyline.size() < 2) {
      return surface::distanceSq(point, startPoint()) <= bandSq;
    }
    int found = 0;
    for (size_t index = 0; index + 1 < polyline.size(); ++index) {
      const Vec2 segmentStart = polyline[index];
      const Vec2 segmentEnd = polyline[index + 1];
      const Vec2 segmentVector = segmentEnd - segmentStart;
      const double segmentLengthSq =
        segmentVector.uCoord * segmentVector.uCoord + segmentVector.vCoord * segmentVector.vCoord;
      double projection = 0.;
      if (segmentLengthSq > kToleranceSq) {
        projection = ((point.uCoord - segmentStart.uCoord) * segmentVector.uCoord +
                      (point.vCoord - segmentStart.vCoord) * segmentVector.vCoord) /
                     segmentLengthSq;
        projection = std::max(0., std::min(1., projection));
      }
      const Vec2 candidate{segmentStart.uCoord + projection * segmentVector.uCoord,
                           segmentStart.vCoord + projection * segmentVector.vCoord};
      if (surface::distanceSq(point, candidate) <= bandSq) {
        return true;
      }
      const bool firstAbove = segmentStart.vCoord > point.vCoord;
      const bool secondAbove = segmentEnd.vCoord > point.vCoord;
      if (firstAbove != secondAbove) {
        const double intersectU =
          segmentStart.uCoord + (point.vCoord - segmentStart.vCoord) * (segmentEnd.uCoord - segmentStart.uCoord) /
                                  (segmentEnd.vCoord - segmentStart.vCoord);
        if (point.uCoord < intersectU) {
          ++found;
        }
      }
    }
    crossings += found;
    return false;
  }

  /// Rightward crossings of a horizontal ray from \a point, with the caller's canonical endpoints so that seams stay consistent.
  int rightwardCrossings(const Vec2& point, const Vec2& canonicalStart, const Vec2& canonicalEnd) const
  {
    auto segmentCrossing = [&](const Vec2& first, const Vec2& second, double exactIntersectU) {
      const bool firstAbove = first.vCoord > point.vCoord;
      const bool secondAbove = second.vCoord > point.vCoord;
      if (firstAbove == secondAbove) {
        return false;
      }
      return point.uCoord < exactIntersectU;
    };

    if (kind == CurveKind::Line) {
      const bool firstAbove = canonicalStart.vCoord > point.vCoord;
      const bool secondAbove = canonicalEnd.vCoord > point.vCoord;
      if (firstAbove == secondAbove) {
        return 0;
      }
      const double intersectU = canonicalStart.uCoord + (point.vCoord - canonicalStart.vCoord) *
                                                          (canonicalEnd.uCoord - canonicalStart.uCoord) /
                                                          (canonicalEnd.vCoord - canonicalStart.vCoord);
      return (point.uCoord < intersectU) ? 1 : 0;
    }

    if (kind == CurveKind::BSpline) {
      // the lines' half-open segment-crossing rule over the polyline, whose ends are the canonical seam vertices
      const auto& polyline = bsplineSamples();
      if (polyline.size() < 2) {
        return 0;
      }
      int crossings = 0;
      for (size_t index = 0; index + 1 < polyline.size(); ++index) {
        // No substitution here: the polyline already ends on the loop-canonical vertices (see
        // setCanonicalEndpoints), so this is the same boundary closestPoint measures against.
        const Vec2 first = polyline[index];
        const Vec2 second = polyline[index + 1];
        const bool firstAbove = first.vCoord > point.vCoord;
        const bool secondAbove = second.vCoord > point.vCoord;
        if (firstAbove == secondAbove) {
          continue;
        }
        const double intersectU =
          first.uCoord + (point.vCoord - first.vCoord) * (second.uCoord - first.uCoord) /
                           (second.vCoord - first.vCoord);
        if (point.uCoord < intersectU) {
          ++crossings;
        }
      }
      return crossings;
    }

    // split the arc into v-monotonic sub-arcs at its extreme angles, where the crossing u is exact
    const double totalSweep = sweep();
    if (std::abs(totalSweep) <= kTolerance || radius <= kTolerance) {
      return 0;
    }
    std::array<double, 8> breakParameters{};
    int breakCount = 0;
    breakParameters[breakCount++] = 0.;
    const double lowAngle = std::min(startAngle, endAngle);
    const double highAngle = std::max(startAngle, endAngle);
    const int firstK = static_cast<int>(std::floor((lowAngle - kHalfPi) / kPi)) - 1;
    const int lastK = static_cast<int>(std::ceil((highAngle - kHalfPi) / kPi)) + 1;
    for (int k = firstK; k <= lastK && breakCount < 7; ++k) {
      const double extremeAngle = kHalfPi + k * kPi;
      if (extremeAngle <= lowAngle + kTolerance || extremeAngle >= highAngle - kTolerance) {
        continue;
      }
      const double extremeParameter = (extremeAngle - startAngle) / totalSweep;
      if (extremeParameter > kTolerance && extremeParameter < 1. - kTolerance) {
        breakParameters[breakCount++] = extremeParameter;
      }
    }
    breakParameters[breakCount++] = 1.;
    std::sort(breakParameters.begin(), breakParameters.begin() + breakCount);

    double ratio = (point.vCoord - center.vCoord) / radius;
    ratio = std::max(-1., std::min(1., ratio));
    const double cosMagnitude = std::sqrt(std::max(0., 1. - ratio * ratio));

    int crossings = 0;
    for (int index = 0; index + 1 < breakCount; ++index) {
      const Vec2 subStart = (index == 0) ? canonicalStart : pointAt(breakParameters[index]);
      const Vec2 subEnd = (index + 2 == breakCount) ? canonicalEnd : pointAt(breakParameters[index + 1]);
      const double midAngle = startAngle + 0.5 * (breakParameters[index] + breakParameters[index + 1]) * totalSweep;
      const double cosSign = std::cos(midAngle) >= 0. ? 1. : -1.;
      const double intersectU = center.uCoord + cosSign * radius * cosMagnitude;
      if (segmentCrossing(subStart, subEnd, intersectU)) {
        ++crossings;
      }
    }
    return crossings;
  }

  /// Reverse the curve's direction in place (start <-> end), keeping the same geometric image.
  void reverseInPlace()
  {
    if (hasCanonicalEndpoints) {
      std::swap(canonicalStart, canonicalEnd);
      bsplineCache.clear();
    }
    if (kind == CurveKind::Line) {
      std::swap(lineStart, lineEnd);
    } else if (kind == CurveKind::Arc) {
      std::swap(startAngle, endAngle);
    } else {
      // B-spline: reverse the poles/weights and complement the knot vector about its span so the
      // parametrization runs the other way (knots stay non-decreasing and clamped).
      std::reverse(poles.begin(), poles.end());
      if (!weights.empty()) {
        std::reverse(weights.begin(), weights.end());
      }
      const double knotSum = knots.front() + knots.back();
      std::vector<double> reversedKnots(knots.size());
      for (size_t index = 0; index < knots.size(); ++index) {
        reversedKnots[index] = knotSum - knots[knots.size() - 1 - index];
      }
      knots = std::move(reversedKnots);
      bsplineCache.clear(); // geometry order changed; recompute lazily
    }
  }
};

/// One closed, oriented boundary loop of Curve2D segments: outer loops wind counter-clockwise, holes clockwise.
struct CurveWire {
  std::vector<Curve2D> curves;
  WireRole role = WireRole::Outer;
  /// The largest representationTolerance() over the curves, fixed when the curves are set.
  double mRepresentationTolerance = 0.;

  /// For each stored curve its input index; reverse() is the only reordering, and sidecar v3 edge identities key on it.
  std::vector<int> sourceCurve;

  /// The stored curve that came from input curve \a inputIndex, or -1 if there is none.
  int storedIndexOfSource(int inputIndex) const
  {
    for (size_t index = 0; index < sourceCurve.size(); ++index) {
      if (sourceCurve[index] == inputIndex) {
        return static_cast<int>(index);
      }
    }
    return -1;
  }

  /// Build and validate the wire from an ordered closed list of curves, joining within \a joinTolerance through \a metric.
  bool initialize(const std::vector<Curve2D>& inputCurves, WireRole wireRole, WireStatus& status,
                  const ParametricMetric& metric = {}, double joinTolerance = kWireJoinTolerance)
  {
    role = wireRole;
    curves = inputCurves;
    mRepresentationTolerance = 0.;
    for (const auto& curve : curves) {
      mRepresentationTolerance = std::max(mRepresentationTolerance, curve.representationTolerance());
    }
    sourceCurve.resize(curves.size());
    for (size_t index = 0; index < curves.size(); ++index) {
      sourceCurve[index] = static_cast<int>(index);
    }

    if (curves.empty()) {
      status = WireStatus::TooFewVertices;
      return false;
    }
    for (size_t index = 0; index < curves.size(); ++index) {
      if (!curves[index].valid()) {
        status = WireStatus::NonFinite;
        return false;
      }
      const Vec2 currentEnd = curves[index].endPoint();
      const Vec2 nextStart = curves[(index + 1) % curves.size()].startPoint();
      if (metric.distanceSq(currentEnd, nextStart) > joinTolerance * joinTolerance) {
        status = WireStatus::Open;
        return false;
      }
    }

    // one vertex value per seam, given to both curves that meet there
    for (size_t index = 0; index < curves.size(); ++index) {
      curves[index].setCanonicalEndpoints(curves[index].startPoint(),
                                          curves[(index + 1) % curves.size()].startPoint());
    }

    const double area = signedArea();
    if (std::abs(area) <= kAreaTolerance) {
      status = WireStatus::ZeroArea;
      return false;
    }

    const bool wantPositiveArea = (role == WireRole::Outer);
    if ((area > 0.) != wantPositiveArea) {
      reverse();
      status = WireStatus::Reversed;
      fillBSplineCaches();
      return true;
    }
    status = WireStatus::Valid;
    fillBSplineCaches();
    return true;
  }

  /// Fill every B-spline's polyline cache now, so that const navigation queries only read it.
  void fillBSplineCaches() const
  {
    for (const auto& curve : curves) {
      if (curve.kind == CurveKind::BSpline) {
        curve.bsplineSamples();
      }
    }
  }

  /// The widest gap between the loop's representation and its boundary, in parametric units; 0 for lines and arcs.
  double representationTolerance() const { return mRepresentationTolerance; }

  /// Reverse the loop orientation in place (order and per-curve direction).
  void reverse()
  {
    std::reverse(curves.begin(), curves.end());
    std::reverse(sourceCurve.begin(), sourceCurve.end());
    for (auto& curve : curves) {
      curve.reverseInPlace();
    }
  }

  /// True if any curve of the loop is a B-spline (whose trimmed-face capacity is only numerically
  /// integrated, so the owning surface must report capacityIsExact() == false).
  bool hasBSpline() const
  {
    for (const auto& curve : curves) {
      if (curve.kind == CurveKind::BSpline) {
        return true;
      }
    }
    return false;
  }

  /// Exact signed area enclosed by the loop (positive when counter-clockwise).
  double signedArea() const
  {
    double area = 0.;
    for (const auto& curve : curves) {
      area += curve.signedAreaContribution();
    }
    return area;
  }

  /// Add the loop's conservative extent, a B-spline's pole hull included, to a parametric bounding box.
  void parametricBounds(Vec2& lower, Vec2& upper) const
  {
    for (const auto& curve : curves) {
      curve.extendBounds(lower, upper);
    }
  }

  /// Add the loop's extent measured on the curves to a parametric bounding box; use it to reject a wire as too big.
  void tightParametricBounds(Vec2& lower, Vec2& upper) const
  {
    for (const auto& curve : curves) {
      curve.extendTightBounds(lower, upper);
    }
  }

  /// Half-width of the on-boundary band in parametric units: the larger of the length floor and the representation tolerance.
  /// A degenerate metric (a pole, an apex) leaves only the representation term.
  double boundaryBand(double lengthFloor) const { return std::max(lengthFloor, mRepresentationTolerance); }

  /// Classify a point against the loop with band floor \a lengthFloor: Boundary within the band, else the crossing parity.
  WireClassification classify(const Vec2& point, double lengthFloor) const
  {
    const double band = boundaryBand(lengthFloor);
    const double bandSq = band * band;
    // Each curve's polyline already ends on the loop-canonical seam vertices, so the half-open
    // crossing convention stays consistent across seams without any substitution here.
    int crossings = 0;
    for (const auto& curve : curves) {
      if (curve.kind == CurveKind::BSpline) {
        if (curve.bsplineBandOrCrossings(point, bandSq, crossings)) {
          return WireClassification::Boundary;
        }
      } else if (curve.distanceSq(point) <= bandSq) {
        return WireClassification::Boundary;
      } else {
        crossings += curve.rightwardCrossings(point, curve.loopStart(), curve.loopEnd());
      }
    }
    return (crossings % 2 == 1) ? WireClassification::Inside : WireClassification::Outside;
  }

  /// \a metric only sizes the on-boundary band; the winding count is topological.
  WireClassification classify(const Vec2& point, const ParametricMetric& metric = {}) const
  {
    return classify(point, trimLengthFloor(metric, point));
  }

  /// Ordered, closed boundary polyline; arcs are sampled into \a segmentsPerArc chords. This is a
  /// mesh-independent hook for visualization and tessellated fallback of curved boundaries.
  std::vector<Vec2> sampledBoundary(int segmentsPerArc = kArcSamples) const
  {
    std::vector<Vec2> samples;
    if (curves.empty()) {
      return samples;
    }
    for (const auto& curve : curves) {
      if (curve.kind == CurveKind::Line) {
        samples.push_back(curve.startPoint());
      } else if (curve.kind == CurveKind::BSpline) {
        // adaptively flatten and append every sample except the closing one (the next curve's
        // start reproduces it)
        std::vector<Vec2> curveSamples;
        curve.bsplineSampleInto(curveSamples);
        for (size_t index = 0; index + 1 < curveSamples.size(); ++index) {
          samples.push_back(curveSamples[index]);
        }
      } else {
        // chords scale with the arc's sweep, so a rim shared with a quadric wall samples identical vertices
        const int arcSteps =
          std::max(1, static_cast<int>(std::lround(segmentsPerArc * std::abs(curve.sweep()) / kTwoPi)));
        for (int step = 0; step < arcSteps; ++step) {
          samples.push_back(curve.pointAt(static_cast<double>(step) / arcSteps));
        }
      }
    }
    samples.push_back(samples.front());
    return samples;
  }
};

/// \name Curve-wire trim helpers for quadric parametric domains (u = phi, v = height or theta)
/// @{

/// Shift \a angle by whole turns to lie as close as possible to the window [uMin, uMax].
inline double unwrapAngleInto(double angle, double uMin, double uMax)
{
  const double windowCenter = 0.5 * (uMin + uMax);
  return angle - kTwoPi * std::round((angle - windowCenter) / kTwoPi);
}

/// Whether a parametric point is in a curve-wire trim (outer loop minus holes); \a boundary reports an on-boundary hit.
inline bool curveTrimContains(const CurveWire& outerWire, const std::vector<CurveWire>& innerWires,
                              const Vec2& point, bool* boundary = nullptr,
                              const ParametricMetric& metric = {})
{
  if (boundary != nullptr) {
    *boundary = false;
  }
  const double lengthFloor = trimLengthFloor(metric, point);
  const auto outerClassification = outerWire.classify(point, lengthFloor);
  if (outerClassification == WireClassification::Outside) {
    return false;
  }
  if (outerClassification == WireClassification::Boundary) {
    if (boundary != nullptr) {
      *boundary = true;
    }
    return true;
  }
  for (const auto& innerWire : innerWires) {
    const auto innerClassification = innerWire.classify(point, lengthFloor);
    if (innerClassification == WireClassification::Boundary) {
      if (boundary != nullptr) {
        *boundary = true;
      }
      return true;
    }
    if (innerClassification == WireClassification::Inside) {
      return false;
    }
  }
  return true;
}

/// Gauss-Legendre nodes per contour sub-interval, and the widest u span one sub-interval covers.
inline constexpr int kContourQuadratureOrder = 20;
inline constexpr double kContourMaxSpanU = 0.25 * kPi;

/// Integrate F(u, v) dv along one directed curve of a trim wire, from \a from to \a to in the
/// curve's own [0, 1] parameter.
template <typename Antiderivative>
double contourIntegralAlongCurve(const Curve2D& curve, const Antiderivative& antiderivative, double from,
                                 double to)
{
  static thread_local std::vector<double> nodes;
  static thread_local std::vector<double> weights;
  if (static_cast<int>(nodes.size()) != kContourQuadratureOrder) {
    gaussLegendre(kContourQuadratureOrder, nodes, weights);
  }
  // split at the interior knots, then into pieces whose u travel is at most kContourMaxSpanU
  static thread_local std::vector<double> breakpoints;
  breakpoints.clear();
  breakpoints.push_back(from);
  curve.appendInteriorKnots(std::min(from, to), std::max(from, to), breakpoints);
  std::sort(breakpoints.begin() + 1, breakpoints.end(),
            [forward = (to >= from)](double first, double second) { return forward ? first < second : first > second; });
  breakpoints.push_back(to);

  double total = 0.;
  for (size_t segment = 0; segment + 1 < breakpoints.size(); ++segment) {
    const double segmentFrom = breakpoints[segment];
    const double segmentTo = breakpoints[segment + 1];
    if (segmentFrom == segmentTo) {
      continue;
    }
    const double travelU = curve.uVariation(std::min(segmentFrom, segmentTo), std::max(segmentFrom, segmentTo));
    const int pieces = std::max(1, static_cast<int>(std::ceil(travelU / kContourMaxSpanU)));
    for (int piece = 0; piece < pieces; ++piece) {
      const double low = segmentFrom + (segmentTo - segmentFrom) * piece / pieces;
      const double high = segmentFrom + (segmentTo - segmentFrom) * (piece + 1) / pieces;
      const double half = 0.5 * (high - low);
      const double mid = 0.5 * (high + low);
      for (int nodeIndex = 0; nodeIndex < kContourQuadratureOrder; ++nodeIndex) {
        const double parameter = mid + half * nodes[nodeIndex];
        const Vec2 point = curve.pointAt(parameter);
        const Vec2 derivative = curve.derivativeAt(parameter);
        total += weights[nodeIndex] * half * antiderivative(point.uCoord, point.vCoord) * derivative.vCoord;
      }
    }
  }
  return total;
}

/// Green's theorem over a wire-trimmed patch: the double integral of f is the contour integral of F dv, F the u-antiderivative of f; seams are bridged.
template <typename Antiderivative>
double integrateOverCurveTrimByParts(const CurveWire& outerWire, const std::vector<CurveWire>& innerWires,
                                     const Antiderivative& antiderivative)
{
  const auto loopIntegral = [&antiderivative](const CurveWire& wire) {
    double total = 0.;
    for (size_t index = 0; index < wire.curves.size(); ++index) {
      const auto& curve = wire.curves[index];
      total += contourIntegralAlongCurve(curve, antiderivative, 0., 1.);
      // seam bridge: a straight run from this curve's end to the next curve's start
      const Vec2 seamFrom = curve.endPoint();
      const Vec2 seamTo = wire.curves[(index + 1) % wire.curves.size()].startPoint();
      const double deltaV = seamTo.vCoord - seamFrom.vCoord;
      if (deltaV != 0.) {
        const Curve2D bridge = Curve2D::makeLine(seamFrom, seamTo);
        total += contourIntegralAlongCurve(bridge, antiderivative, 0., 1.);
      }
    }
    return total;
  };

  double total = loopIntegral(outerWire);
  for (const auto& innerWire : innerWires) {
    total += loopIntegral(innerWire);
  }
  return total;
}

/// Midpoint-rule integral of \a integrand over the trimmed region; kept as the independent check of the contour form.
template <typename Integrand>
double integrateOverCurveTrim(const CurveWire& outerWire, const std::vector<CurveWire>& innerWires,
                              const Integrand& integrand, int samplesPerAxis = 128)
{
  Vec2 lower{std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()};
  Vec2 upper{-std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()};
  outerWire.parametricBounds(lower, upper);
  if (!finite(lower) || !finite(upper) || samplesPerAxis < 1) {
    return 0.;
  }
  const double stepU = (upper.uCoord - lower.uCoord) / samplesPerAxis;
  const double stepV = (upper.vCoord - lower.vCoord) / samplesPerAxis;
  const double cellArea = stepU * stepV;
  double sum = 0.;
  for (int indexU = 0; indexU < samplesPerAxis; ++indexU) {
    const double uCoord = lower.uCoord + (indexU + 0.5) * stepU;
    for (int indexV = 0; indexV < samplesPerAxis; ++indexV) {
      const double vCoord = lower.vCoord + (indexV + 0.5) * stepV;
      if (curveTrimContains(outerWire, innerWires, {uCoord, vCoord})) {
        sum += integrand(uCoord, vCoord) * cellArea;
      }
    }
  }
  return sum;
}

/// Build validated outer and inner trim wires and the outer loop's parametric bounds; rejects a trim wider than a turn in u.
inline bool buildCurveTrim(const std::vector<Curve2D>& outerTrim,
                           const std::vector<std::vector<Curve2D>>& innerTrims, CurveWire& outerWire,
                           std::vector<CurveWire>& innerWires, Vec2& lower, Vec2& upper,
                           std::string& errorMessage, const ParametricMetric& metric = {},
                           double joinTolerance = kWireJoinTolerance)
{
  WireStatus status = WireStatus::Valid;
  if (!outerWire.initialize(outerTrim, WireRole::Outer, status, metric, joinTolerance)) {
    errorMessage = std::string("quadric outer trim wire invalid: ") + wireStatusMessage(status);
    return false;
  }
  innerWires.clear();
  innerWires.reserve(innerTrims.size());
  for (const auto& innerLoop : innerTrims) {
    CurveWire innerWire;
    WireStatus innerStatus = WireStatus::Valid;
    if (!innerWire.initialize(innerLoop, WireRole::Inner, innerStatus, metric, joinTolerance)) {
      errorMessage = std::string("quadric inner trim wire invalid: ") + wireStatusMessage(innerStatus);
      return false;
    }
    innerWires.push_back(std::move(innerWire));
  }
  lower = {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()};
  upper = {-std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()};
  outerWire.parametricBounds(lower, upper);
  if (!finite(lower) || !finite(upper)) {
    errorMessage = "quadric trim wire has non-finite parametric bounds";
    return false;
  }
  if (upper.uCoord - lower.uCoord > kTwoPi + kTolerance) {
    // the pole hull can overshoot the curve; re-measure on the curves before refusing
    Vec2 tightLower{std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()};
    Vec2 tightUpper{-std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()};
    outerWire.tightParametricBounds(tightLower, tightUpper);
    if (!finite(tightLower) || !finite(tightUpper) || tightUpper.uCoord - tightLower.uCoord > kTwoPi + kTolerance) {
      errorMessage = "quadric trim wire spans more than a full turn in phi";
      return false;
    }
    // the wire is admissible; keep the tight box, since the conservative one is not a valid
    // parametric window for a periodic coordinate once it exceeds a full turn
    lower = tightLower;
    upper = tightUpper;
  }
  return true;
}

/// Sub-sample a curve-wire loop so its u span is chorded at \a segmentsPerTurn per turn, matching neighbouring rims.
inline std::vector<Vec2> sampleCurveWireByU(const CurveWire& wire, int segmentsPerTurn = kArcSamples)
{
  std::vector<Vec2> samples;
  for (const auto& curve : wire.curves) {
    if (curve.kind == CurveKind::BSpline) {
      // adaptively flatten in the parameter domain; append every sample except the closing one
      std::vector<Vec2> curveSamples;
      curve.bsplineSampleInto(curveSamples);
      for (size_t index = 0; index + 1 < curveSamples.size(); ++index) {
        samples.push_back(curveSamples[index]);
      }
      continue;
    }
    Vec2 lower{std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()};
    Vec2 upper{-std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()};
    curve.extendBounds(lower, upper);
    const double uSpan = upper.uCoord - lower.uCoord;
    int steps = std::max(1, static_cast<int>(std::lround(segmentsPerTurn * uSpan / kTwoPi)));
    if (curve.kind == CurveKind::Arc) {
      steps = std::max(steps, static_cast<int>(std::lround(segmentsPerTurn * std::abs(curve.sweep()) / kTwoPi)));
      steps = std::max(steps, 1);
    }
    for (int step = 0; step < steps; ++step) {
      samples.push_back(curve.pointAt(static_cast<double>(step) / steps));
    }
  }
  return samples;
}

/// Append the display triangulation of a wire-trimmed quadric patch: the sampled outer loop, ear-clipped; holes are omitted.
template <typename MapUV>
void appendCurveTrimMesh(const CurveWire& outerWire, const MapUV& mapUV, std::vector<Vec3>& vertices,
                         std::vector<std::array<int, 3>>& triangles)
{
  SurfaceWire sampledWire;
  sampledWire.vertices = sampleCurveWireByU(outerWire);
  if (sampledWire.vertices.size() < 3) {
    return;
  }
  const int firstVertexIndex = static_cast<int>(vertices.size());
  for (const auto& sample : sampledWire.vertices) {
    vertices.push_back(mapUV(sample.uCoord, sample.vCoord));
  }
  for (const auto& triangle : triangulateSimpleWire(sampledWire)) {
    triangles.push_back(
      {firstVertexIndex + triangle[0], firstVertexIndex + triangle[1], firstVertexIndex + triangle[2]});
  }
}

/// Append the directed 3D boundary edges of a wire-trimmed quadric patch; a negative \a orientationSign reverses them.
template <typename MapUV>
void appendCurveTrimEdges(const CurveWire& outerWire, const std::vector<CurveWire>& innerWires,
                          const MapUV& mapUV, double orientationSign,
                          std::vector<std::pair<Vec3, Vec3>>& edges)
{
  auto appendLoop = [&](const CurveWire& wire) {
    const auto samples = sampleCurveWireByU(wire);
    const size_t sampleCount = samples.size();
    for (size_t sampleIndex = 0; sampleIndex < sampleCount; ++sampleIndex) {
      const Vec2& current = samples[sampleIndex];
      const Vec2& next = samples[(sampleIndex + 1) % sampleCount];
      const Vec3 edgeStart = mapUV(current.uCoord, current.vCoord);
      const Vec3 edgeEnd = mapUV(next.uCoord, next.vCoord);
      if (orientationSign >= 0.) {
        edges.emplace_back(edgeStart, edgeEnd);
      } else {
        edges.emplace_back(edgeEnd, edgeStart);
      }
    }
  };
  appendLoop(outerWire);
  for (const auto& innerWire : innerWires) {
    appendLoop(innerWire);
  }
}

/// Samples per trim curve when measuring a shared edge's deviation; it never enters a verdict.
inline constexpr int kSharedEdgeSamples = 33;

/// Sample input curve \a index of a curve-wire trim into 3D through \a mapUV; false when out of range or not traceable.
template <typename MapUV>
bool sampleTrimCurveOfCurveWires(const CurveWire& outerWire, const std::vector<CurveWire>& innerWires,
                                 size_t index, const MapUV& mapUV, std::vector<Vec3>& samples)
{
  const CurveWire* wire = nullptr;
  size_t local = index;
  if (local < outerWire.curves.size()) {
    wire = &outerWire;
  } else {
    local -= outerWire.curves.size();
    for (const auto& innerWire : innerWires) {
      if (local < innerWire.curves.size()) {
        wire = &innerWire;
        break;
      }
      local -= innerWire.curves.size();
    }
  }
  if (wire == nullptr) {
    return false;
  }
  const int stored = wire->storedIndexOfSource(static_cast<int>(local));
  if (stored < 0) {
    return false;
  }
  const Curve2D& curve = wire->curves[static_cast<size_t>(stored)];
  samples.clear();
  samples.reserve(kSharedEdgeSamples);
  for (int step = 0; step < kSharedEdgeSamples; ++step) {
    const Vec2 uv = curve.pointAt(static_cast<double>(step) / (kSharedEdgeSamples - 1));
    samples.push_back(mapUV(uv.uCoord, uv.vCoord));
  }
  return true;
}

/// The same for a polygon (vertex-ring) trim, whose curves are all straight segments.
template <typename MapUV>
bool sampleTrimCurveOfSurfaceWires(const SurfaceWire& outerWire, const std::vector<SurfaceWire>& innerWires,
                                   size_t index, const MapUV& mapUV, std::vector<Vec3>& samples)
{
  const SurfaceWire* wire = nullptr;
  size_t local = index;
  if (local < outerWire.vertices.size()) {
    wire = &outerWire;
  } else {
    local -= outerWire.vertices.size();
    for (const auto& innerWire : innerWires) {
      if (local < innerWire.vertices.size()) {
        wire = &innerWire;
        break;
      }
      local -= innerWire.vertices.size();
    }
  }
  if (wire == nullptr) {
    return false;
  }
  const int stored = wire->storedIndexOfSource(static_cast<int>(local));
  if (stored < 0) {
    return false;
  }
  const SurfaceEdge segment = wire->edge(stored);
  samples.clear();
  samples.push_back(mapUV(segment.start.uCoord, segment.start.vCoord));
  samples.push_back(mapUV(segment.end.uCoord, segment.end.vCoord));
  return true;
}
/// @}

/// One trim loop of one face as an ordered 3D polyline, compared with other faces' rims as a curve.
struct SurfaceRim {
  int surfaceIndex = -1;    ///< index of the owning face in the solid's surface list
  bool closed = false;      ///< the polyline returns to its own first point
  std::vector<Vec3> points; ///< consecutive samples; a closed rim does not repeat the first point
};

/// Chain a face's directed chords into rims by matching endpoints within kTolerance, appending them to \a rims.
inline void assembleRims(const std::vector<std::pair<Vec3, Vec3>>& edges, std::vector<SurfaceRim>& rims)
{
  if (edges.empty()) {
    return;
  }
  auto quantize = [](double value) { return static_cast<int64_t>(std::llround(value / kTolerance)); };
  using VertexKey = std::tuple<int64_t, int64_t, int64_t>;
  auto keyOf = [&](const Vec3& point) {
    return VertexKey{quantize(point.xCoord), quantize(point.yCoord), quantize(point.zCoord)};
  };

  // cancel reversed duplicate chords (a self-closing seam) before chaining, keyed by their shared midpoint
  std::vector<bool> consumed(edges.size(), false);
  std::map<VertexKey, std::vector<size_t>> edgesByMidpoint;
  for (size_t edgeIndex = 0; edgeIndex < edges.size(); ++edgeIndex) {
    const Vec3 midpoint = (edges[edgeIndex].first + edges[edgeIndex].second) * 0.5;
    const auto [xKey, yKey, zKey] = keyOf(midpoint);
    bool cancelled = false;
    for (int64_t dx = -1; dx <= 1 && !cancelled; ++dx) {
      for (int64_t dy = -1; dy <= 1 && !cancelled; ++dy) {
        for (int64_t dz = -1; dz <= 1 && !cancelled; ++dz) {
          const auto found = edgesByMidpoint.find(VertexKey{xKey + dx, yKey + dy, zKey + dz});
          if (found == edgesByMidpoint.end()) {
            continue;
          }
          for (const size_t candidate : found->second) {
            if (consumed[candidate] ||
                distanceSq(edges[candidate].first, edges[edgeIndex].second) > kToleranceSq ||
                distanceSq(edges[candidate].second, edges[edgeIndex].first) > kToleranceSq) {
              continue;
            }
            consumed[candidate] = true;
            consumed[edgeIndex] = true;
            cancelled = true;
            break;
          }
        }
      }
    }
    if (!cancelled) {
      edgesByMidpoint[keyOf(midpoint)].push_back(edgeIndex);
    }
  }

  std::map<VertexKey, std::vector<size_t>> edgesByStart;
  for (size_t edgeIndex = 0; edgeIndex < edges.size(); ++edgeIndex) {
    if (!consumed[edgeIndex]) {
      edgesByStart[keyOf(edges[edgeIndex].first)].push_back(edgeIndex);
    }
  }

  // A vertex can land either side of a lattice boundary, so probe the 27 neighbouring cells and
  // accept the first unused chord whose start really is within kTolerance.
  auto findSuccessor = [&](const Vec3& point) -> long long {
    const auto [xKey, yKey, zKey] = keyOf(point);
    for (int64_t dx = -1; dx <= 1; ++dx) {
      for (int64_t dy = -1; dy <= 1; ++dy) {
        for (int64_t dz = -1; dz <= 1; ++dz) {
          const auto found = edgesByStart.find(VertexKey{xKey + dx, yKey + dy, zKey + dz});
          if (found == edgesByStart.end()) {
            continue;
          }
          for (const size_t candidate : found->second) {
            if (!consumed[candidate] && distanceSq(edges[candidate].first, point) <= kToleranceSq) {
              return static_cast<long long>(candidate);
            }
          }
        }
      }
    }
    return -1;
  };

  for (size_t seed = 0; seed < edges.size(); ++seed) {
    if (consumed[seed]) {
      continue;
    }
    consumed[seed] = true;
    SurfaceRim rim;
    rim.points.push_back(edges[seed].first);
    rim.points.push_back(edges[seed].second);
    while (true) {
      if (distanceSq(rim.points.back(), rim.points.front()) <= kToleranceSq) {
        rim.closed = true;
        rim.points.pop_back(); // a closed rim does not repeat its first point
        break;
      }
      const long long next = findSuccessor(rim.points.back());
      if (next < 0) {
        break; // an open chain: the face's boundary is not a set of closed loops
      }
      consumed[static_cast<size_t>(next)] = true;
      rim.points.push_back(edges[static_cast<size_t>(next)].second);
    }
    if (rim.points.size() >= 2) {
      rims.push_back(std::move(rim));
    }
  }
}

/// Abstract analytic surface patch: one support surface plus its trim, with the kernels the navigation needs.
class BoundedSurface
{
 public:
  virtual ~BoundedSurface() = default;

  /// \name Boundary edge identity (sidecar v3): the source edges bounding this face; empty means not stated
  /// @{
  struct BoundaryEdgeRef {
    uint32_t edgeId = 0;     ///< index into the model's edge table; identity, not a coordinate
    bool reversed = false;   ///< this face runs against the edge's own direction
    bool degenerate = false; ///< a cone apex / sphere pole: one point, no length, no partner
    /// Whether trim curve \a i exists to sample for edge \a i; false for a parametric-rectangle trim.
    bool anchored = false;
  };

  void setBoundaryEdges(std::vector<BoundaryEdgeRef> refs) { mBoundaryEdges = std::move(refs); }
  const std::vector<BoundaryEdgeRef>& boundaryEdges() const { return mBoundaryEdges; }

  /// Sample trim curve \a index into 3D, in construction order; false when this face has no such curve.
  virtual bool sampleTrimCurve(size_t index, std::vector<Vec3>& samples) const
  {
    (void)index;
    (void)samples;
    return false;
  }
  /// @}

  /// Accumulate a conservative axis-aligned bounding box of the trimmed patch.
  virtual void conservativeBounds(Vec3& lower, Vec3& upper) const = 0;

  /// One axis-aligned cover box of the sub-patch BVH, as a (lower corner, upper corner) pair.
  using CoverBox = std::pair<Vec3, Vec3>;

  /// Append cover boxes whose union holds the trimmed patch and every point that can realise distanceSqToPatch.
  /// Spheres and tori realise on their whole surface, so they cover it all; the default is conservativeBounds().
  virtual void appendCoverBoxes(std::vector<CoverBox>& boxes) const
  {
    // conservativeBounds only accumulates, so the corners start beyond any geometry
    constexpr double kBig = std::numeric_limits<double>::max();
    CoverBox box{Vec3{kBig, kBig, kBig}, Vec3{-kBig, -kBig, -kBig}};
    conservativeBounds(box.first, box.second);
    boxes.push_back(box);
  }

  /// True if the 3D point lies on the trimmed patch within tolerance.
  virtual bool containsPointOnSurface(const Vec3& point) const = 0;

  /// Append every hit of the ray with the trimmed patch in [minDistance, maxDistance], with the outward normal; no tangential grazes.
  virtual void appendIntersections(const Vec3& rayOrigin, const Vec3& rayDirection, double minDistance,
                                   double maxDistance, std::vector<RayHit>& hits) const = 0;

  /// Squared distance from a 3D point to the trimmed patch (used for Safety).
  virtual double distanceSqToPatch(const Vec3& point) const = 0;

  /// Outward-oriented normal at (or nearest to) the given point.
  virtual Vec3 normalAt(const Vec3& point) const = 0;

  /// The first fundamental form at \a uv, turning parametric displacements into 3D lengths; it varies over the domain and gUU vanishes at poles.
  virtual void parametricMetric(const Vec2& uv, double& gUU, double& gUV, double& gVV) const = 0;

  /// The 3D length squared spanned by a parametric displacement \a delta starting at \a uv.
  double parametricLengthSqAt(const Vec2& uv, const Vec2& delta) const
  {
    double gUU = 0.;
    double gUV = 0.;
    double gVV = 0.;
    parametricMetric(uv, gUU, gUV, gVV);
    return parametricLengthSq(gUU, gUV, gVV, delta);
  }

  /// Signed divergence-theorem contribution to the enclosed volume.
  virtual double capacityContribution() const = 0;

  /// Whether capacityContribution() is analytically exact for this surface.
  virtual bool capacityIsExact() const = 0;

  /// Append this patch's visualization triangulation (navigation must never depend on it).
  virtual void appendDisplayMesh(std::vector<Vec3>& vertices,
                                 std::vector<std::array<int, 3>>& triangles) const = 0;

  /// Append the 3D directed boundary edges of the patch, for solid-closure validation.
  virtual void appendDirectedEdges(std::vector<std::pair<Vec3, Vec3>>& edges) const = 0;

  /// Append the trim boundary as rims, one polyline per loop; the default chains appendDirectedEdges().
  virtual void appendRims(std::vector<SurfaceRim>& rims) const
  {
    std::vector<std::pair<Vec3, Vec3>> edges;
    appendDirectedEdges(edges);
    assembleRims(edges, rims);
  }

 protected:
  std::vector<BoundaryEdgeRef> mBoundaryEdges;
};

/// A bounded planar surface: an infinite plane frame trimmed by one outer wire and optional
/// inner (hole) wires expressed in the plane's local 2D coordinates.
class PlanarBoundedSurface final : public BoundedSurface
{
 public:
  bool initialize(const Vec3& surfaceOrigin, const Vec3& surfaceAxisU, const Vec3& surfaceAxisV,
                  const std::vector<Vec2>& outerWireVertices,
                  const std::vector<std::vector<Vec2>>& innerWireVertices, std::string& errorMessage)
  {
    if (!finite(surfaceOrigin) || !finite(surfaceAxisU) || !finite(surfaceAxisV)) {
      errorMessage = "surface frame contains a non-finite value";
      return false;
    }

    mOrigin = surfaceOrigin;
    mAxisU = surfaceAxisU;
    mAxisV = surfaceAxisV;
    const Vec3 normalVector = cross(mAxisU, mAxisV);
    mAreaScale = norm(normalVector);
    if (mAreaScale <= kTolerance) {
      errorMessage = "surface frame axes are degenerate";
      return false;
    }
    mNormal = normalVector * (1. / mAreaScale);

    mMetricUU = dot(mAxisU, mAxisU);
    mMetricUV = dot(mAxisU, mAxisV);
    mMetricVV = dot(mAxisV, mAxisV);
    const double metricDet = mMetricUU * mMetricVV - mMetricUV * mMetricUV;
    if (std::abs(metricDet) <= kToleranceSq) {
      errorMessage = "surface frame metric is singular";
      return false;
    }
    mInverseMetricDet = 1. / metricDet;

    WireStatus outerStatus = WireStatus::Valid;
    const ParametricMetric metric = parametricMetricOf(*this);
    mTrimBand = trimLengthFloor(metric, Vec2{0., 0.});
    if (!mOuterWire.initialize(outerWireVertices, WireRole::Outer, outerStatus, metric)) {
      errorMessage = std::string("outer wire invalid: ") + wireStatusMessage(outerStatus);
      return false;
    }
    mOuterReoriented = (outerStatus == WireStatus::Reversed);

    mInnerWires.clear();
    mInnerWires.reserve(innerWireVertices.size());
    mInnerReoriented = false;
    for (const auto& innerWireInput : innerWireVertices) {
      SurfaceWire innerWire;
      WireStatus innerStatus = WireStatus::Valid;
      if (!innerWire.initialize(innerWireInput, WireRole::Inner, innerStatus, metric)) {
        errorMessage = std::string("inner wire invalid: ") + wireStatusMessage(innerStatus);
        return false;
      }
      mInnerReoriented = mInnerReoriented || (innerStatus == WireStatus::Reversed);
      mInnerWires.emplace_back(std::move(innerWire));
    }

    const auto ringOf = [this](const SurfaceWire& wire) {
      std::vector<Vec3> ring;
      ring.reserve(wire.vertices.size());
      for (const auto& vertex : wire.vertices) {
        ring.push_back(toGlobal(vertex));
      }
      return ring;
    };
    mOuterRing = ringOf(mOuterWire);
    mInnerRings.clear();
    for (const auto& innerWire : mInnerWires) {
      mInnerRings.push_back(ringOf(innerWire));
    }
    return true;
  }

  /// True if either the outer or any inner wire had to be re-oriented during initialization.
  bool wasReoriented() const { return mOuterReoriented || mInnerReoriented; }

  Vec3 toGlobal(const Vec2& point) const
  {
    return mOrigin + mAxisU * point.uCoord + mAxisV * point.vCoord;
  }

  Vec2 toLocal(const Vec3& point) const
  {
    const Vec3 relativePoint = point - mOrigin;
    const double projectionU = dot(relativePoint, mAxisU);
    const double projectionV = dot(relativePoint, mAxisV);
    return {(projectionU * mMetricVV - projectionV * mMetricUV) * mInverseMetricDet,
            (projectionV * mMetricUU - projectionU * mMetricUV) * mInverseMetricDet};
  }

  double planeDistance(const Vec3& point) const { return dot(point - mOrigin, mNormal); }

  bool containsLocal(const Vec2& point, bool* boundary = nullptr) const
  {
    if (boundary != nullptr) {
      *boundary = false;
    }

    const auto outerClassification = mOuterWire.classify(point, mTrimBand);
    if (outerClassification == WireClassification::Outside) {
      return false;
    }
    if (outerClassification == WireClassification::Boundary) {
      if (boundary != nullptr) {
        *boundary = true;
      }
      return true;
    }

    for (const auto& innerWire : mInnerWires) {
      const auto innerClassification = innerWire.classify(point, mTrimBand);
      if (innerClassification == WireClassification::Boundary) {
        if (boundary != nullptr) {
          *boundary = true;
        }
        return true;
      }
      if (innerClassification == WireClassification::Inside) {
        return false;
      }
    }
    return true;
  }

  bool containsPointOnSurface(const Vec3& point) const override
  {
    if (std::abs(planeDistance(point)) > kTolerance) {
      return false;
    }
    return containsLocal(toLocal(point));
  }

  void appendIntersections(const Vec3& rayOrigin, const Vec3& rayDirection, double minDistance,
                           double maxDistance, std::vector<RayHit>& hits) const override
  {
    const double denominator = dot(mNormal, rayDirection);
    if (std::abs(denominator) <= kTolerance) {
      return;
    }
    const double candidateDistance = dot(mOrigin - rayOrigin, mNormal) / denominator;
    if (candidateDistance < minDistance || candidateDistance > maxDistance) {
      return;
    }
    const Vec3 candidatePoint = rayOrigin + rayDirection * candidateDistance;
    bool onTrimBoundary = false;
    if (!containsLocal(toLocal(candidatePoint), &onTrimBoundary)) {
      return;
    }
    hits.push_back({candidateDistance, mNormal, onTrimBoundary});
  }

  double distanceSqToEdges(const Vec3& point, const std::vector<Vec3>& ring) const
  {
    double bestDistanceSq = std::numeric_limits<double>::infinity();
    for (size_t vertexIndex = 0; vertexIndex < ring.size(); ++vertexIndex) {
      bestDistanceSq =
        std::min(bestDistanceSq, pointSegmentDistanceSq(point, ring[vertexIndex], ring[(vertexIndex + 1) % ring.size()]));
    }
    return bestDistanceSq;
  }

  double distanceSqToPatch(const Vec3& point) const override
  {
    const Vec2 projectedPoint = toLocal(point);
    if (containsLocal(projectedPoint)) {
      const double signedPlaneDistance = planeDistance(point);
      return signedPlaneDistance * signedPlaneDistance;
    }

    double bestDistanceSq = distanceSqToEdges(point, mOuterRing);
    for (const auto& innerRing : mInnerRings) {
      bestDistanceSq = std::min(bestDistanceSq, distanceSqToEdges(point, innerRing));
    }
    return bestDistanceSq;
  }

  Vec3 normalAt(const Vec3&) const override { return mNormal; }

  void conservativeBounds(Vec3& lower, Vec3& upper) const override
  {
    auto extendPoint = [&](const Vec2& surfacePoint) {
      const Vec3 globalPoint = toGlobal(surfacePoint);
      lower.xCoord = std::min(lower.xCoord, globalPoint.xCoord);
      lower.yCoord = std::min(lower.yCoord, globalPoint.yCoord);
      lower.zCoord = std::min(lower.zCoord, globalPoint.zCoord);
      upper.xCoord = std::max(upper.xCoord, globalPoint.xCoord);
      upper.yCoord = std::max(upper.yCoord, globalPoint.yCoord);
      upper.zCoord = std::max(upper.zCoord, globalPoint.zCoord);
    };

    for (const auto& vertex : mOuterWire.vertices) {
      extendPoint(vertex);
    }
    for (const auto& innerWire : mInnerWires) {
      for (const auto& vertex : innerWire.vertices) {
        extendPoint(vertex);
      }
    }
  }

  double area() const
  {
    double parametricArea = std::abs(mOuterWire.signedArea());
    for (const auto& innerWire : mInnerWires) {
      parametricArea -= std::abs(innerWire.signedArea());
    }
    return std::max(0., parametricArea) * mAreaScale;
  }

  /// Constant over the plane, with a cross term: the frame axes need be neither unit-length nor orthogonal.
  void parametricMetric(const Vec2&, double& gUU, double& gUV, double& gVV) const override
  {
    planeParametricMetric(mAxisU, mAxisV, gUU, gUV, gVV);
  }

  double capacityContribution() const override { return dot(mOrigin, mNormal) * area() / 3.; }

  bool capacityIsExact() const override { return true; }

  void appendDisplayMesh(std::vector<Vec3>& vertices, std::vector<std::array<int, 3>>& triangles) const override
  {
    const int firstVertexIndex = static_cast<int>(vertices.size());
    for (const auto& vertex : mOuterWire.vertices) {
      vertices.push_back(toGlobal(vertex));
    }

    const auto localTriangles = triangulateSimpleWire(mOuterWire);
    for (const auto& triangle : localTriangles) {
      triangles.push_back(
        {firstVertexIndex + triangle[0], firstVertexIndex + triangle[1], firstVertexIndex + triangle[2]});
    }
  }

  void appendDirectedEdges(std::vector<std::pair<Vec3, Vec3>>& edges) const override
  {
    auto appendWire = [&](const SurfaceWire& wire) {
      for (size_t vertexIndex = 0; vertexIndex < wire.vertices.size(); ++vertexIndex) {
        const Vec3 edgeStart = toGlobal(wire.vertices[vertexIndex]);
        const Vec3 edgeEnd = toGlobal(wire.vertices[(vertexIndex + 1) % wire.vertices.size()]);
        edges.emplace_back(edgeStart, edgeEnd);
      }
    };
    appendWire(mOuterWire);
    for (const auto& innerWire : mInnerWires) {
      appendWire(innerWire);
    }
  }

  bool sampleTrimCurve(size_t index, std::vector<Vec3>& samples) const override
  {
    return sampleTrimCurveOfSurfaceWires(
      mOuterWire, mInnerWires, index,
      [this](double u, double v) { return toGlobal(Vec2{u, v}); }, samples);
  }

 private:
  Vec3 mOrigin;
  Vec3 mAxisU;
  Vec3 mAxisV;
  Vec3 mNormal;
  double mMetricUU = 0.;
  double mMetricUV = 0.;
  double mMetricVV = 0.;
  double mInverseMetricDet = 0.;
  double mAreaScale = 0.;
  bool mOuterReoriented = false;
  bool mInnerReoriented = false;
  double mTrimBand = 0.; ///< the wires' on-boundary band; the plane's metric is constant
  SurfaceWire mOuterWire;
  std::vector<SurfaceWire> mInnerWires;
  std::vector<Vec3> mOuterRing;               ///< the outer wire's vertices in 3D
  std::vector<std::vector<Vec3>> mInnerRings; ///< the inner wires' vertices in 3D
};

/// A plane trimmed by curved (line/arc/B-spline) loops in an orthonormal frame: exact caps, disks and annuli.
class CurvedPlanarBoundedSurface final : public BoundedSurface
{
 public:
  bool initialize(const Vec3& surfaceOrigin, const Vec3& surfaceAxisU, const Vec3& surfaceAxisV,
                  const std::vector<Curve2D>& outerCurves,
                  const std::vector<std::vector<Curve2D>>& innerCurves, std::string& errorMessage,
                  double joinTolerance = kWireJoinTolerance)
  {
    if (!finite(surfaceOrigin) || !finite(surfaceAxisU) || !finite(surfaceAxisV)) {
      errorMessage = "surface frame contains a non-finite value";
      return false;
    }
    if (std::abs(norm(surfaceAxisU) - 1.) > kTolerance || std::abs(norm(surfaceAxisV) - 1.) > kTolerance ||
        std::abs(dot(surfaceAxisU, surfaceAxisV)) > kTolerance) {
      errorMessage = "curved planar surface requires orthonormal frame axes";
      return false;
    }

    mOrigin = surfaceOrigin;
    mAxisU = surfaceAxisU;
    mAxisV = surfaceAxisV;
    mNormal = cross(mAxisU, mAxisV);

    WireStatus outerStatus = WireStatus::Valid;
    const ParametricMetric metric = parametricMetricOf(*this);
    mTrimFloor = trimLengthFloor(metric, Vec2{0., 0.});
    if (!mOuterWire.initialize(outerCurves, WireRole::Outer, outerStatus, metric, joinTolerance)) {
      errorMessage = std::string("outer wire invalid: ") + wireStatusMessage(outerStatus);
      return false;
    }
    mReoriented = (outerStatus == WireStatus::Reversed);

    mInnerWires.clear();
    mInnerWires.reserve(innerCurves.size());
    for (const auto& innerCurveLoop : innerCurves) {
      CurveWire innerWire;
      WireStatus innerStatus = WireStatus::Valid;
      if (!innerWire.initialize(innerCurveLoop, WireRole::Inner, innerStatus, metric, joinTolerance)) {
        errorMessage = std::string("inner wire invalid: ") + wireStatusMessage(innerStatus);
        return false;
      }
      mReoriented = mReoriented || (innerStatus == WireStatus::Reversed);
      mInnerWires.emplace_back(std::move(innerWire));
    }

    // A B-spline boundary makes the area (hence the capacity contribution) a numeric quadrature,
    // so flag the capacity as inexact (matching the wire-trimmed-quadric policy).
    mCapacityExact = !mOuterWire.hasBSpline();
    for (const auto& innerWire : mInnerWires) {
      mCapacityExact = mCapacityExact && !innerWire.hasBSpline();
    }
    return true;
  }

  /// True if the outer or any inner wire had to be re-oriented during initialization.
  bool wasReoriented() const { return mReoriented; }

  Vec3 toGlobal(const Vec2& point) const { return mOrigin + mAxisU * point.uCoord + mAxisV * point.vCoord; }

  Vec2 toLocal(const Vec3& point) const
  {
    const Vec3 relativePoint = point - mOrigin;
    return {dot(relativePoint, mAxisU), dot(relativePoint, mAxisV)};
  }

  double planeDistance(const Vec3& point) const { return dot(point - mOrigin, mNormal); }

  bool containsLocal(const Vec2& point, bool* boundary = nullptr) const
  {
    if (boundary != nullptr) {
      *boundary = false;
    }

    const auto outerClassification = mOuterWire.classify(point, mTrimFloor);
    if (outerClassification == WireClassification::Outside) {
      return false;
    }
    if (outerClassification == WireClassification::Boundary) {
      if (boundary != nullptr) {
        *boundary = true;
      }
      return true;
    }

    for (const auto& innerWire : mInnerWires) {
      const auto innerClassification = innerWire.classify(point, mTrimFloor);
      if (innerClassification == WireClassification::Boundary) {
        if (boundary != nullptr) {
          *boundary = true;
        }
        return true;
      }
      if (innerClassification == WireClassification::Inside) {
        return false;
      }
    }
    return true;
  }

  bool containsPointOnSurface(const Vec3& point) const override
  {
    if (std::abs(planeDistance(point)) > kTolerance) {
      return false;
    }
    return containsLocal(toLocal(point));
  }

  void appendIntersections(const Vec3& rayOrigin, const Vec3& rayDirection, double minDistance,
                           double maxDistance, std::vector<RayHit>& hits) const override
  {
    const double denominator = dot(mNormal, rayDirection);
    if (std::abs(denominator) <= kTolerance) {
      return;
    }
    const double candidateDistance = dot(mOrigin - rayOrigin, mNormal) / denominator;
    if (candidateDistance < minDistance || candidateDistance > maxDistance) {
      return;
    }
    bool onTrimBoundary = false;
    if (!containsLocal(toLocal(rayOrigin + rayDirection * candidateDistance), &onTrimBoundary)) {
      return;
    }
    hits.push_back({candidateDistance, mNormal, onTrimBoundary});
  }

  double distanceSqToPatch(const Vec3& point) const override
  {
    const Vec2 projectedPoint = toLocal(point);
    const double signedPlaneDistance = planeDistance(point);
    if (containsLocal(projectedPoint)) {
      return signedPlaneDistance * signedPlaneDistance;
    }

    // exact for an orthonormal frame: split into in-plane distance to the trim curves plus the
    // out-of-plane plane distance
    double bestCurveDistanceSq = std::numeric_limits<double>::infinity();
    for (const auto& curve : mOuterWire.curves) {
      bestCurveDistanceSq = std::min(bestCurveDistanceSq, curve.distanceSq(projectedPoint));
    }
    for (const auto& innerWire : mInnerWires) {
      for (const auto& curve : innerWire.curves) {
        bestCurveDistanceSq = std::min(bestCurveDistanceSq, curve.distanceSq(projectedPoint));
      }
    }
    return bestCurveDistanceSq + signedPlaneDistance * signedPlaneDistance;
  }

  Vec3 normalAt(const Vec3&) const override { return mNormal; }

  void conservativeBounds(Vec3& lower, Vec3& upper) const override
  {
    Vec2 parametricLower{std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()};
    Vec2 parametricUpper{-std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()};
    mOuterWire.parametricBounds(parametricLower, parametricUpper);

    // the affine image of the parametric AABB contains the patch; its corners bound the 3D AABB
    for (const double cornerU : {parametricLower.uCoord, parametricUpper.uCoord}) {
      for (const double cornerV : {parametricLower.vCoord, parametricUpper.vCoord}) {
        const Vec3 globalCorner = toGlobal({cornerU, cornerV});
        lower.xCoord = std::min(lower.xCoord, globalCorner.xCoord);
        lower.yCoord = std::min(lower.yCoord, globalCorner.yCoord);
        lower.zCoord = std::min(lower.zCoord, globalCorner.zCoord);
        upper.xCoord = std::max(upper.xCoord, globalCorner.xCoord);
        upper.yCoord = std::max(upper.yCoord, globalCorner.yCoord);
        upper.zCoord = std::max(upper.zCoord, globalCorner.zCoord);
      }
    }
  }

  double area() const
  {
    double parametricArea = std::abs(mOuterWire.signedArea());
    for (const auto& innerWire : mInnerWires) {
      parametricArea -= std::abs(innerWire.signedArea());
    }
    return std::max(0., parametricArea);
  }

  /// The identity form: initialize() rejects a frame whose axes are not orthonormal, so (u, v)
  /// here are already lengths in centimetres. (The polygon-wire plane is the general case.)
  void parametricMetric(const Vec2&, double& gUU, double& gUV, double& gVV) const override
  {
    gUU = 1.;
    gUV = 0.;
    gVV = 1.;
  }

  double capacityContribution() const override { return dot(mOrigin, mNormal) * area() / 3.; }

  bool capacityIsExact() const override { return mCapacityExact; }

  void appendDisplayMesh(std::vector<Vec3>& vertices, std::vector<std::array<int, 3>>& triangles) const override
  {
    // triangulate the sampled outer boundary; holes are ignored in the display mesh (as for the
    // polygonal planar surface, visualization never influences navigation)
    auto samples = mOuterWire.sampledBoundary();
    if (samples.size() < 4) {
      return;
    }
    samples.pop_back(); // drop the closing duplicate

    SurfaceWire sampledWire;
    sampledWire.vertices = std::move(samples);

    const int firstVertexIndex = static_cast<int>(vertices.size());
    for (const auto& vertex : sampledWire.vertices) {
      vertices.push_back(toGlobal(vertex));
    }
    for (const auto& triangle : triangulateSimpleWire(sampledWire)) {
      triangles.push_back(
        {firstVertexIndex + triangle[0], firstVertexIndex + triangle[1], firstVertexIndex + triangle[2]});
    }
  }

  void appendDirectedEdges(std::vector<std::pair<Vec3, Vec3>>& edges) const override
  {
    auto appendWire = [&](const CurveWire& wire) {
      const auto samples = wire.sampledBoundary();
      for (size_t sampleIndex = 0; sampleIndex + 1 < samples.size(); ++sampleIndex) {
        edges.emplace_back(toGlobal(samples[sampleIndex]), toGlobal(samples[sampleIndex + 1]));
      }
    };
    appendWire(mOuterWire);
    for (const auto& innerWire : mInnerWires) {
      appendWire(innerWire);
    }
  }

  bool sampleTrimCurve(size_t index, std::vector<Vec3>& samples) const override
  {
    return sampleTrimCurveOfCurveWires(
      mOuterWire, mInnerWires, index,
      [this](double u, double v) { return toGlobal(Vec2{u, v}); }, samples);
  }

 private:
  Vec3 mOrigin;
  Vec3 mAxisU;
  Vec3 mAxisV;
  Vec3 mNormal;
  bool mReoriented = false;
  bool mCapacityExact = true;
  double mTrimFloor = 0.; ///< the wires' band floor; the plane's metric is constant
  CurveWire mOuterWire;
  std::vector<CurveWire> mInnerWires;
};

/// Cover boxes of a band of revolution between two rim circles: the phi window in chunks, each the box of its two rim arcs.
inline void appendArcBandCoverBoxes(const Vec3& center, const Vec3& axisU, const Vec3& axisV, const Vec3& axisW,
                                    double phiStart, double phiSweep, double heightMin, double heightMax,
                                    double radiusAtMin, double radiusAtMax,
                                    std::vector<BoundedSurface::CoverBox>& boxes)
{
  const int chunks = coverChunkCount(phiSweep);
  for (int chunk = 0; chunk < chunks; ++chunk) {
    const double phiLow = phiStart + phiSweep * chunk / chunks;
    const double phiHigh = phiStart + phiSweep * (chunk + 1) / chunks;
    double lower[3];
    double upper[3];
    for (int dimension = 0; dimension < 3; ++dimension) {
      double radialLow = 0.;
      double radialHigh = 0.;
      sinusoidRange(component(axisU, dimension), component(axisV, dimension), phiLow, phiHigh, radialLow, radialHigh);
      const double centerAtMin = component(center, dimension) + heightMin * component(axisW, dimension);
      const double centerAtMax = component(center, dimension) + heightMax * component(axisW, dimension);
      lower[dimension] = std::min(centerAtMin + radiusAtMin * radialLow, centerAtMax + radiusAtMax * radialLow);
      upper[dimension] = std::max(centerAtMin + radiusAtMin * radialHigh, centerAtMax + radiusAtMax * radialHigh);
    }
    boxes.push_back({Vec3{lower[0], lower[1], lower[2]}, Vec3{upper[0], upper[1], upper[2]}});
  }
}

/// A cylinder of given radius around an axis, trimmed to a (phi, h) rectangle or by curve wires; innerWall points the normal to the axis.
class CylindricalBoundedSurface final : public BoundedSurface
{
 public:
  bool initialize(const Vec3& centerPoint, const Vec3& axis, const Vec3& referenceAxisU, double radius,
                  double heightMin, double heightMax, double phiStart, double phiSweep, bool innerWall,
                  std::string& errorMessage)
  {
    if (!finite(centerPoint) || !finite(axis) || !finite(referenceAxisU) || !std::isfinite(radius) ||
        !std::isfinite(heightMin) || !std::isfinite(heightMax) || !std::isfinite(phiStart) ||
        !std::isfinite(phiSweep)) {
      errorMessage = "cylindrical surface parameter is non-finite";
      return false;
    }
    if (radius <= kTolerance) {
      errorMessage = "cylindrical surface needs a positive radius";
      return false;
    }
    if (heightMax - heightMin <= kTolerance) {
      errorMessage = "cylindrical surface needs a positive height range";
      return false;
    }
    if (phiSweep <= kTolerance || phiSweep > kTwoPi + kTolerance) {
      errorMessage = "cylindrical surface needs an angular sweep in (0, 2pi]";
      return false;
    }
    if (!makeFrame(axis, referenceAxisU, mAxisU, mAxisV, mAxisW, errorMessage)) {
      return false;
    }

    mCenter = centerPoint;
    mRadius = radius;
    mPhiTolerance = angularTolerance(mRadius);
    mHeightMin = heightMin;
    mHeightMax = heightMax;
    mPhiStart = phiStart;
    mPhiSweep = std::min(phiSweep, kTwoPi);
    mNormalSign = innerWall ? -1. : 1.;
    return true;
  }

  /// Wire-trimmed overload: the wires in the (phi[rad], h[cm]) domain decide containment; the window tightens to their bounds.
  bool initialize(const Vec3& centerPoint, const Vec3& axis, const Vec3& referenceAxisU, double radius,
                  double heightMin, double heightMax, double phiStart, double phiSweep, bool innerWall,
                  const std::vector<Curve2D>& outerTrim, const std::vector<std::vector<Curve2D>>& innerTrims,
                  std::string& errorMessage, double joinTolerance = kWireJoinTolerance)
  {
    if (!initialize(centerPoint, axis, referenceAxisU, radius, heightMin, heightMax, phiStart, phiSweep, innerWall,
                    errorMessage)) {
      return false;
    }
    Vec2 lower, upper;
    if (!buildCurveTrim(outerTrim, innerTrims, mTrimOuter, mTrimInner, lower, upper, errorMessage,
                        parametricMetricOf(*this), joinTolerance)) {
      return false;
    }
    mPhiStart = lower.uCoord;
    mPhiSweep = std::min(kTwoPi, upper.uCoord - lower.uCoord);
    mHeightMin = lower.vCoord;
    mHeightMax = upper.vCoord;
    mHasWireTrim = true;
    return true;
  }

  bool hasWireTrim() const { return mHasWireTrim; }

  /// True if the (phi, h) point lies in the trim wire (phi unwrapped into the wire window).
  bool pointInTrim(double phi, double height, bool* boundary = nullptr) const
  {
    const double uCoord = unwrapAngleInto(phi, mPhiStart, mPhiStart + mPhiSweep);
    return curveTrimContains(mTrimOuter, mTrimInner, {uCoord, height}, boundary, parametricMetricOf(*this));
  }

  /// Build an orthonormal frame (U, V, W) with W along \a axis and U the projection of
  /// \a referenceAxisU perpendicular to W. Shared by all axis-symmetric quadric surfaces.
  static bool makeFrame(const Vec3& axis, const Vec3& referenceAxisU, Vec3& axisU, Vec3& axisV, Vec3& axisW,
                        std::string& errorMessage)
  {
    if (norm(axis) <= kTolerance) {
      errorMessage = "surface axis is degenerate";
      return false;
    }
    axisW = normalized(axis);
    const Vec3 projectedU = referenceAxisU - axisW * dot(referenceAxisU, axisW);
    if (norm(projectedU) <= kTolerance) {
      errorMessage = "surface reference axis is parallel to the main axis";
      return false;
    }
    axisU = normalized(projectedU);
    axisV = cross(axisW, axisU); // gives axisU x axisV = axisW
    return true;
  }

  bool fullSweep() const { return mPhiSweep >= kTwoPi - kTolerance; }

  Vec3 toLocal(const Vec3& point) const
  {
    const Vec3 relativePoint = point - mCenter;
    return {dot(relativePoint, mAxisU), dot(relativePoint, mAxisV), dot(relativePoint, mAxisW)};
  }

  bool heightInRange(double height) const
  {
    return height >= mHeightMin - kTolerance && height <= mHeightMax + kTolerance;
  }

  bool phiInSweep(double phi) const
  {
    return angleInSweepRange(phi, mPhiStart, mPhiSweep, mPhiTolerance);
  }

  Vec3 pointAt(double phi, double height) const
  {
    return mCenter + mAxisW * height + (mAxisU * std::cos(phi) + mAxisV * std::sin(phi)) * mRadius;
  }

  bool containsPointOnSurface(const Vec3& point) const override
  {
    const Vec3 localPoint = toLocal(point);
    const double radialDistance = std::hypot(localPoint.xCoord, localPoint.yCoord);
    if (std::abs(radialDistance - mRadius) > kTolerance) {
      return false;
    }
    if (radialDistance <= kTolerance) {
      return !mHasWireTrim && heightInRange(localPoint.zCoord); // phi is undefined on the axis
    }
    const double phi = std::atan2(localPoint.yCoord, localPoint.xCoord);
    if (mHasWireTrim) {
      return pointInTrim(phi, localPoint.zCoord);
    }
    return heightInRange(localPoint.zCoord) && phiInSweep(phi);
  }

  void appendIntersections(const Vec3& rayOrigin, const Vec3& rayDirection, double minDistance,
                           double maxDistance, std::vector<RayHit>& hits) const override
  {
    const Vec3 localOrigin = toLocal(rayOrigin);
    const Vec3 localDirection{dot(rayDirection, mAxisU), dot(rayDirection, mAxisV), dot(rayDirection, mAxisW)};

    const double quadraticA = localDirection.xCoord * localDirection.xCoord +
                              localDirection.yCoord * localDirection.yCoord;
    if (quadraticA <= kToleranceSq) {
      return; // ray parallel to the axis: no transversal crossing of the lateral surface
    }
    const double quadraticB = 2. * (localOrigin.xCoord * localDirection.xCoord +
                                    localOrigin.yCoord * localDirection.yCoord);
    const double quadraticC = localOrigin.xCoord * localOrigin.xCoord +
                              localOrigin.yCoord * localOrigin.yCoord - mRadius * mRadius;
    const double discriminant = quadraticB * quadraticB - 4. * quadraticA * quadraticC;
    if (discriminant <= 0.) {
      return;
    }
    const double sqrtDiscriminant = std::sqrt(discriminant);
    const double firstRoot = (-quadraticB - sqrtDiscriminant) / (2. * quadraticA);
    const double secondRoot = (-quadraticB + sqrtDiscriminant) / (2. * quadraticA);
    if (sameIntersection(firstRoot, secondRoot)) {
      return; // tangential graze: report neither hit so crossing parity stays even
    }

    for (const double candidate : {firstRoot, secondRoot}) {
      if (candidate < minDistance || candidate > maxDistance) {
        continue;
      }
      const double hitU = localOrigin.xCoord + candidate * localDirection.xCoord;
      const double hitV = localOrigin.yCoord + candidate * localDirection.yCoord;
      const double hitHeight = localOrigin.zCoord + candidate * localDirection.zCoord;
      const double hitPhi = std::atan2(hitV, hitU);
      bool onTrimBoundary = false;
      if (mHasWireTrim) {
        if (!pointInTrim(hitPhi, hitHeight, &onTrimBoundary)) {
          continue;
        }
      } else if (!heightInRange(hitHeight) || !phiInSweep(hitPhi)) {
        continue;
      }
      const double radialDistance = std::hypot(hitU, hitV);
      const Vec3 hitNormal = (mAxisU * (hitU / radialDistance) + mAxisV * (hitV / radialDistance)) * mNormalSign;
      hits.push_back({candidate, hitNormal, onTrimBoundary});
    }
  }

  /// Distance to the patch: exact for the parametric rectangle, a lower bound for a wire trim.
  double distanceSqToPatch(const Vec3& point) const override
  {
    const Vec3 localPoint = toLocal(point);
    const double radialDistance = std::hypot(localPoint.xCoord, localPoint.yCoord);
    if (radialDistance <= kTolerance || phiInSweep(std::atan2(localPoint.yCoord, localPoint.xCoord))) {
      return pointSegmentDistanceSq(Vec2{radialDistance, localPoint.zCoord}, Vec2{mRadius, mHeightMin},
                                    Vec2{mRadius, mHeightMax});
    }
    const double distanceToStartSeam =
      pointSegmentDistanceSq(point, pointAt(mPhiStart, mHeightMin), pointAt(mPhiStart, mHeightMax));
    const double endPhi = mPhiStart + mPhiSweep;
    const double distanceToEndSeam =
      pointSegmentDistanceSq(point, pointAt(endPhi, mHeightMin), pointAt(endPhi, mHeightMax));
    return std::min(distanceToStartSeam, distanceToEndSeam);
  }

  Vec3 normalAt(const Vec3& point) const override
  {
    const Vec3 localPoint = toLocal(point);
    const double radialDistance = std::hypot(localPoint.xCoord, localPoint.yCoord);
    if (radialDistance <= kTolerance) {
      return mAxisU * mNormalSign; // ill-defined on the axis; return a stable direction
    }
    return (mAxisU * (localPoint.xCoord / radialDistance) + mAxisV * (localPoint.yCoord / radialDistance)) *
           mNormalSign;
  }

  /// (u, v) = (phi[rad], h[cm]): X_phi has length r and X_h is the unit axis.
  void parametricMetric(const Vec2&, double& gUU, double& gUV, double& gVV) const override
  {
    cylinderParametricMetric(mRadius, gUU, gUV, gVV);
  }

  /// Divergence-theorem contribution over the (phi, h) rectangle; a wire trim uses the contour form, F = (s r / 3)(a sin phi - b cos phi + r phi).
  double capacityContribution() const override
  {
    if (mHasWireTrim) {
      const double centreU = dot(mCenter, mAxisU);
      const double centreV = dot(mCenter, mAxisV);
      const double factor = mNormalSign * mRadius / 3.;
      return integrateOverCurveTrimByParts(mTrimOuter, mTrimInner, [&](double phi, double) {
        return factor * (centreU * std::sin(phi) - centreV * std::cos(phi) + mRadius * phi);
      });
    }
    const double endPhi = mPhiStart + mPhiSweep;
    const double phiFactor = dot(mCenter, mAxisU) * (std::sin(endPhi) - std::sin(mPhiStart)) -
                             dot(mCenter, mAxisV) * (std::cos(endPhi) - std::cos(mPhiStart));
    const double height = mHeightMax - mHeightMin;
    return mNormalSign * mRadius * height * (phiFactor + mRadius * mPhiSweep) / 3.;
  }

  bool capacityIsExact() const override { return !mHasWireTrim; }

  void conservativeBounds(Vec3& lower, Vec3& upper) const override
  {
    // conservative: the AABB of the two full rim circles (partial sweeps get a larger box)
    for (const double height : {mHeightMin, mHeightMax}) {
      const Vec3 rimCenter = mCenter + mAxisW * height;
      for (int dimension = 0; dimension < 3; ++dimension) {
        const double radialExtent = mRadius * std::hypot(component(mAxisU, dimension), component(mAxisV, dimension));
        const double centerValue = component(rimCenter, dimension);
        if (dimension == 0) {
          lower.xCoord = std::min(lower.xCoord, centerValue - radialExtent);
          upper.xCoord = std::max(upper.xCoord, centerValue + radialExtent);
        } else if (dimension == 1) {
          lower.yCoord = std::min(lower.yCoord, centerValue - radialExtent);
          upper.yCoord = std::max(upper.yCoord, centerValue + radialExtent);
        } else {
          lower.zCoord = std::min(lower.zCoord, centerValue - radialExtent);
          upper.zCoord = std::max(upper.zCoord, centerValue + radialExtent);
        }
      }
    }
  }

  /// Cover boxes: the sweep window in angular chunks, which holds every point that realises distanceSqToPatch.
  void appendCoverBoxes(std::vector<CoverBox>& boxes) const override
  {
    appendArcBandCoverBoxes(mCenter, mAxisU, mAxisV, mAxisW, mPhiStart, mPhiSweep, mHeightMin, mHeightMax, mRadius,
                            mRadius, boxes);
  }

  /// Number of chord segments used for rim sampling, consistent with CurveWire::sampledBoundary
  /// so shared circular boundaries close against curved planar caps.
  int rimSegments() const
  {
    return std::max(1, static_cast<int>(std::lround(kArcSamples * mPhiSweep / kTwoPi)));
  }

  void appendDisplayMesh(std::vector<Vec3>& vertices, std::vector<std::array<int, 3>>& triangles) const override
  {
    if (mHasWireTrim) {
      appendCurveTrimMesh(mTrimOuter, [this](double phi, double height) { return pointAt(phi, height); }, vertices, triangles);
      return;
    }
    const int segments = rimSegments();
    const int firstVertexIndex = static_cast<int>(vertices.size());
    for (int step = 0; step <= segments; ++step) {
      const double phi = mPhiStart + mPhiSweep * step / segments;
      vertices.push_back(pointAt(phi, mHeightMin));
      vertices.push_back(pointAt(phi, mHeightMax));
    }
    for (int step = 0; step < segments; ++step) {
      const int base = firstVertexIndex + 2 * step;
      triangles.push_back({base, base + 2, base + 3});
      triangles.push_back({base, base + 3, base + 1});
    }
  }

  void appendDirectedEdges(std::vector<std::pair<Vec3, Vec3>>& edges) const override
  {
    if (mHasWireTrim) {
      // the (phi, h) -> 3D map is orientation-consistent with the outward normal, so a CCW trim
      // loop yields a CCW 3D loop for an outer wall; the sign is just mNormalSign
      appendCurveTrimEdges(mTrimOuter, mTrimInner, [this](double phi, double height) { return pointAt(phi, height); }, mNormalSign, edges);
      return;
    }
    // boundary counter-clockwise seen along the outward normal, so rims shared with caps cancel
    const int segments = rimSegments();
    auto emitEdge = [&](const Vec3& edgeStart, const Vec3& edgeEnd) {
      if (mNormalSign > 0.) {
        edges.emplace_back(edgeStart, edgeEnd);
      } else {
        edges.emplace_back(edgeEnd, edgeStart);
      }
    };
    for (int step = 0; step < segments; ++step) {
      const double phi = mPhiStart + mPhiSweep * step / segments;
      const double nextPhi = mPhiStart + mPhiSweep * (step + 1) / segments;
      emitEdge(pointAt(phi, mHeightMin), pointAt(nextPhi, mHeightMin));
      emitEdge(pointAt(nextPhi, mHeightMax), pointAt(phi, mHeightMax));
    }
    if (!fullSweep()) {
      const double endPhi = mPhiStart + mPhiSweep;
      emitEdge(pointAt(endPhi, mHeightMin), pointAt(endPhi, mHeightMax));
      emitEdge(pointAt(mPhiStart, mHeightMax), pointAt(mPhiStart, mHeightMin));
    }
  }

  bool sampleTrimCurve(size_t index, std::vector<Vec3>& samples) const override
  {
    if (!mHasWireTrim) {
      return false; // a parametric-rectangle trim carries no per-edge curve to sample
    }
    return sampleTrimCurveOfCurveWires(mTrimOuter, mTrimInner, index, [this](double phi, double height) { return pointAt(phi, height); }, samples);
  }

 private:
  Vec3 mCenter;
  Vec3 mAxisU;
  Vec3 mAxisV;
  Vec3 mAxisW;
  double mRadius = 0.;
  double mHeightMin = 0.;
  double mHeightMax = 0.;
  double mPhiStart = 0.;
  double mPhiSweep = kTwoPi;
  double mPhiTolerance = 0.; ///< angularTolerance of the radius
  double mNormalSign = 1.;
  bool mHasWireTrim = false;
  CurveWire mTrimOuter;
  std::vector<CurveWire> mTrimInner;
};

/// A sphere of given radius trimmed to a (theta, phi) rectangle or by curve wires; innerWall points the normal to the centre.
class SphericalBoundedSurface final : public BoundedSurface
{
 public:
  bool initialize(const Vec3& center, const Vec3& polarAxis, const Vec3& referenceAxisU, double radius,
                  double thetaMin, double thetaMax, double phiStart, double phiSweep, bool innerWall,
                  std::string& errorMessage)
  {
    if (!finite(center) || !finite(polarAxis) || !finite(referenceAxisU) || !std::isfinite(radius) ||
        !std::isfinite(thetaMin) || !std::isfinite(thetaMax) || !std::isfinite(phiStart) ||
        !std::isfinite(phiSweep)) {
      errorMessage = "spherical surface parameter is non-finite";
      return false;
    }
    if (radius <= kTolerance) {
      errorMessage = "spherical surface needs a positive radius";
      return false;
    }
    if (thetaMin < -kTolerance || thetaMax > kPi + kTolerance || thetaMax - thetaMin <= kTolerance) {
      errorMessage = "spherical surface needs a polar range within [0, pi]";
      return false;
    }
    if (phiSweep <= kTolerance || phiSweep > kTwoPi + kTolerance) {
      errorMessage = "spherical surface needs an angular sweep in (0, 2pi]";
      return false;
    }
    if (!CylindricalBoundedSurface::makeFrame(polarAxis, referenceAxisU, mAxisU, mAxisV, mAxisW, errorMessage)) {
      return false;
    }

    mCenter = center;
    mRadius = radius;
    mThetaMin = std::max(0., thetaMin);
    mThetaMax = std::min(kPi, thetaMax);
    mPhiStart = phiStart;
    mPhiSweep = std::min(phiSweep, kTwoPi);
    mNormalSign = innerWall ? -1. : 1.;
    return true;
  }

  /// Wire-trimmed overload: the wires in the (phi[rad], theta[rad]) domain decide containment; the window tightens to their bounds.
  bool initialize(const Vec3& center, const Vec3& polarAxis, const Vec3& referenceAxisU, double radius,
                  double thetaMin, double thetaMax, double phiStart, double phiSweep, bool innerWall,
                  const std::vector<Curve2D>& outerTrim, const std::vector<std::vector<Curve2D>>& innerTrims,
                  std::string& errorMessage, double joinTolerance = kWireJoinTolerance)
  {
    if (!initialize(center, polarAxis, referenceAxisU, radius, thetaMin, thetaMax, phiStart, phiSweep, innerWall,
                    errorMessage)) {
      return false;
    }
    Vec2 lower, upper;
    if (!buildCurveTrim(outerTrim, innerTrims, mTrimOuter, mTrimInner, lower, upper, errorMessage,
                        parametricMetricOf(*this), joinTolerance)) {
      return false;
    }
    mPhiStart = lower.uCoord;
    mPhiSweep = std::min(kTwoPi, upper.uCoord - lower.uCoord);
    mThetaMin = std::max(0., lower.vCoord);
    mThetaMax = std::min(kPi, upper.vCoord);
    mHasWireTrim = true;
    return true;
  }

  bool hasWireTrim() const { return mHasWireTrim; }

  /// True if the (phi, theta) point lies in the trim wire (phi unwrapped into the wire window).
  bool pointInTrim(double phi, double theta, bool* boundary = nullptr) const
  {
    const double uCoord = unwrapAngleInto(phi, mPhiStart, mPhiStart + mPhiSweep);
    return curveTrimContains(mTrimOuter, mTrimInner, {uCoord, theta}, boundary, parametricMetricOf(*this));
  }

  bool fullSweep() const { return mPhiSweep >= kTwoPi - kTolerance; }

  Vec3 toLocal(const Vec3& point) const
  {
    const Vec3 relativePoint = point - mCenter;
    return {dot(relativePoint, mAxisU), dot(relativePoint, mAxisV), dot(relativePoint, mAxisW)};
  }

  bool directionInTrim(const Vec3& localPoint, bool* boundary = nullptr) const
  {
    if (boundary != nullptr) {
      *boundary = false;
    }
    const double pointRadius = norm(localPoint);
    if (pointRadius <= kTolerance) {
      return true; // the center is angle-degenerate; every patch point is equidistant
    }
    const double thetaTolerance = angularTolerance(mRadius);
    const double theta = std::acos(std::max(-1., std::min(1., localPoint.zCoord / pointRadius)));
    const double transverseDistance = std::hypot(localPoint.xCoord, localPoint.yCoord);
    if (mHasWireTrim) {
      if (transverseDistance <= kTolerance) {
        // on the polar axis phi is degenerate; accept by the wire's theta (v) range
        return theta >= mThetaMin - thetaTolerance && theta <= mThetaMax + thetaTolerance;
      }
      return pointInTrim(std::atan2(localPoint.yCoord, localPoint.xCoord), theta, boundary);
    }
    if (theta < mThetaMin - thetaTolerance || theta > mThetaMax + thetaTolerance) {
      return false;
    }
    if (transverseDistance <= kTolerance) {
      return true; // on the polar axis phi is degenerate
    }
    return angleInSweepRange(std::atan2(localPoint.yCoord, localPoint.xCoord), mPhiStart, mPhiSweep,
                             thetaTolerance);
  }

  Vec3 pointAt(double theta, double phi) const
  {
    const double sinTheta = std::sin(theta);
    return mCenter + (mAxisU * (sinTheta * std::cos(phi)) + mAxisV * (sinTheta * std::sin(phi)) +
                      mAxisW * std::cos(theta)) *
                       mRadius;
  }

  bool containsPointOnSurface(const Vec3& point) const override
  {
    const Vec3 localPoint = toLocal(point);
    if (std::abs(norm(localPoint) - mRadius) > kTolerance) {
      return false;
    }
    return directionInTrim(localPoint);
  }

  void appendIntersections(const Vec3& rayOrigin, const Vec3& rayDirection, double minDistance,
                           double maxDistance, std::vector<RayHit>& hits) const override
  {
    const Vec3 relativeOrigin = rayOrigin - mCenter;
    const double quadraticA = normSq(rayDirection);
    if (quadraticA <= kToleranceSq) {
      return;
    }
    const double quadraticB = 2. * dot(relativeOrigin, rayDirection);
    const double quadraticC = normSq(relativeOrigin) - mRadius * mRadius;
    const double discriminant = quadraticB * quadraticB - 4. * quadraticA * quadraticC;
    if (discriminant <= 0.) {
      return;
    }
    const double sqrtDiscriminant = std::sqrt(discriminant);
    const double firstRoot = (-quadraticB - sqrtDiscriminant) / (2. * quadraticA);
    const double secondRoot = (-quadraticB + sqrtDiscriminant) / (2. * quadraticA);
    if (sameIntersection(firstRoot, secondRoot)) {
      return; // tangential graze
    }

    for (const double candidate : {firstRoot, secondRoot}) {
      if (candidate < minDistance || candidate > maxDistance) {
        continue;
      }
      const Vec3 localHit = toLocal(rayOrigin + rayDirection * candidate);
      bool onTrimBoundary = false;
      if (!directionInTrim(localHit, &onTrimBoundary)) {
        continue;
      }
      hits.push_back({candidate,
                      (mAxisU * localHit.xCoord + mAxisV * localHit.yCoord + mAxisW * localHit.zCoord) *
                        (mNormalSign / mRadius),
                      onTrimBoundary});
    }
  }

  /// Distance to the patch: exact inside the trim, else the full-sphere distance, a lower bound.
  double distanceSqToPatch(const Vec3& point) const override
  {
    const Vec3 localPoint = toLocal(point);
    const double radialOffset = norm(localPoint) - mRadius;
    return radialOffset * radialOffset;
  }

  Vec3 normalAt(const Vec3& point) const override
  {
    const Vec3 localPoint = toLocal(point);
    const double pointRadius = norm(localPoint);
    if (pointRadius <= kTolerance) {
      return mAxisW * mNormalSign; // ill-defined at the center; return a stable direction
    }
    return (mAxisU * localPoint.xCoord + mAxisV * localPoint.yCoord + mAxisW * localPoint.zCoord) *
           (mNormalSign / pointRadius);
  }

  /// (u, v) = (phi[rad], theta[rad]); gUU vanishes at either pole.
  void parametricMetric(const Vec2& uv, double& gUU, double& gUV, double& gVV) const override
  {
    sphereParametricMetric(mRadius, uv.vCoord, gUU, gUV, gVV);
  }

  /// Divergence-theorem contribution over the (theta, phi) rectangle; a wire trim uses the contour form in (phi, theta).
  double capacityContribution() const override
  {
    if (mHasWireTrim) {
      const double centreU = dot(mCenter, mAxisU);
      const double centreV = dot(mCenter, mAxisV);
      const double centreW = dot(mCenter, mAxisW);
      const double factor = mNormalSign * mRadius * mRadius / 3.;
      return integrateOverCurveTrimByParts(mTrimOuter, mTrimInner, [&](double phi, double theta) {
        const double sinTheta = std::sin(theta);
        return factor * sinTheta *
               (sinTheta * (centreU * std::sin(phi) - centreV * std::cos(phi)) +
                (centreW * std::cos(theta) + mRadius) * phi);
      });
    }
    const double endPhi = mPhiStart + mPhiSweep;
    const double phiFactor = dot(mCenter, mAxisU) * (std::sin(endPhi) - std::sin(mPhiStart)) -
                             dot(mCenter, mAxisV) * (std::cos(endPhi) - std::cos(mPhiStart));
    const double thetaIntegralSinSq =
      0.5 * ((mThetaMax - std::sin(mThetaMax) * std::cos(mThetaMax)) -
             (mThetaMin - std::sin(mThetaMin) * std::cos(mThetaMin)));
    const double thetaIntegralSinCos =
      0.5 * (std::sin(mThetaMax) * std::sin(mThetaMax) - std::sin(mThetaMin) * std::sin(mThetaMin));
    const double thetaIntegralSin = std::cos(mThetaMin) - std::cos(mThetaMax);
    return mNormalSign * mRadius * mRadius *
           (phiFactor * thetaIntegralSinSq + dot(mCenter, mAxisW) * mPhiSweep * thetaIntegralSinCos +
            mRadius * mPhiSweep * thetaIntegralSin) /
           3.;
  }

  bool capacityIsExact() const override { return !mHasWireTrim; }

  void conservativeBounds(Vec3& lower, Vec3& upper) const override
  {
    lower.xCoord = std::min(lower.xCoord, mCenter.xCoord - mRadius);
    lower.yCoord = std::min(lower.yCoord, mCenter.yCoord - mRadius);
    lower.zCoord = std::min(lower.zCoord, mCenter.zCoord - mRadius);
    upper.xCoord = std::max(upper.xCoord, mCenter.xCoord + mRadius);
    upper.yCoord = std::max(upper.yCoord, mCenter.yCoord + mRadius);
    upper.zCoord = std::max(upper.zCoord, mCenter.zCoord + mRadius);
  }

  /// Cover boxes: the whole sphere in (theta, phi) chunks, since distanceSqToPatch ignores the trim.
  void appendCoverBoxes(std::vector<CoverBox>& boxes) const override
  {
    const int thetaChunks = coverChunkCount(kPi);
    const int phiChunks = coverChunkCount(kTwoPi);
    for (int thetaChunk = 0; thetaChunk < thetaChunks; ++thetaChunk) {
      const double thetaLow = kPi * thetaChunk / thetaChunks;
      const double thetaHigh = kPi * (thetaChunk + 1) / thetaChunks;
      for (int phiChunk = 0; phiChunk < phiChunks; ++phiChunk) {
        const double phiLow = kTwoPi * phiChunk / phiChunks;
        const double phiHigh = kTwoPi * (phiChunk + 1) / phiChunks;
        double lower[3];
        double upper[3];
        for (int dimension = 0; dimension < 3; ++dimension) {
          double inPlaneLow = 0.;
          double inPlaneHigh = 0.;
          sinusoidRange(component(mAxisU, dimension), component(mAxisV, dimension), phiLow, phiHigh, inPlaneLow,
                        inPlaneHigh);
          // sin(theta) >= 0 on [0, pi], so the chunk extremes are the theta sinusoid at s's own extremes
          const double axisComponent = component(mAxisW, dimension);
          const double high = sinusoidMaximum(axisComponent, inPlaneHigh, thetaLow, thetaHigh);
          const double low = sinusoidMinimum(axisComponent, inPlaneLow, thetaLow, thetaHigh);
          lower[dimension] = component(mCenter, dimension) + mRadius * low;
          upper[dimension] = component(mCenter, dimension) + mRadius * high;
        }
        boxes.push_back({Vec3{lower[0], lower[1], lower[2]}, Vec3{upper[0], upper[1], upper[2]}});
      }
    }
  }

  int phiSegments() const
  {
    return std::max(1, static_cast<int>(std::lround(kArcSamples * mPhiSweep / kTwoPi)));
  }

  int thetaSegments() const
  {
    return std::max(1, static_cast<int>(std::lround(kArcSamples * (mThetaMax - mThetaMin) / kTwoPi)));
  }

  void appendDisplayMesh(std::vector<Vec3>& vertices, std::vector<std::array<int, 3>>& triangles) const override
  {
    if (mHasWireTrim) {
      appendCurveTrimMesh(mTrimOuter, [this](double phi, double theta) { return pointAt(theta, phi); }, vertices, triangles);
      return;
    }
    const int phiSteps = phiSegments();
    const int thetaSteps = thetaSegments();
    const int firstVertexIndex = static_cast<int>(vertices.size());
    for (int thetaStep = 0; thetaStep <= thetaSteps; ++thetaStep) {
      const double theta = mThetaMin + (mThetaMax - mThetaMin) * thetaStep / thetaSteps;
      for (int phiStep = 0; phiStep <= phiSteps; ++phiStep) {
        vertices.push_back(pointAt(theta, mPhiStart + mPhiSweep * phiStep / phiSteps));
      }
    }
    const int rowLength = phiSteps + 1;
    for (int thetaStep = 0; thetaStep < thetaSteps; ++thetaStep) {
      for (int phiStep = 0; phiStep < phiSteps; ++phiStep) {
        const int base = firstVertexIndex + thetaStep * rowLength + phiStep;
        triangles.push_back({base, base + 1, base + rowLength + 1});
        triangles.push_back({base, base + rowLength + 1, base + rowLength});
      }
    }
  }

  void appendDirectedEdges(std::vector<std::pair<Vec3, Vec3>>& edges) const override
  {
    if (mHasWireTrim) {
      // the (phi, theta) -> 3D map is orientation-*reversed* relative to the outward normal
      // (X_phi x X_theta points inward), so the sign is -mNormalSign
      appendCurveTrimEdges(mTrimOuter, mTrimInner, [this](double phi, double theta) { return pointAt(theta, phi); }, -mNormalSign, edges);
      return;
    }
    // boundary of the (theta, phi) rectangle, traversed counter-clockwise for an outer wall;
    // pole rims are degenerate points and full-sweep phi seams cancel, so both are skipped
    auto emitEdge = [&](const Vec3& edgeStart, const Vec3& edgeEnd) {
      if (mNormalSign > 0.) {
        edges.emplace_back(edgeStart, edgeEnd);
      } else {
        edges.emplace_back(edgeEnd, edgeStart);
      }
    };
    const double thetaTolerance = angularTolerance(mRadius);
    const int phiSteps = phiSegments();
    const double endPhi = mPhiStart + mPhiSweep;
    if (mThetaMin > thetaTolerance) {
      for (int step = 0; step < phiSteps; ++step) {
        const double phi = mPhiStart + mPhiSweep * step / phiSteps;
        const double nextPhi = mPhiStart + mPhiSweep * (step + 1) / phiSteps;
        emitEdge(pointAt(mThetaMin, nextPhi), pointAt(mThetaMin, phi)); // -phi at the small-theta rim
      }
    }
    if (mThetaMax < kPi - thetaTolerance) {
      for (int step = 0; step < phiSteps; ++step) {
        const double phi = mPhiStart + mPhiSweep * step / phiSteps;
        const double nextPhi = mPhiStart + mPhiSweep * (step + 1) / phiSteps;
        emitEdge(pointAt(mThetaMax, phi), pointAt(mThetaMax, nextPhi)); // +phi at the large-theta rim
      }
    }
    if (!fullSweep()) {
      const int thetaSteps = thetaSegments();
      for (int step = 0; step < thetaSteps; ++step) {
        const double theta = mThetaMin + (mThetaMax - mThetaMin) * step / thetaSteps;
        const double nextTheta = mThetaMin + (mThetaMax - mThetaMin) * (step + 1) / thetaSteps;
        emitEdge(pointAt(theta, mPhiStart), pointAt(nextTheta, mPhiStart)); // +theta at phiStart
        emitEdge(pointAt(nextTheta, endPhi), pointAt(theta, endPhi));       // -theta at phiEnd
      }
    }
  }

  bool sampleTrimCurve(size_t index, std::vector<Vec3>& samples) const override
  {
    if (!mHasWireTrim) {
      return false; // a parametric-rectangle trim carries no per-edge curve to sample
    }
    return sampleTrimCurveOfCurveWires(mTrimOuter, mTrimInner, index, [this](double phi, double theta) { return pointAt(theta, phi); }, samples);
  }

 private:
  Vec3 mCenter;
  Vec3 mAxisU;
  Vec3 mAxisV;
  Vec3 mAxisW;
  double mRadius = 0.;
  double mThetaMin = 0.;
  double mThetaMax = kPi;
  double mPhiStart = 0.;
  double mPhiSweep = kTwoPi;
  double mNormalSign = 1.;
  bool mHasWireTrim = false;
  CurveWire mTrimOuter;
  std::vector<CurveWire> mTrimInner;
};

/// A cone whose radius varies linearly with height, trimmed as the cylinder; one radius may be zero (an apex) and slope 0 is a cylinder.
class ConicalBoundedSurface final : public BoundedSurface
{
 public:
  bool initialize(const Vec3& centerPoint, const Vec3& axis, const Vec3& referenceAxisU, double radiusAtMin,
                  double radiusAtMax, double heightMin, double heightMax, double phiStart, double phiSweep,
                  bool innerWall, std::string& errorMessage)
  {
    if (!finite(centerPoint) || !finite(axis) || !finite(referenceAxisU) || !std::isfinite(radiusAtMin) ||
        !std::isfinite(radiusAtMax) || !std::isfinite(heightMin) || !std::isfinite(heightMax) ||
        !std::isfinite(phiStart) || !std::isfinite(phiSweep)) {
      errorMessage = "conical surface parameter is non-finite";
      return false;
    }
    if (radiusAtMin < -kTolerance || radiusAtMax < -kTolerance ||
        std::max(radiusAtMin, radiusAtMax) <= kTolerance) {
      errorMessage = "conical surface needs non-negative radii, at least one positive";
      return false;
    }
    if (heightMax - heightMin <= kTolerance) {
      errorMessage = "conical surface needs a positive height range";
      return false;
    }
    if (phiSweep <= kTolerance || phiSweep > kTwoPi + kTolerance) {
      errorMessage = "conical surface needs an angular sweep in (0, 2pi]";
      return false;
    }
    if (!CylindricalBoundedSurface::makeFrame(axis, referenceAxisU, mAxisU, mAxisV, mAxisW, errorMessage)) {
      return false;
    }

    mCenter = centerPoint;
    mHeightMin = heightMin;
    mHeightMax = heightMax;
    mSlope = (radiusAtMax - radiusAtMin) / (heightMax - heightMin);
    mRadius0 = radiusAtMin - mSlope * heightMin; // radius at h = 0 of the linear law
    mPhiStart = phiStart;
    mPhiSweep = std::min(phiSweep, kTwoPi);
    mNormalSign = innerWall ? -1. : 1.;
    mPhiTolerance = angularTolerance(meanRadius());
    return true;
  }

  /// Wire-trimmed overload: the scalar radii pin r(h); the wires in the (phi[rad], h[cm]) domain decide containment.
  bool initialize(const Vec3& centerPoint, const Vec3& axis, const Vec3& referenceAxisU, double radiusAtMin,
                  double radiusAtMax, double heightMin, double heightMax, double phiStart, double phiSweep,
                  bool innerWall, const std::vector<Curve2D>& outerTrim,
                  const std::vector<std::vector<Curve2D>>& innerTrims, std::string& errorMessage,
                  double joinTolerance = kWireJoinTolerance)
  {
    if (!initialize(centerPoint, axis, referenceAxisU, radiusAtMin, radiusAtMax, heightMin, heightMax, phiStart,
                    phiSweep, innerWall, errorMessage)) {
      return false;
    }
    Vec2 lower, upper;
    if (!buildCurveTrim(outerTrim, innerTrims, mTrimOuter, mTrimInner, lower, upper, errorMessage,
                        parametricMetricOf(*this), joinTolerance)) {
      return false;
    }
    mPhiStart = lower.uCoord;
    mPhiSweep = std::min(kTwoPi, upper.uCoord - lower.uCoord);
    mHeightMin = lower.vCoord;
    mHeightMax = upper.vCoord;
    mPhiTolerance = angularTolerance(meanRadius());
    mHasWireTrim = true;
    return true;
  }

  bool hasWireTrim() const { return mHasWireTrim; }

  /// True if the (phi, h) point lies in the trim wire (phi unwrapped into the wire window).
  bool pointInTrim(double phi, double height, bool* boundary = nullptr) const
  {
    const double uCoord = unwrapAngleInto(phi, mPhiStart, mPhiStart + mPhiSweep);
    return curveTrimContains(mTrimOuter, mTrimInner, {uCoord, height}, boundary, parametricMetricOf(*this));
  }

  bool fullSweep() const { return mPhiSweep >= kTwoPi - kTolerance; }

  double radiusAt(double height) const { return mRadius0 + mSlope * height; }

  double meanRadius() const { return 0.5 * (radiusAt(mHeightMin) + radiusAt(mHeightMax)); }

  Vec3 toLocal(const Vec3& point) const
  {
    const Vec3 relativePoint = point - mCenter;
    return {dot(relativePoint, mAxisU), dot(relativePoint, mAxisV), dot(relativePoint, mAxisW)};
  }

  bool heightInRange(double height) const
  {
    return height >= mHeightMin - kTolerance && height <= mHeightMax + kTolerance;
  }

  bool phiInSweep(double phi) const
  {
    return angleInSweepRange(phi, mPhiStart, mPhiSweep, mPhiTolerance);
  }

  Vec3 pointAt(double phi, double height) const
  {
    return mCenter + mAxisW * height + (mAxisU * std::cos(phi) + mAxisV * std::sin(phi)) * radiusAt(height);
  }

  bool containsPointOnSurface(const Vec3& point) const override
  {
    const Vec3 localPoint = toLocal(point);
    const double surfaceRadius = radiusAt(localPoint.zCoord);
    const double radialDistance = std::hypot(localPoint.xCoord, localPoint.yCoord);
    // |rho - r(h)| overestimates the true surface distance by sqrt(1 + slope^2)
    if (std::abs(radialDistance - surfaceRadius) > kTolerance * std::sqrt(1. + mSlope * mSlope)) {
      return false;
    }
    if (radialDistance <= kTolerance) {
      return !mHasWireTrim && heightInRange(localPoint.zCoord); // phi is undefined near the apex
    }
    const double phi = std::atan2(localPoint.yCoord, localPoint.xCoord);
    if (mHasWireTrim) {
      return pointInTrim(phi, localPoint.zCoord);
    }
    return heightInRange(localPoint.zCoord) && phiInSweep(phi);
  }

  void appendIntersections(const Vec3& rayOrigin, const Vec3& rayDirection, double minDistance,
                           double maxDistance, std::vector<RayHit>& hits) const override
  {
    const Vec3 localOrigin = toLocal(rayOrigin);
    const Vec3 localDirection{dot(rayDirection, mAxisU), dot(rayDirection, mAxisV), dot(rayDirection, mAxisW)};

    // (ox + t dx)^2 + (oy + t dy)^2 = (radius0 + slope * (oz + t dz))^2
    const double surfaceRadiusAtOrigin = mRadius0 + mSlope * localOrigin.zCoord;
    const double quadraticA = localDirection.xCoord * localDirection.xCoord +
                              localDirection.yCoord * localDirection.yCoord -
                              mSlope * mSlope * localDirection.zCoord * localDirection.zCoord;
    const double quadraticB = 2. * (localOrigin.xCoord * localDirection.xCoord +
                                    localOrigin.yCoord * localDirection.yCoord -
                                    mSlope * localDirection.zCoord * surfaceRadiusAtOrigin);
    const double quadraticC = localOrigin.xCoord * localOrigin.xCoord +
                              localOrigin.yCoord * localOrigin.yCoord -
                              surfaceRadiusAtOrigin * surfaceRadiusAtOrigin;

    std::array<double, 2> candidates{};
    int candidateCount = 0;
    if (std::abs(quadraticA) <= kToleranceSq) {
      if (std::abs(quadraticB) <= kToleranceSq) {
        return; // ray runs along the cone surface or its asymptote: no transversal crossing
      }
      candidates[candidateCount++] = -quadraticC / quadraticB;
    } else {
      const double discriminant = quadraticB * quadraticB - 4. * quadraticA * quadraticC;
      if (discriminant <= 0.) {
        return;
      }
      const double sqrtDiscriminant = std::sqrt(discriminant);
      const double firstRoot = (-quadraticB - sqrtDiscriminant) / (2. * quadraticA);
      const double secondRoot = (-quadraticB + sqrtDiscriminant) / (2. * quadraticA);
      if (sameIntersection(firstRoot, secondRoot)) {
        return; // tangential graze (this also covers rays through the exact apex)
      }
      candidates[candidateCount++] = std::min(firstRoot, secondRoot);
      candidates[candidateCount++] = std::max(firstRoot, secondRoot);
    }

    for (int candidateIndex = 0; candidateIndex < candidateCount; ++candidateIndex) {
      const double candidate = candidates[candidateIndex];
      if (candidate < minDistance || candidate > maxDistance) {
        continue;
      }
      const double hitHeight = localOrigin.zCoord + candidate * localDirection.zCoord;
      const double hitSurfaceRadius = radiusAt(hitHeight);
      if (hitSurfaceRadius < -kTolerance) {
        continue; // mirror nappe of the infinite cone
      }
      const double hitU = localOrigin.xCoord + candidate * localDirection.xCoord;
      const double hitV = localOrigin.yCoord + candidate * localDirection.yCoord;
      const double radialDistance = std::hypot(hitU, hitV);
      if (radialDistance <= kTolerance) {
        continue; // apex hit: the normal is undefined there
      }
      const double hitPhi = std::atan2(hitV, hitU);
      bool onTrimBoundary = false;
      if (mHasWireTrim) {
        if (!pointInTrim(hitPhi, hitHeight, &onTrimBoundary)) {
          continue;
        }
      } else if (!heightInRange(hitHeight) || !phiInSweep(hitPhi)) {
        continue;
      }
      const double normalScale = mNormalSign / std::sqrt(1. + mSlope * mSlope);
      const Vec3 hitNormal =
        (mAxisU * (hitU / radialDistance) + mAxisV * (hitV / radialDistance) - mAxisW * mSlope) * normalScale;
      hits.push_back({candidate, hitNormal, onTrimBoundary});
    }
  }

  /// Distance to the patch: exact for the parametric rectangle, a lower bound for a wire trim.
  double distanceSqToPatch(const Vec3& point) const override
  {
    const Vec3 localPoint = toLocal(point);
    const double radialDistance = std::hypot(localPoint.xCoord, localPoint.yCoord);
    if (radialDistance <= kTolerance || phiInSweep(std::atan2(localPoint.yCoord, localPoint.xCoord))) {
      return pointSegmentDistanceSq(Vec2{radialDistance, localPoint.zCoord},
                                    Vec2{radiusAt(mHeightMin), mHeightMin}, Vec2{radiusAt(mHeightMax), mHeightMax});
    }
    const double endPhi = mPhiStart + mPhiSweep;
    const double distanceToStartSeam =
      pointSegmentDistanceSq(point, pointAt(mPhiStart, mHeightMin), pointAt(mPhiStart, mHeightMax));
    const double distanceToEndSeam =
      pointSegmentDistanceSq(point, pointAt(endPhi, mHeightMin), pointAt(endPhi, mHeightMax));
    return std::min(distanceToStartSeam, distanceToEndSeam);
  }

  Vec3 normalAt(const Vec3& point) const override
  {
    const Vec3 localPoint = toLocal(point);
    const double radialDistance = std::hypot(localPoint.xCoord, localPoint.yCoord);
    const double normalScale = mNormalSign / std::sqrt(1. + mSlope * mSlope);
    if (radialDistance <= kTolerance) {
      return (mAxisU - mAxisW * mSlope) * normalScale; // ill-defined on the axis; stable fallback
    }
    return (mAxisU * (localPoint.xCoord / radialDistance) + mAxisV * (localPoint.yCoord / radialDistance) -
            mAxisW * mSlope) *
           normalScale;
  }

  /// (u, v) = (phi[rad], h[cm]): the azimuthal scale is the local radius, and a step in h spans sqrt(1 + slope^2).
  void parametricMetric(const Vec2& uv, double& gUU, double& gUV, double& gVV) const override
  {
    coneParametricMetric(radiusAt(uv.vCoord), mSlope, gUU, gUV, gVV);
  }

  /// Divergence-theorem contribution over the (phi, h) rectangle; a wire trim uses the contour form, as for the cylinder.
  double capacityContribution() const override
  {
    if (mHasWireTrim) {
      const double centreU = dot(mCenter, mAxisU);
      const double centreV = dot(mCenter, mAxisV);
      const double centreW = dot(mCenter, mAxisW);
      return integrateOverCurveTrimByParts(mTrimOuter, mTrimInner, [&](double phi, double height) {
        const double localRadius = radiusAt(height);
        return mNormalSign / 3. * localRadius *
               (centreU * std::sin(phi) - centreV * std::cos(phi) +
                (localRadius - mSlope * (centreW + height)) * phi);
      });
    }
    const double endPhi = mPhiStart + mPhiSweep;
    const double phiFactor = dot(mCenter, mAxisU) * (std::sin(endPhi) - std::sin(mPhiStart)) -
                             dot(mCenter, mAxisV) * (std::cos(endPhi) - std::cos(mPhiStart));
    const double radiusIntegral = mRadius0 * (mHeightMax - mHeightMin) +
                                  0.5 * mSlope * (mHeightMax * mHeightMax - mHeightMin * mHeightMin);
    return mNormalSign * radiusIntegral *
           (phiFactor + (mRadius0 - mSlope * dot(mCenter, mAxisW)) * mPhiSweep) / 3.;
  }

  bool capacityIsExact() const override { return !mHasWireTrim; }

  void conservativeBounds(Vec3& lower, Vec3& upper) const override
  {
    for (const double height : {mHeightMin, mHeightMax}) {
      const Vec3 rimCenter = mCenter + mAxisW * height;
      const double rimRadius = std::max(0., radiusAt(height));
      for (int dimension = 0; dimension < 3; ++dimension) {
        const double radialExtent =
          rimRadius * std::hypot(component(mAxisU, dimension), component(mAxisV, dimension));
        const double centerValue = component(rimCenter, dimension);
        if (dimension == 0) {
          lower.xCoord = std::min(lower.xCoord, centerValue - radialExtent);
          upper.xCoord = std::max(upper.xCoord, centerValue + radialExtent);
        } else if (dimension == 1) {
          lower.yCoord = std::min(lower.yCoord, centerValue - radialExtent);
          upper.yCoord = std::max(upper.yCoord, centerValue + radialExtent);
        } else {
          lower.zCoord = std::min(lower.zCoord, centerValue - radialExtent);
          upper.zCoord = std::max(upper.zCoord, centerValue + radialExtent);
        }
      }
    }
  }

  /// Cover boxes: as for the cylinder, with the rim radii from the linear radius law.
  void appendCoverBoxes(std::vector<CoverBox>& boxes) const override
  {
    appendArcBandCoverBoxes(mCenter, mAxisU, mAxisV, mAxisW, mPhiStart, mPhiSweep, mHeightMin, mHeightMax,
                            std::max(0., radiusAt(mHeightMin)), std::max(0., radiusAt(mHeightMax)), boxes);
  }

  int rimSegments() const
  {
    return std::max(1, static_cast<int>(std::lround(kArcSamples * mPhiSweep / kTwoPi)));
  }

  void appendDisplayMesh(std::vector<Vec3>& vertices, std::vector<std::array<int, 3>>& triangles) const override
  {
    if (mHasWireTrim) {
      appendCurveTrimMesh(mTrimOuter, [this](double phi, double height) { return pointAt(phi, height); }, vertices, triangles);
      return;
    }
    const int segments = rimSegments();
    const int firstVertexIndex = static_cast<int>(vertices.size());
    for (int step = 0; step <= segments; ++step) {
      const double phi = mPhiStart + mPhiSweep * step / segments;
      vertices.push_back(pointAt(phi, mHeightMin));
      vertices.push_back(pointAt(phi, mHeightMax));
    }
    for (int step = 0; step < segments; ++step) {
      const int base = firstVertexIndex + 2 * step;
      // skip triangles that collapse at an apex rim
      if (radiusAt(mHeightMin) > kTolerance) {
        triangles.push_back({base, base + 2, base + 3});
      }
      if (radiusAt(mHeightMax) > kTolerance) {
        triangles.push_back({base, base + 3, base + 1});
      }
    }
  }

  void appendDirectedEdges(std::vector<std::pair<Vec3, Vec3>>& edges) const override
  {
    if (mHasWireTrim) {
      // the (phi, h) -> 3D map is orientation-consistent with the outward normal (as for the
      // cylinder), so the sign is just mNormalSign
      appendCurveTrimEdges(mTrimOuter, mTrimInner, [this](double phi, double height) { return pointAt(phi, height); }, mNormalSign, edges);
      return;
    }
    // same boundary orientation as the cylinder; an apex rim degenerates to a point and is
    // skipped so an apex cone closes against just one cap
    const int segments = rimSegments();
    auto emitEdge = [&](const Vec3& edgeStart, const Vec3& edgeEnd) {
      if (mNormalSign > 0.) {
        edges.emplace_back(edgeStart, edgeEnd);
      } else {
        edges.emplace_back(edgeEnd, edgeStart);
      }
    };
    for (int step = 0; step < segments; ++step) {
      const double phi = mPhiStart + mPhiSweep * step / segments;
      const double nextPhi = mPhiStart + mPhiSweep * (step + 1) / segments;
      if (radiusAt(mHeightMin) > kTolerance) {
        emitEdge(pointAt(phi, mHeightMin), pointAt(nextPhi, mHeightMin));
      }
      if (radiusAt(mHeightMax) > kTolerance) {
        emitEdge(pointAt(nextPhi, mHeightMax), pointAt(phi, mHeightMax));
      }
    }
    if (!fullSweep()) {
      const double endPhi = mPhiStart + mPhiSweep;
      emitEdge(pointAt(endPhi, mHeightMin), pointAt(endPhi, mHeightMax));
      emitEdge(pointAt(mPhiStart, mHeightMax), pointAt(mPhiStart, mHeightMin));
    }
  }

  bool sampleTrimCurve(size_t index, std::vector<Vec3>& samples) const override
  {
    if (!mHasWireTrim) {
      return false; // a parametric-rectangle trim carries no per-edge curve to sample
    }
    return sampleTrimCurveOfCurveWires(mTrimOuter, mTrimInner, index, [this](double phi, double height) { return pointAt(phi, height); }, samples);
  }

 private:
  Vec3 mCenter;
  Vec3 mAxisU;
  Vec3 mAxisV;
  Vec3 mAxisW;
  double mRadius0 = 0.;
  double mSlope = 0.;
  double mHeightMin = 0.;
  double mHeightMax = 0.;
  double mPhiStart = 0.;
  double mPhiSweep = kTwoPi;
  double mPhiTolerance = 0.; ///< angularTolerance of the mean radius of the final window
  double mNormalSign = 1.;
  bool mHasWireTrim = false;
  CurveWire mTrimOuter;
  std::vector<CurveWire> mTrimInner;
};

/// A torus of major radius R and minor radius r, trimmed to a (phiRing, phiTube) rectangle or by curve wires;
/// X(u, v) = centre + (U cos u + V sin u)(R + r cos v) + W r sin v, orientation-consistent with the outward normal.
class TorusBoundedSurface final : public BoundedSurface
{
 public:
  bool initialize(const Vec3& centerPoint, const Vec3& axis, const Vec3& referenceAxisU, double majorRadius,
                  double minorRadius, double phiStart, double phiSweep, double tubeStart, double tubeSweep,
                  bool innerWall, std::string& errorMessage)
  {
    if (!finite(centerPoint) || !finite(axis) || !finite(referenceAxisU) || !std::isfinite(majorRadius) ||
        !std::isfinite(minorRadius) || !std::isfinite(phiStart) || !std::isfinite(phiSweep) ||
        !std::isfinite(tubeStart) || !std::isfinite(tubeSweep)) {
      errorMessage = "toroidal surface parameter is non-finite";
      return false;
    }
    if (majorRadius <= kTolerance || minorRadius <= kTolerance) {
      errorMessage = "toroidal surface needs positive major and minor radii";
      return false;
    }
    if (phiSweep <= kTolerance || phiSweep > kTwoPi + kTolerance) {
      errorMessage = "toroidal surface needs a ring sweep in (0, 2pi]";
      return false;
    }
    if (tubeSweep <= kTolerance || tubeSweep > kTwoPi + kTolerance) {
      errorMessage = "toroidal surface needs a tube sweep in (0, 2pi]";
      return false;
    }
    if (!CylindricalBoundedSurface::makeFrame(axis, referenceAxisU, mAxisU, mAxisV, mAxisW, errorMessage)) {
      return false;
    }

    mCenter = centerPoint;
    mMajorRadius = majorRadius;
    mMinorRadius = minorRadius;
    mRingTolerance = angularTolerance(mMajorRadius);
    mTubeTolerance = angularTolerance(mMinorRadius);
    mPhiStart = phiStart;
    mPhiSweep = std::min(phiSweep, kTwoPi);
    mTubeStart = tubeStart;
    mTubeSweep = std::min(tubeSweep, kTwoPi);
    mNormalSign = innerWall ? -1. : 1.;
    return true;
  }

  /// Wire-trimmed overload: the wires in the (phiRing, phiTube) domain decide containment; a trim wrapping a full turn is refused.
  bool initialize(const Vec3& centerPoint, const Vec3& axis, const Vec3& referenceAxisU, double majorRadius,
                  double minorRadius, double phiStart, double phiSweep, double tubeStart, double tubeSweep,
                  bool innerWall, const std::vector<Curve2D>& outerTrim,
                  const std::vector<std::vector<Curve2D>>& innerTrims, std::string& errorMessage,
                  double joinTolerance = kWireJoinTolerance)
  {
    if (!initialize(centerPoint, axis, referenceAxisU, majorRadius, minorRadius, phiStart, phiSweep, tubeStart,
                    tubeSweep, innerWall, errorMessage)) {
      return false;
    }
    Vec2 lower, upper;
    if (!buildCurveTrim(outerTrim, innerTrims, mTrimOuter, mTrimInner, lower, upper, errorMessage,
                        parametricMetricOf(*this), joinTolerance)) {
      return false;
    }
    if (upper.vCoord - lower.vCoord > kTwoPi + kTolerance) {
      errorMessage = "toroidal trim wire spans more than a full turn in the tube angle";
      return false;
    }
    mPhiStart = lower.uCoord;
    mPhiSweep = std::min(kTwoPi, upper.uCoord - lower.uCoord);
    mTubeStart = lower.vCoord;
    mTubeSweep = std::min(kTwoPi, upper.vCoord - lower.vCoord);
    mHasWireTrim = true;
    return true;
  }

  bool hasWireTrim() const { return mHasWireTrim; }

  /// Whether (phiRing, phiTube) lies in the trim wire, both angles unwrapped into their windows.
  bool pointInTrim(double phiRing, double phiTube, bool* boundary = nullptr) const
  {
    const double uCoord = unwrapAngleInto(phiRing, mPhiStart, mPhiStart + mPhiSweep);
    const double vCoord = unwrapAngleInto(phiTube, mTubeStart, mTubeStart + mTubeSweep);
    return curveTrimContains(mTrimOuter, mTrimInner, {uCoord, vCoord}, boundary, parametricMetricOf(*this));
  }

  bool fullRingSweep() const { return mPhiSweep >= kTwoPi - kTolerance; }
  bool fullTubeSweep() const { return mTubeSweep >= kTwoPi - kTolerance; }

  Vec3 toLocal(const Vec3& point) const
  {
    const Vec3 relativePoint = point - mCenter;
    return {dot(relativePoint, mAxisU), dot(relativePoint, mAxisV), dot(relativePoint, mAxisW)};
  }

  bool ringInSweep(double phiRing) const
  {
    return angleInSweepRange(phiRing, mPhiStart, mPhiSweep, mRingTolerance);
  }

  bool tubeInSweep(double phiTube) const
  {
    return angleInSweepRange(phiTube, mTubeStart, mTubeSweep, mTubeTolerance);
  }

  Vec3 pointAt(double phiRing, double phiTube) const
  {
    const double ringRadius = mMajorRadius + mMinorRadius * std::cos(phiTube);
    return mCenter + (mAxisU * std::cos(phiRing) + mAxisV * std::sin(phiRing)) * ringRadius +
           mAxisW * (mMinorRadius * std::sin(phiTube));
  }

  /// Unit outward normal (pointing away from the tube spine) from a local surface point.
  Vec3 localNormal(const Vec3& localPoint) const
  {
    const double rho = std::hypot(localPoint.xCoord, localPoint.yCoord);
    if (rho <= kTolerance) {
      return mAxisW * (localPoint.zCoord >= 0. ? mNormalSign : -mNormalSign);
    }
    const double radialFactor = (rho - mMajorRadius) / rho;
    Vec3 normal{radialFactor * localPoint.xCoord, radialFactor * localPoint.yCoord, localPoint.zCoord};
    const double length = norm(normal);
    if (length <= kTolerance) {
      return mAxisU * mNormalSign;
    }
    return (mAxisU * normal.xCoord + mAxisV * normal.yCoord + mAxisW * normal.zCoord) * (mNormalSign / length);
  }

  bool containsPointOnSurface(const Vec3& point) const override
  {
    const Vec3 localPoint = toLocal(point);
    const double rho = std::hypot(localPoint.xCoord, localPoint.yCoord);
    const double meridianDistance = std::hypot(rho - mMajorRadius, localPoint.zCoord) - mMinorRadius;
    if (std::abs(meridianDistance) > kTolerance) {
      return false;
    }
    const double phiTube = std::atan2(localPoint.zCoord, rho - mMajorRadius);
    if (rho <= kTolerance) {
      return false; // on the axis phiRing is undefined (only reachable on a horn/spindle torus)
    }
    const double phiRing = std::atan2(localPoint.yCoord, localPoint.xCoord);
    if (mHasWireTrim) {
      return pointInTrim(phiRing, phiTube);
    }
    return ringInSweep(phiRing) && tubeInSweep(phiTube);
  }

  void appendIntersections(const Vec3& rayOrigin, const Vec3& rayDirection, double minDistance,
                           double maxDistance, std::vector<RayHit>& hits) const override
  {
    const Vec3 localOrigin = toLocal(rayOrigin);
    const Vec3 localDirection{dot(rayDirection, mAxisU), dot(rayDirection, mAxisV), dot(rayDirection, mAxisW)};

    // Torus implicit form (local): (|X|^2 + R^2 - r^2)^2 = 4 R^2 (x^2 + y^2). Substituting the ray
    // X = O + t D gives a quartic in t whose leading coefficient is |D|^4 > 0.
    const double dirDotDir = normSq(localDirection);
    if (dirDotDir <= kToleranceSq) {
      return; // degenerate direction
    }
    const double originDotDir = dot(localOrigin, localDirection);
    const double originDotOrigin = normSq(localOrigin);
    const double constantK = mMajorRadius * mMajorRadius - mMinorRadius * mMinorRadius;
    const double transverseE = localDirection.xCoord * localDirection.xCoord +
                               localDirection.yCoord * localDirection.yCoord;
    const double transverseF = localOrigin.xCoord * localDirection.xCoord +
                               localOrigin.yCoord * localDirection.yCoord;
    const double transverseG = localOrigin.xCoord * localOrigin.xCoord +
                               localOrigin.yCoord * localOrigin.yCoord;
    const double fourRSquared = 4. * mMajorRadius * mMajorRadius;

    const double coeff4 = dirDotDir * dirDotDir;
    const double coeff3 = 4. * dirDotDir * originDotDir;
    const double coeff2 =
      4. * originDotDir * originDotDir + 2. * dirDotDir * (originDotOrigin + constantK) - fourRSquared * transverseE;
    const double coeff1 = 4. * originDotDir * (originDotOrigin + constantK) - 2. * fourRSquared * transverseF;
    const double coeff0 = (originDotOrigin + constantK) * (originDotOrigin + constantK) - fourRSquared * transverseG;

    QuarticRoots candidates = solveQuarticReal(coeff4, coeff3, coeff2, coeff1, coeff0);
    if (candidates.empty()) {
      return;
    }
    std::sort(candidates.begin(), candidates.end());

    // an even-sized cluster of near-equal roots is a tangency and is dropped; an odd one is one crossing at its mean
    size_t rootIndex = 0;
    while (rootIndex < candidates.size()) {
      size_t clusterEnd = rootIndex + 1;
      double clusterSum = candidates[rootIndex];
      while (clusterEnd < candidates.size() && sameIntersection(candidates[clusterEnd], candidates[clusterEnd - 1])) {
        clusterSum += candidates[clusterEnd];
        ++clusterEnd;
      }
      const size_t clusterSize = clusterEnd - rootIndex;
      rootIndex = clusterEnd;
      if ((clusterSize & 1u) == 0u) {
        continue; // tangential graze
      }
      const double candidate = clusterSum / static_cast<double>(clusterSize);
      if (candidate < minDistance || candidate > maxDistance) {
        continue;
      }
      const Vec3 localHit = toLocal(rayOrigin + rayDirection * candidate);
      const double rho = std::hypot(localHit.xCoord, localHit.yCoord);
      if (rho <= kTolerance) {
        continue;
      }
      const double phiTube = std::atan2(localHit.zCoord, rho - mMajorRadius);
      const double phiRing = std::atan2(localHit.yCoord, localHit.xCoord);
      bool onTrimBoundary = false;
      if (mHasWireTrim) {
        if (!pointInTrim(phiRing, phiTube, &onTrimBoundary)) {
          continue;
        }
      } else if (!ringInSweep(phiRing) || !tubeInSweep(phiTube)) {
        continue;
      }
      hits.push_back({candidate, localNormal(localHit), onTrimBoundary});
    }
  }

  /// Distance to the patch: exact for the full torus by the meridian distance, a lower bound for a trimmed patch.
  double distanceSqToPatch(const Vec3& point) const override
  {
    const Vec3 localPoint = toLocal(point);
    const double rho = std::hypot(localPoint.xCoord, localPoint.yCoord);
    const double meridianDistance = std::hypot(rho - mMajorRadius, localPoint.zCoord) - mMinorRadius;
    return meridianDistance * meridianDistance;
  }

  Vec3 normalAt(const Vec3& point) const override { return localNormal(toLocal(point)); }

  /// (u, v) = (phiRing[rad], phiTube[rad]): the tube scale is r, the ring scale the distance from the axis.
  void parametricMetric(const Vec2& uv, double& gUU, double& gUV, double& gVV) const override
  {
    torusParametricMetric(mMajorRadius, mMinorRadius, uv.vCoord, gUU, gUV, gVV);
  }

  /// Divergence-theorem contribution over the (phiRing, phiTube) rectangle; a wire trim uses the contour form.
  double capacityContribution() const override
  {
    if (mHasWireTrim) {
      const double centreU = dot(mCenter, mAxisU);
      const double centreV = dot(mCenter, mAxisV);
      const double centreW = dot(mCenter, mAxisW);
      return integrateOverCurveTrimByParts(mTrimOuter, mTrimInner, [&](double phiRing, double phiTube) {
        const double cosTube = std::cos(phiTube);
        const double sinTube = std::sin(phiTube);
        const double rho = mMajorRadius + mMinorRadius * cosTube;
        return mNormalSign * mMinorRadius * rho / 3. *
               (cosTube * (centreU * std::sin(phiRing) - centreV * std::cos(phiRing)) +
                (centreW * sinTube + rho * cosTube + mMinorRadius * sinTube * sinTube) * phiRing);
      });
    }
    // Closed form over u in [u0, u1] (ring) and v in [v0, v1] (tube).
    const double majorR = mMajorRadius;
    const double minorR = mMinorRadius;
    const double u0 = mPhiStart, u1 = mPhiStart + mPhiSweep;
    const double v0 = mTubeStart, v1 = mTubeStart + mTubeSweep;
    const double centerU = dot(mCenter, mAxisU);
    const double centerV = dot(mCenter, mAxisV);
    const double centerW = dot(mCenter, mAxisW);
    const double deltaU = u1 - u0;
    const double deltaV = v1 - v0;
    const double sinIntegralU = std::sin(u1) - std::sin(u0);                                  // integral cos u du
    const double cosIntegralU = std::cos(u0) - std::cos(u1);                                  // integral sin u du
    const double sinIntegralV = std::sin(v1) - std::sin(v0);                                  // integral cos v dv
    const double sinFromCosV = std::cos(v0) - std::cos(v1);                                   // integral sin v dv
    const double cosSquaredV = 0.5 * deltaV + 0.25 * (std::sin(2. * v1) - std::sin(2. * v0)); // integral cos^2 v dv
    const double sinCosV = 0.25 * (std::cos(2. * v0) - std::cos(2. * v1));                    // integral sin v cos v dv

    // centre-independent part, integrated over v then multiplied by the ring span
    const double centerlessV =
      minorR * ((majorR * majorR + minorR * minorR) * sinIntegralV + majorR * minorR * deltaV +
                majorR * minorR * cosSquaredV);
    // W component of the centre offset
    const double centerWpart = minorR * (majorR * sinFromCosV + minorR * sinCosV);
    // U/V components of the centre offset (ring-angle dependent)
    const double centerUVpart =
      (centerU * sinIntegralU + centerV * cosIntegralU) * minorR * (majorR * sinIntegralV + minorR * cosSquaredV);

    const double total = deltaU * centerlessV + deltaU * centerW * centerWpart + centerUVpart;
    return mNormalSign * total / 3.;
  }

  bool capacityIsExact() const override { return !mHasWireTrim; }

  void conservativeBounds(Vec3& lower, Vec3& upper) const override
  {
    // conservative: the AABB of the full torus (partial sweeps get a larger box)
    const double outerRadius = mMajorRadius + mMinorRadius;
    for (int dimension = 0; dimension < 3; ++dimension) {
      const double radialExtent = outerRadius * std::hypot(component(mAxisU, dimension), component(mAxisV, dimension)) +
                                  mMinorRadius * std::abs(component(mAxisW, dimension));
      const double centerValue = component(mCenter, dimension);
      if (dimension == 0) {
        lower.xCoord = std::min(lower.xCoord, centerValue - radialExtent);
        upper.xCoord = std::max(upper.xCoord, centerValue + radialExtent);
      } else if (dimension == 1) {
        lower.yCoord = std::min(lower.yCoord, centerValue - radialExtent);
        upper.yCoord = std::max(upper.yCoord, centerValue + radialExtent);
      } else {
        lower.zCoord = std::min(lower.zCoord, centerValue - radialExtent);
        upper.zCoord = std::max(upper.zCoord, centerValue + radialExtent);
      }
    }
  }

  /// Cover boxes: the full torus in angular chunks, since the meridian projection ignores the trim; a spindle torus uses one box.
  void appendCoverBoxes(std::vector<CoverBox>& boxes) const override
  {
    if (mMajorRadius < mMinorRadius) {
      BoundedSurface::appendCoverBoxes(boxes);
      return;
    }
    const int ringChunks = coverChunkCount(kTwoPi);
    const int tubeChunks = coverChunkCount(kTwoPi);
    for (int ringChunk = 0; ringChunk < ringChunks; ++ringChunk) {
      const double ringLow = kTwoPi * ringChunk / ringChunks;
      const double ringHigh = kTwoPi * (ringChunk + 1) / ringChunks;
      for (int tubeChunk = 0; tubeChunk < tubeChunks; ++tubeChunk) {
        const double tubeLow = kTwoPi * tubeChunk / tubeChunks;
        const double tubeHigh = kTwoPi * (tubeChunk + 1) / tubeChunks;
        double lower[3];
        double upper[3];
        for (int dimension = 0; dimension < 3; ++dimension) {
          double inPlaneLow = 0.;
          double inPlaneHigh = 0.;
          sinusoidRange(component(mAxisU, dimension), component(mAxisV, dimension), ringLow, ringHigh, inPlaneLow,
                        inPlaneHigh);
          // the coordinate is p(u) (R + r cos v) + w r sin v; with R + r cos v >= 0 it is
          // monotone in p, so each extreme is a v sinusoid taken at p's own extreme
          const double axisComponent = component(mAxisW, dimension);
          const double high = sinusoidMaximum(inPlaneHigh, axisComponent, tubeLow, tubeHigh);
          const double low = sinusoidMinimum(inPlaneLow, axisComponent, tubeLow, tubeHigh);
          lower[dimension] = component(mCenter, dimension) + inPlaneLow * mMajorRadius + mMinorRadius * low;
          upper[dimension] = component(mCenter, dimension) + inPlaneHigh * mMajorRadius + mMinorRadius * high;
        }
        boxes.push_back({Vec3{lower[0], lower[1], lower[2]}, Vec3{upper[0], upper[1], upper[2]}});
      }
    }
  }

  int ringSegments() const
  {
    return std::max(1, static_cast<int>(std::lround(kArcSamples * mPhiSweep / kTwoPi)));
  }

  int tubeSegments() const
  {
    return std::max(1, static_cast<int>(std::lround(kArcSamples * mTubeSweep / kTwoPi)));
  }

  void appendDisplayMesh(std::vector<Vec3>& vertices, std::vector<std::array<int, 3>>& triangles) const override
  {
    if (mHasWireTrim) {
      appendCurveTrimMesh(mTrimOuter, [this](double phiRing, double phiTube) { return pointAt(phiRing, phiTube); }, vertices, triangles);
      return;
    }
    const int ringSteps = ringSegments();
    const int tubeSteps = tubeSegments();
    const int firstVertexIndex = static_cast<int>(vertices.size());
    for (int ringStep = 0; ringStep <= ringSteps; ++ringStep) {
      const double phiRing = mPhiStart + mPhiSweep * ringStep / ringSteps;
      for (int tubeStep = 0; tubeStep <= tubeSteps; ++tubeStep) {
        vertices.push_back(pointAt(phiRing, mTubeStart + mTubeSweep * tubeStep / tubeSteps));
      }
    }
    const int rowLength = tubeSteps + 1;
    for (int ringStep = 0; ringStep < ringSteps; ++ringStep) {
      for (int tubeStep = 0; tubeStep < tubeSteps; ++tubeStep) {
        const int base = firstVertexIndex + ringStep * rowLength + tubeStep;
        triangles.push_back({base, base + rowLength, base + rowLength + 1});
        triangles.push_back({base, base + rowLength + 1, base + 1});
      }
    }
  }

  void appendDirectedEdges(std::vector<std::pair<Vec3, Vec3>>& edges) const override
  {
    if (mHasWireTrim) {
      // the (phiRing, phiTube) -> 3D map is orientation-consistent with the outward normal, so
      // the sign is just mNormalSign (as for the cylinder and cone)
      appendCurveTrimEdges(mTrimOuter, mTrimInner, [this](double phiRing, double phiTube) { return pointAt(phiRing, phiTube); }, mNormalSign, edges);
      return;
    }
    // boundary of the (phiRing, phiTube) rectangle traversed counter-clockwise as seen along the
    // outward normal; a full sweep in either angle has no seam there, so it is skipped
    auto emitEdge = [&](const Vec3& edgeStart, const Vec3& edgeEnd) {
      if (mNormalSign > 0.) {
        edges.emplace_back(edgeStart, edgeEnd);
      } else {
        edges.emplace_back(edgeEnd, edgeStart);
      }
    };
    const int ringSteps = ringSegments();
    const int tubeSteps = tubeSegments();
    const double endRing = mPhiStart + mPhiSweep;
    const double endTube = mTubeStart + mTubeSweep;
    if (!fullTubeSweep()) {
      for (int step = 0; step < ringSteps; ++step) {
        const double phiRing = mPhiStart + mPhiSweep * step / ringSteps;
        const double nextRing = mPhiStart + mPhiSweep * (step + 1) / ringSteps;
        emitEdge(pointAt(phiRing, mTubeStart), pointAt(nextRing, mTubeStart)); // +phiRing at tubeStart
        emitEdge(pointAt(nextRing, endTube), pointAt(phiRing, endTube));       // -phiRing at tubeEnd
      }
    }
    if (!fullRingSweep()) {
      for (int step = 0; step < tubeSteps; ++step) {
        const double phiTube = mTubeStart + mTubeSweep * step / tubeSteps;
        const double nextTube = mTubeStart + mTubeSweep * (step + 1) / tubeSteps;
        emitEdge(pointAt(endRing, phiTube), pointAt(endRing, nextTube));     // +phiTube at ringEnd
        emitEdge(pointAt(mPhiStart, nextTube), pointAt(mPhiStart, phiTube)); // -phiTube at ringStart
      }
    }
  }

  bool sampleTrimCurve(size_t index, std::vector<Vec3>& samples) const override
  {
    if (!mHasWireTrim) {
      return false; // a parametric-rectangle trim carries no per-edge curve to sample
    }
    return sampleTrimCurveOfCurveWires(mTrimOuter, mTrimInner, index, [this](double phiRing, double phiTube) { return pointAt(phiRing, phiTube); }, samples);
  }

 private:
  Vec3 mCenter;
  Vec3 mAxisU;
  Vec3 mAxisV;
  Vec3 mAxisW;
  double mMajorRadius = 0.;
  double mMinorRadius = 0.;
  double mPhiStart = 0.;
  double mPhiSweep = kTwoPi;
  double mTubeStart = 0.;
  double mTubeSweep = kTwoPi;
  double mRingTolerance = 0.; ///< angularTolerance of the major radius
  double mTubeTolerance = 0.; ///< angularTolerance of the minor radius
  double mNormalSign = 1.;
  bool mHasWireTrim = false;
  CurveWire mTrimOuter;
  std::vector<CurveWire> mTrimInner;
};

/// How one rim came out of the closure measurement. The four states are exhaustive, and they are
/// exactly the four the ClosureReport rim counters tally.
enum class RimState {
  Matched = 0, ///< every chord has another face within its match band, traversed the other way
  Reversed,    ///< matched, but the partner traverses the shared curve the same way
  Boundary,    ///< some chord has no other face within its match band
  NonManifold  ///< some chord has two or more other faces within the declared tolerance
};

/// One trim loop of one face as measureRimClosure saw it, naming the rim and its worst chord.
struct RimRecord {
  int surfaceIndex = -1;      ///< the owning face's index in the solid's surface list
  int rimIndexOnSurface = -1; ///< which trim loop of that face, in the order the face emits them
  bool closed = false;        ///< the polyline returns to its own first point
  int chords = 0;
  int unmatchedChords = 0;     ///< of them, how many found no other face within their match band
  double length = 0.;          ///< summed chord length, cm
  double unmatchedLength = 0.; ///< how much of it has no other face within the match band, cm
  /// Largest distance from a chord midpoint of this rim to another face's chord, and where: how alone the loneliest chord is.
  double maxIsolation = 0.;
  Vec3 maxIsolationPoint{};
  int maxIsolationFace = -1; ///< the face owning the nearest chord there, or -1 if there was none
  RimState state = RimState::Matched;
};

/// Whether a set of bounded surfaces forms a closed, consistently oriented 2-manifold, by half-edges, rims and edge identities.
struct ClosureReport {
  bool closed = true;                ///< every boundary edge is shared by exactly two faces
  bool orientationConsistent = true; ///< shared edges are traversed in opposite directions
  int boundaryEdges = 0;             ///< edges present on only one face (e.g. a missing face)
  int nonManifoldEdges = 0;          ///< edges shared by more than two faces
  int reversedEdges = 0;             ///< edges shared by two faces in the same direction
  double signedVolume = 0.;          ///< divergence-theorem volume; positive if normals point out

  /// \name Rim-based measurement: the boundary as curves in cm, counted per rim; the verdict when there are no edge identities
  /// @{
  /// Largest distance in cm from any rim chord to the nearest chord of another face; not a seam width.
  double maxRimIsolation = 0.;
  double totalRimLength = 0.;     ///< summed length in cm of every face's trim boundary
  double unmatchedRimLength = 0.; ///< how much of it has no other face within the match band, cm
  double rimEpsilon = 0.;         ///< the declared matching tolerance, in cm
  double rimChordResolution = 0.; ///< the largest amount by which any rim polyline can sit off the
                                  ///< smooth rim it samples, in cm; the per-chord value of this is
                                  ///< what widens the match band
  int rims = 0;                   ///< total number of trim loops over all faces
  int matchedRims = 0;            ///< every chord has another face within the match band,
                                  ///< traversed the opposite way
  int reversedRims = 0;           ///< matched, but the partner traverses the shared curve the
                                  ///< same way (one face's outward normal points inward)
  int nonManifoldRims = 0;        ///< some chord has two or more other faces within rimEpsilon
  int boundaryRims = 0;           ///< some chord has no other face within the match band
  /// One entry per rim, in the order the faces were visited: the detail behind the counters above.
  /// The counters say how many rims are open; these say which, and where.
  std::vector<RimRecord> rimRecords;
  /// @}

  /// \name Closure by edge identity (sidecar v3): an edge is shared when it appears exactly twice, once each way
  /// @{
  /// True when every surface carried a boundary edge list.
  bool edgeIdentityAvailable = false;
  int edgeIncidences = 0;       ///< distinct edge identifiers seen over all faces
  int edgeSharedCount = 0;      ///< appearing exactly twice, opposite sense: a properly shared edge
  int edgeBoundaryCount = 0;    ///< appearing once: a face is missing on the other side
  int edgeNonManifoldCount = 0; ///< appearing three or more times
  int edgeReversedCount = 0;    ///< appearing exactly twice, but with the same sense
  int edgeDegenerateCount = 0;  ///< flagged degenerate (cone apex, sphere pole): excluded from the
                                ///< counts above, because a point has no second face to meet

  /// Largest Hausdorff distance between two faces' realisations of a shared edge, in cm; a measurement, not a verdict.
  double maxSharedEdgeDeviation = 0.;
  uint32_t maxSharedEdgeDeviationEdge = 0;       ///< which edge that was
  Vec3 maxSharedEdgeDeviationPoint{};            ///< and where on it
  int maxSharedEdgeDeviationFaces[2] = {-1, -1}; ///< between which two faces
  int sharedEdgesMeasured = 0;                   ///< shared edges both of whose faces could be sampled
  int sharedEdgesUnmeasured = 0;                 ///< the rest: a parametric-rectangle face names its edges but
                                                 ///< carries no curve for them, so there is nothing to compare
  /// @}
};

/// Measure the Hausdorff distance between the two faces of each shared edge into \a report; it decides nothing.
inline void measureSharedEdgeDeviation(const std::vector<std::unique_ptr<BoundedSurface>>& surfaces,
                                       ClosureReport& report)
{
  // edgeId -> the (surface, slot) pairs claiming it
  std::map<uint32_t, std::vector<std::pair<int, size_t>>> claims;
  for (size_t surfaceIndex = 0; surfaceIndex < surfaces.size(); ++surfaceIndex) {
    if (surfaces[surfaceIndex] == nullptr) {
      continue;
    }
    const auto& refs = surfaces[surfaceIndex]->boundaryEdges();
    for (size_t slot = 0; slot < refs.size(); ++slot) {
      if (refs[slot].degenerate) {
        continue; // a point has no partner and no length to disagree over
      }
      // unanchored claims are collected too, so that an edge whose other side is a
      // parametric-rectangle face is counted as *unmeasured* rather than silently dropped
      claims[refs[slot].edgeId].emplace_back(static_cast<int>(surfaceIndex), slot);
    }
  }

  std::vector<Vec3> first;
  std::vector<Vec3> second;
  for (const auto& [edgeId, holders] : claims) {
    if (holders.size() != 2) {
      continue;
    }
    const auto& [firstSurface, firstSlot] = holders[0];
    const auto& [secondSurface, secondSlot] = holders[1];
    if (!surfaces[static_cast<size_t>(firstSurface)]->sampleTrimCurve(firstSlot, first) ||
        !surfaces[static_cast<size_t>(secondSurface)]->sampleTrimCurve(secondSlot, second) || first.size() < 2 ||
        second.size() < 2) {
      ++report.sharedEdgesUnmeasured;
      continue;
    }
    ++report.sharedEdgesMeasured;
    auto worstAgainst = [](const std::vector<Vec3>& probes, const std::vector<Vec3>& polyline, Vec3& where) {
      double worst = 0.;
      for (const Vec3& probe : probes) {
        double nearest = std::numeric_limits<double>::infinity();
        for (size_t segment = 0; segment + 1 < polyline.size(); ++segment) {
          nearest = std::min(nearest, pointSegmentDistanceSq(probe, polyline[segment], polyline[segment + 1]));
        }
        if (nearest > worst) {
          worst = nearest;
          where = probe;
        }
      }
      return std::sqrt(worst);
    };
    Vec3 forwardPoint{};
    Vec3 backwardPoint{};
    const double forwardWorst = worstAgainst(first, second, forwardPoint);
    const double backwardWorst = worstAgainst(second, first, backwardPoint);
    const double deviation = std::max(forwardWorst, backwardWorst);
    if (deviation > report.maxSharedEdgeDeviation) {
      report.maxSharedEdgeDeviation = deviation;
      report.maxSharedEdgeDeviationEdge = edgeId;
      report.maxSharedEdgeDeviationPoint = forwardWorst >= backwardWorst ? forwardPoint : backwardPoint;
      report.maxSharedEdgeDeviationFaces[0] = firstSurface;
      report.maxSharedEdgeDeviationFaces[1] = secondSurface;
    }
  }
}

/// Measure the face-to-face gaps of \a surfaces as curves into \a report, probing chord midpoints against other faces' chords.
inline void measureRimClosure(const std::vector<std::unique_ptr<BoundedSurface>>& surfaces, double epsilon,
                              ClosureReport& report)
{
  report.rimEpsilon = epsilon;

  std::vector<SurfaceRim> rims;
  std::vector<int> rimIndexOnSurface;
  for (size_t surfaceIndex = 0; surfaceIndex < surfaces.size(); ++surfaceIndex) {
    if (surfaces[surfaceIndex] == nullptr) {
      continue;
    }
    const size_t firstNewRim = rims.size();
    surfaces[surfaceIndex]->appendRims(rims);
    for (size_t rimIndex = firstNewRim; rimIndex < rims.size(); ++rimIndex) {
      rims[rimIndex].surfaceIndex = static_cast<int>(surfaceIndex);
      rimIndexOnSurface.push_back(static_cast<int>(rimIndex - firstNewRim));
    }
  }
  report.rims = static_cast<int>(rims.size());
  if (rims.empty()) {
    return;
  }

  // Flatten to chords with each chord's sagitta: two polylines of one curve differ by it, so it widens the match band.
  // The sagitta is estimated per chord from the turn angle at smooth vertices; a corner has none.
  constexpr double kMaxSmoothTurn = 0.52; // ~30 degrees; a rim sampled at kArcSamples turns by 15
  struct Chord {
    Vec3 start;
    Vec3 end;
    int surfaceIndex;
    double resolution; ///< how far this chord can sit from the smooth rim it samples, in cm
  };
  std::vector<Chord> chords;
  std::vector<std::pair<size_t, size_t>> chordRange(rims.size()); // [first, last) chord of each rim
  for (size_t rimIndex = 0; rimIndex < rims.size(); ++rimIndex) {
    const SurfaceRim& rim = rims[rimIndex];
    chordRange[rimIndex].first = chords.size();
    const size_t pointCount = rim.points.size();
    std::vector<double> vertexSagitta(pointCount, 0.);
    const size_t interiorCount = rim.closed ? pointCount : (pointCount >= 2 ? pointCount - 2 : 0);
    for (size_t offset = 0; offset < interiorCount; ++offset) {
      const size_t middle = rim.closed ? offset : offset + 1;
      const Vec3 incoming = rim.points[middle] - rim.points[(middle + pointCount - 1) % pointCount];
      const Vec3 outgoing = rim.points[(middle + 1) % pointCount] - rim.points[middle];
      const double incomingLength = norm(incoming);
      const double outgoingLength = norm(outgoing);
      if (incomingLength <= kTolerance || outgoingLength <= kTolerance) {
        continue;
      }
      const double turn = std::acos(std::clamp(dot(incoming, outgoing) / (incomingLength * outgoingLength), -1., 1.));
      if (turn > kMaxSmoothTurn) {
        continue; // a corner of the trim, not a sample of a smooth run
      }
      vertexSagitta[middle] = 0.25 * (incomingLength + outgoingLength) * std::tan(0.25 * turn);
      report.rimChordResolution = std::max(report.rimChordResolution, vertexSagitta[middle]);
    }
    const size_t chordCount = rim.closed ? pointCount : pointCount - 1;
    for (size_t pointIndex = 0; pointIndex < chordCount; ++pointIndex) {
      const size_t nextIndex = (pointIndex + 1) % pointCount;
      chords.push_back({rim.points[pointIndex], rim.points[nextIndex], rim.surfaceIndex,
                        std::max(vertexSagitta[pointIndex], vertexSagitta[nextIndex])});
    }
    chordRange[rimIndex].second = chords.size();
  }
  if (chords.empty()) {
    return;
  }

  Vec3 lower{chords.front().start};
  Vec3 upper{chords.front().start};
  auto grow = [&](const Vec3& point) {
    lower = {std::min(lower.xCoord, point.xCoord), std::min(lower.yCoord, point.yCoord),
             std::min(lower.zCoord, point.zCoord)};
    upper = {std::max(upper.xCoord, point.xCoord), std::max(upper.yCoord, point.yCoord),
             std::max(upper.zCoord, point.zCoord)};
  };
  for (const Chord& chord : chords) {
    grow(chord.start);
    grow(chord.end);
  }
  const int gridDimension =
    std::clamp(static_cast<int>(std::cbrt(static_cast<double>(chords.size()))), 1, 32);
  const Vec3 extent = upper - lower;
  const double cellSize =
    std::max({extent.xCoord, extent.yCoord, extent.zCoord, kTolerance}) / gridDimension;
  auto cellOf = [&](double coordinate, double origin) {
    return std::clamp(static_cast<int>((coordinate - origin) / cellSize), 0, gridDimension - 1);
  };
  auto cellIndex = [&](int xCell, int yCell, int zCell) {
    return (xCell * gridDimension + yCell) * gridDimension + zCell;
  };
  std::vector<std::vector<int>> cells(static_cast<size_t>(gridDimension) * gridDimension * gridDimension);
  for (size_t chordIndex = 0; chordIndex < chords.size(); ++chordIndex) {
    const Chord& chord = chords[chordIndex];
    const int xLow = cellOf(std::min(chord.start.xCoord, chord.end.xCoord), lower.xCoord);
    const int xHigh = cellOf(std::max(chord.start.xCoord, chord.end.xCoord), lower.xCoord);
    const int yLow = cellOf(std::min(chord.start.yCoord, chord.end.yCoord), lower.yCoord);
    const int yHigh = cellOf(std::max(chord.start.yCoord, chord.end.yCoord), lower.yCoord);
    const int zLow = cellOf(std::min(chord.start.zCoord, chord.end.zCoord), lower.zCoord);
    const int zHigh = cellOf(std::max(chord.start.zCoord, chord.end.zCoord), lower.zCoord);
    for (int xCell = xLow; xCell <= xHigh; ++xCell) {
      for (int yCell = yLow; yCell <= yHigh; ++yCell) {
        for (int zCell = zLow; zCell <= zHigh; ++zCell) {
          cells[cellIndex(xCell, yCell, zCell)].push_back(static_cast<int>(chordIndex));
        }
      }
    }
  }

  struct Match {
    double distance = std::numeric_limits<double>::infinity();
    int chordIndex = -1;
    /// Another face's chord lies within this chord's match band.
    bool withinBand = false;
    /// The distinct faces found within the declared tolerance alone. Room for three is enough:
    /// only none, one and "more than one" are distinguished, and only the last is used.
    std::array<int, 3> coincidentFaces{-1, -1, -1};
    int coincidentFaceCount = 0;
  };
  // Two bands: shared-edge matching uses the sampling-aware band, non-manifold detection the declared tolerance alone.
  const double maxBand = epsilon + 2. * report.rimChordResolution;
  auto nearestOtherFace = [&](const Vec3& probe, int ownSurfaceIndex, double probeResolution) {
    Match match;
    auto consider = [&](int chordIndex) {
      const Chord& chord = chords[static_cast<size_t>(chordIndex)];
      if (chord.surfaceIndex == ownSurfaceIndex) {
        return;
      }
      const double distance = std::sqrt(pointSegmentDistanceSq(probe, chord.start, chord.end));
      if (distance < match.distance) {
        match.distance = distance;
        match.chordIndex = chordIndex;
      }
      if (distance <= epsilon + probeResolution + chord.resolution) {
        match.withinBand = true;
      }
      if (distance <= epsilon && match.coincidentFaceCount < static_cast<int>(match.coincidentFaces.size())) {
        for (int seen = 0; seen < match.coincidentFaceCount; ++seen) {
          if (match.coincidentFaces[static_cast<size_t>(seen)] == chord.surfaceIndex) {
            return;
          }
        }
        match.coincidentFaces[static_cast<size_t>(match.coincidentFaceCount++)] = chord.surfaceIndex;
      }
    };
    const int xCentre = cellOf(probe.xCoord, lower.xCoord);
    const int yCentre = cellOf(probe.yCoord, lower.yCoord);
    const int zCentre = cellOf(probe.zCoord, lower.zCoord);
    for (int shell = 0; shell < gridDimension; ++shell) {
      // stop once the nearest hit is closer than this shell's inner distance and the shells reach the match band
      const double shellReach = (shell - 1) * cellSize;
      if (shell > 0 && shellReach > std::max(match.distance, maxBand)) {
        break;
      }
      for (int xCell = xCentre - shell; xCell <= xCentre + shell; ++xCell) {
        if (xCell < 0 || xCell >= gridDimension) {
          continue;
        }
        for (int yCell = yCentre - shell; yCell <= yCentre + shell; ++yCell) {
          if (yCell < 0 || yCell >= gridDimension) {
            continue;
          }
          for (int zCell = zCentre - shell; zCell <= zCentre + shell; ++zCell) {
            if (zCell < 0 || zCell >= gridDimension) {
              continue;
            }
            const bool onShell = std::abs(xCell - xCentre) == shell || std::abs(yCell - yCentre) == shell ||
                                 std::abs(zCell - zCentre) == shell;
            if (!onShell) {
              continue; // interior of the shell: visited on an earlier pass
            }
            for (const int chordIndex : cells[cellIndex(xCell, yCell, zCell)]) {
              consider(chordIndex);
            }
          }
        }
      }
    }
    return match;
  };

  report.rimRecords.reserve(rims.size());
  for (size_t rimIndex = 0; rimIndex < rims.size(); ++rimIndex) {
    bool hasUnmatched = false;
    bool hasNonManifold = false;
    int sameDirectionVotes = 0;
    int oppositeDirectionVotes = 0;
    RimRecord record;
    record.surfaceIndex = rims[rimIndex].surfaceIndex;
    record.rimIndexOnSurface = rimIndexOnSurface[rimIndex];
    record.closed = rims[rimIndex].closed;
    record.chords = static_cast<int>(chordRange[rimIndex].second - chordRange[rimIndex].first);
    for (size_t chordIndex = chordRange[rimIndex].first; chordIndex < chordRange[rimIndex].second; ++chordIndex) {
      const Chord& chord = chords[chordIndex];
      const Vec3 along = chord.end - chord.start;
      const double chordLength = norm(along);
      report.totalRimLength += chordLength;
      record.length += chordLength;
      const Vec3 probe = chord.start + along * 0.5;
      const Match match = nearestOtherFace(probe, chord.surfaceIndex, chord.resolution);
      if (std::isfinite(match.distance)) {
        report.maxRimIsolation = std::max(report.maxRimIsolation, match.distance);
        if (match.distance > record.maxIsolation || record.maxIsolationFace < 0) {
          record.maxIsolation = match.distance;
          record.maxIsolationPoint = probe;
          record.maxIsolationFace = chords[static_cast<size_t>(match.chordIndex)].surfaceIndex;
        }
      }
      if (match.coincidentFaceCount > 1) {
        hasNonManifold = true;
      }
      if (!match.withinBand) {
        hasUnmatched = true;
        ++record.unmatchedChords;
        report.unmatchedRimLength += chordLength;
        record.unmatchedLength += chordLength;
        continue;
      }
      const Chord& partner = chords[static_cast<size_t>(match.chordIndex)];
      if (dot(along, partner.end - partner.start) < 0.) {
        ++oppositeDirectionVotes;
      } else {
        ++sameDirectionVotes;
      }
    }
    if (hasNonManifold) {
      ++report.nonManifoldRims;
      record.state = RimState::NonManifold;
    } else if (hasUnmatched) {
      ++report.boundaryRims;
      record.state = RimState::Boundary;
    } else if (sameDirectionVotes > oppositeDirectionVotes) {
      ++report.reversedRims;
      record.state = RimState::Reversed;
    } else {
      ++report.matchedRims;
      record.state = RimState::Matched;
    }
    report.rimRecords.push_back(record);
  }
}

/// Decide closure by counting edge identities when every surface states them: twice opposite is shared, once is open,
/// three or more is non-manifold, twice same-sense is reversed; degenerate edges are excluded.
inline void applyEdgeIdentityClosure(const std::vector<std::unique_ptr<BoundedSurface>>& surfaces,
                                     ClosureReport& report)
{
  size_t surfacesPresent = 0;
  size_t surfacesStatingEdges = 0;
  for (const auto& surface : surfaces) {
    if (surface == nullptr) {
      continue;
    }
    ++surfacesPresent;
    if (!surface->boundaryEdges().empty()) {
      ++surfacesStatingEdges;
    }
  }
  if (surfacesPresent == 0 || surfacesStatingEdges != surfacesPresent) {
    return; // no edge identity, or only some of it: leave the geometric verdict alone
  }
  report.edgeIdentityAvailable = true;

  struct Incidence {
    int forward = 0;
    int reversed = 0;
    int degenerate = 0;
  };
  std::map<uint32_t, Incidence> incidences;
  // which faces own each edge, so a defect can be attributed back to a rim
  std::map<uint32_t, std::vector<int>> owners;
  for (size_t surfaceIndex = 0; surfaceIndex < surfaces.size(); ++surfaceIndex) {
    if (surfaces[surfaceIndex] == nullptr) {
      continue;
    }
    for (const auto& ref : surfaces[surfaceIndex]->boundaryEdges()) {
      Incidence& incidence = incidences[ref.edgeId];
      if (ref.degenerate) {
        ++incidence.degenerate;
      } else if (ref.reversed) {
        ++incidence.reversed;
      } else {
        ++incidence.forward;
      }
      owners[ref.edgeId].push_back(static_cast<int>(surfaceIndex));
    }
  }

  // per face, the worst identity defect any of its edges carries
  std::vector<RimState> faceState(surfaces.size(), RimState::Matched);
  auto worsen = [](RimState& state, RimState candidate) {
    // the enum is not ordered by severity, so spell the precedence out
    auto rank = [](RimState value) {
      switch (value) {
        case RimState::Matched:
          return 0;
        case RimState::Reversed:
          return 1;
        case RimState::Boundary:
          return 2;
        case RimState::NonManifold:
          return 3;
      }
      return 0;
    };
    if (rank(candidate) > rank(state)) {
      state = candidate;
    }
  };

  for (const auto& [edgeId, incidence] : incidences) {
    ++report.edgeIncidences;
    if (incidence.degenerate > 0 && incidence.forward + incidence.reversed == 0) {
      ++report.edgeDegenerateCount;
      continue;
    }
    const int total = incidence.forward + incidence.reversed;
    RimState state = RimState::Matched;
    if (total == 1) {
      ++report.edgeBoundaryCount;
      state = RimState::Boundary;
    } else if (total == 2) {
      if (incidence.forward == 1 && incidence.reversed == 1) {
        ++report.edgeSharedCount;
      } else {
        ++report.edgeReversedCount;
        state = RimState::Reversed;
      }
    } else {
      ++report.edgeNonManifoldCount;
      state = RimState::NonManifold;
    }
    if (state != RimState::Matched) {
      for (const int owner : owners[edgeId]) {
        worsen(faceState[static_cast<size_t>(owner)], state);
      }
    }
  }

  report.closed = (report.edgeBoundaryCount == 0) && (report.edgeNonManifoldCount == 0);
  report.orientationConsistent = (report.edgeReversedCount == 0);

  report.matchedRims = 0;
  report.boundaryRims = 0;
  report.nonManifoldRims = 0;
  report.reversedRims = 0;
  for (RimRecord& record : report.rimRecords) {
    const RimState state = record.surfaceIndex >= 0 && record.surfaceIndex < static_cast<int>(faceState.size())
                             ? faceState[static_cast<size_t>(record.surfaceIndex)]
                             : RimState::Matched;
    record.state = state;
    switch (state) {
      case RimState::NonManifold:
        ++report.nonManifoldRims;
        break;
      case RimState::Boundary:
        ++report.boundaryRims;
        break;
      case RimState::Reversed:
        ++report.reversedRims;
        break;
      case RimState::Matched:
        ++report.matchedRims;
        break;
    }
  }

  measureSharedEdgeDeviation(surfaces, report);
}

/// Validate closure and orientation of \a surfaces by half-edges, measure the rims, and count edge identities when present.
inline ClosureReport validateClosure(const std::vector<std::unique_ptr<BoundedSurface>>& surfaces,
                                     double modelTolerance = 0.)
{
  ClosureReport report;

  auto quantize = [](double value) { return static_cast<int64_t>(std::llround(value / kClosureQuantum)); };
  using VertexKey = std::tuple<int64_t, int64_t, int64_t>;
  auto keyOf = [&](const Vec3& point) {
    return VertexKey{quantize(point.xCoord), quantize(point.yCoord), quantize(point.zCoord)};
  };

  std::vector<std::pair<Vec3, Vec3>> directedEdges;
  for (const auto& surface : surfaces) {
    if (surface != nullptr) {
      surface->appendDirectedEdges(directedEdges);
      report.signedVolume += surface->capacityContribution();
    }
  }

  // For each undirected edge, count occurrences in the forward and reverse directions.
  std::map<std::pair<VertexKey, VertexKey>, std::pair<int, int>> edgeCounts;
  for (const auto& directedEdge : directedEdges) {
    const VertexKey startKey = keyOf(directedEdge.first);
    const VertexKey endKey = keyOf(directedEdge.second);
    if (startKey == endKey) {
      continue; // degenerate edge, already flagged at wire level
    }
    const bool forward = startKey < endKey;
    const auto orderedKey = forward ? std::make_pair(startKey, endKey) : std::make_pair(endKey, startKey);
    auto& counts = edgeCounts[orderedKey];
    if (forward) {
      ++counts.first;
    } else {
      ++counts.second;
    }
  }

  for (const auto& [edgeKey, counts] : edgeCounts) {
    const int total = counts.first + counts.second;
    if (total == 1) {
      ++report.boundaryEdges; // missing neighbouring face
    } else if (total == 2) {
      if (counts.first != 1 || counts.second != 1) {
        ++report.reversedEdges; // both faces traverse the edge the same way
      }
    } else {
      ++report.nonManifoldEdges;
    }
  }

  measureRimClosure(surfaces, modelTolerance > 0. ? modelTolerance : kRimMatchTolerance, report);

  // the verdict is the rim measurement's; the chord counters only describe how faces differ
  report.closed = (report.boundaryRims == 0) && (report.nonManifoldRims == 0);
  report.orientationConsistent = (report.reversedRims == 0);

  // ... unless the surfaces state their edge identities, which then decide by counting
  applyEdgeIdentityClosure(surfaces, report);
  return report;
}

} // namespace o2::cad::surface

#endif
