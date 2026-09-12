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

#define BOOST_TEST_MODULE Test O2BVHSurfaceSolid class
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include "CADSupport/O2BVHSurfaceSolid.h"
#include "CADSupport/O2SurfaceSolidIO.h"
#include "CADSupport/O2SolidHarness.h"

#include "../src/BoundedSurface.h"

#include "TFile.h"
#include "TGeoBBox.h"
#include "TGeoBoolNode.h"
#include "TGeoCompositeShape.h"
#include "TGeoCone.h"
#include "TGeoManager.h"
#include "TGeoMaterial.h"
#include "TGeoMatrix.h"
#include "TGeoMedium.h"
#include "TGeoNode.h"
#include "TGeoShape.h"
#include "TGeoSphere.h"
#include "TGeoTorus.h"
#include "TGeoTube.h"
#include "TGeoVolume.h"
#include "TMath.h"
#include "TNamed.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <string>
#include <memory>
#include <utility>
#include <vector>

namespace
{
using SurfaceSolid = o2::cad::O2BVHSurfaceSolid;
using Point2D = SurfaceSolid::Point2D;
using Point3D = SurfaceSolid::Point3D;
namespace surf = o2::cad::surface;

std::vector<Point2D> rectangleWire(double extentU, double extentV)
{
  return {{0., 0.}, {extentU, 0.}, {extentU, extentV}, {0., extentV}};
}

using BoundaryCurve = SurfaceSolid::PlanarBoundaryCurve;

// A full-circle boundary wire centred at (0,0) as a single +/-2pi arc (clockwise for holes).
std::vector<BoundaryCurve> circleWire(double radius, bool clockwise = false)
{
  return {BoundaryCurve::makeArc({0., 0.}, radius, 0., clockwise ? -surf::kTwoPi : surf::kTwoPi)};
}

// A rectangular trim loop in a quadric's (u, v) parametric domain, as four line boundary curves
// (u = phi, v = height or theta). Wound counter-clockwise; the kernel reorients as needed.
std::vector<BoundaryCurve> paramRectWire(double uMin, double uMax, double vMin, double vMax)
{
  return {BoundaryCurve::makeLine({uMin, vMin}, {uMax, vMin}), BoundaryCurve::makeLine({uMax, vMin}, {uMax, vMax}),
          BoundaryCurve::makeLine({uMax, vMax}, {uMin, vMax}), BoundaryCurve::makeLine({uMin, vMax}, {uMin, vMin})};
}

// Same rectangle as paramRectWire but as internal Curve2D segments, for direct kernel-level tests.
std::vector<surf::Curve2D> paramRectWireCurves(double uMin, double uMax, double vMin, double vMax)
{
  return {surf::Curve2D::makeLine({uMin, vMin}, {uMax, vMin}), surf::Curve2D::makeLine({uMax, vMin}, {uMax, vMax}),
          surf::Curve2D::makeLine({uMax, vMax}, {uMin, vMax}), surf::Curve2D::makeLine({uMin, vMax}, {uMin, vMin})};
}

// Add a planar disk (or annulus when holeRadius > 0) via the general curved-planar API,
// replacing the retired AddPlanarDiskSurface convenience.
bool addDiskSurface(SurfaceSolid& solid, const Point3D& center, const Point3D& axisU, const Point3D& axisV,
                    double radius, double holeRadius = 0.)
{
  std::vector<std::vector<BoundaryCurve>> inners;
  if (holeRadius > 0.) {
    inners.push_back(circleWire(holeRadius, true)); // clockwise hole: no reorientation needed
  }
  return solid.AddCurvedPlanarSurface(center, axisU, axisV, circleWire(radius), inners);
}

// Local frame (origin + parametric axes + rectangle extents) of a box face by index
// (0:+x 1:-x 2:+y 3:-y 4:+z 5:-z), for a box centred at the origin.
struct FaceFrame {
  Point3D origin;
  Point3D axisU;
  Point3D axisV;
  double extentU;
  double extentV;
};

FaceFrame boxFaceFrame(int faceIndex, double halfX, double halfY, double halfZ)
{
  switch (faceIndex) {
    case 0:
      return {{halfX, -halfY, -halfZ}, {0., 1., 0.}, {0., 0., 1.}, 2. * halfY, 2. * halfZ};
    case 1:
      return {{-halfX, -halfY, -halfZ}, {0., 0., 1.}, {0., 1., 0.}, 2. * halfZ, 2. * halfY};
    case 2:
      return {{-halfX, halfY, -halfZ}, {0., 0., 1.}, {1., 0., 0.}, 2. * halfZ, 2. * halfX};
    case 3:
      return {{-halfX, -halfY, -halfZ}, {1., 0., 0.}, {0., 0., 1.}, 2. * halfX, 2. * halfZ};
    case 4:
      return {{-halfX, -halfY, halfZ}, {1., 0., 0.}, {0., 1., 0.}, 2. * halfX, 2. * halfY};
    default:
      return {{-halfX, -halfY, -halfZ}, {0., 1., 0.}, {1., 0., 0.}, 2. * halfY, 2. * halfX};
  }
}

// Add a single box face by index of a box centred at "center". When "reversed" is set the
// face's parametric axes are swapped, which flips the outward normal inward without changing
// the covered rectangle - used to build an orientation-inconsistent fixture.
bool addBoxFace(SurfaceSolid& solid, int faceIndex, double halfX, double halfY, double halfZ, bool reversed = false,
                const Point3D& center = {0., 0., 0.})
{
  FaceFrame frame = boxFaceFrame(faceIndex, halfX, halfY, halfZ);
  if (reversed) {
    std::swap(frame.axisU, frame.axisV);
    std::swap(frame.extentU, frame.extentV);
  }
  for (int dimension = 0; dimension < 3; ++dimension) {
    frame.origin[dimension] += center[dimension];
  }
  return solid.AddPlanarSurface(frame.origin, frame.axisU, frame.axisV, rectangleWire(frame.extentU, frame.extentV));
}

void addBoxSurfaces(SurfaceSolid& solid, double halfX, double halfY, double halfZ,
                    const Point3D& center = {0., 0., 0.})
{
  for (int faceIndex = 0; faceIndex < 6; ++faceIndex) {
    BOOST_REQUIRE(addBoxFace(solid, faceIndex, halfX, halfY, halfZ, false, center));
  }
}

surf::SurfaceWire makeWire(const std::vector<surf::Vec2>& vertices, surf::WireRole role, surf::WireStatus& status)
{
  surf::SurfaceWire wire;
  wire.initialize(vertices, role, status);
  return wire;
}

void checkClose(double value, double reference, double tolerance = 1.e-9)
{
  BOOST_CHECK_SMALL(value - reference, tolerance);
}

std::array<double, 3> unitDirection(double x, double y, double z)
{
  const double length = std::sqrt(x * x + y * y + z * z);
  return {x / length, y / length, z / length};
}

// Compare Contains against a reference ROOT shape on a regular grid. The fractional offsets keep
// grid points away from exact shape boundaries, where inside/outside conventions may differ.
void compareContainsGrid(const SurfaceSolid& solid, const TGeoShape& reference, double extent, int samples)
{
  for (int stepX = 0; stepX < samples; ++stepX) {
    for (int stepY = 0; stepY < samples; ++stepY) {
      for (int stepZ = 0; stepZ < samples; ++stepZ) {
        const double point[3] = {-extent + 2. * extent * (stepX + 0.517) / samples,
                                 -extent + 2. * extent * (stepY + 0.263) / samples,
                                 -extent + 2. * extent * (stepZ + 0.741) / samples};
        BOOST_TEST_CONTEXT("point = (" << point[0] << ", " << point[1] << ", " << point[2] << ")")
        {
          BOOST_CHECK_EQUAL(solid.Contains(point), reference.Contains(point));
        }
      }
    }
  }
}

// Compare the direction-appropriate distance function against a reference ROOT shape.
void compareDistance(const SurfaceSolid& solid, const TGeoShape& reference, const std::array<double, 3>& point,
                     const std::array<double, 3>& direction, double tolerance = 1.e-9)
{
  BOOST_TEST_CONTEXT("point = (" << point[0] << ", " << point[1] << ", " << point[2] << ") direction = ("
                                 << direction[0] << ", " << direction[1] << ", " << direction[2] << ")")
  {
    const bool inside = reference.Contains(point.data());
    BOOST_CHECK_EQUAL(solid.Contains(point.data()), inside);
    if (inside) {
      checkClose(solid.DistFromInside(point.data(), direction.data(), 3),
                 reference.DistFromInside(point.data(), direction.data(), 3), tolerance);
    } else {
      checkClose(solid.DistFromOutside(point.data(), direction.data(), 3),
                 reference.DistFromOutside(point.data(), direction.data(), 3), tolerance);
    }
  }
}

/// @name Closed fixtures for the navigation sweeps
///
/// The distance tests exercise the same solids from several directions rather than one shape per
/// case, so the fixtures are built once here. Each is closed and therefore BVH-backed. The solid
/// is neither copyable nor movable, hence the unique_ptr.
/// @{

std::unique_ptr<SurfaceSolid> makeBoxSolid(const char* name, double halfX, double halfY, double halfZ)
{
  auto solid = std::make_unique<SurfaceSolid>(name);
  addBoxSurfaces(*solid, halfX, halfY, halfZ);
  solid->CloseShape();
  return solid;
}

// innerRadius > 0 gives a hollow tube (an inner wall plus annular caps).
std::unique_ptr<SurfaceSolid> makeTubeSolid(const char* name, double innerRadius, double outerRadius,
                                            double halfHeight)
{
  auto solid = std::make_unique<SurfaceSolid>(name);
  BOOST_REQUIRE(solid->AddCylindricalSurface({0., 0., 0.}, {0., 0., 1.}, {1., 0., 0.}, outerRadius, -halfHeight,
                                             halfHeight));
  if (innerRadius > 0.) {
    BOOST_REQUIRE(solid->AddCylindricalSurface({0., 0., 0.}, {0., 0., 1.}, {1., 0., 0.}, innerRadius, -halfHeight,
                                               halfHeight, 0., surf::kTwoPi, true));
  }
  BOOST_REQUIRE(addDiskSurface(*solid, {0., 0., halfHeight}, {1., 0., 0.}, {0., 1., 0.}, outerRadius, innerRadius));
  BOOST_REQUIRE(addDiskSurface(*solid, {0., 0., -halfHeight}, {1., 0., 0.}, {0., -1., 0.}, outerRadius, innerRadius));
  solid->CloseShape();
  return solid;
}

std::unique_ptr<SurfaceSolid> makeConeSolid(const char* name, double radiusAtBottom, double radiusAtTop,
                                            double halfHeight)
{
  auto solid = std::make_unique<SurfaceSolid>(name);
  BOOST_REQUIRE(solid->AddConicalSurface({0., 0., 0.}, {0., 0., 1.}, {1., 0., 0.}, radiusAtBottom, radiusAtTop,
                                         -halfHeight, halfHeight));
  BOOST_REQUIRE(addDiskSurface(*solid, {0., 0., halfHeight}, {1., 0., 0.}, {0., 1., 0.}, radiusAtTop));
  BOOST_REQUIRE(addDiskSurface(*solid, {0., 0., -halfHeight}, {1., 0., 0.}, {0., -1., 0.}, radiusAtBottom));
  solid->CloseShape();
  return solid;
}

std::unique_ptr<SurfaceSolid> makeSphereSolid(const char* name, double radius)
{
  auto solid = std::make_unique<SurfaceSolid>(name);
  BOOST_REQUIRE(solid->AddSphericalSurface({0., 0., 0.}, {0., 0., 1.}, {1., 0., 0.}, radius));
  solid->CloseShape();
  return solid;
}

std::unique_ptr<SurfaceSolid> makeTorusSolid(const char* name, double majorRadius, double minorRadius)
{
  auto solid = std::make_unique<SurfaceSolid>(name);
  BOOST_REQUIRE(solid->AddToroidalSurface({0., 0., 0.}, {0., 0., 1.}, {1., 0., 0.}, majorRadius, minorRadius));
  solid->CloseShape();
  return solid;
}

// Cylinder barrel closed by two hemispherical endcaps; a mixed-quadric solid with no ROOT
// primitive equivalent, so the loop oracle is the only reference it has.
std::unique_ptr<SurfaceSolid> makeCapsuleSolid(const char* name, double radius, double halfHeight)
{
  auto solid = std::make_unique<SurfaceSolid>(name);
  BOOST_REQUIRE(solid->AddCylindricalSurface({0., 0., 0.}, {0., 0., 1.}, {1., 0., 0.}, radius, -halfHeight,
                                             halfHeight));
  BOOST_REQUIRE(solid->AddSphericalSurface({0., 0., halfHeight}, {0., 0., 1.}, {1., 0., 0.}, radius, 0.,
                                           surf::kPi / 2.));
  BOOST_REQUIRE(solid->AddSphericalSurface({0., 0., -halfHeight}, {0., 0., -1.}, {1., 0., 0.}, radius, 0.,
                                           surf::kPi / 2.));
  solid->CloseShape();
  return solid;
}

/// @}

// Directions probed by the navigation sweeps: the three axes both ways, face and body diagonals,
// and a few skew directions that align with no symmetry of any fixture.
const std::vector<std::array<double, 3>>& probeDirections()
{
  static const std::vector<std::array<double, 3>> directions{
    {1., 0., 0.}, {-1., 0., 0.}, {0., 1., 0.}, {0., -1., 0.}, {0., 0., 1.}, {0., 0., -1.}, unitDirection(1., 1., 0.), unitDirection(1., 0., 1.), unitDirection(0., 1., 1.), unitDirection(1., 1., 1.), unitDirection(-1., 1., -1.), unitDirection(0.37, -0.82, 0.44), unitDirection(-0.91, 0.13, 0.39), unitDirection(0.21, 0.55, -0.81)};
  return directions;
}

// A deterministic point grid over the cube of half-side "extent". The fractional offsets are the
// same irrational-looking shifts the Contains sweeps use, which keeps samples off exact symmetry
// planes and shape boundaries.
std::vector<std::array<double, 3>> probeGrid(double extent, int samples)
{
  std::vector<std::array<double, 3>> points;
  points.reserve(static_cast<size_t>(samples) * samples * samples);
  for (int stepX = 0; stepX < samples; ++stepX) {
    for (int stepY = 0; stepY < samples; ++stepY) {
      for (int stepZ = 0; stepZ < samples; ++stepZ) {
        points.push_back({-extent + 2. * extent * (stepX + 0.517) / samples,
                          -extent + 2. * extent * (stepY + 0.263) / samples,
                          -extent + 2. * extent * (stepZ + 0.741) / samples});
      }
    }
  }
  return points;
}

/// The BVH distance queries must return *exactly* what the all-surfaces loop returns. Both run
/// the same analytic kernels on the same patches and take a minimum over the same hit set; they
/// differ only in the order surfaces are visited and in which of them the BVH lets them skip. So
/// any difference at all -- not merely one above a tolerance -- is a traversal or pruning bug,
/// and exact comparison is the sharpest available oracle. Independent of any mesh reference.
///
/// Ray tmax tightening is an optimization and nothing else, so both settings are checked and must
/// agree with the same loop value.
void checkDistanceAgainstLoop(const SurfaceSolid& solid, const std::array<double, 3>& point,
                              const std::array<double, 3>& direction, double stepmax = TGeoShape::Big())
{
  const double loopOutside = solid.DistFromOutside_Loop(point.data(), direction.data(), stepmax);
  const double loopInside = solid.DistFromInside_Loop(point.data(), direction.data(), stepmax);
  for (const bool pruning : {true, false}) {
    SurfaceSolid::SetRayTMaxPruning(pruning);
    BOOST_TEST_CONTEXT("point = (" << point[0] << ", " << point[1] << ", " << point[2] << ") direction = ("
                                   << direction[0] << ", " << direction[1] << ", " << direction[2]
                                   << ") stepmax = " << stepmax << " pruning = " << pruning)
    {
      BOOST_CHECK_EQUAL(solid.DistFromOutside(point.data(), direction.data(), 3, stepmax), loopOutside);
      BOOST_CHECK_EQUAL(solid.DistFromInside(point.data(), direction.data(), 3, stepmax), loopInside);
    }
  }
  SurfaceSolid::SetRayTMaxPruning(true);
}

// Sweep every grid point against every probe direction, cross-checking BVH against the loop.
void sweepDistanceAgainstLoop(const SurfaceSolid& solid, double extent, int samples)
{
  for (const auto& point : probeGrid(extent, samples)) {
    for (const auto& direction : probeDirections()) {
      checkDistanceAgainstLoop(solid, point, direction);
    }
  }
}

// Sweep both distance functions against a reference ROOT primitive, using each point in the role
// (inside/outside) the reference itself assigns it. Points closer than "skin" to the reference
// boundary are skipped: there the two shapes may legitimately disagree on which side the point is
// on, and the resulting distances are then answers to different questions.
void sweepDistanceAgainstReference(const SurfaceSolid& solid, const TGeoShape& reference, double extent, int samples,
                                   double tolerance = 1.e-9, double skin = 1.e-6)
{
  for (const auto& point : probeGrid(extent, samples)) {
    const bool inside = reference.Contains(point.data());
    if (reference.Safety(point.data(), inside) < skin) {
      continue;
    }
    for (const auto& direction : probeDirections()) {
      BOOST_TEST_CONTEXT("point = (" << point[0] << ", " << point[1] << ", " << point[2] << ") direction = ("
                                     << direction[0] << ", " << direction[1] << ", " << direction[2] << ")")
      {
        if (inside) {
          checkClose(solid.DistFromInside(point.data(), direction.data(), 3),
                     reference.DistFromInside(point.data(), direction.data(), 3), tolerance);
        } else {
          checkClose(solid.DistFromOutside(point.data(), direction.data(), 3),
                     reference.DistFromOutside(point.data(), direction.data(), 3), tolerance);
        }
      }
    }
  }
}

} // namespace

BOOST_AUTO_TEST_CASE(PlanarBoxNavigationMatchesTGeoBBox)
{
  constexpr double halfX = 1.;
  constexpr double halfY = 2.;
  constexpr double halfZ = 3.;

  SurfaceSolid solid("planarBox");
  addBoxSurfaces(solid, halfX, halfY, halfZ);
  solid.CloseShape();

  TGeoBBox reference("referenceBox", halfX, halfY, halfZ);

  BOOST_CHECK(solid.IsDefined());
  BOOST_CHECK_EQUAL(solid.GetNsurfaces(), 6);

  int meshVertices = 0;
  int meshSegments = 0;
  int meshPolygons = 0;
  solid.GetMeshNumbers(meshVertices, meshSegments, meshPolygons);
  BOOST_CHECK_EQUAL(meshVertices, 24);
  BOOST_CHECK_EQUAL(meshSegments, 36);
  BOOST_CHECK_EQUAL(meshPolygons, 12);

  const std::array<std::array<double, 3>, 5> insidePoints{{{0., 0., 0.}, {0.9, 0., 0.}, {1., 0., 0.}, {1., 2., 3.}, {-1., -2., -3.}}};
  for (const auto& point : insidePoints) {
    BOOST_TEST_CONTEXT("point = (" << point[0] << ", " << point[1] << ", " << point[2] << ")")
    {
      BOOST_CHECK(solid.Contains(point.data()));
      BOOST_CHECK(reference.Contains(point.data()));
    }
  }

  const std::array<std::array<double, 3>, 4> outsidePoints{{{1.1, 0., 0.}, {0., 2.1, 0.}, {0., 0., -3.1}, {2., 3., 4.}}};
  for (const auto& point : outsidePoints) {
    BOOST_TEST_CONTEXT("point = (" << point[0] << ", " << point[1] << ", " << point[2] << ")")
    {
      BOOST_CHECK(!solid.Contains(point.data()));
      BOOST_CHECK(!reference.Contains(point.data()));
    }
  }

  const double fromLeft[3] = {-3., 0., 0.};
  const double toRight[3] = {1., 0., 0.};
  checkClose(solid.DistFromOutside(fromLeft, toRight, 3), reference.DistFromOutside(fromLeft, toRight, 3));

  const double fromFront[3] = {0., -5., 0.};
  const double toBack[3] = {0., 1., 0.};
  checkClose(solid.DistFromOutside(fromFront, toBack, 3), reference.DistFromOutside(fromFront, toBack, 3));

  const double fromCenter[3] = {0., 0., 0.};
  const double alongX[3] = {1., 0., 0.};
  const double alongZ[3] = {0., 0., -1.};
  checkClose(solid.DistFromInside(fromCenter, alongX, 3), reference.DistFromInside(fromCenter, alongX, 3));
  checkClose(solid.DistFromInside(fromCenter, alongZ, 3), reference.DistFromInside(fromCenter, alongZ, 3));

  // safeties against analytic distances (TGeo safeties may be weaker underestimates)
  const double outsideSafetyPoint[3] = {2.5, 0., 0.};
  checkClose(solid.Safety(fromCenter, kTRUE), halfX);
  checkClose(solid.Safety(outsideSafetyPoint, kFALSE), 2.5 - halfX);

  const double normalPoint[3] = {halfX, 0., 0.};
  double normal[3] = {0., 0., 0.};
  solid.ComputeNormal(normalPoint, alongX, normal);
  checkClose(normal[0], 1.);
  checkClose(normal[1], 0.);
  checkClose(normal[2], 0.);

  checkClose(solid.Capacity(), 8. * halfX * halfY * halfZ);

  BOOST_CHECK(solid.IsClosed());
  BOOST_CHECK(solid.IsOrientationConsistent());
}

BOOST_AUTO_TEST_CASE(WireValidationAndOrientation)
{
  using surf::WireRole;
  using surf::WireStatus;

  // outer square given clockwise (negative area) must be re-oriented to CCW
  WireStatus reversedStatus = WireStatus::Valid;
  auto reversedOuter = makeWire({{0., 0.}, {0., 1.}, {1., 1.}, {1., 0.}}, WireRole::Outer, reversedStatus);
  BOOST_CHECK(reversedStatus == WireStatus::Reversed);
  BOOST_CHECK_GT(reversedOuter.signedArea(), 0.);

  // outer square already CCW stays valid
  WireStatus outerStatus = WireStatus::Valid;
  auto outerWire = makeWire({{0., 0.}, {1., 0.}, {1., 1.}, {0., 1.}}, WireRole::Outer, outerStatus);
  BOOST_CHECK(outerStatus == WireStatus::Valid);
  BOOST_CHECK_GT(outerWire.signedArea(), 0.);

  // inner (hole) wire must end up clockwise (negative area)
  WireStatus innerStatus = WireStatus::Valid;
  auto innerWire = makeWire({{0., 0.}, {1., 0.}, {1., 1.}, {0., 1.}}, WireRole::Inner, innerStatus);
  BOOST_CHECK(innerStatus == WireStatus::Reversed);
  BOOST_CHECK_LT(innerWire.signedArea(), 0.);

  // degenerate / invalid inputs are rejected with a specific status
  surf::SurfaceWire scratch;
  WireStatus status = WireStatus::Valid;
  BOOST_CHECK(!scratch.initialize({{0., 0.}, {1., 0.}}, WireRole::Outer, status));
  BOOST_CHECK(status == WireStatus::TooFewVertices);

  BOOST_CHECK(!scratch.initialize({{0., 0.}, {1., 0.}, {2., 0.}}, WireRole::Outer, status));
  BOOST_CHECK(status == WireStatus::ZeroArea);

  // self-touching (pinched) loop: a non-adjacent vertex repeats
  BOOST_CHECK(!scratch.initialize({{0., 0.}, {1., 0.}, {0., 0.}, {1., 1.}}, WireRole::Outer, status));
  BOOST_CHECK(status == WireStatus::DegenerateVertex);

  // explicit edge list that does not close is flagged as open
  const std::vector<surf::SurfaceEdge> openEdges{{{0., 0.}, {1., 0.}}, {{1., 0.}, {1., 1.}}, {{1., 1.}, {0.5, 0.5}}};
  BOOST_CHECK(!scratch.initializeFromEdges(openEdges, WireRole::Outer, status));
  BOOST_CHECK(status == WireStatus::Open);

  // a closed edge list is accepted
  const std::vector<surf::SurfaceEdge> closedEdges{
    {{0., 0.}, {1., 0.}}, {{1., 0.}, {1., 1.}}, {{1., 1.}, {0., 1.}}, {{0., 1.}, {0., 0.}}};
  BOOST_CHECK(scratch.initializeFromEdges(closedEdges, WireRole::Outer, status));

  // point classification: inside, outside, and on-edge
  BOOST_CHECK(outerWire.classify({0.5, 0.5}) == surf::WireClassification::Inside);
  BOOST_CHECK(outerWire.classify({1.5, 0.5}) == surf::WireClassification::Outside);
  BOOST_CHECK(outerWire.classify({0.5, 0.}) == surf::WireClassification::Boundary);
}

namespace o2::cad::surface
{
/// A trivial bounded surface, a single 3D triangle, to exercise the BoundedSurface interface.
class DummyBoundedSurface final : public BoundedSurface
{
 public:
  DummyBoundedSurface(const Vec3& firstVertex, const Vec3& secondVertex, const Vec3& thirdVertex)
    : mVertices{firstVertex, secondVertex, thirdVertex}
  {
    mNormal = normalized(cross(secondVertex - firstVertex, thirdVertex - firstVertex));
  }

  void conservativeBounds(Vec3& lower, Vec3& upper) const override
  {
    for (const auto& vertex : mVertices) {
      lower.xCoord = std::min(lower.xCoord, vertex.xCoord);
      lower.yCoord = std::min(lower.yCoord, vertex.yCoord);
      lower.zCoord = std::min(lower.zCoord, vertex.zCoord);
      upper.xCoord = std::max(upper.xCoord, vertex.xCoord);
      upper.yCoord = std::max(upper.yCoord, vertex.yCoord);
      upper.zCoord = std::max(upper.zCoord, vertex.zCoord);
    }
  }

  bool containsPointOnSurface(const Vec3&) const override { return false; }

  void appendIntersections(const Vec3&, const Vec3&, double, double, std::vector<RayHit>&) const override {}

  double distanceSqToPatch(const Vec3& point) const override
  {
    double bestDistanceSq = std::numeric_limits<double>::infinity();
    for (int vertexIndex = 0; vertexIndex < 3; ++vertexIndex) {
      bestDistanceSq = std::min(bestDistanceSq, pointSegmentDistanceSq(point, mVertices[vertexIndex],
                                                                       mVertices[(vertexIndex + 1) % 3]));
    }
    return bestDistanceSq;
  }

  Vec3 normalAt(const Vec3&) const override { return mNormal; }

  /// A triangle carries no parametric domain, so the form is the identity.
  void parametricMetric(const Vec2&, double& gUU, double& gUV, double& gVV) const override
  {
    gUU = 1.;
    gUV = 0.;
    gVV = 1.;
  }

  double capacityContribution() const override { return 0.; }

  bool capacityIsExact() const override { return false; }

  void appendDisplayMesh(std::vector<Vec3>& vertices, std::vector<std::array<int, 3>>& triangles) const override
  {
    const int firstVertexIndex = static_cast<int>(vertices.size());
    for (const auto& vertex : mVertices) {
      vertices.push_back(vertex);
    }
    triangles.push_back({firstVertexIndex, firstVertexIndex + 1, firstVertexIndex + 2});
  }

  void appendDirectedEdges(std::vector<std::pair<Vec3, Vec3>>& edges) const override
  {
    for (int vertexIndex = 0; vertexIndex < 3; ++vertexIndex) {
      edges.emplace_back(mVertices[vertexIndex], mVertices[(vertexIndex + 1) % 3]);
    }
  }

 private:
  std::array<Vec3, 3> mVertices;
  Vec3 mNormal;
};
} // namespace o2::cad::surface

BOOST_AUTO_TEST_CASE(DummyBoundedSurfaceInterface)
{
  auto dummy = std::make_unique<surf::DummyBoundedSurface>(surf::Vec3{0., 0., 0.}, surf::Vec3{1., 0., 0.},
                                                           surf::Vec3{0., 1., 0.});

  surf::Vec3 lower{surf::Vec3{1.e30, 1.e30, 1.e30}};
  surf::Vec3 upper{surf::Vec3{-1.e30, -1.e30, -1.e30}};
  dummy->conservativeBounds(lower, upper);
  checkClose(lower.xCoord, 0.);
  checkClose(upper.xCoord, 1.);
  checkClose(upper.yCoord, 1.);

  const surf::Vec3 normal = dummy->normalAt({0., 0., 0.});
  checkClose(std::abs(normal.zCoord), 1.);
  BOOST_CHECK(!dummy->capacityIsExact());

  std::vector<surf::Vec3> vertices;
  std::vector<std::array<int, 3>> triangles;
  dummy->appendDisplayMesh(vertices, triangles);
  BOOST_CHECK_EQUAL(vertices.size(), 3u);
  BOOST_CHECK_EQUAL(triangles.size(), 1u);

  // a single open triangle is not a closed manifold
  std::vector<std::unique_ptr<surf::BoundedSurface>> surfaces;
  surfaces.emplace_back(std::move(dummy));
  const surf::ClosureReport report = surf::validateClosure(surfaces);
  BOOST_CHECK(!report.closed);
  BOOST_CHECK_EQUAL(report.boundaryEdges, 3);
}

BOOST_AUTO_TEST_CASE(SolidClosureDetectsMissingAndReversedFaces)
{
  constexpr double halfX = 1.;
  constexpr double halfY = 1.5;
  constexpr double halfZ = 2.;

  // missing face: only five of the six box faces are added
  SurfaceSolid missing("missingFaceBox");
  for (int faceIndex = 1; faceIndex < 6; ++faceIndex) {
    BOOST_REQUIRE(addBoxFace(missing, faceIndex, halfX, halfY, halfZ));
  }
  missing.CloseShape(false);
  BOOST_CHECK(!missing.IsClosed());

  // reversed face: the +x face keeps its geometry but has an inward normal
  SurfaceSolid reversed("reversedFaceBox");
  BOOST_REQUIRE(addBoxFace(reversed, 0, halfX, halfY, halfZ, true));
  for (int faceIndex = 1; faceIndex < 6; ++faceIndex) {
    BOOST_REQUIRE(addBoxFace(reversed, faceIndex, halfX, halfY, halfZ));
  }
  reversed.CloseShape(false);
  BOOST_CHECK(reversed.IsClosed());
  BOOST_CHECK(!reversed.IsOrientationConsistent());
}

// The queryable navigation-reliability state.
// A caller must be able to ask "can I trust this solid's navigation answers" and get a single
// answer, rather than having to notice a printed warning; the state must also survive being
// closed with check==false, since diagnostics and reporting are separate concerns.
BOOST_AUTO_TEST_CASE(NavigationReliabilityIsQueryable)
{
  using Reliability = SurfaceSolid::NavigationReliability;
  constexpr double halfX = 1.;
  constexpr double halfY = 1.5;
  constexpr double halfZ = 2.;

  // before CloseShape there are no diagnostics at all
  SurfaceSolid fresh("freshBox");
  for (int faceIndex = 0; faceIndex < 6; ++faceIndex) {
    BOOST_REQUIRE(addBoxFace(fresh, faceIndex, halfX, halfY, halfZ));
  }
  BOOST_CHECK(fresh.GetNavigationReliability() == Reliability::Undetermined);
  BOOST_CHECK(!fresh.IsNavigable());

  fresh.CloseShape(false);
  BOOST_CHECK(fresh.GetNavigationReliability() == Reliability::Reliable);
  BOOST_CHECK(fresh.IsNavigable());
  BOOST_CHECK_EQUAL(fresh.GetBoundaryEdgeCount(), 0);
  BOOST_CHECK_EQUAL(fresh.GetNonManifoldEdgeCount(), 0);
  BOOST_CHECK_EQUAL(fresh.GetReversedEdgeCount(), 0);

  // a missing face leaves boundary edges: the gap case that motivates the whole state
  SurfaceSolid missing("missingFaceBoxState");
  for (int faceIndex = 1; faceIndex < 6; ++faceIndex) {
    BOOST_REQUIRE(addBoxFace(missing, faceIndex, halfX, halfY, halfZ));
  }
  missing.CloseShape(false);
  BOOST_CHECK(missing.GetNavigationReliability() == Reliability::OpenSurfaceSet);
  BOOST_CHECK(!missing.IsNavigable());
  BOOST_CHECK(missing.GetBoundaryEdgeCount() > 0);

  // a reversed face is closed but inconsistently oriented
  SurfaceSolid reversed("reversedFaceBoxState");
  BOOST_REQUIRE(addBoxFace(reversed, 0, halfX, halfY, halfZ, true));
  for (int faceIndex = 1; faceIndex < 6; ++faceIndex) {
    BOOST_REQUIRE(addBoxFace(reversed, faceIndex, halfX, halfY, halfZ));
  }
  reversed.CloseShape(false);
  BOOST_CHECK(reversed.GetNavigationReliability() == Reliability::ReversedFaces);
  BOOST_CHECK(!reversed.IsNavigable());
  BOOST_CHECK(reversed.GetReversedEdgeCount() > 0);

  // duplicated faces: every edge is now shared by four faces. Non-manifold outranks the boundary
  // and orientation cases because parity is not even order-independent on such input.
  SurfaceSolid duplicated("duplicatedFaceBox");
  for (int pass = 0; pass < 2; ++pass) {
    for (int faceIndex = 0; faceIndex < 6; ++faceIndex) {
      BOOST_REQUIRE(addBoxFace(duplicated, faceIndex, halfX, halfY, halfZ));
    }
  }
  duplicated.CloseShape(false);
  BOOST_CHECK(duplicated.GetNavigationReliability() == Reliability::NonManifold);
  BOOST_CHECK(!duplicated.IsNavigable());
  BOOST_CHECK(duplicated.GetNonManifoldEdgeCount() > 0);

  BOOST_CHECK_EQUAL(std::string(SurfaceSolid::GetNavigationReliabilityName(Reliability::Reliable)), "reliable");
  BOOST_CHECK_EQUAL(std::string(SurfaceSolid::GetNavigationReliabilityName(Reliability::OpenSurfaceSet)),
                    "open-surface-set");
  BOOST_CHECK_EQUAL(std::string(SurfaceSolid::GetNavigationReliabilityName(Reliability::NonManifold)), "non-manifold");
}

BOOST_AUTO_TEST_CASE(NumericalConventions)
{
  using surf::WireClassification;
  using surf::WireRole;
  using surf::WireStatus;

  // near-boundary point classification: a unit square wire, points offset from the bottom edge.
  WireStatus status = WireStatus::Valid;
  auto square = makeWire({{0., 0.}, {1., 0.}, {1., 1.}, {0., 1.}}, WireRole::Outer, status);
  BOOST_REQUIRE(status == WireStatus::Valid);

  // within tolerance of an edge -> Boundary, on both sides
  BOOST_CHECK(square.classify({0.5, 0.5 * surf::kTolerance}) == WireClassification::Boundary);
  BOOST_CHECK(square.classify({0.5, -0.5 * surf::kTolerance}) == WireClassification::Boundary);
  // clearly beyond tolerance -> Inside / Outside
  BOOST_CHECK(square.classify({0.5, 1.e3 * surf::kTolerance}) == WireClassification::Inside);
  BOOST_CHECK(square.classify({0.5, -1.e3 * surf::kTolerance}) == WireClassification::Outside);

  // near-tangent rays against a planar surface in the z = 0 plane.
  surf::PlanarBoundedSurface plane;
  std::string planeError;
  BOOST_REQUIRE(plane.initialize({0., 0., 0.}, {1., 0., 0.}, {0., 1., 0.},
                                 {{0., 0.}, {1., 0.}, {1., 1.}, {0., 1.}}, {}, planeError));

  const surf::Vec3 origin{0.5, 0.5, 1.};
  std::vector<surf::RayHit> hits;

  // direction almost parallel to the plane (tiny z component) -> grazing miss
  const surf::Vec3 grazing = surf::normalized({1., 0., 0.1 * surf::kTolerance});
  plane.appendIntersections(origin, grazing, 0., 1.e30, hits);
  BOOST_CHECK(hits.empty());

  // steeper direction -> real intersection at the plane
  const surf::Vec3 steep = surf::normalized({0., 0., -1.});
  plane.appendIntersections(origin, steep, 0., 1.e30, hits);
  BOOST_REQUIRE_EQUAL(hits.size(), 1u);
  checkClose(hits[0].distance, 1.);

  // a hit rejected when it falls below the minimum ray parameter
  hits.clear();
  plane.appendIntersections(origin, steep, 2., 1.e30, hits);
  BOOST_CHECK(hits.empty());

  // duplicate-intersection clustering respects kIntersectionTolerance.
  BOOST_CHECK(surf::sameIntersection(1., 1. + 0.1 * surf::kIntersectionTolerance));
  BOOST_CHECK(!surf::sameIntersection(1., 1. + 1.e3 * surf::kIntersectionTolerance));
}

BOOST_AUTO_TEST_CASE(WireDataModel)
{
  using surf::WireClassification;
  using surf::WireRole;
  using surf::WireStatus;

  // outer square wire: area, orientation, parametric AABB, and boundary sampling.
  WireStatus status = WireStatus::Valid;
  auto square = makeWire({{0., 0.}, {2., 0.}, {2., 3.}, {0., 3.}}, WireRole::Outer, status);
  BOOST_REQUIRE(status == WireStatus::Valid);
  checkClose(square.signedArea(), 6.);

  surf::Vec2 lower{1.e30, 1.e30};
  surf::Vec2 upper{-1.e30, -1.e30};
  square.parametricBounds(lower, upper);
  checkClose(lower.uCoord, 0.);
  checkClose(lower.vCoord, 0.);
  checkClose(upper.uCoord, 2.);
  checkClose(upper.vCoord, 3.);

  // the sampled boundary is the closed vertex ring (first vertex repeated at the end).
  const auto samples = square.sampledBoundary();
  BOOST_CHECK_EQUAL(samples.size(), square.vertices.size() + 1);
  checkClose(samples.front().uCoord, samples.back().uCoord);
  checkClose(samples.front().vCoord, samples.back().vCoord);

  // edge distance / projection (closest point) on the bottom edge.
  const surf::SurfaceEdge bottom{{0., 0.}, {2., 0.}};
  double parameter = -1.;
  const surf::Vec2 projected = bottom.closestPoint({1., 5.}, parameter);
  checkClose(projected.uCoord, 1.);
  checkClose(projected.vCoord, 0.);
  checkClose(parameter, 0.5);
  // projection is clamped to the segment endpoints.
  bottom.closestPoint({-5., 1.}, parameter);
  checkClose(parameter, 0.);
  bottom.closestPoint({5., 1.}, parameter);
  checkClose(parameter, 1.);
  checkClose(std::sqrt(bottom.distanceSq({1., 4.})), 4.);

  // reversed wire: same shape, opposite winding sign, identical parametric AABB.
  WireStatus reversedStatus = WireStatus::Valid;
  auto reversed = makeWire({{0., 0.}, {0., 3.}, {2., 3.}, {2., 0.}}, WireRole::Outer, reversedStatus);
  BOOST_CHECK(reversedStatus == WireStatus::Reversed);
  checkClose(reversed.signedArea(), 6.); // normalized back to CCW (positive)
  surf::Vec2 reversedLower{1.e30, 1.e30};
  surf::Vec2 reversedUpper{-1.e30, -1.e30};
  reversed.parametricBounds(reversedLower, reversedUpper);
  checkClose(reversedUpper.uCoord, 2.);
  checkClose(reversedUpper.vCoord, 3.);

  // open wire via an explicit non-closing edge list is rejected.
  surf::SurfaceWire scratch;
  const std::vector<surf::SurfaceEdge> openEdges{{{0., 0.}, {2., 0.}}, {{2., 0.}, {2., 3.}}, {{2., 3.}, {1., 1.}}};
  BOOST_CHECK(!scratch.initializeFromEdges(openEdges, WireRole::Outer, status));
  BOOST_CHECK(status == WireStatus::Open);

  // point-on-edge classification.
  BOOST_CHECK(square.classify({1., 0.}) == WireClassification::Boundary);
  BOOST_CHECK(square.classify({1., 1.5}) == WireClassification::Inside);
  BOOST_CHECK(square.classify({3., 1.5}) == WireClassification::Outside);

  // square-with-hole: a planar surface with one inner (hole) wire.
  surf::PlanarBoundedSurface holedFace;
  std::string faceError;
  const std::vector<surf::Vec2> outer{{0., 0.}, {4., 0.}, {4., 4.}, {0., 4.}};
  const std::vector<std::vector<surf::Vec2>> holes{{{1., 1.}, {3., 1.}, {3., 3.}, {1., 3.}}};
  BOOST_REQUIRE(holedFace.initialize({0., 0., 0.}, {1., 0., 0.}, {0., 1., 0.}, outer, holes, faceError));

  bool boundary = false;
  BOOST_CHECK(holedFace.containsLocal({0.5, 0.5}, &boundary)); // in material, outside the hole
  BOOST_CHECK(!boundary);
  BOOST_CHECK(!holedFace.containsLocal({2., 2.}));           // inside the hole -> not on the patch
  BOOST_CHECK(holedFace.containsLocal({2., 1.}, &boundary)); // on the hole boundary -> on the patch
  BOOST_CHECK(boundary);

  // the trimmed area accounts for the hole (16 - 4).
  checkClose(holedFace.area(), 12.);
}

BOOST_AUTO_TEST_CASE(TrimmedCurveBoundaries)
{
  using surf::Curve2D;
  using surf::CurveWire;
  using surf::WireClassification;
  using surf::WireRole;
  using surf::WireStatus;

  // --- line curve: endpoint, tangent, bounds, projection -------------------------------------
  const Curve2D line = Curve2D::makeLine({0., 0.}, {4., 0.});
  checkClose(line.startPoint().uCoord, 0.);
  checkClose(line.endPoint().uCoord, 4.);
  const surf::Vec2 lineTangent = line.tangentAt(0.5);
  checkClose(lineTangent.uCoord, 1.);
  checkClose(lineTangent.vCoord, 0.);

  double lineParameter = -1.;
  const surf::Vec2 lineProjection = line.closestPoint({1., 5.}, lineParameter);
  checkClose(lineProjection.uCoord, 1.);
  checkClose(lineProjection.vCoord, 0.);
  checkClose(lineParameter, 0.25);
  checkClose(std::sqrt(line.distanceSq({1., 5.})), 5.);

  // --- arc curve: endpoint, tangent, exact bounds, projection --------------------------------
  // quarter circle of radius 2 centred at the origin, from angle 0 to pi/2.
  const Curve2D quarter = Curve2D::makeArc({0., 0.}, 2., 0., surf::kHalfPi);
  checkClose(quarter.startPoint().uCoord, 2.);
  checkClose(quarter.startPoint().vCoord, 0.);
  checkClose(quarter.endPoint().uCoord, 0.);
  checkClose(quarter.endPoint().vCoord, 2.);
  // tangent at the start of a CCW arc points in +v.
  const surf::Vec2 arcTangent = quarter.tangentAt(0.);
  checkClose(arcTangent.uCoord, 0.);
  checkClose(arcTangent.vCoord, 1.);

  // the quarter arc's exact bounding box is [0, 2] x [0, 2] (no cardinal extreme inside).
  surf::Vec2 arcLower{1.e30, 1.e30};
  surf::Vec2 arcUpper{-1.e30, -1.e30};
  quarter.extendBounds(arcLower, arcUpper);
  checkClose(arcLower.uCoord, 0.);
  checkClose(arcLower.vCoord, 0.);
  checkClose(arcUpper.uCoord, 2.);
  checkClose(arcUpper.vCoord, 2.);

  // projection of a far radial point lands on the circle (distance = |d - r|).
  double arcParameter = -1.;
  const surf::Vec2 arcProjection = quarter.closestPoint({5., 5.}, arcParameter);
  checkClose(std::hypot(arcProjection.uCoord, arcProjection.vCoord), 2.);
  checkClose(arcParameter, 0.5);

  // a full circle's exact bounding box spans the whole diameter in both axes.
  const Curve2D circle = Curve2D::makeCircle({1., -1.}, 3.);
  surf::Vec2 circleLower{1.e30, 1.e30};
  surf::Vec2 circleUpper{-1.e30, -1.e30};
  circle.extendBounds(circleLower, circleUpper);
  checkClose(circleLower.uCoord, -2.);
  checkClose(circleUpper.uCoord, 4.);
  checkClose(circleLower.vCoord, -4.);
  checkClose(circleUpper.vCoord, 2.);

  // --- disk: one full-circle outer wire ------------------------------------------------------
  WireStatus status = WireStatus::Valid;
  CurveWire disk;
  BOOST_REQUIRE(disk.initialize({Curve2D::makeCircle({0., 0.}, 2.)}, WireRole::Outer, status));
  BOOST_CHECK(status == WireStatus::Valid);
  // exact area of the disk is pi * r^2.
  checkClose(disk.signedArea(), surf::kPi * 4., 1.e-9);
  BOOST_CHECK(disk.classify({0., 0.}) == WireClassification::Inside);
  BOOST_CHECK(disk.classify({1.5, 0.}) == WireClassification::Inside);
  BOOST_CHECK(disk.classify({3., 0.}) == WireClassification::Outside);
  BOOST_CHECK(disk.classify({0., 3.}) == WireClassification::Outside);
  BOOST_CHECK(disk.classify({2., 0.}) == WireClassification::Boundary);
  BOOST_CHECK(disk.classify({0., -2.}) == WireClassification::Boundary);

  // a clockwise circle used as an outer wire is re-oriented to counter-clockwise.
  WireStatus reversedStatus = WireStatus::Valid;
  CurveWire reversedDisk;
  BOOST_REQUIRE(reversedDisk.initialize({Curve2D::makeCircle({0., 0.}, 2., true)}, WireRole::Outer, reversedStatus));
  BOOST_CHECK(reversedStatus == WireStatus::Reversed);
  checkClose(reversedDisk.signedArea(), surf::kPi * 4., 1.e-9);

  // --- annulus: outer disk (CCW) minus an inner hole wire (CW) --------------------------------
  WireStatus outerStatus = WireStatus::Valid;
  WireStatus holeStatus = WireStatus::Valid;
  CurveWire outerRing;
  CurveWire innerRing;
  BOOST_REQUIRE(outerRing.initialize({Curve2D::makeCircle({0., 0.}, 3.)}, WireRole::Outer, outerStatus));
  BOOST_REQUIRE(innerRing.initialize({Curve2D::makeCircle({0., 0.}, 1.)}, WireRole::Inner, holeStatus));
  BOOST_CHECK(holeStatus == WireStatus::Reversed); // CCW circle normalized to CW for a hole
  BOOST_CHECK_LT(innerRing.signedArea(), 0.);
  // net annulus area = pi * (R^2 - r^2).
  checkClose(outerRing.signedArea() + innerRing.signedArea(), surf::kPi * (9. - 1.), 1.e-9);

  // a point in the material (between radii) is inside the outer ring and outside the inner hole.
  const surf::Vec2 materialPoint{2., 0.};
  BOOST_CHECK(outerRing.classify(materialPoint) == WireClassification::Inside);
  BOOST_CHECK(innerRing.classify(materialPoint) == WireClassification::Outside);
  // a point inside the hole is inside both rings (so subtracted from the material).
  const surf::Vec2 holePoint{0.2, 0.};
  BOOST_CHECK(outerRing.classify(holePoint) == WireClassification::Inside);
  BOOST_CHECK(innerRing.classify(holePoint) == WireClassification::Inside);

  // --- mixed line + arc loop: a stadium / half-disk closed by a diameter ---------------------
  // upper half-disk: diameter along v = 0 from (-2,0) to (2,0), closed by a CCW semicircle.
  WireStatus halfStatus = WireStatus::Valid;
  CurveWire halfDisk;
  const std::vector<Curve2D> halfDiskCurves{Curve2D::makeLine({-2., 0.}, {2., 0.}),
                                            Curve2D::makeArc({0., 0.}, 2., 0., surf::kPi)};
  BOOST_REQUIRE(halfDisk.initialize(halfDiskCurves, WireRole::Outer, halfStatus));
  BOOST_CHECK(halfStatus == WireStatus::Valid);
  checkClose(halfDisk.signedArea(), 0.5 * surf::kPi * 4., 1.e-9); // half of pi*r^2
  BOOST_CHECK(halfDisk.classify({0., 1.}) == WireClassification::Inside);
  BOOST_CHECK(halfDisk.classify({0., -1.}) == WireClassification::Outside);
  BOOST_CHECK(halfDisk.classify({0., 0.}) == WireClassification::Boundary);

  // an open curve loop is rejected.
  WireStatus openStatus = WireStatus::Valid;
  CurveWire openWire;
  const std::vector<Curve2D> openCurves{Curve2D::makeLine({0., 0.}, {2., 0.}),
                                        Curve2D::makeLine({2., 0.}, {2., 2.})};
  BOOST_CHECK(!openWire.initialize(openCurves, WireRole::Outer, openStatus));
  BOOST_CHECK(openStatus == WireStatus::Open);
}

BOOST_AUTO_TEST_CASE(BSplineTrimCurveKernels)
{
  using surf::Curve2D;
  using surf::CurveWire;
  using surf::Vec2;
  using surf::WireClassification;
  using surf::WireRole;
  using surf::WireStatus;

  // --- non-rational cubic B-spline: validity, clamped endpoints, convex-hull bounds ------------
  const std::vector<Vec2> poles{{0., 0.}, {1., 2.}, {2., -1.}, {3., 1.}, {4., 0.}};
  const std::vector<double> knots{0., 0., 0., 0., 0.5, 1., 1., 1., 1.};
  const Curve2D spline = Curve2D::makeBSpline(3, poles, {}, knots);
  BOOST_CHECK(spline.valid());
  checkClose(spline.startPoint().uCoord, 0.);
  checkClose(spline.startPoint().vCoord, 0.);
  checkClose(spline.endPoint().uCoord, 4.);
  checkClose(spline.endPoint().vCoord, 0.);
  // extendBounds returns the (conservative) control-point convex-hull box
  Vec2 lower{1.e30, 1.e30};
  Vec2 upper{-1.e30, -1.e30};
  spline.extendBounds(lower, upper);
  checkClose(lower.uCoord, 0.);
  checkClose(upper.uCoord, 4.);
  checkClose(lower.vCoord, -1.);
  checkClose(upper.vCoord, 2.);

  // --- rational quadratic B-spline: an exact NURBS quarter circle -----------------------------
  const std::vector<Vec2> circlePoles{{1., 0.}, {1., 1.}, {0., 1.}};
  const std::vector<double> circleWeights{1., std::sqrt(0.5), 1.};
  const std::vector<double> circleKnots{0., 0., 0., 1., 1., 1.};
  const Curve2D quarter = Curve2D::makeBSpline(2, circlePoles, circleWeights, circleKnots);
  BOOST_CHECK(quarter.valid());
  for (int index = 0; index <= 8; ++index) {
    const Vec2 point = quarter.pointAt(static_cast<double>(index) / 8);
    checkClose(std::hypot(point.uCoord, point.vCoord), 1., 1.e-9);
  }

  // --- closed loop (B-spline top + three closing lines): area vs a fine-polygon reference ------
  const std::vector<Curve2D> loop{spline, Curve2D::makeLine({4., 0.}, {4., -3.}),
                                  Curve2D::makeLine({4., -3.}, {0., -3.}), Curve2D::makeLine({0., -3.}, {0., 0.})};
  WireStatus status = WireStatus::Valid;
  CurveWire wire;
  BOOST_REQUIRE(wire.initialize(loop, WireRole::Outer, status));
  double referenceArea = 0.;
  const auto boundarySamples = wire.sampledBoundary();
  for (size_t k = 0; k + 1 < boundarySamples.size(); ++k) {
    referenceArea += 0.5 * (boundarySamples[k].uCoord * boundarySamples[k + 1].vCoord -
                            boundarySamples[k + 1].uCoord * boundarySamples[k].vCoord);
  }
  // wire.signedArea() is the exact Gauss-Legendre value; referenceArea is a chord-polyline
  // approximation of it, so compare at the sampling-accuracy level rather than machine precision
  checkClose(wire.signedArea(), std::abs(referenceArea), 1.e-4);

  // classify inside / outside / boundary
  BOOST_CHECK(wire.classify({2., -1.5}) == WireClassification::Inside);
  BOOST_CHECK(wire.classify({2., -2.9}) == WireClassification::Inside);
  BOOST_CHECK(wire.classify({-1., -1.}) == WireClassification::Outside);
  BOOST_CHECK(wire.classify({2., 5.}) == WireClassification::Outside);
  BOOST_CHECK(wire.classify({0., 0.}) == WireClassification::Boundary);  // on the B-spline start
  BOOST_CHECK(wire.classify({2., -3.}) == WireClassification::Boundary); // on the bottom line

  // --- horizontal-tangent case: a scanline tangent to a smooth apex must not flip parity -------
  // downward arch: quadratic B-spline (0,0) -> apex (1,1) -> (2,0), closed by the baseline.
  const Curve2D arch = Curve2D::makeBSpline(2, {{0., 0.}, {1., 2.}, {2., 0.}}, {}, {0., 0., 0., 1., 1., 1.});
  checkClose(arch.pointAt(0.5).vCoord, 1.); // apex height
  WireStatus archStatus = WireStatus::Valid;
  CurveWire archRegion;
  BOOST_REQUIRE(archRegion.initialize({arch, Curve2D::makeLine({2., 0.}, {0., 0.})}, WireRole::Outer, archStatus));
  BOOST_CHECK(archRegion.classify({1., 0.5}) == WireClassification::Inside);
  BOOST_CHECK(archRegion.classify({1., 1.5}) == WireClassification::Outside);
  // scanline v = 1 is tangent to the apex to the right of these points: a robust kernel counts an
  // even number of crossings so both points classify Outside.
  BOOST_CHECK(archRegion.classify({-1., 1.}) == WireClassification::Outside);
  BOOST_CHECK(archRegion.classify({3., 1.}) == WireClassification::Outside);

  // --- reversal keeps the same geometric image (poles/knots complemented) ----------------------
  Curve2D reversed = spline;
  reversed.reverseInPlace();
  checkClose(reversed.startPoint().uCoord, 4.);
  checkClose(reversed.endPoint().uCoord, 0.);
  checkClose(reversed.pointAt(0.25).uCoord, spline.pointAt(0.75).uCoord, 1.e-9);
  checkClose(reversed.pointAt(0.25).vCoord, spline.pointAt(0.75).vCoord, 1.e-9);
}

// The adaptive sampler must not be fooled by a curve that meets its own chord where it is probed.
// Both halves of the criterion get their own case, because
// each defeats the other's reproducer on its own.
BOOST_AUTO_TEST_CASE(BSplineSamplingIsNotFooledBySymmetry)
{
  using surf::Curve2D;
  using surf::Vec2;

  // --- symmetry about the parameter midpoint, within a single Bezier span ----------------------
  // A cubic Bezier is (P0 + 3 P1 + 3 P2 + P3) / 8 at t = 1/2, so this S-curve passes through
  // (1, 0) -- exactly on its own chord from (0, 0) to (2, 0) -- while bulging by about 0.3 either
  // side of it. A single midpoint probe therefore calls it flat at the very first step and
  // replaces the whole curve with a straight line. That is what happened to the tube-tube junction
  // curve of six ExcavatorArm parts, whose rim vanished entirely as a result.
  const Curve2D sCurve =
    Curve2D::makeBSpline(3, {{0., 0.}, {0.5, 1.}, {1.5, -1.}, {2., 0.}}, {}, {0., 0., 0., 0., 1., 1., 1., 1.});
  BOOST_REQUIRE(sCurve.valid());
  checkClose(sCurve.pointAt(0.5).uCoord, 1., 1.e-12);
  checkClose(sCurve.pointAt(0.5).vCoord, 0., 1.e-12); // the trap: the midpoint is on the chord
  double worstOffChord = 0.;
  for (int step = 0; step <= 64; ++step) {
    worstOffChord = std::max(worstOffChord, std::abs(sCurve.pointAt(static_cast<double>(step) / 64).vCoord));
  }
  BOOST_CHECK(worstOffChord > 0.2); // and the curve really does leave it, by a lot

  std::vector<Vec2> samples;
  sCurve.bsplineSampleInto(samples);
  BOOST_CHECK(samples.size() > 2); // not flattened to its chord
  double worstSampleError = 0.;
  for (int step = 0; step <= 64; ++step) {
    const Vec2 onCurve = sCurve.pointAt(static_cast<double>(step) / 64);
    double nearest = 1.e30;
    for (size_t index = 0; index + 1 < samples.size(); ++index) {
      nearest = std::min(nearest, surf::pointSegmentDistanceSq(onCurve, samples[index], samples[index + 1]));
    }
    worstSampleError = std::max(worstSampleError, std::sqrt(nearest));
  }
  BOOST_CHECK(worstSampleError < 1.e-4); // and the polyline now follows it

  // --- every knot span gets sampled, however flat the curve looks ------------------------------
  // A B-spline is one polynomial piece only *within* a span, so a flatness verdict that straddles
  // a knot is a verdict about a curve the test's own model does not describe. This one is exactly
  // straight, so no probe anywhere can distinguish it from its chord -- and it must still be
  // resolved span by span, because that is the only thing the curve's own structure guarantees.
  std::vector<Vec2> straightPoles;
  std::vector<double> uniformKnots{0., 0., 0., 0.};
  constexpr int spanCount = 8;
  for (int index = 0; index < spanCount + 3; ++index) {
    straightPoles.push_back({static_cast<double>(index), 0.});
  }
  for (int index = 1; index < spanCount; ++index) {
    uniformKnots.push_back(static_cast<double>(index) / spanCount);
  }
  uniformKnots.insert(uniformKnots.end(), {1., 1., 1., 1.});
  const Curve2D straight = Curve2D::makeBSpline(3, straightPoles, {}, uniformKnots);
  BOOST_REQUIRE(straight.valid());
  std::vector<Vec2> straightSamples;
  straight.bsplineSampleInto(straightSamples);
  BOOST_CHECK(static_cast<int>(straightSamples.size()) >= spanCount + 1);
}

BOOST_AUTO_TEST_CASE(CurvedPlanarDiskKernels)
{
  using surf::Curve2D;

  // annulus in the z = 0 plane: outer radius 2, hole radius 1
  surf::CurvedPlanarBoundedSurface annulus;
  std::string error;
  BOOST_REQUIRE(annulus.initialize({0., 0., 0.}, {1., 0., 0.}, {0., 1., 0.},
                                   {Curve2D::makeCircle({0., 0.}, 2.)},
                                   {{Curve2D::makeCircle({0., 0.}, 1., true)}}, error));
  BOOST_CHECK(!annulus.wasReoriented()); // outer CCW, hole CW: both already correctly oriented

  // a skewed (non-orthonormal) frame is rejected
  surf::CurvedPlanarBoundedSurface skewed;
  BOOST_CHECK(!skewed.initialize({0., 0., 0.}, {1., 0., 0.}, {0.5, 1., 0.},
                                 {Curve2D::makeCircle({0., 0.}, 2.)}, {}, error));

  // on-surface classification: material, hole, outside, off-plane
  BOOST_CHECK(annulus.containsPointOnSurface({1.5, 0., 0.}));
  BOOST_CHECK(!annulus.containsPointOnSurface({0.5, 0., 0.}));
  BOOST_CHECK(!annulus.containsPointOnSurface({3., 0., 0.}));
  BOOST_CHECK(!annulus.containsPointOnSurface({1.5, 0., 0.5}));

  // ray intersections: one hit through the material, none through the hole
  std::vector<surf::RayHit> hits;
  annulus.appendIntersections({1.5, 0., 1.}, {0., 0., -1.}, 0., 1.e30, hits);
  BOOST_REQUIRE_EQUAL(hits.size(), 1u);
  checkClose(hits.front().distance, 1.);
  checkClose(hits.front().normal.zCoord, 1.);
  hits.clear();
  annulus.appendIntersections({0.5, 0., 1.}, {0., 0., -1.}, 0., 1.e30, hits);
  BOOST_CHECK(hits.empty());

  // exact patch distances: in-hole (to the hole rim), in-material off-plane, outside the rim,
  // and the combined in-plane plus out-of-plane case
  checkClose(annulus.distanceSqToPatch({0., 0., 0.}), 1.);
  checkClose(annulus.distanceSqToPatch({1.5, 0., 2.}), 4.);
  checkClose(annulus.distanceSqToPatch({4., 0., 0.}), 4.);
  checkClose(annulus.distanceSqToPatch({0.5, 0., 1.}), 1.25);

  // divergence-theorem contribution of an offset disk: (origin . normal) * area / 3
  surf::CurvedPlanarBoundedSurface offsetDisk;
  BOOST_REQUIRE(offsetDisk.initialize({0., 0., 2.}, {1., 0., 0.}, {0., 1., 0.},
                                      {Curve2D::makeCircle({0., 0.}, 1.5)}, {}, error));
  checkClose(offsetDisk.capacityContribution(), 2. * surf::kPi * 1.5 * 1.5 / 3., 1.e-9);
  BOOST_CHECK(offsetDisk.capacityIsExact());
}

BOOST_AUTO_TEST_CASE(CylindricalSurfaceKernels)
{
  // full lateral cylinder, radius 2, height [-3, 3], axis z
  surf::CylindricalBoundedSurface cylinder;
  std::string error;
  BOOST_REQUIRE(cylinder.initialize({0., 0., 0.}, {0., 0., 1.}, {1., 0., 0.}, 2., -3., 3., 0., surf::kTwoPi,
                                    false, error));

  // a transversal ray crosses the lateral surface twice: both hits must be reported
  std::vector<surf::RayHit> hits;
  cylinder.appendIntersections({-5., 0.5, 0.}, {1., 0., 0.}, 0., 1.e30, hits);
  BOOST_REQUIRE_EQUAL(hits.size(), 2u);
  const double chordHalf = std::sqrt(4. - 0.25);
  checkClose(hits[0].distance, 5. - chordHalf);
  checkClose(hits[1].distance, 5. + chordHalf);
  // entering hit: outward normal opposes the ray direction; exiting hit: aligned
  BOOST_CHECK_LT(hits[0].normal.xCoord, 0.);
  BOOST_CHECK_GT(hits[1].normal.xCoord, 0.);

  // tangential graze reports no hits (keeps crossing parity even)
  hits.clear();
  cylinder.appendIntersections({-5., 2., 0.}, {1., 0., 0.}, 0., 1.e30, hits);
  BOOST_CHECK(hits.empty());

  // axis-parallel ray never crosses the lateral surface
  hits.clear();
  cylinder.appendIntersections({0., 0., -5.}, {0., 0., 1.}, 0., 1.e30, hits);
  BOOST_CHECK(hits.empty());

  // exact patch distances: radial, above the rim, and the diagonal rim case
  checkClose(cylinder.distanceSqToPatch({4., 0., 0.}), 4.);
  checkClose(cylinder.distanceSqToPatch({0., 0., 5.}), 8.);
  checkClose(cylinder.distanceSqToPatch({3., 0., 4.}), 2.);

  // half cylinder (phi in [0, pi]): the phi trim filters hits and surface points
  surf::CylindricalBoundedSurface halfCylinder;
  BOOST_REQUIRE(halfCylinder.initialize({0., 0., 0.}, {0., 0., 1.}, {1., 0., 0.}, 2., -3., 3., 0., surf::kPi,
                                        false, error));
  BOOST_CHECK(halfCylinder.containsPointOnSurface({0., 2., 0.}));
  BOOST_CHECK(!halfCylinder.containsPointOnSurface({0., -2., 0.}));
  hits.clear();
  halfCylinder.appendIntersections({0., -5., 0.}, {0., 1., 0.}, 0., 1.e30, hits);
  BOOST_REQUIRE_EQUAL(hits.size(), 1u);
  checkClose(hits.front().distance, 7.);
}

BOOST_AUTO_TEST_CASE(ClosedCylinderMatchesTGeoTube)
{
  constexpr double radius = 2.;
  constexpr double halfHeight = 3.;

  SurfaceSolid solid("closedCylinder");
  BOOST_REQUIRE(solid.AddCylindricalSurface({0., 0., 0.}, {0., 0., 1.}, {1., 0., 0.}, radius, -halfHeight,
                                            halfHeight));
  // cap frames: outward normal is axisU x axisV, so the bottom cap flips axisV
  BOOST_REQUIRE(addDiskSurface(solid, {0., 0., halfHeight}, {1., 0., 0.}, {0., 1., 0.}, radius));
  BOOST_REQUIRE(addDiskSurface(solid, {0., 0., -halfHeight}, {1., 0., 0.}, {0., -1., 0.}, radius));
  solid.CloseShape();

  BOOST_CHECK(solid.IsClosed());
  BOOST_CHECK(solid.IsOrientationConsistent());

  TGeoTube reference("referenceTube", 0., radius, halfHeight);
  compareContainsGrid(solid, reference, 4., 9);

  compareDistance(solid, reference, {5., 0., 0.}, {-1., 0., 0.});
  compareDistance(solid, reference, {0., 0., 5.}, {0., 0., -1.});
  compareDistance(solid, reference, {5., 0.5, 1.}, {-1., 0., 0.});
  compareDistance(solid, reference, {-4., -1., -2.}, unitDirection(1., 0.3, 0.5));
  compareDistance(solid, reference, {0., 0., 0.}, {1., 0., 0.});
  compareDistance(solid, reference, {0., 0., 0.}, {0., 0., 1.});
  compareDistance(solid, reference, {0., 0., 0.}, unitDirection(1., 1., 1.));
  compareDistance(solid, reference, {1., 0.5, -2.}, unitDirection(0.3, -0.4, 0.5));
  compareDistance(solid, reference, {5., 2.5, 0.}, {-1., 0., 0.}); // grazing miss

  // safeties against analytic distances (TGeo safeties may be weaker underestimates, so they
  // are not compared directly)
  const double center[3] = {0., 0., 0.};
  const double insidePoint[3] = {1., 0.5, 1.};
  const double radialOutside[3] = {4., 0., 0.};
  const double axialOutside[3] = {0., 0., 5.};
  const double cornerOutside[3] = {4., 0., 5.};
  checkClose(solid.Safety(center, kTRUE), radius);
  checkClose(solid.Safety(insidePoint, kTRUE), radius - std::sqrt(1.25));
  checkClose(solid.Safety(radialOutside, kFALSE), 2.);
  checkClose(solid.Safety(axialOutside, kFALSE), 2.);
  checkClose(solid.Safety(cornerOutside, kFALSE), std::sqrt(8.)); // exact corner distance

  // normals on the lateral surface and the caps
  double normal[3] = {0., 0., 0.};
  const double sidePoint[3] = {radius, 0., 1.};
  const double alongX[3] = {1., 0., 0.};
  solid.ComputeNormal(sidePoint, alongX, normal);
  checkClose(normal[0], 1.);
  checkClose(normal[1], 0.);
  checkClose(normal[2], 0.);
  const double capPoint[3] = {0.5, 0.5, halfHeight};
  const double alongZ[3] = {0., 0., 1.};
  solid.ComputeNormal(capPoint, alongZ, normal);
  checkClose(normal[2], 1.);

  checkClose(solid.Capacity(), reference.Capacity(), 1.e-9);

  int meshVertices = 0;
  int meshSegments = 0;
  int meshPolygons = 0;
  solid.GetMeshNumbers(meshVertices, meshSegments, meshPolygons);
  BOOST_CHECK_GT(meshVertices, 0);
  BOOST_CHECK_GT(meshPolygons, 0);
}

BOOST_AUTO_TEST_CASE(HollowCylinderMatchesTGeoTube)
{
  constexpr double innerRadius = 1.;
  constexpr double outerRadius = 2.;
  constexpr double halfHeight = 3.;

  SurfaceSolid solid("hollowCylinder");
  BOOST_REQUIRE(solid.AddCylindricalSurface({0., 0., 0.}, {0., 0., 1.}, {1., 0., 0.}, outerRadius, -halfHeight,
                                            halfHeight));
  BOOST_REQUIRE(solid.AddCylindricalSurface({0., 0., 0.}, {0., 0., 1.}, {1., 0., 0.}, innerRadius, -halfHeight,
                                            halfHeight, 0., surf::kTwoPi, true));
  BOOST_REQUIRE(addDiskSurface(solid, {0., 0., halfHeight}, {1., 0., 0.}, {0., 1., 0.}, outerRadius,
                               innerRadius));
  BOOST_REQUIRE(addDiskSurface(solid, {0., 0., -halfHeight}, {1., 0., 0.}, {0., -1., 0.}, outerRadius,
                               innerRadius));
  solid.CloseShape();

  BOOST_CHECK(solid.IsClosed());
  BOOST_CHECK(solid.IsOrientationConsistent());

  TGeoTube reference("referenceHollowTube", innerRadius, outerRadius, halfHeight);
  compareContainsGrid(solid, reference, 4., 9);

  // from the hole the solid is entered through the inner wall
  compareDistance(solid, reference, {0., 0., 0.}, {1., 0., 0.});
  compareDistance(solid, reference, {0., 0., 2.}, unitDirection(0.4, 0.2, -1.));
  // inside the material both walls are exit candidates
  compareDistance(solid, reference, {1.5, 0., 0.}, {1., 0., 0.});
  compareDistance(solid, reference, {1.5, 0., 0.}, {-1., 0., 0.});
  compareDistance(solid, reference, {-1.2, 0.8, 1.}, unitDirection(-0.2, 0.9, 0.4));
  compareDistance(solid, reference, {5., 0., 0.}, {-1., 0., 0.});

  // analytic safety in the middle of the material: 0.5 to either wall
  const double materialPoint[3] = {1.5, 0., 0.};
  checkClose(solid.Safety(materialPoint, kTRUE), 0.5);

  checkClose(solid.Capacity(), reference.Capacity(), 1.e-9);
}

BOOST_AUTO_TEST_CASE(SphereMatchesTGeoSphere)
{
  constexpr double radius = 2.5;

  SurfaceSolid solid("fullSphere");
  BOOST_REQUIRE(solid.AddSphericalSurface({0., 0., 0.}, {0., 0., 1.}, {1., 0., 0.}, radius));
  solid.CloseShape();

  // a full sphere is self-closing: no boundary edges at all
  BOOST_CHECK(solid.IsClosed());
  BOOST_CHECK(solid.IsOrientationConsistent());

  TGeoSphere reference("referenceSphere", 0., radius);
  compareContainsGrid(solid, reference, 3.5, 9);

  compareDistance(solid, reference, {5., 0., 0.}, {-1., 0., 0.});
  compareDistance(solid, reference, {0., 0., 0.}, {1., 0., 0.});
  compareDistance(solid, reference, {0., 0., 0.}, unitDirection(1., 1., 1.));
  compareDistance(solid, reference, {1., 1., 1.}, unitDirection(-0.3, 0.5, 0.8));
  compareDistance(solid, reference, {-4., 0.5, 0.5}, {1., 0., 0.});
  compareDistance(solid, reference, {-4., 2.6, 0.}, {1., 0., 0.}); // clean miss

  // analytic safeties: |distance to center - radius|
  const double insidePoint[3] = {1., 0., 0.};
  const double outsidePoint[3] = {4., 0., 0.};
  checkClose(solid.Safety(insidePoint, kTRUE), radius - 1.);
  checkClose(solid.Safety(outsidePoint, kFALSE), 4. - radius);

  double normal[3] = {0., 0., 0.};
  const double surfacePoint[3] = {radius, 0., 0.};
  const double alongX[3] = {1., 0., 0.};
  solid.ComputeNormal(surfacePoint, alongX, normal);
  checkClose(normal[0], 1.);

  checkClose(solid.Capacity(), reference.Capacity(), 1.e-9);

  int meshVertices = 0;
  int meshSegments = 0;
  int meshPolygons = 0;
  solid.GetMeshNumbers(meshVertices, meshSegments, meshPolygons);
  BOOST_CHECK_GT(meshVertices, 0);
  BOOST_CHECK_GT(meshPolygons, 0);
}

BOOST_AUTO_TEST_CASE(SphericalSectionKernels)
{
  // upper hemisphere shell of radius 2 (theta in [0, pi/2], full phi)
  surf::SphericalBoundedSurface hemisphere;
  std::string error;
  BOOST_REQUIRE(hemisphere.initialize({0., 0., 0.}, {0., 0., 1.}, {1., 0., 0.}, 2., 0., surf::kHalfPi, 0.,
                                      surf::kTwoPi, false, error));

  // divergence contribution of a centred hemisphere shell: 2 pi R^3 / 3
  checkClose(hemisphere.capacityContribution(), 2. * surf::kPi * 8. / 3., 1.e-9);

  // the polar-axis ray meets the sphere twice but only the upper hit is on the patch
  std::vector<surf::RayHit> hits;
  hemisphere.appendIntersections({0., 0., 5.}, {0., 0., -1.}, 0., 1.e30, hits);
  BOOST_REQUIRE_EQUAL(hits.size(), 1u);
  checkClose(hits.front().distance, 3.);
  checkClose(hits.front().normal.zCoord, 1.);

  // a transversal ray at z = 1 stays in the upper hemisphere: both hits reported
  hits.clear();
  hemisphere.appendIntersections({-5., 0., 1.}, {1., 0., 0.}, 0., 1.e30, hits);
  BOOST_CHECK_EQUAL(hits.size(), 2u);

  // the mirrored ray at z = -1 misses the trimmed patch entirely
  hits.clear();
  hemisphere.appendIntersections({-5., 0., -1.}, {1., 0., 0.}, 0., 1.e30, hits);
  BOOST_CHECK(hits.empty());

  // trim-aware surface point classification (equator lies on the trim boundary)
  BOOST_CHECK(hemisphere.containsPointOnSurface({0., 0., 2.}));
  BOOST_CHECK(hemisphere.containsPointOnSurface({2., 0., 0.}));
  BOOST_CHECK(!hemisphere.containsPointOnSurface({0., 0., -2.}));

  // patch distance: exact radially above the pole, conservative lower bound below the equator
  checkClose(hemisphere.distanceSqToPatch({0., 0., 5.}), 9.);
  BOOST_CHECK_LE(hemisphere.distanceSqToPatch({0., 0., -4.}), 4. + 1.e-9);
}

BOOST_AUTO_TEST_CASE(TruncatedConeMatchesTGeoCone)
{
  constexpr double halfHeight = 3.;
  constexpr double radiusAtBottom = 2.;
  constexpr double radiusAtTop = 1.;

  SurfaceSolid solid("truncatedCone");
  BOOST_REQUIRE(solid.AddConicalSurface({0., 0., 0.}, {0., 0., 1.}, {1., 0., 0.}, radiusAtBottom, radiusAtTop,
                                        -halfHeight, halfHeight));
  BOOST_REQUIRE(addDiskSurface(solid, {0., 0., halfHeight}, {1., 0., 0.}, {0., 1., 0.}, radiusAtTop));
  BOOST_REQUIRE(addDiskSurface(solid, {0., 0., -halfHeight}, {1., 0., 0.}, {0., -1., 0.}, radiusAtBottom));
  solid.CloseShape();

  BOOST_CHECK(solid.IsClosed());
  BOOST_CHECK(solid.IsOrientationConsistent());

  TGeoCone reference("referenceCone", halfHeight, 0., radiusAtBottom, 0., radiusAtTop);
  compareContainsGrid(solid, reference, 3.5, 9);

  compareDistance(solid, reference, {5., 0., 0.}, {-1., 0., 0.});
  compareDistance(solid, reference, {0., 0., 5.}, {0., 0., -1.});
  compareDistance(solid, reference, {0., 0., 0.}, {1., 0., 0.});
  compareDistance(solid, reference, {0., 0., 0.}, {0., 0., 1.});
  compareDistance(solid, reference, {0., 0., 0.}, {0., 0., -1.});
  compareDistance(solid, reference, {0.5, -0.3, 1.}, unitDirection(0.6, 0.4, 0.2));
  compareDistance(solid, reference, {-4., 0.2, -2.}, unitDirection(1., 0.05, 0.3));

  // central safety: exact distance to the lateral generator segment (2,-3)-(1,3) in (rho, z);
  // TGeoCone's safety degenerates to 0 on the axis of an rmin = 0 cone, so no direct comparison
  const double center[3] = {0., 0., 0.};
  checkClose(solid.Safety(center, kTRUE), 9. / std::sqrt(37.));

  // lateral-surface normal against the ROOT cone
  double normal[3] = {0., 0., 0.};
  double referenceNormal[3] = {0., 0., 0.};
  const double sidePoint[3] = {1.5, 0., 0.};
  const double alongX[3] = {1., 0., 0.};
  solid.ComputeNormal(sidePoint, alongX, normal);
  reference.ComputeNormal(sidePoint, alongX, referenceNormal);
  checkClose(normal[0], referenceNormal[0]);
  checkClose(normal[1], referenceNormal[1]);
  checkClose(normal[2], referenceNormal[2]);

  checkClose(solid.Capacity(), reference.Capacity(), 1.e-9);
}

BOOST_AUTO_TEST_CASE(ApexConeClosesWithSingleCap)
{
  // full cone: radius 3 at z = -1.5 shrinking to the apex at z = +1.5, closed by one cap
  constexpr double halfHeight = 1.5;
  constexpr double baseRadius = 3.;

  SurfaceSolid solid("apexCone");
  BOOST_REQUIRE(solid.AddConicalSurface({0., 0., 0.}, {0., 0., 1.}, {1., 0., 0.}, baseRadius, 0., -halfHeight,
                                        halfHeight));
  BOOST_REQUIRE(addDiskSurface(solid, {0., 0., -halfHeight}, {1., 0., 0.}, {0., -1., 0.}, baseRadius));
  solid.CloseShape();

  // the apex rim degenerates to a point, so one cap suffices for a closed manifold
  BOOST_CHECK(solid.IsClosed());
  BOOST_CHECK(solid.IsOrientationConsistent());

  // analytic containment: inside iff |z| < halfHeight and rho < r(z) = halfHeight - z
  const auto analyticInside = [&](double x, double y, double z) {
    return std::abs(z) < halfHeight && std::hypot(x, y) < halfHeight - z;
  };
  const std::array<std::array<double, 3>, 7> probePoints{{{0., 0., 0.},
                                                          {1., 0., 0.},
                                                          {1.4, 0., 0.5},
                                                          {0., 0., 1.4},
                                                          {0., 0., 1.6},
                                                          {2., 2., -1.},
                                                          {2., 0., -1.}}};
  for (const auto& probe : probePoints) {
    BOOST_TEST_CONTEXT("point = (" << probe[0] << ", " << probe[1] << ", " << probe[2] << ")")
    {
      BOOST_CHECK_EQUAL(solid.Contains(probe.data()), analyticInside(probe[0], probe[1], probe[2]));
    }
  }

  // radial exit through the slanted surface
  const double insidePoint[3] = {0., 0., -1.};
  const double alongX[3] = {1., 0., 0.};
  checkClose(solid.DistFromInside(insidePoint, alongX, 3), 2.5);

  // central safety is the exact distance to the slanted line rho + z = halfHeight
  const double center[3] = {0., 0., 0.};
  checkClose(solid.Safety(center, kTRUE), halfHeight / std::sqrt(2.), 1.e-9);

  // exact capacity of a full cone: pi R^2 H / 3
  checkClose(solid.Capacity(), surf::kPi * baseRadius * baseRadius * 2. * halfHeight / 3., 1.e-9);
}

BOOST_AUTO_TEST_CASE(ToroidalSurfaceKernels)
{
  // full torus, major radius 3, minor (tube) radius 1, axis z
  constexpr double majorR = 3.;
  constexpr double minorR = 1.;
  surf::TorusBoundedSurface torus;
  std::string error;
  BOOST_REQUIRE(torus.initialize({0., 0., 0.}, {0., 0., 1.}, {1., 0., 0.}, majorR, minorR, 0., surf::kTwoPi, 0.,
                                 surf::kTwoPi, false, error));

  // a ray along +x through the centre crosses the donut four times: at rho = -(R+r), -(R-r),
  // (R-r), (R+r), i.e. distances 6, 8, 12, 14 from the origin at x = -10
  std::vector<surf::RayHit> hits;
  torus.appendIntersections({-10., 0., 0.}, {1., 0., 0.}, 0., 1.e30, hits);
  BOOST_REQUIRE_EQUAL(hits.size(), 4u);
  std::sort(hits.begin(), hits.end(),
            [](const surf::RayHit& a, const surf::RayHit& b) { return a.distance < b.distance; });
  checkClose(hits[0].distance, 6., 1.e-7);
  checkClose(hits[1].distance, 8., 1.e-7);
  checkClose(hits[2].distance, 12., 1.e-7);
  checkClose(hits[3].distance, 14., 1.e-7);
  // crossings alternate enter/exit/enter/exit along the ray
  BOOST_CHECK_LT(hits[0].normal.xCoord, 0.);
  BOOST_CHECK_GT(hits[1].normal.xCoord, 0.);
  BOOST_CHECK_LT(hits[2].normal.xCoord, 0.);
  BOOST_CHECK_GT(hits[3].normal.xCoord, 0.);

  // a z-ray tangent to the outer equator (rho = R + r) touches at a single double root: no hit
  hits.clear();
  torus.appendIntersections({majorR + minorR, 0., -10.}, {0., 0., 1.}, 0., 1.e30, hits);
  BOOST_CHECK(hits.empty());

  // a ray passing above the whole torus (z = 2 r) misses entirely
  hits.clear();
  torus.appendIntersections({-10., 0., 2. * minorR}, {1., 0., 0.}, 0., 1.e30, hits);
  BOOST_CHECK(hits.empty());

  // outward normals: +x at the outer equator, -x (towards the axis) at the inner equator
  const surf::Vec3 outerNormal = torus.normalAt({majorR + minorR, 0., 0.});
  checkClose(outerNormal.xCoord, 1.);
  const surf::Vec3 innerNormal = torus.normalAt({majorR - minorR, 0., 0.});
  checkClose(innerNormal.xCoord, -1.);
  // top of the tube: normal points along +z
  const surf::Vec3 topNormal = torus.normalAt({majorR, 0., minorR});
  checkClose(topNormal.zCoord, 1.);

  // exact meridian distances: radially outside the outer equator and inside the hole
  checkClose(torus.distanceSqToPatch({majorR + minorR + 2., 0., 0.}), 4.);
  checkClose(torus.distanceSqToPatch({0., 0., 0.}), (majorR - minorR) * (majorR - minorR));

  // surface-point classification
  BOOST_CHECK(torus.containsPointOnSurface({majorR + minorR, 0., 0.}));
  BOOST_CHECK(torus.containsPointOnSurface({majorR, 0., minorR}));
  BOOST_CHECK(!torus.containsPointOnSurface({majorR, 0., 0.}));      // tube spine (interior)
  BOOST_CHECK(!torus.containsPointOnSurface({majorR + 5., 0., 0.})); // off the surface

  // exact divergence-theorem capacity of a full torus: 2 pi^2 R r^2
  checkClose(torus.capacityContribution(), 2. * surf::kPi * surf::kPi * majorR * minorR * minorR, 1.e-9);
  BOOST_CHECK(torus.capacityIsExact());

  // partial tube section (a quarter-tube fillet-like patch, phiTube in [0, pi/2], full ring):
  // the trim filters intersections and surface points
  surf::TorusBoundedSurface quarterTube;
  BOOST_REQUIRE(quarterTube.initialize({0., 0., 0.}, {0., 0., 1.}, {1., 0., 0.}, majorR, minorR, 0., surf::kTwoPi, 0.,
                                       surf::kHalfPi, false, error));
  BOOST_CHECK(quarterTube.containsPointOnSurface({majorR + minorR, 0., 0.})); // phiTube = 0 boundary
  BOOST_CHECK(quarterTube.containsPointOnSurface({majorR, 0., minorR}));      // phiTube = pi/2 boundary
  BOOST_CHECK(!quarterTube.containsPointOnSurface({majorR, 0., -minorR}));    // phiTube = -pi/2, off patch
  hits.clear();
  quarterTube.appendIntersections({majorR, 0., -10.}, {0., 0., 1.}, 0., 1.e30, hits);
  BOOST_REQUIRE_EQUAL(hits.size(), 1u); // only the top (+z) tube point is on the quarter patch
  checkClose(hits.front().distance, 10. + minorR, 1.e-7);
}

BOOST_AUTO_TEST_CASE(FullTorusMatchesTGeoTorus)
{
  constexpr double majorR = 3.;
  constexpr double minorR = 1.;

  SurfaceSolid solid("fullTorus");
  BOOST_REQUIRE(solid.AddToroidalSurface({0., 0., 0.}, {0., 0., 1.}, {1., 0., 0.}, majorR, minorR));
  solid.CloseShape();

  // a full torus is self-closing: no boundary edges
  BOOST_CHECK(solid.IsClosed());
  BOOST_CHECK(solid.IsOrientationConsistent());

  // TGeoTorus(R, Rmin, Rmax): a solid torus has Rmin = 0, Rmax = tube radius
  TGeoTorus reference("referenceTorus", majorR, 0., minorR);
  compareContainsGrid(solid, reference, 4.5, 11);

  // analytic x-axis crossings (see the kernel test): from outside and from inside the material
  const double outsidePoint[3] = {-10., 0., 0.};
  const double alongX[3] = {1., 0., 0.};
  checkClose(solid.DistFromOutside(outsidePoint, alongX, 3), 6., 1.e-7);
  const double materialPoint[3] = {majorR + minorR - 0.25, 0., 0.}; // inside the tube on the +x side
  BOOST_CHECK(solid.Contains(materialPoint));
  checkClose(solid.DistFromInside(materialPoint, alongX, 3), 0.25, 1.e-7);

  // a couple of oblique rays cross-checked against the ROOT torus
  compareDistance(solid, reference, {-10., 0.3, 0.2}, {1., 0., 0.}, 1.e-6);
  compareDistance(solid, reference, {0., 0., 5.}, {0., 0., -1.}, 1.e-6); // clean axial miss (through the hole)

  // exact capacity: 2 pi^2 R r^2
  checkClose(solid.Capacity(), reference.Capacity(), 1.e-7);
  checkClose(solid.Capacity(), 2. * surf::kPi * surf::kPi * majorR * minorR * minorR, 1.e-9);

  int meshVertices = 0;
  int meshSegments = 0;
  int meshPolygons = 0;
  solid.GetMeshNumbers(meshVertices, meshSegments, meshPolygons);
  BOOST_CHECK_GT(meshVertices, 0);
  BOOST_CHECK_GT(meshPolygons, 0);
}

BOOST_AUTO_TEST_CASE(WireTrimmedTorusMatchesSection)
{
  // A partial toroidal patch (ring [0, pi/2], tube [0.2, 2.0] - a non-wrapping fillet-like arc)
  // built two ways must classify points identically: with the scalar parametric rectangle and
  // with an equivalent (phiRing, phiTube) line-wire trim. This exercises the wire-trim path
  // (numeric capacity, conservative Safety) and the periodic-in-both-angles unwrapping.
  constexpr double majorR = 4.;
  constexpr double minorR = 1.5;
  constexpr double tubeLow = 0.2;
  constexpr double tubeHigh = 2.0;
  std::string error;

  surf::TorusBoundedSurface scalarSection;
  BOOST_REQUIRE(scalarSection.initialize({0.2, -0.1, 0.3}, {0., 0., 1.}, {1., 0., 0.}, majorR, minorR, 0.,
                                         surf::kHalfPi, tubeLow, tubeHigh - tubeLow, false, error));

  surf::TorusBoundedSurface wireSection;
  const auto wire = paramRectWireCurves(0., surf::kHalfPi, tubeLow, tubeHigh);
  BOOST_REQUIRE(wireSection.initialize({0.2, -0.1, 0.3}, {0., 0., 1.}, {1., 0., 0.}, majorR, minorR, 0., surf::kHalfPi,
                                       tubeLow, tubeHigh - tubeLow, false, wire, {}, error));
  BOOST_CHECK(wireSection.hasWireTrim());

  // classification agrees across a set of on-surface probes at several ring/tube angles
  for (double ring : {0.1, 0.7, 1.2, 1.7, 2.5}) {
    for (double tube : {0.3, 0.8, 1.5, 1.9, 2.6}) {
      const surf::Vec3 probe = scalarSection.pointAt(ring, tube);
      BOOST_TEST_CONTEXT("ring = " << ring << " tube = " << tube)
      {
        BOOST_CHECK_EQUAL(scalarSection.containsPointOnSurface(probe), wireSection.containsPointOnSurface(probe));
      }
    }
  }

  // wire-trim capacity is numeric (flagged inexact) but must approximate the exact scalar value
  BOOST_CHECK(scalarSection.capacityIsExact());
  BOOST_CHECK(!wireSection.capacityIsExact());
  BOOST_CHECK_SMALL(wireSection.capacityContribution() - scalarSection.capacityContribution(),
                    1.e-2 * std::abs(scalarSection.capacityContribution()));
}

BOOST_AUTO_TEST_CASE(BVHConstructionAndTraversal)
{
  constexpr double halfX = 1.;
  constexpr double halfY = 2.;
  constexpr double halfZ = 3.;

  SurfaceSolid solid("bvhBox");
  addBoxSurfaces(solid, halfX, halfY, halfZ);
  BOOST_CHECK(!solid.HasBVH()); // built only in CloseShape
  solid.CloseShape();
  BOOST_REQUIRE(solid.HasBVH());

  // the BVH root box must enclose the exact solid bounds and stay conservative: not tighter
  // than the exact bounds, not looser than the documented expansion (plus float rounding)
  Point3D lower{};
  Point3D upper{};
  BOOST_REQUIRE(solid.GetBVHRootBounds(lower, upper));
  const Point3D exactLower{-halfX, -halfY, -halfZ};
  const Point3D exactUpper{halfX, halfY, halfZ};
  constexpr double boxSlack = 2. * surf::kBVHBoxTolerance;
  for (int dimension = 0; dimension < 3; ++dimension) {
    BOOST_TEST_CONTEXT("dimension = " << dimension)
    {
      BOOST_CHECK(lower[dimension] <= exactLower[dimension]);
      BOOST_CHECK(lower[dimension] >= exactLower[dimension] - boxSlack);
      BOOST_CHECK(upper[dimension] >= exactUpper[dimension]);
      BOOST_CHECK(upper[dimension] <= exactUpper[dimension] + boxSlack);
    }
  }

  // a ray through the box must traverse (at least) the entry and exit face leaves ...
  BOOST_CHECK_GE(solid.CountBVHRayCandidates({-2., 0., 0.}, {1., 0., 0.}), 2);
  // ... while a ray pointing away from the solid reaches no leaf at all
  BOOST_CHECK_EQUAL(solid.CountBVHRayCandidates({0., 5., 0.}, {0., 1., 0.}), 0);

  // two disjoint boxes: BVH pruning with well-separated primitive clusters. The union of two
  // closed manifolds is still a closed manifold, and parity containment handles it naturally.
  constexpr double half = 1.;
  constexpr double centerX = 3.;
  SurfaceSolid twoBoxes("twoBoxes");
  addBoxSurfaces(twoBoxes, half, half, half, {-centerX, 0., 0.});
  addBoxSurfaces(twoBoxes, half, half, half, {centerX, 0., 0.});
  twoBoxes.CloseShape();
  BOOST_REQUIRE(twoBoxes.HasBVH());
  BOOST_CHECK_EQUAL(twoBoxes.GetNsurfaces(), 12);
  BOOST_CHECK(twoBoxes.IsClosed());
  BOOST_CHECK(twoBoxes.IsOrientationConsistent());

  const auto analyticInside = [&](const double* point) {
    return (std::abs(std::abs(point[0]) - centerX) < half) && std::abs(point[1]) < half && std::abs(point[2]) < half;
  };
  constexpr int samples = 9;
  constexpr double extent = 5.;
  for (int stepX = 0; stepX < samples; ++stepX) {
    for (int stepY = 0; stepY < samples; ++stepY) {
      for (int stepZ = 0; stepZ < samples; ++stepZ) {
        const double point[3] = {-extent + 2. * extent * (stepX + 0.517) / samples,
                                 -extent + 2. * extent * (stepY + 0.263) / samples,
                                 -extent + 2. * extent * (stepZ + 0.741) / samples};
        BOOST_TEST_CONTEXT("point = (" << point[0] << ", " << point[1] << ", " << point[2] << ")")
        {
          const bool bvhInside = twoBoxes.Contains(point);
          BOOST_CHECK_EQUAL(bvhInside, twoBoxes.Contains_Loop(point));
          BOOST_CHECK_EQUAL(bvhInside, analyticInside(point));
        }
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(ContainsBoundaryPointsAndCapsule)
{
  constexpr double halfX = 1.;
  constexpr double halfY = 2.;
  constexpr double halfZ = 3.;

  SurfaceSolid box("boundaryBox");
  addBoxSurfaces(box, halfX, halfY, halfZ);
  box.CloseShape();

  // boundary policy: points exactly on faces, edges and vertices count as inside,
  // in the BVH-accelerated path and in the trivial loop alike
  const std::array<std::array<double, 3>, 6> boundaryPoints{{
    {halfX, 0., 0.},         // face
    {0., -halfY, 0.},        // face
    {halfX, halfY, 0.},      // edge
    {-halfX, 0., halfZ},     // edge
    {halfX, halfY, halfZ},   // vertex
    {-halfX, -halfY, -halfZ} // vertex
  }};
  for (const auto& point : boundaryPoints) {
    BOOST_TEST_CONTEXT("point = (" << point[0] << ", " << point[1] << ", " << point[2] << ")")
    {
      BOOST_CHECK(box.Contains(point.data()));
      BOOST_CHECK(box.Contains_Loop(point.data()));
    }
  }

  // capsule: cylinder barrel closed by two spherical endcaps - a mixed quadric fixture with no
  // ROOT primitive equivalent, cross-validated against the trivial loop and the analytic shape
  constexpr double radius = 1.;
  constexpr double halfHeight = 1.5;
  SurfaceSolid capsule("capsule");
  BOOST_REQUIRE(capsule.AddCylindricalSurface({0., 0., 0.}, {0., 0., 1.}, {1., 0., 0.}, radius, -halfHeight,
                                              halfHeight));
  BOOST_REQUIRE(capsule.AddSphericalSurface({0., 0., halfHeight}, {0., 0., 1.}, {1., 0., 0.}, radius, 0.,
                                            surf::kPi / 2.));
  BOOST_REQUIRE(capsule.AddSphericalSurface({0., 0., -halfHeight}, {0., 0., -1.}, {1., 0., 0.}, radius, 0.,
                                            surf::kPi / 2.));
  capsule.CloseShape();
  BOOST_REQUIRE(capsule.HasBVH());
  BOOST_CHECK(capsule.IsClosed());
  BOOST_CHECK(capsule.IsOrientationConsistent());

  const auto capsuleInside = [&](const double* point) {
    const double axialDistance = std::max(0., std::abs(point[2]) - halfHeight);
    return std::hypot(point[0], point[1], axialDistance) < radius;
  };
  constexpr int samples = 9;
  constexpr double extent = 3.;
  for (int stepX = 0; stepX < samples; ++stepX) {
    for (int stepY = 0; stepY < samples; ++stepY) {
      for (int stepZ = 0; stepZ < samples; ++stepZ) {
        const double point[3] = {-extent + 2. * extent * (stepX + 0.517) / samples,
                                 -extent + 2. * extent * (stepY + 0.263) / samples,
                                 -extent + 2. * extent * (stepZ + 0.741) / samples};
        BOOST_TEST_CONTEXT("point = (" << point[0] << ", " << point[1] << ", " << point[2] << ")")
        {
          const bool bvhInside = capsule.Contains(point);
          BOOST_CHECK_EQUAL(bvhInside, capsule.Contains_Loop(point));
          BOOST_CHECK_EQUAL(bvhInside, capsuleInside(point));
        }
      }
    }
  }

  // a few characteristic capsule points, incl. points exactly on the barrel and cap surfaces
  const double onBarrel[3] = {radius, 0., 0.5};
  const double onCapApex[3] = {0., 0., halfHeight + radius};
  const double aboveApex[3] = {0., 0., halfHeight + radius + 0.01};
  const double onRim[3] = {radius, 0., halfHeight}; // shared cylinder/sphere rim
  BOOST_CHECK(capsule.Contains(onBarrel));
  BOOST_CHECK(capsule.Contains(onCapApex));
  BOOST_CHECK(!capsule.Contains(aboveApex));
  BOOST_CHECK(capsule.Contains(onRim));

  // exact capacity: cylinder plus a full sphere from the two hemispheres
  checkClose(capsule.Capacity(),
             surf::kPi * radius * radius * 2. * halfHeight + 4. * surf::kPi * radius * radius * radius / 3., 1.e-9);
}

BOOST_AUTO_TEST_CASE(DistanceBVHMatchesLoopOnAllFixtures)
{
  // The BVH distance queries against their all-surfaces oracle, over every fixture family and a
  // dense point x direction sweep. This is the correctness guard that does not depend on any
  // reference shape: it isolates traversal and pruning from the analytic kernels, which the
  // per-shape cases above already validate against ROOT.
  const std::array<std::pair<std::unique_ptr<SurfaceSolid>, double>, 7> fixtures{{
    {makeBoxSolid("loopBox", 1., 2., 3.), 4.},
    {makeTubeSolid("loopTube", 0., 2., 3.), 4.},
    {makeTubeSolid("loopHollowTube", 1., 2., 3.), 4.},
    {makeConeSolid("loopCone", 2., 1., 3.), 4.},
    {makeSphereSolid("loopSphere", 2.5), 3.5},
    {makeTorusSolid("loopTorus", 3., 1.), 4.5},
    {makeCapsuleSolid("loopCapsule", 1., 1.5), 3.},
  }};

  for (const auto& [solid, extent] : fixtures) {
    BOOST_TEST_CONTEXT("fixture = " << solid->GetName())
    {
      BOOST_REQUIRE(solid->HasBVH());
      sweepDistanceAgainstLoop(*solid, extent, 5);
    }
  }
}

BOOST_AUTO_TEST_CASE(DistanceSweepsMatchRootPrimitives)
{
  // Systematic point x direction sweeps against the ROOT primitives, for both the entering and
  // the exiting query. The per-shape cases above check a handful of hand-picked rays; this walks
  // a grid, so it also covers rays that miss, that graze, and that cross a hole.
  constexpr int samples = 5;

  const auto box = makeBoxSolid("sweepBox", 1., 2., 3.);
  TGeoBBox boxReference("sweepBoxReference", 1., 2., 3.);
  sweepDistanceAgainstReference(*box, boxReference, 4., samples);

  const auto tube = makeTubeSolid("sweepTube", 0., 2., 3.);
  TGeoTube tubeReference("sweepTubeReference", 0., 2., 3.);
  sweepDistanceAgainstReference(*tube, tubeReference, 4., samples);

  const auto hollowTube = makeTubeSolid("sweepHollowTube", 1., 2., 3.);
  TGeoTube hollowTubeReference("sweepHollowTubeReference", 1., 2., 3.);
  sweepDistanceAgainstReference(*hollowTube, hollowTubeReference, 4., samples);

  const auto cone = makeConeSolid("sweepCone", 2., 1., 3.);
  TGeoCone coneReference("sweepConeReference", 3., 0., 2., 0., 1.);
  sweepDistanceAgainstReference(*cone, coneReference, 4., samples);

  const auto sphere = makeSphereSolid("sweepSphere", 2.5);
  TGeoSphere sphereReference("sweepSphereReference", 0., 2.5);
  sweepDistanceAgainstReference(*sphere, sphereReference, 3.5, samples);

  // the torus kernel solves a quartic, so it carries more rounding than the quadric shapes
  const auto torus = makeTorusSolid("sweepTorus", 3., 1.);
  TGeoTorus torusReference("sweepTorusReference", 3., 0., 1.);
  sweepDistanceAgainstReference(*torus, torusReference, 4.5, samples, 1.e-6);
}

BOOST_AUTO_TEST_CASE(DistanceHardCases)
{
  constexpr double halfX = 1.;
  constexpr double halfY = 2.;
  constexpr double halfZ = 3.;
  const auto box = makeBoxSolid("hardCaseBox", halfX, halfY, halfZ);
  TGeoBBox reference("hardCaseBoxReference", halfX, halfY, halfZ);

  // --- rays through a shared edge and a shared vertex -----------------------------------------
  // Both are seen by more than one patch, so the same crossing is reported several times. Taking
  // the minimum over entering hits is insensitive to that, but the BVH and the loop must still
  // see the same set, which is what the loop cross-check asserts.
  const std::array<std::array<double, 3>, 4> throughFeature{{
    {-5., halfY, 0.},    // straight at the x = -1 / y = +2 edge
    {-5., halfY, halfZ}, // straight at the (-1, +2, +3) vertex
    {0., 0., 0.},        // from the centre out through the +y/+z edge
    {-5., -5., -5.},     // body diagonal through the (-1,-2,-3) vertex
  }};
  const std::array<std::array<double, 3>, 4> throughFeatureDirection{{
    {1., 0., 0.},
    {1., 0., 0.},
    unitDirection(0., 1., 1.5),
    unitDirection(1., 2., 3.),
  }};
  for (size_t index = 0; index < throughFeature.size(); ++index) {
    checkDistanceAgainstLoop(*box, throughFeature[index], throughFeatureDirection[index]);
  }

  // --- grazing / tangent rays ------------------------------------------------------------------
  // A ray in the plane of a face never enters: every hit it can report is tangential, and a
  // tangential hit is not a crossing. Both queries must agree with the loop and find nothing.
  const std::array<std::array<double, 3>, 3> grazing{{
    {-5., halfY, 0.}, // in the plane of the y = +2 face
    {halfX, -5., 0.}, // in the plane of the x = +1 face
    {0., 0., halfZ},  // in the plane of the z = +3 face
  }};
  const std::array<std::array<double, 3>, 3> grazingDirection{{
    {1., 0., 0.},
    {0., 1., 0.},
    unitDirection(1., 1., 0.),
  }};
  for (size_t index = 0; index < grazing.size(); ++index) {
    checkDistanceAgainstLoop(*box, grazing[index], grazingDirection[index]);
  }
  // a cylinder tangent ray: the double root must not be reported as two crossings
  const auto tube = makeTubeSolid("hardCaseTube", 0., 2., 3.);
  checkDistanceAgainstLoop(*tube, {-5., 2., 0.}, {1., 0., 0.});
  checkDistanceAgainstLoop(*tube, {-5., 2. - 1.e-7, 0.}, {1., 0., 0.}); // just inside tangency
  checkDistanceAgainstLoop(*tube, {-5., 2. + 1.e-7, 0.}, {1., 0., 0.}); // just outside tangency

  // --- rays starting exactly on a surface -------------------------------------------------------
  // The on-surface convention (a crossing at t = 0 is below kRayTolerance and is not reported) is
  // inherited from the analytic kernels; what matters here is that the BVH reproduces it exactly.
  const std::array<std::array<double, 3>, 4> onSurface{{
    {halfX, 0., 0.},       // on a face
    {-halfX, 0.5, -1.},    // on the opposite face
    {halfX, halfY, 0.},    // on an edge
    {halfX, halfY, halfZ}, // on a vertex
  }};
  for (const auto& point : onSurface) {
    for (const auto& direction : probeDirections()) {
      checkDistanceAgainstLoop(*box, point, direction);
    }
  }
  // just off the surface the answers must be the ordinary ones: entering after ~1e-6 from
  // outside, exiting after the full traversal from inside
  const std::array<double, 3> justOutside{halfX + 1.e-6, 0., 0.};
  const std::array<double, 3> justInside{halfX - 1.e-6, 0., 0.};
  const std::array<double, 3> inward{-1., 0., 0.};
  checkClose(box->DistFromOutside(justOutside.data(), inward.data(), 3), 1.e-6, 1.e-12);
  checkClose(box->DistFromInside(justInside.data(), inward.data(), 3), 2. * halfX - 1.e-6, 1.e-12);
  checkClose(box->DistFromOutside(justOutside.data(), inward.data(), 3),
             reference.DistFromOutside(justOutside.data(), inward.data(), 3), 1.e-12);

  // --- stepmax ----------------------------------------------------------------------------------
  const std::array<double, 3> farOutside{-5., 0., 0.};
  const std::array<double, 3> alongX{1., 0., 0.};
  const double entryDistance = box->DistFromOutside(farOutside.data(), alongX.data(), 3);
  checkClose(entryDistance, 4.);
  // a hit beyond stepmax must not be reported ...
  BOOST_CHECK_EQUAL(box->DistFromOutside(farOutside.data(), alongX.data(), 3, entryDistance * 0.5),
                    TGeoShape::Big());
  // ... including when it lies only just beyond, and the cheap bounding-box reject must agree
  BOOST_CHECK_EQUAL(box->DistFromOutside(farOutside.data(), alongX.data(), 3, entryDistance - 1.e-3),
                    TGeoShape::Big());
  // ... while a stepmax past the hit changes nothing
  checkClose(box->DistFromOutside(farOutside.data(), alongX.data(), 3, entryDistance + 1.e-3), entryDistance);
  checkClose(box->DistFromOutside(farOutside.data(), alongX.data(), 3, 100.), entryDistance);
  // the same for the exiting query
  const std::array<double, 3> center{0., 0., 0.};
  const double exitDistance = box->DistFromInside(center.data(), alongX.data(), 3);
  checkClose(exitDistance, halfX);
  BOOST_CHECK_EQUAL(box->DistFromInside(center.data(), alongX.data(), 3, exitDistance * 0.5), TGeoShape::Big());
  checkClose(box->DistFromInside(center.data(), alongX.data(), 3, exitDistance * 2.), exitDistance);
  // and the loop must honour stepmax identically, at and around the hit
  for (const double stepmax : {entryDistance * 0.5, entryDistance - 1.e-9, entryDistance, entryDistance + 1.e-9,
                               entryDistance * 2.}) {
    checkDistanceAgainstLoop(*box, farOutside, alongX, stepmax);
    checkDistanceAgainstLoop(*box, center, alongX, stepmax);
  }

  // --- a ray that cannot reach the solid at all --------------------------------------------------
  const std::array<double, 3> wayOff{-1000., 0., 0.};
  BOOST_CHECK_EQUAL(box->DistFromOutside(wayOff.data(), alongX.data(), 3, 10.), TGeoShape::Big());
  checkClose(box->DistFromOutside(wayOff.data(), alongX.data(), 3), 999.);
}

BOOST_AUTO_TEST_CASE(RayTMaxPruningIsOptimizationOnly)
{
  // A row of well-separated boxes: a ray along the row enters the first one, after which every
  // node behind it is beyond the tightened bound and must not be visited. Turning the tightening
  // off must cost candidates without changing a single answer.
  constexpr int boxCount = 8;
  constexpr double half = 0.5;
  constexpr double spacing = 4.;

  SurfaceSolid row("prunedRow");
  for (int boxIndex = 0; boxIndex < boxCount; ++boxIndex) {
    addBoxSurfaces(row, half, half, half, {boxIndex * spacing, 0., 0.});
  }
  row.CloseShape();
  BOOST_REQUIRE(row.HasBVH());
  BOOST_CHECK(row.IsClosed());
  BOOST_CHECK_EQUAL(row.GetNsurfaces(), 6 * boxCount);

  const std::array<double, 3> beforeRow{-5., 0., 0.};
  const std::array<double, 3> alongRow{1., 0., 0.};

  BOOST_CHECK(SurfaceSolid::GetRayTMaxPruning()); // on by default

  SurfaceSolid::ResetRayCandidateCounter();
  const double prunedDistance = row.DistFromOutside(beforeRow.data(), alongRow.data(), 3);
  const long long prunedCandidates = SurfaceSolid::GetRayCandidateCount();

  SurfaceSolid::SetRayTMaxPruning(false);
  SurfaceSolid::ResetRayCandidateCounter();
  const double unprunedDistance = row.DistFromOutside(beforeRow.data(), alongRow.data(), 3);
  const long long unprunedCandidates = SurfaceSolid::GetRayCandidateCount();
  SurfaceSolid::SetRayTMaxPruning(true);

  // same answer, and it is the entry face of the first box
  BOOST_CHECK_EQUAL(prunedDistance, unprunedDistance);
  checkClose(prunedDistance, 5. - half);
  // ... reached after strictly less work
  BOOST_CHECK_GT(prunedCandidates, 0);
  BOOST_CHECK_LT(prunedCandidates, unprunedCandidates);

  // the answers stay identical over a full sweep, which is the property that lets the benchmark
  // treat the switch as a pure cost knob
  sweepDistanceAgainstLoop(row, 1.2 * boxCount * spacing / 2., 4);

  // the counter is not touched by the _Loop variants, which visit everything by construction
  SurfaceSolid::ResetRayCandidateCounter();
  row.DistFromOutside_Loop(beforeRow.data(), alongRow.data());
  BOOST_CHECK_EQUAL(SurfaceSolid::GetRayCandidateCount(), 0);
}

BOOST_AUTO_TEST_CASE(RayTMaxPruningKeepsNearTies)
{
  // Two entering candidates a controlled hair apart, one of them behind a very loose bounding
  // box: the geometry in which a mis-set tmax would do its damage.
  //
  // Why this shape of test. A node is culled when the ray *enters its box* beyond tmax, and a box
  // is always entered no later than the patch inside it is hit. A candidate nearer than the
  // current best therefore has a box entered earlier than the current best's hit, and survives
  // any bound at or above that hit -- which is why the implementation's bound (the best hit,
  // rounded up, plus the box inflation) can be argued safe rather than merely measured safe. The
  // narrow window that is left needs a loose box visited first and a tight one entered between
  // the loose patch's box and its hit, so that is what this fixture builds.
  //
  // Fixture: a sphere hit by a near-limb ray far behind where its bounding box starts, plus a
  // small flat patch just in front of that hit, swept over several decades of separation. It is
  // deliberately not a closed manifold (the patch clips into the sphere) and is closed with the
  // diagnostics off: it exists to place the two candidates, not to model a solid. That is
  // legitimate because the oracle is DistFromOutside_Loop, which minimises over the same hits.
  //
  // Scope, honestly: mutation-testing this suite showed that a bound scaled by 0.5 is caught
  // loudly by the sweeps above, while one scaled by 0.999 is caught by neither them nor this
  // case -- with so few primitives, both leaves are box-tested in the same inner-node visit,
  // before any leaf callback has run and tightened anything. So this pins the near-tie geometry
  // and the pruning-on == pruning-off == loop identity; the guarantee against a subtly tight
  // bound rests on the argument above, not on this test.
  constexpr double radius = 2.;
  constexpr double rayOffsetY = 1.9; // near the limb: box entered at x = -2, surface at x = -0.62
  const double sphereHitX = -std::sqrt(radius * radius - rayOffsetY * rayOffsetY);
  const std::array<double, 3> rayOrigin{-10., rayOffsetY, 0.};
  const std::array<double, 3> alongX{1., 0., 0.};
  const double sphereDistance = sphereHitX - rayOrigin[0];

  // relative offsets spanning several decades below the sphere hit, so any tmax that is too
  // tight by anything in that range is caught by at least one of them regardless of how the
  // builder happens to lay out the tree
  for (const double relativeOffset : {1.e-5, 3.e-5, 1.e-4, 3.e-4, 1.e-3, 3.e-3, 1.e-2}) {
    const double patchX = sphereHitX - relativeOffset * sphereDistance;
    BOOST_TEST_CONTEXT("relativeOffset = " << relativeOffset << " patchX = " << patchX)
    {
      SurfaceSolid solid("nearTie");
      BOOST_REQUIRE(solid.AddSphericalSurface({0., 0., 0.}, {0., 0., 1.}, {1., 0., 0.}, radius));
      // axisU x axisV = z x y = -x: the patch faces the incoming ray, so crossing it enters
      BOOST_REQUIRE(solid.AddPlanarSurface({patchX, rayOffsetY - 0.1, -0.1}, {0., 0., 1.}, {0., 1., 0.},
                                           rectangleWire(0.2, 0.2)));
      solid.CloseShape(false);
      BOOST_REQUIRE(solid.HasBVH());

      // the flat patch is the nearer entering crossing, by construction
      const double expected = patchX - rayOrigin[0];
      for (const bool pruning : {true, false}) {
        SurfaceSolid::SetRayTMaxPruning(pruning);
        BOOST_TEST_CONTEXT("pruning = " << pruning)
        {
          const double distance = solid.DistFromOutside(rayOrigin.data(), alongX.data(), 3);
          checkClose(distance, expected, 1.e-9);
          BOOST_CHECK_EQUAL(distance, solid.DistFromOutside_Loop(rayOrigin.data(), alongX.data()));
        }
      }
      SurfaceSolid::SetRayTMaxPruning(true);
    }
  }
}

BOOST_AUTO_TEST_CASE(CurvedPlanarStadiumPrism)
{
  // A stadium (rectangle with two semicircular ends) extruded along z: the two end caps are
  // planar faces with mixed line+arc wires - the general curved-planar case a disk cannot
  // express. Straight sides are flat rectangles; the round ends are half-cylinders.
  constexpr double halfLen = 3.;    // straight half-length along x
  constexpr double radius = 2.;     // corner radius and half-width along y
  constexpr double halfHeight = 4.; // half-height along z

  // Stadium cross-section boundary in the cap's local (u=x, v=y) frame, CCW: bottom line,
  // right semicircle, top line, left semicircle.
  const std::vector<BoundaryCurve> stadiumWire{
    BoundaryCurve::makeLine({-halfLen, -radius}, {halfLen, -radius}),
    BoundaryCurve::makeArc({halfLen, 0.}, radius, -surf::kHalfPi, surf::kHalfPi),
    BoundaryCurve::makeLine({halfLen, radius}, {-halfLen, radius}),
    BoundaryCurve::makeArc({-halfLen, 0.}, radius, surf::kHalfPi, 3. * surf::kHalfPi)};

  SurfaceSolid solid("stadiumPrism");
  // caps (outward +z / -z: the bottom cap flips axisV)
  BOOST_REQUIRE(solid.AddCurvedPlanarSurface({0., 0., halfHeight}, {1., 0., 0.}, {0., 1., 0.}, stadiumWire));
  BOOST_REQUIRE(solid.AddCurvedPlanarSurface({0., 0., -halfHeight}, {1., 0., 0.}, {0., -1., 0.}, stadiumWire));
  // flat side walls at y = +/- radius (outward +/- y)
  BOOST_REQUIRE(solid.AddPlanarSurface({-halfLen, radius, -halfHeight}, {0., 0., 1.}, {1., 0., 0.},
                                       rectangleWire(2. * halfHeight, 2. * halfLen)));
  BOOST_REQUIRE(solid.AddPlanarSurface({-halfLen, -radius, -halfHeight}, {1., 0., 0.}, {0., 0., 1.},
                                       rectangleWire(2. * halfLen, 2. * halfHeight)));
  // round ends as half-cylinders (outer walls)
  BOOST_REQUIRE(solid.AddCylindricalSurface({halfLen, 0., 0.}, {0., 0., 1.}, {1., 0., 0.}, radius, -halfHeight,
                                            halfHeight, -surf::kHalfPi, surf::kPi));
  BOOST_REQUIRE(solid.AddCylindricalSurface({-halfLen, 0., 0.}, {0., 0., 1.}, {1., 0., 0.}, radius, -halfHeight,
                                            halfHeight, surf::kHalfPi, surf::kPi));
  solid.CloseShape();

  BOOST_CHECK(solid.IsClosed());
  BOOST_CHECK(solid.IsOrientationConsistent());

  // exact capacity: (rectangle 2L x 2R + full circle pi R^2) x height 2H
  checkClose(solid.Capacity(), (4. * halfLen * radius + surf::kPi * radius * radius) * 2. * halfHeight, 1.e-6);

  const auto stadiumInside = [&](const double* point) {
    if (std::abs(point[2]) > halfHeight) {
      return false;
    }
    const double ax = std::abs(point[0]);
    const double dx = ax > halfLen ? ax - halfLen : 0.;
    return dx * dx + point[1] * point[1] <= radius * radius;
  };
  // deterministic grid spanning well beyond the solid on every axis
  constexpr int samples = 21;
  const double extentX = 6., extentY = 3.5, extentZ = 5.;
  for (int stepX = 0; stepX < samples; ++stepX) {
    for (int stepY = 0; stepY < samples; ++stepY) {
      for (int stepZ = 0; stepZ < samples; ++stepZ) {
        const double point[3] = {-extentX + 2. * extentX * (stepX + 0.517) / samples,
                                 -extentY + 2. * extentY * (stepY + 0.263) / samples,
                                 -extentZ + 2. * extentZ * (stepZ + 0.741) / samples};
        BOOST_TEST_CONTEXT("point = (" << point[0] << ", " << point[1] << ", " << point[2] << ")")
        {
          BOOST_CHECK_EQUAL(solid.Contains(point), stadiumInside(point));
        }
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(WireTrimmedCylinderMatchesTube)
{
  constexpr double radius = 2.;
  constexpr double halfHeight = 3.;

  // lateral wall via the wire-trim overload: the trim is the full parametric rectangle
  // phi in [0, 2pi] x h in [-hh, hh] expressed as four line edges, which must behave exactly like
  // the scalar rectangle path (equivalence check).
  SurfaceSolid solid("wireTrimmedCylinder");
  BOOST_REQUIRE(solid.AddCylindricalSurface({0., 0., 0.}, {0., 0., 1.}, {1., 0., 0.}, radius, -halfHeight,
                                            halfHeight, 0., surf::kTwoPi, false,
                                            paramRectWire(0., surf::kTwoPi, -halfHeight, halfHeight)));
  BOOST_REQUIRE(addDiskSurface(solid, {0., 0., halfHeight}, {1., 0., 0.}, {0., 1., 0.}, radius));
  BOOST_REQUIRE(addDiskSurface(solid, {0., 0., -halfHeight}, {1., 0., 0.}, {0., -1., 0.}, radius));
  solid.CloseShape();

  BOOST_CHECK(solid.IsClosed());
  BOOST_CHECK(solid.IsOrientationConsistent());

  TGeoTube reference("wireTrimTube", 0., radius, halfHeight);
  compareContainsGrid(solid, reference, 4., 9);
  compareDistance(solid, reference, {5., 0., 0.}, {-1., 0., 0.});
  compareDistance(solid, reference, {0., 0., 5.}, {0., 0., -1.});
  compareDistance(solid, reference, {-4., -1., -2.}, unitDirection(1., 0.3, 0.5));
  compareDistance(solid, reference, {0., 0., 0.}, unitDirection(1., 1., 1.));
  compareDistance(solid, reference, {5., 2.5, 0.}, {-1., 0., 0.}); // grazing miss

  // capacity is numerically integrated for a wire trim; the wall integrand is constant here so it
  // stays accurate, but compare with a relaxed tolerance to reflect the quadrature
  checkClose(solid.Capacity(), reference.Capacity(), 1.e-6);

  double normal[3] = {0., 0., 0.};
  const double sidePoint[3] = {radius, 0., 1.};
  const double alongX[3] = {1., 0., 0.};
  solid.ComputeNormal(sidePoint, alongX, normal);
  checkClose(normal[0], 1.);
  checkClose(normal[1], 0.);
  checkClose(normal[2], 0.);
}

BOOST_AUTO_TEST_CASE(WireTrimmedConeMatchesCone)
{
  constexpr double halfHeight = 3.;
  constexpr double radiusAtBottom = 2.;
  constexpr double radiusAtTop = 1.;

  SurfaceSolid solid("wireTrimmedCone");
  BOOST_REQUIRE(solid.AddConicalSurface({0., 0., 0.}, {0., 0., 1.}, {1., 0., 0.}, radiusAtBottom, radiusAtTop,
                                        -halfHeight, halfHeight, 0., surf::kTwoPi, false,
                                        paramRectWire(0., surf::kTwoPi, -halfHeight, halfHeight)));
  BOOST_REQUIRE(addDiskSurface(solid, {0., 0., halfHeight}, {1., 0., 0.}, {0., 1., 0.}, radiusAtTop));
  BOOST_REQUIRE(addDiskSurface(solid, {0., 0., -halfHeight}, {1., 0., 0.}, {0., -1., 0.}, radiusAtBottom));
  solid.CloseShape();

  BOOST_CHECK(solid.IsClosed());
  BOOST_CHECK(solid.IsOrientationConsistent());

  TGeoCone reference("wireTrimCone", halfHeight, 0., radiusAtBottom, 0., radiusAtTop);
  compareContainsGrid(solid, reference, 3.5, 9);
  compareDistance(solid, reference, {5., 0., 0.}, {-1., 0., 0.});
  compareDistance(solid, reference, {0., 0., 0.}, {1., 0., 0.});
  compareDistance(solid, reference, {-4., 0.2, -2.}, unitDirection(1., 0.05, 0.3));
  // The wire-trimmed cone's capacity used to need a 1e-3 allowance for the grid quadrature; the
  // Green's-theorem contour form is exact on this rectangle-tracing wire, so it can be held to the
  // same tolerance as any untrimmed patch. A regression to the grid rule fails here.
  checkClose(solid.Capacity(), reference.Capacity(), 1.e-9);
}

// Green's theorem for wire-trimmed quadrics.
//
// The integrand does not depend on the second parameter for a cylinder, and depends on the first
// only through sin/cos/identity for all four quadrics, so an antiderivative in u exists in closed
// form and the area integral collapses to a contour integral around the trim wire.
//
// The sharpest possible check: a wire that traces exactly the parametric rectangle must give the
// same number as the rectangle's own closed form, which is analytically exact. Anything the
// contour form got wrong -- a sign, an orientation, a missed seam, a wrong antiderivative --
// shows up here immediately, and the old quadrature could only ever have agreed to ~1e-2.
BOOST_AUTO_TEST_CASE(WireTrimCapacityMatchesTheClosedForm)
{
  std::string error;
  // an off-origin centre and a tilted frame, so every term of every antiderivative is exercised
  // (C.U, C.V and C.W all non-zero) rather than cancelling
  const surf::Vec3 centre{0.7, -1.3, 0.45};
  const surf::Vec3 axis = surf::normalized({0.3, 0.4, 1.});
  const surf::Vec3 reference{1., 0.2, 0.};
  constexpr double kRelative = 1.e-12;

  const auto compare = [&](const char* what, const surf::BoundedSurface& rectangle,
                           const surf::BoundedSurface& wired) {
    BOOST_TEST_CONTEXT(what)
    {
      BOOST_CHECK(!wired.capacityIsExact()); // still reported inexact: see the note below
      const double exact = rectangle.capacityContribution();
      const double contour = wired.capacityContribution();
      BOOST_CHECK_GT(std::abs(exact), 1.e-6); // a zero contribution would prove nothing
      checkClose(contour, exact, kRelative * std::abs(exact));
    }
  };

  {
    surf::CylindricalBoundedSurface rectangle;
    surf::CylindricalBoundedSurface wired;
    const double phiLow = 0.3, phiHigh = 2.4, hLow = -0.8, hHigh = 1.9;
    BOOST_REQUIRE(rectangle.initialize(centre, axis, reference, 1.7, hLow, hHigh, phiLow, phiHigh - phiLow, false,
                                       error));
    BOOST_REQUIRE(wired.initialize(centre, axis, reference, 1.7, hLow, hHigh, phiLow, phiHigh - phiLow, false,
                                   paramRectWireCurves(phiLow, phiHigh, hLow, hHigh), {}, error));
    compare("cylinder", rectangle, wired);
  }
  {
    surf::ConicalBoundedSurface rectangle;
    surf::ConicalBoundedSurface wired;
    const double phiLow = -0.4, phiHigh = 1.9, hLow = 0.2, hHigh = 2.1;
    BOOST_REQUIRE(rectangle.initialize(centre, axis, reference, 1.1, 2.3, hLow, hHigh, phiLow, phiHigh - phiLow,
                                       false, error));
    BOOST_REQUIRE(wired.initialize(centre, axis, reference, 1.1, 2.3, hLow, hHigh, phiLow, phiHigh - phiLow, false,
                                   paramRectWireCurves(phiLow, phiHigh, hLow, hHigh), {}, error));
    compare("cone", rectangle, wired);
  }
  {
    surf::SphericalBoundedSurface rectangle;
    surf::SphericalBoundedSurface wired;
    const double phiLow = 0.2, phiHigh = 2.7, thetaLow = 0.4, thetaHigh = 2.3;
    BOOST_REQUIRE(rectangle.initialize(centre, axis, reference, 2.2, thetaLow, thetaHigh, phiLow, phiHigh - phiLow,
                                       false, error));
    BOOST_REQUIRE(wired.initialize(centre, axis, reference, 2.2, thetaLow, thetaHigh, phiLow, phiHigh - phiLow, false,
                                   paramRectWireCurves(phiLow, phiHigh, thetaLow, thetaHigh), {}, error));
    compare("sphere", rectangle, wired);
  }
  {
    surf::TorusBoundedSurface rectangle;
    surf::TorusBoundedSurface wired;
    const double ringLow = 0.1, ringHigh = 2.2, tubeLow = -0.3, tubeHigh = 1.8;
    BOOST_REQUIRE(rectangle.initialize(centre, axis, reference, 4., 1.4, ringLow, ringHigh - ringLow, tubeLow,
                                       tubeHigh - tubeLow, false, error));
    BOOST_REQUIRE(wired.initialize(centre, axis, reference, 4., 1.4, ringLow, ringHigh - ringLow, tubeLow,
                                   tubeHigh - tubeLow, false,
                                   paramRectWireCurves(ringLow, ringHigh, tubeLow, tubeHigh), {}, error));
    compare("torus", rectangle, wired);
  }

  // A trim the rectangle cannot express, checked against the integrator it replaces. The midpoint
  // rule is a genuinely independent computation -- it needs nothing of the integrand but its value
  // -- so agreement is evidence, but only to its own O(1/N) accuracy, which is the whole reason
  // this change exists. Refining it must walk *towards* the contour answer; that direction is the
  // real assertion here, not either number.
  {
    surf::CylindricalBoundedSurface disk;
    const double radius = 1.7;
    const std::vector<surf::Curve2D> trim{surf::Curve2D::makeCircle({1.0, 0.2}, 0.6)};
    BOOST_REQUIRE(disk.initialize(centre, axis, reference, radius, -2., 2., 0., surf::kTwoPi, false, trim, {},
                                  error));
    const double contour = disk.capacityContribution();

    // the same trim, rebuilt here so the grid rule can be run over it directly
    surf::CurveWire outerWire;
    std::vector<surf::CurveWire> innerWires;
    surf::Vec2 lower, upper;
    BOOST_REQUIRE(surf::buildCurveTrim(trim, {}, outerWire, innerWires, lower, upper, error,
                                       surf::parametricMetricOf(disk)));

    const auto gridRelativeError = [&](int samples) {
      const double grid = surf::integrateOverCurveTrim(
        outerWire, innerWires,
        [&disk, radius](double phi, double height) {
          const surf::Vec3 point = disk.pointAt(phi, height);
          return surf::dot(point, disk.normalAt(point)) * radius / 3.;
        },
        samples);
      return std::abs(grid - contour) / std::abs(contour);
    };
    const double at128 = gridRelativeError(128);
    const double at512 = gridRelativeError(512);
    const double at2048 = gridRelativeError(2048);

    // The grid rule confirms the contour value to its own accuracy -- an independent computation
    // agreeing to 3e-5 is what says the antiderivative route is not just self-consistent.
    BOOST_CHECK_LT(at512, 1.e-4);
    BOOST_CHECK_LT(at2048, 1.e-4);
    // But it cannot do better, and that is the point. At the shipped 128 it is off by 2e-3 --
    // three orders outside the gate's 1e-6 band -- and refining it sixteen-fold does not fix that,
    // because the error is not monotone: the staircase re-phases and 2048 is *worse* than 512
    // (2.9e-5 against 2.4e-5 here; on ExcavatorArm/BucketLink2 the sequence 128..2048 runs 16.004,
    // 17.710, 16.927, 17.244, 17.032 around a true 17.079). So no N could have been the fix.
    BOOST_CHECK_GT(at128, 1.e-3);
    BOOST_CHECK_GT(at2048, at512);
  }
}

BOOST_AUTO_TEST_CASE(WireTrimmedQuadricKernels)
{
  using surf::Curve2D;
  using surf::Vec3;
  std::string error;

  const auto onCylinder = [](double phi, double height) {
    return Vec3{2. * std::cos(phi), 2. * std::sin(phi), height};
  };

  // (1) cylinder wall with a rectangular window (hole) in (phi, h): phi in [2.0, 2.5], h in [-1, 1]
  surf::CylindricalBoundedSurface windowed;
  const std::vector<Curve2D> outer{Curve2D::makeLine({0., -3.}, {surf::kTwoPi, -3.}),
                                   Curve2D::makeLine({surf::kTwoPi, -3.}, {surf::kTwoPi, 3.}),
                                   Curve2D::makeLine({surf::kTwoPi, 3.}, {0., 3.}),
                                   Curve2D::makeLine({0., 3.}, {0., -3.})};
  const std::vector<Curve2D> hole{Curve2D::makeLine({2.0, -1.}, {2.5, -1.}), Curve2D::makeLine({2.5, -1.}, {2.5, 1.}),
                                  Curve2D::makeLine({2.5, 1.}, {2.0, 1.}), Curve2D::makeLine({2.0, 1.}, {2.0, -1.})};
  BOOST_REQUIRE(windowed.initialize({0., 0., 0.}, {0., 0., 1.}, {1., 0., 0.}, 2., -3., 3., 0., surf::kTwoPi, false,
                                    outer, {hole}, error));
  BOOST_CHECK(windowed.containsPointOnSurface(onCylinder(0.5, 0.)));   // material
  BOOST_CHECK(windowed.containsPointOnSurface(onCylinder(2.25, 2.5))); // material above the window
  BOOST_CHECK(!windowed.containsPointOnSurface(onCylinder(2.25, 0.))); // inside the window
  BOOST_CHECK(windowed.containsPointOnSurface(onCylinder(2.25, 1.)));  // on the window edge (boundary)

  // a radial ray into the window is filtered out; a radial ray into material registers one hit
  std::vector<surf::RayHit> hits;
  windowed.appendIntersections({0., 0., 0.}, {std::cos(2.25), std::sin(2.25), 0.}, 0., 1.e30, hits);
  BOOST_CHECK(hits.empty());
  hits.clear();
  windowed.appendIntersections({0., 0., 0.}, {std::cos(0.5), std::sin(0.5), 0.}, 0., 1.e30, hits);
  BOOST_REQUIRE_EQUAL(hits.size(), 1u);
  checkClose(hits.front().distance, 2.);

  // (2) arc trim: a parametric circle (disk in (phi, h)) centred at (pi, 0), radius 0.5
  surf::CylindricalBoundedSurface arcTrim;
  const std::vector<Curve2D> arcOuter{Curve2D::makeCircle({surf::kPi, 0.}, 0.5)};
  BOOST_REQUIRE(arcTrim.initialize({0., 0., 0.}, {0., 0., 1.}, {1., 0., 0.}, 2., -1., 1., 0., surf::kTwoPi, false,
                                   arcOuter, {}, error));
  BOOST_CHECK(arcTrim.containsPointOnSurface(onCylinder(surf::kPi, 0.)));        // centre of the disk
  BOOST_CHECK(!arcTrim.containsPointOnSurface(onCylinder(surf::kPi, 0.6)));      // outside in h
  BOOST_CHECK(!arcTrim.containsPointOnSurface(onCylinder(surf::kPi + 0.6, 0.))); // outside in phi
  BOOST_CHECK_GT(std::abs(arcTrim.capacityContribution()), 0.);

  // (3) sphere section reproduced as a (phi, theta) rectangle wire must match the scalar section
  const auto onSphere = [](double theta, double phi) {
    return Vec3{2. * std::sin(theta) * std::cos(phi), 2. * std::sin(theta) * std::sin(phi), 2. * std::cos(theta)};
  };
  surf::SphericalBoundedSurface sphereWire;
  BOOST_REQUIRE(sphereWire.initialize({0., 0., 0.}, {0., 0., 1.}, {1., 0., 0.}, 2., surf::kHalfPi / 2., surf::kHalfPi,
                                      0., surf::kHalfPi, false,
                                      paramRectWireCurves(0., surf::kHalfPi, surf::kHalfPi / 2., surf::kHalfPi), {},
                                      error));
  surf::SphericalBoundedSurface sphereScalar;
  BOOST_REQUIRE(sphereScalar.initialize({0., 0., 0.}, {0., 0., 1.}, {1., 0., 0.}, 2., surf::kHalfPi / 2.,
                                        surf::kHalfPi, 0., surf::kHalfPi, false, error));
  BOOST_CHECK(sphereWire.containsPointOnSurface(onSphere(surf::kPi / 3., surf::kPi / 4.)));       // inside the section
  BOOST_CHECK(!sphereWire.containsPointOnSurface(onSphere(surf::kPi / 6., surf::kPi / 4.)));      // theta too small
  BOOST_CHECK(!sphereWire.containsPointOnSurface(onSphere(surf::kPi / 3., 3. * surf::kPi / 4.))); // phi outside
  // same story on the sphere: 1e-3 was the grid rule's allowance, not the geometry's
  checkClose(sphereWire.capacityContribution(), sphereScalar.capacityContribution(), 1.e-9);

  // (4) a trim spanning more than a full turn in phi is rejected
  surf::CylindricalBoundedSurface tooWide;
  const std::vector<Curve2D> wideOuter{Curve2D::makeLine({0., -1.}, {7., -1.}), Curve2D::makeLine({7., -1.}, {7., 1.}),
                                       Curve2D::makeLine({7., 1.}, {0., 1.}), Curve2D::makeLine({0., 1.}, {0., -1.})};
  BOOST_CHECK(!tooWide.initialize({0., 0., 0.}, {0., 0., 1.}, {1., 0., 0.}, 2., -1., 1., 0., surf::kTwoPi, false,
                                  wideOuter, {}, error));
}

namespace
{
// Estimate the first fundamental form at (u, v) by central differences of a surface's own
// parametrisation. Checking parametricMetric against this is a proof that the closed form
// describes the map the rest of the kernel actually evaluates -- restating the formula in the
// test would only prove it was copied twice.
template <typename PointAt>
void checkMetricAgainstFiniteDifference(const surf::BoundedSurface& surface, const PointAt& pointAt,
                                        double uCoord, double vCoord, double tolerance = 1.e-6)
{
  const double step = 1.e-5;
  const surf::Vec3 dU = (pointAt(uCoord + step, vCoord) - pointAt(uCoord - step, vCoord)) * (0.5 / step);
  const surf::Vec3 dV = (pointAt(uCoord, vCoord + step) - pointAt(uCoord, vCoord - step)) * (0.5 / step);

  double gUU = 0.;
  double gUV = 0.;
  double gVV = 0.;
  surface.parametricMetric({uCoord, vCoord}, gUU, gUV, gVV);
  checkClose(gUU, dot(dU, dU), tolerance);
  checkClose(gUV, dot(dU, dV), tolerance);
  checkClose(gVV, dot(dV, dV), tolerance);
}
} // namespace

// The first fundamental form of every surface family, against the surface's own parametrisation,
// plus the two degeneracies and the cross term the callers of it have to cope with. This is the
// conversion that makes a parametric tolerance mean a length (findings K3, K5, K12, S10).
BOOST_AUTO_TEST_CASE(ParametricMetricIsTheFirstFundamentalForm)
{
  using surf::Vec2;
  using surf::Vec3;
  std::string error;

  // (1) plane with deliberately non-orthonormal axes: the only family with a cross term, and the
  // only one whose (u, v) are not already lengths.
  const Vec3 axisU{2., 0., 0.};
  const Vec3 axisV{1., 3., 0.}; // not unit, not orthogonal to axisU
  const std::vector<Vec2> unitSquare{{0., 0.}, {1., 0.}, {1., 1.}, {0., 1.}};
  surf::PlanarBoundedSurface plane;
  BOOST_REQUIRE(plane.initialize({0.5, -1., 2.}, axisU, axisV, unitSquare, {}, error));
  checkMetricAgainstFiniteDifference(plane, [&](double u, double v) { return plane.toGlobal({u, v}); }, 0.3, 0.7);
  {
    double gUU = 0.;
    double gUV = 0.;
    double gVV = 0.;
    plane.parametricMetric({0., 0.}, gUU, gUV, gVV);
    checkClose(gUU, 4.);
    checkClose(gUV, 2.); // dot(axisU, axisV) -- zero for every other family
    checkClose(gVV, 10.);
    // and it really measures 3D length: (du, dv) = (1, 0) spans |axisU| = 2 cm
    checkClose(std::sqrt(plane.parametricLengthSqAt({0., 0.}, {1., 0.})), 2.);
    checkClose(std::sqrt(plane.parametricLengthSqAt({0., 0.}, {0., 1.})), std::sqrt(10.));
  }

  // (2) curved planar: initialize() insists on an orthonormal frame, so (u, v) are centimetres.
  surf::CurvedPlanarBoundedSurface curvedPlane;
  BOOST_REQUIRE(curvedPlane.initialize({0., 0., 0.}, {1., 0., 0.}, {0., 1., 0.},
                                       {surf::Curve2D::makeCircle({0., 0.}, 1.)}, {}, error));
  checkMetricAgainstFiniteDifference(
    curvedPlane, [&](double u, double v) { return curvedPlane.toGlobal({u, v}); }, 0.2, -0.4);

  // (3) cylinder, (u, v) = (phi, h). The radius factor is the whole point: the same parametric
  // drift is a different distance on a small hole and on a large cylinder.
  for (const double radius : {0.01, 100.}) {
    surf::CylindricalBoundedSurface cylinder;
    BOOST_REQUIRE(cylinder.initialize({0., 0., 0.}, {0., 0., 1.}, {1., 0., 0.}, radius, -1., 1., 0., surf::kTwoPi,
                                      false, error));
    checkMetricAgainstFiniteDifference(cylinder, [&](double u, double v) { return cylinder.pointAt(u, v); }, 1.1, 0.3, 1.e-4 * radius * radius);
    // a 2e-5 rad join drift is 2e-7 cm on the small cylinder and 2e-3 cm on the large one
    checkClose(std::sqrt(cylinder.parametricLengthSqAt({1.1, 0.3}, {2.e-5, 0.})), 2.e-5 * radius, 1.e-12);
  }

  // (4) sphere, (u, v) = (phi, theta) -- the trim domain's order, the transpose of pointAt's.
  surf::SphericalBoundedSurface sphere;
  BOOST_REQUIRE(sphere.initialize({1., 2., 3.}, {0., 0., 1.}, {1., 0., 0.}, 2.5, 0., surf::kPi, 0., surf::kTwoPi,
                                  false, error));
  checkMetricAgainstFiniteDifference(sphere, [&](double u, double v) { return sphere.pointAt(v, u); }, 0.9, 1.2);
  {
    double gUU = 0.;
    double gUV = 0.;
    double gVV = 0.;
    // at the pole the azimuth degenerates: a phi separation there spans no distance at all
    sphere.parametricMetric({0.9, 0.}, gUU, gUV, gVV);
    checkClose(gUU, 0.);
    checkClose(gVV, 2.5 * 2.5);
    checkClose(sphere.parametricLengthSqAt({0.9, 0.}, {1., 0.}), 0.);
    sphere.parametricMetric({0.9, surf::kPi}, gUU, gUV, gVV);
    checkClose(gUU, 0.);
    // and on the equator it is the full radius
    sphere.parametricMetric({0.9, surf::kHalfPi}, gUU, gUV, gVV);
    checkClose(gUU, 2.5 * 2.5);
  }

  // (5) cone, (u, v) = (phi, h): the azimuthal scale shrinks to zero at the apex, and a step in h
  // walks along the slope rather than along the axis.
  surf::ConicalBoundedSurface cone;
  BOOST_REQUIRE(cone.initialize({0., 0., 0.}, {0., 0., 1.}, {1., 0., 0.}, 0., 4., 0., 2., 0., surf::kTwoPi, false,
                                error));
  checkMetricAgainstFiniteDifference(cone, [&](double u, double v) { return cone.pointAt(u, v); }, 2.0, 1.3);
  {
    double gUU = 0.;
    double gUV = 0.;
    double gVV = 0.;
    cone.parametricMetric({2.0, 0.}, gUU, gUV, gVV); // the apex, where r(h) = 0
    checkClose(gUU, 0.);
    checkClose(gVV, 1. + 2. * 2.); // slope = (4 - 0) / (2 - 0)
    checkClose(cone.parametricLengthSqAt({2.0, 0.}, {1., 0.}), 0.);
    cone.parametricMetric({2.0, 2.}, gUU, gUV, gVV); // the wide end, r = 4
    checkClose(gUU, 16.);
  }

  // (6) torus, (u, v) = (phiRing, phiTube): the ring scale runs from R - r to R + r around the
  // tube, so it is the one family whose gUU varies without any degeneracy.
  surf::TorusBoundedSurface torus;
  BOOST_REQUIRE(torus.initialize({0., 0., 0.}, {0., 0., 1.}, {1., 0., 0.}, 5., 1.5, 0., surf::kTwoPi, 0.,
                                 surf::kTwoPi, false, error));
  checkMetricAgainstFiniteDifference(torus, [&](double u, double v) { return torus.pointAt(u, v); }, 0.7, 2.1);
  {
    double gUU = 0.;
    double gUV = 0.;
    double gVV = 0.;
    torus.parametricMetric({0.7, 0.}, gUU, gUV, gVV); // outside of the ring
    checkClose(gUU, 6.5 * 6.5);
    checkClose(gVV, 1.5 * 1.5);
    torus.parametricMetric({0.7, surf::kPi}, gUU, gUV, gVV); // inside of the ring
    checkClose(gUU, 3.5 * 3.5);
  }
}

// The join tolerance is a length, so the same parametric drift is accepted on a small cylinder
// and refused on a large one. Today's rule cannot tell them apart, which is the whole of K3 -- and
// it is the synthetic form of the measured ST1829909_01 loader rejection (six joins under 3e-5 rad
// on cylinder trims, negligible in arc length, read as three times over a 1e-5 "tolerance").
BOOST_AUTO_TEST_CASE(WireJoinToleranceIsALength)
{
  using surf::Curve2D;
  std::string error;

  // A rectangular (phi, h) trim whose last edge stops `drift` radians short of closing the loop.
  const auto trimWithPhiDrift = [](double drift) {
    return std::vector<Curve2D>{Curve2D::makeLine({0.2, -1.}, {1.2, -1.}), Curve2D::makeLine({1.2, -1.}, {1.2, 1.}),
                                Curve2D::makeLine({1.2, 1.}, {0.2 + drift, 1.}),
                                Curve2D::makeLine({0.2 + drift, 1.}, {0.2 + drift, -1.})};
  };
  const auto acceptsDrift = [&](double radius, double drift) {
    surf::CylindricalBoundedSurface cylinder;
    return cylinder.initialize({0., 0., 0.}, {0., 0., 1.}, {1., 0., 0.}, radius, -1., 1., 0., surf::kTwoPi, false,
                               trimWithPhiDrift(drift), {}, error);
  };

  // The same 2e-5 rad drift, on two radii: 2e-7 cm of arc on the small cylinder against 2e-3 cm
  // on the large one. One is a rounding error, the other is a real gap.
  BOOST_CHECK(acceptsDrift(0.01, 2.e-5)); // the old rule refused this: 2e-5 > 1e-5, radius unseen
  BOOST_CHECK(!acceptsDrift(100., 2.e-5));
  // and the discrimination really is the radius: give the large cylinder a drift small enough to
  // span the same 2e-7 cm and it is accepted; give the small one a gap of 2e-5 cm and it is not.
  BOOST_CHECK(acceptsDrift(100., 2.e-9));
  BOOST_CHECK(!acceptsDrift(0.01, 2.e-3));
  // On a cylinder of radius 1 the two rules coincide up to the change of constant -- which is the
  // only configuration the old one was ever right on, and then only by accident.
  BOOST_CHECK(acceptsDrift(1., 5.e-7));
  BOOST_CHECK(!acceptsDrift(1., 5.e-6)); // the old rule accepted this: 5e-6 < 1e-5
}

// K12: the polygon and curve wire types are fed by the same extractor with the same per-endpoint
// precision, and now judge a join by the same rule. They used to differ by four orders of
// magnitude -- 1e-9 for polygons against 1e-5 for curves -- and in incompatible units.
BOOST_AUTO_TEST_CASE(PolygonAndCurveWiresShareOneJoinRule)
{
  using surf::SurfaceEdge;
  using surf::Vec2;
  using surf::WireRole;
  using surf::WireStatus;

  // A square whose last edge ends `gap` short of the first edge's start, in a domain where
  // (u, v) are already centimetres (the default identity metric).
  const auto polygonAcceptsGap = [](double gap) {
    const std::vector<SurfaceEdge> edges{{{0., 0.}, {1., 0.}},
                                         {{1., 0.}, {1., 1.}},
                                         {{1., 1.}, {0., 1.}},
                                         {{0., 1.}, {gap, 0.}}};
    surf::SurfaceWire wire;
    WireStatus status = WireStatus::Valid;
    return wire.initializeFromEdges(edges, WireRole::Outer, status, {});
  };
  const auto curveAcceptsGap = [](double gap) {
    const std::vector<surf::Curve2D> curves{surf::Curve2D::makeLine({0., 0.}, {1., 0.}),
                                            surf::Curve2D::makeLine({1., 0.}, {1., 1.}),
                                            surf::Curve2D::makeLine({1., 1.}, {0., 1.}),
                                            surf::Curve2D::makeLine({0., 1.}, {gap, 0.})};
    surf::CurveWire wire;
    WireStatus status = WireStatus::Valid;
    return wire.initialize(curves, WireRole::Outer, status, {});
  };

  // inside the 1e-6 cm tolerance. 1e-8 and 1e-7 are the discriminating cases: the polygon wire
  // used to refuse them at 1e-9 while the curve wire accepted them at 1e-5.
  for (const double gap : {0., 1.e-8, 1.e-7}) {
    BOOST_CHECK(polygonAcceptsGap(gap));
    BOOST_CHECK(curveAcceptsGap(gap));
  }
  // outside it. 1e-5 is the mirror case: the curve wire used to accept it and the polygon not.
  for (const double gap : {1.e-5, 1.e-4}) {
    BOOST_CHECK(!polygonAcceptsGap(gap));
    BOOST_CHECK(!curveAcceptsGap(gap));
  }
}

namespace
{
// Helpers writing the surface sidecar binary format documented in
// Detectors/CADSupport/doc/reference/BVHSurfaceSolid.md. Kept independent of the loader implementation so the
// test is a true round-trip through the documented byte layout.
void appendU32(std::vector<char>& bytes, uint32_t value)
{
  const char* raw = reinterpret_cast<const char*>(&value);
  bytes.insert(bytes.end(), raw, raw + sizeof(value));
}

void appendDoubles(std::vector<char>& bytes, std::initializer_list<double> values)
{
  for (const double value : values) {
    const char* raw = reinterpret_cast<const char*>(&value);
    bytes.insert(bytes.end(), raw, raw + sizeof(value));
  }
}

// The fixed header. Version 1 is the three-uint32 form; version 2 appends the model tolerance in
// cm. The default stays at version 1 on purpose: every sidecar test below then doubles as a
// regression test that the reader still accepts the older format.
void appendSidecarHeader(std::vector<char>& bytes, uint32_t nSurfaces, uint32_t version = 1,
                         double modelTolerance = 0., uint32_t nModelEdges = 0)
{
  bytes.insert(bytes.end(), {'O', '2', 'S', 'S'});
  appendU32(bytes, version);
  appendU32(bytes, nSurfaces);
  appendU32(bytes, 0); // reserved
  if (version >= 2) {
    appendDoubles(bytes, {modelTolerance});
  }
  if (version >= 3) {
    appendU32(bytes, nModelEdges); // size of the model's edge table
  }
}

// plane record (type 1) with a single rectangular outer wire of four line-segment edges
void appendPlaneRecord(std::vector<char>& bytes, const FaceFrame& frame)
{
  appendU32(bytes, 1); // surfaceType plane
  appendU32(bytes, 0); // flags
  appendU32(bytes, 9); // nParams
  appendDoubles(bytes, {frame.origin[0], frame.origin[1], frame.origin[2], frame.axisU[0], frame.axisU[1],
                        frame.axisU[2], frame.axisV[0], frame.axisV[1], frame.axisV[2]});
  appendU32(bytes, 1); // nWires
  appendU32(bytes, 0); // wireRole outer
  appendU32(bytes, 4); // nEdges
  const double extentU = frame.extentU;
  const double extentV = frame.extentV;
  const std::array<std::array<double, 4>, 4> edges{{{0., 0., extentU, 0.},
                                                    {extentU, 0., extentU, extentV},
                                                    {extentU, extentV, 0., extentV},
                                                    {0., extentV, 0., 0.}}};
  for (const auto& edge : edges) {
    appendU32(bytes, 0); // curveType line
    appendU32(bytes, 4); // nCurveParams
    appendDoubles(bytes, {edge[0], edge[1], edge[2], edge[3]});
  }
}

// plane record (type 1) for a disk/annulus cap: one full-circle outer arc wire, plus a
// clockwise inner arc wire when holeRadius > 0. Exercises the arc-wire reader path.
void appendDiskPlaneRecord(std::vector<char>& bytes, const Point3D& center, const Point3D& axisU,
                           const Point3D& axisV, double radius, double holeRadius = 0.)
{
  appendU32(bytes, 1); // surfaceType plane
  appendU32(bytes, 0); // flags
  appendU32(bytes, 9); // nParams
  appendDoubles(bytes, {center[0], center[1], center[2], axisU[0], axisU[1], axisU[2], axisV[0], axisV[1], axisV[2]});
  const uint32_t nWires = holeRadius > 0. ? 2u : 1u;
  appendU32(bytes, nWires);
  appendU32(bytes, 0);                                        // outer wire role
  appendU32(bytes, 1);                                        // one edge
  appendU32(bytes, 1);                                        // curveType arc
  appendU32(bytes, 5);                                        // nCurveParams
  appendDoubles(bytes, {0., 0., radius, 0., 2. * surf::kPi}); // cu cv radius phiStart phiSweep (CCW full circle)
  if (holeRadius > 0.) {
    appendU32(bytes, 1); // inner wire role
    appendU32(bytes, 1);
    appendU32(bytes, 1); // arc
    appendU32(bytes, 5);
    appendDoubles(bytes, {0., 0., holeRadius, 0., -2. * surf::kPi}); // clockwise hole
  }
}

std::filesystem::path writeSidecarFile(const std::string& name, const std::vector<char>& bytes)
{
  const auto path = std::filesystem::temp_directory_path() / name;
  std::ofstream out(path, std::ios::binary);
  out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  BOOST_REQUIRE(out.good());
  return path;
}
} // namespace

BOOST_AUTO_TEST_CASE(SurfaceSidecarRoundTrip)
{
  // planar box: six plane records with polygon wires, loaded and compared against TGeoBBox
  constexpr double halfX = 1.;
  constexpr double halfY = 2.;
  constexpr double halfZ = 3.;

  std::vector<char> boxBytes;
  appendSidecarHeader(boxBytes, 6);
  for (int faceIndex = 0; faceIndex < 6; ++faceIndex) {
    appendPlaneRecord(boxBytes, boxFaceFrame(faceIndex, halfX, halfY, halfZ));
  }
  const auto boxPath = writeSidecarFile("o2_sidecar_roundtrip_box.bin", boxBytes);

  SurfaceSolid box("sidecarBox");
  BOOST_REQUIRE(o2::cad::LoadSurfaceSolid(boxPath.string(), box));
  std::filesystem::remove(boxPath);
  BOOST_CHECK_EQUAL(box.GetNsurfaces(), 6);
  box.CloseShape();
  BOOST_CHECK(box.IsClosed());
  BOOST_CHECK(box.IsOrientationConsistent());

  TGeoBBox referenceBox("referenceBox", halfX, halfY, halfZ);
  compareContainsGrid(box, referenceBox, 4., 7);
  compareDistance(box, referenceBox, {5., 0.5, 0.5}, {-1., 0., 0.});
  compareDistance(box, referenceBox, {0., 0., 0.}, unitDirection(1., 1., 1.));
  checkClose(box.Capacity(), referenceBox.Capacity(), 1.e-9);

  // quadric + arc-wire caps: closed cylinder (lateral wall + two disk caps) against TGeoTube
  constexpr double radius = 2.;
  constexpr double halfHeight = 3.;

  std::vector<char> tubeBytes;
  appendSidecarHeader(tubeBytes, 3);
  appendU32(tubeBytes, 2);  // surfaceType cylinder
  appendU32(tubeBytes, 0);  // flags (outer wall)
  appendU32(tubeBytes, 14); // nParams
  appendDoubles(tubeBytes, {0., 0., 0., 0., 0., 1., 1., 0., 0., radius, -halfHeight, halfHeight, 0., 2. * surf::kPi});
  appendU32(tubeBytes, 0); // nWires
  // caps as arc-wire plane records: outward normal is axisU x axisV, so the bottom cap flips axisV
  appendDiskPlaneRecord(tubeBytes, {0., 0., halfHeight}, {1., 0., 0.}, {0., 1., 0.}, radius);
  appendDiskPlaneRecord(tubeBytes, {0., 0., -halfHeight}, {1., 0., 0.}, {0., -1., 0.}, radius);
  const auto tubePath = writeSidecarFile("o2_sidecar_roundtrip_tube.bin", tubeBytes);

  SurfaceSolid tube("sidecarTube");
  BOOST_REQUIRE(o2::cad::LoadSurfaceSolid(tubePath.string(), tube));
  std::filesystem::remove(tubePath);
  BOOST_CHECK_EQUAL(tube.GetNsurfaces(), 3);
  tube.CloseShape();
  BOOST_CHECK(tube.IsClosed());
  BOOST_CHECK(tube.IsOrientationConsistent());

  TGeoTube referenceTube("referenceTube", 0., radius, halfHeight);
  compareContainsGrid(tube, referenceTube, 4., 7);
  compareDistance(tube, referenceTube, {5., 0.5, 1.}, {-1., 0., 0.});
  compareDistance(tube, referenceTube, {0., 0., 0.}, unitDirection(1., 1., 1.));
  checkClose(tube.Capacity(), referenceTube.Capacity(), 1.e-9);

  // malformed input must be rejected without loading surfaces
  const auto badPath = writeSidecarFile("o2_sidecar_bad_magic.bin", {'X', 'X', 'X', 'X', 0, 0, 0, 0});
  SurfaceSolid bad("sidecarBad");
  BOOST_CHECK(!o2::cad::LoadSurfaceSolid(badPath.string(), bad));
  std::filesystem::remove(badPath);
  BOOST_CHECK_EQUAL(bad.GetNsurfaces(), 0);
  BOOST_CHECK(!o2::cad::LoadSurfaceSolid("/nonexistent/o2_sidecar_missing.bin", bad));

  // truncated file: valid header announcing a surface that never follows
  std::vector<char> truncatedBytes;
  appendSidecarHeader(truncatedBytes, 1);
  const auto truncatedPath = writeSidecarFile("o2_sidecar_truncated.bin", truncatedBytes);
  SurfaceSolid truncated("sidecarTruncated");
  BOOST_CHECK(!o2::cad::LoadSurfaceSolid(truncatedPath.string(), truncated));
  std::filesystem::remove(truncatedPath);
}

// Sidecar version 2 carries the source model's own tolerance, so the kernel stops guessing what
// epsilon two faces of an imported solid should agree to. Both versions must load: a v1 file is a
// v2 file that simply does not state one.
BOOST_AUTO_TEST_CASE(SidecarModelToleranceRoundTrip)
{
  constexpr double halfX = 1.;
  constexpr double halfY = 2.;
  constexpr double halfZ = 3.;
  const auto boxBytesWithHeader = [&](uint32_t version, double modelTolerance) {
    std::vector<char> bytes;
    appendSidecarHeader(bytes, 6, version, modelTolerance);
    for (int faceIndex = 0; faceIndex < 6; ++faceIndex) {
      appendPlaneRecord(bytes, boxFaceFrame(faceIndex, halfX, halfY, halfZ));
    }
    return bytes;
  };
  const auto loadFrom = [](const char* name, const std::vector<char>& bytes, SurfaceSolid& solid) {
    const auto path = writeSidecarFile(name, bytes);
    const bool ok = o2::cad::LoadSurfaceSolid(path.string(), solid);
    std::filesystem::remove(path);
    return ok;
  };

  // version 2: the written tolerance reaches the solid untouched, and survives to CloseShape
  SurfaceSolid v2("sidecarV2");
  BOOST_REQUIRE(loadFrom("o2_sidecar_v2.bin", boxBytesWithHeader(2, 3.5e-5), v2));
  BOOST_CHECK_EQUAL(v2.GetNsurfaces(), 6);
  checkClose(v2.GetModelTolerance(), 3.5e-5, 1.e-18);
  v2.CloseShape();
  checkClose(v2.GetModelTolerance(), 3.5e-5, 1.e-18);

  // a v2 file may still state nothing, and "nothing" is zero rather than an invented number
  SurfaceSolid v2Silent("sidecarV2Silent");
  BOOST_REQUIRE(loadFrom("o2_sidecar_v2_silent.bin", boxBytesWithHeader(2, 0.), v2Silent));
  BOOST_CHECK_EQUAL(v2Silent.GetModelTolerance(), 0.);

  // version 1: still loads, and gets the reader's documented fallback rather than zero
  SurfaceSolid v1("sidecarV1");
  BOOST_REQUIRE(loadFrom("o2_sidecar_v1.bin", boxBytesWithHeader(1, 0.), v1));
  BOOST_CHECK_EQUAL(v1.GetNsurfaces(), 6);
  checkClose(v1.GetModelTolerance(), 1.e-6, 1.e-18);

  // a solid nobody told anything keeps zero: "not stated" is not the same as "the fallback"
  SurfaceSolid handBuilt("handBuilt");
  BOOST_CHECK_EQUAL(handBuilt.GetModelTolerance(), 0.);
  handBuilt.SetModelTolerance(1.e-4);
  checkClose(handBuilt.GetModelTolerance(), 1.e-4, 1.e-18);
  handBuilt.SetModelTolerance(-1.); // refused, and the previous value stands
  checkClose(handBuilt.GetModelTolerance(), 1.e-4, 1.e-18);

  // version 3 is understood now (it is a version-2 file that also states its edge identities);
  // it is exercised in SidecarV3EdgeIdentityRoundTrip below.

  // an unknown version is refused rather than reinterpreted
  SurfaceSolid v4("sidecarV4");
  BOOST_CHECK(!loadFrom("o2_sidecar_v4.bin", boxBytesWithHeader(4, 1.e-5), v4));
  BOOST_CHECK_EQUAL(v4.GetNsurfaces(), 0);

  // and a v2 header that stops before its tolerance is a truncated file, not a v1 one
  std::vector<char> stump;
  stump.insert(stump.end(), {'O', '2', 'S', 'S'});
  appendU32(stump, 2);
  appendU32(stump, 6);
  appendU32(stump, 0);
  SurfaceSolid stumped("sidecarV2Stump");
  BOOST_CHECK(!loadFrom("o2_sidecar_v2_stump.bin", stump, stumped));
  BOOST_CHECK_EQUAL(stumped.GetNsurfaces(), 0);
}

BOOST_AUTO_TEST_CASE(WireTrimmedSidecarRoundTrip)
{
  // a cylinder record carrying a (line) trim wire block in its (phi, h) domain must load through
  // the wire-taking Add* overload and navigate like the equivalent scalar cylinder
  constexpr double radius = 2.;
  constexpr double halfHeight = 3.;

  std::vector<char> bytes;
  appendSidecarHeader(bytes, 3);
  appendU32(bytes, 2);  // surfaceType cylinder
  appendU32(bytes, 0);  // flags (outer wall)
  appendU32(bytes, 14); // nParams
  appendDoubles(bytes, {0., 0., 0., 0., 0., 1., 1., 0., 0., radius, -halfHeight, halfHeight, 0., 2. * surf::kPi});
  appendU32(bytes, 1); // nWires
  appendU32(bytes, 0); // outer wire role
  appendU32(bytes, 4); // nEdges
  const std::array<std::array<double, 4>, 4> edges{{{0., -halfHeight, 2. * surf::kPi, -halfHeight},
                                                    {2. * surf::kPi, -halfHeight, 2. * surf::kPi, halfHeight},
                                                    {2. * surf::kPi, halfHeight, 0., halfHeight},
                                                    {0., halfHeight, 0., -halfHeight}}};
  for (const auto& edge : edges) {
    appendU32(bytes, 0); // curveType line
    appendU32(bytes, 4); // nCurveParams
    appendDoubles(bytes, {edge[0], edge[1], edge[2], edge[3]});
  }
  appendDiskPlaneRecord(bytes, {0., 0., halfHeight}, {1., 0., 0.}, {0., 1., 0.}, radius);
  appendDiskPlaneRecord(bytes, {0., 0., -halfHeight}, {1., 0., 0.}, {0., -1., 0.}, radius);
  const auto path = writeSidecarFile("o2_sidecar_wiretrim_cylinder.bin", bytes);

  SurfaceSolid solid("sidecarWireCylinder");
  BOOST_REQUIRE(o2::cad::LoadSurfaceSolid(path.string(), solid));
  std::filesystem::remove(path);
  BOOST_CHECK_EQUAL(solid.GetNsurfaces(), 3);
  solid.CloseShape();
  BOOST_CHECK(solid.IsClosed());
  BOOST_CHECK(solid.IsOrientationConsistent());

  TGeoTube reference("wireTrimSidecarTube", 0., radius, halfHeight);
  compareContainsGrid(solid, reference, 4., 7);
  compareDistance(solid, reference, {5., 0.5, 1.}, {-1., 0., 0.});
  compareDistance(solid, reference, {0., 0., 0.}, unitDirection(1., 1., 1.));
  checkClose(solid.Capacity(), reference.Capacity(), 1.e-6);
}

BOOST_AUTO_TEST_CASE(TorusSidecarRoundTrip)
{
  // a full-torus record (surfaceType 5, 15 params, empty wire block) must load through the
  // scalar AddToroidalSurface path and navigate like TGeoTorus
  constexpr double majorR = 3.;
  constexpr double minorR = 1.;

  std::vector<char> bytes;
  appendSidecarHeader(bytes, 1);
  appendU32(bytes, 5);  // surfaceType torus
  appendU32(bytes, 0);  // flags (outer wall)
  appendU32(bytes, 15); // nParams
  appendDoubles(bytes, {0., 0., 0., 0., 0., 1., 1., 0., 0., majorR, minorR, 0., 2. * surf::kPi, 0., 2. * surf::kPi});
  appendU32(bytes, 0); // nWires (full torus: scalar path)
  const auto path = writeSidecarFile("o2_sidecar_roundtrip_torus.bin", bytes);

  SurfaceSolid solid("sidecarTorus");
  BOOST_REQUIRE(o2::cad::LoadSurfaceSolid(path.string(), solid));
  std::filesystem::remove(path);
  BOOST_CHECK_EQUAL(solid.GetNsurfaces(), 1);
  solid.CloseShape();
  BOOST_CHECK(solid.IsClosed());
  BOOST_CHECK(solid.IsOrientationConsistent());

  TGeoTorus reference("sidecarTorusRef", majorR, 0., minorR);
  compareContainsGrid(solid, reference, 4.5, 9);
  checkClose(solid.Capacity(), reference.Capacity(), 1.e-7);
}

// Wire-join gaps are judged against the tolerance the sidecar itself declares (the
// version-2 model tolerance), with the extractor-precision constant as the floor -- not against
// the bare constant when the model states it cannot do better. This is the ST1829909_01
// rejection: surface 1006's bspline->line join gaps by 5.41e-6 cm on a model that declares
// 4.7e-4 cm, and the 1e-6 constant "is a fallback, not a measurement of the model". The band
// must hold in the loader *and* in the kernel's own wire construction, or the loader would
// accept a wire that Add*Surface rejects moments later.
BOOST_AUTO_TEST_CASE(StreamY_LoaderHonoursTheDeclaredModelTolerance)
{
  constexpr double radius = 2.;
  constexpr double halfHeight = 3.;
  // one seam offset purely in v (cm), so the 3D gap equals the offset on any cylinder:
  // over the 1e-6 cm extractor floor, under the model tolerance the loading case declares
  constexpr double joinGap = 5.e-6;

  const auto cylinderBytes = [&](uint32_t version, double modelTolerance) {
    std::vector<char> bytes;
    appendSidecarHeader(bytes, 3, version, modelTolerance);
    appendU32(bytes, 2);  // surfaceType cylinder
    appendU32(bytes, 0);  // flags (outer wall)
    appendU32(bytes, 14); // nParams
    appendDoubles(bytes, {0., 0., 0., 0., 0., 1., 1., 0., 0., radius, -halfHeight, halfHeight, 0., 2. * surf::kPi});
    appendU32(bytes, 1); // nWires
    appendU32(bytes, 0); // outer wire role
    appendU32(bytes, 5); // nEdges
    // The bottom edge is split in two and the second half starts joinGap off the first half's
    // end -- a mid-wire join like surface 1006's bspline->line seam. Deliberately *not* at the
    // phi-wrap corner: the full-turn seam pair (u = 0 vs u = 2*pi) must stay exactly coincident
    // to cancel in the rim chaining, and the real rejection's face spans only half a turn.
    const std::array<std::array<double, 4>, 5> edges{
      {{0., -halfHeight, surf::kPi, -halfHeight},
       {surf::kPi, -halfHeight + joinGap, 2. * surf::kPi, -halfHeight}, // starts joinGap off edge 0's end
       {2. * surf::kPi, -halfHeight, 2. * surf::kPi, halfHeight},
       {2. * surf::kPi, halfHeight, 0., halfHeight},
       {0., halfHeight, 0., -halfHeight}}};
    for (const auto& edge : edges) {
      appendU32(bytes, 0); // curveType line
      appendU32(bytes, 4); // nCurveParams
      appendDoubles(bytes, {edge[0], edge[1], edge[2], edge[3]});
    }
    appendDiskPlaneRecord(bytes, {0., 0., halfHeight}, {1., 0., 0.}, {0., 1., 0.}, radius);
    appendDiskPlaneRecord(bytes, {0., 0., -halfHeight}, {1., 0., 0.}, {0., -1., 0.}, radius);
    return bytes;
  };
  const auto loadFrom = [](const char* name, const std::vector<char>& bytes, SurfaceSolid& solid) {
    const auto path = writeSidecarFile(name, bytes);
    const bool ok = o2::cad::LoadSurfaceSolid(path.string(), solid);
    std::filesystem::remove(path);
    return ok;
  };

  // a model declaring 1e-4 cm: the 5e-6 cm seam is within the model's own statement, so it loads,
  // closes, and navigates like the gap-free tube (the kernel canonicalizes each seam on accept)
  SurfaceSolid declared("sidecarJoinDeclared");
  BOOST_REQUIRE(loadFrom("o2_sidecar_join_declared.bin", cylinderBytes(2, 1.e-4), declared));
  BOOST_CHECK_EQUAL(declared.GetNsurfaces(), 3);
  declared.CloseShape();
  BOOST_CHECK(declared.IsClosed());
  BOOST_CHECK(declared.IsOrientationConsistent());
  TGeoTube reference("declaredToleranceTube", 0., radius, halfHeight);
  compareContainsGrid(declared, reference, 4., 7);
  compareDistance(declared, reference, {5., 0.5, 1.}, {-1., 0., 0.});

  // a v1 file states nothing, so the extractor-precision floor stands and the same seam is open
  SurfaceSolid silent("sidecarJoinSilent");
  BOOST_CHECK(!loadFrom("o2_sidecar_join_silent.bin", cylinderBytes(1, 0.), silent));
  BOOST_CHECK_EQUAL(silent.GetNsurfaces(), 0);

  // a declared tolerance below the gap does not save it: the model itself calls the seam open
  SurfaceSolid tight("sidecarJoinTight");
  BOOST_CHECK(!loadFrom("o2_sidecar_join_tight.bin", cylinderBytes(2, 2.e-6), tight));
  BOOST_CHECK_EQUAL(tight.GetNsurfaces(), 0);
}

namespace
{
// Append a plane record whose rectangular outer wire has its bottom edge as a degree-3 B-spline
// with collinear poles — geometrically identical to the straight edge, so the box still closes,
// but it exercises the whole B-spline sidecar pipeline (curveType 2 reader -> kernel).
void appendBSplineEdgePlaneRecord(std::vector<char>& bytes, const FaceFrame& frame)
{
  appendU32(bytes, 1); // surfaceType plane
  appendU32(bytes, 0); // flags
  appendU32(bytes, 9); // nParams
  appendDoubles(bytes, {frame.origin[0], frame.origin[1], frame.origin[2], frame.axisU[0], frame.axisU[1],
                        frame.axisU[2], frame.axisV[0], frame.axisV[1], frame.axisV[2]});
  appendU32(bytes, 1); // nWires
  appendU32(bytes, 0); // wireRole outer
  appendU32(bytes, 4); // nEdges
  const double extentU = frame.extentU;
  const double extentV = frame.extentV;
  // edge 0: collinear cubic B-spline from (0, 0) to (extentU, 0)
  appendU32(bytes, 2);                                                                // curveType bspline
  appendU32(bytes, 22);                                                               // nCurveParams = 2 + 2*4 + 4 + 8
  appendDoubles(bytes, {3., 4.,                                                       // degree, nPoles
                        0., 0., extentU / 3., 0., 2. * extentU / 3., 0., extentU, 0., // poles
                        1., 1., 1., 1.,                                               // weights
                        0., 0., 0., 0., 1., 1., 1., 1.});                             // clamped knots
  const std::array<std::array<double, 4>, 3> lines{
    {{extentU, 0., extentU, extentV}, {extentU, extentV, 0., extentV}, {0., extentV, 0., 0.}}};
  for (const auto& edge : lines) {
    appendU32(bytes, 0); // curveType line
    appendU32(bytes, 4); // nCurveParams
    appendDoubles(bytes, {edge[0], edge[1], edge[2], edge[3]});
  }
}

// A rational quadratic B-spline (NURBS) quarter circle from angle a0 to a0 + pi/2, in the (u, v)
// domain, centred at (cu, cv) with radius r. Four of these form an exact circle.
/// A full circle as ONE closed rational B-spline (the standard 9-pole degree-2 NURBS circle),
/// i.e. a wire whose single edge starts and ends at the same point. This is the shape a CAD
/// kernel writes for a tube-tube intersection curve, and it is structurally different from the
/// same circle spelled as four separate quarter arcs.
surf::Curve2D fullCircleBSpline(double cu, double cv, double r)
{
  const double w = std::sqrt(0.5);
  const std::vector<surf::Vec2> poles{{cu + r, cv}, {cu + r, cv + r}, {cu, cv + r}, {cu - r, cv + r}, {cu - r, cv}, {cu - r, cv - r}, {cu, cv - r}, {cu + r, cv - r}, {cu + r, cv}};
  return surf::Curve2D::makeBSpline(2, poles, {1., w, 1., w, 1., w, 1., w, 1.},
                                    {0., 0., 0., 1., 1., 2., 2., 3., 3., 4., 4., 4.});
}

surf::Curve2D quarterCircleBSpline(double cu, double cv, double r, double a0)
{
  const double a1 = a0 + surf::kHalfPi;
  const double aMid = 0.5 * (a0 + a1);
  const std::vector<surf::Vec2> poles{{cu + r * std::cos(a0), cv + r * std::sin(a0)},
                                      {cu + r * std::sqrt(2.) * std::cos(aMid), cv + r * std::sqrt(2.) * std::sin(aMid)},
                                      {cu + r * std::cos(a1), cv + r * std::sin(a1)}};
  return surf::Curve2D::makeBSpline(2, poles, {1., std::sqrt(0.5), 1.}, {0., 0., 0., 1., 1., 1.});
}
} // namespace

// K5: the on-boundary band has to be as wide as the representation it measures against, and
// winding and distance have to measure against the same polyline.
BOOST_AUTO_TEST_CASE(BoundaryBandMatchesTheRepresentation)
{
  using surf::Vec2;
  using surf::WireClassification;
  using surf::WireRole;
  using surf::WireStatus;
  std::string error;

  // A loop of lines and arcs is held exactly and claims no width of its own...
  surf::CurveWire exactWire;
  WireStatus status = WireStatus::Valid;
  BOOST_REQUIRE(exactWire.initialize({surf::Curve2D::makeCircle({0., 0.}, 1.)}, WireRole::Outer, status));
  BOOST_CHECK_EQUAL(exactWire.representationTolerance(), 0.);

  // ...while a B-spline loop is only as good as the polyline it is flattened to.
  surf::CurveWire splineWire;
  BOOST_REQUIRE(splineWire.initialize({fullCircleBSpline(0., 0., 1.)}, WireRole::Outer, status));
  checkClose(splineWire.representationTolerance(), surf::kBSplineFlatness, 1.e-18);

  // The Boundary state must therefore be reachable for a B-spline trim. It was not: a 1e-9 band
  // around a 1e-5 polyline is noise, so a point this close to the curve used to come back Inside
  // or Outside by coin flip.
  const double justInsideTheBand = 0.5 * surf::kBSplineFlatness;
  BOOST_CHECK(splineWire.classify({1. - justInsideTheBand, 0.}) == WireClassification::Boundary);
  BOOST_CHECK(splineWire.classify({1. + justInsideTheBand, 0.}) == WireClassification::Boundary);
  // and well outside the band the answer is decided again, in both directions
  BOOST_CHECK(splineWire.classify({0.5, 0.}) == WireClassification::Inside);
  BOOST_CHECK(splineWire.classify({2.0, 0.}) == WireClassification::Outside);
  // the exact loop keeps its narrow band: the same offset is decidable there
  BOOST_CHECK(exactWire.classify({1. - justInsideTheBand, 0.}) == WireClassification::Inside);

  // The band is a length, so on a surface that stretches the domain it narrows in parametric
  // terms. A 100 cm cylinder resolves 1e-9 cm at 1e-11 rad, not at 1e-9 rad.
  surf::CylindricalBoundedSurface bigCylinder;
  BOOST_REQUIRE(bigCylinder.initialize({0., 0., 0.}, {0., 0., 1.}, {1., 0., 0.}, 100., -1., 1., 0., surf::kTwoPi,
                                       false, error));
  const auto bigMetric = surf::parametricMetricOf(bigCylinder);
  surf::CurveWire squareWire;
  BOOST_REQUIRE(squareWire.initialize({surf::Curve2D::makeLine({0., -1.}, {1., -1.}),
                                       surf::Curve2D::makeLine({1., -1.}, {1., 1.}),
                                       surf::Curve2D::makeLine({1., 1.}, {0., 1.}),
                                       surf::Curve2D::makeLine({0., 1.}, {0., -1.})},
                                      WireRole::Outer, status, bigMetric));
  // 1e-10 rad is 1e-8 cm on this cylinder -- outside a 1e-9 cm band, so the point is decidable
  BOOST_CHECK(squareWire.classify({0.5, 1. - 1.e-10}, bigMetric) == WireClassification::Inside);
  // 1e-12 rad is 1e-10 cm, inside it
  BOOST_CHECK(squareWire.classify({0.5, 1. - 1.e-12}, bigMetric) == WireClassification::Boundary);

  // One polyline: a wire fixes one vertex value per seam, and both the winding polyline and the
  // point-to-curve distance are built from it. Give two quarter arcs endpoints that differ within
  // the join tolerance and the wire must still agree with itself about where its boundary is.
  const double seamDrift = 4.e-7;
  surf::Curve2D first = quarterCircleBSpline(0., 0., 1., 0.);
  surf::Curve2D second = quarterCircleBSpline(0., 0., 1., surf::kHalfPi);
  surf::Curve2D third = quarterCircleBSpline(0., 0., 1., surf::kPi);
  surf::Curve2D fourth = quarterCircleBSpline(0., 0., 1., 3. * surf::kHalfPi);
  second.poles.front() = {second.poles.front().uCoord + seamDrift, second.poles.front().vCoord};
  surf::CurveWire driftedWire;
  BOOST_REQUIRE(driftedWire.initialize({first, second, third, fourth}, WireRole::Outer, status));
  for (const auto& curve : driftedWire.curves) {
    BOOST_CHECK(curve.hasCanonicalEndpoints);
  }
  // every curve now begins exactly where its predecessor ends -- there is one boundary, not two
  for (size_t index = 0; index < driftedWire.curves.size(); ++index) {
    const Vec2 thisEnd = driftedWire.curves[index].loopEnd();
    const Vec2 nextStart = driftedWire.curves[(index + 1) % driftedWire.curves.size()].loopStart();
    BOOST_CHECK_EQUAL(thisEnd.uCoord, nextStart.uCoord);
    BOOST_CHECK_EQUAL(thisEnd.vCoord, nextStart.vCoord);
  }
  // and the polyline the winding walks is the one the distance measures against
  for (const auto& curve : driftedWire.curves) {
    const auto& polyline = curve.bsplineSamples();
    BOOST_REQUIRE(polyline.size() >= 2);
    BOOST_CHECK_EQUAL(polyline.front().uCoord, curve.loopStart().uCoord);
    BOOST_CHECK_EQUAL(polyline.front().vCoord, curve.loopStart().vCoord);
    BOOST_CHECK_EQUAL(polyline.back().uCoord, curve.loopEnd().uCoord);
    BOOST_CHECK_EQUAL(polyline.back().vCoord, curve.loopEnd().vCoord);
  }
  BOOST_CHECK(driftedWire.classify({0., 0.}) == WireClassification::Inside);
  BOOST_CHECK(driftedWire.classify({3., 0.}) == WireClassification::Outside);
}

// The other half of the on-boundary band check.
//
// BoundaryBandMatchesTheRepresentation above pins the *width* of the band. This pins what happens
// to a ray that lands in it. Resolving Boundary as "inside the trim" is a tie-break, not a fact,
// and it is one-sided: the patch keeps a sliver of the band's width past its true trim curve. On a
// Boolean seam that sliver lies in the solid's interior, where a crossing must not be counted, so
// a ray through it gains a spurious crossing and Contains() flips.
//
// Measured on cyl_cross_cyl (two unit cylinders fused, whose seam is transcendental in either
// face's chart, so it has to be carried as a B-spline): every one of 1440 sampled positions along
// the true seam overhangs by 1.0e-5 to 1.9e-5 cm and *none* undercuts -- the floor being the band
// itself and the excess the polyline flattening. That is the single direction-dependent point the
// section 4.2 sweep found, and it is not the root-finding defect (K6) it was filed as.
//
// The kernel cannot remove the sliver -- the data does not say where the seam is to better than
// this -- so it labels it instead, and Contains() re-aims when a shot rests on one. Hence the
// contract here: the flag is set exactly when the answer came from the tie-break, and is *not*
// set for a hit the trim decides on its own, because a flag that fired everywhere would put every
// query on the voting path.
BOOST_AUTO_TEST_CASE(TrimBoundaryHitsAreFlaggedAsAmbiguous)
{
  using surf::Curve2D;
  std::string error;

  // a cylinder of radius 2 carrying a circular B-spline window of radius 0.5 in (phi, h)
  const double centrePhi = surf::kPi;
  const double trimRadius = 0.5;
  surf::CylindricalBoundedSurface splineTrimmed;
  BOOST_REQUIRE(splineTrimmed.initialize({0., 0., 0.}, {0., 0., 1.}, {1., 0., 0.}, 2., -1., 1., 0., surf::kTwoPi,
                                         false,
                                         {quarterCircleBSpline(centrePhi, 0., trimRadius, 0.),
                                          quarterCircleBSpline(centrePhi, 0., trimRadius, surf::kHalfPi),
                                          quarterCircleBSpline(centrePhi, 0., trimRadius, surf::kPi),
                                          quarterCircleBSpline(centrePhi, 0., trimRadius, 3. * surf::kHalfPi)},
                                         {}, error));

  // the same window held exactly, as one arc: it claims no width, so nothing is ever ambiguous
  surf::CylindricalBoundedSurface arcTrimmed;
  BOOST_REQUIRE(arcTrimmed.initialize({0., 0., 0.}, {0., 0., 1.}, {1., 0., 0.}, 2., -1., 1., 0., surf::kTwoPi, false,
                                      {Curve2D::makeCircle({centrePhi, 0.}, trimRadius)}, {}, error));

  // a radial ray that meets the wall at azimuth phi, h = 0
  const auto hitAt = [](const surf::CylindricalBoundedSurface& surface, double phi) {
    std::vector<surf::RayHit> hits;
    surface.appendIntersections({0., 0., 0.}, {std::cos(phi), std::sin(phi), 0.}, 0., 1.e30, hits);
    return hits;
  };

  // The band on this surface is the representation's own tolerance: 1e-5 in (phi, h), since the
  // length floor kTolerance / maxScale is 1e-9 / 2 and loses.
  const double band = surf::kBSplineFlatness;
  const double justInside = 0.5 * band;

  // 1. well inside the window the trim decides by itself -- accepted, and NOT flagged
  {
    const auto hits = hitAt(splineTrimmed, centrePhi);
    BOOST_REQUIRE_EQUAL(hits.size(), 1u);
    BOOST_CHECK(!hits.front().onTrimBoundary);
  }
  // 2. inside the window but within the band of its edge -- accepted, and flagged
  {
    const auto hits = hitAt(splineTrimmed, centrePhi + trimRadius - justInside);
    BOOST_REQUIRE_EQUAL(hits.size(), 1u);
    BOOST_CHECK(hits.front().onTrimBoundary);
  }
  // 3. OUTSIDE the window, still within the band -- accepted anyway, and flagged. This is the
  //    sliver: the tie-break keeps material the trim curve does not enclose.
  {
    const auto hits = hitAt(splineTrimmed, centrePhi + trimRadius + justInside);
    BOOST_REQUIRE_EQUAL(hits.size(), 1u);
    BOOST_CHECK(hits.front().onTrimBoundary);
  }
  // 4. beyond the band the patch really does end
  {
    BOOST_CHECK(hitAt(splineTrimmed, centrePhi + trimRadius + 100. * band).empty());
  }
  // 5. an exactly-held trim has no sliver to label: the same offsets are decided, not flagged
  {
    const auto inside = hitAt(arcTrimmed, centrePhi + trimRadius - justInside);
    BOOST_REQUIRE_EQUAL(inside.size(), 1u);
    BOOST_CHECK(!inside.front().onTrimBoundary);
    BOOST_CHECK(hitAt(arcTrimmed, centrePhi + trimRadius + justInside).empty());
  }
  // 6. and an untrimmed patch never sets it, which is what keeps the fast path fast
  {
    surf::CylindricalBoundedSurface plain;
    BOOST_REQUIRE(plain.initialize({0., 0., 0.}, {0., 0., 1.}, {1., 0., 0.}, 2., -1., 1., 0., surf::kTwoPi, false,
                                   error));
    const auto hits = hitAt(plain, centrePhi);
    BOOST_REQUIRE_EQUAL(hits.size(), 1u);
    BOOST_CHECK(!hits.front().onTrimBoundary);
  }
}

BOOST_AUTO_TEST_CASE(BSplineSidecarRoundTrip)
{
  // a closed box whose first face carries a (collinear) B-spline boundary edge round-trips through
  // the sidecar reader and navigates identically to TGeoBBox
  constexpr double halfX = 1.;
  constexpr double halfY = 2.;
  constexpr double halfZ = 3.;

  std::vector<char> bytes;
  appendSidecarHeader(bytes, 6);
  appendBSplineEdgePlaneRecord(bytes, boxFaceFrame(0, halfX, halfY, halfZ));
  for (int faceIndex = 1; faceIndex < 6; ++faceIndex) {
    appendPlaneRecord(bytes, boxFaceFrame(faceIndex, halfX, halfY, halfZ));
  }
  const auto path = writeSidecarFile("o2_sidecar_bspline_box.bin", bytes);

  SurfaceSolid box("sidecarBSplineBox");
  BOOST_REQUIRE(o2::cad::LoadSurfaceSolid(path.string(), box));
  std::filesystem::remove(path);
  BOOST_CHECK_EQUAL(box.GetNsurfaces(), 6);
  box.CloseShape();
  BOOST_CHECK(box.IsClosed());
  BOOST_CHECK(box.IsOrientationConsistent());

  TGeoBBox reference("bsplineBoxRef", halfX, halfY, halfZ);
  compareContainsGrid(box, reference, 4., 7);
  compareDistance(box, reference, {5., 0.5, 0.5}, {-1., 0., 0.});
  compareDistance(box, reference, {0., 0., 0.}, unitDirection(1., 1., 1.));
  checkClose(box.Capacity(), reference.Capacity(), 1.e-6);
}

BOOST_AUTO_TEST_CASE(BSplineWindowInCylinderWall)
{
  using surf::Curve2D;
  using surf::Vec3;
  std::string error;

  const auto onCylinder = [](double phi, double height) {
    return Vec3{2. * std::cos(phi), 2. * std::sin(phi), height};
  };

  // an exact circular trim in (phi, h) built from four NURBS quarter arcs must classify identically
  // to the same circle expressed as one exact arc Curve2D — validating the B-spline trim path on a
  // quadric against the closed-form arc path.
  const double centrePhi = surf::kPi;
  const double trimRadius = 0.5;
  surf::CylindricalBoundedSurface bsplineDisk;
  const std::vector<Curve2D> bsplineOuter{quarterCircleBSpline(centrePhi, 0., trimRadius, 0.),
                                          quarterCircleBSpline(centrePhi, 0., trimRadius, surf::kHalfPi),
                                          quarterCircleBSpline(centrePhi, 0., trimRadius, surf::kPi),
                                          quarterCircleBSpline(centrePhi, 0., trimRadius, 3. * surf::kHalfPi)};
  BOOST_REQUIRE(bsplineDisk.initialize({0., 0., 0.}, {0., 0., 1.}, {1., 0., 0.}, 2., -1., 1., 0., surf::kTwoPi, false,
                                       bsplineOuter, {}, error));
  BOOST_CHECK(!bsplineDisk.capacityIsExact()); // B-spline (wire) trim -> numeric capacity

  surf::CylindricalBoundedSurface arcDisk;
  BOOST_REQUIRE(arcDisk.initialize({0., 0., 0.}, {0., 0., 1.}, {1., 0., 0.}, 2., -1., 1., 0., surf::kTwoPi, false,
                                   {Curve2D::makeCircle({centrePhi, 0.}, trimRadius)}, {}, error));

  // classification agrees across a grid of the (phi, h) neighbourhood of the trim (skip a thin band
  // around the boundary, where the exact-arc and sampled-B-spline classifications can legitimately
  // differ by the sampling tolerance)
  int compared = 0;
  for (int phiStep = -12; phiStep <= 12; ++phiStep) {
    const double phi = centrePhi + 0.09 * phiStep;
    for (int hStep = -12; hStep <= 12; ++hStep) {
      const double height = 0.09 * hStep;
      // radial distance in (phi, h) from the trim centre; skip the boundary band
      const double distToCentre = std::hypot(phi - centrePhi, height);
      if (std::abs(distToCentre - trimRadius) < 5.e-3) {
        continue;
      }
      const Vec3 point = onCylinder(phi, height);
      BOOST_CHECK_EQUAL(bsplineDisk.containsPointOnSurface(point), arcDisk.containsPointOnSurface(point));
      ++compared;
    }
  }
  BOOST_CHECK_GT(compared, 100);

  // a radial ray into the B-spline window (its centre) registers exactly one wall hit; a ray well
  // outside the window (opposite side of the cylinder) misses the trimmed patch
  std::vector<surf::RayHit> hits;
  bsplineDisk.appendIntersections({0., 0., 0.}, {std::cos(centrePhi), std::sin(centrePhi), 0.}, 0., 1.e30, hits);
  BOOST_REQUIRE_EQUAL(hits.size(), 1u);
  checkClose(hits.front().distance, 2.);
  hits.clear();
  bsplineDisk.appendIntersections({0., 0., 0.}, {std::cos(0.), std::sin(0.), 0.}, 0., 1.e30, hits);
  BOOST_CHECK(hits.empty());
}

// A B-spline wire used as an *inner hole*, and in particular one spelled as a single CLOSED
// B-spline edge. This is what a tube-tube intersection produces and what the converter emits
// constantly: where a boom tube is planted on a fat tube, the fat tube's wall keeps a
// full-rectangle outer wire and carries the intersection curve as one closed B-spline hole.
//
// It regresses a bug that silently deleted such a wire outright. bsplineSampleRecursive used to
// end the recursion when the chord p0->p1 was shorter than the flatness scale; a closed curve has
// p0 == p1 exactly, so a full circle flattened to two coincident points and every polyline-based
// query (winding, closest point, boundary band, display mesh) saw an empty curve. The wire still
// validated and still reported the correct enclosed area, because signedAreaContribution
// integrates the curve by Gauss-Legendre rather than from the polyline -- which is exactly why
// this survived: every check that could have caught it used the analytic path.
//
// Impact: on ExcavatorArm/BoomCylinderOuter_0_1_1_9 a point 0.026 cm inside such a hole was reported as
// lying on the face, and a whole face whose outer wire was one closed B-spline did not exist at
// all. `WireTrimmedQuadricKernels` covers a *line* hole and `BSplineWindowInCylinderWall` covers a
// B-spline outer wire built from four *open* quarter arcs, so neither could see it.
BOOST_AUTO_TEST_CASE(BSplineHoleInCylinderWall)
{
  using surf::Curve2D;
  using surf::Vec3;
  std::string error;

  const auto onCylinder = [](double phi, double height) {
    return Vec3{2. * std::cos(phi), 2. * std::sin(phi), height};
  };

  // full-sweep outer wire (what the converter writes for an untrimmed cylinder wall) ...
  const std::vector<Curve2D> outer{Curve2D::makeLine({0., -3.}, {surf::kTwoPi, -3.}),
                                   Curve2D::makeLine({surf::kTwoPi, -3.}, {surf::kTwoPi, 3.}),
                                   Curve2D::makeLine({surf::kTwoPi, 3.}, {0., 3.}),
                                   Curve2D::makeLine({0., 3.}, {0., -3.})};
  // ... with a circular hole punched in it, expressed once as four NURBS quarter arcs and once as
  // the equivalent exact arc. The arc form is the oracle: it is the already-trusted path.
  const double centrePhi = surf::kPi;
  const double trimRadius = 0.5;
  const std::vector<Curve2D> bsplineHole{quarterCircleBSpline(centrePhi, 0., trimRadius, 0.),
                                         quarterCircleBSpline(centrePhi, 0., trimRadius, surf::kHalfPi),
                                         quarterCircleBSpline(centrePhi, 0., trimRadius, surf::kPi),
                                         quarterCircleBSpline(centrePhi, 0., trimRadius, 3. * surf::kHalfPi)};

  surf::CylindricalBoundedSurface bsplineHoled;
  BOOST_REQUIRE(bsplineHoled.initialize({0., 0., 0.}, {0., 0., 1.}, {1., 0., 0.}, 2., -3., 3., 0., surf::kTwoPi, false,
                                        outer, {bsplineHole}, error));
  surf::CylindricalBoundedSurface arcHoled;
  BOOST_REQUIRE(arcHoled.initialize({0., 0., 0.}, {0., 0., 1.}, {1., 0., 0.}, 2., -3., 3., 0., surf::kTwoPi, false,
                                    outer, {{Curve2D::makeCircle({centrePhi, 0.}, trimRadius)}}, error));

  // the defining property of a hole: its interior is NOT part of the face
  BOOST_CHECK(!arcHoled.containsPointOnSurface(onCylinder(centrePhi, 0.)));     // oracle
  BOOST_CHECK(!bsplineHoled.containsPointOnSurface(onCylinder(centrePhi, 0.))); // the case under test
  // and material well away from the hole still is
  BOOST_CHECK(bsplineHoled.containsPointOnSurface(onCylinder(0.5, 0.)));
  BOOST_CHECK(bsplineHoled.containsPointOnSurface(onCylinder(centrePhi, 2.5)));

  // the two spellings of the same hole must classify identically away from the boundary band
  int compared = 0;
  for (int phiStep = -12; phiStep <= 12; ++phiStep) {
    const double phi = centrePhi + 0.09 * phiStep;
    for (int hStep = -12; hStep <= 12; ++hStep) {
      const double height = 0.09 * hStep;
      if (std::abs(std::hypot(phi - centrePhi, height) - trimRadius) < 5.e-3) {
        continue;
      }
      const Vec3 point = onCylinder(phi, height);
      BOOST_CHECK_EQUAL(bsplineHoled.containsPointOnSurface(point), arcHoled.containsPointOnSurface(point));
      ++compared;
    }
  }
  BOOST_CHECK_GT(compared, 100);

  // a radial ray aimed through the hole must not register a wall hit; one aimed at material must
  std::vector<surf::RayHit> hits;
  bsplineHoled.appendIntersections({0., 0., 0.}, {std::cos(centrePhi), std::sin(centrePhi), 0.}, 0., 1.e30, hits);
  BOOST_CHECK(hits.empty());
  hits.clear();
  bsplineHoled.appendIntersections({0., 0., 0.}, {std::cos(0.5), std::sin(0.5), 0.}, 0., 1.e30, hits);
  BOOST_CHECK_EQUAL(hits.size(), 1u);

  // The same hole as ONE closed B-spline edge rather than four arc segments. This is what the
  // converter actually emits for a tube-tube seam (`_quadric_trim_wire` writes one B-spline per
  // BREP edge, and the intersection curve is a single closed edge).
  surf::CylindricalBoundedSurface singleEdgeHoled;
  BOOST_REQUIRE(singleEdgeHoled.initialize({0., 0., 0.}, {0., 0., 1.}, {1., 0., 0.}, 2., -3., 3., 0., surf::kTwoPi,
                                           false, outer, {{fullCircleBSpline(centrePhi, 0., trimRadius)}}, error));
  BOOST_CHECK(!singleEdgeHoled.containsPointOnSurface(onCylinder(centrePhi, 0.)));
  BOOST_CHECK(singleEdgeHoled.containsPointOnSurface(onCylinder(0.5, 0.)));
  BOOST_CHECK(singleEdgeHoled.containsPointOnSurface(onCylinder(centrePhi, 2.5)));
  for (int phiStep = -12; phiStep <= 12; ++phiStep) {
    const double phi = centrePhi + 0.09 * phiStep;
    for (int hStep = -12; hStep <= 12; ++hStep) {
      const double height = 0.09 * hStep;
      if (std::abs(std::hypot(phi - centrePhi, height) - trimRadius) < 5.e-3) {
        continue;
      }
      const Vec3 point = onCylinder(phi, height);
      BOOST_CHECK_EQUAL(singleEdgeHoled.containsPointOnSurface(point), arcHoled.containsPointOnSurface(point));
    }
  }
}

namespace
{
// The public-API mirror of quarterCircleBSpline, for building a NURBS trim through Add*Surface.
BoundaryCurve quarterCircleBoundaryCurve(double cu, double cv, double r, double a0)
{
  const double a1 = a0 + surf::kHalfPi;
  const double aMid = 0.5 * (a0 + a1);
  const std::vector<Point2D> poles{{cu + r * std::cos(a0), cv + r * std::sin(a0)},
                                   {cu + r * std::sqrt(2.) * std::cos(aMid), cv + r * std::sqrt(2.) * std::sin(aMid)},
                                   {cu + r * std::cos(a1), cv + r * std::sin(a1)}};
  return BoundaryCurve::makeBSpline(2, poles, {1., std::sqrt(0.5), 1.}, {0., 0., 0., 1., 1., 1.});
}

// Assert that two solids are the *same* solid, not merely similar ones: identical closure
// diagnostics and reliability, identical bounding box and capacity, and bit-identical answers
// from all four navigation kernels over the standard probe grid and direction set. This is the
// acceptance criterion for persistence -- a solid that survives a write/read cycle must be
// indistinguishable through the public interface.
void checkSolidsIdentical(const SurfaceSolid& solid, const SurfaceSolid& other, double extent, int samples)
{
  BOOST_CHECK_EQUAL(other.GetNsurfaces(), solid.GetNsurfaces());
  BOOST_CHECK_EQUAL(other.IsDefined(), solid.IsDefined());
  BOOST_CHECK_EQUAL(other.HasBVH(), solid.HasBVH());
  BOOST_CHECK_EQUAL(other.IsClosed(), solid.IsClosed());
  BOOST_CHECK_EQUAL(other.IsOrientationConsistent(), solid.IsOrientationConsistent());
  BOOST_CHECK_EQUAL(static_cast<int>(other.GetNavigationReliability()),
                    static_cast<int>(solid.GetNavigationReliability()));
  BOOST_CHECK_EQUAL(other.GetBoundaryEdgeCount(), solid.GetBoundaryEdgeCount());
  BOOST_CHECK_EQUAL(other.GetNonManifoldEdgeCount(), solid.GetNonManifoldEdgeCount());
  BOOST_CHECK_EQUAL(other.GetReversedEdgeCount(), solid.GetReversedEdgeCount());

  BOOST_CHECK_EQUAL(other.GetDX(), solid.GetDX());
  BOOST_CHECK_EQUAL(other.GetDY(), solid.GetDY());
  BOOST_CHECK_EQUAL(other.GetDZ(), solid.GetDZ());
  for (int dimension = 0; dimension < 3; ++dimension) {
    BOOST_CHECK_EQUAL(other.GetOrigin()[dimension], solid.GetOrigin()[dimension]);
  }
  BOOST_CHECK_EQUAL(other.Capacity(), solid.Capacity());

  for (const auto& point : probeGrid(extent, samples)) {
    BOOST_TEST_CONTEXT("point = (" << point[0] << ", " << point[1] << ", " << point[2] << ")")
    {
      BOOST_CHECK_EQUAL(other.Contains(point.data()), solid.Contains(point.data()));
      BOOST_CHECK_EQUAL(other.Safety(point.data(), solid.Contains(point.data())),
                        solid.Safety(point.data(), solid.Contains(point.data())));
      for (const auto& direction : probeDirections()) {
        BOOST_CHECK_EQUAL(other.DistFromOutside(point.data(), direction.data(), 3),
                          solid.DistFromOutside(point.data(), direction.data(), 3));
        BOOST_CHECK_EQUAL(other.DistFromInside(point.data(), direction.data(), 3),
                          solid.DistFromInside(point.data(), direction.data(), 3));
      }
    }
  }
}

// Write "solid" to a ROOT file and read it back as an independent object.
std::unique_ptr<SurfaceSolid> writeAndReadBack(const SurfaceSolid& solid)
{
  const auto path = std::filesystem::temp_directory_path() /
                    (std::string("o2_bvhsurfacesolid_persist_") + solid.GetName() + ".root");
  {
    TFile file(path.string().c_str(), "RECREATE");
    BOOST_REQUIRE(!file.IsZombie());
    // WriteObject takes a non-const pointer; the call does not modify the solid.
    file.WriteObject(const_cast<SurfaceSolid*>(&solid), "solid");
  }
  std::unique_ptr<SurfaceSolid> restored;
  {
    TFile file(path.string().c_str(), "READ");
    BOOST_REQUIRE(!file.IsZombie());
    restored.reset(file.Get<SurfaceSolid>("solid"));
  }
  std::filesystem::remove(path);
  return restored;
}
} // namespace

// ROOT persistence round trip. The kernel objects behind the solid (BoundedSurface, the BVH, the
// display mesh) are all *derived* state; what has to survive a write/read cycle is the sequence of
// Add*Surface calls the solid was built from, after which CloseShape() reconstructs the rest.
//
// It regresses a bug where nothing at all was streamed: fImpl was transient, so
// a read-back solid came back with zero surfaces, CloseShape(false) then zeroed the streamed
// bounding box, and an *empty* ClosureReport defaults to closed/consistent -- so the husk reported
// NavigationReliability::Reliable and answered "outside" everywhere with full confidence. Any
// TGeoManager::Export/Import of a geometry containing one of these solids silently replaced it by
// an authoritatively-reliable empty point.
BOOST_AUTO_TEST_CASE(PersistenceRoundTrip)
{
  // every surface family and both trim flavours (scalar range and wire trim, the latter with
  // line, arc and B-spline curves) must survive, so each record field is exercised
  const auto box = makeBoxSolid("persistBox", 1., 2., 3.);
  const auto tube = makeTubeSolid("persistTube", 1., 2., 3.); // inner wall + annular arc-wire caps
  const auto cone = makeConeSolid("persistCone", 2., 1., 3.);
  const auto sphere = makeSphereSolid("persistSphere", 2.);
  const auto torus = makeTorusSolid("persistTorus", 3., 1.);
  const auto capsule = makeCapsuleSolid("persistCapsule", 2., 3.);

  for (const auto* solid : {box.get(), tube.get(), cone.get(), sphere.get(), torus.get(), capsule.get()}) {
    BOOST_TEST_CONTEXT("solid = " << solid->GetName())
    {
      const auto restored = writeAndReadBack(*solid);
      BOOST_REQUIRE(restored != nullptr);
      checkSolidsIdentical(*solid, *restored, 4.5, 5);
    }
  }

  // a wire-trimmed cylinder whose window is a NURBS loop: the B-spline degree, poles, weights and
  // knots all have to make the round trip, and the trimmed overload has to be the one replayed
  SurfaceSolid trimmed("persistTrimmed");
  constexpr double radius = 2.;
  constexpr double halfHeight = 3.;
  const std::vector<BoundaryCurve> window{quarterCircleBoundaryCurve(surf::kPi, 0., 0.5, 0.),
                                          quarterCircleBoundaryCurve(surf::kPi, 0., 0.5, surf::kHalfPi),
                                          quarterCircleBoundaryCurve(surf::kPi, 0., 0.5, surf::kPi),
                                          quarterCircleBoundaryCurve(surf::kPi, 0., 0.5, 3. * surf::kHalfPi)};
  BOOST_REQUIRE(trimmed.AddCylindricalSurface({0., 0., 0.}, {0., 0., 1.}, {1., 0., 0.}, radius, -halfHeight,
                                              halfHeight, 0., surf::kTwoPi, false, window));
  BOOST_REQUIRE(addDiskSurface(trimmed, {0., 0., halfHeight}, {1., 0., 0.}, {0., 1., 0.}, radius));
  BOOST_REQUIRE(addDiskSurface(trimmed, {0., 0., -halfHeight}, {1., 0., 0.}, {0., -1., 0.}, radius));
  trimmed.CloseShape(false);
  BOOST_CHECK_EQUAL(trimmed.GetNsurfaces(), 3);

  const auto restoredTrimmed = writeAndReadBack(trimmed);
  BOOST_REQUIRE(restoredTrimmed != nullptr);
  checkSolidsIdentical(trimmed, *restoredTrimmed, 4.5, 5);

  // the model's own tolerance is solid-level state, not derived from the records, so it has to be
  // streamed rather than recomputed on replay
  SurfaceSolid toleranced("persistTolerance");
  for (int faceIndex = 0; faceIndex < 6; ++faceIndex) {
    BOOST_REQUIRE(addBoxFace(toleranced, faceIndex, 1., 2., 3.));
  }
  toleranced.SetModelTolerance(7.25e-5);
  toleranced.CloseShape(false);
  const auto restoredToleranced = writeAndReadBack(toleranced);
  BOOST_REQUIRE(restoredToleranced != nullptr);
  checkClose(restoredToleranced->GetModelTolerance(), 7.25e-5, 1.e-18);

  // an unnavigable solid must come back unnavigable: the failure mode S1 describes is precisely a
  // defective solid that acquires a clean bill of health by losing its surfaces on the way
  SurfaceSolid openBox("persistOpenBox");
  for (int faceIndex = 0; faceIndex < 5; ++faceIndex) { // deliberately missing the sixth face
    BOOST_REQUIRE(addBoxFace(openBox, faceIndex, 1., 2., 3.));
  }
  openBox.CloseShape(false);
  BOOST_REQUIRE(!openBox.IsNavigable());
  BOOST_CHECK_EQUAL(static_cast<int>(openBox.GetNavigationReliability()),
                    static_cast<int>(SurfaceSolid::NavigationReliability::OpenSurfaceSet));

  const auto restoredOpenBox = writeAndReadBack(openBox);
  BOOST_REQUIRE(restoredOpenBox != nullptr);
  BOOST_CHECK(!restoredOpenBox->IsNavigable());
  checkSolidsIdentical(openBox, *restoredOpenBox, 4.5, 5);
}

// A solid that reaches the reader with no surface records -- a file written by an older version,
// or a solid streamed before CloseShape() -- must report Undetermined rather than manufacture a
// clean ClosureReport out of an empty surface set. "I do not know" is the only honest answer, and
// the difference matters: NavigationReliability is the flag callers are told to check.
BOOST_AUTO_TEST_CASE(EmptySolidIsNotReliable)
{
  SurfaceSolid empty("emptySolid");
  BOOST_CHECK_EQUAL(static_cast<int>(empty.GetNavigationReliability()),
                    static_cast<int>(SurfaceSolid::NavigationReliability::Undetermined));
  BOOST_CHECK(!empty.IsNavigable());

  // CloseShape on an empty surface set must not define the shape, with or without checking
  empty.CloseShape(false);
  BOOST_CHECK(!empty.IsDefined());
  BOOST_CHECK_EQUAL(static_cast<int>(empty.GetNavigationReliability()),
                    static_cast<int>(SurfaceSolid::NavigationReliability::Undetermined));
  BOOST_CHECK(!empty.IsNavigable());
  BOOST_CHECK(!empty.IsClosed());
}

namespace
{
// A golden-angle spiral of unit directions: quasi-uniform on the sphere, so no two are
// near-parallel and none aligns with a coordinate axis or a 45-degree symmetry plane. Used to
// test the invariant that containment does not depend on where the parity ray is aimed.
std::vector<std::array<double, 3>> spiralDirections(int count)
{
  std::vector<std::array<double, 3>> directions;
  directions.reserve(count);
  for (int index = 0; index < count; ++index) {
    const double cosTheta = 1. - 2. * (index + 0.5) / count;
    const double sinTheta = std::sqrt(1. - cosTheta * cosTheta);
    const double phi = 2.399963229728653 * index;
    directions.push_back({sinTheta * std::cos(phi), sinTheta * std::sin(phi), cosTheta});
  }
  return directions;
}
} // namespace

// Parity containment answers a topological question, so on a closed, consistently oriented
// 2-manifold it cannot depend on where the ray is aimed. That invariant is what licenses the
// single-shot fast path: Contains() casts one fixed direction and stops.
//
// It is also the sharpest available oracle for the surface set itself -- no reference shape is
// involved, only the solid disagreeing with itself. Measured over the Phase 0 corpus, every part
// the closure check calls Reliable has *zero* direction disagreements in 11k points, and every
// part with disagreements is one the closure check already rejects.
BOOST_AUTO_TEST_CASE(ContainsIsDirectionIndependentOnClosedSolids)
{
  const auto box = makeBoxSolid("dirBox", 1., 2., 3.);
  const auto tube = makeTubeSolid("dirTube", 1., 2., 3.);
  const auto cone = makeConeSolid("dirCone", 2., 1., 3.);
  const auto sphere = makeSphereSolid("dirSphere", 2.);
  const auto torus = makeTorusSolid("dirTorus", 3., 1.);
  const auto capsule = makeCapsuleSolid("dirCapsule", 2., 3.);

  const auto directions = spiralDirections(13);
  for (const auto* solid : {box.get(), tube.get(), cone.get(), sphere.get(), torus.get(), capsule.get()}) {
    BOOST_TEST_CONTEXT("solid = " << solid->GetName())
    {
      BOOST_REQUIRE(solid->IsNavigable());
      for (const auto& point : probeGrid(4.5, 7)) {
        const bool reference = solid->Contains(point.data());
        for (const auto& direction : directions) {
          BOOST_TEST_CONTEXT("point = (" << point[0] << ", " << point[1] << ", " << point[2] << ") direction = ("
                                         << direction[0] << ", " << direction[1] << ", " << direction[2] << ")")
          {
            BOOST_CHECK_EQUAL(solid->ContainsAlongDirection(point.data(), direction.data()), reference);
          }
        }
      }
    }
  }
}

// Section 4.4's re-shoot, on the defect it exists for. A gap in the surface set costs the parity
// ray exactly the crossings that fall inside the gap, so a point is misclassified over the whole
// *shadow* of the gap along the shooting direction -- centimetres of wrong answers arbitrarily far
// from any surface. Aiming the ray somewhere else escapes that shadow, which is why a majority
// over several directions recovers the right answer: measured over the 55 points where the single
// fixed direction disagrees with the OpenCascade oracle on the Phase 0 corpus, not one point is
// wrong in every direction.
//
// The fixture makes the mechanism explicit rather than statistical: the +x face of a box is split
// into two rectangles with a thin strip left out, so a ray leaving along +x from inside sees no
// crossing at all and reports "outside".
BOOST_AUTO_TEST_CASE(ContainsReshootsThroughSurfaceGaps)
{
  constexpr double halfX = 1.;
  constexpr double halfY = 2.;
  constexpr double halfZ = 3.;
  constexpr double gap = 0.05; // half-width in z of the missing strip on the +x face

  SurfaceSolid gapped("gappedBox");
  for (int faceIndex = 1; faceIndex < 6; ++faceIndex) { // every face but +x
    BOOST_REQUIRE(addBoxFace(gapped, faceIndex, halfX, halfY, halfZ));
  }
  // the +x face as two rectangles, leaving z in (-gap, +gap) uncovered. Frame of face 0:
  // origin (halfX, -halfY, -halfZ), axisU = +y, axisV = +z.
  BOOST_REQUIRE(gapped.AddPlanarSurface({halfX, -halfY, -halfZ}, {0., 1., 0.}, {0., 0., 1.},
                                        rectangleWire(2. * halfY, halfZ - gap)));
  BOOST_REQUIRE(gapped.AddPlanarSurface({halfX, -halfY, gap}, {0., 1., 0.}, {0., 0., 1.},
                                        rectangleWire(2. * halfY, halfZ - gap)));
  gapped.CloseShape(false);

  // the gap is what makes the solid unnavigable, and only an unnavigable solid re-shoots
  BOOST_REQUIRE(!gapped.IsNavigable());
  BOOST_CHECK_EQUAL(static_cast<int>(gapped.GetNavigationReliability()),
                    static_cast<int>(SurfaceSolid::NavigationReliability::OpenSurfaceSet));

  // a point deep inside whose +x ray leaves straight through the gap
  const std::array<double, 3> insidePoint{0., 0.3, 0.};
  const std::array<double, 3> throughGap{1., 0., 0.};
  BOOST_CHECK(!gapped.ContainsAlongDirection(insidePoint.data(), throughGap.data())); // the defect itself
  BOOST_CHECK(gapped.Contains(insidePoint.data()));                                   // the re-shoot recovers it
  BOOST_CHECK(gapped.Contains_Loop(insidePoint.data()));                              // ... on both paths

  // the same point in the same box *without* the gap is inside from every direction, so the
  // fixture isolates the gap and not some accident of the point
  const auto intact = makeBoxSolid("intactBox", halfX, halfY, halfZ);
  BOOST_CHECK(intact->Contains(insidePoint.data()));
  BOOST_CHECK(intact->ContainsAlongDirection(insidePoint.data(), throughGap.data()));

  // and the BVH and loop parities still agree everywhere on the defective solid: the re-shoot is
  // applied by one shared helper, so it can never make the two paths differ
  for (const auto& point : probeGrid(4.5, 7)) {
    BOOST_TEST_CONTEXT("point = (" << point[0] << ", " << point[1] << ", " << point[2] << ")")
    {
      BOOST_CHECK_EQUAL(gapped.Contains(point.data()), gapped.Contains_Loop(point.data()));
    }
  }
}

namespace
{
// An L-shaped prism, the concave fixture. Its footprint is ([0,3]x[0,1]) union ([0,1]x[1,2])
// extruded over
// z in [0, height], so it has a *reflex* (concave) vertical edge at x = 1, y = 1 -- the one place
// where a ray can touch the boundary from inside and stay inside, which no convex fixture can
// reproduce. Built from eight planar faces with outward normals, so it is closed and consistently
// oriented.
std::unique_ptr<SurfaceSolid> makeLPrismSolid(const char* name, double height = 1.)
{
  // footprint, counter-clockwise; (1,1) is the reflex vertex
  const std::vector<Point2D> footprint{{0., 0.}, {3., 0.}, {3., 1.}, {1., 1.}, {1., 2.}, {0., 2.}};

  auto solid = std::make_unique<SurfaceSolid>(name);

  // bottom (outward normal -z) and top (+z); axisU x axisV fixes the normal
  std::vector<Point2D> bottomWire;
  bottomWire.reserve(footprint.size());
  for (const auto& vertex : footprint) {
    bottomWire.push_back({vertex[1], vertex[0]}); // (u, v) = (y, x) so that axisU x axisV = -z
  }
  BOOST_REQUIRE(solid->AddPlanarSurface({0., 0., 0.}, {0., 1., 0.}, {1., 0., 0.}, bottomWire));
  BOOST_REQUIRE(solid->AddPlanarSurface({0., 0., height}, {1., 0., 0.}, {0., 1., 0.}, footprint));

  // one vertical wall per footprint edge; axisU along the edge and axisV = +z put the normal at
  // (dy, -dx, 0), which points out of a counter-clockwise footprint
  for (size_t index = 0; index < footprint.size(); ++index) {
    const auto& start = footprint[index];
    const auto& end = footprint[(index + 1) % footprint.size()];
    const double deltaU = end[0] - start[0];
    const double deltaV = end[1] - start[1];
    const double length = std::hypot(deltaU, deltaV);
    BOOST_REQUIRE(solid->AddPlanarSurface({start[0], start[1], 0.}, {deltaU / length, deltaV / length, 0.},
                                          {0., 0., 1.}, rectangleWire(length, height)));
  }
  solid->CloseShape();
  return solid;
}
} // namespace

// S2: the bounding-box pre-check ran *before* the documented "no BVH yet, fall back to the plain
// loop" branch. Before CloseShape() the box is still all zeros, so the pre-check rejected every
// point outside a 1e-9 cube at the origin and the fallback was unreachable -- Contains() was
// effectively disabled on any solid that had not been closed yet.
BOOST_AUTO_TEST_CASE(ContainsWorksBeforeCloseShape)
{
  SurfaceSolid box("preCloseBox");
  addBoxSurfaces(box, 1., 2., 3.);
  BOOST_REQUIRE(!box.HasBVH()); // the premise: no acceleration structure and no bounding box yet

  for (const auto& point : probeGrid(4.5, 5)) {
    BOOST_TEST_CONTEXT("point = (" << point[0] << ", " << point[1] << ", " << point[2] << ")")
    {
      const bool inside = std::abs(point[0]) < 1. && std::abs(point[1]) < 2. && std::abs(point[2]) < 3.;
      BOOST_CHECK_EQUAL(box.Contains(point.data()), inside);
      BOOST_CHECK_EQUAL(box.Contains(point.data()), box.Contains_Loop(point.data()));
    }
  }
}

// S3: a point *on* a face is inside for Contains, but its t = 0 exit was below the minimum ray
// parameter and therefore invisible to DistFromInside, which then returned Big. A navigator that
// asks "how far to the wall" while standing on the wall and is told "never" tunnels straight
// through the geometry. ROOT's own primitives answer 0 here, and so must this.
BOOST_AUTO_TEST_CASE(BoundaryPointsAgreeBetweenContainsAndDistances)
{
  const auto box = makeBoxSolid("boundaryPolicyBox", 1., 2., 3.);
  const std::array<double, 3> onFace{1., 0.5, 0.5}; // exactly on the +x face
  const std::array<double, 3> outward{1., 0., 0.};
  const std::array<double, 3> inward{-1., 0., 0.};

  BOOST_CHECK(box->Contains(onFace.data())); // documented policy: on a face counts as inside

  TGeoBBox reference("boundaryPolicyReference", 1., 2., 3.);
  BOOST_CHECK_EQUAL(box->DistFromInside(onFace.data(), outward.data(), 3), 0.);
  BOOST_CHECK_EQUAL(box->DistFromOutside(onFace.data(), inward.data(), 3), 0.);
  checkClose(box->DistFromInside(onFace.data(), outward.data(), 3),
             reference.DistFromInside(onFace.data(), outward.data(), 3));
  checkClose(box->DistFromOutside(onFace.data(), inward.data(), 3),
             reference.DistFromOutside(onFace.data(), inward.data(), 3));

  // going the other way the far wall is still the answer, so the fix is not "always return 0"
  checkClose(box->DistFromInside(onFace.data(), inward.data(), 3), 2.);

  // the BVH and loop paths must agree on all of it
  for (const auto& direction : {outward, inward}) {
    checkDistanceAgainstLoop(*box, onFace, direction);
  }
}

// S4 / S5: a ray that only *touches* the boundary has not crossed it. Contains() knows this --
// near-equal hits are clustered and a cluster carrying both an entering and an exiting hit
// contributes even parity -- but the distance queries classified every hit on its own, so they
// reported the touch as a crossing. The two then disagree about the same ray: DistFromOutside
// hands the navigator a step to the touch point, Contains says it is still outside once it gets
// there, and the navigator takes zero-length steps forever.
//
// Both flavours are covered. A convex edge graze (box) is the outside-facing case, and the L-prism
// reflex edge is the inside-facing one, which no convex solid can produce: there the ray leaves
// and re-enters the material at a single point and must be reported as never having left.
BOOST_AUTO_TEST_CASE(EdgeGrazesAreNotCrossings)
{
  const double invSqrt2 = 1. / std::sqrt(2.);

  // --- convex: touch the box edge x = +1, y = +2 and stay outside on both sides of the touch
  const auto box = makeBoxSolid("grazeBox", 1., 2., 3.);
  const std::array<double, 3> grazeDirection{invSqrt2, -invSqrt2, 0.};
  const std::array<double, 3> grazeOrigin{1. - 5. * invSqrt2, 2. + 5. * invSqrt2, 0.};
  BOOST_REQUIRE(!box->Contains(grazeOrigin.data()));

  // a point just past the touch is still outside, so nothing was entered ...
  const std::array<double, 3> pastTouch{1. + 1.e-3 * invSqrt2, 2. - 1.e-3 * invSqrt2, 0.};
  BOOST_REQUIRE(!box->Contains(pastTouch.data()));
  // ... and the distance query must say so too
  BOOST_CHECK_EQUAL(box->DistFromOutside(grazeOrigin.data(), grazeDirection.data(), 3), TGeoShape::Big());
  BOOST_CHECK_EQUAL(box->DistFromOutside_Loop(grazeOrigin.data(), grazeDirection.data()), TGeoShape::Big());

  // --- concave: the L-prism's reflex edge at x = 1, y = 1
  const auto prism = makeLPrismSolid("grazePrism");
  BOOST_REQUIRE(prism->IsNavigable());
  checkClose(prism->Capacity(), 4.); // 3x1 plus 1x1, extruded over unit height

  // a ray through the reflex edge along (1,-1): inside before the touch, inside after it, so the
  // touch is not an exit. The real exit is where it leaves the long arm at y = 0.
  const std::array<double, 3> reflexDirection{invSqrt2, -invSqrt2, 0.};
  const std::array<double, 3> reflexOrigin{1. - 0.5 * invSqrt2, 1. + 0.5 * invSqrt2, 0.5};
  BOOST_REQUIRE(prism->Contains(reflexOrigin.data()));
  const std::array<double, 3> pastReflex{1. + 1.e-3 * invSqrt2, 1. - 1.e-3 * invSqrt2, 0.5};
  BOOST_REQUIRE(prism->Contains(pastReflex.data())); // still inside: the touch was not an exit

  const double touchDistance = 0.5;
  const double exitDistance = 0.5 + std::sqrt(2.); // on to (2, 0, 0.5), in the middle of the y = 0 wall
  const double reported = prism->DistFromInside(reflexOrigin.data(), reflexDirection.data(), 3);
  BOOST_CHECK_GT(reported, touchDistance + 1.e-6); // the touch is not the answer ...
  checkClose(reported, exitDistance);              // ... the far wall is
  BOOST_CHECK_EQUAL(prism->DistFromInside_Loop(reflexOrigin.data(), reflexDirection.data()), reported);
}

// The direction-taking DescribeContainsCrossings dumps the crossing list behind ContainsAlongDirection.
BOOST_AUTO_TEST_CASE(DescribeContainsCrossingsTakesAnExplicitDirection)
{
  const auto box = makeBoxSolid("describeBox", 1., 2., 3.);
  const SurfaceSolid::Point3D inside{0.2, 0.3, 0.4};
  const double invSqrt2 = 1. / std::sqrt(2.);
  // +x leaves through x = +1 at 0.8; the unnormalised -z direction leaves through z = -3 at 3.4
  const std::vector<std::pair<SurfaceSolid::Point3D, double>> cases{
    {{1., 0., 0.}, 0.8}, {{invSqrt2, -invSqrt2, 0.}, 0.8 * std::sqrt(2.)}, {{0., 0., -2.}, 3.4}};
  for (const auto& [direction, exitDistance] : cases) {
    BOOST_TEST_CONTEXT("direction = (" << direction[0] << ", " << direction[1] << ", " << direction[2] << ")")
    {
      std::vector<SurfaceSolid::ContainsCrossing> bvhCrossings;
      std::vector<SurfaceSolid::ContainsCrossing> loopCrossings;
      box->DescribeContainsCrossings(inside, direction, bvhCrossings, loopCrossings);
      // from inside a convex box every ray leaves through exactly one face
      BOOST_REQUIRE_EQUAL(bvhCrossings.size(), 1u);
      BOOST_REQUIRE_EQUAL(loopCrossings.size(), 1u);
      BOOST_CHECK_EQUAL(bvhCrossings[0].distance, loopCrossings[0].distance);
      checkClose(bvhCrossings[0].distance, exitDistance);
      BOOST_CHECK_GT(bvhCrossings[0].normalAlignment, 0.); // an exit
      BOOST_CHECK(!bvhCrossings[0].onTrimBoundary);
      BOOST_CHECK(box->ContainsAlongDirection(inside.data(), direction.data()));
    }
  }
}

// The concave fixture earns its keep beyond the single grazing ray: the whole sweep battery is
// run on it, since every invariant the convex fixtures pin (BVH == loop, direction-independent
// parity, Contains consistent with the distance answers) is weaker on shapes with no reflex edge.
BOOST_AUTO_TEST_CASE(LPrismSweeps)
{
  const auto prism = makeLPrismSolid("sweepPrism");
  BOOST_REQUIRE(prism->IsNavigable());

  sweepDistanceAgainstLoop(*prism, 3.5, 5);

  const auto directions = spiralDirections(13);
  for (const auto& point : probeGrid(3.5, 7)) {
    const bool inside = prism->Contains(point.data());
    BOOST_TEST_CONTEXT("point = (" << point[0] << ", " << point[1] << ", " << point[2] << ")")
    {
      BOOST_CHECK_EQUAL(prism->Contains_Loop(point.data()), inside);
      for (const auto& direction : directions) {
        BOOST_CHECK_EQUAL(prism->ContainsAlongDirection(point.data(), direction.data()), inside);
      }
      // the closed-form answer for the extruded L footprint
      const bool expected = point[2] > 0. && point[2] < 1. &&
                            ((point[0] > 0. && point[0] < 3. && point[1] > 0. && point[1] < 1.) ||
                             (point[0] > 0. && point[0] < 1. && point[1] >= 1. && point[1] < 2.));
      BOOST_CHECK_EQUAL(inside, expected);
    }
  }
}

// K1: the B-spline endpoint shortcut assumed a clamped knot vector. A clamped curve interpolates
// its first and last pole, so returning those is exact and free; an unclamped one -- which is what
// OCC writes for a periodic tube-tube intersection curve before SetNotPeriodic -- starts and ends
// strictly inside its control polygon, and the shortcut then returned points that are not on the
// curve at all. Downstream, the wire's edges no longer meet (so it reads as Open and the whole
// face is thrown away) or the off-curve endpoint corrupts the winding classification, since
// CurveWire::classify deliberately uses canonical shared endpoints.
BOOST_AUTO_TEST_CASE(UnclampedBSplineEndpointsAreOnTheCurve)
{
  using surf::Curve2D;
  using surf::Vec2;

  // the same cubic control polygon read twice: once with a clamped knot vector, once with a
  // uniform (unclamped) one. Only the clamped curve may claim its poles as endpoints.
  const std::vector<Vec2> poles{{0., 0.}, {1., 2.}, {3., 2.}, {4., 0.}};
  const Curve2D clamped = Curve2D::makeBSpline(3, poles, {}, {0., 0., 0., 0., 1., 1., 1., 1.});
  const Curve2D uniform = Curve2D::makeBSpline(3, poles, {}, {0., 1., 2., 3., 4., 5., 6., 7.});

  BOOST_REQUIRE(clamped.valid());
  BOOST_REQUIRE(uniform.valid());

  // the endpoints must lie on their own curve, whatever the knot vector says
  for (const auto* curve : {&clamped, &uniform}) {
    const Vec2 start = curve->startPoint();
    const Vec2 end = curve->endPoint();
    const Vec2 evaluatedStart = curve->pointAt(0.);
    const Vec2 evaluatedEnd = curve->pointAt(1.);
    checkClose(start.uCoord, evaluatedStart.uCoord);
    checkClose(start.vCoord, evaluatedStart.vCoord);
    checkClose(end.uCoord, evaluatedEnd.uCoord);
    checkClose(end.vCoord, evaluatedEnd.vCoord);
  }

  // and the two curves really are different, so the test is not vacuous: the clamped one
  // interpolates its outer poles, the uniform one does not come near them
  checkClose(clamped.startPoint().uCoord, 0.);
  checkClose(clamped.endPoint().uCoord, 4.);
  BOOST_CHECK_GT(std::hypot(uniform.startPoint().uCoord - poles.front().uCoord,
                            uniform.startPoint().vCoord - poles.front().vCoord),
                 0.1);

  // a wire closed on the *curve* must validate, which is what the shortcut used to prevent: with
  // poles.front() as the reported start, the joining line would have missed it by that distance
  const Vec2 uniformStart = uniform.startPoint();
  const Vec2 uniformEnd = uniform.endPoint();
  surf::CurveWire wire;
  surf::WireStatus status = surf::WireStatus::Valid;
  BOOST_CHECK(wire.initialize({uniform, Curve2D::makeLine(uniformEnd, uniformStart)}, surf::WireRole::Outer, status));
}

// K2: the full-turn rejection measured the *control-point hull*, not the curve. A closed trim
// curve that wraps nearly a full turn in phi has poles outside its own span (that is what makes
// the hull a conservative bound), so the check saw more than 2*pi and refused a perfectly legal
// through-hole host face -- and a refused face is a face missing from the parity solid, i.e. wrong
// containment throughout its shadow.
BOOST_AUTO_TEST_CASE(NearFullTurnTrimIsNotRejectedOnItsPoleHull)
{
  using surf::Curve2D;
  using surf::Vec2;
  std::string error;

  // A trim wrapping 350 degrees of a cylinder, spelled as two quadratic B-spline spans whose
  // middle poles sit *outside* the span in phi -- which is exactly what makes the control-point
  // hull a conservative bound and not the curve's own extent. The curve stays inside 2*pi; its
  // pole hull does not.
  const double sweep = 350. * surf::kPi / 180.;
  const double overshoot = 0.4;
  const std::vector<Curve2D> outer{
    Curve2D::makeBSpline(2, {{0., -1.}, {-overshoot, 0.}, {0.5 * sweep, 1.}}, {}, {0., 0., 0., 1., 1., 1.}),
    Curve2D::makeBSpline(2, {{0.5 * sweep, 1.}, {sweep + overshoot, 0.}, {sweep, -1.}}, {}, {0., 0., 0., 1., 1., 1.}),
    Curve2D::makeLine({sweep, -1.}, {0., -1.})};

  // the pole hull must genuinely exceed a full turn, otherwise the fixture proves nothing
  Vec2 hullLower{1.e300, 1.e300};
  Vec2 hullUpper{-1.e300, -1.e300};
  surf::CurveWire hullWire;
  surf::WireStatus hullStatus = surf::WireStatus::Valid;
  BOOST_REQUIRE(hullWire.initialize(outer, surf::WireRole::Outer, hullStatus));
  hullWire.parametricBounds(hullLower, hullUpper);
  BOOST_REQUIRE_GT(hullUpper.uCoord - hullLower.uCoord, surf::kTwoPi);

  // ... while the curve itself does not
  Vec2 tightLower{1.e300, 1.e300};
  Vec2 tightUpper{-1.e300, -1.e300};
  hullWire.tightParametricBounds(tightLower, tightUpper);
  BOOST_CHECK_LT(tightUpper.uCoord - tightLower.uCoord, surf::kTwoPi);

  // so the surface must be accepted
  surf::CylindricalBoundedSurface surface;
  BOOST_CHECK_MESSAGE(surface.initialize({0., 0., 0.}, {0., 0., 1.}, {1., 0., 0.}, 2., -2., 2., 0., surf::kTwoPi,
                                         false, outer, {}, error),
                      "near-full-turn trim rejected: " << error);

  // a trim that really does wrap more than a full turn is still refused
  surf::CylindricalBoundedSurface tooWide;
  const std::vector<Curve2D> overWrapped{Curve2D::makeLine({0., -1.}, {surf::kTwoPi + 0.5, -1.}),
                                         Curve2D::makeLine({surf::kTwoPi + 0.5, -1.}, {surf::kTwoPi + 0.5, 1.}),
                                         Curve2D::makeLine({surf::kTwoPi + 0.5, 1.}, {0., 1.}),
                                         Curve2D::makeLine({0., 1.}, {0., -1.})};
  BOOST_CHECK(!tooWide.initialize({0., 0., 0.}, {0., 0., 1.}, {1., 0., 0.}, 2., -2., 2., 0., surf::kTwoPi, false,
                                  overWrapped, {}, error));
}

// K7 claimed a face that fails to build is logged and silently omitted from the parity solid.
// Reading the code does not support that on any production path: Add*Surface returns false, and
// the sidecar loader turns that into a whole-file rejection, which the converter's generated macro
// turns into an exception. What is true is the weaker statement that the *return value* is the
// only signal, so this pins both halves -- the rejection is reported, and nothing is added behind
// the caller's back. Recorded rather than "fixed", in the same spirit as the S6 correction.
BOOST_AUTO_TEST_CASE(RejectedFacesAreNeverSilentlyAdded)
{
  SurfaceSolid solid("rejectingSolid");
  BOOST_REQUIRE(addBoxFace(solid, 0, 1., 2., 3.));
  BOOST_REQUIRE_EQUAL(solid.GetNsurfaces(), 1);

  // degenerate frame (axisU parallel to axisV), a wire with too few vertices, and a zero-radius
  // cylinder: each must be refused, and none may leave a surface behind
  BOOST_CHECK(!solid.AddPlanarSurface({0., 0., 0.}, {1., 0., 0.}, {1., 0., 0.}, rectangleWire(1., 1.)));
  BOOST_CHECK(!solid.AddPlanarSurface({0., 0., 0.}, {1., 0., 0.}, {0., 1., 0.}, {{0., 0.}, {1., 0.}}));
  BOOST_CHECK(!solid.AddCylindricalSurface({0., 0., 0.}, {0., 0., 1.}, {1., 0., 0.}, 0., -1., 1.));
  BOOST_CHECK_EQUAL(solid.GetNsurfaces(), 1);
  BOOST_CHECK_EQUAL(static_cast<int>(solid.GetSurfaceRecords().size()), 1);

  // the loader's contract on the same failure: reject the file rather than return a partial solid
  std::vector<char> bytes;
  appendSidecarHeader(bytes, 1);
  appendU32(bytes, 2);  // cylinder
  appendU32(bytes, 0);  // flags
  appendU32(bytes, 14); // nParams
  appendDoubles(bytes, {0., 0., 0., 0., 0., 1., 1., 0., 0., 0. /* radius */, -1., 1., 0., 2. * surf::kPi});
  appendU32(bytes, 0); // nWires
  const auto path = writeSidecarFile("o2_sidecar_rejected_face.bin", bytes);
  SurfaceSolid loaded("loadedRejecting");
  BOOST_CHECK(!o2::cad::LoadSurfaceSolid(path.string(), loaded));
  std::filesystem::remove(path);
}

// The rim-based closure measurement.
//
// The half-edge check asks whether two faces emitted the *same vertices* along a shared edge. On
// real CAD that question has the answer "no" for reasons that are not gaps: each face samples the
// shared curve independently, so the vertices genuinely are not the same points and no tolerance
// on vertex equality can help. The rim measurement compares the boundaries as curves instead, and
// reports the answer as a length in cm rather than as a chord count.
//
// Nothing derives a verdict from it yet -- IsNavigable() still reads the chord counters -- so
// these tests pin the measurement, not a change of behaviour.
BOOST_AUTO_TEST_CASE(RimClosureMeasuresTheGapInCentimetres)
{
  constexpr double halfX = 1.;
  constexpr double halfY = 1.5;
  constexpr double halfZ = 2.;

  // a closed box: one rim per face, all matched, and no gap at all
  SurfaceSolid closedBox("rimClosedBox");
  addBoxSurfaces(closedBox, halfX, halfY, halfZ);
  closedBox.CloseShape(false);
  BOOST_REQUIRE(closedBox.IsNavigable());
  BOOST_CHECK_EQUAL(closedBox.GetRimCount(), 6);
  BOOST_CHECK_EQUAL(closedBox.GetMatchedRimCount(), 6);
  BOOST_CHECK_EQUAL(closedBox.GetBoundaryRimCount(), 0);
  BOOST_CHECK_SMALL(closedBox.GetMaxRimIsolation(), 1.e-12);
  BOOST_CHECK_SMALL(closedBox.GetUnmatchedRimLength(), 1.e-12);
  // a box has no curved rim, so its polylines are exact and the measurement has no noise floor
  BOOST_CHECK_SMALL(closedBox.GetRimChordResolution(), 1.e-12);
  // the summed perimeter of the six faces, which is what "how much boundary" is measured in
  BOOST_CHECK_CLOSE(closedBox.GetTotalRimLength(), 16. * (halfX + halfY + halfZ), 1.e-9);

  // the same box with the +z face lifted by a known delta. A box's rims are straight, so their
  // sampling resolution is zero and the match band is the declared tolerance alone; the lifted
  // face's rim is then alone by exactly delta, and that is what the isolation reports.
  constexpr double delta = 1.e-3;
  SurfaceSolid shiftedBox("rimShiftedBox");
  for (int faceIndex = 0; faceIndex < 6; ++faceIndex) {
    const Point3D center = faceIndex == 4 ? Point3D{0., 0., delta} : Point3D{0., 0., 0.};
    BOOST_REQUIRE(addBoxFace(shiftedBox, faceIndex, halfX, halfY, halfZ, false, center));
  }
  shiftedBox.CloseShape(false);
  BOOST_CHECK_CLOSE(shiftedBox.GetMaxRimIsolation(), delta, 1.e-6);
  // and the shift is far above the sampling noise floor, so the number means what it says
  BOOST_CHECK(shiftedBox.GetMaxRimIsolation() > shiftedBox.GetRimChordResolution());
}

// The structural failure the rim criterion exists to fix, pinned: two faces that sample one
// shared edge at different chord counts emit different vertices, so vertex matching calls a
// perfectly closed box open -- and open by *chords*, which is how a seven-loop solid came to
// report 1418 boundary edges. Rim matching compares the curves and gets it right, and it is the
// rim answer that IsClosed()/IsNavigable() now report.
BOOST_AUTO_TEST_CASE(RimClosureSurvivesUnequalChordCounts)
{
  constexpr double halfX = 1.;
  constexpr double halfY = 1.5;
  constexpr double halfZ = 2.;

  SurfaceSolid resampled("rimResampledBox");
  for (int faceIndex = 0; faceIndex < 5; ++faceIndex) {
    BOOST_REQUIRE(addBoxFace(resampled, faceIndex, halfX, halfY, halfZ));
  }
  // the last face again, but with every edge split in two: the same rectangle, twice the vertices
  const FaceFrame frame = boxFaceFrame(5, halfX, halfY, halfZ);
  const double extentU = frame.extentU;
  const double extentV = frame.extentV;
  BOOST_REQUIRE(resampled.AddPlanarSurface(frame.origin, frame.axisU, frame.axisV,
                                           {{0., 0.},
                                            {0.5 * extentU, 0.},
                                            {extentU, 0.},
                                            {extentU, 0.5 * extentV},
                                            {extentU, extentV},
                                            {0.5 * extentU, extentV},
                                            {0., extentV},
                                            {0., 0.5 * extentV}}));
  resampled.CloseShape(false);

  // the per-chord counters still see the disagreement -- they compare the vertices the two faces
  // emitted, and those really are different points. That is the defect, and it is why they no
  // longer decide anything.
  BOOST_CHECK(resampled.GetBoundaryEdgeCount() > 0);

  // the verdict comes from the rims, which see one boundary curve per face, all matched, no gap
  BOOST_CHECK(resampled.IsClosed());
  BOOST_CHECK(resampled.IsNavigable());
  BOOST_CHECK_EQUAL(resampled.GetRimCount(), 6);
  BOOST_CHECK_EQUAL(resampled.GetMatchedRimCount(), 6);
  BOOST_CHECK_EQUAL(resampled.GetBoundaryRimCount(), 0);
  BOOST_CHECK_SMALL(resampled.GetMaxRimIsolation(), 1.e-12);
  BOOST_CHECK_SMALL(resampled.GetUnmatchedRimLength(), 1.e-12);
}

// The per-rim records name which loop is open. A rim's state is on the same scale the solid reports, so
// the solid's verdict must be exactly the worst state present -- which is a self-check, not a
// restatement: the two are accumulated independently.
BOOST_AUTO_TEST_CASE(RimReportsNameTheOffendingLoop)
{
  constexpr double halfX = 1.;
  constexpr double halfY = 1.5;
  constexpr double halfZ = 2.;
  constexpr double delta = 1.e-3;

  SurfaceSolid shiftedBox("rimReportBox");
  for (int faceIndex = 0; faceIndex < 6; ++faceIndex) {
    const Point3D center = faceIndex == 4 ? Point3D{0., 0., delta} : Point3D{0., 0., 0.};
    BOOST_REQUIRE(addBoxFace(shiftedBox, faceIndex, halfX, halfY, halfZ, false, center));
  }
  shiftedBox.CloseShape(false);

  const auto& rims = shiftedBox.GetRimReports();
  BOOST_REQUIRE_EQUAL(static_cast<int>(rims.size()), shiftedBox.GetRimCount());

  int boundaryRims = 0;
  auto worst = SurfaceSolid::NavigationReliability::Reliable;
  double openLength = 0.;
  for (const auto& rim : rims) {
    BOOST_CHECK(rim.surface >= 0 && rim.surface < shiftedBox.GetNsurfaces());
    BOOST_CHECK(rim.rimOnSurface >= 0);
    BOOST_CHECK(rim.closed); // a box face's boundary is one closed loop
    BOOST_CHECK(rim.length > 0.);
    openLength += rim.unmatchedLength;
    if (rim.state == SurfaceSolid::NavigationReliability::OpenSurfaceSet) {
      ++boundaryRims;
      BOOST_CHECK(rim.unmatchedChords > 0);
      BOOST_CHECK(rim.unmatchedLength > 0.);
    } else {
      BOOST_CHECK_EQUAL(rim.unmatchedChords, 0);
    }
    worst = std::max(worst, rim.state);
  }
  BOOST_CHECK_EQUAL(boundaryRims, shiftedBox.GetBoundaryRimCount());
  BOOST_CHECK(worst == shiftedBox.GetNavigationReliability());
  BOOST_CHECK_CLOSE(openLength, shiftedBox.GetUnmatchedRimLength(), 1.e-9);

  // the lifted face's own rim is the one that is alone, and it is alone by the lift
  const auto lifted = std::find_if(rims.begin(), rims.end(), [](const auto& rim) { return rim.surface == 4; });
  BOOST_REQUIRE(lifted != rims.end());
  BOOST_CHECK(lifted->state == SurfaceSolid::NavigationReliability::OpenSurfaceSet);
  BOOST_CHECK_CLOSE(lifted->maxIsolation, delta, 1.e-6);
  BOOST_CHECK(lifted->maxIsolationFace >= 0 && lifted->maxIsolationFace != 4);
  // and the worst chord is named where it is: on the +z face, which sits at halfZ + delta
  BOOST_CHECK_CLOSE(lifted->maxIsolationPoint[2], halfZ + delta, 1.e-6);
}

// Rims are counted per boundary loop and measured in centimetres, not counted per chord. A bare
// cylinder wall is the clearest case: two circular rims, sampled at kArcSamples chords each.
BOOST_AUTO_TEST_CASE(RimCountsAreLoopsAndLengthsNotChords)
{
  constexpr double radius = 1.;
  constexpr double halfHeight = 2.;
  SurfaceSolid openTube("rimOpenTube");
  BOOST_REQUIRE(
    openTube.AddCylindricalSurface({0., 0., 0.}, {0., 0., 1.}, {1., 0., 0.}, radius, -halfHeight, halfHeight));
  openTube.CloseShape(false);

  // the chord counter reports every sample of both rims
  BOOST_CHECK(openTube.GetBoundaryEdgeCount() >= 2 * surf::kArcSamples);
  // the rim measurement reports two open loops, and how long they are
  BOOST_CHECK_EQUAL(openTube.GetRimCount(), 2);
  BOOST_CHECK_EQUAL(openTube.GetBoundaryRimCount(), 2);
  BOOST_CHECK_EQUAL(openTube.GetMatchedRimCount(), 0);
  // both rims are open, and the length is the sampled circumference (a chord polygon, so slightly
  // under 2*pi*r); it is the *length* that is reported, not the sample count
  BOOST_CHECK_CLOSE(openTube.GetUnmatchedRimLength(), openTube.GetTotalRimLength(), 1.e-9);
  BOOST_CHECK_CLOSE(openTube.GetTotalRimLength(), 2. * surf::kTwoPi * radius, 1.);

  // the sagitta of a circle of this radius sampled at kArcSamples per turn, from the closed form:
  // r (1 - cos(pi/kArcSamples)). The estimator must recover it, because it is the floor below
  // which a rim gap is how the rims were sampled rather than how far apart the faces are.
  const double exactSagitta = radius * (1. - std::cos(surf::kPi / surf::kArcSamples));
  BOOST_CHECK_CLOSE(openTube.GetRimChordResolution(), exactSagitta, 1.);

  // Two rims and no third: a full-turn patch emits its seam twice, once each way, and that pair
  // bounds nothing -- it cancels in the half-edge check for the same reason. Chained naively it
  // would become a two-point rim straddling the patch, reporting a gap the size of the patch.
}

// The matching tolerance is the model's own declared one when the sidecar states it (version 2),
// and a documented constant when it does not. Before this the kernel had no way to know what
// epsilon two faces of an imported solid should agree to, and guessed.
BOOST_AUTO_TEST_CASE(RimMatchToleranceComesFromTheModel)
{
  SurfaceSolid unstated("rimToleranceUnstated");
  addBoxSurfaces(unstated, 1., 1., 1.);
  unstated.CloseShape(false);
  BOOST_CHECK_EQUAL(unstated.GetModelTolerance(), 0.);
  BOOST_CHECK_CLOSE(unstated.GetRimMatchTolerance(), surf::kRimMatchTolerance, 1.e-9);

  SurfaceSolid stated("rimToleranceStated");
  addBoxSurfaces(stated, 1., 1., 1.);
  stated.SetModelTolerance(2.5e-7);
  stated.CloseShape(false);
  BOOST_CHECK_CLOSE(stated.GetRimMatchTolerance(), 2.5e-7, 1.e-9);
}

// --- Closed-loop quadrature and unclamped B-spline endpoints ---

/// N1. contourIntegralAlongCurve sized its quadrature from the difference between a curve's
/// endpoints. A closed trim loop -- every hole -- has identical endpoints, so it reported zero
/// travel in u and was handed a single interval, and because max(1, ceil(0 / x)) is 1 the interval
/// cap could not reach it at any value. Curve2D::uVariation measures the travel instead.
BOOST_AUTO_TEST_CASE(ClosedCurveReportsItsTravelNotItsEndpointGap)
{
  // a full circle in the (u, v) chart, written as an arc: the endpoints coincide exactly
  const surf::Curve2D circle = surf::Curve2D::makeArc({0., 0.}, 2., 0., 2. * surf::kPi);
  BOOST_CHECK_SMALL(std::abs(circle.endPoint().uCoord - circle.startPoint().uCoord), 1.e-12);
  // u = 2 cos(angle) travels from +2 down to -2 and back: total variation 8, not 0
  BOOST_CHECK_CLOSE(circle.uVariation(0., 1.), 8., 1.e-9);

  // and the same for a closed B-spline, whose poles bound the travel from above
  std::vector<surf::Vec2> poles{{0., 0.}, {1., 1.}, {2., 0.}, {1., -1.}, {0., 0.}};
  std::vector<double> knots{0., 0., 0., 0.25, 0.5, 0.75, 1., 1., 1.};
  const surf::Curve2D loop = surf::Curve2D::makeBSpline(2, poles, {}, knots);
  BOOST_CHECK_SMALL(std::abs(loop.endPoint().uCoord - loop.startPoint().uCoord), 1.e-9);
  BOOST_CHECK_GT(loop.uVariation(0., 1.), 0.5);
}

/// N1, the defect itself: the contour integrator spent one 20-node Gauss-Legendre rule across a
/// B-spline's whole knot domain, and Gauss-Legendre's geometric convergence needs the integrand
/// analytic on the interval it covers -- a B-spline is one polynomial only within a span. The hole
/// here is an exact circle written the way a CAD kernel writes one, a closed rational quadratic
/// over four knot spans. Verified to fail (by 8e-4 absolute) with the knot subdivision removed.
///
/// The endpoint-based interval count is the *second* defect, and it is why this one hid: a closed
/// loop has coincident endpoints, so it reported zero travel in u, and max(1, ceil(0 / x)) is 1 at
/// every x -- a sweep of kContourMaxSpanU moved nothing while the defect sat behind it. That half
/// is pinned by ClosedCurveReportsItsTravelNotItsEndpointGap; on this corpus it is worth 4e-5 cm^3
/// on ExcavatorArm/BoomCylinderInner against the knot subdivision's 1.1e-2, so it is a correctness fix
/// rather than the cause.
BOOST_AUTO_TEST_CASE(HoleInWireTrimIntegratesToTheAnalyticCapacity)
{
  const double radius = 1.2;
  const double height = 1.5;
  const double holeRadius = 0.3;
  const double centreU = surf::kPi;
  const double centreV = 0.5 * height;

  std::vector<o2::cad::O2BVHSurfaceSolid::PlanarBoundaryCurve> outer{
    o2::cad::O2BVHSurfaceSolid::PlanarBoundaryCurve::makeLine({0., 0.}, {2. * surf::kPi, 0.}),
    o2::cad::O2BVHSurfaceSolid::PlanarBoundaryCurve::makeLine({2. * surf::kPi, 0.}, {2. * surf::kPi, height}),
    o2::cad::O2BVHSurfaceSolid::PlanarBoundaryCurve::makeLine({2. * surf::kPi, height}, {0., height}),
    o2::cad::O2BVHSurfaceSolid::PlanarBoundaryCurve::makeLine({0., height}, {0., 0.})};

  // the textbook exact NURBS circle: nine poles on the circumscribed square's corners and edge
  // midpoints, corner weights sqrt(2)/2, four double interior knots
  const double corner = std::sqrt(2.) / 2.;
  const std::array<std::array<double, 2>, 9> unit{{{1., 0.}, {1., 1.}, {0., 1.}, {-1., 1.}, {-1., 0.}, {-1., -1.}, {0., -1.}, {1., -1.}, {1., 0.}}};
  std::vector<o2::cad::O2BVHSurfaceSolid::Point2D> poles;
  for (const auto& pole : unit) {
    poles.push_back({centreU + holeRadius * pole[0], centreV + holeRadius * pole[1]});
  }
  const std::vector<double> weights{1., corner, 1., corner, 1., corner, 1., corner, 1.};
  const std::vector<double> knots{0., 0., 0., 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1., 1., 1.};
  std::vector<std::vector<o2::cad::O2BVHSurfaceSolid::PlanarBoundaryCurve>> holes{
    {o2::cad::O2BVHSurfaceSolid::PlanarBoundaryCurve::makeBSpline(2, poles, weights, knots)}};

  o2::cad::O2BVHSurfaceSolid withHole("withHole");
  BOOST_REQUIRE(withHole.AddCylindricalSurface({0., 0., 0.}, {0., 0., 1.}, {1., 0., 0.}, radius, 0., height, 0.,
                                               2. * surf::kPi, false, outer, holes));
  std::vector<double> contributions;
  withHole.GetSurfaceCapacityContributions(contributions);
  BOOST_REQUIRE_EQUAL(contributions.size(), 1u);

  // f = (r/3)(C.U cos phi + C.V sin phi + r) with the axis through the origin reduces to r^2/3, so
  // the contribution is r^2/3 times the trimmed chart area: 2 pi h minus the hole's pi rho^2.
  const double chartArea = 2. * surf::kPi * height - surf::kPi * holeRadius * holeRadius;
  BOOST_CHECK_CLOSE(contributions[0], radius * radius * chartArea / 3., 1.e-6);
}

/// N2. The loader read a B-spline edge's endpoints off its first and last poles, which are the
/// endpoints only for a clamped knot vector. On an unclamped one the kernel (since K1) evaluates
/// and the loader did not, so the two measured the same wire join between different points.
BOOST_AUTO_TEST_CASE(UnclampedBSplineEndpointsAreEvaluatedNotReadOffThePoles)
{
  // uniform (unclamped) knots: the curve starts well inside the pole polygon
  std::vector<surf::Vec2> poles{{0., 0.}, {1., 2.}, {2., 2.}, {3., 0.}};
  std::vector<double> knots{0., 1., 2., 3., 4., 5., 6., 7.};
  const surf::Curve2D unclamped = surf::Curve2D::makeBSpline(3, poles, {}, knots);
  const surf::Vec2 start = unclamped.startPoint();
  const surf::Vec2 end = unclamped.endPoint();
  // the whole point: neither endpoint is a pole
  BOOST_CHECK_GT(std::abs(start.uCoord - poles.front().uCoord) + std::abs(start.vCoord - poles.front().vCoord), 1.e-3);
  BOOST_CHECK_GT(std::abs(end.uCoord - poles.back().uCoord) + std::abs(end.vCoord - poles.back().vCoord), 1.e-3);
  // and they are on the curve
  BOOST_CHECK_SMALL(std::abs(unclamped.pointAt(0.).uCoord - start.uCoord), 1.e-12);
  BOOST_CHECK_SMALL(std::abs(unclamped.pointAt(1.).vCoord - end.vCoord), 1.e-12);
}

// --------------------------------------------------------------------------------------------
// Sidecar v3 edge identity.
// --------------------------------------------------------------------------------------------

namespace
{
/// The v3 edge identity of a box built by addBoxSurfaces, derived from the geometry *once, here*.
///
/// The converter derives this from `TopExp::MapShapesAndAncestors` on the source B-rep; a unit
/// test has no B-rep, so it keys the identity on the shared 3D endpoints of the box's own trim
/// segments. That is legitimate precisely because it is not what is under test: what is under
/// test is that the kernel decides closure by *counting the identities it is given* and by
/// nothing else, so the identities have to come from somewhere the kernel cannot see.
///
/// `perFace[f]` is face f's list of (edgeId, flags) in the order AddPlanarSurface was given its
/// four rectangle vertices, which is the order SetSurfaceBoundaryEdges expects.
std::vector<std::pair<std::vector<unsigned int>, std::vector<unsigned char>>>
  boxEdgeIdentity(double halfX, double halfY, double halfZ)
{
  using Key = std::array<long long, 6>;
  auto quantize = [](double value) { return static_cast<long long>(std::llround(value * 1.e9)); };
  std::map<Key, unsigned int> edgeIds;
  std::vector<std::pair<std::vector<unsigned int>, std::vector<unsigned char>>> perFace(6);

  for (int faceIndex = 0; faceIndex < 6; ++faceIndex) {
    const FaceFrame frame = boxFaceFrame(faceIndex, halfX, halfY, halfZ);
    const auto corners = rectangleWire(frame.extentU, frame.extentV);
    for (size_t segment = 0; segment < corners.size(); ++segment) {
      const auto& startUV = corners[segment];
      const auto& endUV = corners[(segment + 1) % corners.size()];
      auto toGlobal = [&](const Point2D& uv) {
        return Point3D{frame.origin[0] + frame.axisU[0] * uv[0] + frame.axisV[0] * uv[1],
                       frame.origin[1] + frame.axisU[1] * uv[0] + frame.axisV[1] * uv[1],
                       frame.origin[2] + frame.axisU[2] * uv[0] + frame.axisV[2] * uv[1]};
      };
      const Point3D start = toGlobal(startUV);
      const Point3D end = toGlobal(endUV);
      const Key forwardKey{quantize(start[0]), quantize(start[1]), quantize(start[2]),
                           quantize(end[0]), quantize(end[1]), quantize(end[2])};
      const Key backwardKey{forwardKey[3], forwardKey[4], forwardKey[5],
                            forwardKey[0], forwardKey[1], forwardKey[2]};
      // the lexicographically smaller ordering names the edge; running the other way is "reversed"
      const bool reversed = backwardKey < forwardKey;
      const Key canonical = reversed ? backwardKey : forwardKey;
      const auto inserted = edgeIds.emplace(canonical, static_cast<unsigned int>(edgeIds.size()));
      perFace[faceIndex].first.push_back(inserted.first->second);
      perFace[faceIndex].second.push_back(
        static_cast<unsigned char>(SurfaceSolid::kEdgeAnchored | (reversed ? SurfaceSolid::kEdgeReversed : 0u)));
    }
  }
  BOOST_REQUIRE_EQUAL(edgeIds.size(), 12u); // a box has twelve edges; if it did not, nothing below means anything
  return perFace;
}

/// A closed box carrying its v3 edge identity, ready for CloseShape().
std::unique_ptr<SurfaceSolid> makeIdentifiedBox(const char* name, double halfX, double halfY, double halfZ)
{
  auto solid = std::make_unique<SurfaceSolid>(name);
  addBoxSurfaces(*solid, halfX, halfY, halfZ);
  const auto identity = boxEdgeIdentity(halfX, halfY, halfZ);
  for (int faceIndex = 0; faceIndex < 6; ++faceIndex) {
    BOOST_REQUIRE(solid->SetSurfaceBoundaryEdges(faceIndex, identity[faceIndex].first, identity[faceIndex].second));
  }
  return solid;
}
} // namespace

/// N3, the core of it. Closure is a count of edge identities, and no tolerance, band or sampling
/// enters the verdict. The self-check is the box: 12 edges, every one of them shared by exactly
/// two faces running opposite ways, and the two faces' realisations of each edge coincide exactly.
BOOST_AUTO_TEST_CASE(EdgeIdentityDecidesClosureByCounting)
{
  const auto box = makeIdentifiedBox("identityBox", 1., 2., 3.);
  box->CloseShape(false);

  BOOST_CHECK(box->HasEdgeIdentity());
  BOOST_CHECK_EQUAL(box->GetSourceEdgeCount(), 12);
  BOOST_CHECK_EQUAL(box->GetSharedSourceEdgeCount(), 12);
  BOOST_CHECK_EQUAL(box->GetBoundarySourceEdgeCount(), 0);
  BOOST_CHECK_EQUAL(box->GetNonManifoldSourceEdgeCount(), 0);
  BOOST_CHECK_EQUAL(box->GetReversedSourceEdgeCount(), 0);
  BOOST_CHECK_EQUAL(box->GetDegenerateSourceEdgeCount(), 0);
  BOOST_CHECK(box->IsClosed());
  BOOST_CHECK(box->IsOrientationConsistent());
  BOOST_CHECK(box->IsNavigable());

  // the deviation is a measurement, and on a box built from one set of corners it is exactly zero
  BOOST_CHECK_EQUAL(box->GetMeasuredSharedEdgeCount(), 12);
  BOOST_CHECK_EQUAL(box->GetUnmeasuredSharedEdgeCount(), 0);
  BOOST_CHECK_SMALL(box->GetMaxSharedEdgeDeviation(), 1.e-15);
}

/// A missing face is a missing face however close the survivors happen to lie. Five faces of a box
/// leave four edges used once, and that is decided by counting rather than by how far apart
/// anything is -- the old criterion had to find the nearest chord and compare it against a band.
BOOST_AUTO_TEST_CASE(EdgeIdentityFindsTheMissingFace)
{
  SurfaceSolid openBox("identityOpenBox");
  for (int faceIndex = 0; faceIndex < 5; ++faceIndex) {
    BOOST_REQUIRE(addBoxFace(openBox, faceIndex, 1., 2., 3.));
  }
  const auto identity = boxEdgeIdentity(1., 2., 3.);
  for (int faceIndex = 0; faceIndex < 5; ++faceIndex) {
    BOOST_REQUIRE(openBox.SetSurfaceBoundaryEdges(faceIndex, identity[faceIndex].first, identity[faceIndex].second));
  }
  openBox.CloseShape(false);

  BOOST_CHECK(openBox.HasEdgeIdentity());
  BOOST_CHECK_EQUAL(openBox.GetSourceEdgeCount(), 12);
  BOOST_CHECK_EQUAL(openBox.GetSharedSourceEdgeCount(), 8);
  BOOST_CHECK_EQUAL(openBox.GetBoundarySourceEdgeCount(), 4); // the missing face's own four edges
  BOOST_CHECK(!openBox.IsClosed());
  BOOST_CHECK(!openBox.IsNavigable());
  BOOST_CHECK_EQUAL(static_cast<int>(openBox.GetNavigationReliability()),
                    static_cast<int>(SurfaceSolid::NavigationReliability::OpenSurfaceSet));
}

/// The sense bit carries the orientation, and it is checked. Two faces that traverse a shared edge
/// the same way disagree about which side is out, whatever their normals happen to look like.
BOOST_AUTO_TEST_CASE(EdgeIdentityFindsReversedAndDuplicatedFaces)
{
  {
    const auto box = makeIdentifiedBox("identityReversed", 1., 2., 3.);
    // flip one face's senses: its four edges now read "twice, same way" instead of "twice, opposite"
    auto identity = boxEdgeIdentity(1., 2., 3.);
    for (auto& flag : identity[0].second) {
      flag = static_cast<unsigned char>(flag ^ SurfaceSolid::kEdgeReversed);
    }
    BOOST_REQUIRE(box->SetSurfaceBoundaryEdges(0, identity[0].first, identity[0].second));
    box->CloseShape(false);
    BOOST_CHECK_EQUAL(box->GetReversedSourceEdgeCount(), 4);
    BOOST_CHECK(box->IsClosed()); // still every edge twice: closed, but inconsistently oriented
    BOOST_CHECK(!box->IsOrientationConsistent());
    BOOST_CHECK_EQUAL(static_cast<int>(box->GetNavigationReliability()),
                      static_cast<int>(SurfaceSolid::NavigationReliability::ReversedFaces));
  }
  {
    // a seventh face claiming edges that already have two owners is non-manifold by count
    const auto box = makeIdentifiedBox("identityNonManifold", 1., 2., 3.);
    BOOST_REQUIRE(addBoxFace(*box, 0, 1., 2., 3.));
    const auto identity = boxEdgeIdentity(1., 2., 3.);
    BOOST_REQUIRE(box->SetSurfaceBoundaryEdges(6, identity[0].first, identity[0].second));
    box->CloseShape(false);
    BOOST_CHECK_EQUAL(box->GetNonManifoldSourceEdgeCount(), 4);
    BOOST_CHECK(!box->IsClosed());
    BOOST_CHECK_EQUAL(static_cast<int>(box->GetNavigationReliability()),
                      static_cast<int>(SurfaceSolid::NavigationReliability::NonManifold));
  }
}

/// maxSharedEdgeDeviation is a measurement, and it has to measure the thing it is named after.
/// Move one face of the box bodily by a known delta while it keeps claiming the same edges: the
/// solid is still *closed* by identity (nothing about which edges exist has changed) and the
/// deviation reports the delta. That separation is the whole design -- the verdict says whether
/// the faces are meant to meet, the number says how well they do.
BOOST_AUTO_TEST_CASE(SharedEdgeDeviationMeasuresHowFarApartTheFacesAre)
{
  constexpr double delta = 3.e-4;
  SurfaceSolid shifted("identityShifted");
  for (int faceIndex = 0; faceIndex < 6; ++faceIndex) {
    // face 4 is the +z cap; nudge it along +z, which pulls its whole rim off the four side faces
    const Point3D centre = faceIndex == 4 ? Point3D{0., 0., delta} : Point3D{0., 0., 0.};
    BOOST_REQUIRE(addBoxFace(shifted, faceIndex, 1., 2., 3., false, centre));
  }
  const auto identity = boxEdgeIdentity(1., 2., 3.);
  for (int faceIndex = 0; faceIndex < 6; ++faceIndex) {
    BOOST_REQUIRE(shifted.SetSurfaceBoundaryEdges(faceIndex, identity[faceIndex].first, identity[faceIndex].second));
  }
  shifted.CloseShape(false);

  BOOST_CHECK(shifted.HasEdgeIdentity());
  BOOST_CHECK_EQUAL(shifted.GetSharedSourceEdgeCount(), 12);
  BOOST_CHECK(shifted.IsClosed()); // by identity: the same twelve edges, each used twice
  BOOST_CHECK_EQUAL(shifted.GetMeasuredSharedEdgeCount(), 12);
  checkClose(shifted.GetMaxSharedEdgeDeviation(), delta, 1.e-12);
}

/// The correspondence between edge identity i and trim curve i survives the wire reorientation
/// that CurveWire/SurfaceWire perform on load. A loop handed in with the wrong winding is
/// reversed in place, so storage index i stops being input index i; pairing the wrong two curves
/// would still produce a number, and a plausible one, which is why the mapping is recorded rather
/// than assumed. Handing the same box in with every loop wound the other way must not move the
/// deviation off zero.
BOOST_AUTO_TEST_CASE(TrimCurveIdentitySurvivesWireReorientation)
{
  SurfaceSolid flipped("identityFlippedWinding");
  const auto identity = boxEdgeIdentity(1., 2., 3.);
  std::vector<std::vector<unsigned int>> ids(6);
  std::vector<std::vector<unsigned char>> flags(6);
  for (int faceIndex = 0; faceIndex < 6; ++faceIndex) {
    FaceFrame frame = boxFaceFrame(faceIndex, 1., 2., 3.);
    auto corners = rectangleWire(frame.extentU, frame.extentV);
    // reverse the vertex ring: segment j of the new ring is segment (n-2-j) of the old one, run
    // backwards, and initialize() will reverse it again to restore the winding it wants
    std::reverse(corners.begin(), corners.end());
    BOOST_REQUIRE(flipped.AddPlanarSurface(frame.origin, frame.axisU, frame.axisV, corners));
    const size_t n = corners.size();
    ids[faceIndex].resize(n);
    flags[faceIndex].resize(n);
    for (size_t j = 0; j < n; ++j) {
      const size_t source = (n - 2 - j + n) % n;
      ids[faceIndex][j] = identity[faceIndex].first[source];
      flags[faceIndex][j] = identity[faceIndex].second[source];
    }
    BOOST_REQUIRE(flipped.SetSurfaceBoundaryEdges(faceIndex, ids[faceIndex], flags[faceIndex]));
  }
  flipped.CloseShape(false);

  BOOST_CHECK(flipped.HasEdgeIdentity());
  BOOST_CHECK_EQUAL(flipped.GetSharedSourceEdgeCount(), 12);
  BOOST_CHECK_EQUAL(flipped.GetMeasuredSharedEdgeCount(), 12);
  // the load-bearing assertion: had the reversal gone unrecorded, this would be a box edge long
  BOOST_CHECK_SMALL(flipped.GetMaxSharedEdgeDeviation(), 1.e-12);
}

/// Partial identity is no identity. A face that names no edges looks exactly like a face with no
/// missing neighbours, which is the failure this replaces, so the whole solid falls back on the
/// geometric rim measurement unless *every* face states its edges.
BOOST_AUTO_TEST_CASE(PartialEdgeIdentityFallsBackToTheRimMeasurement)
{
  SurfaceSolid partial("identityPartial");
  addBoxSurfaces(partial, 1., 2., 3.);
  const auto identity = boxEdgeIdentity(1., 2., 3.);
  for (int faceIndex = 0; faceIndex < 5; ++faceIndex) { // one face left silent
    BOOST_REQUIRE(partial.SetSurfaceBoundaryEdges(faceIndex, identity[faceIndex].first, identity[faceIndex].second));
  }
  partial.CloseShape(false);
  BOOST_CHECK(!partial.HasEdgeIdentity());
  BOOST_CHECK_EQUAL(partial.GetSourceEdgeCount(), 0);
  BOOST_CHECK(partial.IsNavigable()); // the geometric verdict, unchanged: the box really is closed

  // and an out-of-range index or mismatched arrays are refused rather than half-applied
  BOOST_CHECK(!partial.SetSurfaceBoundaryEdges(6, identity[0].first, identity[0].second));
  BOOST_CHECK(!partial.SetSurfaceBoundaryEdges(0, {1u, 2u}, {0}));
}

/// The identity is persistent state, not derived state: a solid that loses it on the way through a
/// ROOT file would come back deciding closure by a different rule than the one that was written.
BOOST_AUTO_TEST_CASE(EdgeIdentitySurvivesPersistence)
{
  const auto box = makeIdentifiedBox("identityPersist", 1., 2., 3.);
  box->CloseShape(false);
  BOOST_REQUIRE(box->HasEdgeIdentity());

  const auto restored = writeAndReadBack(*box);
  BOOST_REQUIRE(restored != nullptr);
  BOOST_CHECK(restored->HasEdgeIdentity());
  BOOST_CHECK_EQUAL(restored->GetSourceEdgeCount(), box->GetSourceEdgeCount());
  BOOST_CHECK_EQUAL(restored->GetSharedSourceEdgeCount(), box->GetSharedSourceEdgeCount());
  BOOST_CHECK_EQUAL(restored->GetMeasuredSharedEdgeCount(), box->GetMeasuredSharedEdgeCount());
  checkClose(restored->GetMaxSharedEdgeDeviation(), box->GetMaxSharedEdgeDeviation(), 1.e-18);
  checkSolidsIdentical(*box, *restored, 4.5, 5);
}

/// The version-3 sidecar: the edge identities reach the kernel through the file, a version-2 file
/// still loads and still gets the geometric verdict, and a file that claims v3 without carrying
/// the identities is rejected as truncated rather than parsed into whatever follows.
BOOST_AUTO_TEST_CASE(SidecarV3EdgeIdentityRoundTrip)
{
  constexpr double halfX = 1.;
  constexpr double halfY = 2.;
  constexpr double halfZ = 3.;
  const auto identity = boxEdgeIdentity(halfX, halfY, halfZ);

  const auto boxBytes = [&](uint32_t version, bool writeIdentity) {
    std::vector<char> bytes;
    appendSidecarHeader(bytes, 6, version, 1.e-7, writeIdentity ? 12u : 0u);
    for (int faceIndex = 0; faceIndex < 6; ++faceIndex) {
      appendPlaneRecord(bytes, boxFaceFrame(faceIndex, halfX, halfY, halfZ));
      if (version >= 3) {
        const auto& ids = identity[faceIndex].first;
        const auto& flags = identity[faceIndex].second;
        appendU32(bytes, writeIdentity ? static_cast<uint32_t>(ids.size()) : 0u);
        if (writeIdentity) {
          for (size_t e = 0; e < ids.size(); ++e) {
            appendU32(bytes, ids[e]);
            bytes.push_back(static_cast<char>(flags[e]));
          }
        }
      }
    }
    return bytes;
  };

  SurfaceSolid v3("sidecarV3Identity");
  const auto v3Path = writeSidecarFile("o2_sidecar_v3_identity.bin", boxBytes(3, true));
  BOOST_REQUIRE(o2::cad::LoadSurfaceSolid(v3Path.string(), v3));
  std::filesystem::remove(v3Path);
  v3.CloseShape(false);
  BOOST_CHECK(v3.HasEdgeIdentity());
  BOOST_CHECK_EQUAL(v3.GetSourceEdgeCount(), 12);
  BOOST_CHECK_EQUAL(v3.GetSharedSourceEdgeCount(), 12);
  BOOST_CHECK_EQUAL(v3.GetMeasuredSharedEdgeCount(), 12);
  BOOST_CHECK_SMALL(v3.GetMaxSharedEdgeDeviation(), 1.e-15);
  BOOST_CHECK(v3.IsNavigable());
  TGeoBBox reference("identityBoxReference", halfX, halfY, halfZ);
  compareContainsGrid(v3, reference, 4., 7);
  checkClose(v3.Capacity(), reference.Capacity(), 1.e-9);

  // a v3 file may legitimately state no identities per face, and then it is a v2 file in all but
  // the header: same load, same geometric verdict, and nothing pretends to know the topology
  SurfaceSolid v3Silent("sidecarV3Silent");
  const auto silentPath = writeSidecarFile("o2_sidecar_v3_silent.bin", boxBytes(3, false));
  BOOST_REQUIRE(o2::cad::LoadSurfaceSolid(silentPath.string(), v3Silent));
  std::filesystem::remove(silentPath);
  v3Silent.CloseShape(false);
  BOOST_CHECK(!v3Silent.HasEdgeIdentity());
  BOOST_CHECK(v3Silent.IsNavigable());

  // and the same six faces written as version 2 -- the compatibility statement, in one assertion
  SurfaceSolid v2("sidecarV2StillLoads");
  const auto v2Path = writeSidecarFile("o2_sidecar_v2_still_loads.bin", boxBytes(2, false));
  BOOST_REQUIRE(o2::cad::LoadSurfaceSolid(v2Path.string(), v2));
  std::filesystem::remove(v2Path);
  v2.CloseShape(false);
  BOOST_CHECK(!v2.HasEdgeIdentity());
  BOOST_CHECK(v2.IsNavigable());
  checkClose(v2.GetModelTolerance(), 1.e-7, 1.e-18);

  // a v3 header over a v2 body: the counts it reads are the next record's bytes, and a reader that
  // resize()s to them is killed rather than reporting anything. It must fail as a parse error.
  std::vector<char> mislabelled;
  appendSidecarHeader(mislabelled, 6, 2, 1.e-7);
  mislabelled[4] = 3; // rewrite the version word in place, leaving a version-2 body behind it
  for (int faceIndex = 0; faceIndex < 6; ++faceIndex) {
    appendPlaneRecord(mislabelled, boxFaceFrame(faceIndex, halfX, halfY, halfZ));
  }
  SurfaceSolid mislabelledSolid("sidecarV3Mislabelled");
  const auto badPath = writeSidecarFile("o2_sidecar_v3_mislabelled.bin", mislabelled);
  BOOST_CHECK(!o2::cad::LoadSurfaceSolid(badPath.string(), mislabelledSolid));
  std::filesystem::remove(badPath);
}

/// The point of the exercise, stated as a test: the verdict must not depend on how finely a
/// B-spline trim is flattened. Under the old criterion it did, and *inversely* -- tightening
/// kBSplineFlatness shrank the per-chord sagitta faster than it shrank the disagreement it was
/// standing in for, so a better-resolved solid read as more open.
/// Sampling cannot reach this criterion at all: it counts identities, and the deviation it reports
/// is evaluated on the curves rather than on their polylines.
BOOST_AUTO_TEST_CASE(EdgeIdentityVerdictIsIndependentOfChordSampling)
{
  const auto box = makeIdentifiedBox("identitySampling", 1., 2., 3.);
  box->CloseShape(false);
  const double deviation = box->GetMaxSharedEdgeDeviation();
  const bool navigable = box->IsNavigable();

  // Resample one face's rim at a different chord count by splitting its wire into eight segments
  // instead of four. The rim polylines the geometric measurement compares now differ in phase and
  // in count -- the exact situation section 13 shows moving the old verdict -- while the edge
  // identity, and every curve it names, is untouched.
  SurfaceSolid resampled("identitySamplingResampled");
  const auto identity = boxEdgeIdentity(1., 2., 3.);
  for (int faceIndex = 0; faceIndex < 6; ++faceIndex) {
    const FaceFrame frame = boxFaceFrame(faceIndex, 1., 2., 3.);
    auto corners = rectangleWire(frame.extentU, frame.extentV);
    std::vector<Point2D> dense;
    std::vector<unsigned int> ids;
    std::vector<unsigned char> flags;
    for (size_t segment = 0; segment < corners.size(); ++segment) {
      const auto& a = corners[segment];
      const auto& b = corners[(segment + 1) % corners.size()];
      dense.push_back(a);
      dense.push_back({0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])});
      // both halves of one box edge carry that edge's identity: an edge split for sampling is
      // still one edge, and the count has to see it as one
      for (int half = 0; half < 2; ++half) {
        ids.push_back(identity[faceIndex].first[segment]);
        flags.push_back(identity[faceIndex].second[segment]);
      }
    }
    BOOST_REQUIRE(resampled.AddPlanarSurface(frame.origin, frame.axisU, frame.axisV, dense));
    BOOST_REQUIRE(resampled.SetSurfaceBoundaryEdges(faceIndex, ids, flags));
  }
  resampled.CloseShape(false);

  // every box edge now appears four times (twice per face, split in half), so it is *not* the
  // "exactly twice" case -- which is the honest answer for this deliberately abused fixture, and
  // the assertion worth making is that the count says so rather than that it says "closed"
  BOOST_CHECK(resampled.HasEdgeIdentity());
  BOOST_CHECK_EQUAL(resampled.GetSourceEdgeCount(), 12);
  BOOST_CHECK_EQUAL(resampled.GetNonManifoldSourceEdgeCount(), 12);

  // and the undisturbed box is unmoved by anything sampling-related
  BOOST_CHECK_EQUAL(box->IsNavigable(), navigable);
  checkClose(box->GetMaxSharedEdgeDeviation(), deviation, 1.e-18);
}

// --- Sidecar v3 edge identity ---

// --- Position and scale independence ---
//
// The kernel's length tolerances are absolute, so every number recorded on this branch is a
// statement about centimetre-scale geometry near the origin until someone runs the ladder
// somewhere else. Doing that found one defect, and these tests are what keeps it named.

namespace
{
/// The ray/torus quartic exactly as TorusBoundedSurface::appendIntersections builds it, for a
/// torus of the given radii on the origin with axis z. Kept here rather than reaching into the
/// surface class so the test exercises the solver on coefficients a reader can check by hand.
std::array<double, 5> torusRayQuartic(double majorRadius, double minorRadius,
                                      const Point3D& origin, const Point3D& dir)
{
  const double dirDotDir = dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2];
  const double originDotDir = origin[0] * dir[0] + origin[1] * dir[1] + origin[2] * dir[2];
  const double originDotOrigin =
    origin[0] * origin[0] + origin[1] * origin[1] + origin[2] * origin[2];
  const double constantK = majorRadius * majorRadius - minorRadius * minorRadius;
  const double transverseE = dir[0] * dir[0] + dir[1] * dir[1];
  const double transverseF = origin[0] * dir[0] + origin[1] * dir[1];
  const double transverseG = origin[0] * origin[0] + origin[1] * origin[1];
  const double fourRSquared = 4. * majorRadius * majorRadius;
  return {dirDotDir * dirDotDir,
          4. * dirDotDir * originDotDir,
          4. * originDotDir * originDotDir + 2. * dirDotDir * (originDotOrigin + constantK) -
            fourRSquared * transverseE,
          4. * originDotDir * (originDotOrigin + constantK) - 2. * fourRSquared * transverseF,
          (originDotOrigin + constantK) * (originDotOrigin + constantK) -
            fourRSquared * transverseG};
}

/// The offending ray of the x0.1 sweep, given at unit scale; the whole configuration scales with
/// `scale` so the exact solution scales with it too.
std::vector<double> torusRootsAtScale(double scale)
{
  const Point3D origin{2.094269422822338 * scale, 3.292530879918199 * scale,
                       1.9347519602583996 * scale};
  const Point3D dir{-0.7547297076674779, -0.03154700875395883, -0.655276929704412};
  const auto c = torusRayQuartic(2.5 * scale, 0.8 * scale, origin, dir);
  const auto roots = surf::solveQuarticReal(c[0], c[1], c[2], c[3], c[4]);
  return {roots.begin(), roots.end()};
}
} // namespace

BOOST_AUTO_TEST_CASE(StreamE_TorusQuarticIsScaleCovariantWhereItWorks)
{
  // Ferrari's method is exactly scale-covariant: scaling the geometry and the ray origin by k
  // scales every root by k and nothing else. This is the property the whole sweep rests on, so it
  // is asserted rather than assumed.
  const auto reference = torusRootsAtScale(1.);
  BOOST_REQUIRE_EQUAL(reference.size(), 2u);
  for (const double scale : {0.5, 2., 10.}) {
    const auto roots = torusRootsAtScale(scale);
    BOOST_REQUIRE_EQUAL(roots.size(), reference.size());
    for (size_t i = 0; i < roots.size(); ++i) {
      checkClose(roots[i], reference[i] * scale, 1.e-12);
    }
  }
}

BOOST_AUTO_TEST_CASE(StreamE_TorusQuarticKeepsEveryRootBelowTheOldResolventGuard)
{
  // The resolvent guard must be dimensionless: it used to compare a cm^2 quantity against a
  // length tolerance and silently returned zero roots as the geometry shrank. These three scales
  // must all keep finding both roots of a ray that genuinely crosses the torus twice.
  BOOST_CHECK_EQUAL(torusRootsAtScale(0.15).size(), 2u); // was above the old guard
  BOOST_CHECK_EQUAL(torusRootsAtScale(0.12).size(), 2u); // was below it: every root was lost
  BOOST_CHECK_EQUAL(torusRootsAtScale(0.05).size(), 2u);

  // and the roots are exactly the unit-scale ones scaled down, since Ferrari's method is exactly
  // scale-covariant -- the property StreamE_TorusQuarticIsScaleCovariantWhereItWorks asserts
  // above, now that "where it works" is everywhere.
  const auto reference = torusRootsAtScale(1.);
  BOOST_REQUIRE_EQUAL(reference.size(), 2u);
  for (const double factor : {0.15, 0.12, 0.05, 0.01}) {
    const auto roots = torusRootsAtScale(factor);
    BOOST_REQUIRE_EQUAL(roots.size(), reference.size());
    for (size_t i = 0; i < roots.size(); ++i) {
      BOOST_CHECK_LT(std::abs(roots[i] - reference[i] * factor), 1.e-12 * reference[i] * factor);
    }
  }

  // And the roots really are there: the quartic changes sign across each of the two crossings the
  // unit-scale solve found, scaled down. Without this the counts above could be read as the
  // solver merely agreeing with itself.
  const double scale = 0.12;
  const Point3D origin{2.094269422822338 * scale, 3.292530879918199 * scale,
                       1.9347519602583996 * scale};
  const Point3D dir{-0.7547297076674779, -0.03154700875395883, -0.655276929704412};
  const auto c = torusRayQuartic(2.5 * scale, 0.8 * scale, origin, dir);
  const auto evaluate = [&c](double t) {
    return (((c[0] * t + c[1]) * t + c[2]) * t + c[3]) * t + c[4];
  };
  for (const double root : reference) {
    const double t = root * scale;
    const double span = 0.02 * t;
    BOOST_CHECK_LT(evaluate(t - span) * evaluate(t + span), 0.);
  }
}

// --- Gating any TGeoShape ---
//
// The oracle gate could score exactly one thing: an O2BVHSurfaceSolid loaded from a
// surfaces_<part>.bin sidecar. The four scored queries are TGeoShape virtuals, so the scoring
// loop was never actually specific to that class -- only the loading was. These cases pin the
// two halves of removing that restriction:
//
//   1. the `shape_<part>.root` sidecar convention itself (one TGeoShape-derived object under the
//      key "shape"), through the same save/load pair the harness and the fixture generator use,
//      so producer and consumer cannot drift;
//   2. that the oracle validators really are representation-agnostic -- with a *negative
//      control*, because a validator that reports "0 disagreements" for every input would pass a
//      positive-only test while being structurally incapable of failing.

BOOST_AUTO_TEST_CASE(ShapeSidecarRoundTripsAnyTGeoShape)
{
  namespace harness = o2::cad::harness;
  const auto dir = std::filesystem::temp_directory_path();

  // A TGeoBBox that is *not* centred on the origin. The offset is the point: it is the cheapest
  // way for a frame convention to be silently wrong, so it has to survive the round trip.
  double origin[3] = {1.0, 1.5, 2.0};
  TGeoBBox box("shape", 1.0, 1.5, 2.0, origin);

  // A TGeoCompositeShape, which is what the CSG emitter will actually hand over: a 4 cm cube with
  // an r = 0.8 cm axial through-hole. Built from a TGeoBoolNode rather than from a string
  // expression, so no TGeoManager is needed on either side.
  auto* cube = new TGeoBBox("cube", 2.0, 2.0, 2.0);
  auto* drill = new TGeoTube("drill", 0.0, 0.8, 2.5);
  TGeoCompositeShape composite("shape", new TGeoSubtraction(cube, drill, nullptr, nullptr));

  const std::vector<Point3D> probes{{0.5, 0.5, 0.5}, {1.0, 1.5, 2.0}, {3.0, 1.5, 2.0}, {0.0, 0.0, 0.0}, {1.9, 0.0, 0.0}, {0.0, 0.0, 1.9}, {-1.5, -1.5, 1.0}, {0.79, 0.0, 0.0}, {0.81, 0.0, 0.0}};
  const std::vector<Point3D> directions{{1., 0., 0.}, {0., 1., 0.}, {0., 0., 1.}, {-1., 0., 0.}, {0.6, 0.8, 0.}};

  for (const TGeoShape* original : {static_cast<const TGeoShape*>(&box),
                                    static_cast<const TGeoShape*>(&composite)}) {
    const std::string path = (dir / (std::string("o2_shape_sidecar_") + original->ClassName() + ".root")).string();
    std::string error;
    BOOST_REQUIRE_MESSAGE(harness::saveShapeToRootFile(path, *original, &error), error);

    std::unique_ptr<TGeoShape> loaded(harness::loadShapeFromRootFile(path, &error));
    BOOST_REQUIRE_MESSAGE(loaded != nullptr, error);
    BOOST_CHECK_EQUAL(std::string(loaded->ClassName()), std::string(original->ClassName()));
    // TGeoCompositeShape::Capacity() is Monte-Carlo sampled, so this is a loose check by
    // necessity -- which is precisely why the gate does not treat capacity as a column for
    // composites. 5% is far outside the ~1% MC spread and far inside any real error.
    BOOST_CHECK_CLOSE(loaded->Capacity(), original->Capacity(), 5.0);

    // The queries the gate actually scores must be bit-identical across the round trip.
    for (const auto& p : probes) {
      BOOST_CHECK_EQUAL(loaded->Contains(p.data()), original->Contains(p.data()));
      BOOST_CHECK_EQUAL(loaded->Safety(p.data(), original->Contains(p.data())),
                        original->Safety(p.data(), original->Contains(p.data())));
      for (const auto& d : directions) {
        BOOST_CHECK_EQUAL(loaded->DistFromOutside(p.data(), d.data(), 3),
                          original->DistFromOutside(p.data(), d.data(), 3));
        BOOST_CHECK_EQUAL(loaded->DistFromInside(p.data(), d.data(), 3),
                          original->DistFromInside(p.data(), d.data(), 3));
      }
    }
    std::filesystem::remove(path);
  }
}

BOOST_AUTO_TEST_CASE(ShapeSidecarRefusesWhatIsNotAShape)
{
  namespace harness = o2::cad::harness;
  const auto dir = std::filesystem::temp_directory_path();
  std::string error;

  BOOST_CHECK(harness::loadShapeFromRootFile((dir / "o2_shape_absent.root").string(), &error) == nullptr);
  BOOST_CHECK(!error.empty());

  // A well-formed ROOT file whose "shape" key holds something else must be refused rather than
  // silently ignored: an emitter that writes the wrong object would otherwise look like an
  // emitter that wrote nothing, and the part would quietly lose its column.
  const std::string path = (dir / "o2_shape_not_a_shape.root").string();
  {
    TFile out(path.c_str(), "RECREATE");
    TNamed impostor("shape", "not a shape");
    out.WriteTObject(&impostor, "shape");
    out.Close();
  }
  error.clear();
  BOOST_CHECK(harness::loadShapeFromRootFile(path, &error) == nullptr);
  BOOST_CHECK(!error.empty());
  std::filesystem::remove(path);
}

BOOST_AUTO_TEST_CASE(OracleValidatorsScoreAPlainRootShape)
{
  namespace harness = o2::cad::harness;

  // A ROOT primitive with no connection to O2BVHSurfaceSolid at all, and a *wrong* copy of it:
  // same shape, radius 0.05 cm too large. Everything below is asserted twice, once for each, so
  // no "0 disagreements" here can come from a validator that is unable to report anything else.
  constexpr double kR = 1.5;
  constexpr double kZ = 2.0;
  constexpr double kError = 0.05;
  const TGeoTube truth("truth", 0., kR, kZ);
  const TGeoTube wrong("wrong", 0., kR + kError, kZ);

  // The oracle columns, built analytically from the tube's own closed form rather than from
  // either shape's methods, so `contains` and `safety` are genuinely independent of what is being
  // scored. (The two distance columns below are taken from `truth`, which is independent of
  // `wrong` -- the case that has to be able to fail.)
  const auto trueContains = [](const Point3D& p) {
    return (std::hypot(p[0], p[1]) <= kR && std::fabs(p[2]) <= kZ) ? 1 : 0;
  };
  const auto trueBoundaryDistance = [](const Point3D& p) {
    const double r = std::hypot(p[0], p[1]);
    const double dr = kR - r;
    const double dz = kZ - std::fabs(p[2]);
    if (dr > 0. && dz > 0.) {
      return std::min(dr, dz);
    }
    return std::hypot(std::max(r - kR, 0.), std::max(std::fabs(p[2]) - kZ, 0.));
  };

  std::vector<Point3D> points;
  std::vector<int> containsState;
  std::vector<double> boundaryDistance;
  for (int ix = -6; ix <= 6; ++ix) {
    for (int iy = -6; iy <= 6; ++iy) {
      for (int iz = -4; iz <= 4; ++iz) {
        const Point3D p{0.31 * ix, 0.29 * iy, 0.53 * iz};
        // Points nearer the wall than the wrong shape's error would be legitimately ambiguous
        // for it, so they are dropped: the negative control has to fail on geometry, not on the
        // band. Points in the annulus the two shapes disagree about are deliberately kept.
        if (std::fabs(trueBoundaryDistance(p)) < 1.e-3) {
          continue;
        }
        points.push_back(p);
        containsState.push_back(trueContains(p));
        boundaryDistance.push_back(trueBoundaryDistance(p));
      }
    }
  }
  BOOST_REQUIRE_GT(points.size(), 500u);

  harness::ValidationOptions opt;
  opt.meshBand = 1.e-6; // a synthetic shape has no modelling tolerance to hide behind
  opt.distanceTolerance = 1.e-9;

  auto containsTruth = harness::validateContainsAgainstOracle(&truth, points, containsState,
                                                              boundaryDistance, opt);
  auto containsWrong = harness::validateContainsAgainstOracle(&wrong, points, containsState,
                                                              boundaryDistance, opt);
  BOOST_CHECK_EQUAL(containsTruth.nMismatchUnexplained + containsTruth.nMismatchMissedSurface, 0u);
  BOOST_CHECK_GT(containsWrong.nMismatchUnexplained + containsWrong.nMismatchMissedSurface, 0u);

  auto safetyTruth = harness::validateSafetyAgainstOracle(&truth, points, boundaryDistance, opt);
  auto safetyWrong = harness::validateSafetyAgainstOracle(&wrong, points, boundaryDistance, opt);
  BOOST_CHECK_EQUAL(safetyTruth.nMismatchUnexplained + safetyTruth.nMismatchMissedSurface, 0u);
  BOOST_CHECK_GT(safetyWrong.nMismatchUnexplained + safetyWrong.nMismatchMissedSurface, 0u);

  // Rays from well outside, aimed at points spread through the tube, so the hit rate is not
  // degenerate; the oracle distance is the nearest positive crossing, exactly as occtOracle.py
  // defines it, and the origin classification decides which TGeo entry point is asked.
  std::vector<harness::Ray> rays;
  std::vector<double> rayDistance;
  std::vector<int> originState;
  for (int i = 0; i < 400; ++i) {
    const double phi = 0.0173 * i;
    const double z = -1.9 + 0.0095 * i;
    const Point3D target{0.9 * kR * std::cos(2.1 * phi), 0.9 * kR * std::sin(2.1 * phi), z};
    const Point3D origin{5.0 * std::cos(phi), 5.0 * std::sin(phi), 3.0 - 0.01 * i};
    Point3D dir{target[0] - origin[0], target[1] - origin[1], target[2] - origin[2]};
    const double norm = std::sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
    for (auto& component : dir) {
      component /= norm;
    }
    rays.push_back(harness::Ray{origin, dir});
    rayDistance.push_back(truth.DistFromOutside(origin.data(), dir.data(), 3));
    originState.push_back(trueContains(origin));
  }

  auto distTruth = harness::validateDistanceAgainstOracle(&truth, rays, rayDistance,
                                                          /*wantInside=*/false, opt, originState);
  auto distWrong = harness::validateDistanceAgainstOracle(&wrong, rays, rayDistance,
                                                          /*wantInside=*/false, opt, originState);
  BOOST_CHECK_EQUAL(distTruth.nMismatchUnexplained + distTruth.nMismatchMissedSurface, 0u);
  BOOST_CHECK_GT(distWrong.nMismatchUnexplained + distWrong.nMismatchMissedSurface, 0u);
}
// --- The CSG emitter's two ROOT-side load-bearing claims ---
//
// The emitter itself is Python (Detectors/CADSupport/tools/cadsupport), and its own self-tests live there. What
// belongs here are the two properties of *ROOT* that the emitted file silently depends on. If a
// future ROOT changes either, every CSG part written by this project becomes wrong geometry that
// still loads, and nothing else in the suite would notice.

namespace
{
// Build the emitter's former `placed(primitive, M)` idiom: no TGeoShape in ROOT 6.36 can carry a
// rigid transform (TGeoBBox has fOrigin and nothing else does), and TGeoCompositeShape is the only
// shape that holds a TGeoMatrix at all -- through its TGeoBoolNode, which needs two operands. So a
// recognised tube that was not already on the z axis USED TO BE written as the union of the
// primitive with an identical copy of itself under the same matrix.
//
// It is now written as the bare primitive plus a placement instead. This helper stays, because
// the self-union is
// still exactly the same point set and is therefore the reference the new emission is measured
// against -- see PlacedPrimitiveAnswersExactlyLikeTheSelfUnionComposite.
TGeoCompositeShape* makePlacedTube(const char* name, double rmin, double rmax, double dz,
                                   TGeoMatrix* matrixA, TGeoMatrix* matrixB)
{
  auto* left = new TGeoTube(Form("%s_l", name), rmin, rmax, dz);
  auto* right = new TGeoTube(Form("%s_r", name), rmin, rmax, dz);
  auto* node = new TGeoUnion(left, right, matrixA, matrixB);
  return new TGeoCompositeShape(name, node);
}
} // namespace

BOOST_AUTO_TEST_CASE(CsgSelfUnionCarriesARigidTransformExactly)
{
  // A tube on an axis that is neither a coordinate axis nor through the origin -- i.e. the
  // ExcavatorArm case. Every query on the composite must equal the same query on the bare primitive
  // asked in the primitive's own frame, exactly, not within a band.
  constexpr double kRmin = 0.4;
  constexpr double kRmax = 1.0;
  constexpr double kDz = 5.0;
  const TGeoTube reference("reference", kRmin, kRmax, kDz);

  auto* rotation = new TGeoRotation("csgRot", 0., 0., 0.);
  rotation->RotateX(30.);
  rotation->RotateZ(17.);
  auto* matrixA = new TGeoCombiTrans(0.3, 5.916, 2.0, rotation);
  auto* matrixB = new TGeoCombiTrans(0.3, 5.916, 2.0, rotation);
  const TGeoCombiTrans placement(0.3, 5.916, 2.0, rotation);
  std::unique_ptr<TGeoCompositeShape> placed(
    makePlacedTube("csgPlaced", kRmin, kRmax, kDz, matrixA, matrixB));

  std::size_t probes = 0;
  std::size_t inside = 0;
  std::size_t outside = 0;
  for (int ix = -8; ix <= 8; ++ix) {
    for (int iy = -8; iy <= 8; ++iy) {
      for (int iz = -8; iz <= 8; ++iz) {
        const Point3D master{0.3 + 0.37 * ix, 5.916 + 0.41 * iy, 2.0 + 0.43 * iz};
        Point3D local{};
        placement.MasterToLocal(master.data(), local.data());
        // A point on the wall is decided by floating-point luck on either side; skip a thin
        // shell so the check tests geometry rather than tie-breaking.
        const double r = std::hypot(local[0], local[1]);
        if (std::fabs(r - kRmin) < 1.e-9 || std::fabs(r - kRmax) < 1.e-9 ||
            std::fabs(std::fabs(local[2]) - kDz) < 1.e-9) {
          continue;
        }
        ++probes;
        const bool wanted = reference.Contains(local.data());
        BOOST_REQUIRE_EQUAL(placed->Contains(master.data()), wanted);
        BOOST_REQUIRE_CLOSE_FRACTION(placed->Safety(master.data(), wanted),
                                     reference.Safety(local.data(), wanted), 1.e-12);
        wanted ? ++inside : ++outside;

        for (const auto& dir : {Point3D{1., 0., 0.}, Point3D{0., 1., 0.}, Point3D{0., 0., 1.},
                                Point3D{0.5773502691896258, 0.5773502691896258, 0.5773502691896258}}) {
          Point3D localDir{};
          placement.MasterToLocalVect(dir.data(), localDir.data());
          if (wanted) {
            BOOST_REQUIRE_CLOSE_FRACTION(placed->DistFromInside(master.data(), dir.data(), 3),
                                         reference.DistFromInside(local.data(), localDir.data(), 3),
                                         1.e-12);
          } else {
            const double got = placed->DistFromOutside(master.data(), dir.data(), 3);
            const double want = reference.DistFromOutside(local.data(), localDir.data(), 3);
            if (want > 1.e20) {
              BOOST_REQUIRE_GT(got, 1.e20);
            } else {
              BOOST_REQUIRE_CLOSE_FRACTION(got, want, 1.e-12);
            }
          }
        }
      }
    }
  }
  // A check that cannot fail is not a check: both classes must actually be populated.
  BOOST_CHECK_GT(probes, 2000u);
  BOOST_CHECK_GT(inside, 100u);
  BOOST_CHECK_GT(outside, 100u);

  // The negative half. The same comparison against a primitive 0.05 cm too wide must disagree,
  // otherwise the loop above proves nothing about the transform.
  // The probes are placed *in the shell the two disagree about* and then mapped out to the
  // master frame, rather than being taken from a lattice that might miss a 0.05 cm shell.
  const TGeoTube wrong("wrongReference", kRmin, kRmax + 0.05, kDz);
  std::size_t disagreements = 0;
  std::size_t shellProbes = 0;
  for (int iphi = 0; iphi < 24; ++iphi) {
    const double phi = 2. * M_PI * iphi / 24.;
    for (int iz = -3; iz <= 3; ++iz) {
      const double radius = kRmax + 0.025;
      const Point3D local{radius * std::cos(phi), radius * std::sin(phi), 1.3 * iz};
      Point3D master{};
      placement.LocalToMaster(local.data(), master.data());
      ++shellProbes;
      disagreements += (placed->Contains(master.data()) != wrong.Contains(local.data())) ? 1 : 0;
    }
  }
  BOOST_CHECK_EQUAL(disagreements, shellProbes);
}

BOOST_AUTO_TEST_CASE(CsgTwoLeafUnionRoundTripsAndMatchesTheClosedForm)
{
  // The ExcavatorArm ram, in miniature and in closed form: an eye (a tube on x) plus a rod (a solid
  // cylinder on z), which is what `tier2-tube-union` emits. The union is checked against the
  // membership function of the two cylinders written out by hand, which depends on neither ROOT
  // shape, and then the whole composite is pushed through the shape sidecar and checked again --
  // so a streaming defect that dropped a bool node's matrix would be caught here rather than in
  // a gate run three steps later.
  constexpr double kEyeRmin = 0.7;
  constexpr double kEyeRmax = 1.2;
  constexpr double kEyeDz = 0.75;
  constexpr double kRodR = 0.6;
  constexpr double kRodDz = 3.5;
  constexpr double kRodCentre = 3.5; // rod spans z in [0, 7]

  auto* eyeRotation = new TGeoRotation("csgEyeRot", 90., 90., 0.); // local z -> global x
  auto* eyeMatrix = new TGeoCombiTrans(0., 0., 0., eyeRotation);
  auto* rodMatrix = new TGeoTranslation(0., 0., kRodCentre);
  auto* eye = new TGeoTube("csgEye", kEyeRmin, kEyeRmax, kEyeDz);
  auto* rod = new TGeoTube("csgRod", 0., kRodR, kRodDz);
  auto* node = new TGeoUnion(eye, rod, eyeMatrix, rodMatrix);
  std::unique_ptr<TGeoCompositeShape> ram(new TGeoCompositeShape("csgRam", node));

  const auto closedForm = [&](const Point3D& p) {
    const double rEye = std::hypot(p[1], p[2]);
    const bool inEye = rEye >= kEyeRmin && rEye <= kEyeRmax && std::fabs(p[0]) <= kEyeDz;
    const double rRod = std::hypot(p[0], p[1]);
    const bool inRod = rRod <= kRodR && p[2] >= 0. && p[2] <= 2. * kRodDz;
    return inEye || inRod;
  };
  const auto nearWall = [&](const Point3D& p) {
    const double rEye = std::hypot(p[1], p[2]);
    const double rRod = std::hypot(p[0], p[1]);
    return std::fabs(rEye - kEyeRmin) < 1.e-9 || std::fabs(rEye - kEyeRmax) < 1.e-9 ||
           std::fabs(std::fabs(p[0]) - kEyeDz) < 1.e-9 || std::fabs(rRod - kRodR) < 1.e-9 ||
           std::fabs(p[2]) < 1.e-9 || std::fabs(p[2] - 2. * kRodDz) < 1.e-9;
  };

  const std::filesystem::path path =
    std::filesystem::temp_directory_path() / "o2_csg_ram_shape.root";
  namespace harness = o2::cad::harness;
  std::string error;
  BOOST_REQUIRE_MESSAGE(harness::saveShapeToRootFile(path.string(), *ram, &error), error);
  std::unique_ptr<TGeoShape> loaded(harness::loadShapeFromRootFile(path.string(), &error));
  BOOST_REQUIRE_MESSAGE(loaded != nullptr, error);
  BOOST_CHECK_EQUAL(std::string(loaded->ClassName()), std::string("TGeoCompositeShape"));

  std::size_t inside = 0;
  std::size_t outside = 0;
  for (int ix = -6; ix <= 6; ++ix) {
    for (int iy = -6; iy <= 6; ++iy) {
      for (int iz = -4; iz <= 20; ++iz) {
        const Point3D p{0.23 * ix, 0.27 * iy, 0.41 * iz};
        if (nearWall(p)) {
          continue;
        }
        const bool wanted = closedForm(p);
        BOOST_REQUIRE_EQUAL(ram->Contains(p.data()), wanted);
        BOOST_REQUIRE_EQUAL(loaded->Contains(p.data()), wanted);
        wanted ? ++inside : ++outside;
      }
    }
  }
  BOOST_CHECK_GT(inside, 50u);
  BOOST_CHECK_GT(outside, 500u);

  // Capacity is Monte-Carlo for a composite, so it is *reported* and never gated.
  // The assertion is stated as scatter rather than as accuracy, because scatter needs no exact
  // volume and is the sharper statement: repeated calls returning *different* answers prove the
  // method is sampled, and a spread four orders of magnitude above the gate's 1e-6 band proves
  // that no capacity criterion could ever be applied to a shape written this way. If a future
  // ROOT made TGeoCompositeShape::Capacity() analytic this test would fail, which is the right
  // outcome: the emitter's acceptance policy would then be worth revisiting.
  double minCapacity = ram->Capacity();
  double maxCapacity = minCapacity;
  for (int i = 0; i < 5; ++i) {
    const double sampled = ram->Capacity();
    minCapacity = std::min(minCapacity, sampled);
    maxCapacity = std::max(maxCapacity, sampled);
  }
  BOOST_CHECK_GT(minCapacity, 0.);
  const double spread = (maxCapacity - minCapacity) / (0.5 * (maxCapacity + minCapacity));
  BOOST_CHECK_GT(spread, 1.e-4);

  std::filesystem::remove(path);
}
// --- The CSG emitter's ROOT-side claims ---

// ============================================================================================
// X-ray / geantino transport -- ordered crossing lists
// ============================================================================================
//
// Everything above this block, and everything the oracle gate measures, is a SINGLE-SHOT query:
// from a point, how far to the surface. A transport loop is different in kind -- step, land on
// the boundary, step again from there -- and its failure modes (a zero-length step, a particle
// that enters and never leaves, a crossing found twice, a step that overshoots) cannot be
// expressed as a disagreement on DistFromOutside from an interior sample. These cases pin the
// properties the X-ray benchmark rests on.
//
// They include XRayTransport.h, which is the SAME header the benchmark binary steps with. That
// is deliberate: a test written against a second implementation of the same idea tests neither.

#include "XRayTransport.h"

using namespace o2::cad::xray;
using XRayPoint = o2::cad::harness::Point3D;

/// A box has exactly two crossings and a hollow tube has four -- and the second fact is the one
/// no single-shot query can express, because DistFromOutside reports the first of the four and
/// stops. Both distances are known in closed form, so this needs no oracle and no fixture.
BOOST_AUTO_TEST_CASE(XRayCrossingListsMatchClosedFormOnPrimitives)
{
  StepConfig cfg;
  Robustness stats;
  const XRayPoint origin{-5., 0., 0.};
  const XRayPoint dir{1., 0., 0.};

  TGeoBBox box("xrayBox", 1., 1.5, 2.);
  const auto boxCrossings = stepWithShapeApi(&box, origin, dir, 10., cfg, stats);
  BOOST_REQUIRE_EQUAL(boxCrossings.size(), 2u);
  BOOST_CHECK_SMALL(boxCrossings[0].t - 4., 1.e-12);
  BOOST_CHECK_SMALL(boxCrossings[1].t - 6., 1.e-12);
  BOOST_CHECK_EQUAL(boxCrossings[0].kind, +1);
  BOOST_CHECK_EQUAL(boxCrossings[1].kind, -1);

  TGeoTube tube("xrayTube", 0.5, 1.0, 2.0);
  const auto tubeCrossings = stepWithShapeApi(&tube, origin, dir, 10., cfg, stats);
  BOOST_REQUIRE_EQUAL(tubeCrossings.size(), 4u);
  const double expected[4] = {4.0, 4.5, 5.5, 6.0};
  const int senses[4] = {+1, -1, +1, -1};
  for (int i = 0; i < 4; ++i) {
    BOOST_CHECK_SMALL(tubeCrossings[i].t - expected[i], 1.e-12);
    BOOST_CHECK_EQUAL(tubeCrossings[i].kind, senses[i]);
  }
  BOOST_CHECK_EQUAL(stats.zeroLengthSteps, 0);
  BOOST_CHECK_EQUAL(stats.nonAdvancingSteps, 0);
  BOOST_CHECK_EQUAL(stats.unstickPushes, 0);
}

/// The transport-level BVH == _Loop guard.
///
/// `DistanceBVHMatchesLoopOnAllFixtures` above compares the twins one query at a time from
/// generated points. This compares whole ORDERED CROSSING LISTS produced by stepping, where every
/// query after the first starts from a point the previous query put on a boundary. That is a
/// harder condition and a different one: a traversal-order difference that is invisible on an
/// isolated query can still send the two loops down different sequences of states.
BOOST_AUTO_TEST_CASE(XRayCrossingListsAgreeBetweenBVHAndLoopOnAllFixtures)
{
  StepConfig cfg;
  std::array<std::pair<std::unique_ptr<SurfaceSolid>, double>, 7> fixtures{{
    {makeBoxSolid("xrayLoopBox", 1., 2., 3.), 4.},
    {makeTubeSolid("xrayLoopTube", 0., 2., 3.), 4.},
    {makeTubeSolid("xrayLoopHollowTube", 1., 2., 3.), 4.},
    {makeConeSolid("xrayLoopCone", 2., 1., 3.), 4.},
    {makeSphereSolid("xrayLoopSphere", 2.5), 3.5},
    {makeTorusSolid("xrayLoopTorus", 3., 1.), 4.5},
    {makeCapsuleSolid("xrayLoopCapsule", 1., 1.5), 3.},
  }};
  size_t comparedRays = 0;
  size_t comparedCrossings = 0;
  for (const auto& [solid, extent] : fixtures) {
    BOOST_TEST_CONTEXT("fixture = " << solid->GetName())
    {
      BOOST_REQUIRE(solid->HasBVH());
      const XRayPoint lo{-extent, -extent, -extent};
      const XRayPoint hi{extent, extent, extent};
      // A fan rather than the three axes: a parallel beam is direction-poor, and the point of this
      // case is to exercise many ray/surface configurations per fixture.
      const Raster raster = buildRaster(lo, hi, 9, buildFanBeams(11), 0.);
      for (const auto& ray : raster.rays) {
        Robustness bvhStats;
        Robustness loopStats;
        const auto viaBVH =
          stepWithShapeApi(solid.get(), ray.origin, ray.dir, ray.tMax, cfg, bvhStats);
        // The non-BVH twins, stepped through the identical loop: only the traversal differs.
        const auto viaLoop = stepCrossingsWithKernels(
          ray.origin, ray.dir, ray.tMax, cfg, loopStats,
          [&solid](const double* p) { return solid->Contains_Loop(p); },
          [&solid](const double* p, const double* d) { return solid->DistFromOutside_Loop(p, d); },
          [&solid](const double* p, const double* d) { return solid->DistFromInside_Loop(p, d); });
        BOOST_REQUIRE_EQUAL(viaBVH.size(), viaLoop.size());
        for (size_t i = 0; i < viaBVH.size(); ++i) {
          BOOST_CHECK_EQUAL(viaBVH[i].kind, viaLoop[i].kind);
          // Bit-identical is the contract: both minimise over the same hits from the same kernels.
          BOOST_CHECK_EQUAL(viaBVH[i].t, viaLoop[i].t);
        }
        comparedRays += 1;
        comparedCrossings += viaBVH.size();
      }
    }
  }
  BOOST_CHECK_GT(comparedRays, 2000u);
  BOOST_CHECK_GT(comparedCrossings, 2000u);
}

/// The comparator's own positive AND negative controls. A comparison that cannot fail is not a
/// comparison, and the distinction this one has to preserve is LOST (a wall a track walks
/// through) against DISPLACED (a wrong step length) -- merging them was the first version's bug.
BOOST_AUTO_TEST_CASE(XRayCrossingComparatorCatchesInjectedDefects)
{
  const std::vector<Crossing> truth{{4.0, +1}, {4.5, -1}, {5.5, +1}, {6.0, -1}};
  const double tolerance = 1.e-6;

  ListComparison clean;
  compareLists(truth, truth, {}, {}, tolerance, clean);
  BOOST_CHECK_EQUAL(clean.raysIdentical, 1);
  BOOST_CHECK_EQUAL(clean.matched, 4);
  BOOST_CHECK_EQUAL(clean.missing, 0);
  BOOST_CHECK_EQUAL(clean.extra, 0);
  BOOST_CHECK_EQUAL(clean.displaced, 0);

  auto perturbed = truth;
  perturbed[2].t += 1.e-3;
  ListComparison displaced;
  compareLists(perturbed, truth, {}, {}, tolerance, displaced);
  BOOST_CHECK_EQUAL(displaced.raysIdentical, 0);
  BOOST_CHECK_EQUAL(displaced.displaced, 1);
  BOOST_CHECK_EQUAL(displaced.missing, 0); // a moved crossing is NOT a lost one
  BOOST_CHECK_EQUAL(displaced.extra, 0);
  BOOST_CHECK_SMALL(displaced.worstDeltaT - 1.e-3, 1.e-12);

  auto dropped = truth;
  dropped.erase(dropped.begin() + 1);
  ListComparison lost;
  compareLists(dropped, truth, {}, {}, tolerance, lost);
  BOOST_CHECK_EQUAL(lost.missing, 1);
  BOOST_CHECK_EQUAL(lost.extra, 0);

  auto doubled = truth;
  doubled.insert(doubled.begin() + 1, {4.2, -1});
  ListComparison spurious;
  compareLists(doubled, truth, {}, {}, tolerance, spurious);
  BOOST_CHECK_EQUAL(spurious.extra, 1);
  BOOST_CHECK_EQUAL(spurious.missing, 0);

  auto flipped = truth;
  flipped[1].kind = +1;
  ListComparison sense;
  compareLists(flipped, truth, {}, {}, tolerance, sense);
  BOOST_CHECK_EQUAL(sense.kindMismatch, 1);

  // A crossing moved by LESS than the tolerance must not be reported at all, or every run would
  // drown in last-digit noise.
  auto nudged = truth;
  nudged[0].t += 1.e-9;
  ListComparison quiet;
  compareLists(nudged, truth, {}, {}, tolerance, quiet);
  BOOST_CHECK_EQUAL(quiet.raysIdentical, 1);
  BOOST_CHECK_EQUAL(quiet.displaced, 0);
}

/// The parity audit is the only check in the benchmark that is independent of the stepping: both
/// modes produce an alternating list by construction, so `nonAlternating` can never fire on them.
/// Asking Contains() at the midpoint of every interval is what can contradict a list.
BOOST_AUTO_TEST_CASE(XRayParityAuditContradictsATruncatedList)
{
  StepConfig cfg;
  TGeoBBox box("xrayParityBox", 1., 1., 1.);
  const XRayPoint origin{-5., 0., 0.};
  const XRayPoint dir{1., 0., 0.};

  Robustness good;
  auditCrossingList({{4.0, +1}, {6.0, -1}}, &box, origin, dir, 10., cfg, good);
  BOOST_CHECK_EQUAL(good.parityMismatchIntervals, 0);
  BOOST_CHECK_EQUAL(good.oddCrossingLists, 0);
  BOOST_CHECK_SMALL(good.insideLength - 2., 1.e-12);

  Robustness truncated;
  auditCrossingList({{4.0, +1}}, &box, origin, dir, 10., cfg, truncated);
  BOOST_CHECK_GT(truncated.parityMismatchIntervals, 0);
  BOOST_CHECK_EQUAL(truncated.oddCrossingLists, 1);

  Robustness invented;
  auditCrossingList({{1.0, +1}, {2.0, -1}, {4.0, +1}, {6.0, -1}}, &box, origin, dir, 10., cfg,
                    invented);
  BOOST_CHECK_GT(invented.parityMismatchIntervals, 0);
}

/// The chord integral is EXACT for an axis-aligned box whose raster window is its own bounding
/// box, at every raster density. No convergence argument and no tolerance: either the quadrature
/// is the volume or it is not. This is what fixed the raster geometry -- with the window inflated
/// by 2 % instead, the same box came out 5.1e-02 too large at N = 32.
BOOST_AUTO_TEST_CASE(XRayChordIntegralIsExactForABoxAndConvergesForASphere)
{
  StepConfig cfg;
  TGeoBBox box("xrayVolBox", 1., 1.5, 2.);
  for (const int n : {5, 16, 41}) {
    const Raster raster = buildRaster({-1., -1.5, -2.}, {1., 1.5, 2.}, n, buildBeams("xyz", 0.), 0.);
    Robustness stats;
    std::vector<double> byBeam(raster.beams.size(), 0.);
    for (const auto& ray : raster.rays) {
      const double before = stats.insideLength;
      const auto crossings = stepWithShapeApi(&box, ray.origin, ray.dir, ray.tMax, cfg, stats);
      auditCrossingList(crossings, nullptr, ray.origin, ray.dir, ray.tMax, cfg, stats);
      byBeam[ray.beam] += stats.insideLength - before;
    }
    BOOST_CHECK_SMALL(chordVolume(raster, byBeam) - 24., 1.e-9);
  }

  // A curved silhouette cannot be exact at finite N. The bound below is the MEASURED envelope
  // over N = 24..192 (2e-3), not a convergence rate -- the convergence is NOT monotone in N,
  // because the silhouette cells realign with the lattice at every density. That is the reason
  // this benchmark's volume is quoted with its raster density and never extrapolated.
  TGeoSphere sphere("xrayVolSphere", 0., 1.);
  const double exact = 4. / 3. * 3.14159265358979323846;
  for (const int n : {24, 96}) {
    const Raster raster = buildRaster({-1., -1., -1.}, {1., 1., 1.}, n, buildBeams("z", 0.), 0.);
    Robustness stats;
    for (const auto& ray : raster.rays) {
      const auto crossings = stepWithShapeApi(&sphere, ray.origin, ray.dir, ray.tMax, cfg, stats);
      auditCrossingList(crossings, nullptr, ray.origin, ray.dir, ray.tMax, cfg, stats);
    }
    const double volume = stats.insideLength * raster.cellArea[0];
    BOOST_CHECK_LT(std::fabs(volume - exact) / exact, 2.e-3);
  }
}

/// The raster's own contract, because every number above depends on it: the rays start strictly
/// outside the solid, the lattice covers the bounding box, and a fan is direction-diverse where
/// the axis beams are not. The last property is not cosmetic -- it is why the fan finds the torus
/// quartic defect at x0.1 and the three axis beams do not.
BOOST_AUTO_TEST_CASE(XRayRasterRaysStartOutsideAndFansAreDirectionDiverse)
{
  TGeoBBox box("xrayRasterBox", 1., 1.5, 2.);
  const Raster raster = buildRaster({-1., -1.5, -2.}, {1., 1.5, 2.}, 8, buildBeams("xyz", 0.), 0.);
  BOOST_CHECK_EQUAL(raster.rays.size(), 3u * 8u * 8u);
  for (const auto& ray : raster.rays) {
    BOOST_REQUIRE(!box.Contains(ray.origin.data()));
    // and the far end must be outside too, so the window really does bracket the solid
    const double end[3] = {ray.origin[0] + ray.tMax * ray.dir[0],
                           ray.origin[1] + ray.tMax * ray.dir[1],
                           ray.origin[2] + ray.tMax * ray.dir[2]};
    BOOST_REQUIRE(!box.Contains(end));
  }

  const auto axes = buildBeams("xyz", 0.);
  const auto fan = buildFanBeams(64);
  BOOST_CHECK_EQUAL(axes.size(), 3u);
  BOOST_CHECK_EQUAL(fan.size(), 64u);
  for (const auto& beams : {axes, fan}) {
    for (const auto& beam : beams) {
      BOOST_CHECK_SMALL(dot3(beam.dir, beam.dir) - 1., 1.e-12);
      BOOST_CHECK_SMALL(dot3(beam.u, beam.v), 1.e-12);
      BOOST_CHECK_SMALL(dot3(beam.u, beam.dir), 1.e-12);
      BOOST_CHECK_SMALL(dot3(beam.v, beam.dir), 1.e-12);
    }
  }
  // Direction diversity, stated as a number: the axis beams are mutually orthogonal and nothing
  // else, while no two fan beams are closer than a few degrees and they span the sphere.
  double worstFanAlignment = -1.;
  for (size_t i = 0; i < fan.size(); ++i) {
    for (size_t j = i + 1; j < fan.size(); ++j) {
      worstFanAlignment = std::max(worstFanAlignment, std::fabs(dot3(fan[i].dir, fan[j].dir)));
    }
  }
  BOOST_CHECK_LT(worstFanAlignment, 0.999);
}
// --- X-ray / geantino transport benchmark ---

// --- Dimensionally consistent guards in the quartic root solver ---
//
// solveQuarticReal used to decide all three of its branches with kTolerance -- 1e-9 *cm*, a
// length -- applied to quantities that are not lengths:
//
//   |termQ|      <= kTolerance   selects the biquadratic branch;  termQ scales as L^3
//   resolvent    >  kTolerance   licenses Ferrari's second stage; resolvent scales as L^2
//   |derivative| >  kTolerance   licenses a Newton polishing step; the derivative scales as L^3
//
// Two consequences:
//
//   * the resolvent guard fails and the function returns the EMPTY root set, so a ray silently
//     misses a torus it does cross;
//   * the termQ guard misroutes an asymmetric quartic into the biquadratic branch -- which
//     *assumes* termQ = 0 and forces the roots to be symmetric about -b/4 -- so it returns
//     confidently wrong roots instead of the right ones. That is worse than a miss, because a
//     miss at least leaves a visible gap.
//
// The trigger is the ratio of the ray's lever arm to the feature it hits, not the model's scale:
// the reproducer below is real, *unscaled* ALICE3 geometry, a ray 375 cm from a 0.1 cm tube.
//
// These cases pin the repair from both sides. It is not enough that the previously-failing case
// now works: the branches exist for real reasons, so the biquadratic branch must still be
// *selected* for a true biquadratic, and both branches must still *decline* a configuration that
// genuinely has no real roots. A guard that always passes would satisfy neither.

namespace
{
/// The relative backward error of \a x as a root of a4 x^4 + ... + a0: |p(x)| divided by the sum
/// of the magnitudes of the terms that produced it. Scale-free, so it means the same thing for a
/// torus 400 cm away and one 0.1 cm across, which is the whole point of this block.
double quarticBackwardError(const std::array<double, 5>& coefficients, double x)
{
  double value = 0., magnitude = 0., power = 1.;
  for (int i = 0; i < 5; ++i) {
    const double term = coefficients[4 - i] * power;
    value += term;
    magnitude += std::abs(term);
    power *= x;
  }
  return magnitude > 0. ? std::abs(value) / magnitude : std::abs(value);
}

/// The monic quartic with exactly these four real roots, from the elementary symmetric functions.
std::array<double, 5> quarticFromRoots(double r1, double r2, double r3, double r4)
{
  return {1., -(r1 + r2 + r3 + r4),
          r1 * r2 + r1 * r3 + r1 * r4 + r2 * r3 + r2 * r4 + r3 * r4,
          -(r1 * r2 * r3 + r1 * r2 * r4 + r1 * r3 * r4 + r2 * r3 * r4), r1 * r2 * r3 * r4};
}

std::vector<double> sortedRoots(const std::array<double, 5>& c, surf::QuarticBranch* branch = nullptr)
{
  const auto found = surf::solveQuarticReal(c[0], c[1], c[2], c[3], c[4], branch);
  std::vector<double> roots(found.begin(), found.end());
  std::sort(roots.begin(), roots.end());
  return roots;
}

/// Every returned root must actually be a root, to the precision of the coefficients themselves.
void checkRootsAreRoots(const std::array<double, 5>& c, const std::vector<double>& roots)
{
  for (const double root : roots) {
    BOOST_CHECK_LT(quarticBackwardError(c, root), 1.e-12);
  }
}
} // namespace

BOOST_AUTO_TEST_CASE(StreamM_QuarticFindsTheALICE3ProductionScaleRoots)
{
  // ALICE3 part ST2487462_01, face 47: a torus of R = 5.3 cm and
  // r = 0.1 cm, hit by a ray whose origin is 375 cm away. The crossing lies on the untrimmed
  // surface to 1.7e-14 cm and inside both parameter windows -- the patch is there and the trim
  // admits it -- and the solver returned nothing.
  //
  // It is a *biquadratic* (the ray is perpendicular to the torus axis, so the true termQ is 0),
  // but termQ is evaluated as d - b*c/2 + b^3/8 from terms of magnitude ~1e8 and cancels to
  // -5.96e-08 rather than to 0. That is above the absolute 1e-9 test, so the quartic was routed
  // into the resolvent branch, whose resolvent is 7.1e-15 and fails its own absolute test.
  const std::array<double, 5> c{1.0, -1501.7280000044018, 845808.25396968238, -211752288.545858,
                                19882619385.616932};

  // The reference roots are Newton's method run to convergence on the *exact* binary values of
  // those five coefficients in 60-digit decimal arithmetic, so they are the truth for this input
  // and not another double-precision solve.
  const double firstRoot = 375.3392295779947145;
  const double secondRoot = 375.5247704240909448;

  // The tolerance is not arbitrary. p'(firstRoot) = -14.47, so one ulp of a0 (3.8e-06 at 1.99e10)
  // moves this root by 2.6e-07 cm: the input coefficients do not determine the roots better than
  // that. 1e-06 cm is a few times the conditioning limit and four orders below the 0.1 cm tube
  // whose crossing this is.
  surf::QuarticBranch branch = surf::QuarticBranch::NotAQuartic;
  const auto roots = sortedRoots(c, &branch);
  BOOST_REQUIRE_EQUAL(roots.size(), 2u);
  checkClose(roots[0], firstRoot, 1.e-6);
  checkClose(roots[1], secondRoot, 1.e-6);
  checkRootsAreRoots(c, roots);

  // and it must get there by recognising the biquadratic, not by luck in the resolvent branch
  BOOST_CHECK(branch == surf::QuarticBranch::Biquadratic);
}

BOOST_AUTO_TEST_CASE(StreamM_QuarticIsScaleInvariantOnAnAsymmetricQuartic)
{
  // A thoroughly well-conditioned quartic with four simple real roots {1, 2, 3, 7}, uniformly
  // scaled. Ferrari's method is exactly scale-covariant, so every one of these must return four
  // roots at k times the reference ones -- there is no numerical excuse anywhere in this sweep.
  //
  // Before the repair this collapsed at k = 1e-04, where |termQ| = 5.6e-10 falls under the
  // absolute 1e-09 test: the solver takes the biquadratic branch on a quartic that is not
  // biquadratic and returns *two* roots, 6.5093e-04 and -9.3257e-07, instead of four.
  for (const double k : {1.e6, 1.e3, 1., 1.e-1, 1.e-2, 1.e-3, 1.e-4, 1.e-5, 1.e-6, 1.e-8}) {
    const auto c = quarticFromRoots(1. * k, 2. * k, 3. * k, 7. * k);
    surf::QuarticBranch branch = surf::QuarticBranch::NotAQuartic;
    const auto roots = sortedRoots(c, &branch);
    BOOST_REQUIRE_EQUAL(roots.size(), 4u);
    const double expected[4] = {1. * k, 2. * k, 3. * k, 7. * k};
    for (int i = 0; i < 4; ++i) {
      BOOST_CHECK_LT(std::abs(roots[i] - expected[i]), 1.e-9 * std::abs(expected[i]));
    }
    checkRootsAreRoots(c, roots);
    // The other half of the positive control: an asymmetric quartic must NOT be routed into the
    // biquadratic branch at any scale. A termQ test that always passed would fail here.
    BOOST_CHECK(branch == surf::QuarticBranch::Resolvent);
  }

  // The same statement made about accuracy rather than about the branch, on the family that
  // produced "two confidently wrong roots": at k = 1e-04 the shipped code returns four roots for
  // {-2, -1, 1, 2.1} * k that are wrong by 1.3 % (relative backward error 1.6e-02), because the
  // biquadratic branch forces them to be symmetric about -b/4 and they are not.
  for (const double k : {1., 1.e-2, 1.e-4, 1.e-6}) {
    const auto c = quarticFromRoots(-2. * k, -1. * k, 1. * k, 2.1 * k);
    const auto roots = sortedRoots(c);
    BOOST_REQUIRE_EQUAL(roots.size(), 4u);
    const double expected[4] = {-2. * k, -1. * k, 1. * k, 2.1 * k};
    for (int i = 0; i < 4; ++i) {
      BOOST_CHECK_LT(std::abs(roots[i] - expected[i]), 1.e-9 * std::abs(expected[i]));
    }
    checkRootsAreRoots(c, roots);
  }
}

BOOST_AUTO_TEST_CASE(StreamM_QuarticStillSelectsTheBiquadraticBranch)
{
  // Positive control, first direction. The biquadratic branch is not a fallback: it is the
  // correct, better-conditioned answer whenever the depressed quartic really has no odd term, and
  // a repair that simply widened its guard into irrelevance would be caught by the previous case
  // while a repair that narrowed it away would be caught here.
  {
    // y^4 - 5 y^2 + 4, roots +-1, +-2; termQ is exactly zero
    const std::array<double, 5> c{1., 0., -5., 0., 4.};
    surf::QuarticBranch branch = surf::QuarticBranch::NotAQuartic;
    const auto roots = sortedRoots(c, &branch);
    BOOST_CHECK(branch == surf::QuarticBranch::Biquadratic);
    BOOST_REQUIRE_EQUAL(roots.size(), 4u);
    const double expected[4] = {-2., -1., 1., 2.};
    for (int i = 0; i < 4; ++i) {
      checkClose(roots[i], expected[i], 1.e-12);
    }
  }
  // The same quartic shifted along x and scaled, which is what a torus at a lever arm produces:
  // termQ is zero in exact arithmetic but is computed by cancelling terms of size |b|^3, so the
  // criterion has to be relative to those terms rather than to a fixed length.
  // The centres stop at 100. Beyond that the *depression* step -- p, q, r from b, c, d, e -- is
  // the limit, not the guards: it cancels numbers of size (centre)^k to leave numbers of size
  // (spread)^k, so a quartic whose roots agree to 4 significant figures has lost 8 digits before
  // any branch is chosen and Ferrari's discriminants become noise. Measured on this family, with
  // rounded coefficients, both before and after this change: relative root spread 2e-01 gives
  // 3.9e-14, 2e-02 gives 6.2e-11, 2e-03 gives 9.3e-08, and 2e-04 returns no roots at all. It is a
  // property of Ferrari's method and is not hidden behind a looser tolerance here.
  for (const double centre : {0., 1., 100.}) {
    for (const double k : {1., 1.e-3, 1.e3}) {
      const auto c = quarticFromRoots((centre - 2.) * k, (centre - 1.) * k, (centre + 1.) * k,
                                      (centre + 2.) * k);
      surf::QuarticBranch branch = surf::QuarticBranch::NotAQuartic;
      const auto roots = sortedRoots(c, &branch);
      BOOST_CHECK(branch == surf::QuarticBranch::Biquadratic);
      BOOST_REQUIRE_EQUAL(roots.size(), 4u);
      const double expected[4] = {(centre - 2.) * k, (centre - 1.) * k, (centre + 1.) * k,
                                  (centre + 2.) * k};
      for (int i = 0; i < 4; ++i) {
        BOOST_CHECK_LT(std::abs(roots[i] - expected[i]), 1.e-9 * std::max(1.e-30, std::abs(expected[i])));
      }
      checkRootsAreRoots(c, roots);
    }
  }
}

BOOST_AUTO_TEST_CASE(StreamM_QuarticStillDeclinesDegenerateConfigurations)
{
  // Positive control, second direction. A guard that always passes is not a fix. Each of these
  // must still produce no roots, in both branches, at every scale -- "declines" has to survive
  // the repair as surely as "accepts" does.
  {
    // not a genuine quartic at all
    surf::QuarticBranch branch = surf::QuarticBranch::Biquadratic;
    const auto roots = surf::solveQuarticReal(0., 1., 2., 3., 4., &branch);
    BOOST_CHECK_EQUAL(roots.size(), 0u);
    BOOST_CHECK(branch == surf::QuarticBranch::NotAQuartic);
  }
  for (const double k : {1.e4, 1., 1.e-4, 1.e-8}) {
    // (x^2 + k^2)(x^2 + 4 k^2): no real roots, termQ = 0 -> the biquadratic branch must decline
    const double k2 = k * k;
    const std::array<double, 5> biquadratic{1., 0., 5. * k2, 0., 4. * k2 * k2};
    surf::QuarticBranch branch = surf::QuarticBranch::NotAQuartic;
    BOOST_CHECK_EQUAL(sortedRoots(biquadratic, &branch).size(), 0u);
    BOOST_CHECK(branch == surf::QuarticBranch::Biquadratic);

    // (x^2 + k^2)((x + k)^2 + 4 k^2): no real roots, termQ != 0 -> the resolvent branch must
    // decline, by finding both of Ferrari's quadratics complex rather than by refusing to run
    const std::array<double, 5> asymmetric{1., 2. * k, 6. * k2, 2. * k2 * k, 5. * k2 * k2};
    BOOST_CHECK_EQUAL(sortedRoots(asymmetric, &branch).size(), 0u);
    BOOST_CHECK(branch == surf::QuarticBranch::Resolvent);
  }
}

BOOST_AUTO_TEST_CASE(StreamM_QuarticHasNoCliffAsAQuarticApproachesBiquadratic)
{
  // The defect is a *cliff*: an absolute threshold crossed by a quantity that carries units, so
  // the answer changes discontinuously with the size of the geometry. The repair has to be
  // continuous instead -- as termQ is driven to zero the two branches must agree, because they
  // are the two sides of one limit.
  //
  // {-2, -1, 1, 2 + delta} scaled by k: at delta = 0 the quartic is exactly biquadratic, and
  // delta walks it away from that continuously. Every point must give four correct roots.
  for (const double k : {1., 1.e-2, 1.e-4, 1.e-6}) {
    for (const double delta : {1.e-1, 1.e-3, 1.e-6, 1.e-9, 1.e-12, 0.}) {
      const auto c = quarticFromRoots(-2. * k, -1. * k, 1. * k, (2. + delta) * k);
      const auto roots = sortedRoots(c);
      BOOST_REQUIRE_EQUAL(roots.size(), 4u);
      const double expected[4] = {-2. * k, -1. * k, 1. * k, (2. + delta) * k};
      for (int i = 0; i < 4; ++i) {
        BOOST_CHECK_LT(std::abs(roots[i] - expected[i]), 1.e-8 * std::abs(expected[i]));
      }
      checkRootsAreRoots(c, roots);
    }
  }
}
// --- Tier 0: canonical recognition of NURBS-encoded quadrics
//
// The recognition work itself is entirely converter-side and its own controls live in
// `O2_CADtoTGeo.py --self-test` (18 checks: a NurbsConvert-ed quadric of each kind must be
// recovered, a genuine free-form patch must not, and every accepted face's MEASURED gap must be
// inside the acceptance tolerance).  Nothing of that can be asserted from C++.
//
// What *can* be asserted here, and matters more than it looks, is the kernel-side contract the
// converter measures against.  `_recognized_inner_wall()` decides a recognized quadric's
// `inner_wall` flag by comparing the face's own outward normal with "away from the axis", because
// on a NURBS-encoded quadric `TopoDS` orientation says nothing (on ALICE3: nine
// ALICE3 faces with an exactly antiparallel outward normal, 404 lost crossings, and every closure
// and edge-identity check blind to it because they are all sign-blind).  That measurement is only
// correct if the kernel's own convention is the one it assumes.  This work multiplies the number
// of faces going through that path, so the convention is pinned rather than assumed: if it were
// ever inverted, every recognized face would silently flip and no existing test would notice.
BOOST_AUTO_TEST_CASE(StreamK_InnerWallIsExactlyTheSignOfTheOutwardNormal)
{
  const Point3D centre{0.3, -0.7, 1.1};
  const Point3D axis{0., 0., 1.};
  const Point3D refU{1., 0., 0.};
  constexpr double radius = 2.5;
  constexpr double phi = 0.9;

  // A point on each surface, and the direction "away from the axis / centre" there.
  const double cx = centre[0] + radius * std::cos(phi);
  const double cy = centre[1] + radius * std::sin(phi);
  const Double_t onCylinder[3] = {cx, cy, centre[2] + 0.4};
  const Double_t awayFromAxis[3] = {std::cos(phi), std::sin(phi), 0.};

  for (const bool innerWall : {false, true}) {
    SurfaceSolid solid(innerWall ? "streamK_innerCyl" : "streamK_outerCyl");
    BOOST_REQUIRE(solid.AddCylindricalSurface(centre, axis, refU, radius, -1., 1., 0., surf::kTwoPi, innerWall));
    Double_t n[3] = {0., 0., 0.};
    solid.ComputeNormal(onCylinder, nullptr, n);
    const double alignment = n[0] * awayFromAxis[0] + n[1] * awayFromAxis[1] + n[2] * awayFromAxis[2];
    // Exactly +1 or exactly -1: this is a sign, not a tolerance.
    BOOST_CHECK_CLOSE(alignment, innerWall ? -1. : 1., 1.e-9);
  }

  // The same convention on the cone and on the sphere. All three go through the converter's one
  // `_recognized_inner_wall` measurement, and ALICE3 exercises only the cylinder branch today
  // (recognized planes and spheres are untested there), so the
  // other two are pinned here rather than left to the first model that uses them.
  for (const bool innerWall : {false, true}) {
    SurfaceSolid solid(innerWall ? "streamK_innerCone" : "streamK_outerCone");
    // r(h) = 1 + h over h in [0, 2]: half-angle 45 degrees, apex at h = -1.
    BOOST_REQUIRE(solid.AddConicalSurface(centre, axis, refU, 1., 3., 0., 2., 0., surf::kTwoPi, innerWall));
    const double h = 1.0;
    const double r = 2.0;
    const Double_t onCone[3] = {centre[0] + r * std::cos(phi), centre[1] + r * std::sin(phi), centre[2] + h};
    Double_t n[3] = {0., 0., 0.};
    solid.ComputeNormal(onCone, nullptr, n);
    // The cone's outward normal tilts out of the radial direction by the half angle, so the
    // radial component is what carries the sign -- which is exactly the reasoning
    // `_recognized_inner_wall` relies on, and the reason it can use the radial direction alone.
    const double radial = n[0] * std::cos(phi) + n[1] * std::sin(phi);
    BOOST_CHECK_GT(innerWall ? -radial : radial, 0.5);
  }

  for (const bool innerWall : {false, true}) {
    SurfaceSolid solid(innerWall ? "streamK_innerSph" : "streamK_outerSph");
    BOOST_REQUIRE(solid.AddSphericalSurface(centre, axis, refU, radius, 0., surf::kPi, 0., surf::kTwoPi, innerWall));
    const double theta = 1.1;
    const Double_t onSphere[3] = {centre[0] + radius * std::sin(theta) * std::cos(phi),
                                  centre[1] + radius * std::sin(theta) * std::sin(phi),
                                  centre[2] + radius * std::cos(theta)};
    const double outward[3] = {std::sin(theta) * std::cos(phi), std::sin(theta) * std::sin(phi), std::cos(theta)};
    Double_t n[3] = {0., 0., 0.};
    solid.ComputeNormal(onSphere, nullptr, n);
    const double alignment = n[0] * outward[0] + n[1] * outward[1] + n[2] * outward[2];
    BOOST_CHECK_CLOSE(alignment, innerWall ? -1. : 1., 1.e-9);
  }
}
// --- Placed primitives ---
//
// A recognised primitive whose frame is not the identity used to be emitted as a degenerate
// TGeoCompositeShape -- the primitive unioned with an identical copy of itself under the same
// matrix -- because no TGeoShape in ROOT 6.36 carries a rigid transform. That is still true of
// ROOT; what changed is where the transform lives. The shape is now written in its OWN canonical
// frame and the transform travels beside it, as a TGeoHMatrix under the key "placement" in
// shape_<part>.root. These cases pin the three things that can go wrong with that:
//
//   1. the artefact: the placement must survive the round trip, and its ABSENCE must keep meaning
//      the identity, so that every file written before this convention still loads and still
//      scores exactly as it did;
//   2. the equivalence: the bare primitive queried in its own frame must answer *exactly* like
//      the composite it replaces, with a negative control that moves the count;
//   3. the composition order in geom.C -- `partPlacement * shapePlacement`. That one is silent
//      when wrong: the geometry still builds and the shape is still the right shape, it is simply
//      somewhere else. It is checked by navigating, with a transposed rotation and a reversed
//      product as the controls.

namespace
{
/// The two matrices a placed tube is defined by in these cases: a rotation that is neither
/// symmetric nor axis-aligned, off the origin.
TGeoCombiTrans* makeStreamNPlacement()
{
  auto* rotation = new TGeoRotation("streamNRot", 0., 0., 0.);
  rotation->RotateX(30.);
  rotation->RotateZ(17.);
  rotation->RotateY(-41.);
  return new TGeoCombiTrans(0.3, 5.916, 2.0, rotation);
}
} // namespace

BOOST_AUTO_TEST_CASE(ShapeSidecarRoundTripsAPlacement)
{
  namespace harness = o2::cad::harness;
  const auto dir = std::filesystem::temp_directory_path();
  const std::string path = (dir / "o2_shape_placed.root").string();

  const TGeoTube tube("shape", 0.4, 1.0, 5.0);
  std::unique_ptr<TGeoCombiTrans> placement(makeStreamNPlacement());

  std::string error;
  BOOST_REQUIRE_MESSAGE(harness::saveShapeToRootFile(path, tube, placement.get(), &error), error);

  std::unique_ptr<TGeoShape> loaded(harness::loadShapeFromRootFile(path, &error));
  BOOST_REQUIRE_MESSAGE(loaded != nullptr, error);
  BOOST_CHECK_EQUAL(std::string(loaded->ClassName()), std::string("TGeoTube"));

  std::unique_ptr<TGeoHMatrix> back(harness::loadShapePlacementFromRootFile(path));
  BOOST_REQUIRE(back != nullptr);
  for (int i = 0; i < 9; ++i) {
    BOOST_CHECK_EQUAL(back->GetRotationMatrix()[i], placement->GetRotationMatrix()[i]);
  }
  for (int i = 0; i < 3; ++i) {
    BOOST_CHECK_EQUAL(back->GetTranslation()[i], placement->GetTranslation()[i]);
  }
  // The point of storing it: a point of the part frame reaches the same place through the file as
  // through the original matrix.
  const Point3D master{0.9, 6.2, 3.1};
  Point3D viaFile{};
  Point3D viaOriginal{};
  back->MasterToLocal(master.data(), viaFile.data());
  placement->MasterToLocal(master.data(), viaOriginal.data());
  for (int i = 0; i < 3; ++i) {
    BOOST_CHECK_EQUAL(viaFile[i], viaOriginal[i]);
  }
  std::filesystem::remove(path);
}

BOOST_AUTO_TEST_CASE(AbsentPlacementMeansIdentity)
{
  namespace harness = o2::cad::harness;
  const auto dir = std::filesystem::temp_directory_path();
  const TGeoTube tube("shape", 0.4, 1.0, 5.0);
  std::string error;

  // 1. The historical two-argument overload -- the one every existing shape_*.root was written
  //    with -- must record no placement at all.
  const std::string legacy = (dir / "o2_shape_legacy.root").string();
  BOOST_REQUIRE_MESSAGE(harness::saveShapeToRootFile(legacy, tube, &error), error);
  BOOST_CHECK(harness::loadShapePlacementFromRootFile(legacy) == nullptr);

  // 2. An identity placement is deliberately NOT written, so that "no key" stays the one and only
  //    spelling of the identity.
  const std::string identity = (dir / "o2_shape_identity.root").string();
  TGeoHMatrix unit("unit");
  BOOST_REQUIRE_MESSAGE(harness::saveShapeToRootFile(identity, tube, &unit, &error), error);
  BOOST_CHECK(harness::loadShapePlacementFromRootFile(identity) == nullptr);

  // 3. A file that is not there is the same answer, and must not throw or complain: a part with
  //    no shape sidecar at all is the overwhelmingly common case.
  BOOST_CHECK(harness::loadShapePlacementFromRootFile((dir / "o2_shape_nothing.root").string()) ==
              nullptr);

  std::filesystem::remove(legacy);
  std::filesystem::remove(identity);
}

BOOST_AUTO_TEST_CASE(PlacedPrimitiveAnswersExactlyLikeTheSelfUnionComposite)
{
  // The equivalence the change rests on: the bare primitive queried in its own frame answers like
  // the composite it replaces, on all four scored queries, exactly.
  constexpr double kRmin = 0.4;
  constexpr double kRmax = 1.0;
  constexpr double kDz = 5.0;

  std::unique_ptr<TGeoCombiTrans> placement(makeStreamNPlacement());
  const TGeoTube placedPrimitive("streamNTube", kRmin, kRmax, kDz);
  // The old emission, built here so the two are compared rather than one being trusted.
  std::unique_ptr<TGeoCompositeShape> composite(
    makePlacedTube("streamNComposite", kRmin, kRmax, kDz, new TGeoCombiTrans(*placement),
                   new TGeoCombiTrans(*placement)));

  std::size_t probes = 0;
  std::size_t inside = 0;
  std::size_t disagreements = 0;
  // The negative control travels with the check: the same loop against a 5% fatter tube must
  // disagree, or the loop is not measuring anything.
  const TGeoTube wrong("streamNWrong", kRmin, kRmax * 1.05, kDz);
  std::size_t controlDisagreements = 0;

  for (int ix = -8; ix <= 8; ++ix) {
    for (int iy = -8; iy <= 8; ++iy) {
      for (int iz = -8; iz <= 8; ++iz) {
        const Point3D master{0.3 + 0.37 * ix, 5.916 + 0.41 * iy, 2.0 + 0.43 * iz};
        Point3D local{};
        placement->MasterToLocal(master.data(), local.data());
        const double r = std::hypot(local[0], local[1]);
        if (std::fabs(r - kRmin) < 1.e-9 || std::fabs(r - kRmax) < 1.e-9 ||
            std::fabs(r - kRmax * 1.05) < 1.e-9 || std::fabs(std::fabs(local[2]) - kDz) < 1.e-9) {
          continue;
        }
        ++probes;
        const bool wanted = composite->Contains(master.data());
        if (placedPrimitive.Contains(local.data()) != wanted) {
          ++disagreements;
        }
        if (wrong.Contains(local.data()) != wanted) {
          ++controlDisagreements;
        }
        if (wanted) {
          ++inside;
        }
        BOOST_REQUIRE_CLOSE_FRACTION(placedPrimitive.Safety(local.data(), wanted),
                                     composite->Safety(master.data(), wanted), 1.e-12);
        for (const auto& dir : {Point3D{1., 0., 0.}, Point3D{0., 1., 0.}, Point3D{0., 0., 1.},
                                Point3D{0.5773502691896258, 0.5773502691896258,
                                        0.5773502691896258}}) {
          Point3D localDir{};
          placement->MasterToLocalVect(dir.data(), localDir.data());
          if (wanted) {
            BOOST_REQUIRE_CLOSE_FRACTION(placedPrimitive.DistFromInside(local.data(),
                                                                        localDir.data(), 3),
                                         composite->DistFromInside(master.data(), dir.data(), 3),
                                         1.e-12);
          } else {
            const double got = placedPrimitive.DistFromOutside(local.data(), localDir.data(), 3);
            const double want = composite->DistFromOutside(master.data(), dir.data(), 3);
            if (want > 1.e20) {
              BOOST_REQUIRE_GT(got, 1.e20);
            } else {
              BOOST_REQUIRE_CLOSE_FRACTION(got, want, 1.e-12);
            }
          }
        }
      }
    }
  }
  BOOST_CHECK_EQUAL(disagreements, 0u);
  BOOST_CHECK_GT(controlDisagreements, 0u);
  BOOST_CHECK_GT(inside, 100u);
  BOOST_CHECK_GT(probes, 3000u);
}

BOOST_AUTO_TEST_CASE(PlacedPrimitiveRecoversTheAnalyticCapacity)
{
  // What the degenerate composite cost, stated as a measurement rather than as a claim.
  constexpr double kRmin = 0.4;
  constexpr double kRmax = 1.0;
  constexpr double kDz = 5.0;
  const double analytic = TMath::Pi() * (kRmax * kRmax - kRmin * kRmin) * 2. * kDz;

  const TGeoTube tube("streamNCapTube", kRmin, kRmax, kDz);
  BOOST_CHECK_CLOSE_FRACTION(tube.Capacity(), analytic, 1.e-14);
  // Deterministic: asked twice, the same bits.
  BOOST_CHECK_EQUAL(tube.Capacity(), tube.Capacity());

  std::unique_ptr<TGeoCombiTrans> placement(makeStreamNPlacement());
  std::unique_ptr<TGeoCompositeShape> composite(
    makePlacedTube("streamNCapComposite", kRmin, kRmax, kDz, new TGeoCombiTrans(*placement),
                   new TGeoCombiTrans(*placement)));
  // ... whereas TGeoCompositeShape::Capacity() throws 10000 Monte-Carlo points into the bounding
  // box, so two calls on the same object return different numbers. That is the reason the gate
  // marks a composite `capacityComparable=false`, and the reason a placed primitive that is no
  // longer a composite gets its capacity column back.
  const double first = composite->Capacity();
  const double second = composite->Capacity();
  BOOST_CHECK_NE(first, second);
  BOOST_CHECK_GT(std::fabs(first - analytic) / analytic, 1.e-6);
}

BOOST_AUTO_TEST_CASE(NodeMatrixIsPartPlacementTimesShapePlacement)
{
  // The composition geom.C emits, decided by NAVIGATION rather than by reading the code.
  //
  // The reference is built without ever forming the product: a point of the assembly frame is
  // carried into the part frame by the part placement, then into the shape's frame by the shape
  // placement, and the tube membership is evaluated there. If `partPlacement * shapePlacement` is
  // the right node matrix, ROOT's navigator must reach the same verdict for every point.
  constexpr double kRmin = 0.4;
  constexpr double kRmax = 1.0;
  constexpr double kDz = 5.0;

  std::unique_ptr<TGeoCombiTrans> shapePlacementOwned(makeStreamNPlacement());
  const TGeoHMatrix shapePlacement(*shapePlacementOwned);
  auto* partRotation = new TGeoRotation("streamNPartRot", 37., 24., 61.);
  const TGeoCombiTrans partPlacement(-2.0, 7.0, 1.5, partRotation);

  const auto reference = [&](const Point3D& master, bool& onWall) {
    Point3D partFrame{};
    Point3D shapeFrame{};
    partPlacement.MasterToLocal(master.data(), partFrame.data());
    shapePlacement.MasterToLocal(partFrame.data(), shapeFrame.data());
    const double r = std::hypot(shapeFrame[0], shapeFrame[1]);
    onWall = std::fabs(r - kRmin) < 1.e-9 || std::fabs(r - kRmax) < 1.e-9 ||
             std::fabs(std::fabs(shapeFrame[2]) - kDz) < 1.e-9;
    return r >= kRmin && r <= kRmax && std::fabs(shapeFrame[2]) <= kDz;
  };

  // Every candidate node matrix, including the three ways of getting it wrong. `partOnly` is the
  // bug this test is really for: forgetting to compose at all.
  TGeoHMatrix correct(partPlacement);
  correct.Multiply(&shapePlacement);
  TGeoHMatrix reversed(shapePlacement);
  reversed.Multiply(&partPlacement);
  TGeoHMatrix transposedRotation(shapePlacement);
  {
    double rt[9];
    const double* r = shapePlacement.GetRotationMatrix();
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        rt[3 * i + j] = r[3 * j + i];
      }
    }
    transposedRotation.SetRotation(rt);
    transposedRotation.SetBit(TGeoMatrix::kGeoRotation);
  }
  TGeoHMatrix withTransposed(partPlacement);
  withTransposed.Multiply(&transposedRotation);
  const TGeoHMatrix partOnly(partPlacement);

  const std::vector<std::pair<std::string, const TGeoHMatrix*>> candidates{
    {"part*shape", &correct},
    {"shape*part", &reversed},
    {"part*shape^T", &withTransposed},
    {"part only", &partOnly}};

  // The lattice is centred where the solid actually is -- the translation of the CORRECT node
  // matrix -- and spans more than the tube's own extent. Guessing the centre by adding the two
  // translations put every probe outside the solid, and the controls then reported zero
  // disagreements while being structurally incapable of reporting anything else.
  const double* centre = correct.GetTranslation();

  std::vector<size_t> disagreements(candidates.size(), 0);
  size_t probes = 0;
  size_t insideProbes = 0;

  for (size_t c = 0; c < candidates.size(); ++c) {
    // One manager per candidate, and everything inside it allocated with new: a TGeoShape and a
    // TGeoVolume register themselves with gGeoManager, which frees them.
    auto* manager = new TGeoManager(("streamN_" + std::to_string(c)).c_str(), "composition order");
    auto* material = new TGeoMaterial("Vacuum", 0., 0., 0.);
    auto* medium = new TGeoMedium("Vacuum", 1, material);
    auto* world = new TGeoVolume("TOP", new TGeoBBox("streamNWorld", 30., 30., 30.), medium);
    auto* part = new TGeoVolume("PART", new TGeoTube("streamNNodeTube", kRmin, kRmax, kDz), medium);
    world->AddNode(part, 1, new TGeoHMatrix(*candidates[c].second));
    manager->SetTopVolume(world);
    manager->CloseGeometry();

    size_t localProbes = 0;
    size_t localInside = 0;
    for (int ix = -14; ix <= 14; ++ix) {
      for (int iy = -14; iy <= 14; ++iy) {
        for (int iz = -14; iz <= 14; ++iz) {
          const Point3D master{centre[0] + 0.45 * ix, centre[1] + 0.47 * iy,
                               centre[2] + 0.43 * iz};
          bool onWall = false;
          const bool wanted = reference(master, onWall);
          if (onWall) {
            continue;
          }
          ++localProbes;
          if (wanted) {
            ++localInside;
          }
          TGeoNode* node = manager->FindNode(master[0], master[1], master[2]);
          const bool got = node != nullptr && std::string(node->GetVolume()->GetName()) == "PART";
          if (got != wanted) {
            ++disagreements[c];
          }
        }
      }
    }
    probes = localProbes;
    insideProbes = localInside;
    delete manager;
    gGeoManager = nullptr;
  }

  // The sampling has to be capable of failing: enough points, and enough of them inside.
  BOOST_CHECK_GT(probes, 5000u);
  BOOST_CHECK_GT(insideProbes, 200u);
  BOOST_CHECK_EQUAL(disagreements[0], 0u); // partPlacement * shapePlacement
  BOOST_CHECK_GT(disagreements[1], 0u);    // reversed product
  BOOST_CHECK_GT(disagreements[2], 0u);    // transposed shape rotation
  BOOST_CHECK_GT(disagreements[3], 0u);    // shape placement dropped
}

// ============================================================================================
// The representation cost/memory benchmark's own instruments
// ============================================================================================
//
// These pin the MEASURING apparatus, not the geometry. A per-call cost table is only worth
// reading if the harness that produced it can be shown to move its number when the thing it
// measures moves. Each case below is that demonstration for one column of the benchmark.
//
// They include RepresentationBench.h -- the SAME header the benchmark binary times with.

#include "RepresentationBench.h"

using namespace o2::cad::bench;

/// The timing harness runs the requested passes over a sample set that exercises both branches.
BOOST_AUTO_TEST_CASE(RepBenchTimingHarnessRunsTheRequestedPasses)
{
  TGeoBBox fast("repBenchFast", 1., 1., 1.);
  const o2::cad::harness::Point3D lo{-1., -1., -1.};
  const o2::cad::harness::Point3D hi{1., 1., 1.};
  const QuerySamples samples = buildQuerySamples(&fast, "control", lo, hi, 1500, 1500);

  // The sample set has to be capable of exercising both branches, or three of the four kernels
  // are being timed on an empty vector.
  BOOST_CHECK_GT(samples.insidePoints, 100);
  BOOST_CHECK_LT(samples.insidePoints, static_cast<long long>(samples.points.size()) - 100);
  BOOST_CHECK_EQUAL(samples.outsideRays.size(), 1500u);
  BOOST_CHECK_EQUAL(samples.insideRays.size(), 1500u);

  // And the loop must not have been optimised away: a non-zero checksum, a positive time, and
  // the requested number of passes actually run.
  const TimingStat stat = timeContainsPass(&fast, samples, 1, 5);
  BOOST_CHECK_NE(stat.checksum, 0u);
  BOOST_CHECK_GT(stat.medianNsPerCall, 0.);
  BOOST_CHECK_EQUAL(stat.passes, 5);
  BOOST_CHECK_LE(stat.minNsPerCall, stat.medianNsPerCall);
  BOOST_CHECK_LE(stat.medianNsPerCall, stat.maxNsPerCall);
}

/// The sample set is the whole basis of "the same questions from the same sample sets": every
/// representation of a part is handed this one object. So its labels must agree with the
/// reference that produced them, and rays must actually reach the solid -- a DistFromOutside
/// column measured on rays that all miss prices the early-out, not the kernel.
BOOST_AUTO_TEST_CASE(RepBenchSampleSetIsReproducibleAndActuallyHits)
{
  TGeoTube tube("repBenchTube", 0.3, 1., 2.);
  const o2::cad::harness::Point3D lo{-1., -1., -2.};
  const o2::cad::harness::Point3D hi{1., 1., 2.};
  const QuerySamples a = buildQuerySamples(&tube, "surface", lo, hi, 2000, 2000);
  const QuerySamples b = buildQuerySamples(&tube, "surface", lo, hi, 2000, 2000);

  // Same seed, same bbox, same reference -> bit-identical. Without this the cost table's
  // "same sample set" claim is not checkable from outside.
  BOOST_REQUIRE_EQUAL(a.points.size(), b.points.size());
  for (size_t i = 0; i < a.points.size(); ++i) {
    BOOST_CHECK_EQUAL(a.points[i][0], b.points[i][0]);
    BOOST_CHECK_EQUAL(a.pointIsInside[i], b.pointIsInside[i]);
    BOOST_CHECK_EQUAL(a.pointIsInside[i] != 0, tube.Contains(a.points[i].data()));
  }
  BOOST_CHECK_GT(timeDistOutPass(&tube, a, 1, 3).hitFraction, 0.5);
  BOOST_CHECK_EQUAL(timeDistInPass(&tube, a, 1, 3).hitFraction, 1.);
}

/// Both memory columns have to move when memory moves, and the heap column has to come back when
/// it is released. The 64 MB block is deliberately over glibc's mmap threshold: `uordblks` alone
/// does not see such an allocation at all, which is exactly how this check earned its place.
/// Linux only: `readMemory()` reads /proc/self/statm and mallinfo2, both no-ops elsewhere.
#ifdef __linux__
BOOST_AUTO_TEST_CASE(RepBenchMemoryProbeSeesAnAllocationAndItsRelease)
{
  const MemorySnapshot before = readMemory();
  constexpr size_t kBytes = 64u << 20;
  auto block = std::make_unique<char[]>(kBytes);
  for (size_t i = 0; i < kBytes; i += 4096) {
    block[i] = static_cast<char>(i);
  }
  const MemorySnapshot delta = readMemory() - before;
  BOOST_CHECK_GT(delta.residentBytes, 32LL << 20);
  BOOST_CHECK_GT(delta.heapInUseBytes, 32LL << 20);
  block.reset();
  BOOST_CHECK_LT((readMemory() - before).heapInUseBytes, 8LL << 20);
}
#endif

/// The synthetic boolean ladder is a fixture whose whole purpose is a scaling exponent, so the
/// structure it claims has to be the structure it built -- and the two tree shapes have to be
/// genuinely different, or the "chain vs balanced" column compares a thing with itself.
BOOST_AUTO_TEST_CASE(RepBenchBooleanLadderHasTheStructureItClaims)
{
  auto* manager = new TGeoManager("repBenchLadder", "ladder");
  for (const int k : {2, 4, 8, 16, 32}) {
    const BooleanTreeStats chain =
      booleanTreeStats(buildBooleanLadder(k, LadderShape::Chain, "tC" + std::to_string(k)));
    const BooleanTreeStats balanced =
      booleanTreeStats(buildBooleanLadder(k, LadderShape::Balanced, "tB" + std::to_string(k)));
    BOOST_CHECK_EQUAL(chain.leaves, k);
    BOOST_CHECK_EQUAL(balanced.leaves, k);
    BOOST_CHECK_EQUAL(chain.nodes, k - 1);
    BOOST_CHECK_EQUAL(balanced.nodes, k - 1);
    BOOST_CHECK_EQUAL(chain.depth, k);
    BOOST_CHECK_EQUAL(balanced.depth, 1 + static_cast<int>(std::lround(std::log2(k))));
  }
  // A single leaf is not a composite at all: the ladder must hand back the primitive rather than
  // a one-sided union, or the K=1 baseline row would be priced with boolean machinery.
  TGeoShape* single = buildBooleanLadder(1, LadderShape::Balanced, "tOne");
  BOOST_CHECK(dynamic_cast<TGeoCompositeShape*>(single) == nullptr);
  BOOST_CHECK_EQUAL(booleanTreeStats(single).leaves, 1);
  delete manager;
  gGeoManager = nullptr;
}

// --- Representation cost/memory benchmark ---

// --- BVH-accelerated Safety and ComputeNormal ---
//
// Safety() and ComputeNormal() answer the same question -- which trimmed patch is nearest to this
// point -- and both used to answer it with a bare loop over every patch, which on ALICE3's
// 965-patch solid cost 812 us per call. They now walk the
// BVH that was already there. The oracle is the loop they replaced, kept as Safety_Loop() /
// ComputeNormal_Loop(), and the contract against it is *exact equality*, not agreement to a
// tolerance: both minimise the same distanceSqToPatch over the same patches under the same
// tie-break, so any difference at all is a traversal or pruning bug.

namespace
{
// The benchmark's deterministic LCG, seeded explicitly, so a failure is reproducible from the seed alone.
struct SampleStream {
  explicit SampleStream(std::uint64_t seed) { lcg.state = seed | 1u; }
  o2::cad::bench::detail::Lcg lcg;
  double uniform() { return lcg.next(); }
  double symmetric(double extent) { return (2. * uniform() - 1.) * extent; }
};

/// Points in every regime the two kernels have to survive, for one solid of half-extent \a extent.
///
/// The regimes are not decoration. Pruning is trivially correct where one patch is far nearer than
/// all others and only bites where several are comparably near, which is exactly *on* the surface;
/// and a point far outside the bounding box is the case where the node bound is large and a
/// rounding error in it would prune the whole tree. So the sample is deliberately loaded towards
/// the surface and towards infinity rather than being uniform in the box.
std::vector<std::array<double, 3>> nearestPatchSample(const SurfaceSolid& solid, double extent, int count)
{
  SampleStream stream(0x5EAFE7Full);
  std::vector<std::array<double, 3>> points;
  points.reserve(static_cast<size_t>(count));
  for (int index = 0; index < count; ++index) {
    std::array<double, 3> point{stream.symmetric(extent), stream.symmetric(extent), stream.symmetric(extent)};
    switch (index % 5) {
      case 0: // wherever it landed: inside or outside, generic
        break;
      case 1: { // walked onto the surface along its own normal, i.e. distance ~ 0
        std::array<double, 3> normal{0., 0., 0.};
        const double safety = solid.Safety_Loop(point.data(), solid.Contains(point.data()));
        solid.ComputeNormal_Loop(point.data(), nullptr, normal.data());
        const double sign = solid.Contains(point.data()) ? 1. : -1.;
        for (int dimension = 0; dimension < 3; ++dimension) {
          point[dimension] += sign * safety * normal[dimension];
        }
        break;
      }
      case 2: { // a hair off the surface, at the scale where several patches compete
        std::array<double, 3> normal{0., 0., 0.};
        const double safety = solid.Safety_Loop(point.data(), solid.Contains(point.data()));
        solid.ComputeNormal_Loop(point.data(), nullptr, normal.data());
        const double sign = solid.Contains(point.data()) ? 1. : -1.;
        const double offset = safety - sign * 1.e-9 * std::max(1., extent);
        for (int dimension = 0; dimension < 3; ++dimension) {
          point[dimension] += sign * offset * normal[dimension];
        }
        break;
      }
      case 3: // well outside the bounding box
        for (auto& coordinate : point) {
          coordinate *= 40.;
        }
        break;
      case 4: // very far away, where the node bound is large and its rounding is worst
        for (auto& coordinate : point) {
          coordinate *= 1.e7;
        }
        break;
    }
    points.push_back(point);
  }
  // and the exact centre, where every face of a box is equidistant: the tie the tie-break decides
  points.push_back({0., 0., 0.});
  return points;
}

/// Compare the accelerated nearest-patch kernels against their loop twins at one point. Returns
/// the number of disagreements, so a caller can both assert zero and count.
int countNearestPatchDisagreements(const SurfaceSolid& solid, const std::array<double, 3>& point,
                                   double* worstSafetyGap = nullptr)
{
  int disagreements = 0;
  for (const bool inside : {true, false}) {
    const double accelerated = solid.Safety(point.data(), inside);
    const double reference = solid.Safety_Loop(point.data(), inside);
    if (accelerated != reference) {
      ++disagreements;
    }
    if (worstSafetyGap != nullptr) {
      *worstSafetyGap = std::max(*worstSafetyGap, accelerated - reference);
    }
  }
  // with and without a direction, since the direction flips the sign of the chosen patch's normal
  // and a wrong patch can hide behind that flip
  const std::array<double, 3> direction = unitDirection(0.37, -0.82, 0.44);
  for (const double* dir : {static_cast<const double*>(nullptr), direction.data()}) {
    std::array<double, 3> accelerated{0., 0., 0.};
    std::array<double, 3> reference{0., 0., 0.};
    solid.ComputeNormal(point.data(), dir, accelerated.data());
    solid.ComputeNormal_Loop(point.data(), dir, reference.data());
    for (int dimension = 0; dimension < 3; ++dimension) {
      if (accelerated[dimension] != reference[dimension]) {
        ++disagreements;
      }
    }
  }
  return disagreements;
}

// A solid with many patches whose boxes overlap heavily: eight boxes on a line is the easy case
// for pruning, a shell of small boxes around a sphere is not.
std::unique_ptr<SurfaceSolid> makeManyPatchSolid(const char* name, int ringCount)
{
  auto solid = std::make_unique<SurfaceSolid>(name);
  for (int ring = 0; ring < ringCount; ++ring) {
    const double angle = surf::kTwoPi * ring / ringCount;
    addBoxSurfaces(*solid, 0.4, 0.4, 0.4, {3. * std::cos(angle), 3. * std::sin(angle), 0.});
  }
  solid->CloseShape();
  return solid;
}

// A quarter of a circle of radius r about (cu, cv) in a parametric domain, as a rational quadratic
// B-spline -- the public-API twin of the kernel-level quarterCircleBSpline above.
BoundaryCurve quarterCircleBoundary(double cu, double cv, double r, double a0)
{
  const double a1 = a0 + surf::kHalfPi;
  const double aMid = 0.5 * (a0 + a1);
  std::vector<Point2D> poles{{cu + r * std::cos(a0), cv + r * std::sin(a0)},
                             {cu + r * std::sqrt(2.) * std::cos(aMid), cv + r * std::sqrt(2.) * std::sin(aMid)},
                             {cu + r * std::cos(a1), cv + r * std::sin(a1)}};
  return BoundaryCurve::makeBSpline(2, std::move(poles), {1., std::sqrt(0.5), 1.}, {0., 0., 0., 1., 1., 1.});
}

// Four cylindrical windows cut by B-spline wires in (phi, h), plus two disks. Not a closed solid --
// it is not meant to be navigated -- but it is the *trim* family measured at 2-6 us per
// candidate patch, whose distanceSqToPatch walks a flattened polyline. That is where pruning has
// the most to save and where a wrong bound would cost the most, so the cross-check has to cover it.
std::unique_ptr<SurfaceSolid> makeWireTrimmedSolid(const char* name)
{
  auto solid = std::make_unique<SurfaceSolid>(name);
  for (int window = 0; window < 4; ++window) {
    const double centrePhi = surf::kHalfPi * window + 0.3;
    BOOST_REQUIRE(solid->AddCylindricalSurface(
      {0., 0., 0.}, {0., 0., 1.}, {1., 0., 0.}, 2., -1., 1., 0., surf::kTwoPi, false,
      {quarterCircleBoundary(centrePhi, 0., 0.5, 0.), quarterCircleBoundary(centrePhi, 0., 0.5, surf::kHalfPi),
       quarterCircleBoundary(centrePhi, 0., 0.5, surf::kPi),
       quarterCircleBoundary(centrePhi, 0., 0.5, 3. * surf::kHalfPi)}));
  }
  BOOST_REQUIRE(addDiskSurface(*solid, {0., 0., 1.}, {1., 0., 0.}, {0., 1., 0.}, 2.));
  BOOST_REQUIRE(addDiskSurface(*solid, {0., 0., -1.}, {1., 0., 0.}, {0., -1., 0.}, 2.));
  solid->CloseShape();
  return solid;
}

struct NearestPatchFixture {
  std::unique_ptr<SurfaceSolid> solid;
  double extent;
};

std::vector<NearestPatchFixture> nearestPatchFixtures()
{
  std::vector<NearestPatchFixture> fixtures;
  fixtures.push_back({makeBoxSolid("safetyBox", 1., 2., 3.), 4.});
  fixtures.push_back({makeTubeSolid("safetyTube", 0., 2., 3.), 4.});
  fixtures.push_back({makeTubeSolid("safetyHollowTube", 1., 2., 3.), 4.});
  fixtures.push_back({makeConeSolid("safetyCone", 2., 1., 3.), 4.});
  fixtures.push_back({makeSphereSolid("safetySphere", 2.5), 3.5});
  fixtures.push_back({makeTorusSolid("safetyTorus", 3., 1.), 4.5});
  fixtures.push_back({makeCapsuleSolid("safetyCapsule", 1., 1.5), 3.});
  fixtures.push_back({makeManyPatchSolid("safetyRing", 12), 4.5});
  fixtures.push_back({makeWireTrimmedSolid("safetyWireTrim"), 3.});
  return fixtures;
}
} // namespace

/// The invariant the whole acceleration rests on, pinned at the level it is a property of.
///
/// A node is pruned when the distance from the query point to its bounding box already exceeds the
/// best patch distance found so far. That is only sound if the box distance is a **lower** bound on
/// the distance to every patch inside it -- and the box is built from each surface's own
/// conservativeBounds(), so per surface the statement is
///
///     distance(point, conservativeBounds) <= sqrt(distanceSqToPatch(point))       for every point.
///
/// It holds because every distanceSqToPatch in BoundedSurface.h is realised on the patch's
/// *untrimmed* window -- the wire itself for the planar families, the full rim band for a cylinder
/// or cone, the full sphere, the full torus -- and each family's conservativeBounds() encloses
/// exactly that window. Reading the code says so; this measures it, on every surface family, from
/// points in every regime. If a future surface family returned a distance realised outside its own
/// bounds, Safety() would silently start answering too much and only this case would say so.
BOOST_AUTO_TEST_CASE(StreamS_PatchDistanceIsNeverBelowTheDistanceToItsOwnBoundingBox)
{
  using surf::Vec2;
  using surf::Vec3;
  std::string error;

  surf::PlanarBoundedSurface polygon;
  const std::vector<Vec2> rectangle{{0., 0.}, {2., 0.}, {2., 3.}, {0., 3.}};
  BOOST_REQUIRE(polygon.initialize({0., 0., 0.}, {1., 0., 0.}, {0., 1., 0.}, rectangle, {}, error));
  surf::CurvedPlanarBoundedSurface disk;
  BOOST_REQUIRE(disk.initialize({0., 0., 0.5}, {1., 0., 0.}, {0., 1., 0.},
                                {surf::Curve2D::makeCircle({0., 0.}, 1.5)}, {}, error));
  // a B-spline trim wire on a second planar face: the trim family found to dominate the
  // per-patch cost, and the one whose distanceSqToPatch walks a flattened polyline rather than a
  // closed form -- so the lower-bound claim has to hold for an approximated boundary too
  surf::CurvedPlanarBoundedSurface splineFace;
  BOOST_REQUIRE(splineFace.initialize({0.2, -0.3, 1.1}, {1., 0., 0.}, {0., 1., 0.},
                                      {quarterCircleBSpline(0., 0., 1.2, 0.),
                                       quarterCircleBSpline(0., 0., 1.2, surf::kHalfPi),
                                       quarterCircleBSpline(0., 0., 1.2, surf::kPi),
                                       quarterCircleBSpline(0., 0., 1.2, 3. * surf::kHalfPi)},
                                      {}, error));
  surf::CylindricalBoundedSurface cylinder;
  BOOST_REQUIRE(cylinder.initialize({0.1, -0.2, 0.}, {0., 0., 1.}, {1., 0., 0.}, 2., -1., 1., 0.3,
                                    1.7 * surf::kPi, false, error));
  surf::ConicalBoundedSurface cone;
  BOOST_REQUIRE(cone.initialize({0., 0., 0.}, {0., 1., 0.}, {1., 0., 0.}, 2., 0.5, -1., 1., 0., 1.1 * surf::kPi,
                                false, error));
  surf::SphericalBoundedSurface sphere;
  BOOST_REQUIRE(sphere.initialize({0.3, 0.4, -0.5}, {0., 0., 1.}, {1., 0., 0.}, 1.7, 0.2, 2.4, 0.,
                                  1.3 * surf::kPi, false, error));
  surf::TorusBoundedSurface torus;
  BOOST_REQUIRE(torus.initialize({0., 0., 0.}, {0., 0., 1.}, {1., 0., 0.}, 3., 0.8, 0., 1.4 * surf::kPi, 0.,
                                 1.9 * surf::kPi, false, error));

  const std::vector<const surf::BoundedSurface*> surfaces{&polygon, &disk, &splineFace,
                                                          &cylinder, &cone, &sphere,
                                                          &torus};

  SampleStream stream(0xB0B0Dull);
  size_t checked = 0;
  for (const auto* surface : surfaces) {
    Vec3 lower{TGeoShape::Big(), TGeoShape::Big(), TGeoShape::Big()};
    Vec3 upper{-TGeoShape::Big(), -TGeoShape::Big(), -TGeoShape::Big()};
    surface->conservativeBounds(lower, upper);
    const double boxLower[3] = {lower.xCoord, lower.yCoord, lower.zCoord};
    const double boxUpper[3] = {upper.xCoord, upper.yCoord, upper.zCoord};
    // the box the BVH actually stores is this one inflated outward, which only lowers the bound
    for (int sample = 0; sample < 4000; ++sample) {
      const double scale = (sample % 4 == 3) ? 1.e6 : ((sample % 4 == 2) ? 20. : 5.);
      const Vec3 point{stream.symmetric(scale), stream.symmetric(scale), stream.symmetric(scale)};
      const double coordinates[3] = {point.xCoord, point.yCoord, point.zCoord};
      double boxDistanceSq = 0.;
      for (int dimension = 0; dimension < 3; ++dimension) {
        if (coordinates[dimension] < boxLower[dimension]) {
          const double gap = boxLower[dimension] - coordinates[dimension];
          boxDistanceSq += gap * gap;
        } else if (coordinates[dimension] > boxUpper[dimension]) {
          const double gap = coordinates[dimension] - boxUpper[dimension];
          boxDistanceSq += gap * gap;
        }
      }
      const double patchDistanceSq = surface->distanceSqToPatch(point);
      // the direction of this inequality is the whole safety argument; a tolerance would hide the
      // failure it exists to catch, so it is asserted with the same relative guard the traversal
      // itself applies (1e-12) and nothing more
      BOOST_REQUIRE_LE(boxDistanceSq * (1. - 1.e-12), patchDistanceSq);
      ++checked;
    }
  }
  BOOST_CHECK_EQUAL(checked, 7u * 4000u);
}

/// Accelerated == brute force, exactly, for both kernels, over every fixture and every regime.
BOOST_AUTO_TEST_CASE(StreamS_SafetyAndNormalAreIdenticalToTheAllSurfacesLoop)
{
  size_t comparedPoints = 0;
  double worstSafetyGap = -std::numeric_limits<double>::infinity();
  for (const auto& fixture : nearestPatchFixtures()) {
    BOOST_TEST_CONTEXT("fixture = " << fixture.solid->GetName())
    {
      BOOST_REQUIRE(fixture.solid->HasBVH());
      for (const auto& point : nearestPatchSample(*fixture.solid, fixture.extent, 2000)) {
        BOOST_TEST_CONTEXT("point = (" << point[0] << ", " << point[1] << ", " << point[2] << ")")
        {
          BOOST_REQUIRE_EQUAL(countNearestPatchDisagreements(*fixture.solid, point, &worstSafetyGap), 0);
        }
        ++comparedPoints;
      }
    }
  }
  BOOST_CHECK_EQUAL(comparedPoints, 9u * 2001u);
  // exact equality, so the gap is not merely non-positive but identically zero
  BOOST_CHECK_EQUAL(worstSafetyGap, 0.);
}

/// The traversal must also be right before there is anything to traverse, and on a solid whose
/// surface set is empty -- the two states where the accelerated path has to fall back rather than
/// crash or answer something else.
BOOST_AUTO_TEST_CASE(StreamS_SafetyFallsBackBeforeCloseShapeAndOnAnEmptySolid)
{
  SurfaceSolid open("safetyBeforeClose");
  addBoxSurfaces(open, 1., 2., 3.);
  BOOST_REQUIRE(!open.HasBVH()); // CloseShape not called: no acceleration structure yet
  const std::array<double, 3> probe{0.3, -1.1, 2.2};
  BOOST_CHECK_EQUAL(open.Safety(probe.data(), kTRUE), open.Safety_Loop(probe.data(), kTRUE));
  std::array<double, 3> viaBVH{0., 0., 0.};
  std::array<double, 3> viaLoop{0., 0., 0.};
  open.ComputeNormal(probe.data(), nullptr, viaBVH.data());
  open.ComputeNormal_Loop(probe.data(), nullptr, viaLoop.data());
  BOOST_CHECK_EQUAL(viaBVH[0], viaLoop[0]);
  BOOST_CHECK_EQUAL(viaBVH[1], viaLoop[1]);
  BOOST_CHECK_EQUAL(viaBVH[2], viaLoop[2]);

  SurfaceSolid empty("safetyEmpty");
  empty.CloseShape();
  BOOST_CHECK_EQUAL(empty.Safety(probe.data(), kTRUE), TGeoShape::Big());
  BOOST_CHECK_EQUAL(empty.Safety_Loop(probe.data(), kTRUE), TGeoShape::Big());
  empty.ComputeNormal(probe.data(), nullptr, viaBVH.data());
  empty.ComputeNormal_Loop(probe.data(), nullptr, viaLoop.data());
  BOOST_CHECK_EQUAL(viaBVH[0], viaLoop[0]);
}

/// ComputeNormal's tie-break, isolated. At the exact centre of a box all six faces are the same
/// distance away, so which one wins is decided entirely by the loop's strict `<` -- the first, i.e.
/// the lowest-indexed, patch. A traversal that visits patches in BVH order would legitimately pick
/// a different face and return a different normal, so the accelerated path carries the index
/// tie-break explicitly and declines to prune a node whose bound merely *equals* the current best.
///
/// This is a real configuration, not a contrived one: the centre of a box, the axis of a tube and
/// the centre of a sphere all produce exact ties, and a navigator that asks for a normal there gets
/// an answer that must not depend on how the tree happened to be built.
BOOST_AUTO_TEST_CASE(StreamS_ComputeNormalKeepsTheLowestIndexTieBreak)
{
  const auto box = makeBoxSolid("tieBreakBox", 2., 2., 2.);
  const std::array<double, 3> centre{0., 0., 0.};
  std::array<double, 3> viaBVH{0., 0., 0.};
  std::array<double, 3> viaLoop{0., 0., 0.};
  box->ComputeNormal(centre.data(), nullptr, viaBVH.data());
  box->ComputeNormal_Loop(centre.data(), nullptr, viaLoop.data());
  // all six faces are exactly 2 away, and the loop's strict `<` keeps the first of them
  BOOST_CHECK_EQUAL(box->Safety(centre.data(), kTRUE), box->Safety_Loop(centre.data(), kTRUE));
  BOOST_CHECK_EQUAL(viaBVH[0], viaLoop[0]);
  BOOST_CHECK_EQUAL(viaBVH[1], viaLoop[1]);
  BOOST_CHECK_EQUAL(viaBVH[2], viaLoop[2]);
  // ... and it is genuinely a tie, i.e. the case has something to protect
  BOOST_CHECK_EQUAL(std::abs(viaLoop[0]) + std::abs(viaLoop[1]) + std::abs(viaLoop[2]), 1.);

  // the same on the axis of a hollow tube, where the inner wall and both caps compete
  const auto tube = makeTubeSolid("tieBreakTube", 1., 2., 1.);
  tube->ComputeNormal(centre.data(), nullptr, viaBVH.data());
  tube->ComputeNormal_Loop(centre.data(), nullptr, viaLoop.data());
  BOOST_CHECK_EQUAL(viaBVH[0], viaLoop[0]);
  BOOST_CHECK_EQUAL(viaBVH[1], viaLoop[1]);
  BOOST_CHECK_EQUAL(viaBVH[2], viaLoop[2]);
}

/// The negative control. A cross-check that cannot fail has not passed, so the pruning bound is
/// deliberately replaced by one that is not a lower bound -- the distance to the node's bounding
/// box *centre*, which is larger than the distance to the box for any box with extent -- and the
/// comparison against the loop must then break.
///
/// It must break in a stated direction: an over-large bound prunes subtrees that hold the true
/// nearest patch, so the accelerated Safety comes out **too large**. That is the failure mode that
/// matters (a valid safety may never exceed the true distance to the boundary), and it is the one
/// the sabotage reproduces, so the healthy case is being watched by a test that has been shown to
/// see exactly the thing it is there to see.
BOOST_AUTO_TEST_CASE(StreamS_BreakingThePruningBoundIsCaught)
{
  BOOST_REQUIRE(!SurfaceSolid::GetSafetyBoundUnsoundForTest()); // sound by default

  const auto fixtures = nearestPatchFixtures();
  size_t caughtOnFixtures = 0;
  size_t prunableFixtures = 0;
  size_t sabotagedDisagreements = 0;
  size_t safetyTooLarge = 0;
  size_t safetyTooSmall = 0;
  for (const auto& fixture : fixtures) {
    // With the sub-patch BVH every fixture here has something to prune: a multi-surface solid has
    // one leaf per cover box across its surfaces, and even a single full sphere or torus owns a
    // whole grid of cover-box leaves. (Before sub-patching, single-patch solids were a single
    // unprunable leaf and had to be excluded here; that blind spot is gone by construction.)
    ++prunableFixtures;
    const auto points = nearestPatchSample(*fixture.solid, fixture.extent, 400);
    size_t disagreementsHere = 0;
    SurfaceSolid::SetSafetyBoundUnsoundForTest(true);
    for (const auto& point : points) {
      const double sabotaged = fixture.solid->Safety(point.data(), kTRUE);
      SurfaceSolid::SetSafetyBoundUnsoundForTest(false);
      const double reference = fixture.solid->Safety_Loop(point.data(), kTRUE);
      SurfaceSolid::SetSafetyBoundUnsoundForTest(true);
      if (sabotaged != reference) {
        ++disagreementsHere;
        (sabotaged > reference) ? ++safetyTooLarge : ++safetyTooSmall;
      }
      disagreementsHere += static_cast<size_t>(countNearestPatchDisagreements(*fixture.solid, point));
    }
    SurfaceSolid::SetSafetyBoundUnsoundForTest(false);
    sabotagedDisagreements += disagreementsHere;
    if (disagreementsHere > 0) {
      ++caughtOnFixtures;
      BOOST_TEST_MESSAGE("sabotaged bound caught on " << fixture.solid->GetName() << ": " << disagreementsHere
                                                      << " disagreements over " << points.size() << " points");
    } else {
      BOOST_TEST_MESSAGE("sabotaged bound NOT caught on " << fixture.solid->GetName() << " ("
                                                          << fixture.solid->GetNsurfaces() << " patches)");
    }
  }

  // every fixture is sensitive to the sabotage, not just one lucky one
  BOOST_CHECK_EQUAL(caughtOnFixtures, prunableFixtures);
  BOOST_CHECK_EQUAL(prunableFixtures, fixtures.size());
  BOOST_CHECK_GT(sabotagedDisagreements, 100u);
  // and it fails the dangerous way: too much safety, never too little
  BOOST_CHECK_GT(safetyTooLarge, 0u);
  BOOST_CHECK_EQUAL(safetyTooSmall, 0u);

  // with the sabotage off again the same sample is clean, so the disagreements above are the
  // sabotage and not the fixtures
  for (const auto& fixture : fixtures) {
    for (const auto& point : nearestPatchSample(*fixture.solid, fixture.extent, 400)) {
      BOOST_REQUIRE_EQUAL(countNearestPatchDisagreements(*fixture.solid, point), 0);
    }
  }
  BOOST_CHECK(!SurfaceSolid::GetSafetyBoundUnsoundForTest());
}

/// What the acceleration actually buys, in the currency the defect was measured in: patches handed
/// to distanceSqToPatch per call. The loop's number is GetNsurfaces() by construction; the
/// traversal's is what this counts. The bounds below are deliberately far looser than the measured
/// values so the case pins the *existence* of pruning rather than becoming a performance trap.
BOOST_AUTO_TEST_CASE(StreamS_SafetyVisitsFarFewerPatchesThanTheLoop)
{
  const auto ring = makeManyPatchSolid("candidateRing", 24); // 24 boxes, 144 patches
  BOOST_REQUIRE_EQUAL(ring->GetNsurfaces(), 144);
  const auto points = nearestPatchSample(*ring, 5., 500);

  SurfaceSolid::ResetSafetyCandidateCounter();
  for (const auto& point : points) {
    ring->Safety(point.data(), kTRUE);
  }
  const long long acceleratedCandidates = SurfaceSolid::GetSafetyCandidateCount();
  const double perCall = static_cast<double>(acceleratedCandidates) / points.size();

  BOOST_TEST_MESSAGE("Safety candidates per call: " << perCall << " of " << ring->GetNsurfaces() << " patches");
  BOOST_CHECK_GT(acceleratedCandidates, 0);
  BOOST_CHECK_LT(perCall, 0.4 * ring->GetNsurfaces());

  // the counter is not touched by the loop twin, which visits everything by construction
  SurfaceSolid::ResetSafetyCandidateCounter();
  for (const auto& point : points) {
    ring->Safety_Loop(point.data(), kTRUE);
    std::array<double, 3> normal{0., 0., 0.};
    ring->ComputeNormal_Loop(point.data(), nullptr, normal.data());
  }
  BOOST_CHECK_EQUAL(SurfaceSolid::GetSafetyCandidateCount(), 0);

  // ComputeNormal prunes too, only slightly less: it may not drop a node whose bound ties the
  // current best, because such a node can hold an equally near patch of lower index
  SurfaceSolid::ResetSafetyCandidateCounter();
  for (const auto& point : points) {
    std::array<double, 3> normal{0., 0., 0.};
    ring->ComputeNormal(point.data(), nullptr, normal.data());
  }
  const double normalPerCall = static_cast<double>(SurfaceSolid::GetSafetyCandidateCount()) / points.size();
  BOOST_TEST_MESSAGE("ComputeNormal candidates per call: " << normalPerCall);
  BOOST_CHECK_LT(normalPerCall, 0.4 * ring->GetNsurfaces());
  BOOST_CHECK_GE(normalPerCall, perCall);
}

/// @name The sub-patch BVH
///
/// One conservative box per surface makes every swept quadric a giant leaf: a full cylinder's box
/// is the box of its two full rim circles, a sphere's is the whole ball, and every ray through
/// that box pays an analytic patch intersection that mostly reports nothing. The sub-patch BVH
/// lets each surface contribute several tighter boxes (appendCoverBoxes) and dedups the surfaces
/// a query actually tests, so the leaf boxes hug the geometry and the answers stay bit-identical
/// to the loop twins.
/// @{

namespace
{
using CoverBox = surf::BoundedSurface::CoverBox;

// Squared distance from a point to an axis-aligned box, zero inside; double throughout, so the
// test's bound carries no float rounding of its own.
double coverBoxDistanceSq(const CoverBox& box, const surf::Vec3& point)
{
  double distanceSq = 0.;
  const double coordinates[3] = {point.xCoord, point.yCoord, point.zCoord};
  const double lower[3] = {box.first.xCoord, box.first.yCoord, box.first.zCoord};
  const double upper[3] = {box.second.xCoord, box.second.yCoord, box.second.zCoord};
  for (int dimension = 0; dimension < 3; ++dimension) {
    const double gap = std::max({lower[dimension] - coordinates[dimension],
                                 coordinates[dimension] - upper[dimension], 0.});
    distanceSq += gap * gap;
  }
  return distanceSq;
}

double minCoverBoxDistanceSq(const std::vector<CoverBox>& boxes, const surf::Vec3& point)
{
  double best = std::numeric_limits<double>::infinity();
  for (const auto& box : boxes) {
    best = std::min(best, coverBoxDistanceSq(box, point));
  }
  return best;
}

bool anyCoverBoxContains(const std::vector<CoverBox>& boxes, const surf::Vec3& point, double slack)
{
  for (const auto& box : boxes) {
    if (point.xCoord >= box.first.xCoord - slack && point.xCoord <= box.second.xCoord + slack &&
        point.yCoord >= box.first.yCoord - slack && point.yCoord <= box.second.yCoord + slack &&
        point.zCoord >= box.first.zCoord - slack && point.zCoord <= box.second.zCoord + slack) {
      return true;
    }
  }
  return false;
}

// The two properties every surface's cover boxes owe the traversal, checked against the surface's
// own kernels: (lower bound) the nearest cover box is never farther than distanceSqToPatch, which
// is what makes pruning on a box distance sound for Safety; (coverage) every point of the trimmed
// patch lies in some box, which is what makes a ray traversal that skips the other boxes complete.
void checkCoverBoxProperties(const surf::BoundedSurface& surface, const std::vector<surf::Vec3>& patchPoints,
                             const char* label)
{
  std::vector<CoverBox> boxes;
  surface.appendCoverBoxes(boxes);
  BOOST_TEST_CONTEXT("surface = " << label)
  {
    BOOST_REQUIRE(!boxes.empty());
    for (const auto& point : patchPoints) {
      BOOST_TEST_CONTEXT("patch point = (" << point.xCoord << ", " << point.yCoord << ", " << point.zCoord << ")")
      {
        BOOST_CHECK(anyCoverBoxContains(boxes, point, 1.e-9));
      }
    }
    SampleStream stream(0xC0FEB0C5ull);
    // near the patch, a few radii out, and far away, so the bound is exercised where the box and
    // the patch nearly coincide and where the whole surface is a speck
    constexpr double kProbeScales[3] = {1.5, 8., 300.};
    for (int index = 0; index < 400; ++index) {
      const double scale = kProbeScales[index % 3];
      const surf::Vec3 point{stream.symmetric(scale), stream.symmetric(scale), stream.symmetric(scale)};
      const double patchDistanceSq = surface.distanceSqToPatch(point);
      const double boxDistanceSq = minCoverBoxDistanceSq(boxes, point);
      BOOST_TEST_CONTEXT("point = (" << point.xCoord << ", " << point.yCoord << ", " << point.zCoord << ")")
      {
        BOOST_CHECK_LE(boxDistanceSq, patchDistanceSq * (1. + 1.e-9) + 1.e-18);
      }
    }
  }
}

// A curved family must actually sub-patch: one conservative box would satisfy both properties
// above and tighten nothing, which is the state this whole stream exists to leave behind.
void checkEmitsSeveralCoverBoxes(const surf::BoundedSurface& surface, const char* label)
{
  std::vector<CoverBox> boxes;
  surface.appendCoverBoxes(boxes);
  BOOST_TEST_CONTEXT("surface = " << label)
  {
    BOOST_CHECK_GT(boxes.size(), 1u);
  }
}
} // namespace

/// The cover boxes of every family, against that family's own kernels. The curved families must
/// emit more than one box -- a single conservative box passes the two properties trivially and
/// tightens nothing -- and the properties must hold on awkward frames, partial sweeps and wire
/// trims, not just on the axis-aligned full-sweep cases.
BOOST_AUTO_TEST_CASE(StreamX_CoverBoxesAreATightLowerBoundEnvelopePerFamily)
{
  using surf::Vec3;
  std::string error;

  const Vec3 skewCenter{0.4, -0.2, 0.1};
  const Vec3 skewAxis{0.2, 0.3, 1.};
  const Vec3 referenceU{1., 0., 0.};

  {
    surf::CylindricalBoundedSurface cylinder;
    BOOST_REQUIRE(cylinder.initialize(skewCenter, skewAxis, referenceU, 1.7, -0.8, 1.2, 0.4, 1.9, false, error));
    std::vector<Vec3> patchPoints;
    for (int stepPhi = 0; stepPhi <= 12; ++stepPhi) {
      for (int stepH = 0; stepH <= 4; ++stepH) {
        patchPoints.push_back(cylinder.pointAt(0.4 + 1.9 * stepPhi / 12., -0.8 + 2. * stepH / 4.));
      }
    }
    checkCoverBoxProperties(cylinder, patchPoints, "partial cylinder, skew axis");
    checkEmitsSeveralCoverBoxes(cylinder, "partial cylinder, skew axis");
  }
  {
    surf::CylindricalBoundedSurface cylinder;
    BOOST_REQUIRE(cylinder.initialize({0., 0., 0.}, {0., 0., 1.}, referenceU, 2., -1., 1., 0., surf::kTwoPi,
                                      false, error));
    std::vector<Vec3> patchPoints;
    for (int stepPhi = 0; stepPhi <= 24; ++stepPhi) {
      patchPoints.push_back(cylinder.pointAt(surf::kTwoPi * stepPhi / 24., -1. + 2. * (stepPhi % 5) / 4.));
    }
    checkCoverBoxProperties(cylinder, patchPoints, "full cylinder");
    checkEmitsSeveralCoverBoxes(cylinder, "full cylinder");
  }
  {
    surf::CylindricalBoundedSurface cylinder;
    BOOST_REQUIRE(cylinder.initialize({0., 0., 0.}, {0., 0., 1.}, referenceU, 2., -1., 1., 0., surf::kTwoPi, false,
                                      paramRectWireCurves(0.3, 2.1, -0.5, 0.7), {}, error));
    std::vector<Vec3> patchPoints;
    for (int stepPhi = 0; stepPhi <= 10; ++stepPhi) {
      for (int stepH = 0; stepH <= 4; ++stepH) {
        const double phi = 0.3 + 1.8 * stepPhi / 10.;
        const double height = -0.5 + 1.2 * stepH / 4.;
        if (cylinder.pointInTrim(phi, height)) {
          patchPoints.push_back(cylinder.pointAt(phi, height));
        }
      }
    }
    BOOST_REQUIRE(!patchPoints.empty());
    checkCoverBoxProperties(cylinder, patchPoints, "wire-trimmed cylinder");
  }
  {
    surf::SphericalBoundedSurface sphere;
    BOOST_REQUIRE(sphere.initialize(skewCenter, skewAxis, referenceU, 2.5, 0., surf::kPi, 0., surf::kTwoPi,
                                    false, error));
    std::vector<Vec3> patchPoints;
    for (int stepTheta = 0; stepTheta <= 8; ++stepTheta) {
      for (int stepPhi = 0; stepPhi < 16; ++stepPhi) {
        patchPoints.push_back(sphere.pointAt(surf::kPi * stepTheta / 8., surf::kTwoPi * stepPhi / 16.));
      }
    }
    checkCoverBoxProperties(sphere, patchPoints, "full sphere, skew frame");
    checkEmitsSeveralCoverBoxes(sphere, "full sphere, skew frame");
  }
  {
    // a polar cap: distanceSqToPatch realises on the *whole* sphere (radial projection), so the
    // cover boxes must still cover the full ball surface, not merely the cap
    surf::SphericalBoundedSurface cap;
    BOOST_REQUIRE(cap.initialize({0., 0., 0.}, {0., 0., 1.}, referenceU, 2., 0., 0.6, 0.2, 1.1, false, error));
    std::vector<Vec3> patchPoints;
    for (int stepTheta = 0; stepTheta <= 4; ++stepTheta) {
      for (int stepPhi = 0; stepPhi <= 6; ++stepPhi) {
        patchPoints.push_back(cap.pointAt(0.6 * stepTheta / 4., 0.2 + 1.1 * stepPhi / 6.));
      }
    }
    checkCoverBoxProperties(cap, patchPoints, "spherical cap");
  }
  {
    surf::ConicalBoundedSurface cone;
    BOOST_REQUIRE(cone.initialize(skewCenter, skewAxis, referenceU, 2., 0.5, -0.9, 1.1, 0.7, 2.3, false, error));
    std::vector<Vec3> patchPoints;
    for (int stepPhi = 0; stepPhi <= 10; ++stepPhi) {
      for (int stepH = 0; stepH <= 4; ++stepH) {
        patchPoints.push_back(cone.pointAt(0.7 + 2.3 * stepPhi / 10., -0.9 + 2. * stepH / 4.));
      }
    }
    checkCoverBoxProperties(cone, patchPoints, "partial cone, skew axis");
    checkEmitsSeveralCoverBoxes(cone, "partial cone, skew axis");
  }
  {
    surf::TorusBoundedSurface torus;
    BOOST_REQUIRE(torus.initialize(skewCenter, skewAxis, referenceU, 2.4, 0.7, 0.3, 2.1, -0.4, 1.7, false, error));
    std::vector<Vec3> patchPoints;
    for (int stepRing = 0; stepRing <= 10; ++stepRing) {
      for (int stepTube = 0; stepTube <= 6; ++stepTube) {
        patchPoints.push_back(torus.pointAt(0.3 + 2.1 * stepRing / 10., -0.4 + 1.7 * stepTube / 6.));
      }
    }
    checkCoverBoxProperties(torus, patchPoints, "partial torus, skew axis");
    checkEmitsSeveralCoverBoxes(torus, "partial torus, skew axis");
  }
  {
    surf::TorusBoundedSurface torus;
    BOOST_REQUIRE(torus.initialize({0., 0., 0.}, {0., 0., 1.}, referenceU, 3., 1., 0., surf::kTwoPi, 0.,
                                   surf::kTwoPi, false, error));
    std::vector<Vec3> patchPoints;
    for (int stepRing = 0; stepRing < 16; ++stepRing) {
      for (int stepTube = 0; stepTube < 8; ++stepTube) {
        patchPoints.push_back(torus.pointAt(surf::kTwoPi * stepRing / 16., surf::kTwoPi * stepTube / 8.));
      }
    }
    checkCoverBoxProperties(torus, patchPoints, "full torus");
  }
  {
    surf::PlanarBoundedSurface polygon;
    const std::vector<surf::Vec2> rectangle{{0., 0.}, {2., 0.}, {2., 3.}, {0., 3.}};
    BOOST_REQUIRE(polygon.initialize({0.2, -0.4, 0.5}, {1., 0.2, 0.}, {-0.1, 1., 0.3}, rectangle, {}, error));
    std::vector<Vec3> patchPoints;
    patchPoints.push_back(polygon.toGlobal({0.01, 0.01}));
    patchPoints.push_back(polygon.toGlobal({1.9, 2.9}));
    checkCoverBoxProperties(polygon, patchPoints, "planar polygon");
  }
}

/// Rays that cross a swept quadric's old conservative box while missing the surface itself must
/// reach no patch at all once the leaves are sub-patch boxes. Each case here is a ray the single
/// per-surface box turns into a paid analytic intersection and the sub-patch boxes reject on the
/// box test alone.
BOOST_AUTO_TEST_CASE(StreamX_RaysThroughEmptyBoxRegionsReachNoPatch)
{
  // corner of the ball box, well outside the sphere: rho = |(2.2, 2.2)| = 3.11 > 2.5
  const auto sphere = makeSphereSolid("subBoxSphere", 2.5);
  BOOST_CHECK_EQUAL(sphere->CountBVHRayCandidates({2.2, 2.2, -5.}, {0., 0., 1.}), 0);
  BOOST_CHECK_GE(sphere->CountBVHRayCandidates({0., 0., -5.}, {0., 0., 1.}), 1);

  // along the axis of a solid tube: the barrel patch cannot be hit, only the two caps can
  const auto tube = makeTubeSolid("subBoxTube", 0., 2., 1.);
  BOOST_CHECK_EQUAL(tube->CountBVHRayCandidates({0., 0., -5.}, {0., 0., 1.}), 2);

  // corner of the torus box, outside the outer equator: rho = |(3.4, 3.4)| = 4.8 > R + r = 4
  const auto torus = makeTorusSolid("subBoxTorus", 3., 1.);
  BOOST_CHECK_EQUAL(torus->CountBVHRayCandidates({3.4, 3.4, -5.}, {0., 0., 1.}), 0);
  BOOST_CHECK_GE(torus->CountBVHRayCandidates({3., 0., -5.}, {0., 0., 1.}), 1);

  // behind the back of a quarter cylinder: the full rim circles' box is crossed, the sweep band
  // is nowhere near. Not closed (a bare patch), which the BVH does not require.
  SurfaceSolid quarter("subBoxQuarterCylinder");
  BOOST_REQUIRE(quarter.AddCylindricalSurface({0., 0., 0.}, {0., 0., 1.}, {1., 0., 0.}, 2., -1., 1.,
                                              -surf::kPi / 4., surf::kHalfPi));
  quarter.CloseShape(false);
  BOOST_REQUIRE(quarter.HasBVH());
  BOOST_CHECK_EQUAL(quarter.CountBVHRayCandidates({-1.9, -5., 0.}, {0., 1., 0.}), 0);
  BOOST_CHECK_GE(quarter.CountBVHRayCandidates({5., 0., 0.}, {-1., 0., 0.}), 1);
}

/// With several boxes per surface a ray can enter the same surface's leaves more than once, and a
/// duplicated appendIntersections call would flip parity and corrupt the graze clustering. So the
/// dedup is not an optimization but a correctness requirement, and the sharpest way to pin it is
/// the crossing lists themselves: same multiset, both traversals, on the curved fixtures whose
/// surfaces now own many boxes.
BOOST_AUTO_TEST_CASE(StreamX_CurvedFixturesStayIdenticalToTheLoop)
{
  struct Fixture {
    std::unique_ptr<SurfaceSolid> solid;
    double extent;
  };
  std::vector<Fixture> fixtures;
  fixtures.push_back({makeSphereSolid("subBoxSweepSphere", 2.5), 3.5});
  fixtures.push_back({makeTorusSolid("subBoxSweepTorus", 3., 1.), 4.5});
  fixtures.push_back({makeCapsuleSolid("subBoxSweepCapsule", 1., 1.5), 3.});
  fixtures.push_back({makeConeSolid("subBoxSweepCone", 2., 1., 3.), 4.});
  fixtures.push_back({makeWireTrimmedSolid("subBoxSweepWireTrim"), 3.});

  for (const auto& fixture : fixtures) {
    BOOST_TEST_CONTEXT("fixture = " << fixture.solid->GetName())
    {
      sweepDistanceAgainstLoop(*fixture.solid, fixture.extent, 4);
      for (const auto& point : probeGrid(fixture.extent, 4)) {
        BOOST_TEST_CONTEXT("point = (" << point[0] << ", " << point[1] << ", " << point[2] << ")")
        {
          BOOST_CHECK_EQUAL(fixture.solid->Contains(point.data()), fixture.solid->Contains_Loop(point.data()));
          std::vector<SurfaceSolid::ContainsCrossing> bvhCrossings;
          std::vector<SurfaceSolid::ContainsCrossing> loopCrossings;
          fixture.solid->DescribeContainsCrossings({point[0], point[1], point[2]}, bvhCrossings, loopCrossings);
          BOOST_REQUIRE_EQUAL(bvhCrossings.size(), loopCrossings.size());
          for (size_t index = 0; index < bvhCrossings.size(); ++index) {
            BOOST_CHECK_EQUAL(bvhCrossings[index].distance, loopCrossings[index].distance);
          }
        }
      }
    }
  }
}

/// @}
