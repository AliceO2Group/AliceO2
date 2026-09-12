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

#define BOOST_TEST_MODULE Test O2FlatCSG class
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include "CADSupport/O2FlatCSG.h"
#include "CADSupport/O2SurfaceSolidIO.h"

#include "TFile.h"
#include "TGeoBBox.h"
#include "TGeoShape.h"
#include "TGeoTorus.h"
#include "TGeoTube.h"
#include "TMath.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <vector>

namespace
{
using o2::cad::O2FlatCSG;

/// A small deterministic generator, so a failing case is reproducible from its seed alone.
class Rng
{
 public:
  explicit Rng(unsigned long long seed) : mState(seed) {}
  double uniform(double low, double high)
  {
    mState = mState * 6364136223846793005ULL + 1442695040888963407ULL;
    const double unit = static_cast<double>((mState >> 11) & ((1ULL << 53) - 1)) / static_cast<double>(1ULL << 53);
    return low + unit * (high - low);
  }

 private:
  unsigned long long mState;
};

/// The quadric of the plane with outward unit normal \a n through \a p: Q(x) = n.(x - p).
void planeQuadric(const double n[3], const double p[3], double coeff[10])
{
  for (int index = 0; index < 6; ++index) {
    coeff[index] = 0.;
  }
  coeff[6] = 0.5 * n[0];
  coeff[7] = 0.5 * n[1];
  coeff[8] = 0.5 * n[2];
  coeff[9] = -(n[0] * p[0] + n[1] * p[1] + n[2] * p[2]);
}

/// The quadric of the cylinder of radius \a r about the z axis: Q(x) = x^2 + y^2 - r^2.
void zCylinderQuadric(double r, double coeff[10])
{
  const double values[10] = {1., 0., 0., 1., 0., 0., 0., 0., 0., -r * r};
  for (int index = 0; index < 10; ++index) {
    coeff[index] = values[index];
  }
}

/// The quadric of the cylinder of radius \a r about the tilted axis d = (1,1,1)/sqrt(3):
/// Q(x) = x^T (I - d d^T) x - r^2. Every plane in this file has A = 0 and every upright cylinder
/// has A diagonal, so this is the only halfspace with a genuinely nonzero off-diagonal A -- it
/// exists to exercise the half[row]*half[column] cross term in HalfspaceRange's quadric branch,
/// which a mis-indexed variant (half[column]*half[column]) can get past every other quadric here.
void tiltedCylinderQuadric(double r, double coeff[10])
{
  const double s = 1. / std::sqrt(3.);
  const double d[3] = {s, s, s};
  double a[3][3];
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) {
      a[row][column] = (row == column ? 1. : 0.) - d[row] * d[column];
    }
  }
  coeff[0] = a[0][0];
  coeff[1] = a[0][1];
  coeff[2] = a[0][2];
  coeff[3] = a[1][1];
  coeff[4] = a[1][2];
  coeff[5] = a[2][2];
  coeff[6] = 0.;
  coeff[7] = 0.;
  coeff[8] = 0.;
  coeff[9] = -r * r;
}

/// A box of half-extents (dx, dy, dz) centred on the origin, as one cell of six planes.
void addBoxCell(O2FlatCSG& solid, double dx, double dy, double dz)
{
  const double half[3] = {dx, dy, dz};
  const int first = solid.GetNhalfspaces();
  for (int axis = 0; axis < 3; ++axis) {
    for (int sense = -1; sense <= 1; sense += 2) {
      double normal[3] = {0., 0., 0.};
      double through[3] = {0., 0., 0.};
      normal[axis] = static_cast<double>(sense);
      through[axis] = sense * half[axis];
      double coeff[10];
      planeQuadric(normal, through, coeff);
      solid.AddQuadric(1., coeff);
    }
  }
  solid.AddCell(first, 6, 8. * dx * dy * dz);
}
} // namespace

BOOST_AUTO_TEST_CASE(box_from_six_planes_contains_like_TGeoBBox)
{
  O2FlatCSG solid("box");
  addBoxCell(solid, 3., 4., 5.);
  BOOST_CHECK_EQUAL(solid.GetNcells(), 1);
  BOOST_CHECK_EQUAL(solid.GetNhalfspaces(), 6);

  TGeoBBox reference(3., 4., 5.);
  Rng rng(20260824ULL);
  int scored = 0;
  for (int trial = 0; trial < 20000; ++trial) {
    const double point[3] = {rng.uniform(-6., 6.), rng.uniform(-7., 7.), rng.uniform(-8., 8.)};
    // skip the boundary shell, where the two shapes are allowed to disagree by tolerance
    if (std::abs(std::abs(point[0]) - 3.) < 1.e-9 || std::abs(std::abs(point[1]) - 4.) < 1.e-9 ||
        std::abs(std::abs(point[2]) - 5.) < 1.e-9) {
      continue;
    }
    BOOST_REQUIRE_EQUAL(solid.Contains_Loop(point), reference.Contains(point));
    BOOST_REQUIRE_EQUAL(solid.Contains(point), solid.Contains_Loop(point));
    ++scored;
  }
  BOOST_CHECK_GT(scored, 19000);
}

BOOST_AUTO_TEST_CASE(tube_from_two_cylinders_and_two_planes_contains_like_TGeoTube)
{
  // rmin = 2, rmax = 5, dz = 7: the inner cylinder is a COMPLEMENTED halfspace, which is what
  // makes this cell non-convex and is the case the whole class exists for.
  O2FlatCSG solid("tube");
  double coeff[10];
  zCylinderQuadric(5., coeff);
  solid.AddQuadric(1., coeff);
  zCylinderQuadric(2., coeff);
  solid.AddQuadric(-1., coeff);
  const double up[3] = {0., 0., 1.};
  const double down[3] = {0., 0., -1.};
  const double top[3] = {0., 0., 7.};
  const double bottom[3] = {0., 0., -7.};
  planeQuadric(up, top, coeff);
  solid.AddQuadric(1., coeff);
  planeQuadric(down, bottom, coeff);
  solid.AddQuadric(1., coeff);
  solid.AddCell(0, 4, TMath::Pi() * (25. - 4.) * 14.);

  TGeoTube reference(2., 5., 7.);
  Rng rng(777ULL);
  for (int trial = 0; trial < 20000; ++trial) {
    const double point[3] = {rng.uniform(-6., 6.), rng.uniform(-6., 6.), rng.uniform(-8., 8.)};
    const double radius = std::hypot(point[0], point[1]);
    if (std::abs(radius - 2.) < 1.e-9 || std::abs(radius - 5.) < 1.e-9 ||
        std::abs(std::abs(point[2]) - 7.) < 1.e-9) {
      continue;
    }
    BOOST_TEST_CONTEXT("point = (" << point[0] << ", " << point[1] << ", " << point[2] << ")")
    {
      BOOST_REQUIRE_EQUAL(solid.Contains_Loop(point), reference.Contains(point));
    }
  }
}

BOOST_AUTO_TEST_CASE(two_disjoint_cells_are_a_union)
{
  O2FlatCSG solid("two_boxes");
  addBoxCell(solid, 1., 1., 1.);
  // a second box, centred at x = +10, as six planes of its own
  const int first = solid.GetNhalfspaces();
  const double centre = 10.;
  for (int axis = 0; axis < 3; ++axis) {
    for (int sense = -1; sense <= 1; sense += 2) {
      double normal[3] = {0., 0., 0.};
      double through[3] = {centre, 0., 0.};
      normal[axis] = static_cast<double>(sense);
      through[axis] += (axis == 0 ? sense * 1. : 0.);
      if (axis != 0) {
        through[axis] = sense * 1.;
      }
      double coeff[10];
      planeQuadric(normal, through, coeff);
      solid.AddQuadric(1., coeff);
    }
  }
  solid.AddCell(first, 6, 8.);

  const double inFirst[3] = {0., 0., 0.};
  const double inSecond[3] = {10., 0., 0.};
  const double between[3] = {5., 0., 0.};
  BOOST_CHECK(solid.Contains_Loop(inFirst));
  BOOST_CHECK(solid.Contains_Loop(inSecond));
  BOOST_CHECK(!solid.Contains_Loop(between));
}

BOOST_AUTO_TEST_CASE(box_distances_match_TGeoBBox)
{
  O2FlatCSG solid("box_dist");
  addBoxCell(solid, 3., 4., 5.);
  TGeoBBox reference(3., 4., 5.);

  Rng rng(4242ULL);
  for (int trial = 0; trial < 20000; ++trial) {
    double point[3] = {rng.uniform(-12., 12.), rng.uniform(-12., 12.), rng.uniform(-12., 12.)};
    double dir[3];
    double norm = 0.;
    do {
      for (int index = 0; index < 3; ++index) {
        dir[index] = rng.uniform(-1., 1.);
      }
      norm = std::sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
    } while (norm < 1.e-3);
    for (int index = 0; index < 3; ++index) {
      dir[index] /= norm;
    }
    const bool inside = reference.Contains(point);
    if (inside != static_cast<bool>(solid.Contains_Loop(point))) {
      continue; // a boundary point; classification is tested separately
    }
    const double mine = inside ? solid.DistFromInside_Loop(point, dir, TGeoShape::Big())
                               : solid.DistFromOutside_Loop(point, dir, TGeoShape::Big());
    const double theirs = inside ? reference.DistFromInside(point, dir, 3, TGeoShape::Big(), nullptr)
                                 : reference.DistFromOutside(point, dir, 3, TGeoShape::Big(), nullptr);
    if (theirs >= TGeoShape::Big()) {
      BOOST_REQUIRE_GE(mine, TGeoShape::Big());
    } else {
      BOOST_REQUIRE_SMALL(mine - theirs, 1.e-9);
    }
  }
}

BOOST_AUTO_TEST_CASE(tube_distances_match_TGeoTube_through_the_bore)
{
  // the complemented inner cylinder makes the occupancy along a ray TWO intervals for a ray that
  // crosses the bore, which is the case a convexity assumption would get wrong
  O2FlatCSG solid("tube_dist");
  double coeff[10];
  zCylinderQuadric(5., coeff);
  solid.AddQuadric(1., coeff);
  zCylinderQuadric(2., coeff);
  solid.AddQuadric(-1., coeff);
  const double up[3] = {0., 0., 1.};
  const double down[3] = {0., 0., -1.};
  const double top[3] = {0., 0., 7.};
  const double bottom[3] = {0., 0., -7.};
  planeQuadric(up, top, coeff);
  solid.AddQuadric(1., coeff);
  planeQuadric(down, bottom, coeff);
  solid.AddQuadric(1., coeff);
  solid.AddCell(0, 4, 0.);

  TGeoTube reference(2., 5., 7.);
  // a ray straight along +x at z = 0 enters the wall at x = -5, leaves it at x = -2, re-enters at
  // x = +2 and leaves at x = +5
  const double origin[3] = {-9., 0., 0.};
  const double dir[3] = {1., 0., 0.};
  BOOST_CHECK_SMALL(solid.DistFromOutside_Loop(origin, dir, TGeoShape::Big()) - 4., 1.e-12);

  const double inWall[3] = {-4., 0., 0.};
  BOOST_CHECK_SMALL(solid.DistFromInside_Loop(inWall, dir, TGeoShape::Big()) - 2., 1.e-12);

  const double inBore[3] = {0., 0., 0.};
  BOOST_CHECK(!solid.Contains_Loop(inBore));
  BOOST_CHECK_SMALL(solid.DistFromOutside_Loop(inBore, dir, TGeoShape::Big()) - 2., 1.e-12);

  Rng rng(99ULL);
  for (int trial = 0; trial < 20000; ++trial) {
    double point[3] = {rng.uniform(-9., 9.), rng.uniform(-9., 9.), rng.uniform(-10., 10.)};
    double direction[3];
    double norm = 0.;
    do {
      for (int index = 0; index < 3; ++index) {
        direction[index] = rng.uniform(-1., 1.);
      }
      norm = std::sqrt(direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2]);
    } while (norm < 1.e-3);
    for (int index = 0; index < 3; ++index) {
      direction[index] /= norm;
    }
    const bool inside = reference.Contains(point);
    if (inside != static_cast<bool>(solid.Contains_Loop(point))) {
      continue;
    }
    const double mine = inside ? solid.DistFromInside_Loop(point, direction, TGeoShape::Big())
                               : solid.DistFromOutside_Loop(point, direction, TGeoShape::Big());
    const double theirs = inside ? reference.DistFromInside(point, direction, 3, TGeoShape::Big(), nullptr)
                                 : reference.DistFromOutside(point, direction, 3, TGeoShape::Big(), nullptr);
    if (theirs >= TGeoShape::Big()) {
      BOOST_REQUIRE_GE(mine, TGeoShape::Big());
    } else {
      BOOST_REQUIRE_SMALL(mine - theirs, 1.e-8);
    }
  }
}

BOOST_AUTO_TEST_CASE(a_ray_leaving_one_cell_into_a_touching_one_does_not_stop_between_them)
{
  // two unit boxes sharing the face at x = 1: the union's DistFromInside from the origin along +x
  // is 3, not 1. This is why DistFromInside needs the union across cells and not one cell's exit.
  O2FlatCSG solid("touching");
  addBoxCell(solid, 1., 1., 1.);
  const int first = solid.GetNhalfspaces();
  const double planes[6][2][3] = {{{1., 0., 0.}, {3., 0., 0.}},
                                  {{-1., 0., 0.}, {1., 0., 0.}},
                                  {{0., 1., 0.}, {0., 1., 0.}},
                                  {{0., -1., 0.}, {0., -1., 0.}},
                                  {{0., 0., 1.}, {0., 0., 1.}},
                                  {{0., 0., -1.}, {0., 0., -1.}}};
  for (const auto& plane : planes) {
    double coeff[10];
    planeQuadric(plane[0], plane[1], coeff);
    solid.AddQuadric(1., coeff);
  }
  solid.AddCell(first, 6, 8.);

  const double origin[3] = {0., 0., 0.};
  const double dir[3] = {1., 0., 0.};
  BOOST_CHECK_SMALL(solid.DistFromInside_Loop(origin, dir, TGeoShape::Big()) - 3., 1.e-12);
}

BOOST_AUTO_TEST_CASE(tangential_ray_on_a_cylinder_from_a_point_on_its_surface_has_no_nan_root)
{
  // a ray tangential to a cylinder, starting exactly on its surface, has beta == 0 and gamma == 0
  // together in HalfspaceRoots' quadratic -- the q == 0 case that used to divide 0./0. into a
  // NaN second root instead of recognising the single double root at t = 0
  O2FlatCSG solid("tangent_ray");
  double coeff[10];
  zCylinderQuadric(5., coeff);
  solid.AddQuadric(1., coeff);
  const auto& cylinder = solid.GetHalfspace(0);

  const double origin[3] = {5., 0., 0.};
  const double dir[3] = {0., 1., 0.};
  double roots[4];
  const int found = O2FlatCSG::HalfspaceRoots(cylinder, origin, dir, roots);

  BOOST_REQUIRE_EQUAL(found, 1);
  BOOST_CHECK(std::isfinite(roots[0]));
  BOOST_CHECK_SMALL(roots[0], 1.e-12);

  // the twin: an independent check that the reported root really is one, by plugging it back
  // into the surface equation directly rather than trusting the root-finder's own algebra
  const double hit[3] = {origin[0] + roots[0] * dir[0], origin[1] + roots[0] * dir[1],
                         origin[2] + roots[0] * dir[2]};
  BOOST_CHECK_SMALL(O2FlatCSG::EvalHalfspace(cylinder, hit), 1.e-9);
}

BOOST_AUTO_TEST_CASE(torus_contains_and_distances_match_TGeoTorus)
{
  // a full torus, R = 10, r = 3, about z -- one cell of one halfspace
  O2FlatCSG solid("torus");
  const double centre[3] = {0., 0., 0.};
  const double axis[3] = {0., 0., 1.};
  solid.AddTorus(1., centre, axis, 10., 3.);
  solid.AddCell(0, 1, 2. * TMath::Pi() * TMath::Pi() * 10. * 9.);

  TGeoTorus reference(10., 0., 3.);
  Rng rng(31415ULL);
  int scoredPoints = 0;
  for (int trial = 0; trial < 20000; ++trial) {
    const double point[3] = {rng.uniform(-15., 15.), rng.uniform(-15., 15.), rng.uniform(-5., 5.)};
    const double radial = std::hypot(point[0], point[1]);
    const double distance = std::hypot(radial - 10., point[2]) - 3.;
    if (std::abs(distance) < 1.e-9) {
      continue;
    }
    BOOST_REQUIRE_EQUAL(solid.Contains_Loop(point), reference.Contains(point));
    ++scoredPoints;
  }
  BOOST_CHECK_GT(scoredPoints, 19000);

  for (int trial = 0; trial < 20000; ++trial) {
    double point[3] = {rng.uniform(-20., 20.), rng.uniform(-20., 20.), rng.uniform(-8., 8.)};
    double dir[3];
    double norm = 0.;
    do {
      for (int index = 0; index < 3; ++index) {
        dir[index] = rng.uniform(-1., 1.);
      }
      norm = std::sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
    } while (norm < 1.e-3);
    for (int index = 0; index < 3; ++index) {
      dir[index] /= norm;
    }
    const bool inside = reference.Contains(point);
    if (inside != static_cast<bool>(solid.Contains_Loop(point))) {
      continue;
    }
    const double mine = inside ? solid.DistFromInside_Loop(point, dir, TGeoShape::Big())
                               : solid.DistFromOutside_Loop(point, dir, TGeoShape::Big());
    const double theirs = inside ? reference.DistFromInside(point, dir, 3, TGeoShape::Big(), nullptr)
                                 : reference.DistFromOutside(point, dir, 3, TGeoShape::Big(), nullptr);
    if (theirs >= TGeoShape::Big()) {
      BOOST_REQUIRE_GE(mine, TGeoShape::Big());
    } else {
      // the quartic is the looser of the two solvers; 1e-6 cm on a 10 cm torus
      BOOST_REQUIRE_SMALL(mine - theirs, 1.e-6);
    }
  }
}

BOOST_AUTO_TEST_CASE(a_tilted_torus_is_the_same_solid_as_an_upright_one_rotated)
{
  // the frame handling is where a torus block goes wrong silently, so it gets its own case
  const double axis[3] = {0., 1. / std::sqrt(2.), 1. / std::sqrt(2.)};
  const double centre[3] = {1., 2., 3.};
  O2FlatCSG solid("tilted_torus");
  solid.AddTorus(1., centre, axis, 8., 2.);
  solid.AddCell(0, 1, 0.);

  Rng rng(2718ULL);
  for (int trial = 0; trial < 20000; ++trial) {
    const double point[3] = {rng.uniform(-14., 16.), rng.uniform(-13., 17.), rng.uniform(-12., 18.)};
    // the closed-form signed distance is the reference: sqrt((rho - R)^2 + z^2) - r
    const double offset[3] = {point[0] - centre[0], point[1] - centre[1], point[2] - centre[2]};
    const double along = offset[0] * axis[0] + offset[1] * axis[1] + offset[2] * axis[2];
    double radialVec[3];
    for (int index = 0; index < 3; ++index) {
      radialVec[index] = offset[index] - along * axis[index];
    }
    const double rho = std::sqrt(radialVec[0] * radialVec[0] + radialVec[1] * radialVec[1] +
                                 radialVec[2] * radialVec[2]);
    const double signedDistance = std::hypot(rho - 8., along) - 2.;
    if (std::abs(signedDistance) < 1.e-9) {
      continue;
    }
    BOOST_REQUIRE_EQUAL(static_cast<bool>(solid.Contains_Loop(point)), signedDistance < 0.);
  }

  // Contains_Loop only exercises EvalHalfspace's frame decomposition; HalfspaceRoots has its own,
  // separate one (the pz/dz/pPerp/dPerp block), and the upright case never gives it a non-z axis
  // to get wrong. Compare distances against TGeoTorus by carrying a local (upright, origin-
  // centred) point and direction alongside a world one related by the same rotation that carries
  // the local z axis onto `axis`, so the reference and the shape describe the same solid.
  const double s = 1. / std::sqrt(2.);
  // rotation about the world x axis that sends local (0,0,1) to (0, s, s) == axis
  auto rotateToWorld = [s](const double local[3], double world[3]) {
    world[0] = local[0];
    world[1] = s * local[1] + s * local[2];
    world[2] = -s * local[1] + s * local[2];
  };

  TGeoTorus reference(8., 0., 2.);
  for (int trial = 0; trial < 20000; ++trial) {
    double localPoint[3] = {rng.uniform(-20., 20.), rng.uniform(-20., 20.), rng.uniform(-8., 8.)};
    double localDir[3];
    double norm = 0.;
    do {
      for (int index = 0; index < 3; ++index) {
        localDir[index] = rng.uniform(-1., 1.);
      }
      norm = std::sqrt(localDir[0] * localDir[0] + localDir[1] * localDir[1] + localDir[2] * localDir[2]);
    } while (norm < 1.e-3);
    for (int index = 0; index < 3; ++index) {
      localDir[index] /= norm;
    }
    double worldPoint[3];
    double worldDir[3];
    rotateToWorld(localPoint, worldPoint);
    rotateToWorld(localDir, worldDir);
    for (int index = 0; index < 3; ++index) {
      worldPoint[index] += centre[index];
    }

    const bool inside = reference.Contains(localPoint);
    if (inside != static_cast<bool>(solid.Contains_Loop(worldPoint))) {
      continue;
    }
    const double mine = inside ? solid.DistFromInside_Loop(worldPoint, worldDir, TGeoShape::Big())
                               : solid.DistFromOutside_Loop(worldPoint, worldDir, TGeoShape::Big());
    const double theirs = inside ? reference.DistFromInside(localPoint, localDir, 3, TGeoShape::Big(), nullptr)
                                 : reference.DistFromOutside(localPoint, localDir, 3, TGeoShape::Big(), nullptr);
    if (theirs >= TGeoShape::Big()) {
      BOOST_REQUIRE_GE(mine, TGeoShape::Big());
    } else {
      BOOST_REQUIRE_SMALL(mine - theirs, 1.e-6);
    }
  }
}

BOOST_AUTO_TEST_CASE(the_range_bound_encloses_the_sampled_range)
{
  // the bound must be an ENCLOSURE: over-wide is safe, under-wide is a wrong solid
  O2FlatCSG solid("range");
  double coeff[10];
  zCylinderQuadric(5., coeff);
  const int cylinder = solid.AddQuadric(1., coeff);
  const double normal[3] = {0., 0., 1.};
  const double through[3] = {0., 0., 2.};
  planeQuadric(normal, through, coeff);
  const int plane = solid.AddQuadric(-1., coeff);
  const double centre[3] = {1., 0., 0.};
  const double axis[3] = {0., 0., 1.};
  const int torus = solid.AddTorus(1., centre, axis, 7., 2.);
  tiltedCylinderQuadric(5., coeff);
  const int tilted = solid.AddQuadric(1., coeff);

  Rng rng(555ULL);
  const std::vector<int> halfspaces = {cylinder, plane, torus, tilted};

  auto checkBox = [&](const double* lo, const double* hi) {
    for (int which : halfspaces) {
      double rangeLo = 0.;
      double rangeHi = 0.;
      O2FlatCSG::HalfspaceRange(solid.GetHalfspace(which), lo, hi, rangeLo, rangeHi);
      BOOST_REQUIRE_LE(rangeLo, rangeHi);

      auto checkPoint = [&](const double point[3]) {
        const double value = O2FlatCSG::EvalHalfspace(solid.GetHalfspace(which), point);
        BOOST_REQUIRE_GE(value, rangeLo - 1.e-9);
        BOOST_REQUIRE_LE(value, rangeHi + 1.e-9);
      };

      // deterministic coverage of the box's extremities: a plane's bound is tight exactly AT a
      // corner, so uniform interior sampling has probability zero of ever landing where a
      // slightly under-wide bound would actually be caught
      for (int cx : {0, 1}) {
        for (int cy : {0, 1}) {
          for (int cz : {0, 1}) {
            const double corner[3] = {cx ? hi[0] : lo[0], cy ? hi[1] : lo[1], cz ? hi[2] : lo[2]};
            checkPoint(corner);
          }
        }
      }
      const double mid[3] = {0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), 0.5 * (lo[2] + hi[2])};
      for (int faceAxis = 0; faceAxis < 3; ++faceAxis) {
        for (int side : {0, 1}) {
          double face[3] = {mid[0], mid[1], mid[2]};
          face[faceAxis] = side ? hi[faceAxis] : lo[faceAxis];
          checkPoint(face);
        }
      }
      for (int edgeAxis = 0; edgeAxis < 3; ++edgeAxis) {
        const int other1 = (edgeAxis + 1) % 3;
        const int other2 = (edgeAxis + 2) % 3;
        for (int s1 : {0, 1}) {
          for (int s2 : {0, 1}) {
            double edge[3];
            edge[edgeAxis] = mid[edgeAxis];
            edge[other1] = s1 ? hi[other1] : lo[other1];
            edge[other2] = s2 ? hi[other2] : lo[other2];
            checkPoint(edge);
          }
        }
      }

      // plus random interior samples, as before
      for (int sample = 0; sample < 200; ++sample) {
        const double point[3] = {rng.uniform(lo[0], hi[0]), rng.uniform(lo[1], hi[1]),
                                 rng.uniform(lo[2], hi[2])};
        checkPoint(point);
      }
    }
  };

  for (int trial = 0; trial < 3000; ++trial) {
    double lo[3];
    double hi[3];
    for (int index = 0; index < 3; ++index) {
      const double a = rng.uniform(-12., 12.);
      const double b = a + rng.uniform(0.01, 6.);
      lo[index] = a;
      hi[index] = b;
    }
    checkBox(lo, hi);
  }

  // extreme-aspect-ratio boxes -- one axis ~0.01 wide, another ~24 -- outside the size range the
  // random trials above ever draw (at most 6 wide per axis)
  const double extreme[4][3][2] = {
    {{-0.005, 0.005}, {-12., 12.}, {-0.5, 0.5}},
    {{-12., 12.}, {-0.005, 0.005}, {3., 27.}},
    {{2., 2.01}, {-1., 1.}, {-12., 12.}},
    {{-24., 0.}, {5., 5.01}, {-3., 3.}},
  };
  for (const auto& box : extreme) {
    const double lo[3] = {box[0][0], box[1][0], box[2][0]};
    const double hi[3] = {box[0][1], box[1][1], box[2][1]};
    checkBox(lo, hi);
  }
}

BOOST_AUTO_TEST_CASE(the_boxes_cover_the_solid_and_their_active_lists_are_sound)
{
  // rmin = 2, rmax = 5, dz = 7 again, so there is a bore for the boxes to carve around
  O2FlatCSG solid("boxes");
  double coeff[10];
  zCylinderQuadric(5., coeff);
  solid.AddQuadric(1., coeff);
  zCylinderQuadric(2., coeff);
  solid.AddQuadric(-1., coeff);
  const double up[3] = {0., 0., 1.};
  const double down[3] = {0., 0., -1.};
  const double top[3] = {0., 0., 7.};
  const double bottom[3] = {0., 0., -7.};
  planeQuadric(up, top, coeff);
  solid.AddQuadric(1., coeff);
  planeQuadric(down, bottom, coeff);
  solid.AddQuadric(1., coeff);
  solid.AddCell(0, 4, 0.);
  const double lo[3] = {-5., -5., -7.};
  const double hi[3] = {5., 5., 7.};
  solid.SetCellBBox(0, lo, hi);
  solid.CloseShape();

  BOOST_CHECK_GT(solid.GetNboxes(), 1);

  Rng rng(8080ULL);
  int insideSamples = 0;
  for (int trial = 0; trial < 50000; ++trial) {
    const double point[3] = {rng.uniform(-6., 6.), rng.uniform(-6., 6.), rng.uniform(-8., 8.)};
    if (!solid.Contains_Loop(point)) {
      continue;
    }
    ++insideSamples;
    // COVERAGE: every point of the solid is in some box
    bool covered = false;
    for (int index = 0; index < solid.GetNboxes() && !covered; ++index) {
      const auto& box = solid.GetBox(index);
      covered = point[0] >= box.min[0] && point[0] <= box.max[0] && point[1] >= box.min[1] &&
                point[1] <= box.max[1] && point[2] >= box.min[2] && point[2] <= box.max[2];
    }
    BOOST_REQUIRE(covered);
  }
  BOOST_CHECK_GT(insideSamples, 5000);

  // SOUNDNESS of the active lists: in every box, the active list alone decides membership
  for (int index = 0; index < solid.GetNboxes(); ++index) {
    const auto& box = solid.GetBox(index);
    for (int sample = 0; sample < 200; ++sample) {
      const double point[3] = {rng.uniform(box.min[0], box.max[0]),
                               rng.uniform(box.min[1], box.max[1]),
                               rng.uniform(box.min[2], box.max[2])};
      bool byActive = true;
      for (int slot = 0; slot < box.nActive && byActive; ++slot) {
        byActive = O2FlatCSG::EvalHalfspace(
                     solid.GetHalfspace(solid.GetActive(box.firstActive + slot)), point) <= 0.;
      }
      BOOST_REQUIRE_EQUAL(byActive, solid.CellContains(box.cell, point));
    }
  }
}

BOOST_AUTO_TEST_CASE(a_box_wholly_inside_a_cell_carries_no_active_halfspaces)
{
  O2FlatCSG solid("solid_boxes");
  addBoxCell(solid, 4., 4., 4.);
  const double lo[3] = {-4., -4., -4.};
  const double hi[3] = {4., 4., 4.};
  solid.SetCellBBox(0, lo, hi);
  // Both knobs are pinned, not defaulted: this case is about what the subdivision CAN produce,
  // and the shipped defaults are chosen for query cost, which is
  // a different question. Six levels on a cube is a 4 x 4 x 4 grid, whose innermost eight boxes
  // touch no face.
  solid.SetSplitDepth(6);
  solid.SetMinBoxFraction(0.01);
  solid.CloseShape();
  // a box has six planes and is convex, so subdivision must find interior boxes with an empty list
  int solidBoxes = 0;
  for (int index = 0; index < solid.GetNboxes(); ++index) {
    if (solid.GetBox(index).nActive == 0) {
      ++solidBoxes;
    }
  }
  BOOST_CHECK_GT(solidBoxes, 0);
}

BOOST_AUTO_TEST_CASE(a_cell_without_a_bbox_fails_loudly_instead_of_vanishing)
{
  // cell 0 gets a box; cell 1 (a second, disjoint box) never does -- CloseShape must refuse to
  // build a partial, silently-wrong solid rather than just drop cell 1
  O2FlatCSG solid("missing_bbox");
  addBoxCell(solid, 1., 1., 1.);
  const int first = solid.GetNhalfspaces();
  const double centre[3] = {10., 0., 0.};
  for (int axis = 0; axis < 3; ++axis) {
    for (int sense = -1; sense <= 1; sense += 2) {
      double normal[3] = {0., 0., 0.};
      double through[3] = {centre[0], centre[1], centre[2]};
      normal[axis] = static_cast<double>(sense);
      through[axis] += sense * 1.;
      double coeff[10];
      planeQuadric(normal, through, coeff);
      solid.AddQuadric(1., coeff);
    }
  }
  solid.AddCell(first, 6, 8.);

  const double lo[3] = {-1., -1., -1.};
  const double hi[3] = {1., 1., 1.};
  solid.SetCellBBox(0, lo, hi); // cell 1's box is never set

  solid.CloseShape();
  BOOST_CHECK(!solid.IsClosed());
  BOOST_CHECK_EQUAL(solid.GetNboxes(), 0);
}

BOOST_AUTO_TEST_CASE(an_inverted_cell_bbox_fails_loudly_instead_of_being_kept_as_solid)
{
  // a converter that swapped lo/hi arguments must not get a shape that quietly reports itself
  // closed: an all-axes-inverted box never grows past SplitBox's longest = 0. initialiser, so it
  // would otherwise be kept immediately with an active list computed from a negative-half-extent
  // (hence invalid) range bound -- possibly nActive == 0, which downstream reads as solid material
  O2FlatCSG solid("inverted_bbox");
  addBoxCell(solid, 1., 1., 1.);
  const double lo[3] = {-1., -1., -1.};
  const double hi[3] = {1., 1., 1.};
  solid.SetCellBBox(0, hi, lo); // lo/hi swapped

  solid.CloseShape();
  BOOST_CHECK(!solid.IsClosed());
  BOOST_CHECK_EQUAL(solid.GetNboxes(), 0);
}

BOOST_AUTO_TEST_CASE(a_nan_cell_bbox_fails_loudly_instead_of_defeating_the_inverted_box_check)
{
  // a NaN passes every ordinary "hi < lo" comparison silently (every comparison with NaN is
  // false), so it must be its own check rather than fall through the inverted-box test above --
  // otherwise it would reach HalfspaceRange, produce a NaN range that fails both of SplitBox's
  // drop tests, and get kept as a spurious box
  O2FlatCSG solid("nan_bbox");
  addBoxCell(solid, 1., 1., 1.);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double lo[3] = {-1., -1., -1.};
  const double hi[3] = {1., nan, 1.};
  solid.SetCellBBox(0, lo, hi);

  solid.CloseShape();
  BOOST_CHECK(!solid.IsClosed());
  BOOST_CHECK_EQUAL(solid.GetNboxes(), 0);
}

namespace
{
/// An L-shaped bracket with a bore: three cells, a complemented cylinder, a long diagonal extent.
/// Deliberately the shape a cell-level BVH would handle badly.
///
/// The washer sits ABOVE the arm, at z in [1, 3], so its bore is a genuine hole in the union: the
/// arm spans |y| <= 1 and |z| <= 1, so a washer at |z| <= 1 would have had its own bore filled in
/// by the arm and the solid would have had no cavity anywhere.
///
/// \a planeScale multiplies every PLANE quadric. `sign * Q <= 0` is the same halfspace for any
/// positive scale, so the solid is unchanged -- but the accelerated queries are only bit-identical
/// to their twins when the scale is a power of two; see the rescaled test below and
/// Detectors/CADSupport/doc/reference/Design_FlatCSGSolid.md section 3.1.
void buildBracket(O2FlatCSG& solid, double planeScale = 1.)
{
  double coeff[10];
  const auto scaledPlane = [&](const double* normal, const double* through) {
    planeQuadric(normal, through, coeff);
    for (int index = 0; index < 10; ++index) {
      coeff[index] *= planeScale;
    }
  };
  // cell 0: the long arm, x in [-10, 10], y in [-1, 1], z in [-1, 1]
  const double arm[6][2][3] = {{{1., 0., 0.}, {10., 0., 0.}},
                               {{-1., 0., 0.}, {-10., 0., 0.}},
                               {{0., 1., 0.}, {0., 1., 0.}},
                               {{0., -1., 0.}, {0., -1., 0.}},
                               {{0., 0., 1.}, {0., 0., 1.}},
                               {{0., 0., -1.}, {0., 0., -1.}}};
  int first = solid.GetNhalfspaces();
  for (const auto& plane : arm) {
    scaledPlane(plane[0], plane[1]);
    solid.AddQuadric(1., coeff);
  }
  solid.AddCell(first, 6, 8. * 10. * 1. * 1.);
  const double armLo[3] = {-10., -1., -1.};
  const double armHi[3] = {10., 1., 1.};
  solid.SetCellBBox(0, armLo, armHi);

  // cell 1: the upright, x in [8, 10], y in [-1, 1], z in [1, 12]
  const double upright[6][2][3] = {{{1., 0., 0.}, {10., 0., 0.}},
                                   {{-1., 0., 0.}, {8., 0., 0.}},
                                   {{0., 1., 0.}, {0., 1., 0.}},
                                   {{0., -1., 0.}, {0., -1., 0.}},
                                   {{0., 0., 1.}, {0., 0., 12.}},
                                   {{0., 0., -1.}, {0., 0., 1.}}};
  first = solid.GetNhalfspaces();
  for (const auto& plane : upright) {
    scaledPlane(plane[0], plane[1]);
    solid.AddQuadric(1., coeff);
  }
  solid.AddCell(first, 6, 2. * 2. * 11.);
  const double uprightLo[3] = {8., -1., 1.};
  const double uprightHi[3] = {10., 1., 12.};
  solid.SetCellBBox(1, uprightLo, uprightHi);

  // cell 2: a washer around z at x = -8 and z in [1, 3], with a bore -- a complemented cylinder,
  // so non-convex, and clear of the arm so the bore is empty space
  first = solid.GetNhalfspaces();
  const double centreShift = -8.;
  // outer cylinder about the axis through (-8, 0, *): translate by completing the square
  const double outer[10] = {1., 0., 0., 1., 0., 0., -centreShift, 0., 0.,
                            centreShift * centreShift - 9.};
  solid.AddQuadric(1., outer);
  const double inner[10] = {1., 0., 0., 1., 0., 0., -centreShift, 0., 0.,
                            centreShift * centreShift - 1.};
  solid.AddQuadric(-1., inner);
  const double washer[2][2][3] = {{{0., 0., 1.}, {0., 0., 3.}}, {{0., 0., -1.}, {0., 0., 1.}}};
  for (const auto& plane : washer) {
    scaledPlane(plane[0], plane[1]);
    solid.AddQuadric(1., coeff);
  }
  solid.AddCell(first, 4, TMath::Pi() * (9. - 1.) * 2.);
  const double washerLo[3] = {-11., -3., 1.};
  const double washerHi[3] = {-5., 3., 3.};
  solid.SetCellBBox(2, washerLo, washerHi);
}
} // namespace

BOOST_AUTO_TEST_CASE(the_accelerated_contains_is_bit_identical_to_its_twin)
{
  O2FlatCSG solid("bracket");
  buildBracket(solid);
  solid.CloseShape();
  BOOST_CHECK_GT(solid.GetNboxes(), 3);
  BOOST_CHECK_GT(solid.GetBVHMemory(), 0u);

  Rng rng(123456ULL);
  for (int trial = 0; trial < 200000; ++trial) {
    const double point[3] = {rng.uniform(-13., 13.), rng.uniform(-5., 5.), rng.uniform(-3., 14.)};
    BOOST_REQUIRE_EQUAL(solid.Contains(point), solid.Contains_Loop(point));
  }
}

BOOST_AUTO_TEST_CASE(the_sampled_boundary_points_flip_containment)
{
  O2FlatCSG solid("bracket_points");
  buildBracket(solid);
  solid.CloseShape();
  constexpr int kPoints = 4000;
  std::vector<double> points(3 * kPoints, 0.);
  BOOST_REQUIRE(solid.GetPointsOnSegments(kPoints, points.data()));
  const double zAxis[3] = {0., 0., 1.};
  for (int index = 0; index < kPoints; ++index) {
    const double* point = &points[3 * index];
    double normal[3] = {0., 0., 0.};
    solid.ComputeNormal(point, zAxis, normal);
    double below[3];
    double above[3];
    for (int axis = 0; axis < 3; ++axis) {
      below[axis] = point[axis] - 1.e-6 * normal[axis];
      above[axis] = point[axis] + 1.e-6 * normal[axis];
    }
    BOOST_TEST_CONTEXT("point = (" << point[0] << ", " << point[1] << ", " << point[2] << ")")
    {
      BOOST_CHECK_NE(solid.Contains(below), solid.Contains(above));
    }
  }
}

BOOST_AUTO_TEST_CASE(the_accelerated_distances_are_bit_identical_to_their_twins)
{
  O2FlatCSG solid("bracket_dist");
  buildBracket(solid);
  solid.CloseShape();

  Rng rng(654321ULL);
  for (int trial = 0; trial < 200000; ++trial) {
    double point[3] = {rng.uniform(-16., 16.), rng.uniform(-8., 8.), rng.uniform(-6., 17.)};
    double dir[3];
    double norm = 0.;
    do {
      for (int index = 0; index < 3; ++index) {
        dir[index] = rng.uniform(-1., 1.);
      }
      norm = std::sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
    } while (norm < 1.e-3);
    for (int index = 0; index < 3; ++index) {
      dir[index] /= norm;
    }
    if (solid.Contains_Loop(point)) {
      BOOST_REQUIRE_EQUAL(solid.DistFromInside(point, dir, 3, TGeoShape::Big(), nullptr),
                          solid.DistFromInside_Loop(point, dir, TGeoShape::Big()));
    } else {
      BOOST_REQUIRE_EQUAL(solid.DistFromOutside(point, dir, 3, TGeoShape::Big(), nullptr),
                          solid.DistFromOutside_Loop(point, dir, TGeoShape::Big()));
    }
  }
}

BOOST_AUTO_TEST_CASE(a_ray_along_the_long_arm_crosses_every_cell_it_should)
{
  // the case a per-box clip gets wrong if it forgets to clip: a ray running the length of the
  // bracket passes through many boxes of the same cell, and must see ONE interval, not many
  O2FlatCSG solid("bracket_long");
  buildBracket(solid);
  solid.CloseShape();
  const double dir[3] = {1., 0., 0.};

  // the entry, which one box decides on its own: nothing lies before the arm along z = 0
  const double origin[3] = {-20., 0., 0.};
  BOOST_CHECK_SMALL(solid.DistFromOutside(origin, dir, 3, TGeoShape::Big(), nullptr) - 10., 1.e-12);

  // the exit, which sixteen boxes of cell 0 decide together: the arm is split along x into boxes
  // 1.25 wide, so this is the cross-box join, and a traversal that forgot it would stop at the
  // first box boundary
  const double inArm[3] = {0., 0., 0.};
  // inside the arm at the origin, the exit is x = 10 (the arm and the upright touch at x = 8..10
  // only for z > 1, so along z = 0 the arm alone decides)
  BOOST_CHECK_SMALL(solid.DistFromInside(inArm, dir, 3, TGeoShape::Big(), nullptr) - 10., 1.e-12);

  // An ENTRY that needs the per-cell merge, which is otherwise hard to reach: the twin's rule
  // takes the smallest entry over the intervals whose exit clears TGeoShape::Tolerance(), so a
  // box boundary crossed within the tolerance of the origin cuts the real interval into a
  // sub-tolerance stub the rule would throw away, and the answer would jump from the true entry
  // to the box boundary. This ray starts 1e-11 outside the arm's y = 1 face and 2e-11 before its
  // x = -8.75 box boundary, so both crossings sit inside the tolerance.
  const double grazing[3] = {-8.75 - 2.e-11, 1. + 1.e-11, 0.5};
  const double slant = 1. / std::sqrt(2.);
  const double slantDir[3] = {slant, -slant, 0.};
  const double entered = solid.DistFromOutside(grazing, slantDir, 3, TGeoShape::Big(), nullptr);
  BOOST_CHECK_EQUAL(entered, solid.DistFromOutside_Loop(grazing, slantDir, TGeoShape::Big()));
  // and it really is in the regime where a per-box rule would differ: below the tolerance, and
  // strictly nearer than the x = -8.75 box boundary at 2e-11 * sqrt(2)
  BOOST_CHECK_GT(entered, 0.);
  BOOST_CHECK_LT(entered, TGeoShape::Tolerance());
  BOOST_CHECK_LT(entered, 2.e-11 * std::sqrt(2.));
}

BOOST_AUTO_TEST_CASE(the_accelerated_distances_track_their_twins_when_a_plane_is_rescaled)
{
  // Bit identity between an accelerated query and its twin is a self-check discipline, not a
  // physics requirement: a one-ulp difference in an exit distance is navigationally irrelevant.
  // It is achievable only under the plane convention of design section 3.1, where a unit normal n
  // is stored as 2b = n. There the slab bound (v - o_k) / d_k and the root -0.5*gamma/beta divide
  // numerator and denominator each scaled by exactly one half, so the single IEEE division returns
  // the same double for both, and a box face lying on a plane halfspace is crossed at one value.
  //
  // Rescaling every plane by a NON-POWER-OF-TWO -- 3 here, which is what an unnormalised carrier
  // normal (3, 0, 0) would give -- describes exactly the same solid, but fl(1.5 * d_x) rounds, the
  // root moves off the slab bound by an ulp, and the last bit is lost. Nothing else in the system
  // notices, so this test pins the size of what is lost: the answers must still agree closely.
  O2FlatCSG solid("bracket_scaled");
  buildBracket(solid, 3.);
  solid.CloseShape();
  BOOST_REQUIRE(solid.IsClosed());

  Rng rng(1357911ULL);
  double worst = 0.;
  for (int trial = 0; trial < 200000; ++trial) {
    double point[3] = {rng.uniform(-16., 16.), rng.uniform(-8., 8.), rng.uniform(-6., 17.)};
    double dir[3];
    double norm = 0.;
    do {
      for (int index = 0; index < 3; ++index) {
        dir[index] = rng.uniform(-1., 1.);
      }
      norm = std::sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
    } while (norm < 1.e-3);
    for (int index = 0; index < 3; ++index) {
      dir[index] /= norm;
    }
    const bool inside = solid.Contains_Loop(point);
    // Contains has no arithmetic of its own to lose, so it stays bit-identical under any scale
    BOOST_REQUIRE_EQUAL(solid.Contains(point), inside);
    const double accelerated =
      inside ? solid.DistFromInside(point, dir, 3, TGeoShape::Big(), nullptr)
             : solid.DistFromOutside(point, dir, 3, TGeoShape::Big(), nullptr);
    const double twin = inside ? solid.DistFromInside_Loop(point, dir, TGeoShape::Big())
                               : solid.DistFromOutside_Loop(point, dir, TGeoShape::Big());
    const double slack = std::abs(accelerated - twin);
    worst = std::max(worst, slack);
    // The bound is set just above what the rescale actually costs -- the run below measures
    // 1.24e-14 -- so the assertion, and not only the message under it, is what pins the size of
    // the loss. A looser bound would pass on a rescale that had broken something far larger.
    BOOST_REQUIRE_LE(slack, 1.e-13 * std::max(1., std::abs(twin)));
  }
  BOOST_TEST_MESSAGE("largest accelerated-vs-twin gap under a x3 plane rescale: " << worst);
  // The aggregate is deliberately looser than the relative assertion above: it is an absolute
  // bound on a maximum over a sample, and FMA contraction or a different libm moves the last
  // couple of ulps. 1e-12 still pins the size of the loss a thousand times tighter than the
  // 1e-9 this test used to assert, without being a cross-platform tripwire.
  BOOST_CHECK_LE(worst, 1.e-12);
}

BOOST_AUTO_TEST_CASE(a_shape_that_failed_to_close_still_answers_through_the_loop_twins)
{
  // CloseShape refuses an unset cell bbox and builds nothing, so there is no box array and no BVH.
  // An accelerated query that walked the empty array would answer "no material anywhere" -- the
  // silent vanishing the refusal exists to prevent -- so all three must fall back to the twins.
  O2FlatCSG solid("bracket_unclosed");
  buildBracket(solid);
  // a fourth cell, a box at x in [12, 14], deliberately left without a bounding box
  const double extra[6][2][3] = {{{1., 0., 0.}, {14., 0., 0.}},
                                 {{-1., 0., 0.}, {12., 0., 0.}},
                                 {{0., 1., 0.}, {0., 1., 0.}},
                                 {{0., -1., 0.}, {0., -1., 0.}},
                                 {{0., 0., 1.}, {0., 0., 1.}},
                                 {{0., 0., -1.}, {0., 0., -1.}}};
  double coeff[10];
  const int first = solid.GetNhalfspaces();
  for (const auto& plane : extra) {
    planeQuadric(plane[0], plane[1], coeff);
    solid.AddQuadric(1., coeff);
  }
  solid.AddCell(first, 6, 2. * 2. * 2.);

  solid.CloseShape();
  BOOST_REQUIRE(!solid.IsClosed());
  BOOST_CHECK_EQUAL(solid.GetNboxes(), 0);
  BOOST_CHECK_EQUAL(solid.GetBVHMemory(), 0u);

  // material inside every one of the four cells is still found, and empty space is still empty
  const double inArm[3] = {0., 0., 0.};
  const double inUpright[3] = {9., 0., 6.};
  const double inWasher[3] = {-10.5, 0., 2.};
  const double inExtra[3] = {13., 0., 0.};
  // the washer's bore, which is empty space now that the washer sits above the arm
  const double inBore[3] = {-8., 0., 2.};
  const double outside[3] = {0., 0., 20.};
  BOOST_CHECK(solid.Contains(inArm));
  BOOST_CHECK(solid.Contains(inUpright));
  BOOST_CHECK(solid.Contains(inWasher));
  BOOST_CHECK(solid.Contains(inExtra));
  BOOST_CHECK(!solid.Contains(inBore));
  BOOST_CHECK(!solid.Contains(outside));

  Rng rng(24680ULL);
  for (int trial = 0; trial < 20000; ++trial) {
    const double point[3] = {rng.uniform(-16., 16.), rng.uniform(-5., 5.), rng.uniform(-3., 14.)};
    BOOST_REQUIRE_EQUAL(solid.Contains(point), solid.Contains_Loop(point));
    double dir[3];
    double norm = 0.;
    do {
      for (int index = 0; index < 3; ++index) {
        dir[index] = rng.uniform(-1., 1.);
      }
      norm = std::sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
    } while (norm < 1.e-3);
    for (int index = 0; index < 3; ++index) {
      dir[index] /= norm;
    }
    const bool inside = solid.Contains_Loop(point);
    if (inside) {
      BOOST_REQUIRE_EQUAL(solid.DistFromInside(point, dir, 3, TGeoShape::Big(), nullptr),
                          solid.DistFromInside_Loop(point, dir, TGeoShape::Big()));
    } else {
      BOOST_REQUIRE_EQUAL(solid.DistFromOutside(point, dir, 3, TGeoShape::Big(), nullptr),
                          solid.DistFromOutside_Loop(point, dir, TGeoShape::Big()));
    }
    // Safety falls back to its twin exactly like the other three accelerated queries -- this was
    // asserted for Contains and the distances above but never extended to Safety
    BOOST_REQUIRE_EQUAL(solid.Safety(point, inside), solid.Safety_Loop(point, inside));
  }
}

BOOST_AUTO_TEST_CASE(safety_is_sound_and_matches_its_twin)
{
  O2FlatCSG solid("bracket_safety");
  buildBracket(solid);
  solid.CloseShape();

  Rng rng(24680ULL);
  for (int trial = 0; trial < 50000; ++trial) {
    double point[3] = {rng.uniform(-16., 16.), rng.uniform(-8., 8.), rng.uniform(-6., 17.)};
    const bool inside = solid.Contains_Loop(point);
    const double safety = solid.Safety(point, inside);
    BOOST_REQUIRE_GE(safety, 0.);
    BOOST_REQUIRE_EQUAL(safety, solid.Safety_Loop(point, inside));

    // SOUNDNESS: no point within `safety` of `point` may have the opposite classification
    for (int probe = 0; probe < 40; ++probe) {
      double dir[3];
      double norm = 0.;
      do {
        for (int index = 0; index < 3; ++index) {
          dir[index] = rng.uniform(-1., 1.);
        }
        norm = std::sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
      } while (norm < 1.e-3);
      const double reach = safety * rng.uniform(0., 0.999) / norm;
      const double near[3] = {point[0] + reach * dir[0], point[1] + reach * dir[1],
                              point[2] + reach * dir[2]};
      BOOST_REQUIRE_EQUAL(static_cast<bool>(solid.Contains_Loop(near)), inside);
    }
  }
}

BOOST_AUTO_TEST_CASE(safety_is_sound_when_the_inside_bound_is_actually_nonzero)
{
  // At the class's default split depth, buildBracket's arm is so far from cubic (20 x 2 x 2) that
  // the depth cap fires before any leaf fully detaches from all six faces: EVERY box keeps
  // nActive != 0, so the inside branch's `nActive == 0` selection path -- the one piece of
  // `Safety` whose soundness rests on a structural invariant (design section 4.2's hard guarantee)
  // rather than an exact box-distance formula -- is never taken by the test above. Its probes are
  // then vacuous: with `safety == 0.`, `reach` is always `0.` too, so the "nearby" point IS the
  // query point and the soundness check is trivially true. A deeper split makes solid boxes exist
  // (see the fix-round measurement in the task report), which this case forces so the nActive == 0
  // path is genuinely exercised end to end, not just agreed upon by two implementations at zero.
  O2FlatCSG solid("bracket_safety_deep");
  buildBracket(solid);
  solid.SetSplitDepth(14);
  // The size floor has to come down with the depth cap, or it stops the split first: at the
  // shipped 0.05 the arm is thinner than one minimum box and no leaf ever detaches.
  solid.SetMinBoxFraction(0.002);
  solid.CloseShape();

  bool sawPositiveInsideSafety = false;
  Rng rng(11235813ULL);
  for (int trial = 0; trial < 50000; ++trial) {
    double point[3] = {rng.uniform(-16., 16.), rng.uniform(-8., 8.), rng.uniform(-6., 17.)};
    const bool inside = solid.Contains_Loop(point);
    const double safety = solid.Safety(point, inside);
    BOOST_REQUIRE_GE(safety, 0.);
    BOOST_REQUIRE_EQUAL(safety, solid.Safety_Loop(point, inside));
    if (inside && safety > 0.) {
      sawPositiveInsideSafety = true;
    }

    // the same soundness probes as above, now with genuine reach on at least some trials
    for (int probe = 0; probe < 40; ++probe) {
      double dir[3];
      double norm = 0.;
      do {
        for (int index = 0; index < 3; ++index) {
          dir[index] = rng.uniform(-1., 1.);
        }
        norm = std::sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
      } while (norm < 1.e-3);
      const double reach = safety * rng.uniform(0., 0.999) / norm;
      const double near[3] = {point[0] + reach * dir[0], point[1] + reach * dir[1],
                              point[2] + reach * dir[2]};
      BOOST_REQUIRE_EQUAL(static_cast<bool>(solid.Contains_Loop(near)), inside);
    }
  }
  BOOST_REQUIRE(sawPositiveInsideSafety);
}

BOOST_AUTO_TEST_CASE(capacity_is_the_sum_of_the_cell_volumes)
{
  O2FlatCSG solid("bracket_capacity");
  buildBracket(solid);
  solid.CloseShape();
  const double expected = 8. * 10. * 1. * 1. + 2. * 2. * 11. + TMath::Pi() * (9. - 1.) * 2.;
  BOOST_CHECK_SMALL(solid.Capacity() - expected, 1.e-12);
}

BOOST_AUTO_TEST_CASE(the_bounding_box_is_tight_around_the_retained_boxes)
{
  // ComputeBBox is the union of the RETAINED boxes -- a subset of the union of the cell AABBs --
  // so exact GetDX/GetDY/GetDZ/GetOrigin values depend on subdivision details rather than on the
  // contract. Assert the two legs that matter for navigation correctness instead: the bounding
  // box holds the whole solid, and it does not overshoot past what the cells could possibly reach.
  O2FlatCSG solid("bracket_bbox");
  buildBracket(solid);
  solid.CloseShape();

  // every point of the solid is inside the bounding box
  Rng rng(13579ULL);
  for (int trial = 0; trial < 50000; ++trial) {
    const double point[3] = {rng.uniform(-13., 13.), rng.uniform(-5., 5.), rng.uniform(-3., 14.)};
    if (solid.Contains_Loop(point)) {
      BOOST_REQUIRE(solid.TGeoBBox::Contains(point));
    }
  }

  // the bounding box is contained in the union of the cell AABBs, which buildBracket fixes:
  // arm x in [-10, 10], y in [-1, 1], z in [-1, 1]; upright x in [8, 10], y in [-1, 1], z in
  // [1, 12]; washer x in [-11, -5], y in [-3, 3], z in [1, 3]
  const double cellLo[3][3] = {{-10., -1., -1.}, {8., -1., 1.}, {-11., -3., 1.}};
  const double cellHi[3][3] = {{10., 1., 1.}, {10., 1., 12.}, {-5., 3., 3.}};
  double unionLo[3] = {cellLo[0][0], cellLo[0][1], cellLo[0][2]};
  double unionHi[3] = {cellHi[0][0], cellHi[0][1], cellHi[0][2]};
  for (int cell = 1; cell < 3; ++cell) {
    for (int index = 0; index < 3; ++index) {
      unionLo[index] = std::min(unionLo[index], cellLo[cell][index]);
      unionHi[index] = std::max(unionHi[index], cellHi[cell][index]);
    }
  }
  const double* origin = solid.GetOrigin();
  for (int index = 0; index < 3; ++index) {
    const double dHalf = index == 0 ? solid.GetDX() : (index == 1 ? solid.GetDY() : solid.GetDZ());
    BOOST_CHECK_GE(origin[index] - dHalf, unionLo[index] - 1.e-9);
    BOOST_CHECK_LE(origin[index] + dHalf, unionHi[index] + 1.e-9);
  }
}

BOOST_AUTO_TEST_CASE(the_normal_on_a_face_is_the_face_normal)
{
  O2FlatCSG solid("box_normal");
  addBoxCell(solid, 3., 4., 5.);
  const double lo[3] = {-3., -4., -5.};
  const double hi[3] = {3., 4., 5.};
  solid.SetCellBBox(0, lo, hi);
  solid.CloseShape();

  const double onFace[3] = {3., 1., 1.};
  const double dir[3] = {1., 0., 0.};
  double normal[3] = {0., 0., 0.};
  solid.ComputeNormal(onFace, dir, normal);
  BOOST_CHECK_SMALL(normal[0] - 1., 1.e-12);
  BOOST_CHECK_SMALL(normal[1], 1.e-12);
  BOOST_CHECK_SMALL(normal[2], 1.e-12);
}

BOOST_AUTO_TEST_CASE(the_normal_selection_is_scale_invariant_across_cells)
{
  // Fix round 1: |EvalHalfspace| alone is not a distance -- its gain per unit distance is 1 for a
  // unit plane but ~2R for a cylinder of radius R, so a naive argmin over |f| can pick a distant
  // plane over the surface the point is actually on. Two cells make the point concrete: cell 0 is
  // a cylinder of radius 100 about z, cell 1 a single plane at z = 0.05. The test point sits
  // 0.0005 from the cylinder wall (radially) and 0.05 from the plane -- the cylinder is the true
  // nearest surface by two orders of magnitude, but |f_cylinder| ~= 0.1 > |f_plane| = 0.05, so the
  // unscaled rule would have picked the plane and returned (0, 0, 1) instead of the correct
  // (1, 0, 0). Deliberately not closed: with CloseShape run, the box-restriction half of the fix
  // alone would make this pass trivially (the point's own box never sees the other cell's plane),
  // so this exercises ComputeNormal's cross-cell fallback scan, where only the |f| / |grad f|
  // fix -- not the box restriction -- can be what saves it.
  O2FlatCSG solid("scale_invariance");
  double coeff[10];
  zCylinderQuadric(100., coeff);
  solid.AddQuadric(1., coeff);
  solid.AddCell(0, 1, 0.);

  const double planeNormal[3] = {0., 0., 1.};
  const double planeThrough[3] = {0., 0., 0.05};
  planeQuadric(planeNormal, planeThrough, coeff);
  const int planeFirst = solid.GetNhalfspaces();
  solid.AddQuadric(1., coeff);
  solid.AddCell(planeFirst, 1, 0.);

  BOOST_REQUIRE(!solid.IsClosed());

  const double point[3] = {99.9995, 0., 0.};
  const double dir[3] = {1., 0., 0.};
  double normal[3] = {0., 0., 0.};
  solid.ComputeNormal(point, dir, normal);
  BOOST_CHECK_SMALL(normal[0] - 1., 1.e-9);
  BOOST_CHECK_SMALL(normal[1], 1.e-9);
  BOOST_CHECK_SMALL(normal[2], 1.e-9);
}

BOOST_AUTO_TEST_CASE(a_zero_extent_cell_bbox_does_not_burn_the_whole_cubify_budget)
{
  // Fix round 2: a cell bbox with a genuinely zero extent on one axis passes CloseShape's
  // validation (it rejects only unset, inverted or non-finite boxes, not degenerate-but-flat
  // ones). Without SplitBox's `shortest` floor at `minSize`, that axis's extent stays pinned at
  // zero forever (it is never the longest, so never split), making `longest > 2 * shortest`
  // permanently true and spending the ENTIRE per-path cubify ceiling on a cell a depth-only rule
  // would have resolved in a handful of splits -- roughly `2^kMaxCubifySplits` leaves along every
  // branch instead. A 100 x 100 x 0 slab, subdivided down to the default minSize floor, needs on
  // the order of a dozen splits total once x and y are treated as the only axes that matter; this
  // asserts the box count stays in that regime rather than climbing towards the ceiling.
  O2FlatCSG solid("flat_cell");
  double coeff[10];
  const double planes[6][2][3] = {
    {{1., 0., 0.}, {50., 0., 0.}}, {{-1., 0., 0.}, {-50., 0., 0.}}, {{0., 1., 0.}, {0., 50., 0.}}, {{0., -1., 0.}, {0., -50., 0.}}, {{0., 0., 1.}, {0., 0., 0.}}, {{0., 0., -1.}, {0., 0., 0.}}};
  for (const auto& plane : planes) {
    planeQuadric(plane[0], plane[1], coeff);
    solid.AddQuadric(1., coeff);
  }
  solid.AddCell(0, 6, 0.);
  const double lo[3] = {-50., -50., 0.};
  const double hi[3] = {50., 50., 0.};
  solid.SetCellBBox(0, lo, hi);
  solid.CloseShape();

  BOOST_REQUIRE(solid.IsClosed());
  // measured 272 boxes with the guard in place; a generous margin above that, and two orders of
  // magnitude below what hitting the per-path ceiling on every branch would produce
  BOOST_CHECK_LT(solid.GetNboxes(), 600);
}

BOOST_AUTO_TEST_CASE(a_sidecar_round_trip_reproduces_the_solid)
{
  O2FlatCSG original("bracket_io");
  buildBracket(original);
  original.CloseShape();

  const std::string path = "testFlatCSG_roundtrip.bin";
  BOOST_REQUIRE(o2::cad::WriteFlatCSG(path, original)); // test-only writer

  O2FlatCSG loaded("bracket_io_loaded");
  BOOST_REQUIRE(o2::cad::LoadFlatCSG(path, loaded));
  loaded.CloseShape();

  BOOST_CHECK_EQUAL(loaded.GetNhalfspaces(), original.GetNhalfspaces());
  BOOST_CHECK_EQUAL(loaded.GetNcells(), original.GetNcells());
  BOOST_CHECK_EQUAL(loaded.GetNboxes(), original.GetNboxes());
  BOOST_CHECK_EQUAL(loaded.Capacity(), original.Capacity());

  Rng rng(97531ULL);
  for (int trial = 0; trial < 100000; ++trial) {
    const double point[3] = {rng.uniform(-13., 13.), rng.uniform(-5., 5.), rng.uniform(-3., 14.)};
    BOOST_REQUIRE_EQUAL(loaded.Contains(point), original.Contains(point));
  }
  std::filesystem::remove(path);
}

BOOST_AUTO_TEST_CASE(writing_an_unclosed_shape_is_refused)
{
  // GetCellBBox reads back zeros for a cell whose box was never set -- a finite, non-inverted box
  // that would otherwise pass CloseShape's own validation on reload, silently shipping a
  // degenerate point-box for that cell. WriteFlatCSG refuses before that invariant can ever reach
  // a file: no CloseShape() call at all, and a cell missing a bbox (CloseShape() refused).
  const std::string path = "testFlatCSG_unclosed.bin";

  O2FlatCSG neverClosed("bracket_never_closed");
  buildBracket(neverClosed);
  BOOST_REQUIRE(!neverClosed.IsClosed());
  BOOST_CHECK(!o2::cad::WriteFlatCSG(path, neverClosed));
  BOOST_CHECK(!std::filesystem::exists(path));

  O2FlatCSG refused("bracket_refused_close");
  double coeff[10];
  const double plane[2][3] = {{1., 0., 0.}, {0., 0., 0.}};
  planeQuadric(plane[0], plane[1], coeff);
  refused.AddQuadric(1., coeff);
  refused.AddCell(0, 1, 0.); // no SetCellBBox for this cell -- CloseShape must refuse
  refused.CloseShape();
  BOOST_REQUIRE(!refused.IsClosed());
  BOOST_CHECK(!o2::cad::WriteFlatCSG(path, refused));
  BOOST_CHECK(!std::filesystem::exists(path));
}

BOOST_AUTO_TEST_CASE(a_truncated_sidecar_is_refused_rather_than_half_loaded)
{
  O2FlatCSG original("bracket_trunc");
  buildBracket(original);
  original.CloseShape();
  const std::string path = "testFlatCSG_truncated.bin";
  BOOST_REQUIRE(o2::cad::WriteFlatCSG(path, original));
  std::filesystem::resize_file(path, std::filesystem::file_size(path) - 17);

  O2FlatCSG loaded("bracket_trunc_loaded");
  BOOST_CHECK(!o2::cad::LoadFlatCSG(path, loaded));
  std::filesystem::remove(path);
}

BOOST_AUTO_TEST_CASE(the_shape_survives_a_ROOT_file_without_its_sidecar)
{
  O2FlatCSG original("bracket_root");
  buildBracket(original);
  original.CloseShape();

  const std::string path = "testFlatCSG_shape.root";
  {
    TFile file(path.c_str(), "RECREATE");
    file.WriteObject(&original, "shape");
  }
  O2FlatCSG* restored = nullptr;
  {
    TFile file(path.c_str(), "READ");
    file.GetObject("shape", restored);
  }
  BOOST_REQUIRE(restored != nullptr);
  restored->CloseShape(); // the BVH is not streamed; it is rebuilt

  Rng rng(11223ULL);
  for (int trial = 0; trial < 100000; ++trial) {
    const double point[3] = {rng.uniform(-13., 13.), rng.uniform(-5., 5.), rng.uniform(-3., 14.)};
    BOOST_REQUIRE_EQUAL(restored->Contains(point), original.Contains(point));
  }
  std::filesystem::remove(path);
}

namespace
{
/// One axis-aligned box as its own cell, with its bbox.
void addBoxAsCell(O2FlatCSG& solid, const double* lo, const double* hi)
{
  const int first = solid.GetNhalfspaces();
  double coeff[10];
  for (int axis = 0; axis < 3; ++axis) {
    for (int sense = -1; sense <= 1; sense += 2) {
      double normal[3] = {0., 0., 0.};
      double through[3] = {0., 0., 0.};
      normal[axis] = static_cast<double>(sense);
      through[axis] = sense > 0 ? hi[axis] : lo[axis];
      planeQuadric(normal, through, coeff);
      solid.AddQuadric(1., coeff);
    }
  }
  const int cell = solid.AddCell(first, 6, (hi[0] - lo[0]) * (hi[1] - lo[1]) * (hi[2] - lo[2]));
  solid.SetCellBBox(cell, lo, hi);
}

/// Three touching cells along x. The two tall ones sit together in the BVH, so the far one is
/// tested while the running bound is still the near one's exit, and only the short cell between
/// them then extends the union past it.
void buildStaggeredChain(O2FlatCSG& solid)
{
  const double nearLo[3] = {0., -1., -1.};
  const double nearHi[3] = {1., 20., 1.};
  const double farLo[3] = {5., -1., -1.};
  const double farHi[3] = {6., 20., 1.};
  const double middleLo[3] = {1., -1., -1.};
  const double middleHi[3] = {5., 1., 1.};
  addBoxAsCell(solid, nearLo, nearHi);
  addBoxAsCell(solid, farLo, farHi);
  addBoxAsCell(solid, middleLo, middleHi);
  solid.CloseShape();
}
} // namespace

BOOST_AUTO_TEST_CASE(a_far_box_that_extends_the_union_is_recovered_by_the_unpruned_retry)
{
  O2FlatCSG solid("staggered");
  buildStaggeredChain(solid);

  // along the chain: the answer is the far cell's exit at x = 6, which the pruned traversal can
  // only reach through the middle cell it sees last
  const double point[3] = {0.5, 0., 0.};
  const double dir[3] = {1., 0., 0.};
  O2FlatCSG::ResetUnprunedRetryCounter();
  const double distance = solid.DistFromInside(point, dir, 3, TGeoShape::Big(), nullptr);
  BOOST_CHECK_EQUAL(distance, solid.DistFromInside_Loop(point, dir, TGeoShape::Big()));
  BOOST_CHECK_CLOSE(distance, 5.5, 1.e-9);
  BOOST_CHECK_GT(O2FlatCSG::GetUnprunedRetryCount(), 0);
}

BOOST_AUTO_TEST_CASE(the_pruned_DistFromInside_is_bit_identical_to_its_twin_on_the_staggered_chain)
{
  O2FlatCSG solid("staggered_random");
  buildStaggeredChain(solid);

  Rng rng(97531ULL);
  int inside = 0;
  for (int trial = 0; trial < 100000; ++trial) {
    double point[3] = {rng.uniform(-1., 7.), rng.uniform(-2., 21.), rng.uniform(-2., 2.)};
    if (!solid.Contains_Loop(point)) {
      continue;
    }
    double dir[3];
    double norm = 0.;
    do {
      for (int index = 0; index < 3; ++index) {
        dir[index] = rng.uniform(-1., 1.);
      }
      norm = std::sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
    } while (norm < 1.e-3);
    for (int index = 0; index < 3; ++index) {
      dir[index] /= norm;
    }
    ++inside;
    BOOST_REQUIRE_EQUAL(solid.DistFromInside(point, dir, 3, TGeoShape::Big(), nullptr),
                        solid.DistFromInside_Loop(point, dir, TGeoShape::Big()));
    // a finite step must answer as the twin does with the same step
    BOOST_REQUIRE_EQUAL(solid.DistFromInside(point, dir, 3, 2., nullptr),
                        solid.DistFromInside_Loop(point, dir, 2.));
  }
  BOOST_CHECK_GT(inside, 1000);
}
