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
/// \since 2026-09

#define BOOST_TEST_MODULE Test O2Tessellated class
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include "DetectorsBase/O2Tessellated.h"

#include "TGeoShape.h"

#include <cmath>
#include <limits>
#include <vector>

namespace
{
using o2::base::O2Tessellated;
using Vertex_t = O2Tessellated::Vertex_t;

/// A small deterministic generator, so a failing ray is reproducible from its seed alone.
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

/// Add the twelve outward-wound triangles of an axis-aligned box.
void addBox(O2Tessellated& shape, double cx, double cy, double cz, double hx, double hy, double hz)
{
  const double x0 = cx - hx, x1 = cx + hx;
  const double y0 = cy - hy, y1 = cy + hy;
  const double z0 = cz - hz, z1 = cz + hz;
  const Vertex_t corner[8] = {{x0, y0, z0}, {x1, y0, z0}, {x1, y1, z0}, {x0, y1, z0}, {x0, y0, z1}, {x1, y0, z1}, {x1, y1, z1}, {x0, y1, z1}};
  // each quad is wound counter-clockwise seen from outside, so the facet normal points outward
  const int quad[6][4] = {{0, 3, 2, 1}, {4, 5, 6, 7}, {0, 1, 5, 4}, {2, 3, 7, 6}, {1, 2, 6, 5}, {0, 4, 7, 3}};
  for (const auto& face : quad) {
    shape.AddFacet(corner[face[0]], corner[face[1]], corner[face[2]]);
    shape.AddFacet(corner[face[0]], corner[face[2]], corner[face[3]]);
  }
}

/// The Moeller-Trumbore distance used by O2Tessellated's leaf test, repeated here as the oracle.
double rayTriangleReference(const double* origin, const double* dir, const Vertex_t& v0, const Vertex_t& v1,
                            const Vertex_t& v2)
{
  constexpr double EPS = 1.e-8;
  const double infinity = std::numeric_limits<double>::infinity();
  const double e1[3] = {v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]};
  const double e2[3] = {v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]};
  const double p[3] = {dir[1] * e2[2] - dir[2] * e2[1], dir[2] * e2[0] - dir[0] * e2[2],
                       dir[0] * e2[1] - dir[1] * e2[0]};
  const double det = e1[0] * p[0] + e1[1] * p[1] + e1[2] * p[2];
  if (std::abs(det) <= EPS) {
    return infinity;
  }
  const double tvec[3] = {origin[0] - v0[0], origin[1] - v0[1], origin[2] - v0[2]};
  const double invDet = 1.0 / det;
  const double u = (tvec[0] * p[0] + tvec[1] * p[1] + tvec[2] * p[2]) * invDet;
  if (u < 0.0 || u > 1.0) {
    return infinity;
  }
  const double q[3] = {tvec[1] * e1[2] - tvec[2] * e1[1], tvec[2] * e1[0] - tvec[0] * e1[2],
                       tvec[0] * e1[1] - tvec[1] * e1[0]};
  const double v = (dir[0] * q[0] + dir[1] * q[1] + dir[2] * q[2]) * invDet;
  if (v < 0.0 || u + v > 1.0) {
    return infinity;
  }
  const double t = e2[0] * q[0] + e2[1] * q[1] + e2[2] * q[2];
  return (t * invDet > 0.) ? t * invDet : infinity;
}

/// The unpruned answer: the nearest facet over every facet of the mesh, entering or exiting.
double bruteForce(const O2Tessellated& shape, const double* origin, const double* dir, bool entering)
{
  double best = TGeoShape::Big();
  for (int facet = 0; facet < shape.GetNfacets(); ++facet) {
    const auto& description = shape.GetFacet(facet);
    const Vertex_t& v0 = shape.GetVertex(description[0]);
    const Vertex_t& v1 = shape.GetVertex(description[1]);
    const Vertex_t& v2 = shape.GetVertex(description[2]);
    const double e1[3] = {v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]};
    const double e2[3] = {v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]};
    const double normal[3] = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2],
                              e1[0] * e2[1] - e1[1] * e2[0]};
    const double along = normal[0] * dir[0] + normal[1] * dir[1] + normal[2] * dir[2];
    // the same facing filter the shape applies: entering facets face the ray, exiting ones face away
    if (entering ? (along > 0.) : (along <= 0.)) {
      continue;
    }
    best = std::min(best, rayTriangleReference(origin, dir, v0, v1, v2));
  }
  return best;
}

/// Eight boxes in a row, so every axial ray meets sixteen facets and the BVH has many leaves.
void buildRow(O2Tessellated& shape)
{
  for (int index = 0; index < 8; ++index) {
    addBox(shape, -21. + 6. * index, 0., 0., 2., 3., 4.);
  }
  shape.CloseShape(true, false, false);
}
} // namespace

BOOST_AUTO_TEST_CASE(PrunedRayQueriesEqualTheBruteForceMinimum)
{
  O2Tessellated shape("row");
  buildRow(shape);
  BOOST_CHECK_EQUAL(shape.GetNfacets(), 96);

  Rng rng(20260912);
  int outsideHits = 0;
  int insideHits = 0;
  for (int trial = 0; trial < 4000; ++trial) {
    // origins inside the row and well outside it, so both directions are exercised
    const double origin[3] = {rng.uniform(-40., 40.), rng.uniform(-12., 12.), rng.uniform(-12., 12.)};
    double dir[3] = {rng.uniform(-1., 1.), rng.uniform(-1., 1.), rng.uniform(-1., 1.)};
    const double norm = std::sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
    if (norm < 1.e-6) {
      continue;
    }
    for (int index = 0; index < 3; ++index) {
      dir[index] /= norm;
    }

    const double outside = shape.DistFromOutside(origin, dir, 1, TGeoShape::Big(), nullptr);
    const double inside = shape.DistFromInside(origin, dir, 1, TGeoShape::Big(), nullptr);
    const double outsideReference = bruteForce(shape, origin, dir, true);
    const double insideReference = bruteForce(shape, origin, dir, false);

    BOOST_CHECK_EQUAL(outside, outsideReference);
    BOOST_CHECK_EQUAL(inside, insideReference);
    outsideHits += outsideReference < TGeoShape::Big() ? 1 : 0;
    insideHits += insideReference < TGeoShape::Big() ? 1 : 0;
  }
  // the case is only meaningful if the rays really hit the mesh; this sampling gives about 450
  // entering and 3000 exiting hits
  BOOST_CHECK_GT(outsideHits, 200);
  BOOST_CHECK_GT(insideHits, 200);
}

BOOST_AUTO_TEST_CASE(APrunedRayFindsTheNearestOfManyFacetsAlongIt)
{
  O2Tessellated shape("row");
  buildRow(shape);

  // straight down the row: eight boxes, so sixteen entering and sixteen exiting facets are in line
  const double origin[3] = {-40., 0., 0.};
  const double dir[3] = {1., 0., 0.};
  BOOST_CHECK_EQUAL(shape.DistFromOutside(origin, dir, 1, TGeoShape::Big(), nullptr), 17.);
  BOOST_CHECK_EQUAL(shape.DistFromOutside(origin, dir, 1, TGeoShape::Big(), nullptr),
                    bruteForce(shape, origin, dir, true));

  // from inside the first box, the exit is its own far face and not a later box's
  const double inner[3] = {-21., 0., 0.};
  BOOST_CHECK_EQUAL(shape.DistFromInside(inner, dir, 1, TGeoShape::Big(), nullptr), 2.);
  BOOST_CHECK_EQUAL(shape.DistFromInside(inner, dir, 1, TGeoShape::Big(), nullptr),
                    bruteForce(shape, inner, dir, false));
}
