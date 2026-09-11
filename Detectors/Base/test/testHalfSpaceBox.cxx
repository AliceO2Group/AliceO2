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

/// \file testHalfSpaceBox.cxx
/// \author Sandro Wenzel (CERN)
/// \brief Checks that TGeoGeometryUtils::makeHalfSpaceBox reproduces TGeoHalfSpace

#define BOOST_TEST_MODULE Test HalfSpaceBox
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include "DetectorsBase/TGeoGeometryUtils.h"
#include "TGeoManager.h"
#include "TGeoBBox.h"
#include "TGeoTube.h"
#include "TGeoMatrix.h"
#include "TGeoHalfSpace.h"
#include "TGeoCompositeShape.h"
#include "TMath.h"
#include "TRandom3.h"
#include "TString.h"
#include <cmath>
#include <vector>

namespace
{
struct Plane {
  const char* label;
  double p[3];
  double n[3];
};

// The fifteen half-space cuts of the TPC support structures (Detectors/TPC/simulation/src/Detector.cxx).
// Largest solid any of them is subtracted from is 1.65 x 1.85 x 8.9 cm, hence the 10 cm parent below.
std::vector<Plane> tpcPlanes()
{
  const double slope = TMath::Tan(22. * TMath::DegToRad());
  const double intp = 1.245;
  const double b = slope * slope + 1.;
  const double p1[3] = {intp * slope / b, -intp / b, 0.};
  const double p2[3] = {-intp * slope / b, -intp / b, 0.};
  return {
    {"sp1", {p1[0], p1[1], 0.}, {-p1[0], -p1[1], 0.}},
    {"sp2", {p2[0], p2[1], 0.}, {-p2[0], -p2[1], 0.}},
    {"cutil1", {0., 0.105, 0.}, {0., 1., 0.}},
    {"cutomh1", {0., -1.05, -3.4}, {0., -TMath::Tan(30. * TMath::DegToRad()), 1.}},
    {"cutomh2", {0., -1.05, 3.4}, {0., -TMath::Tan(30. * TMath::DegToRad()), -1.}},
    {"cutomh3", {-1.65, 0., -0.9}, {TMath::Tan(75. * TMath::DegToRad()), 0., 1.}},
    {"cutomh4", {-1.65, 0., 0.9}, {TMath::Tan(75. * TMath::DegToRad()), 0., -1.}},
    {"cutomh5", {1.65, -1.05, 0.}, {-1., -TMath::Tan(20. * TMath::DegToRad()), 0.}},
    {"cutohs1", {0., -0.186, 0.}, {0., -1., 0.}},
    {"cutmmh1", {-1.65, 0., -0.9}, {8., 0., 8. * TMath::Tan(13. * TMath::DegToRad())}},
    {"cutmmh2", {-1.65, 0., 0.9}, {8., 0., -8. * TMath::Tan(13. * TMath::DegToRad())}},
    {"cutmmh3", {0., 1.85, -2.8}, {0., -6.1, 6.1 * TMath::Tan(20. * TMath::DegToRad())}},
    {"cutmmh4", {0., 1.85, 2.8}, {0., -6.1, -6.1 * TMath::Tan(20. * TMath::DegToRad())}},
    {"cutmmh5", {0.75, 0., -8.9}, {2.4 * TMath::Tan(30. * TMath::DegToRad()), 0., 2.4}},
    {"cutmmh6", {0.75, 0., 8.9}, {2.4 * TMath::Tan(30. * TMath::DegToRad()), 0., -2.4}}};
}

// Compares "parent - halfspace" against "parent - box:box_tr" on random points and random rays.
// Points closer than kSurfaceBand to the plane are skipped: on the surface itself the two
// implementations may legitimately round to different sides.
void compare(const TString& tag, const double p[3], const double n[3], double parentHalfSize, double reach,
             TRandom3& rnd, int nPoints, int nRays, double& maxDistDiff)
{
  const double nl = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
  BOOST_REQUIRE(nl > 1e-6);
  constexpr double kSurfaceBand = 1e-9;

  new TGeoBBox(TString::Format("parent_%s", tag.Data()).Data(), parentHalfSize, parentHalfSize, parentHalfSize);
  new TGeoHalfSpace(TString::Format("hs_%s", tag.Data()).Data(), const_cast<double*>(p), const_cast<double*>(n));
  o2::base::TGeoGeometryUtils::makeHalfSpaceBox(TString::Format("bx_%s", tag.Data()).Data(), p, n, reach);

  auto* ref = new TGeoCompositeShape(TString::Format("ref_%s", tag.Data()),
                                     TString::Format("parent_%s-hs_%s", tag.Data(), tag.Data()));
  auto* box = new TGeoCompositeShape(TString::Format("new_%s", tag.Data()),
                                     TString::Format("parent_%s-(bx_%s:bx_%s_tr)", tag.Data(), tag.Data(), tag.Data()));

  const double range = 1.2 * parentHalfSize;
  for (int k = 0; k < nPoints; ++k) {
    double x[3];
    for (int i = 0; i < 3; ++i) {
      x[i] = rnd.Uniform(-range, range);
    }
    const double d = ((x[0] - p[0]) * n[0] + (x[1] - p[1]) * n[1] + (x[2] - p[2]) * n[2]) / nl;
    if (std::abs(d) < kSurfaceBand) {
      continue;
    }
    if (ref->Contains(x) != box->Contains(x)) {
      BOOST_REQUIRE_MESSAGE(false, "containment differs for " << tag.Data() << " at (" << x[0] << "," << x[1] << ","
                                                              << x[2] << "), distance to plane " << d);
    }
  }

  for (int k = 0; k < nRays; ++k) {
    double x[3], dir[3];
    for (int i = 0; i < 3; ++i) {
      x[i] = rnd.Uniform(-3. * range, 3. * range);
      dir[i] = rnd.Uniform(-1., 1.);
    }
    const double dn = std::sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
    if (dn < 1e-6) {
      continue;
    }
    for (int i = 0; i < 3; ++i) {
      dir[i] /= dn;
    }
    const bool inside = ref->Contains(x);
    if (inside != box->Contains(x)) {
      continue; // a point sitting on the surface; covered by the containment loop above
    }
    const double d1 = inside ? ref->DistFromInside(x, dir, 3) : ref->DistFromOutside(x, dir, 3);
    const double d2 = inside ? box->DistFromInside(x, dir, 3) : box->DistFromOutside(x, dir, 3);
    if (d1 > 1e15 && d2 > 1e15) {
      continue; // both miss
    }
    maxDistDiff = std::max(maxDistDiff, std::abs(d1 - d2));
  }
}
} // namespace

BOOST_AUTO_TEST_CASE(HalfSpaceBox_reproduces_TGeoHalfSpace)
{
  auto* geom = new TGeoManager("halfspacetest", "half-space replacement test");
  TRandom3 rnd(20240101);
  double maxDistDiff = 0.;

  // the real TPC cuts
  for (const auto& pl : tpcPlanes()) {
    compare(pl.label, pl.p, pl.n, 10., 100., rnd, 200000, 20000, maxDistDiff);
  }

  // and a spread of arbitrary planes, to pin the rotation for normals in every octant
  for (int i = 0; i < 200; ++i) {
    double p[3], n[3];
    for (int k = 0; k < 3; ++k) {
      p[k] = rnd.Uniform(-5., 5.);
      n[k] = rnd.Uniform(-1., 1.);
    }
    if (std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]) < 1e-3) {
      continue;
    }
    compare(TString::Format("rnd%d", i), p, n, 10., 100., rnd, 20000, 2000, maxDistDiff);
  }

  // the two shapes are not bit-identical, but they must agree to double round-off
  BOOST_CHECK_SMALL(maxDistDiff, 1e-9);
  BOOST_TEST_MESSAGE("maximum ray-distance difference: " << maxDistDiff);
  delete geom;
}

// The composite expressions of the TPC support structures are not all of the simple
// "parent - cut" shape: tpcihs6 subtracts a union and two placed tubes first. That shape is
// what makes a *trailing* "cut:matrix" term unsafe to write unparenthesised, so keep a case
// with the same structure.
BOOST_AUTO_TEST_CASE(HalfSpaceBox_in_a_compound_expression)
{
  auto* geom = new TGeoManager("halfspacetest2", "half-space replacement, compound expression");
  const double shift[3] = {0., -0.175, 0.};
  const double p[3] = {0., 0.105, 0.};
  const double n[3] = {0., 1., 0.};

  new TGeoBBox("tpcihs1", 4.7, 0.66, 2.35);
  new TGeoBBox("tpcihs2", 4.7, 0.485, 1.0, const_cast<double*>(shift));
  new TGeoBBox("tpcihs3", 1.5, 0.485, 2.35, const_cast<double*>(shift));
  new TGeoTube("tpcihs4", 0.0, 2.38, 0.1);
  auto* trans2 = new TGeoTranslation("trans2", 0.0, 2.84, 2.25);
  trans2->RegisterYourself();
  auto* trans3 = new TGeoTranslation("trans3", 0.0, 2.84, -2.25);
  trans3->RegisterYourself();
  new TGeoHalfSpace("cutil1", const_cast<double*>(p), const_cast<double*>(n));
  o2::base::TGeoGeometryUtils::makeHalfSpaceBox("bcutil1", p, n, 100.);

  auto* ref = new TGeoCompositeShape(
    "ref_tpcihs6", "tpcihs1-(tpcihs2+tpcihs3)-(tpcihs4:trans2)-(tpcihs4:trans3)-cutil1");
  auto* box = new TGeoCompositeShape(
    "new_tpcihs6", "tpcihs1-(tpcihs2+tpcihs3)-(tpcihs4:trans2)-(tpcihs4:trans3)-(bcutil1:bcutil1_tr)");

  TRandom3 rnd(20240102);
  long inRef = 0, inBox = 0;
  for (int k = 0; k < 2000000; ++k) {
    double x[3];
    for (int i = 0; i < 3; ++i) {
      x[i] = rnd.Uniform(-6., 6.);
    }
    if (std::abs(x[1] - p[1]) < 1e-9) {
      continue;
    }
    const bool a = ref->Contains(x);
    const bool b = box->Contains(x);
    inRef += a;
    inBox += b;
    if (a != b) {
      BOOST_REQUIRE_MESSAGE(false, "containment differs at (" << x[0] << "," << x[1] << "," << x[2] << ")");
    }
  }
  // guards against both shapes being empty, which would make the comparison vacuous
  BOOST_CHECK_GT(inRef, 0);
  BOOST_CHECK_EQUAL(inRef, inBox);
  delete geom;
}
