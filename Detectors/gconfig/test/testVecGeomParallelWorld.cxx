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

/// The VecGeom parallel world against the TGeoParallelWorld it stands for, on modules of chips with
/// sensors where some chips are misaligned so that they overlap a neighbour and stick out of their module.

#define BOOST_TEST_MODULE Test SimSetup VecGeomParallelWorld
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

// before the VecGeom headers, which define a VECGEOM macro
#include "DetectorsBase/GeometryManager.h"

#include "../src/VecGeomParallelWorld.h"

#include <VecGeom/management/GeoManager.h>
#include <VecGeom/volumes/LogicalVolume.h>
#include <VecGeom/volumes/PlacedVolume.h>

#include <TGeoBBox.h>
#include <TGeoManager.h>
#include <TGeoMaterial.h>
#include <TGeoMatrix.h>
#include <TGeoMedium.h>
#include <TGeoNavigator.h>
#include <TGeoParallelWorld.h>
#include <TGeoPhysicalNode.h>
#include <TGeoVolume.h>
#include <TString.h>

#include <cmath>
#include <random>
#include <string>
#include <vector>

using o2::simsetup::VecGeomParallelWorld;
using V3 = VecGeomParallelWorld::V3;

namespace
{
constexpr int kModules = 3;
constexpr int kChips = 4;

struct Fixture {
  std::vector<std::string> paths; ///< in the order of their priority
  TGeoParallelWorld* tgeo = nullptr;
  VecGeomParallelWorld vecgeom;

  static std::string chipPath(int module, int chip) { return Form("/TOP_1/MOD_%d/CHIP_%d", module, chip); }

  Fixture()
  {
    auto* geom = new TGeoManager("pwtest", "parallel world test");
    auto* medium = new TGeoMedium("vac", 1, new TGeoMaterial("vac", 0., 0., 0.));
    auto* top = geom->MakeBox("TOP", medium, 100., 100., 100.);
    geom->SetTopVolume(top);
    auto* module = geom->MakeBox("MOD", medium, 10., 5., 0.5);
    auto* chip = geom->MakeBox("CHIP", medium, 1., 1., 0.1);
    auto* sensor = geom->MakeBox("SENS", medium, 0.8, 0.8, 0.05);
    chip->AddNode(sensor, 1, new TGeoTranslation(0., 0., 0.03));
    for (int c = 0; c < kChips; ++c) {
      module->AddNode(chip, c, new TGeoTranslation(-6. + 4. * c, 0., 0.));
    }
    top->AddNode(module, 0, new TGeoTranslation(0., 0., -5.));
    top->AddNode(module, 1, new TGeoTranslation(0., 0., 5.));
    auto* rot = new TGeoRotation("r", 30., 20., 10.);
    top->AddNode(module, 2, new TGeoCombiTrans(30., 3., 0., rot));
    geom->CloseGeometry();

    // Misalign: a chip above its neighbour and out of the module in z, one into its neighbour, one out in y,
    // one tilted.
    auto* shift = new TGeoRotation("s", 3., 2., 1.);
    geom->MakePhysicalNode(chipPath(0, 1).c_str())->Align(new TGeoCombiTrans(1.5, 0.2, 0.45, shift));
    geom->MakePhysicalNode(chipPath(0, 3).c_str())->Align(new TGeoCombiTrans(3.2, -0.3, 0.12, shift)); // into chip 2
    geom->MakePhysicalNode(chipPath(1, 2).c_str())->Align(new TGeoTranslation(2., 4.5, 0.));
    geom->MakePhysicalNode(chipPath(2, 0).c_str())->Align(new TGeoCombiTrans(-6., 0., -0.42, shift));
    geom->MakePhysicalNode(chipPath(2, 3).c_str())->Align(new TGeoTranslation(4.5, 0.5, 0.3));
    geom->RefreshPhysicalNodes(false);

    tgeo = geom->CreateParallelWorld("priority");
    tgeo->SetAccelerationMode(TGeoParallelWorld::AccelerationMode::kBVH);
    for (int m = 0; m < kModules; ++m) {
      for (int c = 0; c < kChips; ++c) {
        // sensors before their chips, as ITS adds them
        paths.push_back(chipPath(m, c) + "/SENS_1");
        paths.push_back(chipPath(m, c));
      }
    }
    for (auto const& path : paths) {
      tgeo->AddNode(path.c_str());
    }
    geom->SetUseParallelWorldNav(true);
    BOOST_REQUIRE(tgeo->IsClosed());

    o2::base::GeometryManager::buildVecGeomGeometry(false);
    std::vector<vecgeom::NavigationState> states;
    for (auto const& path : paths) {
      TGeoPhysicalNode pnode(path.c_str());
      vecgeom::NavigationState state;
      state.Clear();
      auto const* placed = vecgeom::GeoManager::Instance().GetWorld();
      state.Push(placed);
      for (int l = 1; l <= pnode.GetLevel(); ++l) {
        const int index = pnode.GetNode(l - 1)->GetVolume()->GetIndex(pnode.GetNode(l));
        placed = placed->GetLogicalVolume()->GetDaughters()[index];
        state.Push(placed);
      }
      states.push_back(state);
    }
    vecgeom.build(std::move(states));
    geom->CdTop();
  }

  int indexOf(TGeoPhysicalNode const* pnode) const
  {
    if (pnode == nullptr) {
      return -1;
    }
    for (std::size_t i = 0; i < paths.size(); ++i) {
      if (paths[i] == pnode->GetName()) {
        return static_cast<int>(i);
      }
    }
    return -2;
  }
};

Fixture& fixture()
{
  static Fixture f;
  return f;
}

/// Points around the modules, where the nodes are.
V3 samplePoint(std::mt19937_64& gen)
{
  static const double centre[kModules][3] = {{0., 0., -5.}, {0., 0., 5.}, {30., 3., 0.}};
  std::uniform_real_distribution<double> u(-1., 1.);
  const auto& c = centre[gen() % kModules];
  return {c[0] + 14. * u(gen), c[1] + 9. * u(gen), c[2] + 4. * u(gen) * u(gen) * u(gen)};
}

V3 sampleDirection(std::mt19937_64& gen)
{
  std::normal_distribution<double> g;
  V3 d(g(gen), g(gen), g(gen));
  return d / d.Mag();
}

const auto noSkip = [](int) { return false; };
} // namespace

BOOST_AUTO_TEST_CASE(locate_agrees_with_tgeo)
{
  auto& f = fixture();
  BOOST_REQUIRE_EQUAL(f.vecgeom.size(), 2 * kModules * kChips);
  std::mt19937_64 gen(1);
  int inside = 0, different = 0, inTwoChips = 0, wrongOrder = 0;
  for (int i = 0; i < 400000; ++i) {
    const V3 p = samplePoint(gen);
    double point[3] = {p[0], p[1], p[2]};
    const int expected = f.indexOf(f.tgeo->FindNode(point));
    const int found = f.vecgeom.findNode(p, noSkip);
    inside += expected >= 0;
    different += found != expected;
    if (found >= 0) {
      // a second chip holding the point must come later; sensors have even indices, their chips follow
      const int chip = found | 1;
      const int other = f.vecgeom.findNode(p, [&](int k) { return k == chip || k == chip - 1; });
      if (other >= 0) {
        ++inTwoChips;
        wrongOrder += other < found;
      }
    }
  }
  BOOST_CHECK_GT(inside, 1000);
  BOOST_CHECK_EQUAL(different, 0);
  BOOST_CHECK_GT(inTwoChips, 10); // two misaligned chips do share points
  BOOST_CHECK_EQUAL(wrongOrder, 0);
}

BOOST_AUTO_TEST_CASE(skipped_node_gives_way_to_the_next)
{
  auto& f = fixture();
  // the centre of the sensor of an aligned chip: sensor first, its chip when the sensor is skipped
  const int sensor = 2 * (0 * kChips + 1);
  vecgeom::Transformation3D m;
  f.vecgeom.state(sensor).TopMatrix(m);
  const V3 p = m.InverseTransform(V3(0., 0., 0.));
  BOOST_CHECK_EQUAL(f.vecgeom.findNode(p, noSkip), sensor);
  BOOST_CHECK_EQUAL(f.vecgeom.findNode(p, [&](int k) { return k == sensor; }), sensor + 1);
}

BOOST_AUTO_TEST_CASE(states_know_their_nodes)
{
  auto& f = fixture();
  for (int chip = 1; chip < f.vecgeom.size(); chip += 2) {
    const int sensor = chip - 1;
    BOOST_CHECK(f.vecgeom.isInNode(chip, f.vecgeom.state(chip)));
    BOOST_CHECK(f.vecgeom.isInNode(chip, f.vecgeom.state(sensor)));
    BOOST_CHECK(!f.vecgeom.isInNode(sensor, f.vecgeom.state(chip)));
    BOOST_CHECK(f.vecgeom.isInAnyNode(f.vecgeom.state(chip)));
    BOOST_CHECK(f.vecgeom.isInAnyNode(f.vecgeom.state(sensor)));
    auto module = f.vecgeom.state(chip);
    module.Pop();
    BOOST_CHECK(!f.vecgeom.isInAnyNode(module));
    if (chip + 2 < f.vecgeom.size()) {
      BOOST_CHECK(!f.vecgeom.isInNode(chip, f.vecgeom.state(chip + 2)));
    }
  }
}

BOOST_AUTO_TEST_CASE(ray_entry_agrees_with_tgeo)
{
  auto& f = fixture();
  std::mt19937_64 gen(2);
  int hits = 0, differentNode = 0, differentStep = 0;
  for (int i = 0; i < 200000; ++i) {
    const V3 p = samplePoint(gen);
    if (f.vecgeom.findNode(p, noSkip) >= 0) {
      continue; // inside a node its surface is the main hierarchy's business
    }
    const V3 d = sampleDirection(gen);
    double point[3] = {p[0], p[1], p[2]};
    double dir[3] = {d[0], d[1], d[2]};
    double expectedStep = 1.e30;
    const int expected = f.indexOf(f.tgeo->FindNextBoundary(point, dir, expectedStep, 1.e30));
    double step = 1.e30;
    const int found = f.vecgeom.distanceToIn(p, d, step, noSkip);
    hits += expected >= 0;
    differentNode += found != expected;
    if (expected >= 0 && found == expected) {
      differentStep += std::abs(step - expectedStep) > 1.e-7;
    }
  }
  BOOST_CHECK_GT(hits, 1000);
  BOOST_CHECK_EQUAL(differentNode, 0);
  BOOST_CHECK_EQUAL(differentStep, 0);
}

BOOST_AUTO_TEST_CASE(ray_entry_honours_bound_and_skip)
{
  auto& f = fixture();
  std::mt19937_64 gen(3);
  int checked = 0;
  for (int i = 0; i < 50000; ++i) {
    const V3 p = samplePoint(gen);
    if (f.vecgeom.findNode(p, noSkip) >= 0) {
      continue;
    }
    const V3 d = sampleDirection(gen);
    double full = 1.e30;
    const int node = f.vecgeom.distanceToIn(p, d, full, noSkip);
    if (node < 0) {
      continue;
    }
    ++checked;
    double shorter = 0.5 * full; // a bound before the node: nothing is seen and the bound stays
    BOOST_CHECK_EQUAL(f.vecgeom.distanceToIn(p, d, shorter, noSkip), -1);
    BOOST_CHECK_EQUAL(shorter, 0.5 * full);
    double other = 1.e30; // the node skipped: another one or none, never closer
    const int next = f.vecgeom.distanceToIn(p, d, other, [&](int k) { return k == node; });
    BOOST_CHECK(next != node);
    BOOST_CHECK(next < 0 || other >= full);
  }
  BOOST_CHECK_GT(checked, 1000);
}

/// A safety is a lower bound of the distance to the nearest node, and the search stops at nodes whose
/// box is farther than the best value so far. Solids underestimate their own distance, so the value is not
/// unique: it lies between the smallest estimate over all nodes and the true distance. TGeo's solids
/// estimate differently, so its value is not compared.
BOOST_AUTO_TEST_CASE(safety_is_bracketed)
{
  auto& f = fixture();
  std::mt19937_64 gen(4);
  int checked = 0, tooSmall = 0, tooLarge = 0, limitBroken = 0;
  for (int i = 0; i < 50000; ++i) {
    const V3 p = samplePoint(gen);
    if (f.vecgeom.findNode(p, noSkip) >= 0) {
      continue;
    }
    ++checked;
    double smallestEstimate = 1.e30;
    for (int k = 0; k < f.vecgeom.size(); ++k) {
      vecgeom::Transformation3D m;
      f.vecgeom.state(k).TopMatrix(m);
      smallestEstimate = std::min(smallestEstimate, f.vecgeom.state(k).Top()->GetUnplacedVolume()->SafetyToIn(m.Transform(p)));
    }
    smallestEstimate = std::max(smallestEstimate, 0.);
    const double found = f.vecgeom.safetyToIn(p, 1.e30);
    tooSmall += found < smallestEstimate - 1.e-9;
    // no ray meets a node closer than the safety
    for (int r = 0; r < 20; ++r) {
      double step = 1.e30;
      if (f.vecgeom.distanceToIn(p, sampleDirection(gen), step, noSkip) >= 0 && step < found - 1.e-9) {
        ++tooLarge;
      }
    }
    // a limit is returned when no node is closer, and never exceeded
    for (double limit : {0.5 * smallestEstimate, 0.5 * found, 2. * found + 1.}) {
      const double bounded = f.vecgeom.safetyToIn(p, limit);
      limitBroken += bounded > limit || bounded < std::min(limit, smallestEstimate) - 1.e-9;
    }
  }
  BOOST_CHECK_GT(checked, 1000);
  BOOST_CHECK_EQUAL(tooSmall, 0);
  BOOST_CHECK_EQUAL(tooLarge, 0);
  BOOST_CHECK_EQUAL(limitBroken, 0);
}
