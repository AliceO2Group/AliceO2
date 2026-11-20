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

/// The relocating VecGeom navigator against TGeo on a geometry of coinciding faces: two copies of a tower
/// side by side that fill their mother, each a stack of tiles that reach the tower's faces, as in a sampling
/// calorimeter. A ray has to visit the same volumes in the same order, and never stop for a zero-length step
/// in a mother it only touches.

#define BOOST_TEST_MODULE Test SimSetup VecGeomG4Navigator
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

// before the VecGeom headers, which define a VECGEOM macro
#include "DetectorsBase/GeometryManager.h"

#include "../src/VecGeomG4Map.h"
#include "../src/VecGeomG4Navigator.h"

#include "TG4RootDetectorConstruction.h"

#include "G4StateManager.hh"
#include "G4SystemOfUnits.hh"
#include "G4TouchableHistory.hh"
#include "G4VExceptionHandler.hh"
#include "G4VPhysicalVolume.hh"

#include <TGeoManager.h>
#include <TGeoMaterial.h>
#include <TGeoMatrix.h>
#include <TGeoMedium.h>
#include <TGeoNode.h>
#include <TGeoVolume.h>
#include <TString.h>

#include <cmath>
#include <memory>
#include <random>
#include <string>
#include <vector>

using o2::simsetup::VecGeomG4Map;
using o2::simsetup::VecGeomG4Navigator;

namespace
{
struct Fixture {
  std::unique_ptr<TG4RootDetectorConstruction> construction;
  VecGeomG4Map map;
  std::unique_ptr<VecGeomG4Navigator> navigator;

  Fixture()
  {
    auto* geom = new TGeoManager("navtest", "coinciding faces");
    auto* medium = new TGeoMedium("alu", 1, new TGeoMaterial("alu", 26.98, 13., 2.7));
    auto* top = geom->MakeBox("TOP", medium, 100., 100., 100.);
    geom->SetTopVolume(top);
    // module > pair > two towers > paper tiles > lead tile; every box reaches the faces of the one around it
    // in x and y, and the towers fill the pair, the pair the module
    auto* module = geom->MakeBox("MODU", medium, 10., 5., 10.);
    auto* pair = geom->MakeBox("PAIR", medium, 10., 5., 10.);
    auto* tower = geom->MakeBox("TOWR", medium, 5., 5., 10.);
    auto* paper = geom->MakeBox("PAPR", medium, 5., 5., 0.5);
    auto* lead = geom->MakeBox("LEAD", medium, 5., 5., 0.3);
    paper->AddNode(lead, 1, new TGeoTranslation(0., 0., 0.));
    for (int i = 0; i < 8; ++i) {
      tower->AddNode(paper, i + 1, new TGeoTranslation(0., 0., -8.75 + 2.5 * i));
    }
    pair->AddNode(tower, 1, new TGeoTranslation(-5., 0., 0.));
    pair->AddNode(tower, 2, new TGeoTranslation(5., 0., 0.));
    module->AddNode(pair, 1, new TGeoTranslation(0., 0., 0.));
    // two modules that touch in y, and one apart
    top->AddNode(module, 1, new TGeoTranslation(0., -5., 0.));
    top->AddNode(module, 2, new TGeoTranslation(0., 5., 0.));
    top->AddNode(module, 3, new TGeoTranslation(40., 0., 0.));
    geom->CloseGeometry();

    construction = std::make_unique<TG4RootDetectorConstruction>(geom);
    auto* world = construction->Construct();
    BOOST_REQUIRE(world != nullptr);
    o2::base::GeometryManager::buildVecGeomGeometry(true);
    map.build(*construction, true);
    navigator = std::make_unique<VecGeomG4Navigator>(map, 1.e-9, false);
    navigator->SetWorldVolume(world);
  }
};

Fixture& fixture()
{
  static Fixture f;
  return f;
}

/// The path of the navigator's touchable, in TGeo's notation.
std::string touchablePath(G4Navigator& nav)
{
  std::unique_ptr<G4TouchableHistory> touchable(nav.CreateTouchableHistory());
  std::string path;
  for (int depth = touchable->GetHistoryDepth(); depth >= 0; --depth) {
    auto const* pv = touchable->GetVolume(depth);
    if (pv == nullptr) {
      return "";
    }
    // TGeo names a node by volume and copy number, and its top node is copy 1
    path += "/" + std::string(pv->GetName()) + "_" + std::to_string(depth == touchable->GetHistoryDepth() ? 1 : pv->GetCopyNo());
  }
  return path;
}

struct Visit {
  std::string path;
  double length; // cm
};

/// The volumes a ray crosses with a non-zero length, and how many zero-length steps came between.
std::vector<Visit> withNavigator(G4Navigator& nav, const double* p, const double* d, int& zeroSteps)
{
  std::vector<Visit> visits;
  G4ThreeVector point(p[0] * cm, p[1] * cm, p[2] * cm);
  const G4ThreeVector dir(d[0], d[1], d[2]);
  auto* pv = nav.LocateGlobalPointAndSetup(point, &dir, false, false);
  for (int k = 0; k < 10000 && pv != nullptr; ++k) {
    G4double safety = 0.;
    const double step = nav.ComputeStep(point, dir, 1.e6 * cm, safety);
    if (!(step < 1.e6 * cm)) {
      break;
    }
    if (step > 1.e-7 * cm) {
      visits.push_back({touchablePath(nav), step / cm});
    } else {
      ++zeroSteps;
    }
    point += step * dir;
    nav.SetGeometricallyLimitedStep();
    pv = nav.LocateGlobalPointAndSetup(point, &dir, true, false);
  }
  return visits;
}

/// Where the navigator is after each boundary along a ray. The locate after boundary `turnAt` gets the
/// reversed direction, as after a scattering at the face, and the walk ends there.
std::vector<std::string> afterBoundaries(G4Navigator& nav, const double* p, const double* d, int turnAt)
{
  std::vector<std::string> paths;
  G4ThreeVector point(p[0] * cm, p[1] * cm, p[2] * cm);
  const G4ThreeVector dir(d[0], d[1], d[2]);
  auto* pv = nav.LocateGlobalPointAndSetup(point, &dir, false, false);
  for (int k = 0; k < 10000 && pv != nullptr; ++k) {
    G4double safety = 0.;
    const double step = nav.ComputeStep(point, dir, 1.e6 * cm, safety);
    if (!(step < 1.e6 * cm)) {
      break;
    }
    point += step * dir;
    nav.SetGeometricallyLimitedStep();
    const G4ThreeVector after = k == turnAt ? -dir : dir;
    pv = nav.LocateGlobalPointAndSetup(point, &after, true, false);
    paths.push_back(pv != nullptr ? touchablePath(nav) : "");
    if (k == turnAt) {
      break;
    }
  }
  return paths;
}

std::vector<Visit> withTGeo(const double* p, const double* d)
{
  std::vector<Visit> visits;
  gGeoManager->InitTrack(p[0], p[1], p[2], d[0], d[1], d[2]);
  for (int k = 0; k < 10000 && !gGeoManager->IsOutside(); ++k) {
    const std::string path = gGeoManager->GetPath();
    gGeoManager->FindNextBoundaryAndStep();
    if (gGeoManager->GetStep() > 1.e-7) {
      visits.push_back({path, gGeoManager->GetStep()});
    }
  }
  return visits;
}
/// Counts the events Geant4 would abort instead of aborting the test.
struct AbortCounter : public G4VExceptionHandler {
  int aborts = 0;
  G4bool Notify(const char*, const char* code, G4ExceptionSeverity severity, const char*) override
  {
    aborts += severity == EventMustBeAborted && std::string(code) == "GeomNav0003";
    return false;
  }
};

/// The top face of the lead tile of the first paper of a tower: module (1-3), tower (1, 2), x offset (cm).
G4ThreeVector leadFace(int module, int tower, double dx = 0.)
{
  const double mx[] = {0., 0., 40.}, my[] = {-5., 5., 0.};
  return G4ThreeVector((mx[module - 1] + (tower == 1 ? -5. : 5.) + dx) * cm, my[module - 1] * cm, -8.45 * cm);
}

/// Puts the track at \a p, as a boundary step there would, without a crossing.
void moveTo(G4Navigator& nav, const G4ThreeVector& p)
{
  const G4ThreeVector momentum(0., 1., 0.);
  nav.LocateGlobalPointWithinVolume(p);
  nav.SetGeometricallyLimitedStep();
  nav.LocateGlobalPointAndSetup(p, &momentum, true, false);
}

/// Chords only, as under a field, alternately into the lead tile below \a p and out of it, at \a p. Returns the
/// first lengthened step (0 if none in \a maxSteps) and counts the zero steps before it.
double chordsUntilPush(G4Navigator& nav, const G4ThreeVector& p, int& zeroSteps, int maxSteps = 30)
{
  // Along y, parallel to the faces between towers, so that a push along them stays in the tower.
  const G4ThreeVector in = G4ThreeVector(0., 1., -1.e-3).unit(), out = G4ThreeVector(0., 1., 2.e-3).unit();
  const G4ThreeVector momentum(0., 1., 0.); // never the direction of a step: every step is a chord
  zeroSteps = 0;
  for (int k = 0; k < maxSteps; ++k) {
    G4double safety = 0.;
    const bool inLead = touchablePath(nav).find("LEAD") != std::string::npos;
    const double step = nav.ComputeStep(p, inLead ? out : in, 1.e6 * cm, safety);
    nav.SetGeometricallyLimitedStep();
    nav.LocateGlobalPointAndSetup(p, &momentum, true, false);
    if (step > 1.e-10 * mm) {
      return step;
    }
    ++zeroSteps;
  }
  return 0.;
}
} // namespace

BOOST_AUTO_TEST_CASE(rays_visit_the_volumes_tgeo_visits)
{
  auto& f = fixture();
  std::mt19937_64 gen(1);
  std::uniform_real_distribution<double> u(-1., 1.);
  std::normal_distribution<double> g;
  int rays = 0, differentCount = 0, differentPath = 0, differentLength = 0, zeroSteps = 0, crossings = 0;
  std::string firstDifference;
  for (int r = 0; r < 20000; ++r) {
    // start inside the two touching modules, off every face
    const double p[3] = {9.9 * u(gen), 9.9 * u(gen), 9.9 * u(gen)};
    double d[3] = {g(gen), g(gen), 0.3 * g(gen)};
    const double norm = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    for (auto& x : d) {
      x /= norm;
    }
    const auto expected = withTGeo(p, d);
    const auto found = withNavigator(*f.navigator, p, d, zeroSteps);
    ++rays;
    crossings += expected.size();
    if (found.size() != expected.size()) {
      ++differentCount;
      if (firstDifference.empty()) {
        firstDifference = Form("ray %d: %zu visits, TGeo %zu", r, found.size(), expected.size());
      }
      continue;
    }
    for (std::size_t i = 0; i < found.size(); ++i) {
      if (found[i].path != expected[i].path) {
        ++differentPath;
        if (firstDifference.empty()) {
          firstDifference = Form("ray %d visit %zu: %s, TGeo %s", r, i, found[i].path.c_str(), expected[i].path.c_str());
        }
        break;
      }
      differentLength += std::abs(found[i].length - expected[i].length) > 1.e-6;
    }
  }
  BOOST_TEST_MESSAGE("rays " << rays << ", volumes crossed " << crossings << ", zero-length steps " << zeroSteps << " " << firstDifference);
  BOOST_CHECK_GT(crossings, 100000);
  BOOST_CHECK_MESSAGE(differentCount == 0, differentCount << " rays with a different number of volumes; " << firstDifference);
  BOOST_CHECK_MESSAGE(differentPath == 0, differentPath << " rays with a different volume; " << firstDifference);
  BOOST_CHECK_EQUAL(differentLength, 0);
  // every face here coincides with others, and none of them is a place to stop
  BOOST_CHECK_EQUAL(zeroSteps, 0);
}

BOOST_AUTO_TEST_CASE(a_turn_at_the_face_does_not_change_the_volume_entered)
{
  auto& f = fixture();
  std::mt19937_64 gen(2);
  std::uniform_real_distribution<double> u(-1., 1.);
  std::normal_distribution<double> g;
  int boundaries = 0, different = 0;
  std::string firstDifference;
  for (int r = 0; r < 2000; ++r) {
    const double p[3] = {9.9 * u(gen), 9.9 * u(gen), 9.9 * u(gen)};
    double d[3] = {g(gen), g(gen), 0.3 * g(gen)};
    const double norm = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    for (auto& x : d) {
      x /= norm;
    }
    const auto straight = afterBoundaries(*f.navigator, p, d, -1);
    for (int k = 0; k < static_cast<int>(straight.size()); ++k) {
      const auto turned = afterBoundaries(*f.navigator, p, d, k);
      ++boundaries;
      if (turned.size() != static_cast<std::size_t>(k) + 1 || turned.back() != straight[k]) {
        ++different;
        if (firstDifference.empty()) {
          firstDifference = Form("ray %d boundary %d: %s, without the turn %s", r, k, turned.empty() ? "-" : turned.back().c_str(), straight[k].c_str());
        }
      }
    }
  }
  BOOST_TEST_MESSAGE("boundaries " << boundaries << " " << firstDifference);
  BOOST_CHECK_GT(boundaries, 10000);
  // the step decides which volume is entered, not the direction the track has after it
  BOOST_CHECK_MESSAGE(different == 0, different << " boundaries where the turn changed the volume; " << firstDifference);
}

BOOST_AUTO_TEST_CASE(a_track_tangent_to_a_field_boundary_is_freed_along_its_momentum)
{
  // A track on the top face of a lead tile, as under a field: the chord points into the tile, the momentum out of it.
  auto& nav = *fixture().navigator;
  const G4ThreeVector face(-5. * cm, -5. * cm, -8.45 * cm);
  const G4ThreeVector chord = G4ThreeVector(1., 0., -1.e-3).unit();
  const G4ThreeVector momentum = G4ThreeVector(1., 0., 1.e-3).unit();
  G4double safety = 0.;
  // reach the face along the chord from the paper above, so the track ends in the lead
  G4ThreeVector point = face - 1. * mm * chord;
  nav.LocateGlobalPointAndSetup(point, &chord, false, false);
  const double first = nav.ComputeStep(point, chord, 1.e6 * cm, safety);
  BOOST_REQUIRE_CLOSE(first, 1. * mm, 1.e-6);
  point += first * chord;
  nav.SetGeometricallyLimitedStep();
  nav.LocateGlobalPointAndSetup(point, &momentum, true, false);
  BOOST_REQUIRE_EQUAL(touchablePath(nav).substr(touchablePath(nav).rfind('/')), "/LEAD_1");

  // Geant4 alternates a straight step along the momentum, which leaves the lead, and a chord step, which enters it again.
  int zeroSteps = 0, pushedChords = 0;
  bool moved = false, freed = false;
  for (int k = 0; k < 24 && !freed; ++k) {
    const bool alongMomentum = k % 2 == 0;
    const G4ThreeVector& dir = alongMomentum ? momentum : chord;
    const double step = nav.ComputeStep(point, dir, 1.e6 * cm, safety);
    BOOST_TEST_MESSAGE("step " << k << (alongMomentum ? " momentum " : " chord ") << step << " mm in " << touchablePath(nav));
    if (step < 1.e-10 * mm) {
      ++zeroSteps;
    } else if (alongMomentum) {
      // the lengthened momentum step moves the track off the face
      point += step * dir;
      moved = true;
    } else if (moved) {
      // the chord reaches the lead only after a real step: the track is off the face
      freed = true;
    } else {
      ++pushedChords;
      break;
    }
    nav.SetGeometricallyLimitedStep();
    nav.LocateGlobalPointAndSetup(point, &momentum, true, false);
  }
  BOOST_TEST_MESSAGE("zero steps " << zeroSteps);
  BOOST_CHECK_GE(zeroSteps, 10);
  // a lengthened chord step is a miss for the field propagator, which then never crosses
  BOOST_CHECK_EQUAL(pushedChords, 0);
  BOOST_CHECK(freed);
}

BOOST_AUTO_TEST_CASE(a_push_restarts_the_run_of_zero_steps_only_in_another_volume_it_reached)
{
  auto& nav = *fixture().navigator;
  const G4ThreeVector momentum(0., 1., 0.);
  int zero = 0;
  // A run of chords on the lead face of tower 1, 2 nm from tower 2: the tenth zero step is pushed by 0.1 nm.
  const G4ThreeVector p(-2.e-7 * cm, -5. * cm, -8.45 * cm);
  nav.LocateGlobalPointAndSetup(p, &momentum, false, false);
  BOOST_CHECK_CLOSE(chordsUntilPush(nav, p, zero), 1.e-7 * mm, 0.01);
  BOOST_CHECK_EQUAL(zero, 9);
  // Moved along the same face by ten times the push, still in the volumes of the run: no restart, the next zero
  // step is pushed at once, and the push goes on growing.
  const G4ThreeVector same(-1.e-7 * cm, -5. * cm, -8.45 * cm);
  moveTo(nav, same);
  double push = 0.;
  for (int k = 1; k < 4; ++k) {
    push = chordsUntilPush(nav, same, zero);
    BOOST_CHECK_EQUAL(zero, 0);
    BOOST_CHECK_CLOSE(push, 1.e-7 * mm * std::pow(4., k), 0.01);
  }
  // Into tower 2, a volume the run never visited, but by less than half the last push: no restart either.
  const G4ThreeVector close(1.e-7 * cm, -5. * cm, -8.45 * cm);
  BOOST_REQUIRE_LT((close - same).mag(), 0.5 * push);
  moveTo(nav, close);
  BOOST_CHECK_NE(touchablePath(nav).find("TOWR_2"), std::string::npos);
  BOOST_CHECK_GT(chordsUntilPush(nav, close, zero), 0.);
  BOOST_CHECK_EQUAL(zero, 0);
  // Into another module, far and new: the run starts again.
  const G4ThreeVector far = leadFace(2, 1);
  moveTo(nav, far);
  BOOST_CHECK_GT(chordsUntilPush(nav, far, zero), 0.);
  BOOST_CHECK_EQUAL(zero, 9);
}

BOOST_AUTO_TEST_CASE(a_track_pushed_from_place_to_place_is_aborted_after_its_fourth_run)
{
  auto& nav = *fixture().navigator;
  static AbortCounter handler; // registered with Geant4 for the rest of the process
  handler.aborts = 0;
  const G4ThreeVector momentum(0., 1., 0.);
  int zero = 0;
  // Four places, each reached by a push and each a volume the previous run did not visit: three restarts.
  const G4ThreeVector places[] = {leadFace(1, 1), leadFace(1, 2), leadFace(2, 1), leadFace(2, 2)};
  nav.LocateGlobalPointAndSetup(places[0], &momentum, false, false);
  for (const auto& p : places) {
    if (&p != &places[0]) {
      moveTo(nav, p);
    }
    BOOST_CHECK_GT(chordsUntilPush(nav, p, zero), 0.);
    BOOST_CHECK_EQUAL(zero, 9);
  }
  // A fifth place does not restart the run: it goes on to the abort.
  const G4ThreeVector last = leadFace(3, 1);
  moveTo(nav, last);
  int steps = 0;
  for (; steps < 30 && handler.aborts == 0; ++steps) {
    chordsUntilPush(nav, last, zero, 1);
  }
  BOOST_CHECK_EQUAL(handler.aborts, 1);
  // The fourth run had reached ten zero steps, so fifteen more reach the abort at twenty-five.
  BOOST_CHECK_EQUAL(steps, 15);
}
