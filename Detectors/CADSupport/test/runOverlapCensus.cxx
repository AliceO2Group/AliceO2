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

/// \file runOverlapCensus.cxx
/// \brief "Is this assembly legal?", as a routine run rather than a bespoke investigation.
///
/// Built as `o2-bench-cadsupport-overlap`.
///
/// Takes any TGeo geometry file and answers, pair by pair and by name, whether the placed solids
/// compose into a world TGeo and Geant4 will accept -- separating the pairs that *share a face*
/// (legal, and the normal state of an assembly) from the pairs that *interpenetrate* (illegal, and
/// silently wrong transport). `--inject` translates one node first, which is the positive control:
/// a check that cannot fail has not passed.

#include "CADSupport/O2OverlapCheck.h"
#include "CADSupport/O2BVHSurfaceSolid.h"
#include "DetectorsBase/O2Tessellated.h"

#include "TGeoBBox.h"
#include "TGeoManager.h"
#include "TGeoMatrix.h"
#include "TGeoNode.h"
#include "TGeoTube.h"
#include "TGeoVolume.h"
#include "TFile.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace o2::cad;

namespace
{

struct Options {
  std::string geometry;
  std::string topVolume;
  std::string json;
  std::vector<std::string> injections;
  OverlapOptions check;
  bool rootCheck = false;
  int rootNmesh = 0;
  double rootOvlp = 0.001;
  bool selfTest = false;
  bool listPairs = false;
};

void usage(const char* argv0)
{
  std::cout
    << "usage: " << argv0 << " --geometry <geom.root> [options]\n"
    << "   or: " << argv0 << " --self-test\n\n"
    << "  --geometry PATH      ROOT file holding a TGeoManager (as written by geom.C's\n"
    << "                       build_and_export, or by any other producer)\n"
    << "  --top NAME           volume whose daughters are censused (default: the top volume)\n"
    << "  --points N           boundary points sampled per solid (default 20000). Coverage only:\n"
    << "                       every individual answer is exact, so this bounds false NEGATIVES\n"
    << "  --tol CM             depth below which a containment is a shared face, not an overlap\n"
    << "                       (default 1e-6)\n"
    << "  --residual CM        a sampled point further than this from its own solid's boundary is\n"
    << "                       discarded rather than used as evidence (default 1e-6)\n"
    << "  --pad CM             bounding-box inflation for the pairwise rejection (default 0.1).\n"
    << "                       Scopes which DISJOINT pairs get measured; never hides an overlap\n"
    << "  --volume-samples N   Monte-Carlo estimate of the shared volume of each illegal pair\n"
    << "  --inject NAME:DX,DY,DZ   translate a node by (dx,dy,dz) cm before the census. The\n"
    << "                       positive control; may be repeated\n"
    << "  --root-check [N]     also run TGeoManager::CheckOverlaps for comparison, optionally\n"
    << "                       after SetNmeshPoints(N)\n"
    << "  --root-ovlp CM       the tolerance handed to CheckOverlaps (default 0.001)\n"
    << "  --list-pairs         print every tested pair, not only the illegal ones\n"
    << "  --json PATH          write the census as JSON\n"
    << "  --self-test          analytic controls, no geometry file needed; exits non-zero on any\n"
    << "                       failure\n\n"
    << "Exit code is the number of illegal pairs, capped at 250; 251 on a usage or load error.\n";
}

bool parseInjection(const std::string& spec, std::string& name, double shift[3])
{
  const auto colon = spec.rfind(':');
  if (colon == std::string::npos) {
    return false;
  }
  name = spec.substr(0, colon);
  return std::sscanf(spec.c_str() + colon + 1, "%lf,%lf,%lf", &shift[0], &shift[1], &shift[2]) == 3;
}

/// Translate a placed node by \a shift cm in the mother frame. Used only by --inject.
bool injectShift(TGeoVolume* mother, const std::string& nodeName, const double shift[3])
{
  for (int index = 0; index < mother->GetNdaughters(); ++index) {
    TGeoNode* node = mother->GetNode(index);
    if (nodeName != node->GetVolume()->GetName() && nodeName != node->GetName()) {
      continue;
    }
    auto* nodeWithMatrix = dynamic_cast<TGeoNodeMatrix*>(node);
    if (nodeWithMatrix == nullptr) {
      return false;
    }
    auto* replacement = new TGeoHMatrix(*node->GetMatrix());
    const double* translation = replacement->GetTranslation();
    replacement->SetDx(translation[0] + shift[0]);
    replacement->SetDy(translation[1] + shift[1]);
    replacement->SetDz(translation[2] + shift[2]);
    replacement->RegisterYourself();
    nodeWithMatrix->SetMatrix(replacement);
    return true;
  }
  return false;
}

void printCensus(const OverlapCensus& census, bool listPairs)
{
  std::printf("\n%d placed solids -> %d pairs; %d survive the bounding-box rejection (%.2f %%)\n", census.nSolids,
              census.nPairsTotal, census.nPairsTested,
              census.nPairsTotal > 0 ? 100. * census.nPairsTested / census.nPairsTotal : 0.);
  std::printf("disjoint %d | touching %d | INTERPENETRATING %d | contained %d | extruding %d      (%.1f s)\n",
              census.nDisjoint, census.nTouching, census.nInterpenetrating, census.nContained, census.nExtruding,
              census.elapsedSeconds);
  std::printf("points rejected as not on their own solid: %d; worst accepted residual %.3e cm\n",
              census.nPointsRejected, census.worstResidualCm);

  std::printf("\n%-28s %-10s %10s %10s %8s %8s %6s\n", "solid", "shape", "requested", "accepted", "rejected",
              "residual", "onSeg");
  for (const auto& solid : census.solids) {
    std::printf("%-28s %-10s %10d %10d %8d %8.1e %6s\n", solid.name.c_str(),
                solid.shapeClass.size() > 10 ? solid.shapeClass.substr(solid.shapeClass.size() - 10).c_str()
                                             : solid.shapeClass.c_str(),
                solid.requested, solid.accepted, solid.rejected, solid.worstResidualCm,
                solid.usedPointsOnSegments ? "yes" : "no");
  }

  std::printf("\n%-46s %-17s %13s %8s %8s %13s\n", "pair", "verdict", "depth(cm)", "AinB", "BinA", "sep/vol");
  for (const auto& pair : census.pairs) {
    const bool illegal =
      pair.verdict == OverlapVerdict::Interpenetrating || pair.verdict == OverlapVerdict::Contained;
    if (!listPairs && !illegal) {
      continue;
    }
    char label[128];
    std::snprintf(label, sizeof(label), "%s | %s", pair.nameA.c_str(), pair.nameB.c_str());
    char trailer[64] = "";
    if (pair.sharedVolumeCm3 >= 0.) {
      std::snprintf(trailer, sizeof(trailer), "V=%.4e", pair.sharedVolumeCm3);
    } else if (pair.separationCm >= 0.) {
      std::snprintf(trailer, sizeof(trailer), "gap=%.4e", pair.separationCm);
    }
    std::printf("%-46s %-17s %13.6e %8d %8d %13s\n", label, OverlapVerdictName(pair.verdict), pair.depthCm,
                pair.deepPointsAInsideB, pair.deepPointsBInsideA, trailer);
  }
  for (const auto& pair : census.extrusions) {
    std::printf("%-46s %-17s %13.6e %8d %8s %13s\n", (pair.nameA + " extrudes " + pair.nameB).c_str(), "EXTRUSION",
                pair.depthCm, pair.deepPointsAInsideB, "-", "");
  }
}

nlohmann::json censusToJson(const OverlapCensus& census)
{
  nlohmann::json out;
  out["nSolids"] = census.nSolids;
  out["nPairsTotal"] = census.nPairsTotal;
  out["nPairsTested"] = census.nPairsTested;
  out["nDisjoint"] = census.nDisjoint;
  out["nTouching"] = census.nTouching;
  out["nInterpenetrating"] = census.nInterpenetrating;
  out["nContained"] = census.nContained;
  out["nExtruding"] = census.nExtruding;
  out["illegal"] = census.illegalCount();
  out["nPointsRejected"] = census.nPointsRejected;
  out["worstResidualCm"] = census.worstResidualCm;
  out["elapsedSeconds"] = census.elapsedSeconds;
  for (const auto& solid : census.solids) {
    out["solids"].push_back({{"name", solid.name},
                             {"shape", solid.shapeClass},
                             {"requested", solid.requested},
                             {"accepted", solid.accepted},
                             {"rejected", solid.rejected},
                             {"worstResidualCm", solid.worstResidualCm},
                             {"usedPointsOnSegments", solid.usedPointsOnSegments}});
  }
  auto pairJson = [](const OverlapPair& pair) {
    return nlohmann::json{{"a", pair.nameA},
                          {"b", pair.nameB},
                          {"verdict", OverlapVerdictName(pair.verdict)},
                          {"depthCm", pair.depthCm},
                          {"deepestPoint", pair.deepestPoint},
                          {"deepestPointFrom", pair.deepestPointFrom},
                          {"pointsAInsideB", pair.pointsAInsideB},
                          {"pointsBInsideA", pair.pointsBInsideA},
                          {"deepPointsAInsideB", pair.deepPointsAInsideB},
                          {"deepPointsBInsideA", pair.deepPointsBInsideA},
                          {"sampledA", pair.sampledA},
                          {"sampledB", pair.sampledB},
                          {"separationCm", pair.separationCm},
                          {"sharedVolumeCm3", pair.sharedVolumeCm3},
                          {"sharedVolumeErrCm3", pair.sharedVolumeErrCm3},
                          {"sharedVolumeHits", pair.sharedVolumeHits}};
  };
  for (const auto& pair : census.pairs) {
    out["pairs"].push_back(pairJson(pair));
  }
  for (const auto& pair : census.extrusions) {
    out["extrusions"].push_back(pairJson(pair));
  }
  return out;
}

// ---------------------------------------------------------------------------------------------
// Self-test: the three populations, built from arithmetic, with the controls that make them mean
// something. No geometry file, no build directory, no model.
// ---------------------------------------------------------------------------------------------

int gChecks = 0;
int gFailures = 0;

void check(bool condition, const std::string& what)
{
  gChecks++;
  if (!condition) {
    gFailures++;
    std::printf("  FAIL  %s\n", what.c_str());
  } else {
    std::printf("  ok    %s\n", what.c_str());
  }
}

TGeoVolume* makeWorld(const char* name)
{
  auto* manager = new TGeoManager(name, name);
  auto* material = new TGeoMaterial("vac", 0., 0., 0.);
  auto* medium = new TGeoMedium("vac", 1, material);
  auto* world = manager->MakeBox("world", medium, 100., 100., 100.);
  manager->SetTopVolume(world);
  return world;
}

/// A unit cube as an O2BVHSurfaceSolid, so the controls run on the representation this branch
/// ships rather than only on ROOT's primitives.
O2BVHSurfaceSolid* makeSurfaceBox(const char* name, double halfX, double halfY, double halfZ)
{
  auto* solid = new O2BVHSurfaceSolid(name);
  const double faces[6][3] = {{1., 0., 0.}, {-1., 0., 0.}, {0., 1., 0.}, {0., -1., 0.}, {0., 0., 1.}, {0., 0., -1.}};
  const double half[3] = {halfX, halfY, halfZ};
  for (const auto& normal : faces) {
    const int axis = (normal[0] != 0.) ? 0 : ((normal[1] != 0.) ? 1 : 2);
    const int axisU = (axis + 1) % 3;
    const int axisV = (axis + 2) % 3;
    O2BVHSurfaceSolid::Point3D origin{0., 0., 0.};
    origin[axis] = normal[axis] * half[axis];
    O2BVHSurfaceSolid::Point3D directionU{0., 0., 0.};
    O2BVHSurfaceSolid::Point3D directionV{0., 0., 0.};
    directionU[axisU] = 1.;
    directionV[axisV] = 1.;
    // Wind the quad so its normal points out of the box.
    const double sign = normal[axis];
    std::vector<O2BVHSurfaceSolid::Point2D> wire;
    const double extentU = half[axisU];
    const double extentV = half[axisV];
    if (sign > 0) {
      wire = {{-extentU, -extentV}, {extentU, -extentV}, {extentU, extentV}, {-extentU, extentV}};
    } else {
      wire = {{-extentU, -extentV}, {-extentU, extentV}, {extentU, extentV}, {extentU, -extentV}};
    }
    solid->AddPlanarSurface(origin, directionU, directionV, wire, {});
  }
  solid->CloseShape(false);
  return solid;
}

int selfTest()
{
  std::printf("== o2-bench-cadsupport-overlap self-test ==\n");

  OverlapOptions options;
  options.pointsPerSolid = 4000;
  options.checkExtrusion = false;

  // --- 1. Two boxes sharing a face exactly: TOUCHING, and it must not be called an overlap. ---
  {
    TGeoVolume* world = makeWorld("touch");
    auto* left = new TGeoVolume("left", makeSurfaceBox("leftBox", 1., 1., 1.), world->GetMedium());
    auto* right = new TGeoVolume("right", makeSurfaceBox("rightBox", 1., 1., 1.), world->GetMedium());
    world->AddNode(left, 1, new TGeoTranslation(-1., 0., 0.));
    world->AddNode(right, 1, new TGeoTranslation(1., 0., 0.));
    gGeoManager->CloseGeometry();
    const OverlapCensus census = CheckWorldOverlaps(world, options);
    check(census.nPairsTested == 1, "touching: the pair survives the box rejection");
    check(census.nTouching == 1 && census.nInterpenetrating == 0,
          "touching: a shared face is TOUCHING, not an overlap");
    check(!census.pairs.empty() && census.pairs[0].pointsAInsideB + census.pairs[0].pointsBInsideA > 0,
          "touching: the check was capable of firing (points ARE found inside)");
    check(!census.pairs.empty() && census.pairs[0].depthCm <= options.depthTolerance,
          "touching: the depth is at the tolerance, i.e. zero");
    delete gGeoManager;
  }

  // --- 2. The same two boxes moved 0.2 cm into each other: INTERPENETRATING, at that depth. ---
  {
    TGeoVolume* world = makeWorld("overlap");
    auto* left = new TGeoVolume("left", makeSurfaceBox("leftBox", 1., 1., 1.), world->GetMedium());
    auto* right = new TGeoVolume("right", makeSurfaceBox("rightBox", 1., 1., 1.), world->GetMedium());
    world->AddNode(left, 1, new TGeoTranslation(-1., 0., 0.));
    world->AddNode(right, 1, new TGeoTranslation(0.8, 0., 0.));
    gGeoManager->CloseGeometry();
    OverlapOptions withVolume = options;
    withVolume.volumeSamples = 200000;
    const OverlapCensus census = CheckWorldOverlaps(world, withVolume);
    check(census.nInterpenetrating == 1, "injected 0.2 cm: INTERPENETRATING");
    const double depth = census.pairs.empty() ? 0. : census.pairs[0].depthCm;
    check(std::abs(depth - 0.2) < 1e-9,
          "injected 0.2 cm: the depth is the injected displacement (" + std::to_string(depth) + ")");
    const double volume = census.pairs.empty() ? -1. : census.pairs[0].sharedVolumeCm3;
    check(std::abs(volume - 0.8) < 0.02,
          "injected 0.2 cm: shared volume 0.2 x 2 x 2 = 0.8 cm3 (" + std::to_string(volume) + ")");
    delete gGeoManager;
  }

  // --- 3. The negative control: the same two boxes 0.2 cm APART must not fire. ---
  {
    TGeoVolume* world = makeWorld("gap");
    auto* left = new TGeoVolume("left", makeSurfaceBox("leftBox", 1., 1., 1.), world->GetMedium());
    auto* right = new TGeoVolume("right", makeSurfaceBox("rightBox", 1., 1., 1.), world->GetMedium());
    world->AddNode(left, 1, new TGeoTranslation(-1., 0., 0.));
    world->AddNode(right, 1, new TGeoTranslation(1.2, 0., 0.));
    gGeoManager->CloseGeometry();
    const OverlapCensus census = CheckWorldOverlaps(world, options);
    check(census.nDisjoint == 1 && census.illegalCount() == 0, "0.2 cm gap: disjoint, nothing flagged");
    const double separation = census.pairs.empty() ? -1. : census.pairs[0].separationCm;
    check(std::abs(separation - 0.2) < 1e-9,
          "0.2 cm gap: the separation is recovered (" + std::to_string(separation) + ")");
    delete gGeoManager;
  }

  // --- 4. A tenth of a micron: the tolerance is a decision, and it is measured, not assumed. ---
  {
    TGeoVolume* world = makeWorld("thin");
    auto* left = new TGeoVolume("left", makeSurfaceBox("leftBox", 1., 1., 1.), world->GetMedium());
    auto* right = new TGeoVolume("right", makeSurfaceBox("rightBox", 1., 1., 1.), world->GetMedium());
    world->AddNode(left, 1, new TGeoTranslation(-1., 0., 0.));
    world->AddNode(right, 1, new TGeoTranslation(1. - 1e-5, 0., 0.));
    gGeoManager->CloseGeometry();
    const OverlapCensus loose = CheckWorldOverlaps(world, options);
    check(loose.nInterpenetrating == 1, "1e-5 cm interpenetration is resolved at the default 1e-6 tolerance");
    OverlapOptions coarse = options;
    coarse.depthTolerance = 1e-4;
    const OverlapCensus blunted = CheckWorldOverlaps(world, coarse);
    check(blunted.nInterpenetrating == 0 && blunted.nTouching == 1,
          "CONTROL: at a 1e-4 tolerance the same 1e-5 interpenetration reads as touching");
    delete gGeoManager;
  }

  // --- 5. Containment, which is legal only as a declared mother/daughter. ---
  {
    TGeoVolume* world = makeWorld("nested");
    auto* outer = new TGeoVolume("outer", makeSurfaceBox("outerBox", 3., 3., 3.), world->GetMedium());
    auto* inner = new TGeoVolume("inner", makeSurfaceBox("innerBox", 1., 1., 1.), world->GetMedium());
    world->AddNode(outer, 1, new TGeoTranslation(0., 0., 0.));
    world->AddNode(inner, 1, new TGeoTranslation(0., 0., 0.));
    gGeoManager->CloseGeometry();
    const OverlapCensus census = CheckWorldOverlaps(world, options);
    check(census.nContained == 1, "a solid wholly inside another is CONTAINED, not merely overlapping");
    delete gGeoManager;
  }

  // --- 6. A curved contact: a press fit exact in the model must not read as an overlap. This is
  //        the case ROOT's checker got wrong, and the sagitta of its 24-gon is 8.6e-3 cm. ---
  {
    TGeoVolume* world = makeWorld("press");
    auto* pin = new TGeoVolume("pin", new TGeoTube("pinTube", 0., 1., 5.), world->GetMedium());
    auto* sleeve = new TGeoVolume("sleeve", new TGeoTube("sleeveTube", 1., 2., 5.), world->GetMedium());
    world->AddNode(pin, 1, new TGeoTranslation(0., 0., 0.));
    world->AddNode(sleeve, 1, new TGeoTranslation(0., 0., 0.));
    gGeoManager->CloseGeometry();
    const OverlapCensus census = CheckWorldOverlaps(world, options);
    check(census.nInterpenetrating == 0 && census.nTouching == 1,
          "an exact press fit on a cylinder is TOUCHING, not an 8.6e-3 cm overlap");
    const double depth = census.pairs.empty() ? -1. : census.pairs[0].depthCm;
    check(depth < 1e-6, "press fit: depth " + std::to_string(depth) + " is below the 24-gon sagitta 8.6e-3 by 4 decades");
    delete gGeoManager;
  }

  // --- 7. The residual filter: a point that is not on its own solid is not evidence. ---
  {
    auto* box = new TGeoBBox("residualBox", 1., 1., 1.);
    std::vector<double> points;
    int rejected = 0;
    double worst = 0.;
    const int accepted = SampleBoundaryPoints(box, 4000, 1e-6, points, rejected, worst);
    check(accepted > 0 && rejected == 0, "TGeoBBox: every sampled point is on the box");
    check(worst < 1e-9, "TGeoBBox: worst accepted residual " + std::to_string(worst) + " is at round-off");
  }

  // --- 8. The sampling contract on the shape this branch ships. ---
  {
    auto* solid = makeSurfaceBox("contractBox", 1., 2., 3.);
    const int meshVertices = solid->GetNmeshVertices();
    std::vector<double> buffer(3 * (meshVertices + 5000), -1.2345e33);
    check(solid->GetPointsOnSegments(meshVertices + 5000, buffer.data()),
          "GetPointsOnSegments fills the buffer when asked for more than the mesh");
    int unfilled = 0;
    double worst = 0.;
    for (int index = 0; index < meshVertices + 5000; ++index) {
      if (buffer[3 * index] == -1.2345e33) {
        unfilled++;
        continue;
      }
      worst = std::max(worst, solid->Safety(&buffer[3 * index], solid->Contains(&buffer[3 * index])));
    }
    check(unfilled == 0, "GetPointsOnSegments leaves no slot unwritten");
    check(worst < O2BVHSurfaceSolid::kSurfacePointTolerance,
          "every generated point is on the solid (worst " + std::to_string(worst) + ")");
    check(!solid->GetPointsOnSegments(meshVertices - 1, buffer.data()),
          "below the mesh size it declines, so ROOT falls back to the full exact vertex set");
    delete solid;
  }

  std::printf("\n%d checks, %d failures\n", gChecks, gFailures);
  return gFailures == 0 ? 0 : 1;
}

} // namespace

int main(int argc, char** argv)
{
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    auto next = [&](const char* what) -> std::string {
      if (index + 1 >= argc) {
        std::cerr << "error: " << what << " needs a value\n";
        std::exit(251);
      }
      return argv[++index];
    };
    if (argument == "-h" || argument == "--help") {
      usage(argv[0]);
      return 0;
    } else if (argument == "--geometry") {
      options.geometry = next("--geometry");
    } else if (argument == "--top") {
      options.topVolume = next("--top");
    } else if (argument == "--points") {
      options.check.pointsPerSolid = std::atoi(next("--points").c_str());
    } else if (argument == "--tol") {
      options.check.depthTolerance = std::atof(next("--tol").c_str());
    } else if (argument == "--residual") {
      options.check.residualTolerance = std::atof(next("--residual").c_str());
    } else if (argument == "--pad") {
      options.check.padCm = std::atof(next("--pad").c_str());
    } else if (argument == "--volume-samples") {
      options.check.volumeSamples = std::atoi(next("--volume-samples").c_str());
    } else if (argument == "--inject") {
      options.injections.push_back(next("--inject"));
    } else if (argument == "--root-check") {
      options.rootCheck = true;
      if (index + 1 < argc && argv[index + 1][0] != '-') {
        options.rootNmesh = std::atoi(argv[++index]);
      }
    } else if (argument == "--root-ovlp") {
      options.rootOvlp = std::atof(next("--root-ovlp").c_str());
    } else if (argument == "--list-pairs") {
      options.listPairs = true;
    } else if (argument == "--json") {
      options.json = next("--json");
    } else if (argument == "--self-test") {
      options.selfTest = true;
    } else {
      std::cerr << "error: unknown argument " << argument << "\n";
      usage(argv[0]);
      return 251;
    }
  }

  if (options.selfTest) {
    return selfTest();
  }
  if (options.geometry.empty()) {
    usage(argv[0]);
    return 251;
  }

  TGeoManager::Import(options.geometry.c_str());
  if (gGeoManager == nullptr) {
    std::cerr << "error: no TGeoManager in " << options.geometry << "\n";
    return 251;
  }
  TGeoVolume* top = options.topVolume.empty() ? gGeoManager->GetTopVolume()
                                              : gGeoManager->GetVolume(options.topVolume.c_str());
  if (top == nullptr) {
    std::cerr << "error: no such volume: " << options.topVolume << "\n";
    return 251;
  }

  for (const auto& specification : options.injections) {
    std::string name;
    double shift[3] = {0., 0., 0.};
    if (!parseInjection(specification, name, shift)) {
      std::cerr << "error: cannot parse --inject " << specification << " (expected NAME:DX,DY,DZ)\n";
      return 251;
    }
    if (!injectShift(top, name, shift)) {
      std::cerr << "error: --inject names no daughter of " << top->GetName() << ": " << name << "\n";
      return 251;
    }
    std::printf("# injected: %s by (%g, %g, %g) cm\n", name.c_str(), shift[0], shift[1], shift[2]);
  }

  std::printf("# geometry %s, top volume %s, %d points per solid, depth tolerance %g cm, pad %g cm\n",
              options.geometry.c_str(), top->GetName(), options.check.pointsPerSolid, options.check.depthTolerance,
              options.check.padCm);

  const OverlapCensus census = CheckWorldOverlaps(top, options.check);
  printCensus(census, options.listPairs);

  if (options.rootCheck) {
    std::printf("\n== TGeoManager::CheckOverlaps, for comparison (nmesh %s, ovlp %g) ==\n",
                options.rootNmesh > 0 ? std::to_string(options.rootNmesh).c_str() : "default", options.rootOvlp);
    gGeoManager->GetGeomPainter();
    if (options.rootNmesh > 0) {
      gGeoManager->SetNmeshPoints(options.rootNmesh);
    }
    gGeoManager->CheckOverlaps(options.rootOvlp);
    gGeoManager->PrintOverlaps();
  }

  if (!options.json.empty()) {
    nlohmann::json out = censusToJson(census);
    out["geometry"] = options.geometry;
    out["top"] = top->GetName();
    out["options"] = {{"pointsPerSolid", options.check.pointsPerSolid},
                      {"depthToleranceCm", options.check.depthTolerance},
                      {"residualToleranceCm", options.check.residualTolerance},
                      {"padCm", options.check.padCm},
                      {"volumeSamples", options.check.volumeSamples}};
    out["injections"] = options.injections;
    std::ofstream stream(options.json);
    stream << out.dump(1) << "\n";
    std::printf("\nwrote %s\n", options.json.c_str());
  }

  return std::min(census.illegalCount(), 250);
}
