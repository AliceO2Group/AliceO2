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

/// \file testGeometryDoctorPlacements.cxx
/// \brief Runs o2-sim-geometry-doctor on small geometries with a shared assembly placed twice.

#include <TGeoManager.h>
#include <TGeoMaterial.h>
#include <TGeoMatrix.h>
#include <TGeoMedium.h>
#include <TGeoVolume.h>

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <sys/wait.h>

using json = nlohmann::json;

namespace
{

int failures = 0;

void check(bool condition, const std::string& what)
{
  std::cout << (condition ? "  ok    " : "  FAIL  ") << what << '\n';
  failures += condition ? 0 : 1;
}

enum class Layout { Ideal,
                    Extruded,
                    Shadowed,
                    Supported,
                    Grid };

constexpr int kGrid = 17; ///< 17^3 copies, so the per-placement pass is split into many units

/// A hall with an assembly of a sensitive gas box and a frame placed twice in it.
/// Extruded moves copy 2 out of the hall; shadowed places a panel, before the
/// copies, that contains the gas of copy 2, as an aligned slat inside a support
/// panel. Supported puts that panel into an assembly placed twice, the second copy
/// over the gas of copy 2: TGeo gives the overlap to the panel, VecGeom to the gas,
/// so the second panel is lost only for VecGeom. Grid places a smaller assembly 17^3 times and moves copies 2 and the last
/// one out of the hall.
void buildGeometry(Layout layout, const std::string& file)
{
  delete gGeoManager;
  new TGeoManager("doctorTest", "geometry doctor placement test");
  auto* air = new TGeoMaterial("Air", 14.6, 7.3, 0.0012);
  auto* gas = new TGeoMaterial("Gas", 39.9, 18., 0.0018);
  auto* steel = new TGeoMaterial("Steel", 55.8, 26., 7.87);
  double params[20] = {0.};
  auto* airMedium = new TGeoMedium("Air", 1, air, params);
  auto* steelMedium = new TGeoMedium("Steel", 3, steel, params);
  params[0] = 1.; // sensitive
  auto* gasMedium = new TGeoMedium("Gas", 2, gas, params);

  auto* world = gGeoManager->MakeBox("World", airMedium, 400., 400., 400.);
  gGeoManager->SetTopVolume(world);
  auto* hall = gGeoManager->MakeBox("Hall", airMedium, 100., 100., 100.);
  world->AddNode(hall, 1);

  auto* module = new TGeoVolumeAssembly("Module");
  if (layout == Layout::Grid) {
    module->AddNode(gGeoManager->MakeBox("Gas", gasMedium, 1., 1., 1.), 1);
    module->AddNode(gGeoManager->MakeBox("Frame", steelMedium, 1., 1., 1.), 1, new TGeoTranslation(0., 3., 0.));
    const int last = kGrid * kGrid * kGrid;
    for (int copy = 1; copy <= last; ++copy) {
      const int i = (copy - 1) % kGrid, j = (copy - 1) / kGrid % kGrid, k = (copy - 1) / (kGrid * kGrid);
      double x = -80. + 10. * i, y = -80. + 10. * j, z = -80. + 10. * k;
      if (copy == 2 || copy == last) {
        x = copy == 2 ? 150. : -150.;
        y = z = 0.;
      }
      hall->AddNode(module, copy, new TGeoTranslation(x, y, z));
    }
  } else {
    module->AddNode(gGeoManager->MakeBox("Gas", gasMedium, 5., 5., 5.), 1);
    module->AddNode(gGeoManager->MakeBox("Frame", steelMedium, 5., 5., 5.), 1, new TGeoTranslation(0., 20., 0.));
    if (layout == Layout::Shadowed) {
      hall->AddNode(gGeoManager->MakeBox("Panel", steelMedium, 10., 8., 8.), 1, new TGeoTranslation(50., 0., 0.));
    }
    if (layout == Layout::Supported) {
      auto* support = new TGeoVolumeAssembly("Support");
      support->AddNode(gGeoManager->MakeBox("Panel", steelMedium, 10., 8., 8.), 1);
      hall->AddNode(support, 1, new TGeoTranslation(0., -60., 0.));
      hall->AddNode(support, 2, new TGeoTranslation(50., 0., 0.));
    }
    hall->AddNode(module, 1, new TGeoTranslation(-50., 0., 0.));
    hall->AddNode(module, 2, new TGeoTranslation(layout == Layout::Extruded ? 150. : 50., 0., 0.));
  }
  gGeoManager->CloseGeometry();
  gGeoManager->Export(file.c_str());
}

struct Run {
  int exitCode = -1;
  json result;
  std::string text;
};

Run runDoctor(const std::string& exe, const std::string& name, const std::string& extra)
{
  const std::string command = exe + " --geometry-file " + name + ".root --reachability-only --output-prefix " + name +
                              " " + extra + " > " + name + ".log 2>&1";
  std::cout << command << '\n';
  Run run;
  const std::string jsonFile = name + "-reachability.json";
  std::remove(jsonFile.c_str()); // never read the file of an earlier run
  const int status = std::system(command.c_str());
  run.exitCode = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
  check(run.exitCode == 0 || run.exitCode == 3, name + ": exit code 0 or 3");
  std::ifstream in(jsonFile);
  std::stringstream buffer;
  buffer << in.rdbuf();
  run.text = buffer.str();
  try {
    run.result = json::parse(run.text);
  } catch (const std::exception& e) {
    std::cout << "  cannot parse " << name << "-reachability.json: " << e.what() << '\n';
  }
  const bool valid = run.result.is_object() && run.result.contains("findings");
  check(valid, name + "-reachability.json written");
  if (!valid) {
    run.result = json{{"findings", json::array()}};
  }
  return run;
}

const json* findingFor(const Run& run, const std::string& path)
{
  for (const auto& finding : run.result["findings"]) {
    if (finding["path"] == path) {
      return &finding;
    }
  }
  return nullptr;
}

bool mentions(const Run& run, const std::string& part)
{
  for (const auto& finding : run.result["findings"]) {
    if (finding["path"].get<std::string>().find(part) != std::string::npos) {
      return true;
    }
  }
  return false;
}

std::set<std::string> deadPaths(const Run& run)
{
  std::set<std::string> dead;
  for (const auto& finding : run.result["findings"]) {
    if (finding["kind"] == "dead") {
      dead.insert(finding["path"].get<std::string>());
    }
  }
  return dead;
}

} // namespace

int main(int argc, char** argv)
{
  if (argc < 2) {
    std::cerr << "usage: " << argv[0] << " <path to o2-sim-geometry-doctor>\n";
    return 1;
  }
  const std::string exe = argv[1];
  buildGeometry(Layout::Ideal, "ideal.root");
  buildGeometry(Layout::Extruded, "extruded.root");
  buildGeometry(Layout::Shadowed, "shadowed.root");
  buildGeometry(Layout::Supported, "supported.root");
  buildGeometry(Layout::Grid, "grid.root");
  const std::string gas2 = "/World_1/Hall_1/Module_2/Gas_1";
  const std::string frame2 = "/World_1/Hall_1/Module_2/Frame_1";

  std::cout << "ideal\n";
  auto ideal = runDoctor(exe, "ideal", "--reachability-jobs 2");
  check(ideal.exitCode == 0, "exit code 0");
  check(ideal.result["findings"].empty(), "no finding");
  check(ideal.result["placements"] == 6, "6 non-assembly placements");

  std::cout << "extruded\n";
  auto extruded = runDoctor(exe, "extruded", "--reachability-jobs 2");
  check(extruded.exitCode == 3, "exit code 3");
  check(extruded.result["findings"].size() == 2, "two findings");
  const json* gas = findingFor(extruded, gas2);
  check(gas != nullptr && (*gas)["kind"] == "dead" && (*gas)["scope"] == "placement" && (*gas)["sensitive"] == true &&
          (*gas)["group"] == "/World_1/Hall_1/Module_2" && (*gas)["instances"] == 2,
        "gas of copy 2 dead, placement scope, sensitive, grouped under Module_2, 2 instances");
  const json* frame = findingFor(extruded, frame2);
  check(frame != nullptr && (*frame)["kind"] == "dead" && (*frame)["sensitive"] == false, "frame of copy 2 dead");
  check(!mentions(extruded, "Module_1"), "nothing for copy 1");

  std::cout << "extruded, control without the per-placement pass\n";
  auto control = runDoctor(exe, "extruded", "--reachability-jobs 2 --placement-samples 0");
  check(control.exitCode == 0, "exit code 0: the per-object audit alone is blind to copy 2");
  check(control.result["findings"].empty(), "no finding");

  std::cout << "shadowed\n";
  auto shadowed = runDoctor(exe, "shadowed", "--reachability-jobs 2");
  check(!mentions(shadowed, "Module_1"), "nothing for copy 1");
  const json* shadowedGas = findingFor(shadowed, gas2);
  check(shadowedGas != nullptr && (*shadowedGas)["kind"] == "dead" && (*shadowedGas)["scope"] == "placement",
        "the panel, placed first, takes the gas of copy 2: dead, placement scope");
  auto shadowedControl = runDoctor(exe, "shadowed", "--reachability-jobs 2 --placement-samples 0");
  check(findingFor(shadowedControl, gas2) == nullptr, "the per-object audit alone does not see it");

  std::cout << "supported, both navigators\n";
  auto both = runDoctor(exe, "supported", "--reachability-jobs 2 --navigator both");
  const json* panel2 = findingFor(both, "/World_1/Hall_1/Support_2/Panel_1");
  std::cout << "  navigator used: " << both.result.value("navigator", "?") << '\n';
  if (both.result.value("navigator", "") == "both") {
    check(panel2 != nullptr && (*panel2)["scope"] == "placement" && (*panel2)["kind"] == "disagree" &&
            (*panel2)["reached"] == 1. && (*panel2)["vgReached"].get<double>() < 0.9,
          "the second panel, lost only for VecGeom, is reported as a disagreement");
  }

  std::cout << "grid of " << kGrid * kGrid * kGrid << " copies, cut into many units\n";
  const std::string last = std::to_string(kGrid * kGrid * kGrid);
  const std::set<std::string> gridDead{"/World_1/Hall_1/Module_2/Gas_1", "/World_1/Hall_1/Module_2/Frame_1",
                                       "/World_1/Hall_1/Module_" + last + "/Gas_1",
                                       "/World_1/Hall_1/Module_" + last + "/Frame_1"};
  auto gridOne = runDoctor(exe, "grid", "--reachability-jobs 1");
  check(gridOne.exitCode == 3 && deadPaths(gridOne) == gridDead && gridOne.result["findings"].size() == 4,
        "copies 2 and " + last + " dead, nothing else");
  const std::string gridFirst = gridOne.text;
  auto gridFour = runDoctor(exe, "grid", "--reachability-jobs 4");
  check(!gridFirst.empty() && gridFirst == gridFour.text, "grid: identical JSON with 1 and 4 threads");

  std::cout << "reproducibility, 1 against 4 threads\n";
  for (const std::string name : {"extruded", "shadowed"}) {
    auto one = runDoctor(exe, name, "--reachability-jobs 1");
    const std::string first = one.text;
    auto four = runDoctor(exe, name, "--reachability-jobs 4");
    check(!first.empty() && first == four.text, name + ": identical JSON");
  }

  std::cout << "baseline\n";
  std::system("cp ideal-reachability.json baseline-ideal.json");
  auto against = runDoctor(exe, "extruded", "--reachability-jobs 2 --reachability-baseline baseline-ideal.json");
  int fresh = 0;
  for (const auto& finding : against.result["findings"]) {
    fresh += finding.value("new", false) ? 1 : 0;
  }
  check(against.exitCode == 3 && fresh == 2, "extruded against ideal: two new dead, exit code 3");
  auto itself = runDoctor(exe, "ideal", "--reachability-jobs 2 --reachability-baseline baseline-ideal.json");
  check(itself.exitCode == 0 && itself.result["findings"].empty(), "ideal against itself: nothing new, exit code 0");
  std::system("cp extruded-reachability.json baseline-extruded.json");
  auto same = runDoctor(exe, "extruded", "--reachability-jobs 2 --reachability-baseline baseline-extruded.json");
  fresh = 0;
  for (const auto& finding : same.result["findings"]) {
    fresh += finding.value("new", false) ? 1 : 0;
  }
  check(same.exitCode == 0 && fresh == 0, "extruded against itself: findings known, nothing new, exit code 0");

  std::cout << "extruded with the VecGeom navigator\n";
  auto vecgeom = runDoctor(exe, "extruded", "--reachability-jobs 2 --navigator vecgeom");
  std::cout << "  navigator used: " << vecgeom.result.value("navigator", "?") << '\n';
  check(vecgeom.exitCode == 3 && deadPaths(vecgeom) == std::set<std::string>{frame2, gas2},
        "the same dead placements");

  std::cout << (failures == 0 ? "all checks passed" : std::to_string(failures) + " checks failed") << '\n';
  return failures == 0 ? 0 : 1;
}
