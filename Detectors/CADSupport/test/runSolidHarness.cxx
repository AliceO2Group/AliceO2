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

/// \file runSolidHarness.cxx
/// \brief Front-end for the O2SolidHarness validation / performance comparison harness.
///
/// Built as o2-bench-cadsupport-solid-harness (see Detectors/CADSupport/CMakeLists.txt). Loads
/// paired surfaces_*.bin / facets_*.bin parts from a test-part database (see
/// Detectors/CADSupport/validation/makeTestPartDB.py) and, for each, validates and times
/// O2BVHSurfaceSolid (candidate) against O2Tessellated (reference). Usage and reading rules are in
/// Detectors/CADSupport/doc/reference/SolidNavigationHarness.md.

#include "CADSupport/O2SolidHarness.h"
#include "CADSupport/O2BVHSurfaceSolid.h"
#include "DetectorsBase/O2Tessellated.h"
#include "CADSupport/O2SurfaceSolidIO.h"

#include "TGeoBBox.h"
#include "TGeoCompositeShape.h"
#include "TGeoMatrix.h"
#include "TGeoScaledShape.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <iostream>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using json = nlohmann::json;
using namespace o2::cad;
using namespace o2::cad::harness;
using o2::base::O2Tessellated;

namespace
{

struct Options {
  std::string db;
  std::string explicitSurfaces;
  std::string explicitFacets;
  std::string partsPattern;
  int points = 5000;
  int rays = 5000;
  uint64_t seed = 1;
  std::set<std::string> only = {"contains", "distout", "distin", "safety"};
  bool loopCrosscheck = false;
  bool pruningAb = false;
  bool allRims = false; ///< print every rim, not only the ones that are not cleanly matched
  std::string jsonOut;
  int warmup = 1;
  int repeat = 3;
  std::string dumpSamples;   ///< directory to write per-part sample sets into, for the OCCT oracle
  std::string refAnswers;    ///< directory holding the oracle's answers for those sample sets
  std::string loadSamples;   ///< directory to read per-part sample sets from, instead of generating
  bool edgeIdentity = false; ///< report the sidecar-v3 edge-identity block
  std::string explicitShape; ///< ad-hoc mode: the shape_<part>.root sidecar to score alongside
};

struct Part {
  std::string id;
  std::string model;
  std::string surfaces;
  std::string facets;
  /// The `shape_<VOL>_<LID>.root` sidecar, when the part has one. Optional by construction: it is
  /// the future CSG emitter's output and no part has one today.
  std::string shape;
};

/// `surfaces_<suffix>.bin` -> `shape_<suffix>.root` in the same directory.
///
/// Derived rather than only read from the manifest so that a shape sidecar dropped next to the
/// other artifacts is picked up by a `--skip-convert` re-score, which is the loop anyone
/// developing an emitter will actually run. `makeTestPartDB.py` records the same path under the
/// manifest's `"shape"` key when it indexes the database, and that entry wins when present.
std::string deriveShapeSidecarPath(const std::string& surfacesPath)
{
  const auto slash = surfacesPath.find_last_of('/');
  const std::string dir = slash == std::string::npos ? std::string() : surfacesPath.substr(0, slash + 1);
  std::string base = slash == std::string::npos ? surfacesPath : surfacesPath.substr(slash + 1);
  const std::string prefix = "surfaces_";
  const std::string suffix = ".bin";
  if (base.rfind(prefix, 0) != 0 || base.size() <= prefix.size() + suffix.size() ||
      base.compare(base.size() - suffix.size(), suffix.size(), suffix) != 0) {
    return {};
  }
  const std::string stem = base.substr(prefix.size(), base.size() - prefix.size() - suffix.size());
  return dir + "shape_" + stem + ".root";
}

bool fileExists(const std::string& path)
{
  if (path.empty()) {
    return false;
  }
  std::ifstream probe(path);
  return static_cast<bool>(probe);
}

void printUsage(const char* argv0)
{
  std::cout << "Usage: " << argv0 << " --db <dir> [--parts <substring>] [--points N] [--rays N] [--seed N]\n"
                                     "                              [--only contains,distout,distin,safety] [--loop-crosscheck]\n"
                                     "                              [--pruning-ab] [--json <out.json>] [--warmup N] [--repeat N]\n"
                                     "   or: "
            << argv0 << " --surfaces <file> --facets <file> [--shape <file>] [options as above]\n\n"
                        "  Every representation a part has is scored side by side against the same oracle answers:\n"
                        "    surface  surfaces_<part>.bin  -> O2BVHSurfaceSolid   (the historical candidate)\n"
                        "    mesh     facets_<part>.bin    -> O2Tessellated       (also the sampling reference)\n"
                        "    shape    shape_<part>.root    -> any TGeoShape       (the CSG emitter's hand-over)\n"
                        "  The `shape` sidecar is one ROOT file holding one TGeoShape-derived object under the key\n"
                        "  \"shape\", in cm, plus an OPTIONAL TGeoHMatrix under the key \"placement\" taking it from\n"
                        "  its own frame into the part's; absent means identity, and points and rays are transformed\n"
                        "  into the shape's frame before it is asked. See CADSupport/O2SolidHarness.h.\n\n"
                        "  --loop-crosscheck  also run the surface solid's non-BVH _Loop twins and require exact\n"
                        "                     agreement; this is the correctness guard that does not involve the mesh\n"
                        "  --pruning-ab       re-run the distance kernels with ray tmax pruning disabled, reporting\n"
                        "                     the BVH candidate counts and ns/call both ways (prices the optimization)\n"
                        "  --rims             list every trim loop, not only the ones that are not cleanly matched;\n"
                        "                     the same records go into --json unconditionally\n"
                        "  --dump-samples D   write each part's sample set to D/samples_<part>.json\n"
                        "  --load-samples D   read each part's sample set from D/samples_<part>.json instead of\n"
                        "                     generating it. The generator derives its points from the *mesh*, so two\n"
                        "                     runs on differently-tessellated shapes cannot be compared point by\n"
                        "                     point; loading a frozen (and, for a transformed shape, transformed) set\n"
                        "                     removes the mesh from the comparison entirely. --points/--rays/--seed\n"
                        "                     are then ignored and the file's counts are used.\n"
                        "  --edge-identity    report the sidecar-v3 edge-identity block (source-edge counts and the\n"
                        "                     max shared-edge deviation) on stdout; it is always in --json\n"
                        "  --ref-answers D    validate against D/answers_<part>.json instead of the mesh; those are\n"
                        "                     produced by Detectors/CADSupport/validation/occtOracle.py from the part's .brep, so a\n"
                        "                     disagreement outside the model tolerance is a defect, not chording\n\n"
                        "OCCT oracle round trip:\n"
                        "  "
            << argv0 << " --db <db> --dump-samples /tmp/o\n"
                        "  occtOracle.py --brep <part>.brep --samples /tmp/o/samples_<part>.json \\\n"
                        "                --out /tmp/o/answers_<part>.json\n"
                        "  "
            << argv0 << " --db <db> --ref-answers /tmp/o\n\n"
                        "perf record entry point (single kernel, one part):\n"
                        "  perf record -g "
            << argv0 << " --db <db> --parts ExcavatorArm --only distout --rays 200000\n";
}

std::set<std::string> splitCsv(const std::string& s)
{
  std::set<std::string> out;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    if (!tok.empty()) {
      out.insert(tok);
    }
  }
  return out;
}

bool parseArgs(int argc, char** argv, Options& opt)
{
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&](const char* flag) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error(std::string("missing value for ") + flag);
      }
      return argv[++i];
    };
    if (a == "--db") {
      opt.db = next("--db");
    } else if (a == "--surfaces") {
      opt.explicitSurfaces = next("--surfaces");
    } else if (a == "--facets") {
      opt.explicitFacets = next("--facets");
    } else if (a == "--shape") {
      opt.explicitShape = next("--shape");
    } else if (a == "--parts") {
      opt.partsPattern = next("--parts");
    } else if (a == "--points") {
      opt.points = std::stoi(next("--points"));
    } else if (a == "--rays") {
      opt.rays = std::stoi(next("--rays"));
    } else if (a == "--seed") {
      opt.seed = std::stoull(next("--seed"));
    } else if (a == "--only") {
      opt.only = splitCsv(next("--only"));
    } else if (a == "--loop-crosscheck") {
      opt.loopCrosscheck = true;
    } else if (a == "--pruning-ab") {
      opt.pruningAb = true;
    } else if (a == "--rims") {
      opt.allRims = true;
    } else if (a == "--json") {
      opt.jsonOut = next("--json");
    } else if (a == "--warmup") {
      opt.warmup = std::stoi(next("--warmup"));
    } else if (a == "--repeat") {
      opt.repeat = std::stoi(next("--repeat"));
    } else if (a == "--dump-samples") {
      opt.dumpSamples = next("--dump-samples");
    } else if (a == "--ref-answers") {
      opt.refAnswers = next("--ref-answers");
    } else if (a == "--load-samples") {
      opt.loadSamples = next("--load-samples");
    } else if (a == "--edge-identity") {
      opt.edgeIdentity = true;
    } else if (a == "-h" || a == "--help") {
      printUsage(argv[0]);
      return false;
    } else {
      throw std::runtime_error("unrecognized option: " + a);
    }
  }
  if (opt.db.empty() && (opt.explicitSurfaces.empty() || opt.explicitFacets.empty())) {
    throw std::runtime_error("either --db <dir> or both --surfaces/--facets are required");
  }
  return true;
}

std::vector<Part> collectParts(const Options& opt)
{
  std::vector<Part> parts;
  if (!opt.explicitSurfaces.empty()) {
    Part part{"adhoc", "adhoc", opt.explicitSurfaces, opt.explicitFacets, opt.explicitShape};
    if (part.shape.empty()) {
      part.shape = deriveShapeSidecarPath(part.surfaces);
    }
    parts.push_back(std::move(part));
    return parts;
  }
  const std::string manifestPath = opt.db + "/manifest.json";
  std::ifstream in(manifestPath);
  if (!in) {
    throw std::runtime_error("cannot open " + manifestPath);
  }
  json manifest;
  in >> manifest;
  for (const auto& p : manifest.at("parts")) {
    Part part;
    part.id = p.at("id").get<std::string>();
    part.model = p.at("model").get<std::string>();
    part.surfaces = p.at("surfaces").get<std::string>();
    part.facets = p.at("facets").get<std::string>();
    part.shape = p.value("shape", std::string());
    if (part.shape.empty()) {
      part.shape = deriveShapeSidecarPath(part.surfaces);
    }
    if (!opt.partsPattern.empty()) {
      const bool idMatch = part.id.find(opt.partsPattern) != std::string::npos;
      const bool modelMatch = part.model.find(opt.partsPattern) != std::string::npos;
      if (!idMatch && !modelMatch) {
        continue;
      }
    }
    parts.push_back(std::move(part));
  }
  return parts;
}

json validationToJson(const ValidationResult& r)
{
  json j;
  j["nSamples"] = r.nSamples;
  j["nAgree"] = r.nAgree;
  j["nMismatchWithinBand"] = r.nMismatchWithinBand;
  j["nMismatchMissedSurface"] = r.nMismatchMissedSurface;
  j["nMismatchUnexplained"] = r.nMismatchUnexplained;
  j["nNoVerdict"] = r.nNoVerdict;
  j["nRelabelled"] = r.nRelabelled;
  j["worstDeviation"] = r.worstDeviation;
  json offenders = json::array();
  for (const auto& o : r.worstOffenders) {
    offenders.push_back({{"point", {o.point[0], o.point[1], o.point[2]}},
                         {"dir", {o.dir[0], o.dir[1], o.dir[2]}},
                         {"candidateValue", o.candidateValue},
                         {"referenceValue", o.referenceValue},
                         {"deviation", o.deviation},
                         {"referenceSafety", o.referenceSafety},
                         {"incidenceCosine", o.incidenceCosine}});
  }
  j["worstOffenders"] = offenders;
  return j;
}

// The sample/answer JSON contract shared with Detectors/CADSupport/validation/occtOracle.py. Bump on both sides
// together; the oracle refuses a version it does not speak rather than guessing.
constexpr int kOracleFormatVersion = 1;

/// Part ids carry '/' and other path-hostile characters; the oracle round trip pairs files by
/// this sanitized form on both sides.
std::string sanitizePartId(const std::string& id)
{
  std::string out;
  out.reserve(id.size());
  for (const char c : id) {
    out.push_back((std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '.') ? c : '_');
  }
  return out;
}

json pointsToJson(const std::vector<Point3D>& points)
{
  json array = json::array();
  for (const auto& p : points) {
    array.push_back({p[0], p[1], p[2]});
  }
  return array;
}

json raysToJson(const std::vector<Ray>& rays)
{
  json array = json::array();
  for (const auto& r : rays) {
    array.push_back({{"o", {r.origin[0], r.origin[1], r.origin[2]}},
                     {"d", {r.dir[0], r.dir[1], r.dir[2]}}});
  }
  return array;
}

/// Serialize a sample set so an external oracle can answer exactly the same queries. The samples
/// come from a seeded mt19937_64 inside the harness, so nothing outside can regenerate them;
/// dumping is the only way to ask another implementation about the same points.
void writeSamples(const std::string& dir, const std::string& partId, const SampleSet& samples)
{
  json doc;
  doc["version"] = kOracleFormatVersion;
  doc["part"] = partId;
  doc["bboxMin"] = {samples.bboxMin[0], samples.bboxMin[1], samples.bboxMin[2]};
  doc["bboxMax"] = {samples.bboxMax[0], samples.bboxMax[1], samples.bboxMax[2]};
  doc["points"] = {{"bulk", pointsToJson(samples.bulkPoints)},
                   {"boundary", pointsToJson(samples.boundaryPoints)},
                   {"inside", pointsToJson(samples.insidePoints)}};
  doc["rays"] = {{"outside", raysToJson(samples.outsideRays)},
                 {"inside", raysToJson(samples.insideRays)}};
  const std::string path = dir + "/samples_" + sanitizePartId(partId) + ".json";
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("cannot write " + path);
  }
  out << doc.dump(1);
  std::printf("  wrote samples: %s\n", path.c_str());
}

std::vector<Point3D> pointsFromJson(const json& array)
{
  std::vector<Point3D> points;
  points.reserve(array.size());
  for (const auto& p : array) {
    points.push_back(Point3D{p.at(0).get<double>(), p.at(1).get<double>(), p.at(2).get<double>()});
  }
  return points;
}

std::vector<Ray> raysFromJson(const json& array)
{
  std::vector<Ray> rays;
  rays.reserve(array.size());
  for (const auto& r : array) {
    const auto& o = r.at("o");
    const auto& d = r.at("d");
    rays.push_back(Ray{Point3D{o.at(0).get<double>(), o.at(1).get<double>(), o.at(2).get<double>()},
                       Point3D{d.at(0).get<double>(), d.at(1).get<double>(), d.at(2).get<double>()}});
  }
  return rays;
}

/// Read back a sample set written by writeSamples(), so that two runs on differently tessellated or
/// transformed shapes ask exactly the same questions.
SampleSet readSamples(const std::string& dir, const std::string& partId)
{
  const std::string path = dir + "/samples_" + sanitizePartId(partId) + ".json";
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("cannot read " + path);
  }
  json doc;
  in >> doc;
  const int version = doc.value("version", -1);
  if (version != kOracleFormatVersion) {
    throw std::runtime_error(path + ": sample format version " + std::to_string(version) +
                             ", this harness speaks " + std::to_string(kOracleFormatVersion));
  }
  SampleSet samples;
  for (int i = 0; i < 3; ++i) {
    samples.bboxMin[i] = doc.at("bboxMin").at(i).get<double>();
    samples.bboxMax[i] = doc.at("bboxMax").at(i).get<double>();
  }
  samples.bulkPoints = pointsFromJson(doc.at("points").at("bulk"));
  samples.boundaryPoints = pointsFromJson(doc.at("points").at("boundary"));
  samples.insidePoints = pointsFromJson(doc.at("points").at("inside"));
  samples.outsideRays = raysFromJson(doc.at("rays").at("outside"));
  samples.insideRays = raysFromJson(doc.at("rays").at("inside"));
  std::printf("  loaded samples: %s (bulk=%zu boundary=%zu inside=%zu outRays=%zu inRays=%zu)\n",
              path.c_str(), samples.bulkPoints.size(), samples.boundaryPoints.size(),
              samples.insidePoints.size(), samples.outsideRays.size(), samples.insideRays.size());
  return samples;
}

/// Oracle answers for one part, or `has == false` when no answer file exists for it.
struct OracleAnswers {
  bool has = false;
  double tolerance = 0.;
  double capacity = 0.;
  bool valid = false;
  /// The BREP's own bounding box, in the frame the oracle answered in. Every candidate must live
  /// in that same frame; this is what makes that checkable instead of assumed.
  bool hasBbox = false;
  Point3D bboxMin{};
  Point3D bboxMax{};
  std::map<std::string, std::vector<int>> containsState;
  /// Per ray category, the oracle's classification of each ray *origin* (1/0/-1). This is what
  /// makes the distance columns soundly categorised rather than categorised by the reference mesh.
  std::map<std::string, std::vector<int>> originContains;
  std::map<std::string, std::vector<double>> boundaryDistance;
  std::map<std::string, std::vector<double>> distOutside;
  std::map<std::string, std::vector<double>> distInside;
};

template <typename T>
std::map<std::string, std::vector<T>> readColumns(const json& parent, const char* key)
{
  std::map<std::string, std::vector<T>> columns;
  if (!parent.contains(key)) {
    return columns;
  }
  for (const auto& [category, values] : parent.at(key).items()) {
    columns[category] = values.template get<std::vector<T>>();
  }
  return columns;
}

OracleAnswers loadOracleAnswers(const std::string& dir, const std::string& partId)
{
  OracleAnswers answers;
  const std::string path = dir + "/answers_" + sanitizePartId(partId) + ".json";
  std::ifstream in(path);
  if (!in) {
    std::printf("  oracle: no answers file %s, skipping oracle validation\n", path.c_str());
    return answers;
  }
  json doc;
  in >> doc;
  const int version = doc.value("version", -1);
  if (version != kOracleFormatVersion) {
    throw std::runtime_error(path + ": answer format version " + std::to_string(version) +
                             ", this harness speaks " + std::to_string(kOracleFormatVersion));
  }
  answers.has = true;
  answers.tolerance = doc.value("tolerance", 0.);
  answers.capacity = doc.value("capacity", 0.);
  answers.valid = doc.value("valid", false);
  if (doc.contains("bboxMin") && doc.contains("bboxMax")) {
    answers.hasBbox = true;
    for (int i = 0; i < 3; ++i) {
      answers.bboxMin[i] = doc.at("bboxMin").at(i).get<double>();
      answers.bboxMax[i] = doc.at("bboxMax").at(i).get<double>();
    }
  }
  answers.containsState = readColumns<int>(doc, "contains");
  answers.originContains = readColumns<int>(doc, "originContains");
  answers.boundaryDistance = readColumns<double>(doc, "safetyUpperBound");
  answers.distOutside = readColumns<double>(doc, "distFromOutside");
  answers.distInside = readColumns<double>(doc, "distFromInside");
  return answers;
}

/// Concatenate the oracle's per-category columns in the same order the harness concatenates its
/// point categories, so index i of the merged column belongs to point i of `allPoints`.
///
/// Each column is padded to its category's *point count* before the next one is appended. That
/// padding is not cosmetic: the oracle answers `contains` for every point but caps the expensive
/// boundary-distance query, so its columns have different lengths per category. Concatenating
/// them raw would shift every later category's answers onto the wrong points -- a silent,
/// systematic mis-scoring rather than an error.
template <typename T>
std::vector<T> mergeCategories(const std::map<std::string, std::vector<T>>& columns,
                               const std::array<size_t, 3>& categorySizes, T missing)
{
  static constexpr std::array<const char*, 3> kOrder = {"bulk", "boundary", "inside"};
  std::vector<T> merged;
  for (size_t categoryIndex = 0; categoryIndex < kOrder.size(); ++categoryIndex) {
    const size_t expected = categorySizes[categoryIndex];
    const auto it = columns.find(kOrder[categoryIndex]);
    const size_t available = it == columns.end() ? 0 : std::min(expected, it->second.size());
    for (size_t i = 0; i < available; ++i) {
      merged.push_back(it->second[i]);
    }
    merged.insert(merged.end(), expected - available, missing);
  }
  return merged;
}

json timingToJson(const TimingResult& t)
{
  return {{"nCalls", t.nCalls}, {"nsPerCall", t.nsPerCall}, {"checksum", t.checksum}};
}

void printValidation(const std::string& name, const ValidationResult& r)
{
  // Scored = everything the reference was willing to answer. Reporting the percentage against
  // nSamples would let a reference that abstains on half the points look like agreement.
  const size_t scored = r.nSamples - r.nNoVerdict;
  const double agreePct = scored ? 100. * static_cast<double>(r.nAgree) / static_cast<double>(scored) : 0.;
  std::printf(
    "  %-10s scored=%-7zu agree=%6.2f%%  mismatch(band=%zu,missed=%zu,unexplained=%zu)"
    "  noVerdict=%zu  worstDev=%.6g\n",
    name.c_str(), scored, agreePct, r.nMismatchWithinBand, r.nMismatchMissedSurface,
    r.nMismatchUnexplained, r.nNoVerdict, r.worstDeviation);
  if (r.nRelabelled > 0) {
    // Not a candidate result: it says how many rays the sample generator had put in the wrong
    // category, which is a statement about the reference mesh. Printed so an improvement in these
    // columns is never mistaken for a kernel improvement.
    std::printf("  %-10s relabelled=%zu ray(s) by the oracle's own origin classification\n",
                name.c_str(), r.nRelabelled);
  }
  if (r.nMismatchUnexplained > 0 || r.nMismatchMissedSurface > 0) {
    const size_t nShow = std::min<size_t>(3, r.worstOffenders.size());
    for (size_t i = 0; i < nShow; ++i) {
      const auto& o = r.worstOffenders[i];
      std::printf("      offender[%zu]: point=(%.6g,%.6g,%.6g) dir=(%.6g,%.6g,%.6g) cand=%.6g ref=%.6g dev=%.6g refSafety=%.6g\n",
                  i, o.point[0], o.point[1], o.point[2], o.dir[0], o.dir[1], o.dir[2], o.candidateValue,
                  o.referenceValue, o.deviation, o.referenceSafety);
    }
  }
}

void printTiming(const std::string& name, const TimingResult& candidate, const TimingResult& reference)
{
  const double ratio = reference.nsPerCall > 0. ? candidate.nsPerCall / reference.nsPerCall : 0.;
  std::printf("  %-10s candidate=%9.1f ns/call   reference=%9.1f ns/call   ratio(cand/ref)=%.2fx\n", name.c_str(),
              candidate.nsPerCall, reference.nsPerCall, ratio);
}

// What the BVH traversal buys over the all-surfaces loop on the *same* shape: unlike the
// candidate/reference ratio this compares like with like, so it prices the acceleration structure
// alone rather than analytic patches against triangles.
void printLoopSpeedup(const std::string& name, const TimingResult& bvh, const TimingResult& loop)
{
  const double speedup = bvh.nsPerCall > 0. ? loop.nsPerCall / bvh.nsPerCall : 0.;
  std::printf("  %-10s   BVH=%9.1f ns/call   _Loop=%9.1f ns/call   speedup(loop/bvh)=%.2fx\n", name.c_str(),
              bvh.nsPerCall, loop.nsPerCall, speedup);
}

double toSeconds(std::chrono::steady_clock::time_point t0, std::chrono::steady_clock::time_point t1)
{
  return std::chrono::duration<double>(t1 - t0).count();
}

// ------------------------------------------------------------------------------------------
// Representations: the same part, scored several ways against one set of oracle answers
// ------------------------------------------------------------------------------------------
//
// The four scored queries are TGeoShape virtuals, so the scoring loop below has no business
// knowing what it is scoring. Everything that is specific to O2BVHSurfaceSolid -- closure, rims,
// NavigationReliability, the _Loop twins, the BVH candidate counters -- hangs off `surfaceSolid`,
// which is null for every other representation, and is reported only where it means something.
// A TGeoCompositeShape has no rims and no closure; reporting "reliable" or "not navigable" for it
// would be a category error, so those keys are simply absent from its entry and a
// `closureApplicable: false` says why.

struct Representation {
  std::string name;   ///< "surface" | "mesh" | "shape"
  std::string source; ///< the file it was loaded from
  const TGeoShape* shape = nullptr;
  const O2BVHSurfaceSolid* surfaceSolid = nullptr; ///< non-null only for "surface"
  int primitives = 0;                              ///< patches / triangles / -1 when not countable
  const char* primitiveKind = "";
  /// The shape's own frame, expressed in the part frame; null means the two are the same.
  ///
  /// Only the `shape` representation can have one, and only since a placed primitive stopped
  /// being written as a degenerate TGeoCompositeShape. **Points and rays are transformed into the
  /// shape's frame** rather than the shape being wrapped in something that carries the matrix.
  /// The reason is that this is the only arrangement under which the object the gate scores is
  /// the object the converter emitted: `shapeClass` is the real class, `Capacity()` is the real
  /// analytic capacity, and nothing between the sample and the shape can absorb an error. A
  /// wrapper (or a one-node TGeoVolume) would reintroduce exactly the indirection this change
  /// removed, and its own bounding box would be the inflated corner hull again.
  const TGeoMatrix* placement = nullptr;
};

/// A point of the part frame, expressed in the shape's own frame.
Point3D toLocal(const TGeoMatrix* placement, const Point3D& p)
{
  if (placement == nullptr) {
    return p;
  }
  Point3D out{};
  placement->MasterToLocal(p.data(), out.data());
  return out;
}

/// A direction of the part frame, expressed in the shape's own frame. A rigid transform preserves
/// lengths, so every distance the oracle states along a ray is unchanged by this -- which is why
/// the oracle's answers can be compared against the transformed query without touching them.
Ray toLocal(const TGeoMatrix* placement, const Ray& r)
{
  if (placement == nullptr) {
    return r;
  }
  Ray out{};
  placement->MasterToLocal(r.origin.data(), out.origin.data());
  placement->MasterToLocalVect(r.dir.data(), out.dir.data());
  return out;
}

/// Transformed copies of a sample vector. Returns an EMPTY vector when there is no placement, so
/// that the overwhelmingly common unplaced case selects the caller's own vector by reference and
/// copies nothing.
template <typename T>
std::vector<T> toLocal(const TGeoMatrix* placement, const std::vector<T>& in)
{
  if (placement == nullptr) {
    return {};
  }
  std::vector<T> out;
  out.reserve(in.size());
  for (const auto& item : in) {
    out.push_back(toLocal(placement, item));
  }
  return out;
}

/// How the shape computes Capacity(), and therefore whether comparing it against the OCCT volume
/// is a measurement or noise.
///
/// `TGeoCompositeShape::Capacity()` throws 10000 accepted Monte-Carlo points into the bounding
/// box (TGeoCompositeShape.cxx:282), so its relative error is ~1e-2 -- four orders of magnitude
/// above the 1e-6 gate band. It is reported, and explicitly marked not comparable, rather than
/// silently producing a failure that means nothing. Every other ROOT shape in this version
/// computes Capacity in closed form (checked: TGeoCompositeShape is the only Capacity() in
/// geom/geom/src that touches gRandom).
struct CapacityKind {
  const char* method = "root-analytic";
  bool comparable = true;
};

bool usesMonteCarloCapacity(const TGeoShape* shape)
{
  if (shape == nullptr) {
    return false;
  }
  if (shape->InheritsFrom(TGeoCompositeShape::Class())) {
    return true;
  }
  // TGeoScaledShape::Capacity() forwards to the shape it wraps, so a scaled composite is just as
  // sampled as a bare one.
  if (const auto* scaled = dynamic_cast<const TGeoScaledShape*>(shape)) {
    return usesMonteCarloCapacity(scaled->GetShape());
  }
  return false;
}

CapacityKind capacityKindOf(const Representation& rep)
{
  if (rep.surfaceSolid != nullptr) {
    // Divergence theorem in closed form over the analytic faces.
    return {"exact-divergence", true};
  }
  if (dynamic_cast<const O2Tessellated*>(rep.shape) != nullptr) {
    // Exact for the mesh (signed tetrahedra over its own triangles), deterministic, and therefore
    // a real measurement -- of the chording deficit, not of a bug.
    return {"mesh-divergence", true};
  }
  if (usesMonteCarloCapacity(rep.shape)) {
    return {"root-montecarlo", false};
  }
  return {"root-analytic", true};
}

/// Max deviation, in cm, between a shape's own bounding box and the oracle's, over all six faces.
///
/// This is the frame check. A TGeoShape answers in its own local frame and the oracle answers in
/// the .brep's; if an emitter writes a shape in the assembly frame instead of the part frame,
/// every column below fills with plausible-looking nonsense and nothing else would notice.
/// Returns -1 when the shape does not derive from TGeoBBox (nothing in ROOT's shape library that
/// matters here fails that) or when the answer file predates the bbox fields.
///
/// With a placement, the shape's box has to be carried into the part frame before it can be
/// compared, and the only frame-independent way to do that is to transform the eight corners and
/// take their axis-aligned hull. For a rotated body that hull is strictly larger than the body, so
/// the number becomes conservative -- exactly as it already was for a TGeoCompositeShape, whose
/// TGeoBoolNode::ComputeBBox does the same thing internally. It is a *frame* check, not a
/// tightness measurement, and it still moves by the size of a frame error.
double bboxDeviationFromOracle(const TGeoShape* shape, const OracleAnswers& oracle,
                               const TGeoMatrix* placement = nullptr)
{
  if (!oracle.hasBbox) {
    return -1.;
  }
  const auto* box = dynamic_cast<const TGeoBBox*>(shape);
  if (box == nullptr) {
    return -1.;
  }
  const double half[3] = {box->GetDX(), box->GetDY(), box->GetDZ()};
  double lo[3];
  double hi[3];
  for (int i = 0; i < 3; ++i) {
    lo[i] = box->GetOrigin()[i] - half[i];
    hi[i] = box->GetOrigin()[i] + half[i];
  }
  if (placement != nullptr) {
    double outLo[3] = {1.e300, 1.e300, 1.e300};
    double outHi[3] = {-1.e300, -1.e300, -1.e300};
    for (int corner = 0; corner < 8; ++corner) {
      const double local[3] = {(corner & 1) ? hi[0] : lo[0], (corner & 2) ? hi[1] : lo[1],
                               (corner & 4) ? hi[2] : lo[2]};
      double master[3];
      placement->LocalToMaster(local, master);
      for (int i = 0; i < 3; ++i) {
        outLo[i] = std::min(outLo[i], master[i]);
        outHi[i] = std::max(outHi[i], master[i]);
      }
    }
    std::copy(std::begin(outLo), std::end(outLo), std::begin(lo));
    std::copy(std::begin(outHi), std::end(outHi), std::begin(hi));
  }
  double worst = 0.;
  for (int i = 0; i < 3; ++i) {
    worst = std::max(worst, std::fabs(lo[i] - oracle.bboxMin[i]));
    worst = std::max(worst, std::fabs(hi[i] - oracle.bboxMax[i]));
  }
  return worst;
}

/// Everything the gate reads out of one (candidate, oracle answers) pair. Deliberately typed on
/// TGeoShape*: this is the whole point of the abstraction.
///
/// The returned object is exactly the historical `partJson["oracle"]` block, so the surface
/// representation's columns are produced by the same code that produced them before and the
/// existing path stays inert.
json scoreAgainstOracle(const TGeoShape* candidate, const OracleAnswers& oracle,
                        const ValidationOptions& oracleOpt, const std::vector<Point3D>& allPointsIn,
                        const std::vector<int>& containsState,
                        const std::vector<double>& boundaryDistance, const SampleSet& samplesIn,
                        const std::set<std::string>& only, const std::string& label,
                        const std::string& capacityLabel, const TGeoMatrix* placement = nullptr)
{
  // The samples are stated in the part frame -- the frame the oracle answered in. A shape that
  // carries a placement answers in its own, so the *questions* move and the answers do not: a
  // rigid transform preserves both the inside/outside relation and every distance along a ray.
  const std::vector<Point3D> localPoints = toLocal(placement, allPointsIn);
  const std::vector<Ray> localOutsideRays = toLocal(placement, samplesIn.outsideRays);
  const std::vector<Ray> localInsideRays = toLocal(placement, samplesIn.insideRays);
  const std::vector<Point3D>& allPoints = placement != nullptr ? localPoints : allPointsIn;
  const std::vector<Ray>& outsideRays =
    placement != nullptr ? localOutsideRays : samplesIn.outsideRays;
  const std::vector<Ray>& insideRays = placement != nullptr ? localInsideRays : samplesIn.insideRays;
  json oracleJson;
  oracleJson["tolerance"] = oracle.tolerance;
  oracleJson["capacity"] = oracle.capacity;
  oracleJson["valid"] = oracle.valid;
  const double capacity = candidate->Capacity();
  oracleJson["capacityCandidate"] = capacity;
  oracleJson["capacityRelativeDeviation"] =
    oracle.capacity != 0. ? (capacity - oracle.capacity) / oracle.capacity : 0.;
  std::printf("  %s: capacity candidate=%.6g reference=%.6g relDev=%.3g\n", capacityLabel.c_str(),
              capacity, oracle.capacity, oracleJson["capacityRelativeDeviation"].get<double>());

  if (only.count("contains")) {
    auto v = validateContainsAgainstOracle(candidate, allPoints, containsState, boundaryDistance,
                                           oracleOpt);
    printValidation(label + ":contains", v);
    oracleJson["contains"] = validationToJson(v);
  }
  const auto originStateFor = [&oracle](const char* category) {
    const auto it = oracle.originContains.find(category);
    return it == oracle.originContains.end() ? std::vector<int>{} : it->second;
  };
  if (only.count("distout")) {
    const auto it = oracle.distOutside.find("outside");
    if (it != oracle.distOutside.end()) {
      auto v = validateDistanceAgainstOracle(candidate, outsideRays, it->second,
                                             /*wantInside=*/false, oracleOpt,
                                             originStateFor("outside"));
      printValidation(label + ":distout", v);
      oracleJson["distout"] = validationToJson(v);
    }
  }
  if (only.count("distin")) {
    const auto it = oracle.distInside.find("inside");
    if (it != oracle.distInside.end()) {
      auto v = validateDistanceAgainstOracle(candidate, insideRays, it->second,
                                             /*wantInside=*/true, oracleOpt, originStateFor("inside"));
      printValidation(label + ":distin", v);
      oracleJson["distin"] = validationToJson(v);
    }
  }
  if (only.count("safety")) {
    auto v = validateSafetyAgainstOracle(candidate, allPoints, boundaryDistance, oracleOpt);
    printValidation(label + ":safety", v);
    oracleJson["safety"] = validationToJson(v);
  }
  return oracleJson;
}

/// Disagreements outside tolerance, summed over the four columns. This is the invariant the
/// project defends and it is a *different* number from the gate total, so it is computed once
/// here and reported next to every representation rather than reconstructed by each consumer.
size_t countDisagreements(const json& oracleJson)
{
  size_t bad = 0;
  for (const char* key : {"contains", "distout", "distin", "safety"}) {
    if (!oracleJson.contains(key)) {
      continue;
    }
    const auto& column = oracleJson.at(key);
    bad += column.value("nMismatchUnexplained", size_t{0});
    bad += column.value("nMismatchMissedSurface", size_t{0});
  }
  return bad;
}

} // namespace

int main(int argc, char** argv)
{
  Options opt;
  try {
    if (!parseArgs(argc, argv, opt)) {
      return 0;
    }
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    printUsage(argv[0]);
    return 1;
  }

  std::vector<Part> parts;
  try {
    parts = collectParts(opt);
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  if (parts.empty()) {
    std::cerr << "no parts matched (pattern='" << opt.partsPattern << "')\n";
    return 1;
  }

  json jsonReport = json::array();
  std::vector<std::string> unreliableParts;

  for (const auto& part : parts) {
    std::printf("=== %s (%s) ===\n", part.id.c_str(), part.model.c_str());

    O2BVHSurfaceSolid surf(part.id.c_str());
    if (!LoadSurfaceSolid(part.surfaces, surf)) {
      std::cerr << "  skip: LoadSurfaceSolid failed for " << part.surfaces << "\n";
      continue;
    }
    auto t0 = std::chrono::steady_clock::now();
    surf.CloseShape(true);
    auto t1 = std::chrono::steady_clock::now();
    const double surfCloseSeconds = toSeconds(t0, t1);

    O2Tessellated mesh(part.id.c_str());
    if (!LoadFacetSolid(part.facets, mesh)) {
      std::cerr << "  skip: LoadFacetSolid failed for " << part.facets << "\n";
      continue;
    }
    t0 = std::chrono::steady_clock::now();
    mesh.CloseShape();
    t1 = std::chrono::steady_clock::now();
    const double meshCloseSeconds = toSeconds(t0, t1);

    const TGeoShape* candidate = &surf;
    const TGeoShape* reference = &mesh;

    // Every representation this part has, in the order they are reported. `surface` first so the
    // historical candidate keeps its place; `mesh` second because it is also the sampling
    // reference; `shape` last because it is optional and does not exist yet for any converted
    // part -- it is the slot the CSG emitter writes into.
    std::vector<Representation> representations;
    representations.push_back({"surface", part.surfaces, &surf, &surf, surf.GetNsurfaces(), "patches"});
    representations.push_back({"mesh", part.facets, &mesh, nullptr, mesh.GetNfacets(), "triangles"});
    std::unique_ptr<TGeoShape> rootShape;
    std::unique_ptr<TGeoHMatrix> rootShapePlacement;
    if (fileExists(part.shape)) {
      std::string shapeError;
      rootShape.reset(loadShapeFromRootFile(part.shape, &shapeError));
      if (rootShape) {
        rootShapePlacement.reset(loadShapePlacementFromRootFile(part.shape));
        std::printf("  shape sidecar: %s -> %s \"%s\"%s\n", part.shape.c_str(),
                    rootShape->ClassName(), rootShape->GetName(),
                    rootShapePlacement ? "  (placed: queries are transformed into its own frame)"
                                       : "");
        representations.push_back({"shape", part.shape, rootShape.get(), nullptr, -1,
                                   rootShape->ClassName(), rootShapePlacement.get()});
      } else {
        std::printf("  shape sidecar: *** %s\n", shapeError.c_str());
      }
    }

    const Point3D bboxMin{mesh.GetOrigin()[0] - mesh.GetDX(), mesh.GetOrigin()[1] - mesh.GetDY(),
                          mesh.GetOrigin()[2] - mesh.GetDZ()};
    const Point3D bboxMax{mesh.GetOrigin()[0] + mesh.GetDX(), mesh.GetOrigin()[1] + mesh.GetDY(),
                          mesh.GetOrigin()[2] + mesh.GetDZ()};

    std::printf("  surfaces=%d triangles=%d  closeShape: surface=%.4fs mesh=%.4fs\n", surf.GetNsurfaces(),
                mesh.GetNfacets(), surfCloseSeconds, meshCloseSeconds);

    // Label every measurement with whether its subject is a closed manifold at all.
    const auto reliability = surf.GetNavigationReliability();
    const char* reliabilityName = O2BVHSurfaceSolid::GetNavigationReliabilityName(reliability);
    const bool navigable = surf.IsNavigable();
    std::printf("  navigation: %s%s  (boundary=%d non-manifold=%d reversed=%d)\n", reliabilityName,
                navigable ? "" : "  *** UNRELIABLE: results below are not a measurement of accuracy ***",
                surf.GetBoundaryEdgeCount(), surf.GetNonManifoldEdgeCount(), surf.GetReversedEdgeCount());
    // The same boundary measured as curves, in cm. The isolation is how alone the loneliest chord
    // is, *not* a seam width; the chord resolution is next to it because it is what widens the
    // band each chord is matched in, over the declared tolerance.
    std::printf(
      "  rim isolation: max %.3g cm (chord resolution %.3g cm, declared tolerance %.3g cm); rims %d "
      "(matched=%d boundary=%d non-manifold=%d reversed=%d), open %.3g of %.3g cm\n",
      surf.GetMaxRimIsolation(), surf.GetRimChordResolution(), surf.GetRimMatchTolerance(), surf.GetRimCount(),
      surf.GetMatchedRimCount(), surf.GetBoundaryRimCount(), surf.GetNonManifoldRimCount(),
      surf.GetReversedRimCount(), surf.GetUnmatchedRimLength(), surf.GetTotalRimLength());
    // Sidecar v3: closure decided by edge *identity* rather than by proximity. The
    // deviation is a measured cm number and deliberately not a verdict -- it says how far the two
    // faces that provably share an edge actually are, which is the first defensible answer this
    // project has had to that question. Always in --json; on stdout only when asked, because a
    // 19-part run is already dense.
    if (opt.edgeIdentity) {
      if (surf.HasEdgeIdentity()) {
        std::printf(
          "  edge identity: %d source edge(s) (shared=%d boundary=%d non-manifold=%d "
          "reversed=%d degenerate=%d), max shared-edge deviation %.4g cm\n",
          surf.GetSourceEdgeCount(), surf.GetSharedSourceEdgeCount(),
          surf.GetBoundarySourceEdgeCount(), surf.GetNonManifoldSourceEdgeCount(),
          surf.GetReversedSourceEdgeCount(), surf.GetDegenerateSourceEdgeCount(),
          surf.GetMaxSharedEdgeDeviation());
      } else {
        std::printf("  edge identity: absent (sidecar predates v3); closure fell back to proximity\n");
      }
    }
    // Name the offending rims; the line above gives only their count and length.
    json rimsJson = json::array();
    for (const auto& rim : surf.GetRimReports()) {
      const char* stateName = O2BVHSurfaceSolid::GetNavigationReliabilityName(rim.state);
      const bool clean = rim.state == O2BVHSurfaceSolid::NavigationReliability::Reliable;
      if (opt.allRims || !clean) {
        std::printf(
          "    rim face=%d loop=%d %s %s: %d chords, %.4g cm (%d chords / %.4g cm unmatched); "
          "loneliest chord %.3g cm from face %d at (%.4g, %.4g, %.4g)\n",
          rim.surface, rim.rimOnSurface, rim.closed ? "closed" : "OPEN-CHAIN", stateName, rim.chords,
          rim.length, rim.unmatchedChords, rim.unmatchedLength, rim.maxIsolation, rim.maxIsolationFace,
          rim.maxIsolationPoint[0], rim.maxIsolationPoint[1], rim.maxIsolationPoint[2]);
      }
      rimsJson.push_back({{"face", rim.surface},
                          {"loop", rim.rimOnSurface},
                          {"state", stateName},
                          {"closed", rim.closed},
                          {"chords", rim.chords},
                          {"unmatchedChords", rim.unmatchedChords},
                          {"length", rim.length},
                          {"unmatchedLength", rim.unmatchedLength},
                          {"maxIsolation", rim.maxIsolation},
                          {"maxIsolationFace", rim.maxIsolationFace},
                          {"maxIsolationPoint", rim.maxIsolationPoint}});
    }
    if (!navigable) {
      unreliableParts.push_back(part.id + " (" + reliabilityName + ")");
    }

    SampleConfig cfg;
    cfg.nBulk = opt.points;
    cfg.nBoundary = opt.points;
    cfg.nInside = std::max(1, opt.points / 2);
    cfg.nOutsideRays = opt.rays;
    cfg.nInsideRays = std::max(1, opt.rays / 2);
    cfg.seed = opt.seed;
    const SampleSet samples = opt.loadSamples.empty() ? generateSamples(reference, bboxMin, bboxMax, cfg)
                                                      : readSamples(opt.loadSamples, part.id);

    long long candidatesSampled = 0;
    const size_t nProbe = std::min<size_t>(200, samples.outsideRays.size());
    for (size_t i = 0; i < nProbe; ++i) {
      const auto& r = samples.outsideRays[i];
      const int n = surf.CountBVHRayCandidates(r.origin, r.dir);
      if (n > 0) {
        candidatesSampled += n;
      }
    }
    std::printf("  BVH ray candidates: sum=%lld over %zu probe rays\n", candidatesSampled, nProbe);

    json partJson;
    partJson["id"] = part.id;
    partJson["model"] = part.model;
    partJson["nSurfaces"] = surf.GetNsurfaces();
    partJson["nTriangles"] = mesh.GetNfacets();
    partJson["closeShapeSecondsSurface"] = surfCloseSeconds;
    partJson["closeShapeSecondsMesh"] = meshCloseSeconds;
    partJson["bvhRayCandidatesSampled"] = candidatesSampled;
    partJson["bvhRayCandidatesProbeRays"] = nProbe;
    partJson["navigation"] = {{"reliability", reliabilityName},
                              {"navigable", navigable},
                              {"boundaryEdges", surf.GetBoundaryEdgeCount()},
                              {"nonManifoldEdges", surf.GetNonManifoldEdgeCount()},
                              {"reversedEdges", surf.GetReversedEdgeCount()},
                              {"maxRimIsolation", surf.GetMaxRimIsolation()},
                              {"rimChordResolution", surf.GetRimChordResolution()},
                              {"rimMatchTolerance", surf.GetRimMatchTolerance()},
                              {"totalRimLength", surf.GetTotalRimLength()},
                              {"unmatchedRimLength", surf.GetUnmatchedRimLength()},
                              {"rims", surf.GetRimCount()},
                              {"matchedRims", surf.GetMatchedRimCount()},
                              {"boundaryRims", surf.GetBoundaryRimCount()},
                              {"nonManifoldRims", surf.GetNonManifoldRimCount()},
                              {"reversedRims", surf.GetReversedRimCount()},
                              {"hasEdgeIdentity", surf.HasEdgeIdentity()},
                              {"sourceEdges", surf.GetSourceEdgeCount()},
                              {"sharedSourceEdges", surf.GetSharedSourceEdgeCount()},
                              {"boundarySourceEdges", surf.GetBoundarySourceEdgeCount()},
                              {"nonManifoldSourceEdges", surf.GetNonManifoldSourceEdgeCount()},
                              {"reversedSourceEdges", surf.GetReversedSourceEdgeCount()},
                              {"degenerateSourceEdges", surf.GetDegenerateSourceEdgeCount()},
                              {"maxSharedEdgeDeviation", surf.GetMaxSharedEdgeDeviation()},
                              {"rimDetail", rimsJson}};

    std::vector<Point3D> allPoints = samples.bulkPoints;
    allPoints.insert(allPoints.end(), samples.boundaryPoints.begin(), samples.boundaryPoints.end());
    allPoints.insert(allPoints.end(), samples.insidePoints.begin(), samples.insidePoints.end());
    const std::array<size_t, 3> categorySizes{samples.bulkPoints.size(), samples.boundaryPoints.size(),
                                              samples.insidePoints.size()};

    if (!opt.dumpSamples.empty()) {
      writeSamples(opt.dumpSamples, part.id, samples);
    }

    // Ground-truth validation, when the oracle has answered this part. Kept separate from the
    // mesh comparison below rather than replacing it: the mesh columns stay comparable with every
    // measurement recorded so far, while these columns are the ones a gate can be written against.
    if (!opt.refAnswers.empty()) {
      const OracleAnswers oracle = loadOracleAnswers(opt.refAnswers, part.id);
      if (oracle.has) {
        ValidationOptions oracleOpt;
        // The band is now the model's own declared tolerance instead of a guessed mesh sagitta.
        // A floor keeps a perfectly-toleranced synthetic fixture from demanding bit equality.
        oracleOpt.meshBand = std::max(oracle.tolerance, oracleOpt.distanceTolerance);
        const auto boundaryDistance =
          mergeCategories<double>(oracle.boundaryDistance, categorySizes, -1.);
        const auto containsState = mergeCategories<int>(oracle.containsState, categorySizes, -1);

        std::printf("  oracle: %s tolerance=%.3g capacity=%.6g cm^3 (band=%.3g)\n",
                    oracle.valid ? "valid" : "*** NOT BRepCheck-VALID ***", oracle.tolerance,
                    oracle.capacity, oracleOpt.meshBand);

        // The historical block, unchanged in content: the exact-surface representation's columns
        // under `oracle`, printed with the same "O:" labels. Everything written here before this
        // refactor is still written here, by the same code, so the existing path is inert.
        json oracleJson = scoreAgainstOracle(candidate, oracle, oracleOpt, allPoints, containsState,
                                             boundaryDistance, samples, opt.only, "O", "oracle");
        partJson["oracle"] = oracleJson;

        // New: the same four columns for every other representation the part has, against the
        // same answers. This is what makes a CSG-emitted or tessellated part scoreable at all,
        // and it is the shape the tiered coverage scorecard needs -- parallel columns, not
        // alternatives behind a flag.
        json representationsJson = json::array();
        for (const auto& rep : representations) {
          const bool isSurface = rep.surfaceSolid != nullptr;
          json repJson;
          repJson["name"] = rep.name;
          repJson["source"] = rep.source;
          repJson["shapeClass"] = rep.shape->ClassName();
          if (rep.primitives >= 0) {
            repJson["primitives"] = rep.primitives;
            repJson["primitiveKind"] = rep.primitiveKind;
          }
          const auto capacityKind = capacityKindOf(rep);
          repJson["capacityMethod"] = capacityKind.method;
          repJson["capacityComparable"] = capacityKind.comparable;
          // The frame check, per representation: a candidate whose box does not sit where the
          // oracle's box sits is not being asked the same questions the oracle answered.
          repJson["bboxDeviationFromOracle"] =
            bboxDeviationFromOracle(rep.shape, oracle, rep.placement);
          // The placement, mirrored into the scorecard as a 3x4 row-major [R | t] so that a
          // Python consumer never has to open the .root file, and null when there is none.
          if (rep.placement != nullptr) {
            const double* rot = rep.placement->GetRotationMatrix();
            const double* tr = rep.placement->GetTranslation();
            repJson["placement"] = {{rot[0], rot[1], rot[2], tr[0]},
                                    {rot[3], rot[4], rot[5], tr[1]},
                                    {rot[6], rot[7], rot[8], tr[2]}};
          } else {
            repJson["placement"] = nullptr;
          }

          // Closure / rims / NavigationReliability are O2BVHSurfaceSolid concepts. A
          // TGeoCompositeShape has neither, and a triangle mesh has a different notion entirely,
          // so those keys exist only where the question has an answer. `closureApplicable`
          // records the decision explicitly instead of leaving a reader to infer it from an
          // absent field.
          repJson["closureApplicable"] = isSurface;
          if (isSurface) {
            repJson["reliability"] = reliabilityName;
            repJson["navigable"] = navigable;
          } else if (rep.name == "mesh") {
            // O2Tessellated's own, differently-named watertightness statement. Deliberately not
            // called `navigable`: it is a property of the triangle soup, decided by half-edge
            // counting over chords, and it is not the same claim.
            repJson["meshClosedBody"] = mesh.IsClosedBody();
          }

          if (isSurface) {
            // Already computed above; scoring the same shape twice would only cost time and
            // invite the two copies to drift.
            repJson["oracle"] = oracleJson;
          } else {
            std::printf("  --- representation '%s' (%s) against the same oracle answers ---\n",
                        rep.name.c_str(), rep.shape->ClassName());
            repJson["oracle"] = scoreAgainstOracle(rep.shape, oracle, oracleOpt, allPoints,
                                                   containsState, boundaryDistance, samples,
                                                   opt.only, "R:" + rep.name,
                                                   "oracle[" + rep.name + "]", rep.placement);
          }
          repJson["disagreements"] = countDisagreements(repJson["oracle"]);
          representationsJson.push_back(std::move(repJson));
        }
        partJson["representations"] = std::move(representationsJson);
      }
    }

    if (opt.only.count("contains")) {
      auto v = validateContains(candidate, reference, allPoints);
      printValidation("contains", v);
      auto tc = timeContains(candidate, allPoints, opt.warmup, opt.repeat);
      auto tr = timeContains(reference, allPoints, opt.warmup, opt.repeat);
      printTiming("contains", tc, tr);
      partJson["contains"] = {{"validation", validationToJson(v)},
                              {"timingCandidate", timingToJson(tc)},
                              {"timingReference", timingToJson(tr)}};
    }
    if (opt.only.count("distout")) {
      auto v = validateDistFromOutside(candidate, reference, samples.outsideRays);
      printValidation("distout", v);
      auto tc = timeDistFromOutside(candidate, samples.outsideRays, opt.warmup, opt.repeat);
      auto tr = timeDistFromOutside(reference, samples.outsideRays, opt.warmup, opt.repeat);
      printTiming("distout", tc, tr);
      // the all-surfaces baseline: what the BVH traversal buys over visiting every patch
      auto tl = timeRayKernel(samples.outsideRays, opt.warmup, opt.repeat,
                              [&surf](const Point3D& o, const Point3D& d) {
                                return surf.DistFromOutside_Loop(o.data(), d.data());
                              });
      printLoopSpeedup("distout", tc, tl);
      partJson["distout"] = {{"validation", validationToJson(v)},
                             {"timingCandidate", timingToJson(tc)},
                             {"timingReference", timingToJson(tr)},
                             {"timingCandidateLoop", timingToJson(tl)}};
    }
    if (opt.only.count("distin")) {
      auto v = validateDistFromInside(candidate, reference, samples.insideRays);
      printValidation("distin", v);
      auto tc = timeDistFromInside(candidate, samples.insideRays, opt.warmup, opt.repeat);
      auto tr = timeDistFromInside(reference, samples.insideRays, opt.warmup, opt.repeat);
      printTiming("distin", tc, tr);
      auto tl = timeRayKernel(samples.insideRays, opt.warmup, opt.repeat,
                              [&surf](const Point3D& o, const Point3D& d) {
                                return surf.DistFromInside_Loop(o.data(), d.data());
                              });
      printLoopSpeedup("distin", tc, tl);
      partJson["distin"] = {{"validation", validationToJson(v)},
                            {"timingCandidate", timingToJson(tc)},
                            {"timingReference", timingToJson(tr)},
                            {"timingCandidateLoop", timingToJson(tl)}};
    }
    if (opt.only.count("safety")) {
      // Never compared against each other (see ground rules): each shape's Safety() is checked
      // against its own DistFrom{Inside,Outside} contract independently.
      auto vc = validateSafety(candidate, allPoints);
      auto vr = validateSafety(reference, allPoints);
      printValidation("safety(cand)", vc);
      printValidation("safety(ref)", vr);
      auto tc = timeSafety(candidate, allPoints, opt.warmup, opt.repeat);
      auto tr = timeSafety(reference, allPoints, opt.warmup, opt.repeat);
      printTiming("safety", tc, tr);
      partJson["safety"] = {{"validationCandidate", validationToJson(vc)},
                            {"validationReference", validationToJson(vr)},
                            {"timingCandidate", timingToJson(tc)},
                            {"timingReference", timingToJson(tr)}};
    }

    if (opt.loopCrosscheck) {
      // Independent of the tessellated reference entirely: separates
      // BVH/traversal bugs from surface-kernel bugs.
      // The distance twins must agree *exactly*, not within a tolerance: both take a
      // minimum over the same hits from the same kernels and differ only in which surfaces the
      // BVH lets them skip, so any difference at all is a traversal or pruning bug.
      size_t containsAgree = 0;
      size_t crossingDumps = 0;
      constexpr size_t kMaxCrossingDumps = 3;
      std::vector<O2BVHSurfaceSolid::ContainsCrossing> bvhCrossings;
      std::vector<O2BVHSurfaceSolid::ContainsCrossing> loopCrossings;
      for (const auto& p : allPoints) {
        if (surf.Contains(p.data()) == surf.Contains_Loop(p.data())) {
          ++containsAgree;
          continue;
        }
        // A parity disagreement between two paths over the same kernels means the two hit lists
        // differ. Print them: the difference is the diagnosis, and guessing at it has already
        // cost this project one three-item plan built on a wrong premise.
        if (crossingDumps++ >= kMaxCrossingDumps) {
          continue;
        }
        surf.DescribeContainsCrossings(p, bvhCrossings, loopCrossings);
        std::printf("      BVH!=Loop at (%.9g,%.9g,%.9g): BVH=%d (%zu crossings) Loop=%d (%zu crossings)\n",
                    p[0], p[1], p[2], static_cast<int>(surf.Contains(p.data())), bvhCrossings.size(),
                    static_cast<int>(surf.Contains_Loop(p.data())), loopCrossings.size());
        const size_t nShow = std::max(bvhCrossings.size(), loopCrossings.size());
        for (size_t i = 0; i < nShow; ++i) {
          const char* bvhKind = i < bvhCrossings.size()
                                  ? (bvhCrossings[i].normalAlignment < 0. ? "ENTER" : "EXIT ")
                                  : "-----";
          const char* loopKind = i < loopCrossings.size()
                                   ? (loopCrossings[i].normalAlignment < 0. ? "ENTER" : "EXIT ")
                                   : "-----";
          const double bvhT = i < bvhCrossings.size() ? bvhCrossings[i].distance : -1.;
          const double loopT = i < loopCrossings.size() ? loopCrossings[i].distance : -1.;
          std::printf("        [%2zu] BVH %s t=%-18.12g   Loop %s t=%-18.12g%s\n", i, bvhKind, bvhT,
                      loopKind, loopT,
                      (i < bvhCrossings.size() && i < loopCrossings.size() &&
                       std::fabs(bvhT - loopT) > 1.e-12)
                        ? "   <-- differs"
                        : "");
        }
      }
      std::printf("  loop-crosscheck contains: BVH==Loop for %zu/%zu points\n", containsAgree, allPoints.size());
      partJson["loopCrosscheckContains"] = {{"agree", containsAgree}, {"total", allPoints.size()}};

      size_t outAgree = 0;
      double worstOutDeviation = 0.;
      for (const auto& r : samples.outsideRays) {
        const double bvh = surf.DistFromOutside(r.origin.data(), r.dir.data(), 3);
        const double loop = surf.DistFromOutside_Loop(r.origin.data(), r.dir.data());
        if (bvh == loop) {
          ++outAgree;
        } else {
          worstOutDeviation = std::max(worstOutDeviation, std::fabs(bvh - loop));
        }
      }
      std::printf("  loop-crosscheck distout : BVH==Loop for %zu/%zu rays (worstDev=%.6g)\n", outAgree,
                  samples.outsideRays.size(), worstOutDeviation);
      partJson["loopCrosscheckDistOutside"] = {
        {"agree", outAgree}, {"total", samples.outsideRays.size()}, {"worstDeviation", worstOutDeviation}};

      size_t inAgree = 0;
      double worstInDeviation = 0.;
      for (const auto& r : samples.insideRays) {
        const double bvh = surf.DistFromInside(r.origin.data(), r.dir.data(), 3);
        const double loop = surf.DistFromInside_Loop(r.origin.data(), r.dir.data());
        if (bvh == loop) {
          ++inAgree;
        } else {
          worstInDeviation = std::max(worstInDeviation, std::fabs(bvh - loop));
        }
      }
      std::printf("  loop-crosscheck distin  : BVH==Loop for %zu/%zu rays (worstDev=%.6g)\n", inAgree,
                  samples.insideRays.size(), worstInDeviation);
      partJson["loopCrosscheckDistInside"] = {
        {"agree", inAgree}, {"total", samples.insideRays.size()}, {"worstDeviation", worstInDeviation}};
    }

    if (opt.pruningAb) {
      // Prices the ray tmax tightening: the same rays run with it on and off, reporting both the
      // surface patches the traversal actually handed to the leaf callback and the wall time. The
      // answers must be bit-identical -- the switch is a cost knob, never a semantic one, and a
      // mismatch here is a bug in the tightening rather than a measurement.
      json pruningJson;
      size_t identical = 0;
      std::vector<double> prunedValues;
      prunedValues.reserve(samples.outsideRays.size());

      O2BVHSurfaceSolid::SetRayTMaxPruning(true);
      O2BVHSurfaceSolid::ResetRayCandidateCounter();
      for (const auto& r : samples.outsideRays) {
        prunedValues.push_back(surf.DistFromOutside(r.origin.data(), r.dir.data(), 3));
      }
      const long long prunedCandidates = O2BVHSurfaceSolid::GetRayCandidateCount();
      auto tPruned = timeDistFromOutside(candidate, samples.outsideRays, opt.warmup, opt.repeat);

      O2BVHSurfaceSolid::SetRayTMaxPruning(false);
      O2BVHSurfaceSolid::ResetRayCandidateCounter();
      for (size_t i = 0; i < samples.outsideRays.size(); ++i) {
        const auto& r = samples.outsideRays[i];
        if (surf.DistFromOutside(r.origin.data(), r.dir.data(), 3) == prunedValues[i]) {
          ++identical;
        }
      }
      const long long unprunedCandidates = O2BVHSurfaceSolid::GetRayCandidateCount();
      auto tUnpruned = timeDistFromOutside(candidate, samples.outsideRays, opt.warmup, opt.repeat);
      O2BVHSurfaceSolid::SetRayTMaxPruning(true);

      const double candidateRatio =
        unprunedCandidates > 0 ? static_cast<double>(prunedCandidates) / static_cast<double>(unprunedCandidates) : 0.;
      const double speedup = tPruned.nsPerCall > 0. ? tUnpruned.nsPerCall / tPruned.nsPerCall : 0.;
      std::printf("  tmax-pruning A/B (distout, %zu rays): identical=%zu/%zu\n", samples.outsideRays.size(), identical,
                  samples.outsideRays.size());
      std::printf("      candidates: pruned=%lld  unpruned=%lld  (%.1f%% of the work)\n", prunedCandidates,
                  unprunedCandidates, 100. * candidateRatio);
      std::printf("      time      : pruned=%9.1f ns/call  unpruned=%9.1f ns/call  speedup=%.2fx\n",
                  tPruned.nsPerCall, tUnpruned.nsPerCall, speedup);

      pruningJson["identical"] = identical;
      pruningJson["total"] = samples.outsideRays.size();
      pruningJson["candidatesPruned"] = prunedCandidates;
      pruningJson["candidatesUnpruned"] = unprunedCandidates;
      pruningJson["timingPruned"] = timingToJson(tPruned);
      pruningJson["timingUnpruned"] = timingToJson(tUnpruned);
      partJson["tmaxPruningAB"] = std::move(pruningJson);
    }

    jsonReport.push_back(std::move(partJson));
  }

  // Repeated at the end because per-part lines scroll away in a 19-part run, and because the whole
  // point of item 4 is that no future reader can attribute an "unexplained" column to mesh
  // chording without first seeing whether the subject was a closed manifold at all.
  if (!unreliableParts.empty()) {
    std::printf(
      "\n*** %zu of %zu part(s) are NOT navigable; their accuracy columns above measure an\n"
      "*** undefined answer, not the exact solid's error.\n",
      unreliableParts.size(), parts.size());
    for (const auto& id : unreliableParts) {
      std::printf("***   %s\n", id.c_str());
    }
  } else {
    std::printf("\nAll %zu part(s) closed consistently oriented manifolds: navigation results are meaningful.\n",
                parts.size());
  }

  // The tiered scorecard, in its most compact form: how many disagreements outside tolerance each
  // representation of each part has. Printed here because the per-part blocks scroll away, and
  // because "which representation would have accepted this part" is the question the converter's
  // dispatch policy will be written against.
  bool anyRepresentations = false;
  for (const auto& partJson : jsonReport) {
    anyRepresentations = anyRepresentations || partJson.contains("representations");
  }
  if (anyRepresentations) {
    std::printf("\n=== REPRESENTATION SCORECARD (disagreements outside tolerance, all four columns) ===\n");
    for (const auto& partJson : jsonReport) {
      if (!partJson.contains("representations")) {
        continue;
      }
      std::printf("  %-46s", partJson.at("id").get<std::string>().c_str());
      for (const auto& rep : partJson.at("representations")) {
        const double capacityDeviation =
          rep.at("oracle").value("capacityRelativeDeviation", 0.);
        const bool capacityComparable = rep.value("capacityComparable", false);
        char capacityText[32];
        if (capacityComparable) {
          std::snprintf(capacityText, sizeof(capacityText), "%.2g", std::fabs(capacityDeviation));
        } else {
          std::snprintf(capacityText, sizeof(capacityText), "n/a");
        }
        std::printf("  %s=%zu (cap %s)", rep.at("name").get<std::string>().c_str(),
                    rep.value("disagreements", size_t{0}), capacityText);
      }
      std::printf("\n");
    }
  }

  if (!opt.jsonOut.empty()) {
    std::ofstream out(opt.jsonOut);
    out << jsonReport.dump(1);
    std::printf("\nWrote %s\n", opt.jsonOut.c_str());
  }

  return 0;
}
