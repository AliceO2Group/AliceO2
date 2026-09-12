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

/// \file runXRayBenchmark.cxx
/// \brief X-ray / geantino transport benchmark: ordered crossing lists, by stepping.
///
/// Built as `o2-bench-cadsupport-xray`.
///
/// WHY THIS EXISTS, in one paragraph. Everything the oracle gate measures is a *single-shot*
/// query: from a sampled point, how far to the surface. A transport loop is different in kind --
/// step, land *on* the boundary, step again from there -- and that is where geometry navigation
/// actually fails: zero-length steps, ping-ponging on a face, a particle that enters and never
/// exits, a crossing found twice, a step that overshoots into the next volume. None of those can
/// be expressed as a disagreement on `distout` from an interior sample, so the existing gate is
/// structurally blind to all of them. This benchmark shoots a structured parallel-beam raster
/// through a part and produces, per ray, the ORDERED CROSSING LIST -- the sequence of entry/exit
/// distances -- by stepping, two independent ways, and compares the lists (not aggregates)
/// against OpenCascade.
///
/// TWO STEPPING MODES, and the reason both exist:
///   (a) `shape`  -- a direct shape-API loop: Contains() to establish the starting state, then
///                   alternating DistFromOutside()/DistFromInside(), advancing the point, until
///                   the ray leaves the raster window. Depends on nothing but the shape.
///   (b) `nav`    -- the real TGeoNavigator: the part placed in a TGeoVolume inside a minimal
///                   world, transported with FindNextBoundaryAndStep(). The production path.
/// If (a) and (b) disagree, that isolates *the shape* from *the navigator* immediately; with only
/// (b) one cannot tell which of the two lied. Both are always reported.
///
/// THREE-STAGE ROUND TRIP, mirroring the oracle gate:
///   1. `--dump-rays D`      writes D/xrays_<part>.json: the raster window and every ray.
///   2. `xrayOracle.py`      answers exactly those rays from the part's .brep, in OpenCascade,
///                           into D/crossings_<part>.json.
///   3. `--ref-crossings D`  steps both modes over the same rays and scores the lists.
/// The rays are written and read rather than regenerated on both sides for the same reason the
/// sample sets are: a comparison is only evidence if both sides answered the same question.
///
/// NOT REQUIRED: a tessellated mesh. The raster is structured and deterministic, so unlike
/// `generateSamples()` nothing here rejection-samples through `O2Tessellated`. That is what makes
/// this instrument runnable on a model whose meshing does not fit in memory.

#include "RepresentationBench.h"
#include "XRayTransport.h"

#include "CADSupport/O2SolidHarness.h"
#include "CADSupport/O2BVHSurfaceSolid.h"
#include "DetectorsBase/O2Tessellated.h"
#include "CADSupport/O2FlatCSG.h"
#include "CADSupport/O2SurfaceSolidIO.h"

#include "TGeoBBox.h"
#include "TGeoManager.h"
#include "TGeoMaterial.h"
#include "TGeoMatrix.h"
#include "TGeoMedium.h"
#include "TGeoNavigator.h"
#include "TGeoNode.h"
#include "TGeoSphere.h"
#include "TGeoTube.h"
#include "TGeoVolume.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using json = nlohmann::json;
using namespace o2::cad;
using namespace o2::cad::harness;
using namespace o2::cad::xray;
using namespace o2::cad::bench;
using o2::base::O2Tessellated;

namespace
{

// The xrays_/crossings_ JSON contract shared with Detectors/CADSupport/validation/xrayOracle.py. Bump on both
// sides together; the oracle refuses a version it does not speak rather than guessing.
constexpr int kXRayFormatVersion = 2;

json comparisonToJson(const ListComparison& c)
{
  return json{{"rays", c.rays},
              {"raysIdentical", c.raysIdentical},
              {"raysStructural", c.raysStructural},
              {"matched", c.matched},
              {"displacedCrossings", c.displaced},
              {"missingCrossings", c.missing},
              {"extraCrossings", c.extra},
              {"kindMismatch", c.kindMismatch},
              {"worstDeltaT", c.worstDeltaT},
              {"worstOrigin", {c.worstOrigin[0], c.worstOrigin[1], c.worstOrigin[2]}},
              {"worstDir", {c.worstDir[0], c.worstDir[1], c.worstDir[2]}},
              {"worstReason", c.worstReason}};
}

json robustnessToJson(const Robustness& r)
{
  return json{{"rays", r.rays},
              {"raysWithCrossings", r.raysWithCrossings},
              {"crossings", r.crossings},
              {"steps", r.steps},
              {"zeroLengthSteps", r.zeroLengthSteps},
              {"nonAdvancingSteps", r.nonAdvancingSteps},
              {"unstickPushes", r.unstickPushes},
              {"iterationCapHits", r.iterationCapHits},
              {"unterminated", r.unterminated},
              {"oddCrossingLists", r.oddCrossingLists},
              {"nonAlternating", r.nonAlternating},
              {"duplicateCrossings", r.duplicateCrossings},
              {"parityMismatchIntervals", r.parityMismatchIntervals},
              {"parityMismatchNearBoundary", r.parityMismatchNearBoundary},
              {"originInside", r.originInside},
              {"boundaryWithoutTransition", r.boundaryWithoutTransition},
              {"originOutsideWorld", r.originOutsideWorld},
              {"insideLengthCm", r.insideLength},
              {"seconds", r.seconds}};
}

// ------------------------------------------------------------------------------------------
// Mode (b): the real TGeoNavigator
// ------------------------------------------------------------------------------------------

/// One part in a minimal world, transported with FindNextBoundaryAndStep().
///
/// The crossing distance is taken as (projection of the point *before* the step onto the ray) +
/// GetStep(), rather than by accumulating GetStep(): the navigator moves the point a little past
/// each boundary, and reprojecting absorbs that push instead of letting it accumulate.
class NavigatorTransport
{
 public:
  /// Builds the world INSIDE the caller's manager, deliberately.
  ///
  /// Constructing a second TGeoManager here is what the first version did, and it segfaulted:
  /// `TGeoManager`'s constructor DELETES the existing `gGeoManager`, and that manager owns the
  /// shape being handed in (TGeoShape registers itself in `gGeoManager`'s shape list on
  /// construction). The world therefore has to be built in the manager the shape already belongs
  /// to, and everything created here is freed with it.
  /// `placement` is the shape's own frame expressed in the part frame, or null when they are the
  /// same. Mode (b) carries it on the NODE rather than transforming the rays, which is the whole
  /// point of having a navigator: the rays stay in the part frame, ROOT performs the transform it
  /// would perform in production, and mode (a) -- which transforms the rays by hand -- becomes an
  /// independent check of it rather than a restatement.
  NavigatorTransport(TGeoManager* manager, TGeoShape* shape, const Point3D& bboxMin,
                     const Point3D& bboxMax, const TGeoMatrix* placement = nullptr)
  {
    mManager = manager;
    auto* material = new TGeoMaterial("Vacuum", 0., 0., 0.);
    auto* medium = new TGeoMedium("Vacuum", 1, material);
    double half[3];
    double centre[3];
    for (int k = 0; k < 3; ++k) {
      centre[k] = 0.5 * (bboxMax[k] + bboxMin[k]);
      half[k] = 0.5 * (bboxMax[k] - bboxMin[k]) + 0.05 * (bboxMax[k] - bboxMin[k]) + 0.1;
    }
    auto* worldBox = new TGeoBBox("xrayWorld", half[0], half[1], half[2], centre);
    mWorld = new TGeoVolume("TOP", worldBox, medium);
    mPart = new TGeoVolume("PART", shape, medium);
    // Identity unless the shape carries a placement, in which case this is where it is applied.
    mWorld->AddNode(mPart, 1, placement != nullptr ? new TGeoHMatrix(*placement) : nullptr);
    mManager->SetTopVolume(mWorld);
    mManager->CloseGeometry();
    mManager->SetNsegments(80);
    mNavigator = mManager->GetCurrentNavigator();
  }

  /// Owns nothing: the manager handed in outlives this object and frees the world with itself.
  ~NavigatorTransport() = default;

  NavigatorTransport(const NavigatorTransport&) = delete;
  NavigatorTransport& operator=(const NavigatorTransport&) = delete;

  bool valid() const { return mNavigator != nullptr; }

  std::vector<Crossing> transport(const Point3D& origin, const Point3D& dir, double tMax,
                                  const StepConfig& cfg, Robustness& stats)
  {
    std::vector<Crossing> crossings;
    mNavigator->InitTrack(origin.data(), dir.data());
    if (mNavigator->IsOutside()) {
      // The world is built to contain every ray of the raster, so this cannot fire on a correct
      // configuration -- and it gets its own counter precisely so that a wrong one is never
      // mistaken for a geometry defect.
      ++stats.originOutsideWorld;
      return crossings;
    }
    bool inPart = (mNavigator->GetCurrentVolume() == mPart);
    if (inPart) {
      ++stats.originInside;
    }
    int iter = 0;
    for (; iter < cfg.maxIter; ++iter) {
      const double* before = mNavigator->GetCurrentPoint();
      double tBefore = 0.;
      for (int k = 0; k < 3; ++k) {
        tBefore += (before[k] - origin[k]) * dir[k];
      }
      mNavigator->FindNextBoundaryAndStep(TGeoShape::Big(), kFALSE);
      const double step = mNavigator->GetStep();
      ++stats.steps;
      const double tCross = tBefore + step;
      if (step <= cfg.zeroStep) {
        ++stats.zeroLengthSteps;
      }
      if (!(tCross > tBefore)) {
        ++stats.nonAdvancingSteps;
      }
      if (mNavigator->IsOutside() || tCross > tMax || !(step < TGeoShape::Big())) {
        break;
      }
      const bool nowIn = (mNavigator->GetCurrentVolume() == mPart);
      if (nowIn != inPart) {
        crossings.push_back({tCross, nowIn ? +1 : -1});
        inPart = nowIn;
      } else {
        ++stats.boundaryWithoutTransition;
      }
    }
    if (iter >= cfg.maxIter) {
      ++stats.iterationCapHits;
    }
    if (inPart) {
      ++stats.unterminated;
    }
    return crossings;
  }

 private:
  TGeoManager* mManager = nullptr;
  TGeoVolume* mWorld = nullptr;
  TGeoVolume* mPart = nullptr;
  TGeoNavigator* mNavigator = nullptr;
};

// ------------------------------------------------------------------------------------------
// Reading a crossing list, and comparing two of them
// ------------------------------------------------------------------------------------------

// ------------------------------------------------------------------------------------------
// The raster
// ------------------------------------------------------------------------------------------
//
// A structured parallel-beam raster, not Monte Carlo. Cell centres of an N x N lattice over the
// raster window, one beam per axis. Structured wins for two independent reasons: the chord
// integral converges far better than random sampling (boundary cells are the whole error budget
// and their count grows as N rather than N^2), and a lattice deliberately produces the grazing,
// edge-on and vertex-on rays a random direction essentially never generates -- which is where a
// transport loop stalls.

// ------------------------------------------------------------------------------------------
// Options, part collection, IO
// ------------------------------------------------------------------------------------------

struct Options {
  std::string db;
  std::string explicitSurfaces;
  std::string explicitFacets;
  std::string explicitShape;
  std::string explicitFlatCSG;
  /// `O2FlatCSG::SetSplitDepth` / `SetMinBoxFraction` for every flat subject, or < 0 / < 0 to
  /// leave the class defaults alone. These exist so the split knobs can be swept from outside
  /// the class, which is how their defaults were chosen (Design_FlatCSGSolid.md section 9).
  int flatSplitDepth = -1;
  double flatMinBoxFraction = -1.;
  std::string partsPattern;
  int raster = 48;
  std::string axesSpec = "xyz";
  /// Transverse padding of the raster window over the part's bounding box, cm. Kept absolute and
  /// small: it is a first-order systematic on the chord volume (see buildRaster). It exists only
  /// to cover the fact that a tessellated bounding box is INSCRIBED -- measured at 1e-4 to 1e-3 cm
  /// on these corpora -- so a zero margin would clip the true solid's silhouette.
  double margin = 1.e-3;
  /// Rotate every beam off its coordinate axis by this many degrees. At 0 the beams are exactly
  /// axis-aligned, which keeps a box's chord integral exact but samples a very special family of
  /// ray/surface configurations; a non-zero tilt makes them generic.
  double tiltDegrees = 0.;
  /// When > 0, replace the axis beams by this many Fibonacci-spiral directions. A parallel-beam
  /// raster is direction-poor and a direction-dependent defect is invisible to it; see
  /// buildFanBeams.
  int fanBeams = 0;
  std::string dumpRays;
  std::string refCrossings;
  std::string jsonOut;
  /// `flatcsg` is deliberately NOT here: a flat part's `shape_*.root` already holds the same
  /// `O2FlatCSG`, so a database run would score one solid twice. Naming `--flatcsg <file>`
  /// adds it, and `--representations` can ask for it by name.
  std::set<std::string> representations = {"surface", "mesh", "shape"};
  bool skipNavigator = false;
  bool selfTest = false;
  /// The representation cost/memory comparison: per-call ns for
  /// the four navigation kernels plus transport, and two memory numbers, per representation, from
  /// ONE shared sample set per part.
  bool perf = false;
  int perfPoints = 4096;
  int perfRays = 4096;
  int perfPasses = 9;
  int perfWarmup = 2;
  /// Comma-separated leaf counts for the synthetic boolean ladder. Needs no database and no model.
  std::string ladderSpec;
  StepConfig step;
};

struct Part {
  std::string id;
  std::string model;
  std::string surfaces;
  std::string facets;
  std::string shape;
  std::string flatcsg;
};

/// The file one named representation of \a part reads. One place, because four call sites
/// used to spell the same three-way conditional and a fourth representation would have made
/// each of them a place to forget it.
const std::string& sourceFor(const Part& part, const std::string& name)
{
  if (name == "surface") {
    return part.surfaces;
  }
  if (name == "mesh") {
    return part.facets;
  }
  if (name == "flatcsg") {
    return part.flatcsg;
  }
  return part.shape;
}

/// Every representation name, in the order the tables print them.
const std::array<std::string, 4>& allRepresentations()
{
  static const std::array<std::string, 4> names{"surface", "mesh", "shape", "flatcsg"};
  return names;
}

std::string deriveSidecarPath(const std::string& surfacesPath, const char* prefixOut,
                              const char* suffixOut)
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
  return dir + prefixOut + stem + suffixOut;
}

bool fileExists(const std::string& path)
{
  if (path.empty()) {
    return false;
  }
  std::ifstream probe(path);
  return static_cast<bool>(probe);
}

/// Must match sanitizePartId() in runSolidHarness.cxx and sanitize_part_id() in the gate scripts.
std::string sanitizePartId(const std::string& id)
{
  std::string out;
  out.reserve(id.size());
  for (const char c : id) {
    out.push_back((std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '.') ? c : '_');
  }
  return out;
}

void printUsage(const char* argv0)
{
  std::cout << "X-ray / geantino transport benchmark -- ordered crossing lists, by stepping.\n\n"
               "Usage: "
            << argv0 << " --db <dir> [--parts <substring>] [--raster N] [--axes xyz]\n"
                        "                          [--dump-rays D] [--ref-crossings D] [--json out.json]\n"
                        "   or: "
            << argv0 << " --surfaces <f> [--facets <f>] [--shape <f>] [--flatcsg <f>]\n"
                        "                          [options as above]\n"
                        "   or: "
            << argv0 << " --self-test\n\n"
                        "  --raster N        N x N parallel rays per beam axis (default 48). Structured, not random:\n"
                        "                    the chord integral converges as the boundary-cell count (~N) rather than\n"
                        "                    as sqrt of the sample count, and a lattice generates the edge-on and\n"
                        "                    vertex-on rays that stall a transport loop.\n"
                        "  --axes xyz        which beam axes to fire (subset of x,y,z; default all three)\n"
                        "  --beams N         fire N Fibonacci-spiral directions instead of the axis beams. A parallel\n"
                        "                    beam is DIRECTION-POOR: three axes are three directions however many rays\n"
                        "                    are fired, and a direction-dependent defect (the torus quartic) is\n"
                        "                    invisible to them. Use this whenever hunting one.\n"
                        "  --tilt DEG        rotate every beam off its axis by DEG (default 0). An axis-aligned beam\n"
                        "                    is a special family of configurations; a tilted one is generic. The known\n"
                        "                    torus quartic defect is invisible at tilt 0 and visible at tilt 12.\n"
                        "  --dump-rays D     write D/xrays_<part>.json (the raster window and every ray) and exit\n"
                        "  --ref-crossings D read D/crossings_<part>.json (Detectors/CADSupport/validation/xrayOracle.py) and score\n"
                        "                    the crossing LISTS against it, per representation, per mode\n"
                        "  --flatcsg <f>     an o2::cad::O2FlatCSG sidecar (flatcsg_*.bin) as its own subject.\n"
                        "                    NOT in the default set -- a flat part's shape_*.root already holds\n"
                        "                    the same solid -- so name it here, or in --representations.\n"
                        "                    This is how the flat halfspace solid is scored against the SAME part\n"
                        "                    emitted as a plain TGeoCompositeShape through --shape: two subjects,\n"
                        "                    one raster, one sample set (Design_FlatCSGSolid.md section 9).\n"
                        "  --flat-split-depth N        override O2FlatCSG::SetSplitDepth on every flat subject\n"
                        "  --flat-min-box-fraction X   override O2FlatCSG::SetMinBoxFraction likewise. The two\n"
                        "                    knobs are swept from here rather than from a test, so the defaults in\n"
                        "                    the header rest on the same instrument that reports the query cost.\n"
                        "  --representations surface,mesh,shape,flatcsg   which to run (default: all present)\n"
                        "  --no-navigator    skip mode (b); mode (a) depends on nothing but the shape\n"
                        "  --perf            the representation cost/memory comparison: per-call ns for Contains,\n"
                        "                    Safety, DistFromOutside and DistFromInside, plus transport ns/ray and\n"
                        "                    ns/crossing, plus structural and measured memory -- for every\n"
                        "                    representation, from ONE shared sample set per part. Warm cache; the\n"
                        "                    reported number is the median over --perf-passes complete passes and the\n"
                        "                    min/max spread is printed with it.\n"
                        "  --perf-points N   query points per part (default 4096)\n"
                        "  --perf-rays N     rays per distance kernel (default 4096)\n"
                        "  --perf-passes N   timed passes (default 9); --perf-warmup N untimed first (default 2)\n"
                        "  --ladder 2,4,8    the synthetic boolean ladder: unions of K TGeoTubes as a left-deep CHAIN\n"
                        "                    and as a BALANCED tree, timed with the same kernels. Needs no database:\n"
                        "                    every genuine boolean in the corpus is a 2-leaf union, so the corpus\n"
                        "                    cannot answer how a composite scales with leaf count and this fixture is\n"
                        "                    what does.\n"
                        "  --push X          distance advanced past a found crossing (cm, default 1e-9 = kRayTolerance)\n"
                        "  --unstick-push X  the nudge a stalled step is repaired with (cm, default 1e-6); every use\n"
                        "                    is counted in `unstickPushes`\n"
                        "  --max-iter N      transport iteration cap per ray (default 512)\n"
                        "  --self-test       analytic self-checks (box, tube, sphere) plus the synthetic controls that\n"
                        "                    prove the comparison can fail. Needs no database and no oracle.\n\n"
                        "Three-stage round trip:\n"
                        "  "
            << argv0 << " --db <db> --dump-rays /tmp/x\n"
                        "  xrayOracle.py --brep <part>.brep --rays /tmp/x/xrays_<part>.json \\\n"
                        "                --out /tmp/x/crossings_<part>.json\n"
                        "  "
            << argv0 << " --db <db> --ref-crossings /tmp/x --json /tmp/x/xray.json\n";
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
    } else if (a == "--flatcsg") {
      opt.explicitFlatCSG = next("--flatcsg");
    } else if (a == "--flat-split-depth") {
      opt.flatSplitDepth = std::stoi(next("--flat-split-depth"));
    } else if (a == "--flat-min-box-fraction") {
      opt.flatMinBoxFraction = std::stod(next("--flat-min-box-fraction"));
    } else if (a == "--parts") {
      opt.partsPattern = next("--parts");
    } else if (a == "--raster") {
      opt.raster = std::stoi(next("--raster"));
    } else if (a == "--axes") {
      opt.axesSpec = next("--axes");
    } else if (a == "--beams") {
      opt.fanBeams = std::stoi(next("--beams"));
    } else if (a == "--tilt") {
      opt.tiltDegrees = std::stod(next("--tilt"));
    } else if (a == "--margin") {
      opt.margin = std::stod(next("--margin"));
    } else if (a == "--dump-rays") {
      opt.dumpRays = next("--dump-rays");
    } else if (a == "--ref-crossings") {
      opt.refCrossings = next("--ref-crossings");
    } else if (a == "--json") {
      opt.jsonOut = next("--json");
    } else if (a == "--representations") {
      opt.representations = splitCsv(next("--representations"));
    } else if (a == "--no-navigator") {
      opt.skipNavigator = true;
    } else if (a == "--perf") {
      opt.perf = true;
    } else if (a == "--perf-points") {
      opt.perfPoints = std::stoi(next("--perf-points"));
    } else if (a == "--perf-rays") {
      opt.perfRays = std::stoi(next("--perf-rays"));
    } else if (a == "--perf-passes") {
      opt.perfPasses = std::stoi(next("--perf-passes"));
    } else if (a == "--perf-warmup") {
      opt.perfWarmup = std::stoi(next("--perf-warmup"));
    } else if (a == "--ladder") {
      opt.ladderSpec = next("--ladder");
    } else if (a == "--push") {
      opt.step.push = std::stod(next("--push"));
    } else if (a == "--unstick-push") {
      opt.step.unstickPush = std::stod(next("--unstick-push"));
    } else if (a == "--zero-step") {
      opt.step.zeroStep = std::stod(next("--zero-step"));
    } else if (a == "--max-iter") {
      opt.step.maxIter = std::stoi(next("--max-iter"));
    } else if (a == "--self-test") {
      opt.selfTest = true;
    } else if (a == "-h" || a == "--help") {
      printUsage(argv[0]);
      return false;
    } else {
      throw std::runtime_error("unrecognized option: " + a);
    }
  }
  if (!opt.selfTest && opt.ladderSpec.empty() && opt.db.empty() && opt.explicitSurfaces.empty() &&
      opt.explicitShape.empty() && opt.explicitFlatCSG.empty()) {
    throw std::runtime_error(
      "either --db <dir>, --surfaces/--shape/--flatcsg <file>, "
      "--ladder <counts> or --self-test is required");
  }
  // Naming a sidecar means "score this", whatever the default set says.
  if (!opt.explicitFlatCSG.empty()) {
    opt.representations.insert("flatcsg");
  }
  return true;
}

std::vector<Part> collectParts(const Options& opt)
{
  std::vector<Part> parts;
  if (!opt.explicitSurfaces.empty() || !opt.explicitShape.empty() ||
      !opt.explicitFlatCSG.empty()) {
    Part part{"adhoc", "adhoc", opt.explicitSurfaces, opt.explicitFacets, opt.explicitShape,
              opt.explicitFlatCSG};
    // The siblings are only DERIVED from a `surfaces_*.bin` stem. Naming a shape or a sidecar
    // directly means "score exactly this", which is how one part is emitted two ways and the two
    // scored against each other; guessing a third subject from that name would be inventing one.
    if (!part.surfaces.empty()) {
      if (part.facets.empty()) {
        part.facets = deriveSidecarPath(part.surfaces, "facets_", ".bin");
      }
      if (part.shape.empty()) {
        part.shape = deriveSidecarPath(part.surfaces, "shape_", ".root");
      }
      if (part.flatcsg.empty()) {
        part.flatcsg = deriveSidecarPath(part.surfaces, "flatcsg_", ".bin");
      }
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
    part.model = p.value("model", std::string("?"));
    part.surfaces = p.value("surfaces", std::string());
    part.facets = p.value("facets", std::string());
    part.shape = p.value("shape", std::string());
    if (part.shape.empty()) {
      part.shape = deriveSidecarPath(part.surfaces, "shape_", ".root");
    }
    part.flatcsg = p.value("flatcsg", std::string());
    if (part.flatcsg.empty()) {
      part.flatcsg = deriveSidecarPath(part.surfaces, "flatcsg_", ".bin");
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

void writeRays(const std::string& dir, const std::string& partId, const Raster& raster,
               const std::string& bboxSource)
{
  json doc;
  doc["version"] = kXRayFormatVersion;
  doc["part"] = partId;
  doc["windowMin"] = {raster.windowMin[0], raster.windowMin[1], raster.windowMin[2]};
  doc["windowMax"] = {raster.windowMax[0], raster.windowMax[1], raster.windowMax[2]};
  doc["raster"] = raster.n;
  json beams = json::array();
  for (const auto& beam : raster.beams) {
    beams.push_back({{"label", beam.label}, {"dir", {beam.dir[0], beam.dir[1], beam.dir[2]}}});
  }
  doc["beams"] = beams;
  doc["cellArea"] = raster.cellArea;
  doc["transverseMargin"] = raster.transverseMargin;
  doc["windowExcess"] = raster.windowExcess;
  doc["bboxSource"] = bboxSource;
  json rays = json::array();
  for (const auto& r : raster.rays) {
    rays.push_back({{"o", {r.origin[0], r.origin[1], r.origin[2]}},
                    {"d", {r.dir[0], r.dir[1], r.dir[2]}},
                    {"tmax", r.tMax},
                    {"beam", r.beam}});
  }
  doc["rays"] = std::move(rays);
  const std::string path = dir + "/xrays_" + sanitizePartId(partId) + ".json";
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("cannot write " + path);
  }
  out << doc.dump();
  std::printf("  wrote %s (%zu rays)\n", path.c_str(), raster.rays.size());
}

/// The oracle's answer for one part: the ordered crossing list per ray, plus its own chord volume.
struct OracleCrossings {
  bool has = false;
  double tolerance = 1.e-7;
  double capacity = 0.;
  double volumeChord = 0.;
  bool valid = false;
  std::vector<std::vector<Crossing>> perRay;
  std::vector<bool> ambiguous;
  long long ambiguousRays = 0;
  Raster raster;
};

OracleCrossings loadOracleCrossings(const std::string& dir, const std::string& partId)
{
  OracleCrossings out;
  const std::string path = dir + "/crossings_" + sanitizePartId(partId) + ".json";
  std::ifstream in(path);
  if (!in) {
    return out;
  }
  json doc;
  in >> doc;
  if (doc.value("version", 0) != kXRayFormatVersion) {
    throw std::runtime_error(path + ": unsupported format version");
  }
  out.has = true;
  out.tolerance = doc.value("tolerance", 1.e-7);
  out.capacity = doc.value("capacity", 0.);
  out.volumeChord = doc.value("volumeChord", 0.);
  out.valid = doc.value("valid", false);
  out.ambiguousRays = doc.value("ambiguousRays", 0);
  out.raster.n = doc.value("raster", 0);
  out.raster.transverseMargin = doc.value("transverseMargin", 0.);
  const auto& window0 = doc.at("windowMin");
  const auto& window1 = doc.at("windowMax");
  for (int k = 0; k < 3; ++k) {
    out.raster.windowMin[k] = window0[k].get<double>();
    out.raster.windowMax[k] = window1[k].get<double>();
  }
  out.raster.cellArea = doc.at("cellArea").get<std::vector<double>>();
  out.raster.windowExcess = doc.value("windowExcess", std::vector<double>(out.raster.cellArea.size(), 0.));
  for (const auto& b : doc.at("beams")) {
    Beam beam;
    beam.label = b.at("label").get<std::string>();
    for (int k = 0; k < 3; ++k) {
      beam.dir[k] = b.at("dir")[k].get<double>();
    }
    out.raster.beams.push_back(std::move(beam));
  }
  out.raster.rays.reserve(doc.at("rays").size());
  for (const auto& r : doc.at("rays")) {
    RayDef ray;
    for (int k = 0; k < 3; ++k) {
      ray.origin[k] = r.at("o")[k].get<double>();
      ray.dir[k] = r.at("d")[k].get<double>();
    }
    ray.tMax = r.at("tmax").get<double>();
    ray.beam = r.at("beam").get<int>();
    out.raster.rays.push_back(ray);
    std::vector<Crossing> crossings;
    const auto& ts = r.at("t");
    const auto& kinds = r.at("k");
    for (size_t i = 0; i < ts.size(); ++i) {
      crossings.push_back({ts[i].get<double>(), kinds[i].get<int>()});
    }
    out.perRay.push_back(std::move(crossings));
    // A ray OCCT itself declined to classify somewhere along its length. Excluded from the
    // comparison rather than scored either way -- the same treatment `nNoVerdict` gets in the
    // sample gate, for the same reason: there is no ground truth to compare against there.
    out.ambiguous.push_back(r.value("amb", false));
  }
  return out;
}

/// A ray of the part frame, expressed in a placed shape's own frame.
///
/// Mode (a) speaks to the shape API directly, so it is the caller's job to put the query in the
/// shape's frame. A rigid transform preserves lengths, so every `t` in the resulting crossing list
/// is the same number it would have been in the part frame -- which is why the lists produced this
/// way are compared against the oracle's, and against mode (b)'s, without any further correction.
void toShapeFrame(const TGeoMatrix* placement, const Point3D& origin, const Point3D& dir,
                  Point3D& localOrigin, Point3D& localDir)
{
  if (placement == nullptr) {
    localOrigin = origin;
    localDir = dir;
    return;
  }
  placement->MasterToLocal(origin.data(), localOrigin.data());
  placement->MasterToLocalVect(dir.data(), localDir.data());
}

/// A shape's bounding box carried into the part frame: the axis-aligned hull of the eight
/// transformed corners. Conservative for a rotated body, which is exactly what a raster window and
/// a navigator world both need.
void placedBox(const TGeoBBox& box, const TGeoMatrix* placement, Point3D& lo, Point3D& hi)
{
  const double half[3] = {box.GetDX(), box.GetDY(), box.GetDZ()};
  for (int k = 0; k < 3; ++k) {
    lo[k] = box.GetOrigin()[k] - half[k];
    hi[k] = box.GetOrigin()[k] + half[k];
  }
  if (placement == nullptr) {
    return;
  }
  Point3D outLo{1.e300, 1.e300, 1.e300};
  Point3D outHi{-1.e300, -1.e300, -1.e300};
  for (int corner = 0; corner < 8; ++corner) {
    const double local[3] = {(corner & 1) ? hi[0] : lo[0], (corner & 2) ? hi[1] : lo[1],
                             (corner & 4) ? hi[2] : lo[2]};
    double master[3];
    placement->LocalToMaster(local, master);
    for (int k = 0; k < 3; ++k) {
      outLo[k] = std::min(outLo[k], master[k]);
      outHi[k] = std::max(outHi[k], master[k]);
    }
  }
  lo = outLo;
  hi = outHi;
}

/// The tightest CONTAINING bounding box available for a part, and where it came from.
///
/// The order is a measurement: the surface solid's box is conservative, while the shape's and the
/// mesh's are tight. So: shape, else mesh, else surface, and say which.
bool resolveBoundingBox(const Part& part, const Options& opt, Point3D& lo, Point3D& hi,
                        std::string& source)
{
  struct Candidate {
    const char* name;
    const std::string& path;
  };
  const Candidate candidates[4] = {{"shape", part.shape},
                                   {"flatcsg", part.flatcsg},
                                   {"mesh", part.facets},
                                   {"surface", part.surfaces}};
  for (const auto& candidate : candidates) {
    if (!opt.representations.count(candidate.name) || !fileExists(candidate.path)) {
      continue;
    }
    auto* manager = new TGeoManager("xrayBBox", "bbox probe");
    TGeoShape* shape = nullptr;
    std::unique_ptr<TGeoHMatrix> placement;
    if (std::string(candidate.name) == "surface") {
      auto* solid = new O2BVHSurfaceSolid(part.id.c_str());
      if (LoadSurfaceSolid(candidate.path, *solid)) {
        solid->CloseShape(true);
        shape = solid;
      }
    } else if (std::string(candidate.name) == "mesh") {
      auto* solid = new O2Tessellated(part.id.c_str());
      if (LoadFacetSolid(candidate.path, *solid)) {
        solid->CloseShape();
        shape = solid;
      }
    } else if (std::string(candidate.name) == "flatcsg") {
      auto* solid = new O2FlatCSG(part.id.c_str());
      if (LoadFlatCSG(candidate.path, *solid)) {
        solid->CloseShape();
        shape = solid;
      }
    } else {
      shape = loadShapeFromRootFile(candidate.path, nullptr);
      // The window must be stated in the PART frame, so a placed shape's box is carried through
      // its placement first. Skipping this would raster a rotated tube against the box of the tube
      // at the origin -- a window that misses the part entirely.
      placement.reset(loadShapePlacementFromRootFile(candidate.path));
    }
    const auto* box = dynamic_cast<const TGeoBBox*>(shape);
    if (box != nullptr) {
      placedBox(*box, placement.get(), lo, hi);
      source = candidate.name;
      delete manager;
      gGeoManager = nullptr;
      return true;
    }
    delete manager;
    gGeoManager = nullptr;
  }
  return false;
}

// ------------------------------------------------------------------------------------------
// --perf: per-call cost and memory, per representation, from one shared sample set
// ------------------------------------------------------------------------------------------
//
// Everything here answers one question -- "what does asking this representation a navigation
// question cost, and what does holding it cost" -- and it answers it under three constraints that
// are the whole difference between a benchmark and a stopwatch:
//
//   * SAME QUESTIONS. The point and ray sets are built once per part, from a designated reference
//     representation's own Contains(), and handed unchanged to all three. `partitionedBy` is
//     reported so nobody has to guess which one.
//   * WARM CACHE, and said so. Every kernel is warmed before it is timed and every part fits in
//     cache, so these are steady-state numbers for a single resident solid. A real simulation
//     holds thousands of solids and misses; the ratios here are an upper bound on how well the
//     cheaper representation does there, not a prediction of it.
//   * LOAD EXCLUDED FROM THE KERNEL, AND REPORTED SEPARATELY, because loading dominates the run
//     on a large model.

json timingToJson(const TimingStat& t)
{
  json out{{"callsPerPass", t.callsPerPass},
           {"passes", t.passes},
           {"nsPerCallMedian", t.medianNsPerCall},
           {"nsPerCallMin", t.minNsPerCall},
           {"nsPerCallMax", t.maxNsPerCall},
           {"spread", t.spread},
           {"checksum", t.checksum}};
  if (t.hitFraction >= 0.) {
    out["hitFraction"] = t.hitFraction;
  }
  return out;
}

/// A representation, loaded, with everything the cost table needs to say about it.
struct LoadedRep {
  TGeoManager* manager = nullptr;
  TGeoShape* shape = nullptr;
  const O2BVHSurfaceSolid* surfaceSolid = nullptr;
  const O2FlatCSG* flatSolid = nullptr;
  std::unique_ptr<TGeoHMatrix> placement;
  StructuralMemory structural;
  MemorySnapshot loadDelta;  ///< across the file read
  MemorySnapshot closeDelta; ///< across CloseShape(), i.e. the acceleration structure
  double loadSeconds = 0.;
  double closeSeconds = 0.;
  bool meshClosedBody = true;
  bool ok = false;
};

/// Load one representation into its own TGeoManager, measuring what it cost to do so.
///
/// The split between `loadDelta` and `closeDelta` is deliberate and it is where the surface
/// solid's memory actually is: `LoadSurfaceSolid` reads the sidecar, `CloseShape` builds the BVH,
/// and lumping the two together would attribute an acceleration structure to a file format.
LoadedRep loadRepresentation(const std::string& name, const std::string& source,
                             const std::string& partId, int flatSplitDepth = -1,
                             double flatMinBoxFraction = -1.)
{
  LoadedRep rep;
  rep.manager = new TGeoManager(("perf_" + name).c_str(), "representation benchmark");
  rep.structural.sidecarBytes = fileBytes(source);
  const MemorySnapshot before = readMemory();
  const auto t0 = std::chrono::steady_clock::now();
  if (name == "surface") {
    auto* solid = new O2BVHSurfaceSolid(partId.c_str());
    if (!LoadSurfaceSolid(source, *solid)) {
      return rep;
    }
    const auto t1 = std::chrono::steady_clock::now();
    rep.loadSeconds = std::chrono::duration<double>(t1 - t0).count();
    rep.loadDelta = readMemory() - before;
    const MemorySnapshot beforeClose = readMemory();
    solid->CloseShape(true);
    rep.closeSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
    rep.closeDelta = readMemory() - beforeClose;
    rep.shape = solid;
    rep.surfaceSolid = solid;
    rep.structural.primitives = solid->GetNsurfaces();
    // The patch count and the sidecar are the two EXACT numbers a surface solid has. The trim
    // wires are variable-length per patch and live behind a private type, so the in-memory
    // arithmetic is not available from outside; the sidecar bytes bound it from below and the
    // measured heap delta bounds it from above, and both are printed rather than one guessed
    // number in between.
    rep.structural.bytes = rep.structural.sidecarBytes;
    rep.structural.formula = "patches=" + std::to_string(rep.structural.primitives) +
                             "; bytes = sidecar on disk (in-memory trim arrays are not "
                             "introspectable; see measured heap delta)";
  } else if (name == "mesh") {
    auto* solid = new O2Tessellated(partId.c_str());
    if (!LoadFacetSolid(source, *solid)) {
      return rep;
    }
    const auto t1 = std::chrono::steady_clock::now();
    rep.loadSeconds = std::chrono::duration<double>(t1 - t0).count();
    rep.loadDelta = readMemory() - before;
    const MemorySnapshot beforeClose = readMemory();
    solid->CloseShape();
    rep.closeSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
    rep.closeDelta = readMemory() - beforeClose;
    rep.shape = solid;
    rep.meshClosedBody = solid->IsClosedBody();
    rep.structural.primitives = solid->GetNfacets();
    // Exact, and the one representation whose in-memory size IS arithmetic: three index arrays
    // per facet plus a deduplicated vertex array plus one outward normal per facet.
    const long long nF = solid->GetNfacets();
    const long long nV = solid->GetNvertices();
    rep.structural.bytes = nV * static_cast<long long>(sizeof(O2Tessellated::Vertex_t)) +
                           nF * static_cast<long long>(sizeof(TGeoFacet)) +
                           nF * static_cast<long long>(sizeof(O2Tessellated::Vertex_t));
    rep.structural.formula =
      std::to_string(nV) + " vertices x " + std::to_string(sizeof(O2Tessellated::Vertex_t)) +
      " B + " + std::to_string(nF) + " facets x " + std::to_string(sizeof(TGeoFacet)) +
      " B + " + std::to_string(nF) + " normals x " + std::to_string(sizeof(O2Tessellated::Vertex_t)) + " B";
  } else if (name == "flatcsg") {
    auto* solid = new O2FlatCSG(partId.c_str());
    if (!LoadFlatCSG(source, *solid)) {
      return rep;
    }
    if (flatSplitDepth >= 0) {
      solid->SetSplitDepth(flatSplitDepth);
    }
    if (flatMinBoxFraction >= 0.) {
      solid->SetMinBoxFraction(flatMinBoxFraction);
    }
    const auto t1 = std::chrono::steady_clock::now();
    rep.loadSeconds = std::chrono::duration<double>(t1 - t0).count();
    rep.loadDelta = readMemory() - before;
    const MemorySnapshot beforeClose = readMemory();
    solid->CloseShape();
    rep.closeSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
    rep.closeDelta = readMemory() - beforeClose;
    if (!solid->IsClosed()) {
      // A refused CloseShape leaves a shape that answers through its `_Loop` twins -- correct, and
      // orders of magnitude slower. Timing it as if it were the accelerated path would be a
      // measurement of the wrong thing, so the representation is dropped instead.
      return rep;
    }
    rep.shape = solid;
    rep.flatSolid = solid;
    rep.structural.primitives = solid->GetNcells();
    // Exact for everything this class owns: the halfspace blocks, the cell table, the sub-cell
    // boxes with their concatenated active lists, and the BVH the class reports for itself.
    const long long nH = solid->GetNhalfspaces();
    const long long nC = solid->GetNcells();
    const long long nB = solid->GetNboxes();
    long long active = 0;
    for (int i = 0; i < solid->GetNboxes(); ++i) {
      active += solid->GetBox(i).nActive;
    }
    const long long bvh = static_cast<long long>(solid->GetBVHMemory());
    rep.structural.bytes = nH * static_cast<long long>(sizeof(FlatCSGHalfspace)) +
                           nC * static_cast<long long>(sizeof(FlatCSGCell)) +
                           nC * 6 * static_cast<long long>(sizeof(double)) +
                           nB * static_cast<long long>(sizeof(FlatCSGBox)) +
                           active * static_cast<long long>(sizeof(int)) + bvh;
    rep.structural.formula =
      std::to_string(nH) + " halfspaces x " + std::to_string(sizeof(FlatCSGHalfspace)) + " B + " +
      std::to_string(nC) + " cells x " + std::to_string(sizeof(FlatCSGCell) + 6 * sizeof(double)) +
      " B + " + std::to_string(nB) + " boxes x " + std::to_string(sizeof(FlatCSGBox)) + " B + " +
      std::to_string(active) + " active x " + std::to_string(sizeof(int)) + " B + BVH " +
      std::to_string(bvh) + " B";
  } else {
    std::string error;
    rep.shape = loadShapeFromRootFile(source, &error);
    if (rep.shape == nullptr) {
      return rep;
    }
    rep.placement.reset(loadShapePlacementFromRootFile(source));
    rep.loadSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    rep.loadDelta = readMemory() - before;
    const BooleanTreeStats tree = booleanTreeStats(rep.shape);
    rep.structural.primitives = tree.leaves;
    // A composite is a handful of objects: the node count is exact and tiny, and that is the
    // headline of the whole memory column.
    rep.structural.bytes = tree.leaves * 200 + tree.nodes * 200;
    rep.structural.formula = "leaves=" + std::to_string(tree.leaves) +
                             " nodes=" + std::to_string(tree.nodes) +
                             " depth=" + std::to_string(tree.depth) +
                             "; bytes ~ (leaves+nodes) x 200 B (ROOT object overhead dominates)";
  }
  rep.ok = true;
  return rep;
}

/// The same sample set, expressed in a placed shape's own frame.
///
/// A rigid transform preserves lengths, so every distance the kernels return is the number it
/// would have been in the part frame. This is the same argument mode (a) of the transport loop
/// makes, and it is why the timing of a placed primitive is comparable with everything else.
QuerySamples toShapeFrame(const QuerySamples& in, const TGeoMatrix* placement)
{
  if (placement == nullptr) {
    return in;
  }
  QuerySamples out = in;
  for (auto& p : out.points) {
    Point3D q;
    placement->MasterToLocal(p.data(), q.data());
    p = q;
  }
  auto move = [&](std::vector<Ray>& rays) {
    for (auto& r : rays) {
      Point3D o;
      Point3D d;
      placement->MasterToLocal(r.origin.data(), o.data());
      placement->MasterToLocalVect(r.dir.data(), d.data());
      r.origin = o;
      r.dir = d;
    }
  };
  move(out.outsideRays);
  move(out.insideRays);
  return out;
}

/// The one part of this that is about O2BVHSurfaceSolid rather than about representations.
///
/// An aggregate says *that*, never *where*. If the surface solid is slower than a two-leaf
/// composite, "the BVH surface solid is slow" is not a finding -- it is a restatement. These four
/// numbers localise it: how many patches the BVH hands to the leaf callback per ray query, what
/// the same query costs with the acceleration structure's tmax pruning switched off, what it
/// costs with no BVH at all (the `_Loop` twin), and therefore what one patch intersection costs.
/// Nothing here is optimised; it is measured and reported.
json localiseSurfaceSolid(const O2BVHSurfaceSolid* solid, const QuerySamples& s, int warmup, int passes)
{
  json out;
  const auto* shape = static_cast<const TGeoShape*>(solid);
  (void)shape;

  const bool pruningWas = O2BVHSurfaceSolid::GetRayTMaxPruning();

  O2BVHSurfaceSolid::SetRayTMaxPruning(true);
  O2BVHSurfaceSolid::ResetRayCandidateCounter();
  for (const auto& ray : s.outsideRays) {
    volatile double sink = solid->DistFromOutside(ray.origin.data(), ray.dir.data(), 3, TGeoShape::Big(), nullptr);
    (void)sink;
  }
  const long long prunedCandidates = O2BVHSurfaceSolid::GetRayCandidateCount();
  const TimingStat pruned = timePasses(static_cast<long long>(s.outsideRays.size()), warmup, passes, [&]() {
    uint64_t acc = 0;
    for (const auto& ray : s.outsideRays) {
      acc ^= static_cast<uint64_t>(
        solid->DistFromOutside(ray.origin.data(), ray.dir.data(), 3, TGeoShape::Big(), nullptr) * 1.e6);
    }
    return acc;
  });

  O2BVHSurfaceSolid::SetRayTMaxPruning(false);
  O2BVHSurfaceSolid::ResetRayCandidateCounter();
  for (const auto& ray : s.outsideRays) {
    volatile double sink = solid->DistFromOutside(ray.origin.data(), ray.dir.data(), 3, TGeoShape::Big(), nullptr);
    (void)sink;
  }
  const long long unprunedCandidates = O2BVHSurfaceSolid::GetRayCandidateCount();
  const TimingStat unpruned = timePasses(static_cast<long long>(s.outsideRays.size()), warmup, passes, [&]() {
    uint64_t acc = 0;
    for (const auto& ray : s.outsideRays) {
      acc ^= static_cast<uint64_t>(
        solid->DistFromOutside(ray.origin.data(), ray.dir.data(), 3, TGeoShape::Big(), nullptr) * 1.e6);
    }
    return acc;
  });
  O2BVHSurfaceSolid::SetRayTMaxPruning(pruningWas);

  const TimingStat loop = timePasses(static_cast<long long>(s.outsideRays.size()), warmup, passes, [&]() {
    uint64_t acc = 0;
    for (const auto& ray : s.outsideRays) {
      acc ^= static_cast<uint64_t>(
        solid->DistFromOutside_Loop(ray.origin.data(), ray.dir.data()) * 1.e6);
    }
    return acc;
  });
  const TimingStat containsLoop = timePasses(static_cast<long long>(s.points.size()), warmup, passes, [&]() {
    uint64_t acc = 0;
    for (const auto& p : s.points) {
      acc ^= solid->Contains_Loop(p.data()) ? 1u : 0u;
    }
    return acc;
  });

  // --- the nearest-patch queries: Safety() and ComputeNormal() -------------------------------
  //
  // Same shape of measurement; the `_Loop` twins are the unaccelerated kernels, so "before" and
  // "after" run in the same binary on the same sample set.
  //
  // The disagreement counter travels with the timing on purpose: the twins must return bit-
  // identical answers, and a speed ratio quoted without it would price two different kernels.
  O2BVHSurfaceSolid::ResetSafetyCandidateCounter();
  long long safetyDisagreements = 0;
  long long normalDisagreements = 0;
  for (size_t index = 0; index < s.points.size(); ++index) {
    const bool inside = s.pointIsInside[index] != 0;
    if (solid->Safety(s.points[index].data(), inside) != solid->Safety_Loop(s.points[index].data(), inside)) {
      ++safetyDisagreements;
    }
    Point3D viaBVH{0., 0., 0.};
    Point3D viaLoop{0., 0., 0.};
    solid->ComputeNormal(s.points[index].data(), nullptr, viaBVH.data());
    solid->ComputeNormal_Loop(s.points[index].data(), nullptr, viaLoop.data());
    if (viaBVH != viaLoop) {
      ++normalDisagreements;
    }
  }
  O2BVHSurfaceSolid::ResetSafetyCandidateCounter();
  for (size_t index = 0; index < s.points.size(); ++index) {
    volatile double sink = solid->Safety(s.points[index].data(), s.pointIsInside[index] != 0);
    (void)sink;
  }
  const long long safetyCandidates = O2BVHSurfaceSolid::GetSafetyCandidateCount();

  const TimingStat safetyBVH = timePasses(static_cast<long long>(s.points.size()), warmup, passes, [&]() {
    uint64_t acc = 0;
    for (size_t index = 0; index < s.points.size(); ++index) {
      acc ^= static_cast<uint64_t>(solid->Safety(s.points[index].data(), s.pointIsInside[index] != 0) * 1.e6);
    }
    return acc;
  });
  const TimingStat safetyLoop = timePasses(static_cast<long long>(s.points.size()), warmup, passes, [&]() {
    uint64_t acc = 0;
    for (size_t index = 0; index < s.points.size(); ++index) {
      acc ^= static_cast<uint64_t>(solid->Safety_Loop(s.points[index].data(), s.pointIsInside[index] != 0) * 1.e6);
    }
    return acc;
  });
  const TimingStat normalBVH = timePasses(static_cast<long long>(s.points.size()), warmup, passes, [&]() {
    uint64_t acc = 0;
    Point3D normal{0., 0., 0.};
    for (const auto& p : s.points) {
      solid->ComputeNormal(p.data(), nullptr, normal.data());
      acc ^= static_cast<uint64_t>(normal[0] * 1.e6);
    }
    return acc;
  });
  const TimingStat normalLoop = timePasses(static_cast<long long>(s.points.size()), warmup, passes, [&]() {
    uint64_t acc = 0;
    Point3D normal{0., 0., 0.};
    for (const auto& p : s.points) {
      solid->ComputeNormal_Loop(p.data(), nullptr, normal.data());
      acc ^= static_cast<uint64_t>(normal[0] * 1.e6);
    }
    return acc;
  });

  const double rays = static_cast<double>(std::max<size_t>(1, s.outsideRays.size()));
  const double points = static_cast<double>(std::max<size_t>(1, s.points.size()));
  out["safetyBVHNs"] = safetyBVH.medianNsPerCall;
  out["safetyLoopNs"] = safetyLoop.medianNsPerCall;
  out["safetySpeedup"] = safetyBVH.medianNsPerCall > 0. ? safetyLoop.medianNsPerCall / safetyBVH.medianNsPerCall : 0.;
  out["normalBVHNs"] = normalBVH.medianNsPerCall;
  out["normalLoopNs"] = normalLoop.medianNsPerCall;
  out["bvhCandidatesPerSafetyCall"] = safetyCandidates / points;
  out["loopCandidatesPerSafetyCall"] = static_cast<double>(solid->GetNsurfaces());
  out["safetyDisagreements"] = safetyDisagreements;
  out["normalDisagreements"] = normalDisagreements;
  out["nearestPatchComparedPoints"] = static_cast<long long>(s.points.size());
  std::printf(
    "      safety:   %.1f ns BVH vs %.1f ns _Loop (%.1fx) | %.2f candidates/call of %d | "
    "normal %.1f ns vs %.1f ns | disagreements %lld safety / %lld normal in %zu points\n",
    safetyBVH.medianNsPerCall, safetyLoop.medianNsPerCall, out["safetySpeedup"].get<double>(),
    safetyCandidates / points, solid->GetNsurfaces(), normalBVH.medianNsPerCall,
    normalLoop.medianNsPerCall, safetyDisagreements, normalDisagreements, s.points.size());

  out["patches"] = solid->GetNsurfaces();
  out["bvhCandidatesPerDistOutCall"] = prunedCandidates / rays;
  out["loopCandidatesPerDistOutCall"] = unprunedCandidates / rays;
  out["distOutPrunedNs"] = pruned.medianNsPerCall;
  out["distOutUnprunedNs"] = unpruned.medianNsPerCall;
  out["distOutLoopNs"] = loop.medianNsPerCall;
  out["containsLoopNs"] = containsLoop.medianNsPerCall;
  out["nsPerBVHCandidate"] =
    prunedCandidates > 0 ? pruned.medianNsPerCall * rays / static_cast<double>(prunedCandidates) : 0.;
  std::printf(
    "      localise: %d patches | %.1f BVH candidates/distout call (unpruned %.1f) | "
    "distout %.1f ns pruned, %.1f ns unpruned, %.1f ns _Loop | %.2f ns per candidate patch | "
    "Contains_Loop %.1f ns\n",
    solid->GetNsurfaces(), prunedCandidates / rays, unprunedCandidates / rays,
    pruned.medianNsPerCall, unpruned.medianNsPerCall, loop.medianNsPerCall,
    out["nsPerBVHCandidate"].get<double>(), containsLoop.medianNsPerCall);
  return out;
}

void printTiming(const char* label, const TimingStat& t)
{
  std::printf("      %-14s %9.1f ns/call  [%9.1f .. %9.1f, spread %5.1f%%]", label,
              t.medianNsPerCall, t.minNsPerCall, t.maxNsPerCall, 100. * t.spread);
  if (t.hitFraction >= 0.) {
    std::printf("  hit %5.1f%%", 100. * t.hitFraction);
  }
  std::printf("\n");
}

/// The synthetic boolean ladder: `--ladder 2,4,8,16,32`.
///
/// It exists because the corpus cannot answer the question it answers. Reported for both tree
/// shapes and with the leaf count verified from the built tree rather than from the request --
/// a fixture that claims 32 leaves and holds 16 would show sublinear scaling and be wrong.
json runLadder(const Options& opt)
{
  json out = json::array();
  std::vector<int> counts;
  {
    std::stringstream ss(opt.ladderSpec);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
      if (!tok.empty()) {
        counts.push_back(std::stoi(tok));
      }
    }
  }
  std::printf("=== synthetic boolean ladder: unions of K overlapping TGeoTubes ===\n");
  std::printf(
    "    Every genuine boolean in the corpus is a 2-leaf union of two TGeoTubes, so the\n"
    "    corpus cannot say how a composite scales with K. This can.\n\n");
  for (const int k : counts) {
    for (const auto shapeKind : {LadderShape::Chain, LadderShape::Balanced}) {
      const char* kindName = shapeKind == LadderShape::Chain ? "chain" : "balanced";
      auto* manager = new TGeoManager("ladder", "boolean ladder");
      const std::string tag = std::string("L") + kindName + std::to_string(k);
      const MemorySnapshot before = readMemory();
      TGeoShape* shape = buildBooleanLadder(k, shapeKind, tag);
      const MemorySnapshot after = readMemory();
      if (shape == nullptr) {
        delete manager;
        gGeoManager = nullptr;
        continue;
      }
      const BooleanTreeStats tree = booleanTreeStats(shape);
      const auto* box = dynamic_cast<const TGeoBBox*>(shape);
      const Point3D lo{box->GetOrigin()[0] - box->GetDX(), box->GetOrigin()[1] - box->GetDY(),
                       box->GetOrigin()[2] - box->GetDZ()};
      const Point3D hi{box->GetOrigin()[0] + box->GetDX(), box->GetOrigin()[1] + box->GetDY(),
                       box->GetOrigin()[2] + box->GetDZ()};
      const QuerySamples samples =
        buildQuerySamples(shape, "self", lo, hi, opt.perfPoints, opt.perfRays);
      const TimingStat contains = timeContainsPass(shape, samples, opt.perfWarmup, opt.perfPasses);
      const TimingStat safety = timeSafetyPass(shape, samples, opt.perfWarmup, opt.perfPasses);
      const TimingStat distOut = timeDistOutPass(shape, samples, opt.perfWarmup, opt.perfPasses);
      const TimingStat distIn = timeDistInPass(shape, samples, opt.perfWarmup, opt.perfPasses);
      std::printf("  --- K=%-3d %-9s (leaves=%lld nodes=%lld depth=%d, %.1f%% of points inside) ---\n",
                  k, kindName, tree.leaves, tree.nodes, tree.depth,
                  100. * static_cast<double>(samples.insidePoints) /
                    static_cast<double>(std::max<size_t>(1, samples.points.size())));
      printTiming("Contains", contains);
      printTiming("Safety", safety);
      printTiming("DistFromOutside", distOut);
      printTiming("DistFromInside", distIn);
      out.push_back({{"leavesRequested", k},
                     {"treeShape", kindName},
                     {"leaves", tree.leaves},
                     {"nodes", tree.nodes},
                     {"depth", tree.depth},
                     {"buildResidentBytes", (after - before).residentBytes},
                     {"buildHeapBytes", (after - before).heapInUseBytes},
                     {"insideFraction", static_cast<double>(samples.insidePoints) /
                                          static_cast<double>(std::max<size_t>(1, samples.points.size()))},
                     {"contains", timingToJson(contains)},
                     {"safety", timingToJson(safety)},
                     {"distFromOutside", timingToJson(distOut)},
                     {"distFromInside", timingToJson(distIn)}});
      delete manager;
      gGeoManager = nullptr;
    }
  }
  return out;
}

// ------------------------------------------------------------------------------------------
// Self-test: analytic references, and the controls that prove the comparison can fail
// ------------------------------------------------------------------------------------------

int selfTest()
{
  int failures = 0;
  auto check = [&](const char* name, bool ok, const std::string& detail = {}) {
    std::printf("  [%s] %s%s\n", ok ? "ok  " : "FAIL", name,
                ok || detail.empty() ? "" : ("   " + detail).c_str());
    if (!ok) {
      ++failures;
    }
  };

  StepConfig cfg;
  Robustness stats;

  // 1. A box: exactly two crossings, at analytically known distances.
  {
    TGeoBBox box("selftestBox", 1., 1.5, 2.);
    const Point3D origin{-5., 0., 0.};
    const Point3D dir{1., 0., 0.};
    auto crossings = stepWithShapeApi(&box, origin, dir, 10., cfg, stats);
    check("box: exactly two crossings along a central ray", crossings.size() == 2,
          "got " + std::to_string(crossings.size()));
    if (crossings.size() == 2) {
      check("box: enter at 4.0 cm", std::fabs(crossings[0].t - 4.) < 1.e-9);
      check("box: exit at 6.0 cm", std::fabs(crossings[1].t - 6.) < 1.e-9);
      check("box: kinds are enter then exit", crossings[0].kind == +1 && crossings[1].kind == -1);
    }
  }

  // 2. A hollow tube: FOUR crossings along a diameter. This is the case a single-shot `distout`
  //    query cannot express at all -- it reports the first of the four and stops.
  {
    TGeoTube tube("selftestTube", 0.5, 1.0, 2.0);
    const Point3D origin{-5., 0., 0.};
    const Point3D dir{1., 0., 0.};
    auto crossings = stepWithShapeApi(&tube, origin, dir, 10., cfg, stats);
    check("hollow tube: four crossings along a diameter", crossings.size() == 4,
          "got " + std::to_string(crossings.size()));
    if (crossings.size() == 4) {
      const double expect[4] = {4.0, 4.5, 5.5, 6.0};
      bool ok = true;
      for (int i = 0; i < 4; ++i) {
        ok = ok && std::fabs(crossings[i].t - expect[i]) < 1.e-9;
      }
      check("hollow tube: crossings at 4.0 / 4.5 / 5.5 / 6.0 cm", ok);
      check("hollow tube: in, out, in, out",
            crossings[0].kind == +1 && crossings[1].kind == -1 && crossings[2].kind == +1 &&
              crossings[3].kind == -1);
    }
  }

  // 3a. A BOX's chord integral is EXACT, at every raster density, when the window is its own
  //     bounding box. That is the sharpest available self-check on the volume instrument: no
  //     convergence argument, no tolerance -- either the quadrature is the volume or it is not.
  //     It is also what fixed the raster geometry: with the window inflated by 2 % instead, this
  //     same box came out 5.1e-02 too large at N = 32.
  {
    TGeoBBox box("selftestVolBox", 1., 1.5, 2.);
    const Point3D bboxMin{-1., -1.5, -2.};
    const Point3D bboxMax{1., 1.5, 2.};
    for (const int n : {7, 32}) {
      Raster raster = buildRaster(bboxMin, bboxMax, n, buildBeams("xyz", 0.), 0.);
      Robustness s;
      std::vector<double> byAxis(3, 0.);
      for (const auto& ray : raster.rays) {
        const double before = s.insideLength;
        auto crossings = stepWithShapeApi(&box, ray.origin, ray.dir, ray.tMax, cfg, s);
        auditCrossingList(crossings, nullptr, ray.origin, ray.dir, ray.tMax, cfg, s);
        byAxis[ray.beam] += s.insideLength - before;
      }
      const double volume = chordVolume(raster, byAxis);
      check(("box 2 x 3 x 4 cm: chord integral is EXACT at N=" + std::to_string(n)).c_str(),
            std::fabs(volume - 24.) < 1.e-9, "got " + std::to_string(volume));
    }
  }

  // 3b. A sphere's chord integral against its closed-form volume. A curved silhouette cannot be
  //     exact at finite N, so this is where the ACHIEVED PRECISION of the volume instrument is
  //     measured -- and the measurement says the convergence is NOT monotone in N (the silhouette
  //     cells realign with the lattice at every density), so the honest statement is an envelope
  //     at a stated density, never an extrapolation.
  {
    TGeoSphere sphere("selftestSphere", 0., 1.);
    const Point3D bboxMin{-1., -1., -1.};
    const Point3D bboxMax{1., 1., 1.};
    const double exact = 4. / 3. * 3.14159265358979323846;
    double worst = 0.;
    for (const int n : {24, 48, 96, 192}) {
      Raster raster = buildRaster(bboxMin, bboxMax, n, buildBeams("z", 0.), 0.);
      Robustness s;
      for (const auto& ray : raster.rays) {
        auto crossings = stepWithShapeApi(&sphere, ray.origin, ray.dir, ray.tMax, cfg, s);
        auditCrossingList(crossings, nullptr, ray.origin, ray.dir, ray.tMax, cfg, s);
      }
      const double volume = s.insideLength * raster.cellArea[0];
      const double rel = std::fabs(volume - exact) / exact;
      worst = std::max(worst, rel);
      std::printf(
        "        sphere r=1: raster %3d x %3d -> V = %.8f cm^3, exact %.8f, "
        "relative %.3e\n",
        n, n, volume, exact, rel);
    }
    // The bound is the MEASURED envelope over N = 24..192, not a convergence rate. If a future
    // change makes the quadrature worse than this it is a regression; if the envelope itself has
    // to be widened, that is a result to report rather than a constant to tune.
    check("sphere chord integral stays inside the measured 2e-3 envelope for N = 24..192",
          worst < 2.e-3, "worst rel=" + std::to_string(worst));
  }

  // 4. THE CONTROLS. A comparison that cannot fail is not a comparison. Take a correct crossing
  //    list and (i) perturb one distance, (ii) drop one crossing, (iii) duplicate one, and require
  //    the comparator to name each.
  {
    const std::vector<Crossing> truth{{4.0, +1}, {4.5, -1}, {5.5, +1}, {6.0, -1}};
    const Point3D o{-5., 0., 0.};
    const Point3D d{1., 0., 0.};

    ListComparison clean;
    compareLists(truth, truth, o, d, 1.e-6, clean);
    check("control 0: identical lists compare clean",
          clean.raysIdentical == 1 && clean.missing == 0 && clean.extra == 0 &&
            clean.matched == 4);

    auto perturbed = truth;
    perturbed[2].t += 1.e-3;
    ListComparison shifted;
    compareLists(perturbed, truth, o, d, 1.e-6, shifted);
    check("control 1: a crossing moved by 1e-3 cm is CAUGHT, and as DISPLACED not as lost",
          shifted.raysIdentical == 0 && shifted.displaced == 1 && shifted.missing == 0 &&
            shifted.extra == 0 && std::fabs(shifted.worstDeltaT - 1.e-3) < 1.e-12,
          "displaced=" + std::to_string(shifted.displaced) + " missing=" +
            std::to_string(shifted.missing) + " dt=" + std::to_string(shifted.worstDeltaT));

    auto dropped = truth;
    dropped.erase(dropped.begin() + 1);
    ListComparison lost;
    compareLists(dropped, truth, o, d, 1.e-6, lost);
    check("control 2: a dropped crossing is CAUGHT as `missing`",
          lost.missing == 1 && lost.extra == 0, "missing=" + std::to_string(lost.missing));

    auto doubled = truth;
    doubled.insert(doubled.begin() + 1, {4.2, -1});
    ListComparison spurious;
    compareLists(doubled, truth, o, d, 1.e-6, spurious);
    check("control 3: an extra crossing is CAUGHT as `extra`",
          spurious.extra == 1 && spurious.missing == 0, "extra=" + std::to_string(spurious.extra));

    // A crossing at the right place but with the wrong sense (enter where the truth exits) is a
    // different defect and must not be absorbed into `matched`.
    auto flipped = truth;
    flipped[1].kind = +1;
    ListComparison sense;
    compareLists(flipped, truth, o, d, 1.e-6, sense);
    check("control 4: a crossing with the wrong sense is CAUGHT", sense.kindMismatch == 1);
  }

  // 5b. THE TIMING HARNESS'S OWN NEGATIVE CONTROL. A timing harness that cannot distinguish a
  //     deliberately slowed shape from a fast one is not measuring what it claims. So:
  //     time the same kernels on a TGeoBBox and on a TGeoBBox carrying ballast, and require the
  //     number to MOVE, in the right direction, on all four kernels.
  {
    TGeoBBox fast("perfControlFast", 1., 1., 1.);
    BallastShape slow("perfControlSlow", 1., 1., 1., 60);
    const Point3D lo{-1., -1., -1.};
    const Point3D hi{1., 1., 1.};
    const QuerySamples samples = buildQuerySamples(&fast, "control", lo, hi, 2000, 2000);
    check("control 5: the shared sample set has both inside and outside points",
          samples.insidePoints > 100 &&
            samples.insidePoints < static_cast<long long>(samples.points.size()) - 100,
          "inside=" + std::to_string(samples.insidePoints) + " of " +
            std::to_string(samples.points.size()));
    check("control 6: the sample partition is consistent with the reference it came from", [&] {
      for (size_t i = 0; i < samples.points.size(); ++i) {
        if (fast.Contains(samples.points[i].data()) != (samples.pointIsInside[i] != 0)) {
          return false;
        }
      }
      return true;
    }());
    check("control 7: DistFromOutside rays actually hit (an all-miss set times the early-out)",
          timeDistOutPass(&fast, samples, 1, 3).hitFraction > 0.5);

    struct Kernel {
      const char* name;
      TimingStat (*fn)(const TGeoShape*, const QuerySamples&, int, int);
    };
    const Kernel kernels[4] = {{"Contains", &timeContainsPass},
                               {"Safety", &timeSafetyPass},
                               {"DistFromOutside", &timeDistOutPass},
                               {"DistFromInside", &timeDistInPass}};
    for (const auto& kernel : kernels) {
      const TimingStat quick = kernel.fn(&fast, samples, 2, 7);
      const TimingStat heavy = kernel.fn(&slow, samples, 2, 7);
      const double ratio = quick.medianNsPerCall > 0. ? heavy.medianNsPerCall / quick.medianNsPerCall : 0.;
      check((std::string("control 8: ballast is VISIBLE on ") + kernel.name +
             " (the timing harness can move its own number)")
              .c_str(),
            ratio > 2.,
            std::string("fast=") + std::to_string(quick.medianNsPerCall) + " ns slow=" +
              std::to_string(heavy.medianNsPerCall) + " ns ratio=" + std::to_string(ratio));
      check((std::string("control 9: the ") + kernel.name +
             " timing loop was not elided (non-zero checksum, positive time)")
              .c_str(),
            quick.checksum != 0 && quick.medianNsPerCall > 0. && quick.passes == 7);
    }
  }

  // 5c. THE MEMORY PROBE'S NEGATIVE CONTROL. Both memory numbers must move when memory is taken,
  //     and the heap number must come back when it is given up. Without this the "resident delta"
  //     column could be reporting allocator noise and nobody would know.
  {
    const MemorySnapshot before = readMemory();
    constexpr size_t kBytes = 64u << 20;
    auto* block = new char[kBytes];
    for (size_t i = 0; i < kBytes; i += 4096) {
      block[i] = static_cast<char>(i); // touch every page: an untouched mmap is not resident
    }
    const MemorySnapshot held = readMemory();
    const MemorySnapshot delta = held - before;
    check("control 10: the resident probe sees a 64 MB touched allocation",
          delta.residentBytes > 32LL << 20,
          "delta=" + std::to_string(delta.residentBytes >> 20) + " MB");
    check("control 11: the heap probe sees a 64 MB allocation",
          delta.heapInUseBytes > 32LL << 20,
          "delta=" + std::to_string(delta.heapInUseBytes >> 20) + " MB");
    delete[] block;
    const MemorySnapshot released = readMemory() - before;
    check("control 12: the heap probe sees it released again (the resident one need not)",
          released.heapInUseBytes < 8LL << 20,
          "still=" + std::to_string(released.heapInUseBytes >> 20) + " MB");
  }

  // 5d. THE STRUCTURAL MEMORY CONTROL. The exact column has to depend on the geometry, so build
  //     the same tree twice at different sizes and require the count -- and the derived byte
  //     figure -- to follow. A structural number that does not move with the structure is a
  //     constant with a units label.
  {
    auto* manager = new TGeoManager("perfControlLadder", "structural control");
    TGeoShape* small = buildBooleanLadder(4, LadderShape::Balanced, "ctlS");
    TGeoShape* big = buildBooleanLadder(32, LadderShape::Balanced, "ctlB");
    const BooleanTreeStats a = booleanTreeStats(small);
    const BooleanTreeStats b = booleanTreeStats(big);
    check("control 13: the ladder builds the leaf count it was asked for",
          a.leaves == 4 && b.leaves == 32,
          "got " + std::to_string(a.leaves) + " and " + std::to_string(b.leaves));
    check("control 14: a balanced ladder's depth is logarithmic in its leaf count",
          a.depth == 3 && b.depth == 6,
          "depth " + std::to_string(a.depth) + " and " + std::to_string(b.depth));
    TGeoShape* chain = buildBooleanLadder(32, LadderShape::Chain, "ctlC");
    const BooleanTreeStats c = booleanTreeStats(chain);
    check(
      "control 15: a chain ladder of the same leaf count is deeper, so the two tree shapes "
      "really are different fixtures",
      c.leaves == 32 && c.depth == 32,
      "leaves=" + std::to_string(c.leaves) + " depth=" + std::to_string(c.depth));
    delete manager;
    gGeoManager = nullptr;
  }

  // 5. The parity audit's own control: hand it a list with a crossing removed and require the
  //    midpoint Contains() check to contradict it.
  {
    TGeoBBox box("selftestBox2", 1., 1., 1.);
    const Point3D origin{-5., 0., 0.};
    const Point3D dir{1., 0., 0.};
    Robustness good;
    auditCrossingList({{4.0, +1}, {6.0, -1}}, &box, origin, dir, 10., cfg, good);
    check("parity audit: a correct list has no parity mismatch", good.parityMismatchIntervals == 0);
    Robustness bad;
    auditCrossingList({{4.0, +1}}, &box, origin, dir, 10., cfg, bad);
    check("parity audit: a truncated list is CAUGHT by Contains() at the midpoints",
          bad.parityMismatchIntervals > 0 && bad.oddCrossingLists == 1);
  }

  std::printf("\n%s: %d failure(s)\n", failures == 0 ? "SELF-TEST PASSED" : "SELF-TEST FAILED",
              failures);
  return failures == 0 ? 0 : 1;
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

  if (opt.selfTest) {
    return selfTest();
  }

  if (!opt.ladderSpec.empty()) {
    json ladder = runLadder(opt);
    if (!opt.jsonOut.empty()) {
      std::ofstream out(opt.jsonOut);
      out << json{{"ladder", std::move(ladder)}}.dump(1);
      std::printf("\nreport: %s\n", opt.jsonOut.c_str());
    }
    return 0;
  }

  std::vector<Beam> beams;
  std::vector<Part> parts;
  try {
    beams = opt.fanBeams > 0 ? buildFanBeams(opt.fanBeams)
                             : buildBeams(opt.axesSpec, opt.tiltDegrees);
    if (beams.empty()) {
      throw std::runtime_error("no beam selected (--axes)");
    }
    parts = collectParts(opt);
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  if (parts.empty()) {
    std::cerr << "no parts matched (pattern='" << opt.partsPattern << "')\n";
    return 1;
  }

  json report = json::array();

  // ---- --perf: the representation cost/memory comparison ---------------------------------
  if (opt.perf) {
    std::printf(
      "Per-call costs are WARM-CACHE, single-threaded, median of %d passes after %d "
      "warmup passes.\nEvery representation of a part answers the SAME sample set.\n\n",
      opt.perfPasses, opt.perfWarmup);
    for (const auto& part : parts) {
      std::printf("=== %s (%s) ===\n", part.id.c_str(), part.model.c_str());
      Point3D lo{};
      Point3D hi{};
      std::string bboxSource;
      if (!resolveBoundingBox(part, opt, lo, hi, bboxSource)) {
        std::printf("  skip: no representation could supply a bounding box\n");
        continue;
      }
      const Raster raster = buildRaster(lo, hi, opt.raster, beams, opt.margin);

      // The sample set is built ONCE, from the first representation present in the order
      // surface -> mesh -> shape, and every representation is then asked exactly it. The order is
      // a preference for the representation whose Contains() is exact, not an accident: the
      // partition is a fixed label, so it wants to come from the most trustworthy classifier
      // available, and it is reported either way.
      QuerySamples samples;
      std::string partitionedBy;
      for (const auto& candidate : allRepresentations()) {
        const std::string& source = sourceFor(part, candidate);
        if (!opt.representations.count(candidate) || !fileExists(source)) {
          continue;
        }
        LoadedRep rep = loadRepresentation(candidate, source, part.id, opt.flatSplitDepth,
                                           opt.flatMinBoxFraction);
        if (rep.ok) {
          // Points are drawn in the PART frame; a placed shape classifies them in its own.
          QuerySamples inFrame =
            buildQuerySamples(rep.shape, candidate, lo, hi, opt.perfPoints, opt.perfRays);
          if (rep.placement) {
            // Undo the frame so the stored set is the part frame's, as every other consumer
            // expects. Drawing in the shape frame and unmapping is equivalent and simpler than
            // threading the matrix through the generator.
            for (auto& p : inFrame.points) {
              Point3D q;
              rep.placement->LocalToMaster(p.data(), q.data());
              p = q;
            }
            for (auto* rays : {&inFrame.outsideRays, &inFrame.insideRays}) {
              for (auto& r : *rays) {
                Point3D o;
                Point3D d;
                rep.placement->LocalToMaster(r.origin.data(), o.data());
                rep.placement->LocalToMasterVect(r.dir.data(), d.data());
                r.origin = o;
                r.dir = d;
              }
            }
          }
          samples = std::move(inFrame);
          partitionedBy = candidate;
        }
        delete rep.manager;
        gGeoManager = nullptr;
        if (!partitionedBy.empty()) {
          break;
        }
      }
      if (partitionedBy.empty()) {
        std::printf("  skip: no representation loaded\n");
        continue;
      }
      samples.partitionedBy = partitionedBy;
      std::printf(
        "  samples: %zu points (%.1f%% inside), %zu outside rays, %zu inside rays, "
        "partitioned by '%s'; raster %d x %d x %zu beams = %zu rays\n",
        samples.points.size(),
        100. * static_cast<double>(samples.insidePoints) /
          static_cast<double>(std::max<size_t>(1, samples.points.size())),
        samples.outsideRays.size(), samples.insideRays.size(), partitionedBy.c_str(),
        raster.n, raster.n, raster.beams.size(), raster.rays.size());

      json partJson;
      partJson["id"] = part.id;
      partJson["model"] = part.model;
      partJson["partitionedBy"] = partitionedBy;
      partJson["insideFraction"] = static_cast<double>(samples.insidePoints) /
                                   static_cast<double>(std::max<size_t>(1, samples.points.size()));
      partJson["bboxSource"] = bboxSource;
      json repsJson = json::array();

      for (const auto& candidate : allRepresentations()) {
        const std::string& source = sourceFor(part, candidate);
        if (!opt.representations.count(candidate) || !fileExists(source)) {
          continue;
        }
        LoadedRep rep = loadRepresentation(candidate, source, part.id, opt.flatSplitDepth,
                                           opt.flatMinBoxFraction);
        if (!rep.ok) {
          std::printf("  [skip %s] would not load from %s\n", candidate.c_str(), source.c_str());
          delete rep.manager;
          gGeoManager = nullptr;
          continue;
        }
        const QuerySamples local = toShapeFrame(samples, rep.placement.get());
        std::printf("  --- %-8s %-22s (%lld %s, load %.3f s + close %.3f s) ---\n", candidate.c_str(),
                    rep.shape->ClassName(), rep.structural.primitives,
                    candidate == "mesh"      ? "triangles"
                    : candidate == "surface" ? "patches"
                    : candidate == "flatcsg" ? "cells"
                                             : "leaves",
                    rep.loadSeconds, rep.closeSeconds);

        const TimingStat contains = timeContainsPass(rep.shape, local, opt.perfWarmup, opt.perfPasses);
        const TimingStat safety = timeSafetyPass(rep.shape, local, opt.perfWarmup, opt.perfPasses);
        const TimingStat distOut = timeDistOutPass(rep.shape, local, opt.perfWarmup, opt.perfPasses);
        const TimingStat distIn = timeDistInPass(rep.shape, local, opt.perfWarmup, opt.perfPasses);
        printTiming("Contains", contains);
        printTiming("Safety", safety);
        printTiming("DistFromOutside", distOut);
        printTiming("DistFromInside", distIn);

        // Full geantino transport over the raster, timed the same way: several complete passes,
        // median reported. This is the number a simulation actually pays, and it is the only one
        // that composes the four kernels in the order a transport does.
        Robustness statsTransport;
        long long crossings = 0;
        const TimingStat transport =
          timePasses(static_cast<long long>(raster.rays.size()), opt.perfWarmup, opt.perfPasses, [&]() {
            uint64_t acc = 0;
            Robustness s;
            long long found = 0;
            for (const auto& ray : raster.rays) {
              Point3D o;
              Point3D d;
              toShapeFrame(rep.placement.get(), ray.origin, ray.dir, o, d);
              const auto list = stepWithShapeApi(rep.shape, o, d, ray.tMax, opt.step, s);
              found += static_cast<long long>(list.size());
              acc += list.size();
            }
            statsTransport = s;
            crossings = found;
            return acc;
          });
        // `crossings` is set from the last pass; every pass sees the same rays, so it is the
        // per-pass crossing count and the ns/crossing below is exact rather than averaged over a
        // varying denominator. It is counted from the returned lists rather than from the
        // Robustness bookkeeping, which only fills in `crossings` when the per-ray audit runs --
        // and the audit is deliberately NOT run inside a timed pass, because Contains() at every
        // interval midpoint would put a fifth kernel into a transport measurement.
        const double nsPerCrossing =
          crossings > 0 ? transport.medianNsPerCall * static_cast<double>(raster.rays.size()) /
                            static_cast<double>(crossings)
                        : 0.;
        std::printf(
          "      %-14s %9.1f ns/ray   [%9.1f .. %9.1f, spread %5.1f%%]  %.1f ns/crossing "
          "(%lld crossings, %lld steps)\n",
          "transport", transport.medianNsPerCall, transport.minNsPerCall,
          transport.maxNsPerCall, 100. * transport.spread, nsPerCrossing, crossings,
          statsTransport.steps);

        const MemorySnapshot total{rep.loadDelta.residentBytes + rep.closeDelta.residentBytes,
                                   rep.loadDelta.heapInUseBytes + rep.closeDelta.heapInUseBytes};
        std::printf(
          "      memory: structural %lld B (%s)\n"
          "              sidecar on disk %lld B | measured heap +%lld B (load %lld + close "
          "%lld) | resident +%lld B\n",
          rep.structural.bytes, rep.structural.formula.c_str(),
          rep.structural.sidecarBytes, total.heapInUseBytes, rep.loadDelta.heapInUseBytes,
          rep.closeDelta.heapInUseBytes, total.residentBytes);
        if (candidate == "mesh" && !rep.meshClosedBody) {
          std::printf(
            "      *** meshClosedBody = FALSE: this mesh is INVALID, not merely "
            "inaccurate. Read no accuracy column of this row as a safety statement. ***\n");
        }

        json repJson;
        repJson["name"] = candidate;
        repJson["source"] = source;
        repJson["shapeClass"] = rep.shape->ClassName();
        repJson["primitives"] = rep.structural.primitives;
        repJson["loadSeconds"] = rep.loadSeconds;
        repJson["closeSeconds"] = rep.closeSeconds;
        repJson["structuralBytes"] = rep.structural.bytes;
        repJson["structuralFormula"] = rep.structural.formula;
        repJson["sidecarBytes"] = rep.structural.sidecarBytes;
        repJson["heapBytesLoad"] = rep.loadDelta.heapInUseBytes;
        repJson["heapBytesClose"] = rep.closeDelta.heapInUseBytes;
        repJson["heapBytesTotal"] = total.heapInUseBytes;
        repJson["residentBytesTotal"] = total.residentBytes;
        repJson["capacity"] = rep.shape->Capacity();
        repJson["placed"] = (rep.placement != nullptr);
        repJson["contains"] = timingToJson(contains);
        repJson["safety"] = timingToJson(safety);
        repJson["distFromOutside"] = timingToJson(distOut);
        repJson["distFromInside"] = timingToJson(distIn);
        repJson["transport"] = timingToJson(transport);
        repJson["transportNsPerCrossing"] = nsPerCrossing;
        repJson["transportCrossings"] = crossings;
        repJson["transportSteps"] = statsTransport.steps;
        repJson["transportUnterminated"] = statsTransport.unterminated;
        repJson["transportParityMismatch"] = statsTransport.parityMismatchIntervals;
        if (candidate == "mesh") {
          repJson["meshClosedBody"] = rep.meshClosedBody;
        }
        if (rep.surfaceSolid != nullptr) {
          repJson["localise"] = localiseSurfaceSolid(rep.surfaceSolid, local, opt.perfWarmup,
                                                     opt.perfPasses);
        }
        if (rep.flatSolid != nullptr) {
          // The two counts the crossover is regressed against (Design_FlatCSGSolid.md section 9),
          // plus the box structure the split knobs move.
          long long active = 0;
          long long worst = 0;
          for (int i = 0; i < rep.flatSolid->GetNboxes(); ++i) {
            const long long n = rep.flatSolid->GetBox(i).nActive;
            active += n;
            worst = std::max(worst, n);
          }
          repJson["flatCells"] = rep.flatSolid->GetNcells();
          repJson["flatHalfspaces"] = rep.flatSolid->GetNhalfspaces();
          repJson["flatBoxes"] = rep.flatSolid->GetNboxes();
          repJson["flatActiveTotal"] = active;
          repJson["flatActiveMean"] =
            rep.flatSolid->GetNboxes() > 0
              ? static_cast<double>(active) / static_cast<double>(rep.flatSolid->GetNboxes())
              : 0.;
          repJson["flatActiveMax"] = worst;
          repJson["flatBVHBytes"] = static_cast<long long>(rep.flatSolid->GetBVHMemory());
          repJson["flatSplitDepth"] = opt.flatSplitDepth;
          repJson["flatMinBoxFraction"] = opt.flatMinBoxFraction;
          repJson["flatCloseSeconds"] = rep.closeSeconds;
        }
        repsJson.push_back(std::move(repJson));
        delete rep.manager;
        gGeoManager = nullptr;
      }
      partJson["representations"] = std::move(repsJson);
      report.push_back(std::move(partJson));
      std::printf("\n");
    }
    if (!opt.jsonOut.empty()) {
      std::ofstream out(opt.jsonOut);
      out << report.dump(1);
      std::printf("\nreport: %s\n", opt.jsonOut.c_str());
    }
    return 0;
  }

  for (const auto& part : parts) {
    std::printf("=== %s (%s) ===\n", part.id.c_str(), part.model.c_str());
    json partJson;
    partJson["id"] = part.id;
    partJson["model"] = part.model;

    // ---- the raster window -------------------------------------------------------------
    // In scoring mode it comes from the oracle's answer file, so the two sides cannot possibly be
    // asking about different rays; otherwise it is built here from the tightest containing
    // bounding box the part has.
    Raster raster;
    OracleCrossings oracle;
    std::string bboxSource = "?";
    if (!opt.refCrossings.empty()) {
      try {
        oracle = loadOracleCrossings(opt.refCrossings, part.id);
      } catch (const std::exception& e) {
        std::cerr << "  error reading crossings: " << e.what() << "\n";
        continue;
      }
      if (!oracle.has) {
        std::printf("  skip: no crossings file for this part in %s\n", opt.refCrossings.c_str());
        continue;
      }
      raster = oracle.raster;
      opt.step.matchTolerance = std::max(oracle.tolerance, 1.e-6);
      std::printf(
        "  oracle: %s tolerance=%.3g capacity=%.6g cm^3 chordVolume=%.6g cm^3 "
        "(%lld ambiguous ray(s))\n",
        oracle.valid ? "valid" : "*** NOT BRepCheck-VALID ***", oracle.tolerance,
        oracle.capacity, oracle.volumeChord, oracle.ambiguousRays);
    } else {
      Point3D lo{};
      Point3D hi{};
      if (!resolveBoundingBox(part, opt, lo, hi, bboxSource)) {
        std::printf("  skip: no representation could supply a bounding box\n");
        continue;
      }
      raster = buildRaster(lo, hi, opt.raster, beams, opt.margin);
      std::printf(
        "  raster: %d x %d x %zu beam(s) = %zu rays (tilt %.3g deg); window from the "
        "'%s' bounding box + %.3g cm, cross-section excess %.3g\n",
        raster.n, raster.n, raster.beams.size(), raster.rays.size(), opt.tiltDegrees,
        bboxSource.c_str(), raster.transverseMargin, raster.windowExcess.front());
    }

    // `--dump-rays` writes the raster and stops: the oracle answers it next, and the scoring pass
    // then reads the rays back from the oracle's file. Nothing is stepped here.
    if (!opt.dumpRays.empty()) {
      writeRays(opt.dumpRays, part.id, raster, bboxSource);
      continue;
    }

    // ---- representations ---------------------------------------------------------------
    struct RepSpec {
      std::string name;
      std::string source;
    };
    std::vector<RepSpec> specs;
    if (opt.representations.count("surface") && fileExists(part.surfaces)) {
      specs.push_back({"surface", part.surfaces});
    }
    if (opt.representations.count("mesh") && fileExists(part.facets)) {
      specs.push_back({"mesh", part.facets});
    }
    if (opt.representations.count("shape") && fileExists(part.shape)) {
      specs.push_back({"shape", part.shape});
    }
    if (opt.representations.count("flatcsg") && fileExists(part.flatcsg)) {
      specs.push_back({"flatcsg", part.flatcsg});
    }
    if (specs.empty()) {
      std::printf("  skip: no representation available\n");
      continue;
    }

    json repsJson = json::array();

    for (const auto& spec : specs) {
      // A fresh TGeoManager per representation: it owns the shape (TGeoShape registers itself in
      // gGeoManager on construction, so any other arrangement double-frees) and it carries the
      // one-part world mode (b) transports through.
      auto* manager = new TGeoManager(("xray_" + spec.name).c_str(), "X-ray benchmark world");
      TGeoShape* shape = nullptr;
      // The shape's own frame, when it is not the part frame. Mode (a) transforms each ray into
      // it; mode (b) puts it on the node. Owned here: the manager owns the shape, not the matrix.
      std::unique_ptr<TGeoHMatrix> placement;
      double loadSeconds = 0.;
      int primitives = -1;
      const char* primitiveKind = "";
      const auto tLoad0 = std::chrono::steady_clock::now();
      if (spec.name == "surface") {
        auto* solid = new O2BVHSurfaceSolid(part.id.c_str());
        if (!LoadSurfaceSolid(spec.source, *solid)) {
          std::printf("  [skip %s] LoadSurfaceSolid failed for %s\n", spec.name.c_str(),
                      spec.source.c_str());
          delete manager;
          gGeoManager = nullptr;
          continue;
        }
        solid->CloseShape(true);
        primitives = solid->GetNsurfaces();
        primitiveKind = "patches";
        shape = solid;
      } else if (spec.name == "mesh") {
        auto* solid = new O2Tessellated(part.id.c_str());
        if (!LoadFacetSolid(spec.source, *solid)) {
          std::printf("  [skip %s] LoadFacetSolid failed for %s\n", spec.name.c_str(),
                      spec.source.c_str());
          delete manager;
          gGeoManager = nullptr;
          continue;
        }
        solid->CloseShape();
        primitives = solid->GetNfacets();
        primitiveKind = "triangles";
        shape = solid;
      } else if (spec.name == "flatcsg") {
        auto* solid = new O2FlatCSG(part.id.c_str());
        if (opt.flatSplitDepth >= 0) {
          solid->SetSplitDepth(opt.flatSplitDepth);
        }
        if (opt.flatMinBoxFraction >= 0.) {
          solid->SetMinBoxFraction(opt.flatMinBoxFraction);
        }
        if (!LoadFlatCSG(spec.source, *solid)) {
          std::printf("  [skip %s] LoadFlatCSG failed for %s\n", spec.name.c_str(),
                      spec.source.c_str());
          delete manager;
          gGeoManager = nullptr;
          continue;
        }
        solid->CloseShape();
        if (!solid->IsClosed()) {
          std::printf(
            "  [skip %s] CloseShape refused %s, so the shape would answer through its "
            "_Loop twins and the row would not be the accelerated path\n",
            spec.name.c_str(), spec.source.c_str());
          delete manager;
          gGeoManager = nullptr;
          continue;
        }
        primitives = solid->GetNcells();
        primitiveKind = "cells";
        shape = solid;
      } else {
        std::string error;
        shape = loadShapeFromRootFile(spec.source, &error);
        if (shape == nullptr) {
          std::printf("  [skip %s] %s\n", spec.name.c_str(), error.c_str());
          delete manager;
          gGeoManager = nullptr;
          continue;
        }
        placement.reset(loadShapePlacementFromRootFile(spec.source));
        primitiveKind = shape->ClassName();
      }
      loadSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - tLoad0).count();

      const auto* box = dynamic_cast<const TGeoBBox*>(shape);

      std::printf("  --- %-8s %-28s (%d %s, load %.3f s) ---\n", spec.name.c_str(),
                  shape->ClassName(), primitives, primitiveKind, loadSeconds);

      json repJson;
      repJson["name"] = spec.name;
      repJson["source"] = spec.source;
      repJson["shapeClass"] = shape->ClassName();
      repJson["primitives"] = primitives;
      repJson["primitiveKind"] = primitiveKind;
      repJson["loadSeconds"] = loadSeconds;
      repJson["capacity"] = shape->Capacity();
      repJson["placed"] = (placement != nullptr);

      // ---- mode (a): the shape API ------------------------------------------------------
      Robustness statsA;
      std::vector<double> insideByAxisA(raster.beams.size(), 0.);
      std::vector<std::vector<Crossing>> listsA(raster.rays.size());
      ListComparison vsOracleA;
      {
        const auto t0 = std::chrono::steady_clock::now();
        for (size_t i = 0; i < raster.rays.size(); ++i) {
          const auto& ray = raster.rays[i];
          const double before = statsA.insideLength;
          Point3D o;
          Point3D d;
          toShapeFrame(placement.get(), ray.origin, ray.dir, o, d);
          listsA[i] = stepWithShapeApi(shape, o, d, ray.tMax, opt.step, statsA);
          auditCrossingList(listsA[i], shape, o, d, ray.tMax, opt.step, statsA);
          insideByAxisA[ray.beam] += statsA.insideLength - before;
        }
        statsA.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      }
      repJson["modeA"] = robustnessToJson(statsA);
      repJson["modeA"]["volumeChordCm3"] = chordVolume(raster, insideByAxisA);
      json perAxisA = json::object();
      for (size_t b = 0; b < raster.beams.size(); ++b) {
        perAxisA[raster.beams[b].label] = insideByAxisA[b] * raster.cellArea[b];
      }
      repJson["modeA"]["volumeChordPerAxisCm3"] = perAxisA;
      if (oracle.has) {
        for (size_t i = 0; i < raster.rays.size() && i < oracle.perRay.size(); ++i) {
          if (oracle.ambiguous[i]) {
            continue; // OCCT declined somewhere along this ray; there is no ground truth to score
          }
          compareLists(listsA[i], oracle.perRay[i], raster.rays[i].origin, raster.rays[i].dir,
                       opt.step.matchTolerance, vsOracleA);
        }
        repJson["modeA"]["vsOracle"] = comparisonToJson(vsOracleA);
      }
      std::printf(
        "    (a) shape API : %lld rays, %lld crossings, %.4f s | zero=%lld stall=%lld "
        "nonAdv=%lld cap=%lld unterm=%lld odd=%lld dup=%lld parity=%lld\n",
        statsA.rays, statsA.crossings, statsA.seconds, statsA.zeroLengthSteps,
        statsA.unstickPushes, statsA.nonAdvancingSteps, statsA.iterationCapHits,
        statsA.unterminated, statsA.oddCrossingLists, statsA.duplicateCrossings,
        statsA.parityMismatchIntervals);
      if (oracle.has) {
        std::printf(
          "        vs OCCT : %lld/%lld rays identical, LOST=%lld extra=%lld "
          "displaced=%lld kind=%lld worst dt=%.3g cm\n",
          vsOracleA.raysIdentical, vsOracleA.rays, vsOracleA.missing, vsOracleA.extra,
          vsOracleA.displaced, vsOracleA.kindMismatch, vsOracleA.worstDeltaT);
        if (!vsOracleA.worstReason.empty() && vsOracleA.worstReason != "deltaT") {
          std::printf("        worst   : %s at o=(%.6g, %.6g, %.6g) d=(%.4g, %.4g, %.4g)\n",
                      vsOracleA.worstReason.c_str(), vsOracleA.worstOrigin[0],
                      vsOracleA.worstOrigin[1], vsOracleA.worstOrigin[2], vsOracleA.worstDir[0],
                      vsOracleA.worstDir[1], vsOracleA.worstDir[2]);
        }
      }
      std::printf("        volume  : chord integral %.8g cm^3 (Capacity %.8g)\n",
                  repJson["modeA"]["volumeChordCm3"].get<double>(), shape->Capacity());

      // ---- mode (b): the real navigator -------------------------------------------------
      if (!opt.skipNavigator && box != nullptr) {
        Robustness statsB;
        std::vector<double> insideByAxisB(raster.beams.size(), 0.);
        ListComparison vsOracleB;
        ListComparison aVsB;
        // The world must contain the part AND every ray of the raster, start to finish. Deriving
        // it from the axis-aligned window is not enough once the beams are tilted: a rotated
        // lattice reaches outside the part's own box, and the first version of this loop reported
        // 5358 lost crossings at a 27 degree tilt that were entirely its own undersized world.
        Point3D wMin;
        Point3D wMax;
        placedBox(*box, placement.get(), wMin, wMax);
        for (const auto& ray : raster.rays) {
          for (int k = 0; k < 3; ++k) {
            const double end = ray.origin[k] + ray.tMax * ray.dir[k];
            wMin[k] = std::min({wMin[k], ray.origin[k], end});
            wMax[k] = std::max({wMax[k], ray.origin[k], end});
          }
        }
        NavigatorTransport transport(manager, shape, wMin, wMax, placement.get());
        const auto t0 = std::chrono::steady_clock::now();
        for (size_t i = 0; i < raster.rays.size(); ++i) {
          const auto& ray = raster.rays[i];
          const double before = statsB.insideLength;
          auto listB = transport.transport(ray.origin, ray.dir, ray.tMax, opt.step, statsB);
          // The shape is handed in here as well, deliberately: in mode (b) the parity audit
          // compares the NAVIGATOR's crossing list against the SHAPE's own Contains(), which is a
          // genuine cross-check between the two and not a tautology.
          Point3D o;
          Point3D d;
          toShapeFrame(placement.get(), ray.origin, ray.dir, o, d);
          auditCrossingList(listB, shape, o, d, ray.tMax, opt.step, statsB);
          insideByAxisB[ray.beam] += statsB.insideLength - before;
          if (oracle.has && i < oracle.perRay.size() && !oracle.ambiguous[i]) {
            compareLists(listB, oracle.perRay[i], ray.origin, ray.dir, opt.step.matchTolerance,
                         vsOracleB);
          }
          compareLists(listB, listsA[i], ray.origin, ray.dir, opt.step.matchTolerance, aVsB);
        }
        statsB.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        repJson["modeB"] = robustnessToJson(statsB);
        repJson["modeB"]["volumeChordCm3"] = chordVolume(raster, insideByAxisB);
        if (oracle.has) {
          repJson["modeB"]["vsOracle"] = comparisonToJson(vsOracleB);
        }
        repJson["modeAvsB"] = comparisonToJson(aVsB);
        std::printf(
          "    (b) navigator: %lld rays, %lld crossings, %.4f s | zero=%lld nonAdv=%lld "
          "cap=%lld unterm=%lld odd=%lld dup=%lld noTransition=%lld outsideWorld=%lld\n",
          statsB.rays, statsB.crossings, statsB.seconds, statsB.zeroLengthSteps,
          statsB.nonAdvancingSteps, statsB.iterationCapHits, statsB.unterminated,
          statsB.oddCrossingLists, statsB.duplicateCrossings,
          statsB.boundaryWithoutTransition, statsB.originOutsideWorld);
        if (oracle.has) {
          std::printf(
            "        vs OCCT : %lld/%lld rays identical, LOST=%lld extra=%lld "
            "displaced=%lld worst dt=%.3g cm\n",
            vsOracleB.raysIdentical, vsOracleB.rays, vsOracleB.missing, vsOracleB.extra,
            vsOracleB.displaced, vsOracleB.worstDeltaT);
        }
        std::printf(
          "        (a)vs(b): %lld/%lld rays identical, LOST=%lld extra=%lld "
          "displaced=%lld worst dt=%.3g cm\n",
          aVsB.raysIdentical, aVsB.rays, aVsB.missing, aVsB.extra, aVsB.displaced,
          aVsB.worstDeltaT);
        std::printf("        volume  : chord integral %.8g cm^3\n",
                    repJson["modeB"]["volumeChordCm3"].get<double>());
      }

      repsJson.push_back(std::move(repJson));
      delete manager; // frees the shape, the world and the navigator with it
      gGeoManager = nullptr;
    }

    partJson["raster"] = {{"n", raster.n},
                          {"rays", raster.rays.size()},
                          {"cellArea", raster.cellArea},
                          {"transverseMargin", raster.transverseMargin},
                          {"windowExcess", raster.windowExcess},
                          {"windowMin", {raster.windowMin[0], raster.windowMin[1], raster.windowMin[2]}},
                          {"windowMax", {raster.windowMax[0], raster.windowMax[1], raster.windowMax[2]}}};
    if (oracle.has) {
      // Three volume numbers, and they answer three different questions. `volumeChordCm3` is
      // OCCT's OWN chord integral over these same rays, so comparing a candidate against it is
      // immune to the raster's own error; `capacity` is OCCT's exact volume, so
      // (oracle chord - capacity) IS the raster's achieved precision, measured at this density;
      // and each representation's `capacity` is the number the sample gate already scores.
      partJson["oracle"] = {{"tolerance", oracle.tolerance},
                            {"capacity", oracle.capacity},
                            {"volumeChordCm3", oracle.volumeChord},
                            {"chordVsExactRelative",
                             oracle.capacity != 0.
                               ? (oracle.volumeChord - oracle.capacity) / oracle.capacity
                               : 0.},
                            {"ambiguousRays", oracle.ambiguousRays},
                            {"valid", oracle.valid}};
      std::printf(
        "  raster precision: OCCT chord integral %.8g vs OCCT exact %.8g "
        "-> %.3e relative (N=%d, %zu rays)\n",
        oracle.volumeChord, oracle.capacity,
        partJson["oracle"]["chordVsExactRelative"].get<double>(), raster.n,
        raster.rays.size());
    }
    partJson["representations"] = std::move(repsJson);
    report.push_back(std::move(partJson));
  }

  if (!opt.jsonOut.empty()) {
    std::ofstream out(opt.jsonOut);
    out << report.dump(1);
    std::printf("\nreport: %s\n", opt.jsonOut.c_str());
  }
  return 0;
}
