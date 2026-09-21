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

/// \file XRayTransport.h
/// \brief The X-ray transport benchmark's algorithms: stepping, auditing and comparing ordered
///        crossing lists.
///
/// Header-only, and deliberately NOT in CADSupport: this is a measuring instrument, and the
/// project's rule is that an instrument must not change the thing it measures. Putting it here
/// means `o2-bench-cadsupport-solid-harness` and the oracle gate are untouched -- not one
/// object file of the existing path is rebuilt differently.
///
/// It is a header rather than code inside runXRayBenchmark.cxx for one reason: the unit tests in
/// testBVHSurfaceSolid.cxx must exercise THE SAME stepping loop and THE SAME comparator that the
/// benchmark runs. A test against a second implementation of the same idea tests neither.

#ifndef ALICEO2_BASE_XRAYTRANSPORT_H_
#define ALICEO2_BASE_XRAYTRANSPORT_H_

#include "CADSupport/O2SolidHarness.h"

#include "TGeoShape.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <vector>

namespace o2
{
namespace cad
{
namespace xray
{

using o2::cad::harness::Point3D;

/// A single boundary crossing along a ray: the distance from the ray origin and whether the ray
/// is entering (+1) or leaving (-1) the solid there.
struct Crossing {
  double t = 0.;
  int kind = 0;
};

/// Everything a transport loop can do wrong that a single-shot distance query cannot express.
/// Every counter here is a *count of events*, never a rate, so two runs can be added.
struct Robustness {
  long long rays = 0;
  long long raysWithCrossings = 0;
  long long crossings = 0;
  long long steps = 0;
  long long zeroLengthSteps = 0;    ///< a step at or below `zeroStep` (default 1e-9 cm)
  long long nonAdvancingSteps = 0;  ///< the accumulated distance did not increase
  long long unstickPushes = 0;      ///< a stalled step that had to be nudged to continue
  long long iterationCapHits = 0;   ///< the loop hit `maxIter` without leaving the window
  long long unterminated = 0;       ///< the ray ended INSIDE the solid: entered and never left
  long long oddCrossingLists = 0;   ///< odd number of crossings (the same event, counted as the
                                    ///< brief names it; equal to `unterminated` by construction
                                    ///< in mode (a) and an independent number in mode (b))
  long long nonAlternating = 0;     ///< two consecutive crossings of the same kind
  long long duplicateCrossings = 0; ///< two crossings closer together than the match tolerance
  /// A parity mismatch whose midpoint is within the match tolerance of the boundary: excused,
  /// counted, and never folded into `parityMismatchIntervals`. Same principle as the sample
  /// gate's `nNoVerdict`.
  long long parityMismatchNearBoundary = 0;
  long long parityMismatchIntervals = 0;   ///< Contains() at an interval midpoint contradicts the
                                           ///< in/out state the crossing list implies. This is the
                                           ///< one check that is INDEPENDENT of the stepping: the
                                           ///< list alternates by construction in both modes, so
                                           ///< without it "non-alternating" could never fire.
  long long originInside = 0;              ///< a raster ray whose origin was not outside the solid
  long long boundaryWithoutTransition = 0; ///< mode (b): a boundary was crossed but the volume did
                                           ///< not change (a re-entry into the same volume)
  /// mode (b): the ray origin was not inside the navigator's world at all, so the transport never
  /// started; counted apart so a misconfigured world is not mistaken for a geometry defect.
  long long originOutsideWorld = 0;
  double insideLength = 0.; ///< summed inside-segment length, cm (the chord integral)
  double seconds = 0.;
};

struct StepConfig {
  /// Distance the point is advanced *past* a found crossing before the next query. This is the
  /// crux of a transport loop: land exactly on a face and the next query re-finds the same
  /// crossing at zero. Default 1e-9 cm = the kernel's own kRayTolerance, so the recorded crossing
  /// distances carry a known bias of at most (k-1) * push over a k-crossing ray, i.e. below 1e-8
  /// cm -- two orders under the 1e-6 cm comparison band.
  double push = 1.e-9;
  /// A step at or below this is a stall, not progress.
  double zeroStep = 1.e-9;
  /// What a stalled loop is nudged by to continue, mirroring what a navigator has to do. Every
  /// use is counted (`unstickPushes`): it is a repair, and a repair that is not counted is a lie.
  double unstickPush = 1.e-6;
  int maxIter = 512;
  /// Crossing-list match tolerance, cm. Set from the model's own declared tolerance where the
  /// oracle supplies one, floored here.
  double matchTolerance = 1.e-6;
};

// ------------------------------------------------------------------------------------------
// Mode (a): the direct shape-API stepping loop
// ------------------------------------------------------------------------------------------
//
// Contains() to establish the starting state, then alternating DistFromOutside/DistFromInside,
// advancing the point along the ray, until the accumulated distance leaves the raster window.
// `stepmax` is deliberately NOT used to bound the query: its semantics differ between shape
// implementations (some return the crossing, some return stepmax, some return Big), and this loop
// must be a measurement of the crossing list rather than of that convention. The window is
// enforced on the returned crossing distance instead.

/// The stepping loop, parameterised on the three kernels it calls.
///
/// Templated so the SAME loop can be driven by `O2BVHSurfaceSolid`'s BVH entry points and by its
/// non-BVH `_Loop` twins. That turns the project's existing single-query "BVH == _Loop" guard into
/// a transport-level one: every query after the first starts from a point the previous query put
/// on a boundary, so a traversal-order difference that is invisible on an isolated query can still
/// send the two down different sequences of states.
template <typename ContainsFn, typename DistOutFn, typename DistInFn>
std::vector<Crossing> stepCrossingsWithKernels(const Point3D& origin, const Point3D& dir,
                                               double tMax, const StepConfig& cfg,
                                               Robustness& stats, ContainsFn contains,
                                               DistOutFn distFromOutside, DistInFn distFromInside)
{
  std::vector<Crossing> crossings;
  double point[3] = {origin[0], origin[1], origin[2]};
  bool inside = contains(point);
  if (inside) {
    ++stats.originInside;
  }
  double t = 0.;
  int iter = 0;
  for (; iter < cfg.maxIter; ++iter) {
    const double step = inside ? distFromInside(point, dir.data()) : distFromOutside(point, dir.data());
    ++stats.steps;
    if (!(step < TGeoShape::Big())) {
      break; // no further crossing along this ray
    }
    const double tCross = t + step;
    if (tCross > tMax) {
      break; // beyond the raster window: not this ray's business
    }
    if (step <= cfg.zeroStep) {
      ++stats.zeroLengthSteps;
    }
    crossings.push_back({tCross, inside ? -1 : +1});
    inside = !inside;
    double advance = step + cfg.push;
    if (!(advance > 0.)) {
      ++stats.nonAdvancingSteps;
      advance = cfg.unstickPush;
      ++stats.unstickPushes;
    } else if (step <= cfg.zeroStep) {
      advance = step + cfg.unstickPush;
      ++stats.unstickPushes;
    }
    t += advance;
    if (t > tMax) {
      break;
    }
    for (int k = 0; k < 3; ++k) {
      point[k] = origin[k] + t * dir[k];
    }
  }
  if (iter >= cfg.maxIter) {
    ++stats.iterationCapHits;
  }
  if (inside) {
    ++stats.unterminated;
  }
  return crossings;
}

/// Mode (a): the same loop driven by the ordinary TGeoShape virtuals.
inline std::vector<Crossing> stepWithShapeApi(const TGeoShape* shape, const Point3D& origin,
                                              const Point3D& dir, double tMax,
                                              const StepConfig& cfg, Robustness& stats)
{
  return stepCrossingsWithKernels(
    origin, dir, tMax, cfg, stats, [shape](const double* p) { return shape->Contains(p); },
    [shape](const double* p, const double* d) {
      return shape->DistFromOutside(p, d, 3, TGeoShape::Big(), nullptr);
    },
    [shape](const double* p, const double* d) {
      return shape->DistFromInside(p, d, 3, TGeoShape::Big(), nullptr);
    });
}

/// Book the per-ray consistency properties of one crossing list. Split out because both modes and
/// the oracle's own answer go through it, so a defect in one cannot be excused by a different
/// bookkeeping in another.
inline void auditCrossingList(const std::vector<Crossing>& crossings, const TGeoShape* shape,
                              const Point3D& origin, const Point3D& dir, double tMax,
                              const StepConfig& cfg, Robustness& stats)
{
  ++stats.rays;
  stats.crossings += static_cast<long long>(crossings.size());
  if (!crossings.empty()) {
    ++stats.raysWithCrossings;
  }
  if (crossings.size() % 2 != 0) {
    ++stats.oddCrossingLists;
  }
  for (size_t i = 1; i < crossings.size(); ++i) {
    if (crossings[i].kind == crossings[i - 1].kind) {
      ++stats.nonAlternating;
    }
    if (std::fabs(crossings[i].t - crossings[i - 1].t) <= cfg.matchTolerance) {
      ++stats.duplicateCrossings;
    }
  }
  // The inside-segment length: the chord integral's contribution from this ray.
  for (size_t i = 0; i + 1 < crossings.size(); i += 2) {
    if (crossings[i].kind == +1 && crossings[i + 1].kind == -1) {
      stats.insideLength += crossings[i + 1].t - crossings[i].t;
    }
  }
  // The independent check. Both stepping modes produce an alternating list *by construction*, so
  // `nonAlternating` above can never fire on them; asking the shape's own Contains() at the
  // midpoint of every interval is the only way this instrument can contradict itself.
  if (shape != nullptr) {
    std::vector<double> edges;
    edges.push_back(0.);
    for (const auto& c : crossings) {
      edges.push_back(c.t);
    }
    edges.push_back(tMax);
    bool expectInside = false;
    for (size_t i = 0; i + 1 < edges.size(); ++i) {
      const double mid = 0.5 * (edges[i] + edges[i + 1]);
      if (edges[i + 1] - edges[i] > 8. * cfg.matchTolerance) {
        double p[3];
        for (int k = 0; k < 3; ++k) {
          p[k] = origin[k] + mid * dir[k];
        }
        const bool actuallyInside = shape->Contains(p);
        if (actuallyInside != expectInside) {
          // Classify before counting. A midpoint within the match tolerance of the boundary has no
          // defined answer on either side, exactly as the sample gate's `nNoVerdict` points do;
          // counting it as a contradiction would manufacture defects out of near-tangency.
          // Safety() is only paid for on a mismatch, which is rare.
          // Safety() must be asked with the state the shape ITSELF reports; asking it with the
          // state the crossing list expects makes a plain outside point look like a boundary
          // point (TGeoBBox::Safety(p, in=true) goes negative there) and silently excuses every
          // real contradiction. That mistake made this counter read 0 on a deliberately
          // truncated list.
          if (shape->Safety(p, actuallyInside ? kTRUE : kFALSE) <= cfg.matchTolerance) {
            ++stats.parityMismatchNearBoundary;
          } else {
            ++stats.parityMismatchIntervals;
          }
        }
      }
      expectInside = !expectInside;
    }
  }
}

/// How two crossing lists differ, with LOST and DISPLACED kept apart.
///
/// That separation is the whole localising value of comparing lists rather than aggregates. A
/// crossing the candidate never found is a wall a track walks through; a crossing it found half a
/// millimetre late is a step length that is slightly wrong. Both are defects, they have completely
/// different consequences for transport, and a single "disagreements" count merges them.
struct ListComparison {
  long long rays = 0;
  long long raysIdentical = 0;  ///< the whole ordered list matched, position and sense
  long long raysStructural = 0; ///< the lists have different lengths or senses
  long long matched = 0;
  long long displaced = 0; ///< same position in both lists, more than `tolerance` apart
  long long missing = 0;   ///< in the reference, absent from the candidate
  long long extra = 0;     ///< in the candidate, absent from the reference
  long long kindMismatch = 0;
  double worstDeltaT = 0.; ///< max |dt| over positionally matched crossings, cm
  Point3D worstOrigin{};
  Point3D worstDir{};
  std::string worstReason;
};

inline void compareLists(const std::vector<Crossing>& candidate, const std::vector<Crossing>& reference,
                         const Point3D& origin, const Point3D& dir, double tolerance, ListComparison& out)
{
  ++out.rays;
  bool sameShape = candidate.size() == reference.size();
  for (size_t i = 0; sameShape && i < candidate.size(); ++i) {
    sameShape = candidate[i].kind == reference[i].kind;
  }
  if (sameShape) {
    // Same number of crossings in the same order with the same senses: every difference is a
    // position, so report the positions and never manufacture a missing/extra pair out of one
    // displaced crossing.
    bool identical = true;
    for (size_t i = 0; i < candidate.size(); ++i) {
      const double delta = std::fabs(candidate[i].t - reference[i].t);
      ++out.matched;
      if (delta > tolerance) {
        ++out.displaced;
        identical = false;
      }
      if (delta > out.worstDeltaT) {
        out.worstDeltaT = delta;
        out.worstOrigin = origin;
        out.worstDir = dir;
        out.worstReason = delta > tolerance ? "displaced crossing" : "deltaT";
      }
    }
    out.raysIdentical += identical;
    return;
  }

  // Structurally different: walk both lists and attribute each unpaired crossing to the side it
  // came from. This is the branch that names a LOST wall.
  ++out.raysStructural;
  size_t i = 0;
  size_t j = 0;
  while (i < candidate.size() && j < reference.size()) {
    const double delta = candidate[i].t - reference[j].t;
    if (std::fabs(delta) <= tolerance) {
      ++out.matched;
      if (candidate[i].kind != reference[j].kind) {
        ++out.kindMismatch;
      }
      if (std::fabs(delta) > out.worstDeltaT) {
        out.worstDeltaT = std::fabs(delta);
      }
      ++i;
      ++j;
    } else if (delta < 0.) {
      ++out.extra;
      ++i;
    } else {
      ++out.missing;
      ++j;
    }
  }
  out.extra += static_cast<long long>(candidate.size() - i);
  out.missing += static_cast<long long>(reference.size() - j);
  if (out.worstReason.empty() || out.worstReason == "deltaT" ||
      out.worstReason == "displaced crossing") {
    out.worstOrigin = origin;
    out.worstDir = dir;
    out.worstReason = reference.size() > candidate.size() ? "MISSING crossing" : "EXTRA crossing";
  }
}

struct RayDef {
  Point3D origin{};
  Point3D dir{};
  double tMax = 0.;
  int beam = 0; ///< index into Raster::beams
};

/// One parallel beam: a direction and the orthonormal frame the lattice is laid out in.
///
/// Beams are directions, not axes, because a tilted beam produces generic ray/surface
/// configurations that an axis-aligned beam misses.
struct Beam {
  Point3D dir{};
  Point3D u{};
  Point3D v{};
  std::string label;
};

struct Raster {
  int n = 0;
  std::vector<Beam> beams;
  std::vector<double> cellArea; ///< per beam, cm^2
  /// Fractional excess of the raster window's cross-section over the part's own bounding box,
  /// per beam. Not decoration: at finite N a window wider than the silhouette biases the chord
  /// integral UPWARD by about this much, because the cells straddling the silhouette are counted
  /// whole. It is reported next to every volume so the systematic is never invisible.
  std::vector<double> windowExcess;
  std::vector<RayDef> rays;
  double transverseMargin = 0.;
  Point3D windowMin{}; ///< the part bbox plus the margin, in world coordinates (the world box)
  Point3D windowMax{};
};

inline double dot3(const Point3D& a, const Point3D& b)
{
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

inline Point3D normalize3(const Point3D& a)
{
  const double norm = std::sqrt(dot3(a, a));
  return {a[0] / norm, a[1] / norm, a[2] / norm};
}

inline Point3D cross3(const Point3D& a, const Point3D& b)
{
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}

/// The beams for `axesSpec` (a subset of x, y, z), each tilted by `tiltDegrees`.
///
/// At tilt 0 the frame is exactly the two remaining coordinate axes, so an axis-aligned box's
/// chord integral stays EXACT (every cell centre is inside its own bounding box; see buildRaster).
/// A non-zero tilt rotates the beam by `tilt` about one transverse axis and by 0.618 * tilt about
/// the other -- an irrational-looking ratio on purpose, so no beam lands on a symmetry plane of a
/// part that was drawn on a coordinate grid.
/// `count` beams spread over the sphere by the Fibonacci spiral, deterministic and seed-free.
///
/// This exists because of a measurement, not for completeness. A parallel-beam raster is
/// DIRECTION-POOR: three axes (or three tilted axes) are three directions, however many rays are
/// fired. The known torus quartic defect fires on a configuration that depends on the ray
/// DIRECTION, so it is invisible to a 3-beam raster of 27648 rays and visible to a fan of many
/// directions. Impact-parameter density and direction density are different resolutions and a
/// benchmark that only has the first will report a clean sheet on a defect it cannot see.
inline std::vector<Beam> buildFanBeams(int count)
{
  std::vector<Beam> beams;
  const double golden = 3.14159265358979323846 * (3. - std::sqrt(5.));
  for (int i = 0; i < count; ++i) {
    // Only the upper hemisphere is needed: a beam and its reverse sample the same lines.
    const double z = (count == 1) ? 1. : 1. - static_cast<double>(i) / static_cast<double>(count);
    const double radius = std::sqrt(std::max(0., 1. - z * z));
    const double theta = golden * i;
    Beam beam;
    beam.dir = normalize3({radius * std::cos(theta), radius * std::sin(theta), z});
    // A transverse frame: Gram-Schmidt off whichever axis the beam is least aligned with.
    int least = 0;
    for (int k = 1; k < 3; ++k) {
      if (std::fabs(beam.dir[k]) < std::fabs(beam.dir[least])) {
        least = k;
      }
    }
    Point3D seed{};
    seed[least] = 1.;
    const double projection = dot3(seed, beam.dir);
    beam.u = normalize3({seed[0] - projection * beam.dir[0], seed[1] - projection * beam.dir[1],
                         seed[2] - projection * beam.dir[2]});
    beam.v = cross3(beam.dir, beam.u);
    beam.label = "f" + std::to_string(i);
    beams.push_back(std::move(beam));
  }
  return beams;
}

inline std::vector<Beam> buildBeams(const std::string& axesSpec, double tiltDegrees)
{
  std::vector<Beam> beams;
  const double t = std::tan(tiltDegrees * 3.14159265358979323846 / 180.);
  for (const char c : axesSpec) {
    int axis = -1;
    if (c == 'x' || c == 'X') {
      axis = 0;
    } else if (c == 'y' || c == 'Y') {
      axis = 1;
    } else if (c == 'z' || c == 'Z') {
      axis = 2;
    } else {
      continue;
    }
    const int iu = (axis + 1) % 3;
    const int iv = (axis + 2) % 3;
    Point3D w{};
    Point3D u{};
    Point3D v{};
    w[axis] = 1.;
    u[iu] = 1.;
    v[iv] = 1.;
    Beam beam;
    if (t == 0.) {
      beam.dir = w;
      beam.u = u;
      beam.v = v;
      beam.label = std::string(1, "xyz"[axis]);
    } else {
      Point3D dir{w[0] + t * u[0] + 0.618 * t * v[0], w[1] + t * u[1] + 0.618 * t * v[1],
                  w[2] + t * u[2] + 0.618 * t * v[2]};
      beam.dir = normalize3(dir);
      // Gram-Schmidt the transverse frame off the original in-plane axis.
      Point3D uu{u[0] - dot3(u, beam.dir) * beam.dir[0], u[1] - dot3(u, beam.dir) * beam.dir[1],
                 u[2] - dot3(u, beam.dir) * beam.dir[2]};
      beam.u = normalize3(uu);
      beam.v = cross3(beam.dir, beam.u);
      beam.label = std::string(1, "xyz"[axis]) + "+t";
    }
    beams.push_back(std::move(beam));
  }
  return beams;
}

/// The transverse window and the longitudinal start are deliberately DECOUPLED.
///
/// Transverse: the window is the bounding box's own extent IN THE BEAM'S FRAME plus
/// `transverseMargin`, which should be as small as the bounding box's reliability allows, because
/// the window excess is a first-order systematic on the volume.
///
/// Longitudinal: the ray must start strictly OUTSIDE the solid, or Contains() at the origin is a
/// coin toss on the face and the whole transport starts in the wrong state. That margin is
/// therefore generous and costs nothing -- it is along the ray, not across it.
inline Raster buildRaster(const Point3D& bboxMin, const Point3D& bboxMax, int n,
                          const std::vector<Beam>& beams, double transverseMargin)
{
  Raster raster;
  raster.n = n;
  raster.beams = beams;
  raster.transverseMargin = transverseMargin;
  for (int k = 0; k < 3; ++k) {
    raster.windowMin[k] = bboxMin[k] - transverseMargin;
    raster.windowMax[k] = bboxMax[k] + transverseMargin;
  }
  for (const auto& beam : beams) {
    // Project the eight bounding-box corners into the beam frame; the window is their extent.
    double lo[3] = {1.e300, 1.e300, 1.e300};
    double hi[3] = {-1.e300, -1.e300, -1.e300};
    for (int corner = 0; corner < 8; ++corner) {
      const Point3D p{(corner & 1) ? bboxMax[0] : bboxMin[0], (corner & 2) ? bboxMax[1] : bboxMin[1],
                      (corner & 4) ? bboxMax[2] : bboxMin[2]};
      const double coordinate[3] = {dot3(p, beam.u), dot3(p, beam.v), dot3(p, beam.dir)};
      for (int k = 0; k < 3; ++k) {
        lo[k] = std::min(lo[k], coordinate[k]);
        hi[k] = std::max(hi[k], coordinate[k]);
      }
    }
    const double uLo = lo[0] - transverseMargin;
    const double vLo = lo[1] - transverseMargin;
    const double du = (hi[0] - lo[0] + 2. * transverseMargin) / n;
    const double dv = (hi[1] - lo[1] + 2. * transverseMargin) / n;
    raster.cellArea.push_back(du * dv);
    const double bboxArea = (hi[0] - lo[0]) * (hi[1] - lo[1]);
    raster.windowExcess.push_back(bboxArea > 0. ? (du * dv * n * n) / bboxArea - 1. : 0.);
    const double extent = hi[2] - lo[2];
    const double lead = 0.05 * extent + 1.e-3;
    const double wStart = lo[2] - lead;
    const int index = static_cast<int>(raster.cellArea.size()) - 1;
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        const double uu = uLo + (i + 0.5) * du;
        const double vv = vLo + (j + 0.5) * dv;
        RayDef ray;
        ray.beam = index;
        for (int k = 0; k < 3; ++k) {
          ray.origin[k] = uu * beam.u[k] + vv * beam.v[k] + wStart * beam.dir[k];
          ray.dir[k] = beam.dir[k];
        }
        ray.tMax = extent + 2. * lead;
        raster.rays.push_back(ray);
      }
    }
  }
  return raster;
}

/// Each beam is an independent estimate of the same volume; the reported number is their mean and
/// the per-beam spread is the honest error bar.
inline double chordVolume(const Raster& raster, const std::vector<double>& insideLengthPerBeam)
{
  double sum = 0.;
  size_t used = 0;
  for (size_t i = 0; i < raster.beams.size() && i < insideLengthPerBeam.size(); ++i) {
    sum += insideLengthPerBeam[i] * raster.cellArea[i];
    ++used;
  }
  return used > 0 ? sum / static_cast<double>(used) : 0.;
}

} // namespace xray
} // namespace cad
} // namespace o2

#endif
