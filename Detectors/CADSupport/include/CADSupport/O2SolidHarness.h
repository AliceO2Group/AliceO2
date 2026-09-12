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

/// \file O2SolidHarness.h
/// \brief Validation and timing harness for TGeoShape navigation, typed on plain `TGeoShape*`.

#ifndef ALICEO2_CADSUPPORT_O2SOLIDHARNESS_
#define ALICEO2_CADSUPPORT_O2SOLIDHARNESS_

#include "TGeoShape.h"

class TGeoMatrix;
class TGeoHMatrix;

#include <array>
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace o2
{
namespace cad
{
namespace harness
{

using Point3D = std::array<double, 3>;

struct Ray {
  Point3D origin{};
  Point3D dir{}; // unit vector by convention (TGeo contract); not renormalized by the harness
};

/// Parameters of `generateSamples`; the counts are targets, and a category may come back short.
struct SampleConfig {
  int nBulk = 2000;               ///< uniform points over the inflated bbox
  int nBoundary = 2000;           ///< points within `boundaryBand` of the reference surface
  int nInside = 1000;             ///< points accepted by the reference Contains()
  int nOutsideRays = 4000;        ///< rays from outside origins, for DistFromOutside
  int nInsideRays = 2000;         ///< rays from inside origins, for DistFromInside
  double bboxInflate = 0.15;      ///< fractional bbox half-extent padding for bulk/outside sampling
  double boundaryBand = -1.;      ///< absolute distance (cm); <0 auto-picks 1e-3 * bbox diagonal
  double aimedRayFraction = 0.5;  ///< fraction of rays aimed at a random interior bbox point rather
                                  ///< than an isotropic direction (keeps DistFromOutside hit rates
                                  ///< non-degenerate)
  int maxRejectionAttempts = 200; ///< attempts per accepted sample before giving up on that category
  uint64_t seed = 1;              ///< every SampleSet is fully determined by this and the bbox
};

struct SampleSet {
  Point3D bboxMin{};
  Point3D bboxMax{};
  std::vector<Point3D> bulkPoints;
  std::vector<Point3D> boundaryPoints;
  std::vector<Point3D> insidePoints;
  std::vector<Ray> outsideRays;
  std::vector<Ray> insideRays;
};

/// A deterministic sample set from `cfg.seed` and the bbox; \a reference, the trusted mesh, classifies the points.
SampleSet generateSamples(const TGeoShape* reference, const Point3D& bboxMin, const Point3D& bboxMax,
                          const SampleConfig& cfg = {});

// ---- Validation ----------------------------------------------------------------------------------

/// One worst-case disagreement, with enough state (point/direction/values) to reproduce it
/// directly outside the harness.
struct Offender {
  Point3D point{};
  Point3D dir{}; // zero for point-only queries (Contains, Safety)
  double candidateValue = 0.;
  double referenceValue = 0.;
  double deviation = 0.;
  double referenceSafety = 0.; // point queries: reference distance to its own surface
  double incidenceCosine = 1.; // ray queries: |cos| between ray and surface normal at the hit,
                               // i.e. how much surface uncertainty this ray amplifies
};

struct ValidationResult {
  size_t nSamples = 0;
  size_t nAgree = 0;
  size_t nMismatchWithinBand = 0;    // explainable by the reference's own imprecision (see below)
  size_t nMismatchMissedSurface = 0; // one side found no crossing where the other did
  size_t nMismatchUnexplained = 0;
  size_t nNoVerdict = 0;  // oracle mode only: the reference declined to answer
  size_t nRelabelled = 0; // ray queries, oracle mode: origins whose category the oracle
                          // contradicted, so the other TGeo entry point was asked
  double worstDeviation = 0.;
  std::vector<Offender> worstOffenders; // bounded by opt.maxOffenders, worst-first
};

/// `nMismatchMissedSurface` counts a candidate that misses a wall the reference hits, or tunnels
/// to a farther one; such a mismatch is never explained away as mesh chording.
struct ValidationOptions {
  double distanceTolerance = 1.e-6; ///< absolute agreement tolerance for distances (cm)
  double meshBand = 1.e-2;          ///< the reference's positional uncertainty (cm): chord sagitta or model tolerance
  /// Floor of the incidence cosine that scales the distance allowance, so a tangent ray cannot excuse an unbounded error.
  double minIncidenceCosine = 1.e-2;
  double stepmax = TGeoShape::Big();
  size_t maxOffenders = 10;
};

ValidationResult validateContains(const TGeoShape* candidate, const TGeoShape* reference,
                                  const std::vector<Point3D>& points, const ValidationOptions& opt = {});

ValidationResult validateDistFromOutside(const TGeoShape* candidate, const TGeoShape* reference,
                                         const std::vector<Ray>& rays, const ValidationOptions& opt = {});

ValidationResult validateDistFromInside(const TGeoShape* candidate, const TGeoShape* reference,
                                        const std::vector<Ray>& rays, const ValidationOptions& opt = {});

/// Check one shape's Safety() lower-bound contract against its own DistFrom* along six probe directions; never compares two shapes.
ValidationResult validateSafety(const TGeoShape* shape, const std::vector<Point3D>& points,
                                const ValidationOptions& opt = {});

// ---- Validation against the OpenCascade oracle: a disagreement beyond the model tolerance is a defect ----

/// `oracleState`: 1 inside, 0 outside, -1 declined; `oracleBoundaryDistance` may cover only a prefix of \a points.
ValidationResult validateContainsAgainstOracle(const TGeoShape* candidate,
                                               const std::vector<Point3D>& points,
                                               const std::vector<int>& oracleState,
                                               const std::vector<double>& oracleBoundaryDistance,
                                               const ValidationOptions& opt = {});

/// `oracleDistance`: the nearest positive crossing, or >= Big() for a miss. `oracleOriginState` (1, 0, -1), when present,
/// decides per ray which entry point is asked, and a -1 origin abstains; otherwise `wantInside` decides.
ValidationResult validateDistanceAgainstOracle(const TGeoShape* candidate,
                                               const std::vector<Ray>& rays,
                                               const std::vector<double>& oracleDistance,
                                               bool wantInside, const ValidationOptions& opt = {},
                                               const std::vector<int>& oracleOriginState = {});

/// Safety's contract against the oracle's exact distance: `0 <= safety <= trueDistance`.
ValidationResult validateSafetyAgainstOracle(const TGeoShape* candidate,
                                             const std::vector<Point3D>& points,
                                             const std::vector<double>& oracleBoundaryDistance,
                                             const ValidationOptions& opt = {});

// ---- Timing --------------------------------------------------------------------------------------

struct TimingResult {
  size_t nCalls = 0;
  double nsPerCall = 0.;
  uint64_t checksum = 0; ///< accumulated from the results so the optimizer cannot elide the calls
};

namespace detail
{
/// Checksum mixer the timing loops accumulate results through, so the optimizer cannot elide the
/// measured calls. Exposed only because `timeRayKernel` below is a template.
uint64_t mixDouble(uint64_t acc, double value);
} // namespace detail

/// Time a per-ray kernel `kernel(origin, dir)` exactly like the `timeDistFrom*` functions, e.g. a `_Loop` twin.
template <typename RayKernel>
TimingResult timeRayKernel(const std::vector<Ray>& rays, int warmupRepeats, int timedRepeats, RayKernel&& kernel)
{
  for (int warmup = 0; warmup < warmupRepeats; ++warmup) {
    for (const auto& ray : rays) {
      volatile double sink = kernel(ray.origin, ray.dir);
      (void)sink;
    }
  }
  uint64_t checksum = 0;
  const auto start = std::chrono::steady_clock::now();
  for (int repeat = 0; repeat < timedRepeats; ++repeat) {
    for (const auto& ray : rays) {
      checksum = detail::mixDouble(checksum, kernel(ray.origin, ray.dir));
    }
  }
  const auto stop = std::chrono::steady_clock::now();
  TimingResult result;
  result.nCalls = rays.size() * static_cast<size_t>(timedRepeats);
  const double nanoseconds = std::chrono::duration<double, std::nano>(stop - start).count();
  result.nsPerCall = result.nCalls > 0 ? nanoseconds / static_cast<double>(result.nCalls) : 0.;
  result.checksum = checksum;
  return result;
}

TimingResult timeContains(const TGeoShape* shape, const std::vector<Point3D>& points, int warmupRepeats,
                          int timedRepeats);
TimingResult timeDistFromOutside(const TGeoShape* shape, const std::vector<Ray>& rays, int warmupRepeats,
                                 int timedRepeats, double stepmax = TGeoShape::Big());
TimingResult timeDistFromInside(const TGeoShape* shape, const std::vector<Ray>& rays, int warmupRepeats,
                                int timedRepeats);
TimingResult timeSafety(const TGeoShape* shape, const std::vector<Point3D>& points, int warmupRepeats,
                        int timedRepeats);

// ---- The `shape_<part>.root` sidecar -------------------------------------------------------------
//
//   * one file per part, `shape_<VOL>_<LID>.root`, next to the part's other sidecars;
//   * one TGeoShape-derived object under the key "shape" (the first such key is the fallback);
//   * lengths in centimetres;
//   * an optional TGeoHMatrix under "placement" takes the shape's frame to the part's (`local -> part`);
//     no key means the identity;
//   * a TGeoCompositeShape is written whole and needs no TGeoManager.

/// Read the single TGeoShape of a `shape_<part>.root` sidecar; nullptr on failure, with the reason in `*error`. The caller owns it.
TGeoShape* loadShapeFromRootFile(const std::string& path, std::string* error = nullptr);

/// Read the shape's placement, or nullptr when there is none, meaning the identity. The caller owns it.
TGeoHMatrix* loadShapePlacementFromRootFile(const std::string& path);

/// Write a shape sidecar, with \a placement under "placement" unless it is null or the identity.
bool saveShapeToRootFile(const std::string& path, const TGeoShape& shape, std::string* error = nullptr);
bool saveShapeToRootFile(const std::string& path, const TGeoShape& shape,
                         const TGeoMatrix* placement, std::string* error);

} // namespace harness
} // namespace cad
} // namespace o2

#endif
