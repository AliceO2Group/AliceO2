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

/// \file RepresentationBench.h
/// \brief Per-call cost, memory and the synthetic boolean ladder: the measuring parts of the
///        representation comparison.
///
/// Header-only, and deliberately NOT in CADSupport, for the same reason `XRayTransport.h` is
/// not: an instrument must not change the thing it measures. Nothing in the gate path or in
/// `libO2CADSupport` is rebuilt differently because this file exists.
///
/// It is a header rather than code inside runXRayBenchmark.cxx so that the unit tests exercise
/// THE SAME timing loop, THE SAME memory probe and THE SAME ladder the benchmark reports from. A
/// test written against a second implementation of the same idea tests neither.
///
/// THREE THINGS THIS FILE IS CAREFUL ABOUT, each bought with a known way of getting it wrong:
///
///  1. **One wall clock is not a measurement.** Every kernel is timed over several complete
///     passes and reported as the MEDIAN with the min/max spread beside it, never as a single
///     elapsed time. A single pass on a shared interactive machine is a sample of the machine's
///     mood as much as of the kernel.
///  2. **The same questions, from the same sample sets, for every representation.** The point and
///     ray sets are built ONCE per part from a reference representation's own classification and
///     handed unchanged to all three. Letting each representation partition its own inside/outside
///     set would compare three different questions and call the answer a speed ratio.
///  3. **Two memory numbers, because they answer different questions.** A STRUCTURAL count (exact,
///     derived from the shape's own counters and element sizes) and a MEASURED resident/heap delta
///     (noisy, allocator-dependent, but the only one that sees what the shape actually asked the
///     allocator for). Where they disagree the structural one is the exact statement and the
///     measured one is the honest one; both are printed.

#ifndef ALICEO2_BASE_REPRESENTATIONBENCH_H_
#define ALICEO2_BASE_REPRESENTATIONBENCH_H_

#include "CADSupport/O2SolidHarness.h"

#include "TGeoBBox.h"
#include "TGeoBoolNode.h"
#include "TGeoCompositeShape.h"
#include "TGeoMatrix.h"
#include "TGeoShape.h"
#include "TGeoTube.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#ifdef __linux__
#include <malloc.h>
#include <unistd.h>
#endif

namespace o2
{
namespace cad
{
namespace bench
{

using o2::cad::harness::Point3D;
using o2::cad::harness::Ray;

// ------------------------------------------------------------------------------------------
// 1. Timing: several passes, a robust statistic, and the spread
// ------------------------------------------------------------------------------------------

/// The result of timing one kernel over several complete passes of the same sample set.
///
/// `median` is the reported number and `min`/`max` are the honest error bar. The minimum is
/// singled out in the printouts as well, because on a machine with other tenants it is the
/// closest thing to the kernel's own cost: noise can only ever make a pass slower.
struct TimingStat {
  long long callsPerPass = 0;
  int passes = 0;
  double medianNsPerCall = 0.;
  double minNsPerCall = 0.;
  double maxNsPerCall = 0.;
  /// (max - min) / median, as a fraction. A number that is quoted with every timing rather than
  /// hidden, because it is what says whether two representations that differ by 10 % differ.
  double spread = 0.;
  uint64_t checksum = 0; ///< accumulated from the results so the optimizer cannot elide the calls
  /// Fraction of calls that returned a finite (< TGeoShape::Big()) distance. A distance kernel
  /// that never hits anything is fast for a reason that has nothing to do with its speed, so this
  /// travels with every ray timing. Meaningless (and left at -1) for point queries.
  double hitFraction = -1.;
};

namespace detail
{
/// The checksum mixer, identical in spirit to O2SolidHarness's: the timed loop must not be
/// removable by the optimizer, and a `volatile` sink costs a store per call.
inline uint64_t mix(uint64_t acc, double value)
{
  uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  acc ^= bits + 0x9e3779b97f4a7c15ULL + (acc << 6) + (acc >> 2);
  return acc;
}
} // namespace detail

/// Time `pass()` -- one complete sweep over the sample set, returning a checksum -- over
/// `warmupPasses` untimed and `passes` timed repetitions, and report the median ns/call.
///
/// The warmup is not decoration: the first pass over a freshly loaded shape pays for the page
/// faults of its own data and for the branch predictor's ignorance, and on the mesh
/// representation that alone was measured at more than 2x the steady-state cost. Every number
/// this function returns is therefore a WARM-CACHE number, and the caller is expected to say so.
template <typename Pass>
TimingStat timePasses(long long callsPerPass, int warmupPasses, int passes, Pass&& pass)
{
  TimingStat stat;
  stat.callsPerPass = callsPerPass;
  if (callsPerPass <= 0 || passes <= 0) {
    return stat;
  }
  for (int i = 0; i < warmupPasses; ++i) {
    stat.checksum = detail::mix(stat.checksum, static_cast<double>(pass()));
  }
  std::vector<double> perPass;
  perPass.reserve(passes);
  for (int i = 0; i < passes; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    const uint64_t sum = pass();
    const auto t1 = std::chrono::steady_clock::now();
    stat.checksum = detail::mix(stat.checksum, static_cast<double>(sum));
    perPass.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count() /
                      static_cast<double>(callsPerPass));
  }
  std::sort(perPass.begin(), perPass.end());
  stat.passes = passes;
  stat.minNsPerCall = perPass.front();
  stat.maxNsPerCall = perPass.back();
  stat.medianNsPerCall = perPass[perPass.size() / 2];
  stat.spread = stat.medianNsPerCall > 0.
                  ? (stat.maxNsPerCall - stat.minNsPerCall) / stat.medianNsPerCall
                  : 0.;
  return stat;
}

// ------------------------------------------------------------------------------------------
// 2. Memory: one exact number and one measured number
// ------------------------------------------------------------------------------------------

/// A point-in-time reading of what this process is holding.
///
/// `residentBytes` comes from /proc/self/statm and is what the operating system sees: it includes
/// the allocator's unreturned arenas and every page the process has ever touched, so it is a
/// generous upper bound and it never goes down when a vector is freed. `heapInUseBytes` comes
/// from mallinfo2 and is what glibc believes is currently handed out to the program -- much
/// closer to the structural number and much less noisy, but blind to anything allocated outside
/// malloc. Both are reported; neither is the truth on its own.
///
/// `uordblks` ALONE IS NOT THE HEAP. glibc services any request over M_MMAP_THRESHOLD (128 kB by
/// default) with its own mmap and books it in `hblkhd`, not in `uordblks` -- so a 64 MB
/// allocation moved this counter by exactly zero until `hblkhd` was added. That was caught by
/// this file's own negative control (`control 11` in the benchmark self-test) and it is the
/// reason the control exists: a memory column that cannot see 64 MB is not a memory column.
struct MemorySnapshot {
  long long residentBytes = 0;
  long long heapInUseBytes = 0;
};

inline MemorySnapshot readMemory()
{
  MemorySnapshot out;
#ifdef __linux__
  std::ifstream statm("/proc/self/statm");
  if (statm) {
    long long totalPages = 0;
    long long residentPages = 0;
    statm >> totalPages >> residentPages;
    out.residentBytes = residentPages * static_cast<long long>(::sysconf(_SC_PAGESIZE));
  }
#if defined(__GLIBC__) && (__GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 33))
  const struct mallinfo2 info = ::mallinfo2();
  out.heapInUseBytes = static_cast<long long>(info.uordblks) + static_cast<long long>(info.hblkhd);
#endif
#endif
  return out;
}

inline MemorySnapshot operator-(const MemorySnapshot& a, const MemorySnapshot& b)
{
  return {a.residentBytes - b.residentBytes, a.heapInUseBytes - b.heapInUseBytes};
}

/// The exact structural size of a representation, derived from its own counters.
///
/// This is the number that is a property of the geometry rather than of the allocator, and it is
/// the one to quote when asking "what would N of these cost". `formula` records how it was
/// arrived at, so a reader can check it rather than trust it.
struct StructuralMemory {
  long long primitives = 0;   ///< triangles / analytic patches / boolean leaves
  long long bytes = 0;        ///< the arithmetic below, exact for the arrays it counts
  long long sidecarBytes = 0; ///< the file the representation was loaded from, on disk
  std::string formula;
};

/// Bytes a `.bin`/`.root` sidecar occupies on disk. Exact, and the one memory number that needs
/// no assumption about anybody's allocator.
inline long long fileBytes(const std::string& path)
{
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  return in ? static_cast<long long>(in.tellg()) : 0;
}

// ------------------------------------------------------------------------------------------
// 3. The sample sets -- built once per part, handed unchanged to every representation
// ------------------------------------------------------------------------------------------

/// The four kernels take two kinds of input and the split between inside and outside has to be
/// made by SOMETHING. It is made once, by a designated reference representation, and recorded --
/// `partitionedBy` travels with every table this produces. Letting each representation classify
/// its own points would mean `DistFromInside` is timed on a different set for each of them, and
/// a ratio between those is not a speed comparison.
struct QuerySamples {
  std::string partitionedBy;
  std::vector<Point3D> points;     ///< all query points, mixed inside/outside, in bbox order
  std::vector<char> pointIsInside; ///< the reference's Contains() for each, as a fixed label
  std::vector<Ray> outsideRays;    ///< origin outside per the reference, aimed into the bbox
  std::vector<Ray> insideRays;     ///< origin inside per the reference, isotropic direction
  long long insidePoints = 0;
};

namespace detail
{
/// A tiny explicit LCG. Not for statistical quality -- for the property that matters here, which
/// is that the sample set is a pure function of the seed and the bounding box and is therefore
/// reproducible across representations, across runs and across machines.
struct Lcg {
  uint64_t state = 88172645463325252ULL;
  double next()
  {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<double>((state >> 11) & ((1ULL << 53) - 1)) / static_cast<double>(1ULL << 53);
  }
};
} // namespace detail

/// Build the shared sample sets from `reference`'s own classification.
///
/// `nPoints` points are drawn uniformly over the bounding box inflated by `inflate`, so the set
/// contains both interior and exterior points in whatever ratio the part's own fill factor gives
/// -- which is the realistic mixture a navigator sees, and is a per-part property that is
/// reported rather than forced. Rays are drawn until the requested counts are met or the attempt
/// budget runs out; outside rays are AIMED at a random point of the bounding box, because an
/// isotropically-directed ray from outside misses a thin part almost always and would time the
/// miss path exclusively.
inline QuerySamples buildQuerySamples(const TGeoShape* reference, const std::string& referenceName,
                                      const Point3D& bboxMin, const Point3D& bboxMax, int nPoints,
                                      int nRays, uint64_t seed = 20260802ULL, double inflate = 0.12)
{
  QuerySamples out;
  out.partitionedBy = referenceName;
  detail::Lcg rng{seed};
  Point3D lo{};
  Point3D hi{};
  Point3D centre{};
  for (int k = 0; k < 3; ++k) {
    const double half = 0.5 * (bboxMax[k] - bboxMin[k]);
    centre[k] = 0.5 * (bboxMax[k] + bboxMin[k]);
    lo[k] = centre[k] - half * (1. + inflate);
    hi[k] = centre[k] + half * (1. + inflate);
  }
  auto drawPoint = [&]() {
    Point3D p{};
    for (int k = 0; k < 3; ++k) {
      p[k] = lo[k] + (hi[k] - lo[k]) * rng.next();
    }
    return p;
  };
  out.points.reserve(nPoints);
  out.pointIsInside.reserve(nPoints);
  for (int i = 0; i < nPoints; ++i) {
    const Point3D p = drawPoint();
    const bool in = reference->Contains(p.data());
    out.points.push_back(p);
    out.pointIsInside.push_back(in ? 1 : 0);
    out.insidePoints += in ? 1 : 0;
  }
  const int budget = 400 * std::max(1, nRays);
  int attempts = 0;
  while (static_cast<int>(out.outsideRays.size()) < nRays && attempts < budget) {
    ++attempts;
    const Point3D p = drawPoint();
    if (reference->Contains(p.data())) {
      continue;
    }
    // Aim at a random point of the (uninflated) bounding box: a thin part is missed by an
    // isotropic direction almost always, and a DistFromOutside timing dominated by misses prices
    // the early-out rather than the kernel.
    Point3D target{};
    for (int k = 0; k < 3; ++k) {
      target[k] = bboxMin[k] + (bboxMax[k] - bboxMin[k]) * rng.next();
    }
    Point3D d{target[0] - p[0], target[1] - p[1], target[2] - p[2]};
    const double norm = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    if (!(norm > 0.)) {
      continue;
    }
    for (int k = 0; k < 3; ++k) {
      d[k] /= norm;
    }
    out.outsideRays.push_back({p, d});
  }
  attempts = 0;
  while (static_cast<int>(out.insideRays.size()) < nRays && attempts < budget) {
    ++attempts;
    const Point3D p = drawPoint();
    if (!reference->Contains(p.data())) {
      continue;
    }
    const double z = 2. * rng.next() - 1.;
    const double phi = 2. * 3.14159265358979323846 * rng.next();
    const double r = std::sqrt(std::max(0., 1. - z * z));
    out.insideRays.push_back({p, {r * std::cos(phi), r * std::sin(phi), z}});
  }
  return out;
}

// ------------------------------------------------------------------------------------------
// 4. The four kernel passes
// ------------------------------------------------------------------------------------------
//
// Each is a closure over the shared sample set that performs exactly `callsPerPass` virtual calls
// and returns a checksum. They exist as named functions rather than as lambdas at the call site
// so that the unit tests time the same loop bodies the benchmark does.

inline TimingStat timeContainsPass(const TGeoShape* shape, const QuerySamples& s, int warmup, int passes)
{
  return timePasses(static_cast<long long>(s.points.size()), warmup, passes, [&]() {
    uint64_t acc = 0;
    for (const auto& p : s.points) {
      acc = detail::mix(acc, shape->Contains(p.data()) ? 1. : 0.);
    }
    return acc;
  });
}

/// Safety is asked with the FIXED label from the reference partition, not with each shape's own
/// Contains(). Two reasons, and the second is the one that matters: `Safety(p, in)` takes
/// different branches for in/out on every implementation here, so a shape that disagreed about a
/// point would be timed on a different branch; and asking each shape's own Contains() first would
/// price two kernels and call it one.
inline TimingStat timeSafetyPass(const TGeoShape* shape, const QuerySamples& s, int warmup, int passes)
{
  return timePasses(static_cast<long long>(s.points.size()), warmup, passes, [&]() {
    uint64_t acc = 0;
    for (size_t i = 0; i < s.points.size(); ++i) {
      acc = detail::mix(acc, shape->Safety(s.points[i].data(), s.pointIsInside[i] ? kTRUE : kFALSE));
    }
    return acc;
  });
}

inline TimingStat timeDistOutPass(const TGeoShape* shape, const QuerySamples& s, int warmup, int passes)
{
  long long hits = 0;
  long long calls = 0;
  TimingStat stat = timePasses(static_cast<long long>(s.outsideRays.size()), warmup, passes, [&]() {
    uint64_t acc = 0;
    for (const auto& ray : s.outsideRays) {
      const double d = shape->DistFromOutside(ray.origin.data(), ray.dir.data(), 3, TGeoShape::Big(), nullptr);
      hits += (d < TGeoShape::Big()) ? 1 : 0;
      ++calls;
      acc = detail::mix(acc, d);
    }
    return acc;
  });
  stat.hitFraction = calls > 0 ? static_cast<double>(hits) / static_cast<double>(calls) : -1.;
  return stat;
}

inline TimingStat timeDistInPass(const TGeoShape* shape, const QuerySamples& s, int warmup, int passes)
{
  long long hits = 0;
  long long calls = 0;
  TimingStat stat = timePasses(static_cast<long long>(s.insideRays.size()), warmup, passes, [&]() {
    uint64_t acc = 0;
    for (const auto& ray : s.insideRays) {
      const double d = shape->DistFromInside(ray.origin.data(), ray.dir.data(), 3, TGeoShape::Big(), nullptr);
      hits += (d < TGeoShape::Big()) ? 1 : 0;
      ++calls;
      acc = detail::mix(acc, d);
    }
    return acc;
  });
  stat.hitFraction = calls > 0 ? static_cast<double>(hits) / static_cast<double>(calls) : -1.;
  return stat;
}

// ------------------------------------------------------------------------------------------
// 5. The synthetic boolean ladder
// ------------------------------------------------------------------------------------------
//
// Every genuine boolean in the corpus today is a 2-leaf union of two TGeoTubes, so the corpus
// cannot say how a composite scales with leaf count
// and no amount of running it harder will make it. This builds the missing fixture: unions of
// 2, 4, 8, ... TGeoTubes, in the two tree shapes an emitter can plausibly produce.
//
// The two shapes are the point of the experiment.
//   * CHAIN   -- (((t0 + t1) + t2) + t3) ... : depth K-1, the natural output of a fold over a
//                list of leaves. Every query descends the whole spine.
//   * BALANCED-- a complete binary tree of depth ceil(log2 K). This is what a BVH over primitives
//                would give you for free, minus the bounding-box rejection.
// If the two scale the same way, tree shape is not where the cost is and a BVH-over-primitives
// CSG solid has nothing to win from restructuring alone. If they separate, the gap IS the prize.

enum class LadderShape { Chain,
                         Balanced };

/// One rung of the ladder: `leaves` overlapping tubes on a line, unioned in the requested shape.
///
/// Overlapping rather than disjoint, deliberately: the corpus's booleans are two COAXIAL tubes
/// with shared interior, and a union of disjoint bodies is an easier question (a point is inside
/// at most one leaf, so a short-circuiting evaluator stops early on every interior query). The
/// pitch of 0.8 against a radius of 0.5 gives every leaf a genuine overlap with its neighbour.
///
/// Returns a shape owned by the current gGeoManager, like every other TGeoShape.
inline TGeoShape* buildBooleanLadder(int leaves, LadderShape shape, const std::string& tag)
{
  if (leaves < 1) {
    return nullptr;
  }
  const double rMin = 0.2;
  const double rMax = 0.5;
  const double dz = 1.0;
  const double pitch = 0.8;
  auto leafName = [&](int i) { return tag + "_leaf" + std::to_string(i); };
  std::vector<TGeoShape*> nodes;
  std::vector<TGeoMatrix*> offsets;
  for (int i = 0; i < leaves; ++i) {
    auto* tube = new TGeoTube(leafName(i).c_str(), rMin, rMax, dz);
    nodes.push_back(tube);
    auto* m = new TGeoTranslation((i - 0.5 * (leaves - 1)) * pitch, 0., 0.);
    m->SetName((leafName(i) + "_m").c_str());
    m->RegisterYourself();
    offsets.push_back(m);
  }
  if (leaves == 1) {
    return nodes.front();
  }
  int serial = 0;
  auto join = [&](TGeoShape* a, TGeoMatrix* ma, TGeoShape* b, TGeoMatrix* mb) -> TGeoShape* {
    auto* node = new TGeoUnion(a, b, ma, mb);
    auto* composite = new TGeoCompositeShape((tag + "_u" + std::to_string(serial++)).c_str(), node);
    return composite;
  };
  if (shape == LadderShape::Chain) {
    TGeoShape* acc = nodes[0];
    TGeoMatrix* accMatrix = offsets[0];
    for (int i = 1; i < leaves; ++i) {
      acc = join(acc, accMatrix, nodes[i], offsets[i]);
      accMatrix = nullptr; // the accumulated composite is already in the common frame
    }
    return acc;
  }
  std::vector<TGeoShape*> level = nodes;
  std::vector<TGeoMatrix*> levelMatrix = offsets;
  while (level.size() > 1) {
    std::vector<TGeoShape*> next;
    std::vector<TGeoMatrix*> nextMatrix;
    for (size_t i = 0; i < level.size(); i += 2) {
      if (i + 1 < level.size()) {
        next.push_back(join(level[i], levelMatrix[i], level[i + 1], levelMatrix[i + 1]));
        nextMatrix.push_back(nullptr);
      } else {
        next.push_back(level[i]);
        nextMatrix.push_back(levelMatrix[i]);
      }
    }
    level.swap(next);
    levelMatrix.swap(nextMatrix);
  }
  return level.front();
}

/// Walk a shape's boolean tree and count leaves, internal nodes and depth.
///
/// The structural memory number for a composite, and the control the ladder's own self-test
/// checks: a fixture that claims 32 leaves and holds 16 is not a scaling experiment.
struct BooleanTreeStats {
  long long leaves = 0;
  long long nodes = 0; ///< TGeoCompositeShape / TGeoBoolNode pairs
  int depth = 0;
};

inline BooleanTreeStats booleanTreeStats(const TGeoShape* shape)
{
  BooleanTreeStats out;
  const auto* composite = dynamic_cast<const TGeoCompositeShape*>(shape);
  if (composite == nullptr || composite->GetBoolNode() == nullptr) {
    out.leaves = 1;
    out.depth = 1;
    return out;
  }
  const TGeoBoolNode* node = composite->GetBoolNode();
  const BooleanTreeStats left = booleanTreeStats(node->GetLeftShape());
  const BooleanTreeStats right = booleanTreeStats(node->GetRightShape());
  out.leaves = left.leaves + right.leaves;
  out.nodes = left.nodes + right.nodes + 1;
  out.depth = 1 + std::max(left.depth, right.depth);
  return out;
}

// ------------------------------------------------------------------------------------------
// 6. The negative control for the timing harness itself
// ------------------------------------------------------------------------------------------

/// A TGeoBBox that is deliberately slower than a TGeoBBox, by a controllable amount.
///
/// The timing harness's negative control: every kernel timed here is also timed against this,
/// and the ratio must exceed a stated factor.
///
/// The burn loop is a data dependency on the point, so it cannot be hoisted or constant-folded,
/// and its result is folded into the returned value so it cannot be dropped.
class BallastShape : public TGeoBBox
{
 public:
  BallastShape(const char* name, double dx, double dy, double dz, int burn)
    : TGeoBBox(name, dx, dy, dz), mBurn(burn) {}

  double ballast(const double* point) const
  {
    double acc = 1.;
    for (int i = 0; i < mBurn; ++i) {
      acc = std::sqrt(acc * acc + point[i % 3] * point[i % 3] + 1.);
    }
    return acc;
  }

  bool Contains(const double* point) const override
  {
    return TGeoBBox::Contains(point) && ballast(point) > 0.;
  }
  double Safety(const double* point, bool in = kTRUE) const override
  {
    return TGeoBBox::Safety(point, in) + 0. * ballast(point);
  }
  double DistFromOutside(const double* point, const double* dir, int iact = 1,
                         double step = TGeoShape::Big(), double* safe = nullptr) const override
  {
    return TGeoBBox::DistFromOutside(point, dir, iact, step, safe) + 0. * ballast(point);
  }
  double DistFromInside(const double* point, const double* dir, int iact = 1,
                        double step = TGeoShape::Big(), double* safe = nullptr) const override
  {
    return TGeoBBox::DistFromInside(point, dir, iact, step, safe) + 0. * ballast(point);
  }

 private:
  int mBurn = 0;
};

} // namespace bench
} // namespace cad
} // namespace o2

#endif
