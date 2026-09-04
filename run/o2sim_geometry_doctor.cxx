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

/// \file o2sim_geometry_doctor.cxx
/// \brief Audits a placed geometry against the magnetic field it will be transported in.
///
/// The tool reads a geometry file and a magnetic field, and answers two questions
/// that cannot be answered by looking at either one alone:
///
///   * where does the geometry fail to express the field's structure -- a logical
///     volume placed both inside and outside the field, or a mother whose own
///     material straddles the boundary, so that no per-medium field flag can be
///     right for all of its placements;
///   * where does it already express it wrongly -- a medium with ifield == 0,
///     which asks the transport engine for straight lines, sitting in real field.
///
/// It also answers one question about the geometry alone, and therefore runs that
/// part without a field under --reachability-only: does every placement actually
/// occupy the space it was built in? A daughter outside its mother, or one shadowed
/// by an overlapping sibling, is never reached by the navigator, carries no material
/// and produces no hits, and nothing in the construction code says so.
///
/// It detects and proposes. It never modifies a geometry.
///
/// The field enters in two ways. A support model (an outer bound on where |B|
/// exceeds a threshold, maximised over phi) supplies the geometric argument: a
/// placement whose extent misses every band of the model is field-free, and that
/// is the only way this tool ever concludes "field-free". Sampling enters only in
/// the opposite direction, to disprove -- finding field inside a volume settles
/// the question, finding none does not.
///
/// Reading the geometry needs no detector code: a medium's ifield flag and its
/// sensitivity are GSTMED parameters 1 and 0 of the TGeoMedium, and are recovered
/// from the file itself.

#include "Field/MagneticField.h"

#include <TFile.h>
#include <TGeoBBox.h>
#include <TGeoCone.h>
#include <TGeoEltu.h>
#include <TGeoManager.h>
#include <TGeoMatrix.h>
#include <TGeoMedium.h>
#include <TGeoNavigator.h>
#include <TGeoNode.h>
#include <TGeoPcon.h>
#include <TGeoPgon.h>
#include <TGeoShape.h>
#include <TGeoTube.h>
#include <TGeoVolume.h>
#include <TRandom3.h>
#include <TVectorD.h>

#include <boost/program_options.hpp>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace bpo = boost::program_options;
using json = nlohmann::json;

namespace
{

// ---------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------

/// printf into a std::string, for the tables this tool prints and stores.
template <typename... Args>
std::string form(const char* fmt, Args... args)
{
  char buffer[4096];
  std::snprintf(buffer, sizeof(buffer), fmt, args...);
  return std::string(buffer);
}

/// Writes every line to stdout as it is produced and keeps a copy for the report file.
class Report
{
 public:
  void operator()(const std::string& line)
  {
    std::cout << line << '\n';
    mLines.push_back(line);
  }
  void write(const std::string& path) const
  {
    std::ofstream out(path);
    for (const auto& line : mLines) {
      out << line << '\n';
    }
  }

 private:
  std::vector<std::string> mLines;
};

/// Progress and diagnostics go to stderr so that the report on stdout stays a report.
void progress(const std::string& line) { std::cerr << line << std::endl; }

// ---------------------------------------------------------------------------
// the field
// ---------------------------------------------------------------------------

// A serialized MagneticField keeps its measured map in a transient member, so a
// reader has to call CreateField() again to get a usable field back. That call
// carries a trap: it feeds mMultipicativeFactorSolenoid/Dipole back through
// setters which negate under the LHC polarity convention, so CreateField() is not
// idempotent and the second call inverts the polarity of both the measured map
// and the machine compensators. |B| is untouched, which is exactly why the flip
// survives any magnitude-based check -- a 200k-point round trip on |B| passes
// while every field vector points the wrong way.
//
// A field file therefore has to carry reference field VECTORS taken from the live
// object at write time. The loader re-evaluates them, repairs a pure global flip,
// and refuses the file if what it gets back is anything other than what was
// written. A file without those probes cannot be verified and is refused too;
// --field-current builds the field from scratch instead.

constexpr const char* kFieldObjectKey = "MagneticField";
constexpr const char* kFieldProbeKey = "ReferenceProbes";

/// 0 = reproduces the probes, 1 = exactly negated, -1 = neither.
int compareToProbes(o2::field::MagneticField* field, const TVectorD& probes)
{
  const int n = probes.GetNrows() / 6;
  bool same = true;
  bool flipped = true;
  for (int i = 0; i < n; ++i) {
    double x[3] = {probes[6 * i], probes[6 * i + 1], probes[6 * i + 2]};
    double b[3] = {0., 0., 0.};
    field->Field(x, b);
    for (int k = 0; k < 3; ++k) {
      const double want = probes[6 * i + 3 + k];
      same = same && (b[k] == want);
      flipped = flipped && (b[k] == -want);
    }
  }
  return same ? 0 : (flipped ? 1 : -1);
}

o2::field::MagneticField* loadFieldFromFile(const std::string& path)
{
  TFile* file = TFile::Open(path.c_str());
  if (file == nullptr || file->IsZombie()) {
    progress("error: cannot open field file " + path);
    return nullptr;
  }
  auto* field = dynamic_cast<o2::field::MagneticField*>(file->Get(kFieldObjectKey));
  auto* probes = dynamic_cast<TVectorD*>(file->Get(kFieldProbeKey));
  if (field == nullptr) {
    progress(form("error: no '%s' object in %s", kFieldObjectKey, path.c_str()));
    return nullptr;
  }
  if (probes == nullptr) {
    progress(form(
      "error: no '%s' in %s -- the field cannot be verified against what was written, "
      "and a silently inverted field is exactly what this check exists to catch. "
      "Use --field-current instead.",
      kFieldProbeKey, path.c_str()));
    return nullptr;
  }
  const TVectorD reference(*probes);
  file->Close();
  delete file;

  // Reload the parameterisation from this file, whatever path was stored at write
  // time, so that the file is self-contained and relocatable.
  field->setDataFileName(path.c_str());
  field->CreateField();

  int comparison = compareToProbes(field, reference);
  if (comparison == 1) {
    field->setFactorSolenoid(-field->getFactorSolenoid());
    field->setFactorDipole(-field->getFactorDipole());
    comparison = compareToProbes(field, reference);
    if (comparison == 0) {
      progress("field: polarity flip from the non-idempotent CreateField() detected and repaired");
    }
  }
  if (comparison != 0) {
    progress(form(
      "error: the field reloaded from %s does not reproduce its own reference probes; "
      "refusing to hand back a field that is not the one written",
      path.c_str()));
    return nullptr;
  }
  progress(form("field: %s verified against %d reference probe vectors", path.c_str(), reference.GetNrows() / 6));
  return field;
}

double fieldMag(o2::field::MagneticField* field, double x, double y, double z)
{
  double point[3] = {x, y, z};
  double b[3] = {0., 0., 0.};
  field->Field(point, b);
  return std::sqrt(b[0] * b[0] + b[1] * b[1] + b[2] * b[2]);
}

double fieldMagCyl(o2::field::MagneticField* field, double r, double phi, double z)
{
  return fieldMag(field, r * std::cos(phi), r * std::sin(phi), z);
}

// ---------------------------------------------------------------------------
// the field-support model
// ---------------------------------------------------------------------------

// The model states, per threshold, a list of z-bands each carrying the radial
// intervals in which |B| exceeds that threshold, maximised over phi. It is an
// OUTER bound: a point outside every band has |B| <= threshold. That direction is
// what lets a sampled quantity support a geometric argument about a volume.
//
// Two features of the real field dictate the sampling, and both were found by
// this model getting them wrong first:
//
//  * the LHC machine elements are hard cylinders with a discontinuous edge (the
//    A-side compensator aperture is exactly r < 4.0 cm), so every threshold
//    crossing is bisected rather than left on the grid;
//
//  * the measured map's coverage is a BOX in (x, y), not a cylinder. Between the
//    box's inscribed and corner radius the field survives only in ~2 degree wedges
//    at the four corners -- at r = 194.5, z = -797.9 it is 8.2 kG at phi = 132.8
//    degrees and exactly zero at 127.5 and 135. A model sampling 16 phi values
//    steps straight over those wedges and declares 8 kG of dipole field
//    unsupported. phi sampling is therefore bounded by ARC LENGTH, so the angular
//    resolution follows the feature size at every radius.
//
// Interval edges are stored as best estimates with a separate edge uncertainty
// rather than pre-inflated, because the study this tool comes from turns on a
// volume whose inner radius is exactly the field boundary: it is separated from
// the field by exactly zero, and no amount of inflation may promote that to a
// margin.

constexpr int kMinPhiSamples = 24;
constexpr double kPhiArcStep = 3.0;    // cm, the azimuthal sampling bound
constexpr double kBisectionTol = 0.01; // cm, also the model's edge uncertainty
constexpr double kZStepCoarse = 1.0;   // cm
constexpr double kZStepRefine = 0.1;   // cm, used wherever the radial structure changes
constexpr double kScanRMax = 2100.;    // cm, outer reach of the scan
constexpr double kScanRMaxFine = 900.; // cm, beyond this the radial grid is coarse
constexpr double kTightMargin = 0.05;  // cm, for boundaries that are analytically hard
constexpr long kViolationScanPoints = 400000;

struct Interval {
  double lo, hi;
};

struct Band {
  double zlo, zhi;
  std::vector<Interval> iv;
};

struct Model {
  double thresholdKG = 0.;
  std::vector<Band> bands;
};

struct Support {
  std::vector<Model> models;
  double edgeUncertainty = kBisectionTol;
  double marginStrict = 5.0;
  double marginTight = kTightMargin;
  double zmin = 0., zmax = 0., rmax = 0.;
  std::string parameterisation;

  /// Separation in the (z,r) half-plane between a placement's extent and the
  /// support; 0 means they intersect. The support is a union over phi, so the
  /// nearest support point is reachable at the volume's own phi and this planar
  /// distance is the true 3-D distance.
  double separation(int t, double vzmin, double vzmax, double vrmin, double vrmax) const
  {
    double best = 1e30;
    for (const auto& band : models[t].bands) {
      const double dz = std::max(0., std::max(band.zlo - vzmax, vzmin - band.zhi));
      for (const auto& iv : band.iv) {
        const double lo = iv.lo - edgeUncertainty;
        const double hi = iv.hi + edgeUncertainty;
        const double dr = std::max(0., std::max(lo - vrmax, vrmin - hi));
        best = std::min(best, std::sqrt(dz * dz + dr * dr));
        if (best <= 0.) {
          return 0.;
        }
      }
    }
    return (best > 1e29) ? 1e30 : best;
  }

  bool supportAt(int t, double z, double r) const { return separation(t, z, z, r, r) <= 0.; }

  /// How deeply the extent penetrates the expanded support, 0 if it does not.
  /// This separates "overlaps by less than the model can resolve" -- a volume
  /// sharing a boundary surface with the field and nothing else -- from
  /// "genuinely reaches into the field".
  double penetration(int t, double vzmin, double vzmax, double vrmin, double vrmax) const
  {
    double worst = 0.;
    for (const auto& band : models[t].bands) {
      if (band.zlo > vzmax || band.zhi < vzmin) {
        continue;
      }
      for (const auto& iv : band.iv) {
        const double lo = iv.lo - edgeUncertainty;
        const double hi = iv.hi + edgeUncertainty;
        if (lo > vrmax || hi < vrmin) {
          continue;
        }
        worst = std::max(worst, std::min(hi - vrmin, vrmax - lo));
      }
    }
    return worst;
  }

  /// Is the whole extent inside the support? Then the placement is in the field
  /// everywhere and there is nothing left to disprove. This is what makes the tool
  /// tractable: most of the ~10^5 passive placements sit deep inside the solenoid,
  /// and sampling each of them to rediscover that would dominate the runtime for
  /// no information. It is a coverage walk rather than a single-band lookup,
  /// because bands are cut wherever the radial structure changes and a long volume
  /// crosses many of them.
  bool coveredBySupport(int t, double vzmin, double vzmax, double vrmin, double vrmax) const
  {
    std::vector<std::pair<double, double>> covering;
    for (const auto& band : models[t].bands) {
      if (band.zhi < vzmin || band.zlo > vzmax) {
        continue;
      }
      for (const auto& iv : band.iv) {
        if (iv.lo - edgeUncertainty <= vrmin && iv.hi + edgeUncertainty >= vrmax) {
          covering.push_back({band.zlo, band.zhi});
          break;
        }
      }
    }
    std::sort(covering.begin(), covering.end());
    double frontier = vzmin;
    for (const auto& segment : covering) {
      if (segment.first > frontier + 1e-9) {
        return false;
      }
      frontier = std::max(frontier, segment.second);
      if (frontier >= vzmax) {
        return true;
      }
    }
    return frontier >= vzmax;
  }

  /// Outside its domain the model makes no claim, and the answer is not OUT.
  bool inDomain(double vzmin, double vzmax, double vrmax) const
  {
    return vzmin >= zmin && vzmax <= zmax && vrmax <= rmax;
  }
};

int phiSamplesAt(double r)
{
  if (r <= 0.) {
    return 1;
  }
  return std::max(kMinPhiSamples, (int)std::ceil(2 * M_PI * r / kPhiArcStep));
}

double maxFieldOverPhi(o2::field::MagneticField* field, double r, double z)
{
  if (r == 0.) {
    return fieldMag(field, 0., 0., z);
  }
  const int n = phiSamplesAt(r);
  double worst = 0.;
  for (int i = 0; i < n; ++i) {
    worst = std::max(worst, fieldMagCyl(field, r, 2 * M_PI * i / n, z));
  }
  return worst;
}

std::vector<double> radialGrid()
{
  std::vector<double> grid;
  for (double r = 0.; r < 20.; r += 0.1) {
    grid.push_back(r);
  }
  for (double r = 20.; r < 100.; r += 1.0) {
    grid.push_back(r);
  }
  for (double r = 100.; r < 800.; r += 2.0) {
    grid.push_back(r);
  }
  for (double r = 800.; r <= kScanRMaxFine; r += 10.0) {
    grid.push_back(r);
  }
  // A coarse extension so that an unexpected far feature is not invisible by construction.
  for (double r = kScanRMaxFine + 25.; r <= kScanRMax; r += 25.0) {
    grid.push_back(r);
  }
  return grid;
}

/// Crossing radius between rOut (|B| <= threshold) and rIn (|B| > threshold).
/// The two may be given in either radial order: assuming rIn < rOut turns every
/// falling edge into a grid midpoint, which understates the support radius, and
/// understating support is the one direction this model may not err in.
double bisectCrossing(o2::field::MagneticField* field, double z, double rOut, double rIn, double threshold)
{
  for (int i = 0; i < 60 && std::fabs(rOut - rIn) > kBisectionTol; ++i) {
    const double middle = 0.5 * (rOut + rIn);
    if (maxFieldOverPhi(field, middle, z) > threshold) {
      rIn = middle;
    } else {
      rOut = middle;
    }
  }
  return 0.5 * (rOut + rIn);
}

struct Slice {
  double z = 0., zlo = 0., zhi = 0.;
  std::vector<std::vector<Interval>> iv; // one per threshold
};

Slice sliceAt(o2::field::MagneticField* field, double z, const std::vector<double>& thresholds)
{
  Slice slice;
  slice.z = z;
  slice.iv.resize(thresholds.size());
  const std::vector<double> grid = radialGrid();
  std::vector<double> b(grid.size());
  for (size_t i = 0; i < grid.size(); ++i) {
    b[i] = maxFieldOverPhi(field, grid[i], z);
  }
  for (size_t t = 0; t < thresholds.size(); ++t) {
    bool open = false;
    Interval current{0., 0.};
    for (size_t i = 0; i < grid.size(); ++i) {
      const bool above = b[i] > thresholds[t];
      if (above && !open) {
        current.lo = (i > 0) ? std::max(0., bisectCrossing(field, z, grid[i - 1], grid[i], thresholds[t])) : grid[i];
        open = true;
      } else if (!above && open) {
        current.hi = bisectCrossing(field, z, grid[i], grid[i - 1], thresholds[t]);
        slice.iv[t].push_back(current);
        open = false;
      }
    }
    if (open) {
      current.hi = grid.back();
      slice.iv[t].push_back(current);
    }
  }
  return slice;
}

void mergeIntervals(std::vector<Interval>& into, const std::vector<Interval>& from)
{
  into.insert(into.end(), from.begin(), from.end());
  if (into.empty()) {
    return;
  }
  std::sort(into.begin(), into.end(), [](const Interval& a, const Interval& b) { return a.lo < b.lo; });
  std::vector<Interval> merged{into.front()};
  for (size_t i = 1; i < into.size(); ++i) {
    if (into[i].lo <= merged.back().hi + 1e-9) {
      merged.back().hi = std::max(merged.back().hi, into[i].hi);
    } else {
      merged.push_back(into[i]);
    }
  }
  into.swap(merged);
}

bool sameStructure(const std::vector<Interval>& a, const std::vector<Interval>& b)
{
  if (a.size() != b.size()) {
    return false;
  }
  for (size_t i = 0; i < a.size(); ++i) {
    if (std::fabs(a[i].lo - b[i].lo) > 0.02 || std::fabs(a[i].hi - b[i].hi) > 0.02) {
      return false;
    }
  }
  return true;
}

Support buildSupport(o2::field::MagneticField* field, const std::vector<double>& thresholds, double zmin, double zmax)
{
  progress(form(
    "support: scanning z %.0f..%.0f, dz %.1f cm refined to %.1f, r to %.0f cm, "
    "phi by arc length <= %.1f cm (%d samples at r=200)",
    zmin, zmax, kZStepCoarse, kZStepRefine, kScanRMax, kPhiArcStep, phiSamplesAt(200.)));

  std::vector<Slice> slices;
  Slice previous;
  bool havePrevious = false;
  for (double z = zmin; z <= zmax + 1e-9; z += kZStepCoarse) {
    Slice slice = sliceAt(field, z, thresholds);
    bool changed = false;
    for (size_t t = 0; havePrevious && t < thresholds.size(); ++t) {
      changed = changed || !sameStructure(previous.iv[t], slice.iv[t]);
    }
    if (changed) {
      for (double zz = previous.z + kZStepRefine; zz < z - 1e-9; zz += kZStepRefine) {
        slices.push_back(sliceAt(field, zz, thresholds));
      }
    }
    slices.push_back(slice);
    previous = slice;
    havePrevious = true;
    if (std::fmod(z - zmin, 500.) < kZStepCoarse / 2) {
      progress(form("support:   ... z = %.0f", z));
    }
  }

  for (size_t i = 0; i < slices.size(); ++i) {
    const double zPrev = (i == 0) ? slices[i].z - kZStepCoarse : slices[i - 1].z;
    const double zNext = (i + 1 == slices.size()) ? slices[i].z + kZStepCoarse : slices[i + 1].z;
    slices[i].zlo = 0.5 * (zPrev + slices[i].z);
    slices[i].zhi = 0.5 * (slices[i].z + zNext);
  }

  // A band carries the union of the intervals of ITS OWN slices, extended by half
  // a coarse step at each end so that neighbouring bands overlap and a point
  // between two samples is claimed by both. Unioning a band with its bracketing
  // slices instead is sound only for a band one step wide: applied to a merged
  // band it smears a neighbour's support across the whole length, and the field-
  // free windows this tool exists to find are precisely empty bands between two
  // populated ones. Conservatism has to stay local or it destroys them.
  Support support;
  support.models.resize(thresholds.size());
  for (size_t t = 0; t < thresholds.size(); ++t) {
    support.models[t].thresholdKG = thresholds[t];
    size_t i = 0;
    while (i < slices.size()) {
      size_t j = i;
      while (j + 1 < slices.size() && sameStructure(slices[j].iv[t], slices[j + 1].iv[t])) {
        ++j;
      }
      Band band;
      band.zlo = slices[i].zlo - 0.5 * kZStepCoarse;
      band.zhi = slices[j].zhi + 0.5 * kZStepCoarse;
      for (size_t k = i; k <= j; ++k) {
        mergeIntervals(band.iv, slices[k].iv[t]);
      }
      if (!band.iv.empty()) {
        support.models[t].bands.push_back(band);
      }
      i = j + 1;
    }
  }
  support.zmin = zmin;
  support.zmax = zmax;
  support.rmax = kScanRMax;
  support.parameterisation = field->getParameterName();
  return support;
}

/// The model claims to be an outer bound. Test that against the field itself:
/// where the model reports no support, the real field must not exceed the
/// threshold. This is sampling used in the one direction it is allowed, to
/// disprove, and it is the only check that catches a model which is merely
/// plausible. A violated bound invalidates every OUT verdict downstream, so it is
/// fatal rather than a warning.
bool violationScan(o2::field::MagneticField* field, const Support& support, Report& report)
{
  TRandom3 random(10001);
  std::vector<long> violations(support.models.size(), 0);
  std::vector<double> worst(support.models.size(), 0.);
  std::vector<double> worstR(support.models.size(), 0.);
  std::vector<double> worstZ(support.models.size(), 0.);
  for (long i = 0; i < kViolationScanPoints; ++i) {
    // Half the points anywhere in the domain, half near the axis where the
    // machine elements are narrow enough to hide between uniform samples.
    const bool nearAxis = (i % 2 == 1);
    const double z = nearAxis ? random.Uniform(std::max(support.zmin, -2200.), std::min(support.zmax, 2200.))
                              : random.Uniform(support.zmin, support.zmax);
    const double r = random.Uniform(0., nearAxis ? 20. : support.rmax);
    const double b = fieldMagCyl(field, r, random.Uniform(0., 2 * M_PI), z);
    for (size_t t = 0; t < support.models.size(); ++t) {
      if (b > support.models[t].thresholdKG && !support.supportAt(t, z, r)) {
        ++violations[t];
        if (b > worst[t]) {
          worst[t] = b;
          worstR[t] = r;
          worstZ[t] = z;
        }
      }
    }
  }
  bool ok = true;
  for (size_t t = 0; t < support.models.size(); ++t) {
    report(form("  outer bound at %6.1f G: %ld / %ld sampled points with field outside every band%s",
                support.models[t].thresholdKG * 1000., violations[t], kViolationScanPoints,
                violations[t] == 0 ? "  (bound holds)" : "   <-- THE MODEL IS NOT AN OUTER BOUND"));
    if (violations[t] != 0) {
      report(form("      worst: |B| = %.4f kG at r = %.3f, z = %.3f", worst[t], worstR[t], worstZ[t]));
      ok = false;
    }
  }
  return ok;
}

json supportToJson(const Support& support, const std::string& fieldSource)
{
  const std::time_t now = std::time(nullptr);
  char stamp[64];
  std::strftime(stamp, sizeof(stamp), "%Y-%m-%dT%H:%M:%S", std::gmtime(&now));

  json out;
  out["schema"] = "o2-sim-geometry-doctor/field_support/1";
  out["generated_utc"] = stamp;
  out["field_source"] = fieldSource;
  out["parameterisation"] = support.parameterisation;
  out["units"] = "kGauss, cm";
  out["semantics"] =
    "Outer bound on the support of |B|, maximised over phi. A point outside every band, "
    "after expanding intervals by edge_uncertainty_cm, has |B| <= threshold.";
  out["resolution"] = {{"dz_coarse", kZStepCoarse},
                       {"dz_refine", kZStepRefine},
                       {"dr_near_axis", 0.1},
                       {"phi_arc_step_cm", kPhiArcStep},
                       {"phi_min_samples", kMinPhiSamples},
                       {"bisection_tol_cm", kBisectionTol},
                       {"edge_uncertainty_cm", support.edgeUncertainty}};
  out["domain"] = {{"zmin", support.zmin}, {"zmax", support.zmax}, {"rmax", support.rmax}};
  out["recommended_margins_cm"] = {{"strict", support.marginStrict}, {"tight", support.marginTight}};
  out["models"] = json::array();
  for (const auto& model : support.models) {
    json m;
    m["threshold_kG"] = model.thresholdKG;
    m["threshold_gauss"] = model.thresholdKG * 1000.;
    m["n_bands"] = model.bands.size();
    m["bands"] = json::array();
    for (const auto& band : model.bands) {
      json b;
      b["zlo"] = band.zlo;
      b["zhi"] = band.zhi;
      b["iv"] = json::array();
      for (const auto& iv : band.iv) {
        b["iv"].push_back(json::array({iv.lo, iv.hi}));
      }
      m["bands"].push_back(b);
    }
    out["models"].push_back(m);
  }
  return out;
}

/// An empty or truncated model would declare the whole geometry field-free, which
/// is the worst direction to fail in, so every problem here is fatal.
bool supportFromJson(const json& in, Support& support)
{
  try {
    support.edgeUncertainty = in.at("resolution").at("edge_uncertainty_cm").get<double>();
    support.marginStrict = in.at("recommended_margins_cm").at("strict").get<double>();
    support.marginTight = in.at("recommended_margins_cm").at("tight").get<double>();
    support.zmin = in.at("domain").at("zmin").get<double>();
    support.zmax = in.at("domain").at("zmax").get<double>();
    support.rmax = in.at("domain").at("rmax").get<double>();
    support.parameterisation = in.value("parameterisation", std::string());
    for (const auto& m : in.at("models")) {
      Model model;
      model.thresholdKG = m.at("threshold_kG").get<double>();
      for (const auto& b : m.at("bands")) {
        Band band;
        band.zlo = b.at("zlo").get<double>();
        band.zhi = b.at("zhi").get<double>();
        for (const auto& iv : b.at("iv")) {
          band.iv.push_back({iv.at(0).get<double>(), iv.at(1).get<double>()});
        }
        model.bands.push_back(band);
      }
      support.models.push_back(model);
    }
  } catch (const std::exception& e) {
    progress(std::string("error: cannot read the support model: ") + e.what());
    return false;
  }
  if (support.models.empty() || support.models.front().bands.empty()) {
    progress(
      "error: the support model is empty -- refusing to continue, since an empty model "
      "would declare the whole geometry field-free");
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// reachability
// ---------------------------------------------------------------------------

/// A placement can be perfectly built and still occupy no space. TGeo never
/// descends into a daughter lying outside its mother, and an overlapping sibling
/// can shadow one that does not; either way the volume carries no material, takes
/// no steps and produces no hits, and nothing in the construction code complains.
/// Only the navigator can settle it, so ask it: draw points inside a placement's
/// own shape and check that FindNode() comes back through that placement.
///
/// One representative path per node object, which is the granularity at which the
/// common defect lives -- a daughter outside its mother is a property of the node,
/// not of the path that reaches it. A node whose mother is itself placed many
/// times is therefore sampled once, in the first of those placements.
///
/// Two questions are asked of every point, and they are not the same question.
/// *Reachability* asks whether the navigator's path passes through this placement
/// at all, so a point that lands in one of its own daughters counts. *Self
/// material* asks the stronger question: for a point that is nominally this
/// volume's own material -- inside its shape and inside none of its daughters --
/// FindNode() must return exactly this path, not a prefix of it and not something
/// else. A mother whose own medium is entirely taken by an overlapping foreign
/// volume is still "reached" through its daughters, and only the second question
/// sees that its material is gone.
struct Reach {
  std::string medium, mother, worstPath;
  long sampled = 0;
  double fraction = 1.;
  long ownSampled = 0;     ///< points that are nominally this volume's own material
  double ownFraction = 1.; ///< of those, the share the navigator actually gives it
};

constexpr int kReachRejectionTries = 400;

/// Defined with the placement table below. Deliberately the same predicate the
/// field classification already uses for "own material", so the two parts of this
/// tool cannot disagree about what a volume's own material is.
bool insideAnyDaughter(TGeoVolume* volume, const double* local);

/// One placement to sample: the node, where it sits, and the chain of nodes that
/// reaches it. The chain is what the navigator's answer is compared against --
/// node identity rather than a path string, so no formatting or copy-number
/// ambiguity can enter the comparison.
struct ReachTask {
  TGeoNode* node = nullptr;
  TGeoHMatrix matrix;
  std::string path;
  std::vector<TGeoNode*> chain; ///< top node first, this node last
};

struct ReachResult {
  bool sampled = false;
  long drawn = 0, reached = 0, ownDrawn = 0, ownReached = 0;
};

/// Collects one representative placement per node object. This half of the audit
/// is inherently sequential -- it carries the matrix chain down the tree and prunes
/// on node identity -- but it is also cheap, because it draws no points.
class ReachCollector
{
 public:
  void walk(TGeoNode* node) { walk(node, TGeoHMatrix(), "", {}); }
  std::vector<ReachTask>& tasks() { return mTasks; }
  long nodesVisited() const { return mVisited; }

 private:
  void walk(TGeoNode* node, const TGeoHMatrix& parent, const std::string& path, std::vector<TGeoNode*> chain);
  std::set<TGeoNode*> mSeen;
  std::vector<ReachTask> mTasks;
  long mVisited = 0;
};

void ReachCollector::walk(TGeoNode* node, const TGeoHMatrix& parent, const std::string& path,
                          std::vector<TGeoNode*> chain)
{
  if (!mSeen.insert(node).second) {
    return; // this node object, and therefore its whole subtree, is already covered
  }
  TGeoHMatrix here = parent;
  here.Multiply(node->GetMatrix());
  const std::string myPath = path + "/" + node->GetName();
  chain.push_back(node);
  ++mVisited;

  // an assembly is expanded away at closure, so FindNode never returns one
  if (!node->GetVolume()->IsAssembly()) {
    ReachTask task;
    task.node = node;
    task.matrix = here;
    task.path = myPath;
    task.chain = chain;
    mTasks.push_back(std::move(task));
  }
  for (int i = 0; i < node->GetNdaughters(); ++i) {
    walk(node->GetDaughter(i), here, myPath, chain);
  }
}

/// Rejection sampling against the shape itself. A TGeoCompositeShape inherits
/// TGeoBBox, so its DX/DY/DZ describe a box that still contains the holes and
/// subtractions -- only Contains() knows the difference. GetOrigin() matters too:
/// the box need not be centred on the local origin.
bool samplePoint(TGeoShape* shape, TRandom3& random, double* local)
{
  auto* box = dynamic_cast<TGeoBBox*>(shape);
  if (box == nullptr) {
    return false;
  }
  const double* origin = box->GetOrigin();
  for (int attempt = 0; attempt < kReachRejectionTries; ++attempt) {
    local[0] = origin[0] + box->GetDX() * (2. * random.Rndm() - 1.);
    local[1] = origin[1] + box->GetDY() * (2. * random.Rndm() - 1.);
    local[2] = origin[2] + box->GetDZ() * (2. * random.Rndm() - 1.);
    if (shape->Contains(local)) {
      return true;
    }
  }
  return false;
}

/// One generator per placement, seeded from its index. A single shared generator
/// would make every placement's numbers depend on the order the others were
/// sampled in -- which is the walk order in a serial run and nothing at all in a
/// parallel one. Seeding per placement makes the audit reproducible and identical
/// whatever --jobs is set to.
unsigned int seedFor(size_t index)
{
  unsigned long long x = 20260901ull + 0x9E3779B97F4A7C15ull * (index + 1);
  x ^= x >> 30;
  x *= 0xBF58476D1CE4E5B9ull;
  x ^= x >> 27;
  return (unsigned int)(x >> 33) | 1u;
}

/// Does the TGeo navigator's current path pass through this placement, and did it
/// stop exactly there? Compared node by node rather than as a path prefix: a
/// string prefix accepts `.../X_1` for `.../X_10`, and copy numbers 1 and 10 in
/// one mother are common enough in ALICE for that to matter.
bool tgeoPassesThrough(TGeoNavigator* nav, const std::vector<TGeoNode*>& chain, bool& exact)
{
  const int depth = (int)chain.size() - 1;
  const int level = nav->GetLevel();
  if (level < depth) {
    return false;
  }
  for (int d = 0; d <= depth; ++d) {
    if (nav->GetMother(level - d) != chain[d]) {
      return false;
    }
  }
  exact = (level == depth);
  return true;
}

void sampleTask(const ReachTask& task, size_t index, int samples, TGeoNavigator* nav, ReachResult& out)
{
  TGeoVolume* volume = task.node->GetVolume();
  const bool hasDaughters = volume->GetNdaughters() > 0;
  TRandom3 random(seedFor(index));
  for (int i = 0; i < samples; ++i) {
    double local[3], global[3];
    if (!samplePoint(volume->GetShape(), random, local)) {
      break;
    }
    ++out.drawn;
    // nominally this volume's own material: inside its shape, inside none of its
    // daughters. A leaf owns every point of its shape, so skip the walk there.
    const bool own = !hasDaughters || !insideAnyDaughter(volume, local);
    if (own) {
      ++out.ownDrawn;
    }
    task.matrix.LocalToMaster(local, global);

    // FindNode() resumes from wherever the navigator currently is, so without
    // this the audit asks each question from inside the very placement it is
    // testing and that placement wins every genuinely ambiguous point. Two
    // mutually overlapping volumes then both report themselves fully reached.
    // Starting from the top makes the answer the navigator's own, and the same
    // one a track crossing the region would get.
    nav->CdTop();
    if (nav->FindNode(global[0], global[1], global[2]) == nullptr) {
      continue;
    }
    bool exact = false;
    if (!tgeoPassesThrough(nav, task.chain, exact)) {
      continue;
    }
    ++out.reached;
    if (own && exact) {
      ++out.ownReached; // it stopped here, so the material really is this one's
    }
  }
  out.sampled = out.drawn > 0;
}

/// Prints the audit and returns how many placements the navigator cannot reach at all.
long reportReachability(int samples, int jobs, Report& report)
{
  if (samples <= 0) {
    return 0;
  }
  progress("reachability: asking the navigator to find every placement from inside its own shape");
  ReachCollector collector;
  collector.walk(gGeoManager->GetTopNode());
  auto& tasks = collector.tasks();

  int threads = jobs > 0 ? jobs : (int)std::thread::hardware_concurrency();
  threads = std::max(1, std::min<int>(threads, (int)tasks.size()));
  // Every placement is sampled independently, so the only shared state is the
  // geometry itself. ROOT serves that per thread: SetMaxThreads allocates the
  // per-thread shape data (composite shapes and voxel finders cache into it) and
  // each worker claims its own navigator, without which they would all drive one.
  if (threads > 1) {
    gGeoManager->SetMaxThreads(threads);
  }
  progress(form("reachability: %zu placements, %d point%s each, %d thread%s", tasks.size(), samples,
                samples == 1 ? "" : "s", threads, threads == 1 ? "" : "s"));

  std::vector<ReachResult> results(tasks.size());
  std::atomic<size_t> next{0};
  auto worker = [&]() {
    TGeoNavigator* nav = threads > 1 ? gGeoManager->AddNavigator() : gGeoManager->GetCurrentNavigator();
    // handed out one at a time: a placement's cost spans orders of magnitude, so a
    // static split would leave most threads waiting on the few expensive ones
    for (size_t i = next++; i < tasks.size(); i = next++) {
      sampleTask(tasks[i], i, samples, nav, results[i]);
    }
  };
  if (threads > 1) {
    std::vector<std::thread> pool;
    pool.reserve(threads);
    for (int i = 0; i < threads; ++i) {
      pool.emplace_back(worker);
    }
    for (auto& thread : pool) {
      thread.join();
    }
  } else {
    worker();
  }

  long sampled = 0, unsampleable = 0;
  std::vector<Reach> dead, partial;
  for (size_t i = 0; i < tasks.size(); ++i) {
    const auto& result = results[i];
    if (!result.sampled) {
      ++unsampleable; // a sliver too thin for the rejection budget; says nothing
      continue;
    }
    ++sampled;
    TGeoNode* node = tasks[i].node;
    auto* medium = node->GetVolume()->GetMedium();
    Reach entry;
    entry.medium = medium != nullptr ? medium->GetName() : "(none)";
    entry.mother = node->GetMotherVolume() != nullptr ? node->GetMotherVolume()->GetName() : "-";
    entry.worstPath = tasks[i].path;
    entry.sampled = result.drawn;
    entry.ownSampled = result.ownDrawn;
    entry.fraction = double(result.reached) / result.drawn;
    entry.ownFraction = result.ownDrawn > 0 ? double(result.ownReached) / result.ownDrawn : 1.;
    // Either number can fail on its own. A mother almost entirely filled by its
    // daughters keeps a high reached fraction while the sliver of its own medium
    // is taken by a foreign volume, and that sliver is the material that
    // disappears -- so classify on whichever of the two is worse.
    if (entry.fraction == 0.) {
      dead.push_back(entry);
    } else if (entry.fraction < 0.999 || entry.ownFraction < 0.999) {
      partial.push_back(entry);
    }
  }

  report(form("reachability: %ld node objects visited, %ld sampled, %ld too thin to sample",
              collector.nodesVisited(), sampled, unsampleable));
  report(form("  %ld placements the navigator never reaches, %zu it reaches only in part", (long)dead.size(),
              partial.size()));
  // worst first: with hundreds of small overlaps the walk order is not a ranking
  std::sort(partial.begin(), partial.end(), [](const Reach& a, const Reach& b) {
    return std::min(a.fraction, a.ownFraction) < std::min(b.fraction, b.ownFraction);
  });
  if (!dead.empty()) {
    report("  unreachable -- these carry no material and produce no hits:");
    report(form("    %-12s %-18s %10s  %s", "mother", "medium", "sampled", "path"));
    for (const auto& entry : dead) {
      report(form("    %-12s %-18s %10ld  %s", entry.mother.c_str(), entry.medium.c_str(), entry.sampled,
                  entry.worstPath.c_str()));
    }
  }
  for (size_t i = 0; i < partial.size() && i < 20; ++i) {
    const auto& entry = partial[i];
    if (i == 0) {
      report("  partially shadowed -- an overlapping sibling or an extruding placement.");
      report("  'reached' is how much of the placement the navigator enters at all; 'own kept'");
      report("  how much of the medium this volume was given to carry survives as its own:");
      report(form("    %-12s %-18s %8s %9s  %s", "mother", "medium", "reached", "own kept", "path"));
    }
    report(form("    %-12s %-18s %7.1f%% %8.1f%%  %s", entry.mother.c_str(), entry.medium.c_str(),
                100. * entry.fraction, 100. * entry.ownFraction, entry.worstPath.c_str()));
  }
  if (partial.size() > 20) {
    report(form("    ... and %zu more, all above %.1f%%", partial.size() - 20,
                100. * std::min(partial[19].fraction, partial[19].ownFraction)));
  }

  report("");
  return (long)dead.size();
}

// ---------------------------------------------------------------------------
// the placement table
// ---------------------------------------------------------------------------

constexpr int kMaxDepth = 14;
constexpr size_t kMaxRows = 400000;
constexpr double kSampleDr = 0.05;   // cm, radial step of the disproof scan
constexpr double kSampleArc = 0.5;   // cm, azimuthal step of the disproof scan
constexpr double kSampleDzMax = 2.0; // cm
constexpr long kMaxSamplesPerRow = 4000000;

struct Row {
  std::string path, lv, medium, mother, shape;
  std::string effectiveMother; ///< nearest non-assembly ancestor: whose material really surrounds this
  std::string verdict = "UNCLASSIFIED";
  int ifield = -1;
  int copyNo = 0, nDaughters = 0, depth = 0;
  bool sensitive = false, assembly = false, approximateExtent = false, resolved = true;
  double zmin = 0., zmax = 0., rmin = 0., rmax = 0.;
  double separation = -1., penetration = 0.;
  double maxB = -1., minB = -1.;
  long nSampled = 0;
  bool wholeVolumeSampled = false;
  double wholeMaxB = -1., wholeMinB = -1.;
  TGeoNode* node = nullptr;
  TGeoHMatrix matrix;
};

bool isOutFamily(const std::string& verdict)
{
  return verdict == "OUT" || verdict == "OUT_TIGHT" || verdict == "OUT_BOUNDARY";
}

/// Everything that is not an OUT-family verdict counts as in-field, including
/// UNKNOWN and OUTSIDE_DOMAIN. Defaulting the unclear cases to "has field" is the
/// only safe direction: the cost of being wrong the other way is straight-line
/// transport through a real field.
bool isInFamily(const std::string& verdict)
{
  return verdict == "IN" || verdict == "IN_COVERED" || verdict == "UNKNOWN" || verdict == "OUTSIDE_DOMAIN";
}

/// Radial extent from the SHAPE, not from bounding-box corners: the corners of a
/// box around an on-axis tube all sit at the same radius and report rmin == rmax.
/// Two shapes need care beyond that. TGeoPgon's Rmin/Rmax are apothems, so the
/// circumscribed radius is Rmax / cos(pi/n). And TGeoEltu MUST be tested before
/// TGeoTube: it derives from TGeoTube and reuses fRmin/fRmax to hold its two
/// semi-axes, so the tube branch makes a solid elliptical pipe report an inner
/// radius of 3.175 cm, as though it avoided the beam axis it plainly contains.
/// That is the dangerous direction -- it promotes an in-field volume to field-free.
bool shapeRadii(TGeoShape* shape, double& rmin, double& rmax)
{
  if (auto* pgon = dynamic_cast<TGeoPgon*>(shape)) {
    rmin = 1e30;
    rmax = -1e30;
    for (int i = 0; i < pgon->GetNz(); ++i) {
      rmin = std::min(rmin, pgon->GetRmin(i));
      rmax = std::max(rmax, pgon->GetRmax(i));
    }
    const double edges = pgon->GetNedges() > 2 ? pgon->GetNedges() : 3;
    rmax /= std::cos(M_PI / edges);
    return true;
  }
  if (auto* pcon = dynamic_cast<TGeoPcon*>(shape)) {
    rmin = 1e30;
    rmax = -1e30;
    for (int i = 0; i < pcon->GetNz(); ++i) {
      rmin = std::min(rmin, pcon->GetRmin(i));
      rmax = std::max(rmax, pcon->GetRmax(i));
    }
    return true;
  }
  if (auto* cone = dynamic_cast<TGeoCone*>(shape)) {
    rmin = std::min(cone->GetRmin1(), cone->GetRmin2());
    rmax = std::max(cone->GetRmax1(), cone->GetRmax2());
    return true;
  }
  if (auto* eltu = dynamic_cast<TGeoEltu*>(shape)) {
    rmin = 0.; // solid, so it contains the axis
    rmax = std::max(eltu->GetA(), eltu->GetB());
    return true;
  }
  if (auto* tube = dynamic_cast<TGeoTube*>(shape)) {
    rmin = tube->GetRmin();
    rmax = tube->GetRmax();
    return true;
  }
  return false;
}

bool zPreserving(const TGeoHMatrix& m)
{
  const Double_t* r = m.GetRotationMatrix();
  return std::fabs(std::fabs(r[8]) - 1.) < 1e-9 && std::fabs(r[2]) < 1e-9 && std::fabs(r[5]) < 1e-9 &&
         std::fabs(r[6]) < 1e-9 && std::fabs(r[7]) < 1e-9;
}

void extentOf(TGeoNode* node, const TGeoHMatrix& matrix, Row& row)
{
  TGeoShape* shape = node->GetVolume()->GetShape();
  auto* box = dynamic_cast<TGeoBBox*>(shape);
  if (box == nullptr) {
    // No bounding box means no analytic extent; claim everything, which classifies
    // as in-field and never as OUT.
    row.zmin = row.zmax = 0.;
    row.rmin = 0.;
    row.rmax = 1e9;
    row.approximateExtent = true;
    return;
  }

  row.zmin = 1e30;
  row.zmax = -1e30;
  double boxRmax = 0.;
  const double dx = box->GetDX(), dy = box->GetDY(), dz = box->GetDZ();
  const double* origin = box->GetOrigin();
  for (int i = 0; i < 8; ++i) {
    double local[3] = {origin[0] + ((i & 1) ? dx : -dx), origin[1] + ((i & 2) ? dy : -dy),
                       origin[2] + ((i & 4) ? dz : -dz)};
    double global[3];
    matrix.LocalToMaster(local, global);
    row.zmin = std::min(row.zmin, global[2]);
    row.zmax = std::max(row.zmax, global[2]);
    boxRmax = std::max(boxRmax, std::hypot(global[0], global[1]));
  }

  const double* translation = matrix.GetTranslation();
  const double offAxis = std::hypot(translation[0] + origin[0], translation[1] + origin[1]);
  double localRmin = 0., localRmax = 0.;
  if (zPreserving(matrix) && shapeRadii(shape, localRmin, localRmax)) {
    row.rmin = std::max(0., localRmin - offAxis);
    row.rmax = localRmax + offAxis;
    row.approximateExtent = false;
  } else {
    row.rmax = boxRmax;
    row.rmin = (offAxis <= std::hypot(dx, dy)) ? 0. : std::max(0., offAxis - std::hypot(dx, dy));
    row.approximateExtent = true;
  }
}

// ---------------------------------------------------------------------------
// the doctor
// ---------------------------------------------------------------------------

struct ContainerProposal {
  std::string mother, motherPath;
  double zlo, zhi, rmax;
  int nDaughters;
  bool clearedByStrictMargin;
  bool sensitive;
};

struct SharedVolume {
  std::string lv;
  std::vector<const Row*> out, in;
};

class Doctor
{
 public:
  Doctor(o2::field::MagneticField* field, const Support& support) : mField(field), mSupport(support)
  {
    mThreshold = support.models.front().thresholdKG;
  }

  void walk(TGeoNode* node) { walk(node, nullptr, TGeoHMatrix(), 0, "", ""); }
  void classifyAll();
  void findFindings();

  const std::vector<Row>& rows() const { return mRows; }
  size_t nPruned() const { return mPruned; }
  const std::vector<const Row*>& reverseAudit() const { return mReverse; }
  const std::vector<SharedVolume>& sharedVolumes() const { return mShared; }
  const std::vector<const Row*>& straddlingMothers() const { return mStraddling; }
  const std::vector<ContainerProposal>& containers() const { return mContainers; }
  bool hasSensitive(TGeoVolume* volume);

 private:
  void walk(TGeoNode* node, TGeoVolume* mother, const TGeoHMatrix& parent, int depth, const std::string& path,
            const std::string& effectiveMother);
  void classify(Row& row);
  void disproofScan(Row& row);
  void wholeVolumeScan(Row& row);
  bool ownMaterialAt(const Row& row, const double* global) const;

  o2::field::MagneticField* mField;
  const Support& mSupport;
  double mThreshold;
  std::vector<Row> mRows;
  std::map<TGeoVolume*, int> mSensitiveCache;
  size_t mPruned = 0;

  std::vector<const Row*> mReverse;
  std::vector<SharedVolume> mShared;
  std::vector<const Row*> mStraddling;
  std::vector<ContainerProposal> mContainers;
};

bool Doctor::hasSensitive(TGeoVolume* volume)
{
  auto cached = mSensitiveCache.find(volume);
  if (cached != mSensitiveCache.end() && cached->second >= 0) {
    return cached->second == 1;
  }
  mSensitiveCache[volume] = 0; // guard against re-entry
  auto* medium = volume->GetMedium();
  bool found = medium != nullptr && medium->GetParam(0) != 0.;
  for (int i = 0; i < volume->GetNdaughters() && !found; ++i) {
    found = hasSensitive(volume->GetNode(i)->GetVolume());
  }
  mSensitiveCache[volume] = found ? 1 : 0;
  return found;
}

/// A volume is structural if it is an assembly or made of its mother's medium
/// (barrel inside cave, caveRB24 inside cave). The walk descends through those and
/// through fully passive subtrees, and prunes detector modules: this tool is about
/// the passive geometry, and a proposal touching a sensitive subtree is refused by
/// default anyway, because sensitive volumes are resolved by name after
/// construction and re-parenting one silently produces zero hits.
bool isStructural(TGeoNode* node, TGeoVolume* mother)
{
  if (mother == nullptr) {
    return true;
  }
  TGeoVolume* volume = node->GetVolume();
  if (volume->IsAssembly()) {
    return true;
  }
  auto* mine = volume->GetMedium();
  auto* theirs = mother->GetMedium();
  return mine != nullptr && theirs != nullptr && std::strcmp(mine->GetName(), theirs->GetName()) == 0;
}

void Doctor::walk(TGeoNode* node, TGeoVolume* mother, const TGeoHMatrix& parent, int depth, const std::string& path,
                  const std::string& effectiveMother)
{
  if (mRows.size() >= kMaxRows || depth > kMaxDepth) {
    return;
  }
  TGeoHMatrix here = parent;
  here.Multiply(node->GetMatrix());
  TGeoVolume* volume = node->GetVolume();
  const std::string myPath = path + "/" + volume->GetName() + "_" + std::to_string(node->GetNumber());

  Row row;
  row.path = myPath;
  row.lv = volume->GetName();
  row.mother = mother != nullptr ? mother->GetName() : "";
  row.effectiveMother = effectiveMother;
  row.shape = volume->GetShape()->ClassName();
  row.copyNo = node->GetNumber();
  row.nDaughters = volume->GetNdaughters();
  row.depth = depth;
  row.assembly = volume->IsAssembly();
  auto* medium = volume->GetMedium();
  row.medium = medium != nullptr ? medium->GetName() : "(none)";
  row.ifield = medium != nullptr ? (int)medium->GetParam(1) : -1;
  row.sensitive = medium != nullptr && medium->GetParam(0) != 0.;
  row.node = node;
  row.matrix = here;
  extentOf(node, here, row);
  mRows.push_back(row);

  if (hasSensitive(volume) && !isStructural(node, mother)) {
    ++mPruned;
    return;
  }
  for (int i = 0; i < node->GetNdaughters(); ++i) {
    walk(node->GetDaughter(i), volume, here, depth + 1, myPath, row.assembly ? effectiveMother : myPath);
  }
}

// A mother's shape includes the space its daughters occupy, but its MATERIAL is
// only what they leave over. That distinction is the whole point of these scans:
// caveRB24's shape reaches the beam axis, and what its daughters leave over is a
// sliver of cave air at r ~ 3-4 cm, between an oval beam pipe and the circular
// field cylinder, carrying 13.2 kG.
bool insideAnyDaughter(TGeoVolume* volume, const double* local)
{
  for (int i = 0; i < volume->GetNdaughters(); ++i) {
    TGeoNode* daughter = volume->GetNode(i);
    double inDaughter[3];
    daughter->GetMatrix()->MasterToLocal(local, inDaughter);
    if (!daughter->GetVolume()->GetShape()->Contains(inDaughter)) {
      continue;
    }
    if (daughter->GetVolume()->IsAssembly()) {
      if (insideAnyDaughter(daughter->GetVolume(), inDaughter)) {
        return true;
      }
    } else {
      return true;
    }
  }
  return false;
}

bool Doctor::ownMaterialAt(const Row& row, const double* global) const
{
  double local[3];
  row.matrix.MasterToLocal(global, local);
  if (!row.node->GetVolume()->GetShape()->Contains(local)) {
    return false;
  }
  return !insideAnyDaughter(row.node->GetVolume(), local);
}

double thinnestDaughter(TGeoVolume* volume)
{
  double thinnest = 1e30;
  for (int i = 0; i < volume->GetNdaughters(); ++i) {
    auto* box = dynamic_cast<TGeoBBox*>(volume->GetNode(i)->GetVolume()->GetShape());
    if (box != nullptr) {
      thinnest = std::min(thinnest, 2 * std::min(box->GetDX(), std::min(box->GetDY(), box->GetDZ())));
    }
  }
  return thinnest;
}

/// Sampling restricted to where an answer can be: the intersection of the
/// placement's extent with the support bands. Everywhere else the field is below
/// threshold by construction, so sampling it would cost time and tell nothing.
void Doctor::disproofScan(Row& row)
{
  const double thinnest = thinnestDaughter(row.node->GetVolume());
  row.resolved = (row.nDaughters == 0) || (kSampleDr <= std::max(0.1, thinnest));
  long budget = kMaxSamplesPerRow;
  for (const auto& band : mSupport.models.front().bands) {
    const double z0 = std::max(row.zmin, band.zlo);
    const double z1 = std::min(row.zmax, band.zhi);
    if (z1 < z0) {
      continue;
    }
    for (const auto& iv : band.iv) {
      const double r0 = std::max(row.rmin, iv.lo - mSupport.edgeUncertainty);
      const double r1 = std::min(row.rmax, iv.hi + mSupport.edgeUncertainty);
      if (r1 < r0) {
        continue;
      }
      const double dz = std::min(kSampleDzMax, std::max(0.25, (z1 - z0) / 200.));
      for (double z = z0; z <= z1 + 1e-9 && budget > 0; z += dz) {
        for (double r = r0; r <= r1 + 1e-9 && budget > 0; r += kSampleDr) {
          const int nphi = (r <= 0.) ? 1 : std::max(8, (int)std::ceil(2 * M_PI * r / kSampleArc));
          for (int i = 0; i < nphi && budget > 0; ++i) {
            const double phi = 2 * M_PI * i / nphi;
            double global[3] = {r * std::cos(phi), r * std::sin(phi), z};
            --budget;
            if (!ownMaterialAt(row, global)) {
              continue;
            }
            ++row.nSampled;
            const double b = fieldMag(mField, global[0], global[1], global[2]);
            row.maxB = std::max(row.maxB, b);
            row.minB = (row.minB < 0.) ? b : std::min(row.minB, b);
          }
        }
      }
    }
  }
  if (budget <= 0) {
    row.resolved = false;
  }
}

/// A coarse scan over the whole placement, to see whether its own material spans
/// both sides of the predicate. Cheap, and it is what makes "the mother straddles
/// the boundary" a measurement rather than an assertion.
void Doctor::wholeVolumeScan(Row& row)
{
  row.wholeVolumeSampled = true;
  const double dz = std::max(1.0, (row.zmax - row.zmin) / 300.);
  const double dr = std::max(0.5, (row.rmax - row.rmin) / 200.);
  long budget = 2000000;
  for (double z = row.zmin; z <= row.zmax && budget > 0; z += dz) {
    for (double r = row.rmin; r <= row.rmax && budget > 0; r += dr) {
      const int nphi = (r <= 0.) ? 1 : std::max(8, (int)std::ceil(2 * M_PI * r / std::max(2.0, dr)));
      for (int i = 0; i < nphi && budget > 0; ++i) {
        const double phi = 2 * M_PI * i / nphi;
        double global[3] = {r * std::cos(phi), r * std::sin(phi), z};
        --budget;
        if (!ownMaterialAt(row, global)) {
          continue;
        }
        const double b = fieldMag(mField, global[0], global[1], global[2]);
        row.wholeMaxB = std::max(row.wholeMaxB, b);
        row.wholeMinB = (row.wholeMinB < 0.) ? b : std::min(row.wholeMinB, b);
      }
    }
  }
}

void Doctor::classify(Row& row)
{
  if (row.assembly) {
    row.verdict = "ASSEMBLY"; // virtual: no material, so nothing to flag
    return;
  }
  if (!mSupport.inDomain(row.zmin, row.zmax, row.rmax)) {
    row.verdict = "OUTSIDE_DOMAIN";
    return;
  }
  row.separation = mSupport.separation(0, row.zmin, row.zmax, row.rmin, row.rmax);
  if (row.separation >= mSupport.marginStrict) {
    row.verdict = "OUT";
    return;
  }
  if (row.separation >= mSupport.marginTight) {
    row.verdict = "OUT_TIGHT";
    return;
  }
  row.penetration = mSupport.penetration(0, row.zmin, row.zmax, row.rmin, row.rmax);
  if (mSupport.coveredBySupport(0, row.zmin, row.zmax, row.rmin, row.rmax)) {
    row.verdict = "IN_COVERED"; // in the field by containment, nothing to disprove
    return;
  }
  disproofScan(row);
  if (row.maxB > mThreshold) {
    row.verdict = "IN";
  } else if (row.separation > 0. || row.penetration <= 2 * mSupport.edgeUncertainty) {
    // Touches the support only to within what the model can resolve. Field-freeness
    // then rests on an analytic claim about a hard boundary, so it is emitted for a
    // human to confirm and never treated as established.
    row.verdict = "OUT_BOUNDARY";
  } else {
    row.verdict = "UNKNOWN";
  }
}

void Doctor::classifyAll()
{
  size_t done = 0;
  for (auto& row : mRows) {
    classify(row);
    if (++done % 50000 == 0) {
      progress(form("classify:   %zu / %zu placements", done, mRows.size()));
    }
  }
  for (auto& row : mRows) {
    // Only a straddler can be heterogeneous: a placement wholly inside the support
    // is uniformly in the field, one that is OUT is uniformly out of it.
    if (row.nDaughters > 0 && !row.assembly &&
        (row.verdict == "IN" || row.verdict == "UNKNOWN" || row.verdict == "OUT_BOUNDARY")) {
      wholeVolumeScan(row);
    }
  }
}

void Doctor::findFindings()
{
  // --- the reverse audit: ifield == 0 media that reach into the field ---------
  // An ifield == 0 medium tells the transport engine to move in a straight line.
  // Where that is wrong it is a physics bug, so the finding must carry the field
  // actually present in the volume's own material, measured. Rows short-circuited
  // as IN_COVERED were never sampled, so they are sampled now.
  for (auto& row : mRows) {
    if (row.assembly || row.ifield != 0 || row.medium == "dummy" || row.medium == "(none)") {
      continue;
    }
    if (isInFamily(row.verdict)) {
      if (!row.wholeVolumeSampled) {
        wholeVolumeScan(row);
      }
      mReverse.push_back(&row);
    }
  }

  // --- shared logical volumes placed on both sides (signature C) --------------
  std::map<std::string, std::vector<const Row*>> byVolume;
  for (const auto& row : mRows) {
    if (!row.assembly) {
      byVolume[row.lv].push_back(&row);
    }
  }
  for (const auto& entry : byVolume) {
    SharedVolume shared;
    shared.lv = entry.first;
    for (const auto* row : entry.second) {
      if (isOutFamily(row->verdict)) {
        shared.out.push_back(row);
      } else if (isInFamily(row->verdict)) {
        shared.in.push_back(row);
      }
    }
    if (!shared.out.empty() && !shared.in.empty()) {
      mShared.push_back(shared);
    }
  }
  std::sort(mShared.begin(), mShared.end(),
            [](const SharedVolume& a, const SharedVolume& b) { return a.out.size() > b.out.size(); });

  // --- mothers whose own material straddles the predicate (signature D) -------
  for (const auto& row : mRows) {
    if (row.nDaughters > 0 && !row.assembly && row.wholeVolumeSampled && row.wholeMaxB > mThreshold &&
        row.wholeMinB <= mThreshold) {
      mStraddling.push_back(&row);
    }
  }

  // --- missing containers -----------------------------------------------------
  // Where a container belongs is not where the daughters have a spatial gap: a
  // beam pipe is a continuous chain of volumes with no gap wider than a flange, so
  // gap-based clustering returns one cluster spanning everything and proposes
  // nothing. What separates the daughters is the PREDICATE. A maximal run of
  // consecutive field-free daughters is exactly a group that cannot be given its
  // own medium today, because the only thing enclosing it is the mother's air,
  // which is not field-free along its whole length.
  for (const auto* mother : mStraddling) {
    std::vector<const Row*> kids;
    for (const auto& row : mRows) {
      if (row.effectiveMother == mother->path && !row.assembly) {
        kids.push_back(&row);
      }
    }
    std::sort(kids.begin(), kids.end(), [](const Row* a, const Row* b) { return a->zmin < b->zmin; });

    size_t i = 0;
    while (i < kids.size()) {
      if (!isOutFamily(kids[i]->verdict)) {
        ++i;
        continue;
      }
      size_t j = i;
      double zlo = kids[i]->zmin, zhi = kids[i]->zmax, rmax = kids[i]->rmax;
      while (j + 1 < kids.size() && isOutFamily(kids[j + 1]->verdict)) {
        ++j;
        zlo = std::min(zlo, kids[j]->zmin);
        zhi = std::max(zhi, kids[j]->zmax);
        rmax = std::max(rmax, kids[j]->rmax);
      }
      const int n = (int)(j - i + 1);

      // A container must not swallow anything outside its run. Rejecting any run
      // with an intruder is too blunt: a 0.05 cm overhang from a neighbouring pipe
      // section would kill a 51-daughter proposal. The right answer to a 0.05 cm
      // overhang is to move the container's edge, so the edges are clamped past any
      // overhanging neighbour and only a residual overlap makes the run unusable.
      double clampedLo = zlo, clampedHi = zhi;
      std::vector<const Row*> intruders;
      for (size_t k = 0; k < kids.size(); ++k) {
        if (k >= i && k <= j) {
          continue;
        }
        const Row* other = kids[k];
        if (other->zmax <= zlo + 0.01 || other->zmin >= zhi - 0.01 || other->rmin >= rmax - 0.01) {
          continue;
        }
        intruders.push_back(other);
      }
      for (const Row* other : intruders) {
        if (other->zmin <= clampedLo + 1e-9 && other->zmax > clampedLo) {
          clampedLo = other->zmax;
        }
        if (other->zmax >= clampedHi - 1e-9 && other->zmin < clampedHi) {
          clampedHi = other->zmin;
        }
      }
      bool residual = false;
      for (const Row* other : intruders) {
        residual = residual || (other->zmax > clampedLo + 0.01 && other->zmin < clampedHi - 0.01 &&
                                other->rmin < rmax - 0.01);
      }

      const double separation =
        (clampedHi > clampedLo) ? mSupport.separation(0, clampedLo, clampedHi, 0., rmax) : -1.;
      if (n >= 2 && separation >= mSupport.marginTight && !residual) {
        ContainerProposal proposal;
        proposal.mother = mother->lv;
        proposal.motherPath = mother->path;
        proposal.zlo = clampedLo;
        proposal.zhi = clampedHi;
        proposal.rmax = rmax;
        proposal.nDaughters = n;
        proposal.clearedByStrictMargin = separation >= mSupport.marginStrict;
        proposal.sensitive = false;
        for (size_t k = i; k <= j; ++k) {
          proposal.sensitive = proposal.sensitive || hasSensitive(kids[k]->node->GetVolume());
        }
        mContainers.push_back(proposal);
      }
      i = j + 1;
    }
  }
}

// ---------------------------------------------------------------------------
// outputs
// ---------------------------------------------------------------------------

void writePlacementCsv(const std::vector<Row>& rows, const std::string& path)
{
  std::FILE* out = std::fopen(path.c_str(), "w");
  if (out == nullptr) {
    progress("error: cannot write " + path);
    return;
  }
  std::fprintf(out,
               "path,lv,medium,ifield,shape,mother,copy,ndaughters,sensitive,assembly,approx,"
               "zmin,zmax,rmin,rmax,verdict,separation_cm,penetration_cm,maxB_kG,nsampled\n");
  for (const auto& row : rows) {
    std::fprintf(out, "%s,%s,%s,%d,%s,%s,%d,%d,%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%s,%.4f,%.4f,%.6f,%ld\n", row.path.c_str(),
                 row.lv.c_str(), row.medium.c_str(), row.ifield, row.shape.c_str(), row.mother.c_str(), row.copyNo,
                 row.nDaughters, (int)row.sensitive, (int)row.assembly, (int)row.approximateExtent, row.zmin, row.zmax,
                 row.rmin, row.rmax, row.verdict.c_str(), row.separation, row.penetration, row.maxB, row.nSampled);
  }
  std::fclose(out);
}

json placementJson(const Row& row)
{
  return json{{"path", row.path},
              {"copy", row.copyNo},
              {"z", json::array({row.zmin, row.zmax})},
              {"r", json::array({row.rmin, row.rmax})},
              {"verdict", row.verdict},
              {"separation_cm", row.separation},
              {"max_B_kG", std::max(row.maxB, row.wholeMaxB)}};
}

json proposalsToJson(Doctor& doctor, const Support& support, const std::string& geometryFile,
                     const std::string& fieldSource)
{
  const std::time_t now = std::time(nullptr);
  char stamp[64];
  std::strftime(stamp, sizeof(stamp), "%Y-%m-%dT%H:%M:%S", std::gmtime(&now));

  json out;
  out["schema"] = "o2-sim-geometry-doctor/proposals/1";
  out["generated_utc"] = stamp;
  out["geometry"] = geometryFile;
  out["field_source"] = fieldSource;
  out["threshold_kG"] = support.models.front().thresholdKG;
  out["margins_cm"] = {{"strict", support.marginStrict},
                       {"tight", support.marginTight},
                       {"edge_uncertainty", support.edgeUncertainty}};
  out["proposals"] = json::array();

  for (const auto& shared : doctor.sharedVolumes()) {
    bool refused = false;
    for (const auto* row : shared.out) {
      refused = refused || row->sensitive || doctor.hasSensitive(row->node->GetVolume());
    }
    json entry;
    entry["signature"] = "shared-volume";
    entry["action"] = "split the logical volume, so that its field-free placements can carry a field-free medium";
    entry["logical_volume"] = shared.lv;
    entry["n_out"] = shared.out.size();
    entry["n_in"] = shared.in.size();
    entry["status"] = refused ? "refused by default (sensitive path)" : "proposed";
    entry["out_placements"] = json::array();
    for (const auto* row : shared.out) {
      entry["out_placements"].push_back(placementJson(*row));
    }
    entry["in_placements"] = json::array();
    for (const auto* row : shared.in) {
      entry["in_placements"].push_back(placementJson(*row));
    }
    out["proposals"].push_back(entry);
  }

  for (const auto* row : doctor.reverseAudit()) {
    const double maxB = std::max(row->maxB, row->wholeMaxB);
    json entry;
    entry["signature"] = "reverse-audit";
    entry["action"] = maxB > support.models.front().thresholdKG
                        ? "the field-free medium assignment is wrong: straight-line transport inside real field"
                        : "the field-free medium reaches field support but no field was found in its own material, review";
    entry["path"] = row->path;
    entry["logical_volume"] = row->lv;
    entry["medium"] = row->medium;
    entry["copy"] = row->copyNo;
    entry["z"] = json::array({row->zmin, row->zmax});
    entry["r"] = json::array({row->rmin, row->rmax});
    entry["min_B_kG"] = row->wholeMinB;
    entry["max_B_kG"] = maxB;
    entry["verdict"] = row->verdict;
    out["proposals"].push_back(entry);
  }

  for (const auto& container : doctor.containers()) {
    json entry;
    entry["signature"] = "missing-container";
    entry["action"] = "insert a container with a field-free medium and re-parent the cluster into it";
    entry["mother"] = container.mother;
    entry["mother_path"] = container.motherPath;
    entry["z"] = json::array({container.zlo, container.zhi});
    entry["rmax"] = container.rmax;
    entry["n_daughters"] = container.nDaughters;
    entry["status"] = container.sensitive ? "refused by default (sensitive path)" : "proposed";
    entry["clearance"] = container.clearedByStrictMargin ? "strict margin" : "tight margin";
    out["proposals"].push_back(entry);
  }

  for (const auto* row : doctor.straddlingMothers()) {
    json entry;
    entry["signature"] = "heterogeneous-mother";
    entry["action"] =
      "the mother's own material spans both sides of the predicate, so no per-medium flag can "
      "express it; it needs a container";
    entry["path"] = row->path;
    entry["logical_volume"] = row->lv;
    entry["medium"] = row->medium;
    entry["min_B_kG"] = row->wholeMinB;
    entry["max_B_kG"] = row->wholeMaxB;
    entry["n_daughters"] = row->nDaughters;
    out["proposals"].push_back(entry);
  }
  return out;
}

// ---------------------------------------------------------------------------
// the anchor self-check
// ---------------------------------------------------------------------------

// A unit test for this tool would need a placed ALICE geometry and a field map,
// neither of which belongs in the repository, so the regression gate is instead a
// file of expectations that any ALICE Run 3 geometry must satisfy, checked against
// a real run with --verify-anchors. See run/geometry-doctor-anchors.json.
//
// Supported expectations: OUT (every placement of the volume is in the OUT
// family), IN, NOT_OUT, ASSEMBLY, and REVERSE_AUDIT_FLAGGED. An anchor may also
// require a placement count and a lower bound on the field found in the volume's
// own material.

bool verifyAnchors(const std::string& path, Doctor& doctor, Report& report)
{
  std::ifstream in(path);
  if (!in) {
    report("  cannot open the anchor file " + path);
    return false;
  }
  json anchors;
  try {
    in >> anchors;
  } catch (const std::exception& e) {
    report(std::string("  cannot parse the anchor file: ") + e.what());
    return false;
  }

  std::set<std::string> flaggedByReverseAudit;
  for (const auto* row : doctor.reverseAudit()) {
    flaggedByReverseAudit.insert(row->lv);
  }

  bool allPassed = true;
  report(form("  %-22s %-22s %-10s %s", "volume", "expected", "verdict", "evidence"));
  for (const auto& anchor : anchors.at("anchors")) {
    const auto volume = anchor.at("volume").get<std::string>();
    const auto expected = anchor.at("expect").get<std::string>();

    int placements = 0;
    int failures = 0;
    double worstSeparation = 1e30;
    double bestField = -1.;
    std::string reported;
    for (const auto& row : doctor.rows()) {
      if (row.lv != volume) {
        continue;
      }
      ++placements;
      bool ok = true;
      if (expected == "OUT") {
        ok = isOutFamily(row.verdict) || row.verdict == "ASSEMBLY";
      } else if (expected == "IN") {
        ok = row.verdict == "IN";
      } else if (expected == "NOT_OUT") {
        ok = !isOutFamily(row.verdict);
      } else if (expected == "ASSEMBLY") {
        ok = row.verdict == "ASSEMBLY";
      } else if (expected == "REVERSE_AUDIT_FLAGGED") {
        ok = true; // decided below, on the volume rather than the placement
      } else {
        report("  unknown expectation '" + expected + "' for " + volume);
        return false;
      }
      failures += ok ? 0 : 1;
      bestField = std::max(bestField, std::max(row.maxB, row.wholeMaxB));
      if (row.separation < worstSeparation) {
        worstSeparation = row.separation;
        reported = row.verdict;
      }
    }

    if (expected == "REVERSE_AUDIT_FLAGGED") {
      failures = flaggedByReverseAudit.count(volume) > 0 ? 0 : 1;
      reported = failures == 0 ? "flagged" : "not flagged";
    }
    if (placements == 0) {
      failures = 1;
      reported = "not placed";
    }
    if (anchor.contains("placements") && placements != anchor.at("placements").get<int>()) {
      failures += 1;
    }
    if (anchor.contains("min_max_B_kG") && bestField < anchor.at("min_max_B_kG").get<double>()) {
      failures += 1;
    }

    allPassed = allPassed && failures == 0;
    report(form("  %-22s %-22s %-10s %d placement(s), separation %.3f cm, max|B| %.4f kG   %s", volume.c_str(),
                expected.c_str(), reported.c_str(), placements, worstSeparation > 1e29 ? -1. : worstSeparation,
                bestField, failures == 0 ? "PASS" : "FAIL"));
  }
  return allPassed;
}

// ---------------------------------------------------------------------------

struct Options {
  std::string geometryFile;
  std::string fieldFile;
  int fieldCurrent = 0;
  std::string supportFile;
  std::string anchorFile;
  std::vector<double> thresholdsGauss;
  double margin = 5.0;
  int reachSamples = 32;
  int reachJobs = 0;
  bool reachabilityOnly = false;
  std::string outputPrefix = "geometry-doctor";
};

} // namespace

int main(int argc, char** argv)
{
  Options options;
  bpo::options_description description(
    "Audits a placed geometry against the magnetic field it will be transported "
    "in, and reports where the two do not fit together.\n\nOptions");
  description.add_options()                                                                             //
    ("help,h", "print this help message")                                                               //
    ("geometry-file", bpo::value<std::string>(&options.geometryFile)->required(),                       //
     "the geometry to audit, e.g. o2sim_geometry.root")                                                 //
    ("field-file", bpo::value<std::string>(&options.fieldFile),                                         //
     "a serialized MagneticField carrying reference probe vectors")                                     //
    ("field-current", bpo::value<int>(&options.fieldCurrent),                                           //
     "build the nominal field for this L3 current instead, e.g. -5")                                    //
    ("support-file", bpo::value<std::string>(&options.supportFile),                                     //
     "field-support model cache: read it if it exists, otherwise write it")                             //
    ("threshold", bpo::value<std::vector<double>>(&options.thresholdsGauss)->composing(),               //
     "field threshold in Gauss, repeatable; the lowest one decides the verdicts (default 1 and 10)")    //
    ("margin", bpo::value<double>(&options.margin)->default_value(5.0),                                 //
     "clearance in cm a placement must keep from the field support to be called field-free")            //
    ("output-prefix", bpo::value<std::string>(&options.outputPrefix)->default_value("geometry-doctor"), //
     "prefix for the report, the proposals and the placement table")                                    //
    ("verify-anchors", bpo::value<std::string>(&options.anchorFile),                                    //
     "check the classification against known-good volumes listed in this JSON file")                    //
    ("reachability-samples", bpo::value<int>(&options.reachSamples)->default_value(1000),               //
     "points drawn inside each placement for the reachability audit; 0 disables it. Below a few "       //
     "hundred the audit reports genuine placements as partially shadowed")                              //
    ("reachability-jobs", bpo::value<int>(&options.reachJobs)->default_value(0),                        //
     "threads for the reachability audit; 0 uses every core. The answer does not depend on it")         //
    ("reachability-only", bpo::bool_switch(&options.reachabilityOnly),                                  //
     "run only the reachability audit, which needs no magnetic field");

  bpo::variables_map arguments;
  try {
    bpo::store(bpo::parse_command_line(argc, argv, description), arguments);
    if (arguments.count("help") != 0u) {
      std::cout << description << '\n';
      return 0;
    }
    bpo::notify(arguments);
  } catch (const bpo::error& e) {
    std::cerr << "error: " << e.what() << "\n\n"
              << description << '\n';
    return 1;
  }

  const bool haveFieldFile = arguments.count("field-file") != 0u;
  const bool haveFieldCurrent = arguments.count("field-current") != 0u;

  // The reachability audit is a question about the geometry alone, so it is the one
  // part of this tool that can run without a field.
  if (options.reachabilityOnly) {
    TGeoManager::Import(options.geometryFile.c_str());
    if (gGeoManager == nullptr) {
      std::cerr << "error: no TGeoManager in " << options.geometryFile << '\n';
      return 1;
    }
    Report report;
    report("ALICE simulation geometry doctor -- reachability audit");
    report("");
    report("  geometry      : " + options.geometryFile);
    report(form("  volumes       : %d, media %d", gGeoManager->GetListOfVolumes()->GetEntries(),
                gGeoManager->GetListOfMedia()->GetEntries()));
    report("");
    const long dead = reportReachability(options.reachSamples, options.reachJobs, report);
    const std::string reportPath = options.outputPrefix + "-report.txt";
    report("wrote " + reportPath);
    report.write(reportPath);
    return dead == 0 ? 0 : 3;
  }

  if (haveFieldFile == haveFieldCurrent) {
    std::cerr << "error: give exactly one of --field-file and --field-current\n";
    return 1;
  }
  if (options.thresholdsGauss.empty()) {
    options.thresholdsGauss = {1., 10.};
  }
  std::sort(options.thresholdsGauss.begin(), options.thresholdsGauss.end());
  std::vector<double> thresholds; // kGauss, as the field itself reports
  for (double gauss : options.thresholdsGauss) {
    thresholds.push_back(gauss * 1e-3);
  }

  const std::string fieldSource =
    haveFieldFile ? options.fieldFile : form("createNominalField(%d)", options.fieldCurrent);
  o2::field::MagneticField* field =
    haveFieldFile ? loadFieldFromFile(options.fieldFile) : o2::field::MagneticField::createNominalField(options.fieldCurrent);
  if (field == nullptr) {
    std::cerr << "error: no usable magnetic field\n";
    return 1;
  }

  Report report;
  report("ALICE simulation geometry doctor");
  report("");
  report("  geometry      : " + options.geometryFile);
  report("  field         : " + fieldSource + ", parameterisation " + field->getParameterName());

  // --- the field-support model ------------------------------------------------
  Support support;
  bool supportFromCache = false;
  if (!options.supportFile.empty()) {
    std::ifstream cache(options.supportFile);
    if (cache) {
      json cached;
      try {
        cache >> cached;
      } catch (const std::exception& e) {
        std::cerr << "error: cannot parse " << options.supportFile << ": " << e.what() << '\n';
        return 1;
      }
      if (!supportFromJson(cached, support)) {
        return 1;
      }
      supportFromCache = true;
    }
  }

  if (supportFromCache) {
    if (support.models.size() != thresholds.size()) {
      std::cerr << "error: " << options.supportFile << " carries " << support.models.size()
                << " thresholds but " << thresholds.size() << " were requested\n";
      return 1;
    }
    for (size_t t = 0; t < thresholds.size(); ++t) {
      if (std::fabs(support.models[t].thresholdKG - thresholds[t]) > 1e-9) {
        std::cerr << "error: " << options.supportFile << " was built for a different threshold ("
                  << support.models[t].thresholdKG * 1000. << " G against " << thresholds[t] * 1000. << " G)\n";
        return 1;
      }
    }
    if (!support.parameterisation.empty() && support.parameterisation != field->getParameterName()) {
      std::cerr << "error: " << options.supportFile << " was built for parameterisation "
                << support.parameterisation << ", not " << field->getParameterName() << '\n';
      return 1;
    }
    report("  support model : " + options.supportFile + " (cached)");
  } else {
    support = buildSupport(field, thresholds, -3000., 3000.);
    if (!options.supportFile.empty()) {
      std::ofstream out(options.supportFile);
      out << supportToJson(support, fieldSource).dump(1, '\t') << '\n';
      report("  support model : built and written to " + options.supportFile);
    } else {
      report("  support model : built for this run");
    }
  }
  support.marginStrict = options.margin;

  std::string bandCounts;
  for (const auto& model : support.models) {
    bandCounts += form("%s%.1f G -> %zu bands", bandCounts.empty() ? "" : ", ", model.thresholdKG * 1000.,
                       model.bands.size());
  }
  report("                  " + bandCounts);
  report(form("                  margins: strict %.2f cm, tight %.2f cm, edge uncertainty %.2f cm",
              support.marginStrict, support.marginTight, support.edgeUncertainty));
  report("");

  // The model is only worth anything if it really is an outer bound, and only the
  // field itself can say so.
  report("outer-bound check");
  if (!violationScan(field, support, report)) {
    report("");
    report("The support model is not an outer bound on this field, so no placement can be called");
    report("field-free from it. Refusing to classify.");
    report.write(options.outputPrefix + "-report.txt");
    return 1;
  }
  report("");

  // --- the geometry -----------------------------------------------------------
  TGeoManager::Import(options.geometryFile.c_str());
  if (gGeoManager == nullptr) {
    std::cerr << "error: no TGeoManager in " << options.geometryFile << '\n';
    return 1;
  }
  report(form("  volumes       : %d, media %d", gGeoManager->GetListOfVolumes()->GetEntries(),
              gGeoManager->GetListOfMedia()->GetEntries()));
  report("");
  reportReachability(options.reachSamples, options.reachJobs, report);

  Doctor doctor(field, support);
  doctor.walk(gGeoManager->GetTopNode());
  report(form("  placements    : %zu classified, %zu detector subtrees pruned", doctor.rows().size(),
              doctor.nPruned()));
  report("");

  progress("classify: sampling the field inside every placement that reaches the support");
  doctor.classifyAll();
  doctor.findFindings();

  std::map<std::string, int> verdicts;
  for (const auto& row : doctor.rows()) {
    ++verdicts[row.verdict];
  }
  report("verdicts");
  for (const auto& verdict : verdicts) {
    report(form("  %-16s %7d", verdict.first.c_str(), verdict.second));
  }
  report("");

  // --- the reverse audit ------------------------------------------------------
  const double threshold = support.models.front().thresholdKG;
  std::map<std::string, std::pair<int, double>> reverseByVolume;
  for (const auto* row : doctor.reverseAudit()) {
    auto& entry = reverseByVolume[row->lv + " [" + row->medium + "]"];
    ++entry.first;
    entry.second = std::max(entry.second, std::max(row->maxB, row->wholeMaxB));
  }
  int inRealField = 0;
  for (const auto& entry : reverseByVolume) {
    inRealField += entry.second.second > threshold ? 1 : 0;
  }
  report(form("reverse audit: %zu placements carry a field-free medium yet reach into the field support",
              doctor.reverseAudit().size()));
  report(form("  %zu logical volumes, %d of them with real field in their own material",
              reverseByVolume.size(), inRealField));
  report(form("  %-46s %11s %16s", "volume [medium]", "placements", "max |B| [kG]"));
  for (const auto& entry : reverseByVolume) {
    report(form("  %-46s %11d %16.4f%s", entry.first.c_str(), entry.second.first, entry.second.second,
                entry.second.second > threshold ? "   <-- straight-line transport in real field" : ""));
  }
  report("");

  // --- the forward findings ---------------------------------------------------
  report(form("shared volumes: %zu logical volumes are placed both out of and into the field",
              doctor.sharedVolumes().size()));
  report(form("  %-28s %8s %8s  %s", "logical volume", "out", "in", "status"));
  for (size_t i = 0; i < doctor.sharedVolumes().size() && i < 20; ++i) {
    const auto& shared = doctor.sharedVolumes()[i];
    bool refused = false;
    bool approximate = false;
    for (const auto* row : shared.out) {
      refused = refused || row->sensitive || doctor.hasSensitive(row->node->GetVolume());
      approximate = approximate || row->approximateExtent;
    }
    report(form("  %-28s %8zu %8zu  %s%s", shared.lv.c_str(), shared.out.size(), shared.in.size(),
                refused ? "refused by default (sensitive)" : "proposed",
                approximate ? "  [extent approximate]" : ""));
  }
  if (doctor.sharedVolumes().size() > 20) {
    report(form("  ... and %zu more, all of them in the proposals file", doctor.sharedVolumes().size() - 20));
  }
  report("");

  report(form("heterogeneous mothers: %zu whose own material straddles the predicate",
              doctor.straddlingMothers().size()));
  for (size_t i = 0; i < doctor.straddlingMothers().size() && i < 10; ++i) {
    const auto* row = doctor.straddlingMothers()[i];
    report(form("  %-40s |B| in own material %.4g .. %.4f kG, %d daughters", row->lv.c_str(), row->wholeMinB,
                row->wholeMaxB, row->nDaughters));
  }
  report("");

  report(form("missing containers: %zu daughter clusters lie wholly on the field-free side",
              doctor.containers().size()));
  for (const auto& container : doctor.containers()) {
    report(form("  in %-14s z %9.2f .. %9.2f  rmax %7.2f  %3d daughters  %s%s", container.mother.c_str(),
                container.zlo, container.zhi, container.rmax, container.nDaughters,
                container.clearedByStrictMargin ? "clear by the strict margin" : "clear by the tight margin",
                container.sensitive ? "  [refused: sensitive]" : ""));
  }
  report("");

  // --- the anchor self-check --------------------------------------------------
  bool anchorsPassed = true;
  if (!options.anchorFile.empty()) {
    report("anchors");
    anchorsPassed = verifyAnchors(options.anchorFile, doctor, report);
    report(anchorsPassed ? "  all anchors reproduced" : "  ANCHORS FAILED");
    report("");
  }

  // --- outputs ----------------------------------------------------------------
  const std::string proposalsPath = options.outputPrefix + "-proposals.json";
  const std::string tablePath = options.outputPrefix + "-placements.csv";
  const std::string reportPath = options.outputPrefix + "-report.txt";
  std::ofstream proposals(proposalsPath);
  proposals << proposalsToJson(doctor, support, options.geometryFile, fieldSource).dump(1, '\t') << '\n';
  writePlacementCsv(doctor.rows(), tablePath);
  report("wrote " + proposalsPath + ", " + tablePath + " and " + reportPath);
  report.write(reportPath);

  return anchorsPassed ? 0 : 2;
}
