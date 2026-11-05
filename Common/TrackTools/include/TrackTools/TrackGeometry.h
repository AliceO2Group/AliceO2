// Copyright 2026-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_TRACKTOOLS_TRACKGEOMETRY_H_
#define ALICEO2_TRACKTOOLS_TRACKGEOMETRY_H_

/**
 * @file TrackGeometry.h
 * @brief Lightweight O2 track-geometry helpers for propagation to cylindrical radii and TPC row/sector-edge accounting.
 *
 * The header provides two related layers:
 *
 * 1. `o2::recohelpers`
 *    - propagation of an `o2::track::TrackPar` to a cylindrical radius;
 *    - H: fast analytic helix transport in a uniform Bz field;
 *    - K0: O2 `TrackPar` parameter propagation in Bz, without material correction;
 *    - derived global position/momentum, position azimuth, momentum azimuth, tan(lambda), and fractional sector position;
 *    - scalar and batch interfaces for primary-like and local O2 track parameters.
 *
 * 2. `o2::tpc::recohelpers`
 *    - TPC row sampling geometry from `o2::tpc::Mapper` (F6 centre correction pending);
 *    - H: analytic-helix row traversal;
 *    - L: stack-local linear approximation derived from helix endpoints;
 *    - K: O2 `Propagator` row traversal with `MatCorrType::USEMatCorrNONE`;
 *    - row counts, approximate track lengths, and 152-row masks for Crossed, Dead, Excluded, and Available rows;
 *    - configurable longitudinal, local-inclination, and active-pad-edge guard acceptance.
 *
 * The TPC classification currently samples the Mapper's inner-row reference radius (F6: pad-centre correction pending).
 * `Dead` is defined by the physical Mapper active-pad boundary, while
 * `Excluded` is the additional configurable guard measured inward from that active-pad edge. The nominal 20-degree
 * sector boundary is not the guard reference.
 *
 * This header is intentionally self-contained and header-only so the same definitions can be used from compiled O2
 * code and from PyROOT/commissioning wrappers.
 *
 * @par Propagation hierarchy and implementation status
 *
 * The frozen TrackGeometry design distinguishes the following transport levels:
 *
 * - **H** — analytic helix, constant signed Bz, no material.
 *   Implemented for the generic state-at-radius API and for TPC row traversal.
 * - **L** — stack-local linear approximation.
 *   Implemented for the TPC row/length approximation path only.
 * - **K0** — O2 `TrackPar` / TrackParametrization, constant signed Bz, no material.
 *   Implemented for the generic state-at-radius API.
 * - **K1** — O2 `Propagator`, initialized O2 magnetic-field machinery, material OFF.
 *   The current TPC row evaluator named **K** already uses `Propagator` with
 *   `MatCorrType::USEMatCorrNONE` and therefore exercises the K1 transport ingredients for this bounded TPC-row use
 *   case. The common generic K0/K1/K2 state-at-radius dispatch and final public K1 naming are **not yet implemented**.
 * - **K2** — O2 `Propagator`, initialized O2 magnetic-field machinery, material ON.
 *   **Not yet implemented.** The production material mode and material-map initialization must be explicitly verified
 *   during K2 commissioning before this mode is exposed.
 *
 * The intended diagnostic decomposition is:
 * `H-K0` = implementation/convention closure,
 * `K1-K0` = realistic-field/Propagator effect,
 * `K2-K1` = material effect.
 *
 * Planned functionality is documented here deliberately so that the public contract is visible before implementation;
 * planned modes must not be presented by wrappers as available until the corresponding C++ path and validation exist.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <cstddef>
#include <vector>
#include <utility>

#include "CommonConstants/MathConstants.h"
#include "MathUtils/Utils.h"
#include "ReconstructionDataFormats/Track.h"
#include "TPCBase/Mapper.h"
#include "DetectorsBase/Propagator.h"

namespace o2::recohelpers
{

/** @brief Status returned by the cylindrical-radius track-parameter evaluators. */
enum class TrackParametersStatus : int32_t {
  OK = 0,
  InvalidInput = 1,
  UnreachableRadius = 2,
  WrongDirection = 3,
  PropagationFailed = 4,
  MomentumInvalid = 5
};

/** @brief Global track state evaluated at a requested cylindrical radius.
 *
 * Invalid components remain NaN. `valid()` is true only for `TrackParametersStatus::OK`.
 */
struct TrackParametersAtRadius {
  float gx = std::numeric_limits<float>::quiet_NaN();
  float gy = std::numeric_limits<float>::quiet_NaN();
  float gz = std::numeric_limits<float>::quiet_NaN();
  float px = std::numeric_limits<float>::quiet_NaN();
  float py = std::numeric_limits<float>::quiet_NaN();
  float pz = std::numeric_limits<float>::quiet_NaN();
  TrackParametersStatus status = TrackParametersStatus::InvalidInput;

  bool valid() const { return status == TrackParametersStatus::OK; }
};

/** @brief Global azimuth of the propagated position, atan2(y,x), or NaN for an invalid state. */
inline float statePhiPos(const TrackParametersAtRadius& state)
{
  return state.valid() ? std::atan2(state.gy, state.gx) : std::numeric_limits<float>::quiet_NaN();
}

/** @brief Global azimuth of the propagated momentum, atan2(py,px), or NaN for an invalid state. */
inline float statePhiMom(const TrackParametersAtRadius& state)
{
  return state.valid() ? std::atan2(state.py, state.px) : std::numeric_limits<float>::quiet_NaN();
}

/** @brief tan(lambda)=pz/pT of the propagated state, or NaN for an invalid state. */
inline float stateTgl(const TrackParametersAtRadius& state)
{
  if (!state.valid()) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  const float pt = std::hypot(state.px, state.py);
  return pt > 0.f ? state.pz / pt : std::numeric_limits<float>::quiet_NaN();
}

/** @brief Fractional azimuthal position inside a periodic sector pattern.
 *
 * @param phi Global azimuth [rad].
 * @param nSectors Number of equal azimuthal sectors.
 * @param sectorPhaseRad Azimuth corresponding to fractional sector coordinate 0.
 * @return Fractional sector coordinate in [0,1), or NaN for invalid input.
 *
 * This is a periodic coordinate, not a metric distance to the physical active-pad edge.
 */
inline float dSector(float phi, int nSectors = 18, float sectorPhaseRad = 0.f)
{
  if (!std::isfinite(phi) || !std::isfinite(sectorPhaseRad) || nSectors <= 0) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  const float x = 0.5f * nSectors * (phi - sectorPhaseRad) / o2::constants::math::PI;
  const float phase = x - std::floor(x);
  return phase < 1.f ? phase : 0.f;
}

/** @brief `dSector()` evaluated from the propagated global position azimuth. */
inline float stateDSector(const TrackParametersAtRadius& state, int nSectors = 18, float sectorPhaseRad = 0.f)
{
  return dSector(statePhiPos(state), nSectors, sectorPhaseRad);
}

/** @brief Build an O2 `TrackPar` from a reduced primary-like state.
 *
 * @param phi Momentum azimuth at the supplied vertex [rad].
 * @param tgl tan(lambda).
 * @param qpt Signed q/pT [(GeV/c)^-1].
 * @param vx,vy,vz Vertex coordinates [cm].
 *
 * The transverse momentum magnitude is reconstructed as 1/|qpt| and the charge sign from qpt.
 */
inline o2::track::TrackPar makeTrackParFromPrimaryApproximation(float phi, float tgl, float qpt,
                                                                float vx = 0.f, float vy = 0.f, float vz = 0.f)
{
  const float pt = 1.f / std::abs(qpt);
  const int charge = qpt >= 0.f ? 1 : -1;
  o2::track::TrackPar::dim3_t xyz{{vx, vy, vz}};
  o2::track::TrackPar::dim3_t pxyz{{pt * std::cos(phi), pt * std::sin(phi), pt * tgl}};
  return o2::track::TrackPar(xyz, pxyz, charge, false);
}

/// Backward-compatible reduced-primary constructor used by existing commissioning code.
inline o2::track::TrackPar makePrimaryTrackPar(float phi, float tgl, float qpt, float vz = 0.f)
{
  return makeTrackParFromPrimaryApproximation(phi, tgl, qpt, 0.f, 0.f, vz);
}

/** @brief Extract finite global position and momentum from an O2 `TrackPar`. */
inline bool extractGlobalState(const o2::track::TrackPar& track,
                               o2::track::TrackPar::dim3_t& xyz,
                               o2::track::TrackPar::dim3_t& pxyz)
{
  track.getXYZGlo(xyz);
  if (!track.getPxPyPzGlo(pxyz)) {
    return false;
  }
  for (int i = 0; i < 3; ++i) {
    if (!std::isfinite(xyz[i]) || !std::isfinite(pxyz[i])) {
      return false;
    }
  }
  return true;
}

/** @brief Check whether the radial derivative is compatible with the requested propagation direction. */
inline bool directionMatches(o2::track::DirType dir, float drds)
{
  constexpr float eps = 1.e-6f;
  if (dir == o2::track::DirType::DirOutward) {
    return drds >= -eps;
  }
  if (dir == o2::track::DirType::DirInward) {
    return drds <= eps;
  }
  return true;
}

/** @brief H transport: analytic helix intersection with a cylindrical radius in uniform Bz.
 *
 * The implementation solves the transverse circle/circle intersection analytically, selects the first solution
 * compatible with the requested direction, and transports z with the constant input tan(lambda). Double precision is
 * used internally for the geometric construction, while the public state remains float.
 *
 * @param track Input O2 track parameters.
 * @param radius Target cylindrical radius [cm].
 * @param bzKG Uniform Bz field in O2 kG convention.
 * @param dir Required radial propagation direction.
 * @return Global state and status at the requested radius.
 */
inline TrackParametersAtRadius getParametersAtRadiusH(const o2::track::TrackPar& track, float radius, float bzKG,
                                                      o2::track::DirType dir = o2::track::DirType::DirOutward)
{
  TrackParametersAtRadius out;
  if (!(radius > 0.f) || !std::isfinite(radius) || !std::isfinite(bzKG) || !track.isValid()) {
    out.status = TrackParametersStatus::InvalidInput;
    return out;
  }

  o2::track::TrackPar::dim3_t xyz{}, pxyz{};
  if (!extractGlobalState(track, xyz, pxyz)) {
    out.status = TrackParametersStatus::MomentumInvalid;
    return out;
  }

  // Use double precision for the analytic construction even though TrackPar and
  // the public result are float-based.  For high-pT tracks the circle centre is
  // O(1/curvature) away while the requested detector radius is O(10^2 cm).
  // Performing the circle/circle intersection in float loses several microrad
  // through cancellation.  K0 is therefore the numerical oracle for the final
  // float state, while H keeps the fast analytic transport.
  const double x0 = static_cast<double>(xyz[0]);
  const double y0 = static_cast<double>(xyz[1]);
  const double z0 = static_cast<double>(xyz[2]);
  const double px0 = static_cast<double>(pxyz[0]);
  const double py0 = static_cast<double>(pxyz[1]);
  const double pz0 = static_cast<double>(pxyz[2]);
  const double targetR = static_cast<double>(radius);
  const double pt = std::hypot(px0, py0);
  if (!(pt > 0.) || !std::isfinite(pt)) {
    out.status = TrackParametersStatus::MomentumInvalid;
    return out;
  }

  const double phi0 = std::atan2(py0, px0);
  const double tgl = pz0 / pt;
  const double curv = static_cast<double>(track.getQ2Pt()) *
                      static_cast<double>(bzKG) *
                      static_cast<double>(o2::constants::math::B2C);
  constexpr double eps = 1.e-10;
  constexpr double curvEps = 1.e-12;
  const double twoPi = 2.0 * static_cast<double>(o2::constants::math::PI);

  bool foundAny = false;
  bool foundDirection = false;
  double bestSxy = std::numeric_limits<double>::infinity();
  double bestX = 0., bestY = 0., bestPhiMom = 0.;

  auto directionMatchesD = [&](double drds) {
    constexpr double dirEps = 1.e-10;
    if (dir == o2::track::DirType::DirOutward) {
      return drds >= -dirEps;
    }
    if (dir == o2::track::DirType::DirInward) {
      return drds <= dirEps;
    }
    return true;
  };

  auto consider = [&](double x, double y, double phiMom, double sxy) {
    if (!std::isfinite(sxy) || sxy < -eps) {
      return;
    }
    foundAny = true;
    sxy = std::max(0.0, sxy);
    const double drds = (x * std::cos(phiMom) + y * std::sin(phiMom)) / targetR;
    if (!directionMatchesD(drds)) {
      return;
    }
    foundDirection = true;
    if (sxy < bestSxy) {
      bestSxy = sxy;
      bestX = x;
      bestY = y;
      bestPhiMom = phiMom;
    }
  };

  if (std::abs(curv) < curvEps) {
    const double ux = px0 / pt, uy = py0 / pt;
    const double b = x0 * ux + y0 * uy;
    const double c = x0 * x0 + y0 * y0 - targetR * targetR;
    double disc = b * b - c;
    if (disc < -eps) {
      out.status = TrackParametersStatus::UnreachableRadius;
      return out;
    }
    disc = std::max(0.0, disc);
    const double root = std::sqrt(disc);
    for (const double sxy : {-b - root, -b + root}) {
      consider(x0 + sxy * ux, y0 + sxy * uy, phi0, sxy);
    }
  } else {
    const double xc = x0 - std::sin(phi0) / curv;
    const double yc = y0 + std::cos(phi0) / curv;
    const double rho = 1.0 / std::abs(curv);
    const double d = std::hypot(xc, yc);
    if (!(d > eps) || d > targetR + rho + eps || d < std::abs(targetR - rho) - eps) {
      out.status = TrackParametersStatus::UnreachableRadius;
      return out;
    }

    const double a = (targetR * targetR - rho * rho + d * d) / (2.0 * d);
    double h2 = targetR * targetR - a * a;
    if (h2 < -eps) {
      out.status = TrackParametersStatus::UnreachableRadius;
      return out;
    }
    h2 = std::max(0.0, h2);
    const double h = std::sqrt(h2);
    const double bx = a * xc / d, by = a * yc / d;
    const double ox = -yc * h / d, oy = xc * h / d;

    const std::array<std::array<double, 2>, 2> intersections{{{{bx + ox, by + oy}},
                                                              {{bx - ox, by - oy}}}};
    for (const auto& xy : intersections) {
      const double x = xy[0], y = xy[1];
      double sinPhi = curv * (x - xc);
      double cosPhi = -curv * (y - yc);
      const double norm = std::hypot(sinPhi, cosPhi);
      if (!(norm > 0.) || !std::isfinite(norm)) {
        continue;
      }
      sinPhi /= norm;
      cosPhi /= norm;
      const double phiMom = std::atan2(sinPhi, cosPhi);
      double delta = phiMom - phi0;
      if (curv > 0.) {
        while (delta < -eps) {
          delta += twoPi;
        }
      } else {
        while (delta > eps) {
          delta -= twoPi;
        }
      }
      consider(x, y, phiMom, delta / curv);
    }
  }

  if (!foundDirection || !std::isfinite(bestSxy)) {
    out.status = foundAny ? TrackParametersStatus::WrongDirection : TrackParametersStatus::UnreachableRadius;
    return out;
  }

  out.gx = static_cast<float>(bestX);
  out.gy = static_cast<float>(bestY);
  out.gz = static_cast<float>(z0 + bestSxy * tgl);
  out.px = static_cast<float>(pt * std::cos(bestPhiMom));
  out.py = static_cast<float>(pt * std::sin(bestPhiMom));
  out.pz = static_cast<float>(pz0);
  out.status = TrackParametersStatus::OK;
  return out;
}

/** @brief K0 transport: O2 `TrackPar` propagation to a cylindrical radius in Bz, without material correction.
 *
 * `getXatLabR()` locates the local-x target corresponding to the requested laboratory radius and
 * `propagateParamTo()` performs the O2 parameter transport. A state is reported as OK only when its final global
 * radius agrees with the requested cylinder within 2e-3 cm; an off-radius one-step result is PropagationFailed.
 * This path is used as an independent O2 transport reference for the fast analytic H implementation.
 *
 * @param track Input O2 track parameters.
 * @param radius Target cylindrical radius [cm].
 * @param bzKG Bz field in O2 kG convention.
 * @param dir Required radial propagation direction.
 * @return Global state and status at the requested radius.
 */
inline TrackParametersAtRadius getParametersAtRadiusK0(const o2::track::TrackPar& track, float radius, float bzKG,
                                                       o2::track::DirType dir = o2::track::DirType::DirOutward)
{
  TrackParametersAtRadius out;
  if (!(radius > 0.f) || !std::isfinite(radius) || !std::isfinite(bzKG) || !track.isValid()) {
    out.status = TrackParametersStatus::InvalidInput;
    return out;
  }

  float xAtR = 0.f;
  if (!track.getXatLabR(radius, xAtR, bzKG, dir)) {
    out.status = TrackParametersStatus::UnreachableRadius;
    return out;
  }

  auto propagated = track;
  if (!propagated.propagateParamTo(xAtR, bzKG)) {
    out.status = TrackParametersStatus::PropagationFailed;
    return out;
  }

  o2::track::TrackPar::dim3_t xyz{}, pxyz{};
  if (!extractGlobalState(propagated, xyz, pxyz)) {
    out.status = TrackParametersStatus::MomentumInvalid;
    return out;
  }

  // A successful internal TrackPar update is not sufficient: the returned
  // global state must actually lie on the requested cylinder (Architect A5).
  constexpr float radiusToleranceCm = 2.e-3f;
  const float resultRadius = std::hypot(xyz[0], xyz[1]);
  if (!std::isfinite(resultRadius) || std::abs(resultRadius - radius) > radiusToleranceCm) {
    out.status = TrackParametersStatus::PropagationFailed;
    return out;
  }

  out.gx = xyz[0];
  out.gy = xyz[1];
  out.gz = xyz[2];
  out.px = pxyz[0];
  out.py = pxyz[1];
  out.pz = pxyz[2];
  out.status = TrackParametersStatus::OK;
  return out;
}

/** @brief Backward-compatible scalar wrapper returning only the propagated position azimuth using K0. */
inline float phiAtRadius(const o2::track::TrackPar& track, float radius, float bzKG,
                         o2::track::DirType dir = o2::track::DirType::DirOutward)
{
  return statePhiPos(getParametersAtRadiusK0(track, radius, bzKG, dir));
}

/** @brief Backward-compatible scalar wrapper returning only the propagated fractional sector coordinate using K0. */
inline float dSectorAtRadius(const o2::track::TrackPar& track, float radius, float bzKG, int nSectors = 18,
                             o2::track::DirType dir = o2::track::DirType::DirOutward, float sectorPhaseRad = 0.f)
{
  return stateDSector(getParametersAtRadiusK0(track, radius, bzKG, dir), nSectors, sectorPhaseRad);
}

/** @brief Position azimuth at radius for the reduced primary-like parameterization, evaluated with K0. */
inline float primaryPhiAtRadius(float phi, float tgl, float qpt, float radius, float bzKG, float vz = 0.f)
{
  if (!std::isfinite(phi) || !std::isfinite(tgl) || !std::isfinite(qpt) || std::abs(qpt) < 1.e-12f) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  return phiAtRadius(makePrimaryTrackPar(phi, tgl, qpt, vz), radius, bzKG, o2::track::DirType::DirOutward);
}

/** @brief Fractional sector coordinate at radius for the reduced primary-like parameterization, evaluated with K0. */
inline float primaryDSectorAtRadius(float phi, float tgl, float qpt, float radius, float bzKG,
                                    int nSectors = 18, float vz = 0.f, float sectorPhaseRad = 0.f)
{
  const float phiR = primaryPhiAtRadius(phi, tgl, qpt, radius, bzKG, vz);
  return dSector(phiR, nSectors, sectorPhaseRad);
}

/** @brief Structure-of-arrays result for batch cylindrical-radius propagation.
 *
 * Besides the global position/momentum, the batch result stores commonly used derived observables and one status value
 * per input track.
 */
struct TrackParametersAtRadiusBatchResult {
  std::vector<float> gx, gy, gz, px, py, pz;
  std::vector<float> phiPos, phiMom, tgl, dSector;
  std::vector<int32_t> status;

  void resize(std::size_t n)
  {
    gx.resize(n);
    gy.resize(n);
    gz.resize(n);
    px.resize(n);
    py.resize(n);
    pz.resize(n);
    phiPos.resize(n);
    phiMom.resize(n);
    tgl.resize(n);
    dSector.resize(n);
    status.resize(n);
  }

  void set(std::size_t i, const TrackParametersAtRadius& state, int nSectors, float sectorPhaseRad)
  {
    gx[i] = state.gx;
    gy[i] = state.gy;
    gz[i] = state.gz;
    px[i] = state.px;
    py[i] = state.py;
    pz[i] = state.pz;
    phiPos[i] = statePhiPos(state);
    phiMom[i] = statePhiMom(state);
    tgl[i] = stateTgl(state);
    dSector[i] = stateDSector(state, nSectors, sectorPhaseRad);
    status[i] = static_cast<int32_t>(state.status);
  }
};

template <typename Evaluator>
inline TrackParametersAtRadiusBatchResult primaryTrackParametersAtRadiusBatch(
  const float* phi, const float* tgl, const float* qpt,
  const float* vx, const float* vy, const float* vz,
  std::size_t n, float radius, float bzKG, int nSectors, float sectorPhaseRad, Evaluator&& evaluate)
{
  TrackParametersAtRadiusBatchResult out;
  out.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    if (!phi || !tgl || !qpt || !std::isfinite(qpt[i]) || std::abs(qpt[i]) < 1.e-12f) {
      TrackParametersAtRadius invalid;
      invalid.status = TrackParametersStatus::InvalidInput;
      out.set(i, invalid, nSectors, sectorPhaseRad);
      continue;
    }
    const auto track = makeTrackParFromPrimaryApproximation(phi[i], tgl[i], qpt[i],
                                                            vx ? vx[i] : 0.f,
                                                            vy ? vy[i] : 0.f,
                                                            vz ? vz[i] : 0.f);
    out.set(i, evaluate(track, radius, bzKG), nSectors, sectorPhaseRad);
  }
  return out;
}

/** @brief Batch H transport for reduced primary-like inputs. */
inline TrackParametersAtRadiusBatchResult primaryTrackParametersAtRadiusHBatch(
  const float* phi, const float* tgl, const float* qpt,
  const float* vx, const float* vy, const float* vz,
  std::size_t n, float radius, float bzKG, int nSectors = 18, float sectorPhaseRad = 0.f)
{
  return primaryTrackParametersAtRadiusBatch(
    phi, tgl, qpt, vx, vy, vz, n, radius, bzKG, nSectors, sectorPhaseRad,
    [](const o2::track::TrackPar& track, float r, float bz) {
      return getParametersAtRadiusH(track, r, bz, o2::track::DirType::DirOutward);
    });
}

/** @brief Batch K0 transport for reduced primary-like inputs. */
inline TrackParametersAtRadiusBatchResult primaryTrackParametersAtRadiusK0Batch(
  const float* phi, const float* tgl, const float* qpt,
  const float* vx, const float* vy, const float* vz,
  std::size_t n, float radius, float bzKG, int nSectors = 18, float sectorPhaseRad = 0.f)
{
  return primaryTrackParametersAtRadiusBatch(
    phi, tgl, qpt, vx, vy, vz, n, radius, bzKG, nSectors, sectorPhaseRad,
    [](const o2::track::TrackPar& track, float r, float bz) {
      return getParametersAtRadiusK0(track, r, bz, o2::track::DirType::DirOutward);
    });
}

/** @brief Build an O2 `TrackPar` from the standard local O2 track parameterization. */
inline o2::track::TrackPar makeTrackParFromLocalParameters(float x, float alpha, float y, float z,
                                                           float snp, float tgl, float qpt, int absCharge = 1)
{
  o2::track::TrackPar::params_t par{{y, z, snp, tgl, qpt}};
  return o2::track::TrackPar(x, alpha, par, std::max(1, std::abs(absCharge)));
}

template <typename Evaluator>
inline TrackParametersAtRadiusBatchResult localTrackParametersAtRadiusBatch(
  const float* x, const float* alpha, const float* y, const float* z,
  const float* snp, const float* tgl, const float* qpt,
  std::size_t n, float radius, float bzKG, int nSectors, float sectorPhaseRad, Evaluator&& evaluate)
{
  TrackParametersAtRadiusBatchResult out;
  out.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    if (!x || !alpha || !y || !z || !snp || !tgl || !qpt) {
      TrackParametersAtRadius invalid;
      invalid.status = TrackParametersStatus::InvalidInput;
      out.set(i, invalid, nSectors, sectorPhaseRad);
      continue;
    }
    const auto track = makeTrackParFromLocalParameters(x[i], alpha[i], y[i], z[i], snp[i], tgl[i], qpt[i]);
    out.set(i, evaluate(track, radius, bzKG), nSectors, sectorPhaseRad);
  }
  return out;
}

/** @brief Batch H transport for tracks supplied in local O2 parameterization. */
inline TrackParametersAtRadiusBatchResult localTrackParametersAtRadiusHBatch(
  const float* x, const float* alpha, const float* y, const float* z,
  const float* snp, const float* tgl, const float* qpt,
  std::size_t n, float radius, float bzKG, int nSectors = 18, float sectorPhaseRad = 0.f)
{
  return localTrackParametersAtRadiusBatch(
    x, alpha, y, z, snp, tgl, qpt, n, radius, bzKG, nSectors, sectorPhaseRad,
    [](const o2::track::TrackPar& track, float r, float bz) {
      return getParametersAtRadiusH(track, r, bz, o2::track::DirType::DirOutward);
    });
}

/** @brief Batch K0 transport for tracks supplied in local O2 parameterization. */
inline TrackParametersAtRadiusBatchResult localTrackParametersAtRadiusK0Batch(
  const float* x, const float* alpha, const float* y, const float* z,
  const float* snp, const float* tgl, const float* qpt,
  std::size_t n, float radius, float bzKG, int nSectors = 18, float sectorPhaseRad = 0.f)
{
  return localTrackParametersAtRadiusBatch(
    x, alpha, y, z, snp, tgl, qpt, n, radius, bzKG, nSectors, sectorPhaseRad,
    [](const o2::track::TrackPar& track, float r, float bz) {
      return getParametersAtRadiusK0(track, r, bz, o2::track::DirType::DirOutward);
    });
}

/// Backward-compatible batch wrappers retained for existing commissioning notebooks/tests.
inline std::vector<float> primaryPhiAtRadiusBatch(const float* phi, const float* tgl, const float* qpt, const float* vz,
                                                  std::size_t n, float radius, float bzKG)
{
  auto batch = primaryTrackParametersAtRadiusK0Batch(phi, tgl, qpt, nullptr, nullptr, vz, n, radius, bzKG);
  return std::move(batch.phiPos);
}

inline std::vector<float> primaryDSectorAtRadiusBatch(const float* phi, const float* tgl, const float* qpt, const float* vz,
                                                      std::size_t n, float radius, float bzKG, int nSectors = 18,
                                                      float sectorPhaseRad = 0.f)
{
  auto batch = primaryTrackParametersAtRadiusK0Batch(phi, tgl, qpt, nullptr, nullptr, vz, n, radius, bzKG,
                                                     nSectors, sectorPhaseRad);
  return std::move(batch.dSector);
}

/** @brief Legacy compact batch result containing only position azimuth and fractional sector coordinate. */
struct PrimaryTrackGeometryBatchResult {
  std::vector<float> phi;
  std::vector<float> dSector;
};

inline PrimaryTrackGeometryBatchResult primaryTrackGeometryAtRadiusBatch(
  const float* phi, const float* tgl, const float* qpt, const float* vz,
  std::size_t n, float radius, float bzKG, int nSectors = 18, float sectorPhaseRad = 0.f)
{
  auto batch = primaryTrackParametersAtRadiusK0Batch(phi, tgl, qpt, nullptr, nullptr, vz, n, radius, bzKG,
                                                     nSectors, sectorPhaseRad);
  return {std::move(batch.phiPos), std::move(batch.dSector)};
}

} // namespace o2::recohelpers

namespace o2::tpc::recohelpers
{

/**
 * @name TPC row and sector-edge accounting
 *
 * These helpers evaluate the first outward TPC trajectory branch at sampled Mapper row radii and classify every accepted
 * crossed row using the physical active-pad edge plus an optional inward guard.
 */
///@{

/** @brief Status returned by the TPC row/sector-edge evaluators. */
enum class SectorEdgeStatus : int32_t {
  OK = 0,
  InvalidInput = 1,
  NoRowsCrossed = 2,
  PropagationFailed = 3,
  FieldNotInitialized = 4
};

/** @brief Reduced primary-track seed used by the TPC row evaluators.
 *
 * The seed assumes x=y=0 at the primary vertex. Bz follows the O2 kG convention.
 */
struct TPCPrimaryTrackSeed {
  float phi = 0.f;                                      // momentum azimuth at the primary vertex [rad]
  float tgl = 0.f;                                      // tan(lambda)
  float qpt = 0.f;                                      // signed q/pT [(GeV/c)^-1]
  float vz = 0.f;                                       // primary vertex z [cm]
  float bzKG = std::numeric_limits<float>::quiet_NaN(); // signed O2 field convention [kG]; must be supplied
};

/** @brief Configuration for TPC row acceptance and sector-edge classification.
 *
 * A row is `Crossed` only when its row-centre cylindrical surface is reached on the first outward trajectory branch and
 * the propagated state satisfies:
 *   - |z| <= `zMaxCm`;
 *   - |snp| < `maxSnp`.
 *
 * The row-reachability limit uses radial-frame `snp = sin(phiMom-phiPos)` from global position/momentum
 * azimuths. This is invariant under O2 tracking-frame rotation; the Mapper sector frame remains in use only
 * for local-y active-pad-edge classification.
 * Rows outside this acceptance are not Crossed and therefore are not Dead, Excluded, or Available.
 *
 * Every Crossed row belongs to exactly one of three mutually exclusive classes:
 *   - `Dead`: row-centre position lies outside the Mapper active-pad boundary; independent of guard width.
 *   - `Excluded`: position is inside the active-pad boundary but closer than the configured guard to that boundary.
 *   - `Available`: neither Dead nor Excluded.
 *
 * Therefore, independently for row counts, masks, and approximate lengths:
 * `Crossed = Dead + Excluded + Available`.
 *
 * `guardLowerCm` and `guardUpperCm` are measured inward from the physical active-pad edge on the negative and positive
 * local-y sides respectively. They are not measured from the nominal 20-degree sector boundary.
 */
struct TPCSectorEdgeConfig {
  float guardLowerCm = 0.f; // active-side exclusion adjacent to negative local-y edge
  float guardUpperCm = 0.f; // active-side exclusion adjacent to positive local-y edge
  float zMaxCm = 250.f;     // symmetric TPC longitudinal acceptance |z| <= zMaxCm
  float maxSnp = 0.9f;      // radial-frame |sin(phiMom-phiPos)| < maxSnp; not O2 TrackPar::getSnp()
};

/** @brief Result of one TPC row/sector-edge evaluation.
 *
 * `nRows*` count accepted row-centre intersections. `length*Cm` accumulates the corresponding approximate track length
 * as radial row thickness multiplied by ds/dr. The four 152-row masks deliberately retain redundant information so the
 * partition identity can be checked directly.
 *
 * `maxRadiusCm` is the radius of the last accepted sampled row; NaN if none. It is not the continuous
 * mathematical turning radius of a helix and is not a propagation-failure sentinel.
 * `firstDeadRow` and `lastDeadRow` delimit the first/last crossed rows classified outside the active-pad boundary;
 * they remain -1 when no Dead row is found.
 */
struct TPCSectorEdgeResult {
  float lengthCrossedCm = 0.f;
  float lengthDeadCm = 0.f;
  float lengthExcludedCm = 0.f;
  float lengthAvailableCm = 0.f;
  float maxRadiusCm = std::numeric_limits<float>::quiet_NaN(); // last accepted row radius [cm], not turning radius

  uint16_t nRowsCrossed = 0;
  uint16_t nRowsDead = 0;
  uint16_t nRowsExcluded = 0;
  uint16_t nRowsAvailable = 0;

  std::array<uint32_t, 5> crossedMask{};
  std::array<uint32_t, 5> deadMask{};
  std::array<uint32_t, 5> excludedMask{};
  std::array<uint32_t, 5> availableMask{};

  int16_t firstDeadRow = -1;
  int16_t lastDeadRow = -1;
  SectorEdgeStatus status = SectorEdgeStatus::OK;
};

/** @brief Structure-of-arrays batch result for the TPC row/sector-edge evaluators.
 *
 * The batch API exposes counts, approximate lengths, and status. Per-row masks remain available from the scalar API.
 */
struct TPCSectorEdgeBatchResult {
  std::vector<float> lengthCrossedCm;
  std::vector<float> lengthDeadCm;
  std::vector<float> lengthExcludedCm;
  std::vector<float> lengthAvailableCm;
  std::vector<float> maxRadiusCm;

  std::vector<int16_t> nRowsCrossed;
  std::vector<int16_t> nRowsDead;
  std::vector<int16_t> nRowsExcluded;
  std::vector<int16_t> nRowsAvailable;

  std::vector<int32_t> status;

  void resize(std::size_t n)
  {
    lengthCrossedCm.resize(n);
    lengthDeadCm.resize(n);
    lengthExcludedCm.resize(n);
    lengthAvailableCm.resize(n);
    maxRadiusCm.resize(n);
    nRowsCrossed.resize(n);
    nRowsDead.resize(n);
    nRowsExcluded.resize(n);
    nRowsAvailable.resize(n);
    status.resize(n);
  }

  void set(std::size_t i, const TPCSectorEdgeResult& r)
  {
    lengthCrossedCm[i] = r.lengthCrossedCm;
    lengthDeadCm[i] = r.lengthDeadCm;
    lengthExcludedCm[i] = r.lengthExcludedCm;
    lengthAvailableCm[i] = r.lengthAvailableCm;
    maxRadiusCm[i] = r.maxRadiusCm;
    nRowsCrossed[i] = static_cast<int16_t>(r.nRowsCrossed);
    nRowsDead[i] = static_cast<int16_t>(r.nRowsDead);
    nRowsExcluded[i] = static_cast<int16_t>(r.nRowsExcluded);
    nRowsAvailable[i] = static_cast<int16_t>(r.nRowsAvailable);
    status[i] = static_cast<int32_t>(r.status);
  }
};

/** @brief Internal global trajectory sample at one TPC row-centre radius. */
struct TrajectoryPoint {
  float x = 0.f;
  float y = 0.f;
  float z = 0.f;
  float phiPos = 0.f;
  float phiMom = 0.f;
  float dsDr = 0.f;
  bool valid = false;
};

/** @brief Debug-only row state; trajectory[row] is the state at that sampled TPC row.
 * Append even the first state rejected by maxSnp. With tracing disabled no momentum conversion is performed.
 */
inline void appendTrajectoryState(std::vector<o2::recohelpers::TrackParametersAtRadius>& trajectory,
                                  const TrajectoryPoint& p, float qpt, float tgl)
{
  const float pt = 1.f / std::abs(qpt);
  trajectory.push_back({p.x, p.y, p.z, pt * std::cos(p.phiMom), pt * std::sin(p.phiMom),
                        pt * tgl, o2::recohelpers::TrackParametersStatus::OK});
}

/** @brief Set one of the 152 TPC row bits in the five-word row mask. */
inline void setMaskBit(std::array<uint32_t, 5>& mask, int row)
{
  mask[row >> 5] |= uint32_t(1) << (row & 31);
}

/** @brief Return the Mapper pad-region index containing a global TPC row, or -1 if none. */
inline int regionForRow(const o2::tpc::Mapper& mapper, int row)
{
  for (int i = 0; i < mapper.getNumberOfPadRegions(); ++i) {
    const auto& r = mapper.getPadRegionInfo(i);
    const int first = r.getGlobalRowOffset(), last = first + r.getNumberOfPadRows();
    if (row >= first && row < last) {
      return i;
    }
  }
  return -1;
}

/** @brief Current sampled TPC row radius [cm]; F6 physical pad-centre correction remains pending. */
inline float rowRadius(const o2::tpc::Mapper& mapper, int row)
{
  const int ireg = regionForRow(mapper, row);
  if (ireg < 0) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  const auto& r = mapper.getPadRegionInfo(ireg);
  return r.getRadiusFirstRow() + (row - r.getGlobalRowOffset()) * r.getPadHeight();
}

/** @brief Radial row thickness, taken from the Mapper pad height [cm]. */
inline float rowLengthRadial(const o2::tpc::Mapper& mapper, int row)
{
  const int ireg = regionForRow(mapper, row);
  return ireg < 0 ? 0.f : mapper.getPadRegionInfo(ireg).getPadHeight();
}

/** @brief Active half-width of one TPC row in local-y from Mapper pad count and pad width [cm]. */
inline float rowActiveHalfWidth(const o2::tpc::Mapper& mapper, int row)
{
  const int ireg = regionForRow(mapper, row);
  if (ireg < 0) {
    return 0.f;
  }
  const auto& r = mapper.getPadRegionInfo(ireg);
  return 0.5f * mapper.getNumberOfPadsInRowSector(row) * r.getPadWidth();
}

/** @brief Radial-frame sin(phiMom-phiPos); independent of the O2 TrackPar alpha rotation.
 *
 * This is the row-stopping convention ratified for PHASE_0_3. The Mapper sector-local
 * inclination (phiMom-alphaSector) is distinct and does not control row reachability.
 */
inline float trajectoryRadialSnp(const TrajectoryPoint& p)
{
  if (!p.valid || !std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.phiMom)) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  return std::sin(p.phiMom - std::atan2(p.y, p.x));
}

/** @brief Backward-compatible internal spelling; use trajectoryRadialSnp for new code. */
inline float trajectoryLocalSnp(const TrajectoryPoint& p) { return trajectoryRadialSnp(p); }

/** @brief True when a valid row-centre trajectory point satisfies the configured z and snp acceptance. */
inline bool rowWithinAcceptance(const TrajectoryPoint& p, const TPCSectorEdgeConfig& cfg)
{
  const float snp = trajectoryRadialSnp(p);
  return p.valid && std::isfinite(p.z) && std::abs(p.z) <= cfg.zMaxCm &&
         std::isfinite(snp) && std::abs(snp) < cfg.maxSnp;
}

/** @brief True when the radial-frame inclination reaches the configured stopping boundary. */
inline bool reachedMaxSnp(const TrajectoryPoint& p, const TPCSectorEdgeConfig& cfg)
{
  const float snp = trajectoryRadialSnp(p);
  return std::isfinite(snp) && std::abs(snp) >= cfg.maxSnp;
}

/** @brief Validate the complete input state and report its global reference radius.
 * An outward-only row query does not back-propagate to rows inside this radius.
 */
inline bool trackOutwardReference(const o2::track::TrackPar& track, float& radius)
{
  o2::track::TrackPar::dim3_t xyz{}, pxyz{};
  if (!track.isValid() || !o2::recohelpers::extractGlobalState(track, xyz, pxyz)) {
    return false;
  }
  radius = std::hypot(xyz[0], xyz[1]);
  if (!std::isfinite(radius) || !std::isfinite(track.getQ2Pt()) ||
      std::abs(track.getQ2Pt()) < 1.e-12f) {
    return false;
  }
  // Negative radial motion at the supplied reference is not the first outward branch.
  return radius < 1.e-5f || xyz[0] * pxyz[0] + xyz[1] * pxyz[1] >= 0.f;
}

/** @brief One TPC row trajectory sample from a complete O2 local five-parameter track. */
inline TrajectoryPoint helixAtRadius(const o2::track::TrackPar& track, float radius, float bzKG)
{
  TrajectoryPoint p;
  if (!track.isValid() || !std::isfinite(bzKG) || !(radius > 0.f)) {
    return p;
  }
  const auto state = o2::recohelpers::getParametersAtRadiusH(
    track, radius, bzKG, o2::track::DirType::DirOutward);
  if (!state.valid()) {
    return p;
  }
  p.x = state.gx;
  p.y = state.gy;
  p.z = state.gz;
  p.phiPos = o2::recohelpers::statePhiPos(state);
  p.phiMom = o2::recohelpers::statePhiMom(state);
  const float c = std::abs(std::cos(p.phiMom - p.phiPos));
  const float tgl = o2::recohelpers::stateTgl(state);
  p.dsDr = std::sqrt(1.f + tgl * tgl) / std::max(c, 1.e-6f);
  p.valid = std::isfinite(p.dsDr);
  return p;
}

/** @brief Legacy reduced-primary adapter to the same complete-track H implementation. */
inline TrajectoryPoint primaryHelixAtRadius(const TPCPrimaryTrackSeed& seed, float radius)
{
  if (!std::isfinite(seed.phi) || !std::isfinite(seed.tgl) || !std::isfinite(seed.qpt) ||
      !std::isfinite(seed.bzKG) || std::abs(seed.qpt) < 1.e-12f) {
    return {};
  }
  return helixAtRadius(o2::recohelpers::makeTrackParFromPrimaryApproximation(
                         seed.phi, seed.tgl, seed.qpt, 0.f, 0.f, seed.vz),
                       radius, seed.bzKG);
}

/** @brief Apply acceptance and classify one crossed row as Dead, Excluded, or Available.
 *
 * The physical active-pad edge is `N_pads(row) * padWidth / 2` in sector-local y. A point outside that edge is Dead.
 * Otherwise the side-dependent guard is measured inward from that edge; points inside the guard are Excluded and the
 * remaining crossed rows are Available.
 *
 * This function updates row counts, approximate lengths, row masks, and first/last Dead row consistently.
 */
inline void accumulateRow(const o2::tpc::Mapper& mapper, int row, const TrajectoryPoint& p, float rowLength,
                          const TPCSectorEdgeConfig& cfg, TPCSectorEdgeResult& out)
{
  if (!rowWithinAcceptance(p, cfg) || !std::isfinite(rowLength) || rowLength <= 0.f) {
    return;
  }

  const float phi = std::atan2(p.y, p.x);
  const int sector = o2::math_utils::angle2Sector(phi);
  const float alpha = o2::math_utils::sector2Angle(sector);
  const float localY = -p.x * std::sin(alpha) + p.y * std::cos(alpha);
  const float halfWidth = rowActiveHalfWidth(mapper, row);
  const float edgeDistance = halfWidth - std::abs(localY);
  const float guard = localY < 0.f ? cfg.guardLowerCm : cfg.guardUpperCm;

  ++out.nRowsCrossed;
  out.maxRadiusCm = rowRadius(mapper, row);
  out.lengthCrossedCm += rowLength;
  setMaskBit(out.crossedMask, row);

  if (edgeDistance < 0.f) {
    ++out.nRowsDead;
    out.lengthDeadCm += rowLength;
    setMaskBit(out.deadMask, row);
    if (out.firstDeadRow < 0) {
      out.firstDeadRow = row;
    }
    out.lastDeadRow = row;
  } else if (edgeDistance < guard) {
    ++out.nRowsExcluded;
    out.lengthExcludedCm += rowLength;
    setMaskBit(out.excludedMask, row);
  } else {
    ++out.nRowsAvailable;
    out.lengthAvailableCm += rowLength;
    setMaskBit(out.availableMask, row);
  }
}

/** @brief H row evaluator: analytic helix sampled at the current Mapper row radii.
 *
 * Rows are processed from inner to outer radius. Evaluation stops when radial |snp| reaches `cfg.maxSnp`.
 * The result uses the common Crossed/Dead/Excluded/Available classification in `accumulateRow()`.
 */
inline TPCSectorEdgeResult sectorEdgeHelixRows(const o2::track::TrackPar& track, float bzKG, const TPCSectorEdgeConfig& cfg = {},
                                               std::vector<o2::recohelpers::TrackParametersAtRadius>* trajectory = nullptr)
{
  TPCSectorEdgeResult out;
  if (trajectory) {
    trajectory->clear();
  }
  float startRadius = 0.f;
  if (!std::isfinite(bzKG) || !trackOutwardReference(track, startRadius)) {
    out.status = SectorEdgeStatus::InvalidInput;
    return out;
  }
  const auto& mapper = o2::tpc::Mapper::instance();

  for (int row = 0; row < mapper.getNumberOfRows(); ++row) {
    const float r = rowRadius(mapper, row);
    if (r + 1.e-4f < startRadius) {
      if (trajectory) {
        trajectory->emplace_back(); // Preserve trajectory[row] indexing for skipped rows.
      }
      continue; // A full track can start inside the TPC; do not count earlier rows.
    }
    const auto p = helixAtRadius(track, r, bzKG);
    if (!p.valid) {
      break; // a larger radius cannot re-enter the first outward primary-helix branch
    }
    if (trajectory) {
      appendTrajectoryState(*trajectory, p, track.getQ2Pt(), track.getTgl());
    }
    if (reachedMaxSnp(p, cfg)) {
      break;
    }
    accumulateRow(mapper, row, p, rowLengthRadial(mapper, row) * p.dsDr, cfg, out);
  }

  if (!out.nRowsCrossed) {
    out.status = SectorEdgeStatus::NoRowsCrossed;
  }
  return out;
}

/** @brief Reduced-primary compatibility adapter: construct TrackPar and call the full H evaluator. */
inline TPCSectorEdgeResult sectorEdgeHelixRows(const TPCPrimaryTrackSeed& seed, const TPCSectorEdgeConfig& cfg = {},
                                               std::vector<o2::recohelpers::TrackParametersAtRadius>* trajectory = nullptr)
{
  if (!std::isfinite(seed.phi) || !std::isfinite(seed.tgl) || !std::isfinite(seed.qpt) ||
      !std::isfinite(seed.bzKG) || std::abs(seed.qpt) < 1.e-12f) {
    TPCSectorEdgeResult r;
    r.status = SectorEdgeStatus::InvalidInput;
    return r;
  }
  return sectorEdgeHelixRows(o2::recohelpers::makeTrackParFromPrimaryApproximation(
                               seed.phi, seed.tgl, seed.qpt, 0.f, 0.f, seed.vz),
                             seed.bzKG, cfg, trajectory);
}

/** @brief Conservative path-continuity guard against a later outward orbit.
 *
 * This is a gross-branch-rejection criterion, NOT an H/K numerical-agreement tolerance.
 * The generous factor 10 avoids flagging near-boundary valid segments; a long detour
 * around a curl cannot be accepted merely because its final radial direction is outward.
 */
inline bool plausibleFirstOutwardStep(float dr, float ds, float tgl, float maxSnp)
{
  if (!(dr > 0.f) || !(ds > 0.f) || !std::isfinite(ds) || !std::isfinite(tgl) ||
      !(maxSnp > 0.f && maxSnp < 1.f)) {
    return false;
  }
  const float minRadialCos = std::sqrt(1.f - maxSnp * maxSnp);
  return ds <= 10.f * dr * std::sqrt(1.f + tgl * tgl) / minRadialCos;
}

/** @brief K row evaluator: O2 `Propagator` traversal of TPC row radii without material correction.
 *
 * The first eligible row is initialized from analytic H propagation of the supplied complete TrackPar,
 * after which the O2 Propagator advances the track
 * row by row with `MatCorrType::USEMatCorrNONE`. The radial-frame maxSnp stop and common row classification
 * are applied independently of the Propagator's rotating internal tracking frame.
 * Numerical propagation failures return `PropagationFailed`; a physical snp stop returns `OK` with the last
 * accepted row reported in `maxRadiusCm`.
 *
 * The O2 magnetic field must already be initialized; otherwise `FieldNotInitialized` is returned.
 */
inline TPCSectorEdgeResult sectorEdgeO2PropagatorRows(const o2::track::TrackPar& inputTrack, float bzKG,
                                                      const TPCSectorEdgeConfig& cfg = {},
                                                      std::vector<o2::recohelpers::TrackParametersAtRadius>* trajectory = nullptr)
{
  TPCSectorEdgeResult out;
  if (trajectory) {
    trajectory->clear();
  }
  const auto& mapper = o2::tpc::Mapper::instance();
  auto* prop = o2::base::Propagator::Instance();
  if (!prop->hasMagFieldSet()) {
    out.status = SectorEdgeStatus::FieldNotInitialized;
    return out;
  }
  float startRadius = 0.f;
  if (!std::isfinite(bzKG) || !trackOutwardReference(inputTrack, startRadius) ||
      !std::isfinite(cfg.maxSnp) || !(cfg.maxSnp > 0.f && cfg.maxSnp < 1.f)) {
    out.status = SectorEdgeStatus::InvalidInput;
    return out;
  }

  int firstRow = 0;
  while (firstRow < mapper.getNumberOfRows() &&
         rowRadius(mapper, firstRow) + 1.e-4f < startRadius) {
    ++firstRow;
  }
  if (firstRow == mapper.getNumberOfRows()) {
    out.status = SectorEdgeStatus::NoRowsCrossed;
    return out;
  }
  const float r0 = rowRadius(mapper, firstRow);
  if (trajectory) {
    trajectory->resize(firstRow); // Keep diagnostic trace indexed by global TPC row.
  }
  // Preserve the reviewed K first-row seeding convention, but use the complete
  // original track rather than reconstructing a primary vertex. K field-map
  // transport from the original reference remains a separate physics-validation item.
  const auto p0 = helixAtRadius(inputTrack, r0, bzKG);
  if (!p0.valid) {
    out.status = SectorEdgeStatus::NoRowsCrossed;
    return out;
  }

  const float qpt = inputTrack.getQ2Pt();
  const float pt = 1.f / std::abs(qpt);
  const int charge = qpt >= 0.f ? 1 : -1;
  o2::track::TrackPar::dim3_t xyz{{p0.x, p0.y, p0.z}};
  o2::track::TrackPar::dim3_t pxyz{{pt * std::cos(p0.phiMom), pt * std::sin(p0.phiMom), pt * inputTrack.getTgl()}};
  o2::track::TrackPar track(xyz, pxyz, charge, true);
  o2::track::TrackLTIntegral path;
  path.clear();
  path.setTimeNotNeeded();
  float previousPathCm = 0.f;
  float previousRadiusCm = r0;

  for (int row = firstRow; row < mapper.getNumberOfRows(); ++row) {
    const float radius = rowRadius(mapper, row);
    if (row > firstRow) {
      // Record accumulated transported path length. A later outward *orbit* can have positive dr/ds
      // and lie on the requested cylinder; radialCos > 0 alone would accept that wrong branch.
      // With radial |snp| < maxSnp, a first-branch step should be short relative to its radial
      // increment. This deliberately generous guard only rejects gross discontinuities; numerical
      // H/K row-mask agreement is checked separately by the physics oracle.
      if (!prop->propagateToR(track, radius, false, o2::base::Propagator::MAX_SIN_PHI,
                              o2::base::Propagator::MAX_STEP,
                              o2::base::Propagator::MatCorrType::USEMatCorrNONE, &path)) {
        out.status = SectorEdgeStatus::PropagationFailed;
        return out;
      }
      const float deltaPathCm = path.getL() - previousPathCm;
      const float deltaRadiusCm = radius - previousRadiusCm;
      if (!plausibleFirstOutwardStep(deltaRadiusCm, deltaPathCm, track.getTgl(), cfg.maxSnp)) {
        out.status = SectorEdgeStatus::PropagationFailed;
        return out;
      }
      previousPathCm = path.getL();
      previousRadiusCm = radius;
    }

    const auto xyzg = track.getXYZGlo();
    TrajectoryPoint p;
    p.x = xyzg.X();
    p.y = xyzg.Y();
    p.z = xyzg.Z();
    p.phiPos = track.getPhiPos();
    p.phiMom = track.getPhi();
    const float dphi = p.phiMom - p.phiPos;
    const float radialCos = std::cos(dphi);
    // propagateToR() may return a state on a different branch or slightly off the target cylinder.
    // Neither may contribute a crossed row. The radial-frame snp limit is checked independently below.
    constexpr float radiusToleranceCm = 2.e-3f;
    const float actualRadius = std::hypot(p.x, p.y);
    if (!std::isfinite(actualRadius) || std::abs(actualRadius - radius) > radiusToleranceCm ||
        !std::isfinite(radialCos)) {
      out.status = SectorEdgeStatus::PropagationFailed;
      return out;
    }
    if (radialCos <= 0.f) {
      break; // reached the transverse turning point or a returning arc
    }
    p.dsDr = std::sqrt(1.f + track.getTgl() * track.getTgl()) / radialCos;
    p.valid = true;
    if (trajectory) {
      appendTrajectoryState(*trajectory, p, track.getQ2Pt(), track.getTgl());
    }
    if (reachedMaxSnp(p, cfg)) {
      break;
    }
    accumulateRow(mapper, row, p, rowLengthRadial(mapper, row) * p.dsDr, cfg, out);
  }

  if (!out.nRowsCrossed && out.status == SectorEdgeStatus::OK) {
    out.status = SectorEdgeStatus::NoRowsCrossed;
  }
  return out;
}

/** @brief Reduced-primary compatibility adapter to complete-state K row evaluation. */
inline TPCSectorEdgeResult sectorEdgeO2PropagatorRows(const TPCPrimaryTrackSeed& seed,
                                                      const TPCSectorEdgeConfig& cfg = {},
                                                      std::vector<o2::recohelpers::TrackParametersAtRadius>* trajectory = nullptr)
{
  if (!std::isfinite(seed.phi) || !std::isfinite(seed.tgl) || !std::isfinite(seed.qpt) ||
      !std::isfinite(seed.bzKG) || std::abs(seed.qpt) < 1.e-12f) {
    TPCSectorEdgeResult r;
    r.status = SectorEdgeStatus::InvalidInput;
    return r;
  }
  return sectorEdgeO2PropagatorRows(o2::recohelpers::makeTrackParFromPrimaryApproximation(
                                      seed.phi, seed.tgl, seed.qpt, 0.f, 0.f, seed.vz),
                                    seed.bzKG, cfg, trajectory);
}

/** @brief L row evaluator: stack-local linear approximation to the analytic helix.
 *
 * Rows with equal Mapper pad height are treated as one stack. Analytic H endpoints define linear interpolation in phi
 * and z versus radius. If the nominal outer endpoint is unreachable or reaches maxSnp first, the last reachable row
 * below maxSnp becomes the endpoint of a retained partial stack and the outward traversal stops after that stack.
 * Row position, momentum direction, and ds/dr are then evaluated from the local approximation before classification.
 */
inline TPCSectorEdgeResult sectorEdgeStackLinear(const o2::track::TrackPar& track, float bzKG, const TPCSectorEdgeConfig& cfg = {},
                                                 std::vector<o2::recohelpers::TrackParametersAtRadius>* trajectory = nullptr)
{
  TPCSectorEdgeResult out;
  if (trajectory) {
    trajectory->clear();
  }
  float startRadius = 0.f;
  if (!std::isfinite(bzKG) || !trackOutwardReference(track, startRadius)) {
    out.status = SectorEdgeStatus::InvalidInput;
    return out;
  }
  const auto& mapper = o2::tpc::Mapper::instance();

  int first = 0;
  while (first < mapper.getNumberOfRows()) {
    const float h = rowLengthRadial(mapper, first);
    int last = first;
    while (last + 1 < mapper.getNumberOfRows() && std::abs(rowLengthRadial(mapper, last + 1) - h) < 1.e-6f) {
      ++last;
    }

    if (rowRadius(mapper, first) + 1.e-4f < startRadius) {
      if (trajectory) {
        trajectory->emplace_back(); // Preserve global row indexing.
      }
      ++first; // Ignore rows inside the complete track's reference radius.
      continue;
    }
    const float r0 = rowRadius(mapper, first);
    const auto p0 = helixAtRadius(track, r0, bzKG);
    if (!p0.valid || reachedMaxSnp(p0, cfg)) {
      break;
    }

    // The frozen Linear contract keeps a reachable partial stack. If the nominal
    // outer endpoint is unreachable or already beyond maxSnp, walk inward to the
    // last reachable row below maxSnp and use that row as the interpolation endpoint.
    int endpoint = last;
    auto p1 = helixAtRadius(track, rowRadius(mapper, endpoint), bzKG);
    bool stopAfterStack = false;
    while (endpoint > first && (!p1.valid || reachedMaxSnp(p1, cfg))) {
      --endpoint;
      p1 = helixAtRadius(track, rowRadius(mapper, endpoint), bzKG);
      stopAfterStack = true;
    }
    if (!p1.valid || reachedMaxSnp(p1, cfg)) {
      break;
    }
    stopAfterStack = stopAfterStack || endpoint < last;

    const float r1 = rowRadius(mapper, endpoint);
    float dphi = p1.phiPos - p0.phiPos;
    o2::math_utils::detail::bringToPMPi(dphi);
    const float dr = std::max(r1 - r0, 1.e-6f);
    const float dphiDr = dphi / dr, dzDr = (p1.z - p0.z) / dr;

    for (int row = first; row <= endpoint; ++row) {
      const float r = rowRadius(mapper, row), f = (r - r0) / dr;
      const float phi = p0.phiPos + f * dphi;
      TrajectoryPoint p;
      p.x = r * std::cos(phi);
      p.y = r * std::sin(phi);
      p.z = p0.z + f * (p1.z - p0.z);
      p.phiPos = phi;
      p.phiMom = phi + std::atan(r * dphiDr);
      p.dsDr = std::sqrt(1.f + (r * dphiDr) * (r * dphiDr) + dzDr * dzDr);
      // When the limit leaves a single reachable row, finite-difference derivatives are undefined.
      // Reuse the actual helix tangent and path length at that row rather than setting the slopes to zero.
      if (endpoint == first) {
        p.phiMom = p0.phiMom;
        p.dsDr = p0.dsDr;
      }
      p.valid = true;
      if (trajectory) {
        appendTrajectoryState(*trajectory, p, track.getQ2Pt(), track.getTgl());
      }
      if (reachedMaxSnp(p, cfg)) {
        stopAfterStack = true;
        break;
      }
      accumulateRow(mapper, row, p, rowLengthRadial(mapper, row) * p.dsDr, cfg, out);
    }

    if (stopAfterStack) {
      break;
    }
    first = last + 1;
  }

  if (!out.nRowsCrossed) {
    out.status = SectorEdgeStatus::NoRowsCrossed;
  }
  return out;
}

/** @brief Reduced-primary compatibility adapter to complete-state L row evaluation. */
inline TPCSectorEdgeResult sectorEdgeStackLinear(const TPCPrimaryTrackSeed& seed, const TPCSectorEdgeConfig& cfg = {},
                                                 std::vector<o2::recohelpers::TrackParametersAtRadius>* trajectory = nullptr)
{
  if (!std::isfinite(seed.phi) || !std::isfinite(seed.tgl) || !std::isfinite(seed.qpt) ||
      !std::isfinite(seed.bzKG) || std::abs(seed.qpt) < 1.e-12f) {
    TPCSectorEdgeResult r;
    r.status = SectorEdgeStatus::InvalidInput;
    return r;
  }
  return sectorEdgeStackLinear(o2::recohelpers::makeTrackParFromPrimaryApproximation(
                                 seed.phi, seed.tgl, seed.qpt, 0.f, 0.f, seed.vz),
                               seed.bzKG, cfg, trajectory);
}

template <typename Evaluator>
inline TPCSectorEdgeBatchResult sectorEdgeBatch(const float* phi, const float* tgl, const float* qpt, const float* vz,
                                                std::size_t n, float bzKG, const TPCSectorEdgeConfig& cfg, Evaluator&& evaluate)
{
  TPCSectorEdgeBatchResult out;
  out.resize(n);

  if (!phi || !tgl || !qpt) {
    for (std::size_t i = 0; i < n; ++i) {
      TPCSectorEdgeResult r;
      r.status = SectorEdgeStatus::InvalidInput;
      out.set(i, r);
    }
    return out;
  }

  for (std::size_t i = 0; i < n; ++i) {
    TPCPrimaryTrackSeed seed;
    seed.phi = phi[i];
    seed.tgl = tgl[i];
    seed.qpt = qpt[i];
    seed.vz = vz ? vz[i] : 0.f;
    seed.bzKG = bzKG;
    out.set(i, evaluate(seed, cfg));
  }
  return out;
}

/** @brief Batch wrapper for the analytic H TPC row evaluator. */
inline TPCSectorEdgeBatchResult sectorEdgeHelixRowsBatch(const float* phi, const float* tgl, const float* qpt, const float* vz,
                                                         std::size_t n, float bzKG, const TPCSectorEdgeConfig& cfg = {})
{
  return sectorEdgeBatch(phi, tgl, qpt, vz, n, bzKG, cfg,
                         [](const TPCPrimaryTrackSeed& seed, const TPCSectorEdgeConfig& c) { return sectorEdgeHelixRows(seed, c); });
}

/** @brief Batch wrapper for the stack-linear L TPC row evaluator. */
inline TPCSectorEdgeBatchResult sectorEdgeStackLinearBatch(const float* phi, const float* tgl, const float* qpt, const float* vz,
                                                           std::size_t n, float bzKG, const TPCSectorEdgeConfig& cfg = {})
{
  return sectorEdgeBatch(phi, tgl, qpt, vz, n, bzKG, cfg,
                         [](const TPCPrimaryTrackSeed& seed, const TPCSectorEdgeConfig& c) { return sectorEdgeStackLinear(seed, c); });
}

/** @brief Batch wrapper for the O2 Propagator K TPC row evaluator, without material correction. */
inline TPCSectorEdgeBatchResult sectorEdgeO2PropagatorRowsBatch(const float* phi, const float* tgl, const float* qpt, const float* vz,
                                                                std::size_t n, float bzKG, const TPCSectorEdgeConfig& cfg = {})
{
  return sectorEdgeBatch(phi, tgl, qpt, vz, n, bzKG, cfg,
                         [](const TPCPrimaryTrackSeed& seed, const TPCSectorEdgeConfig& c) {
                           return sectorEdgeO2PropagatorRows(seed, c);
                         });
}

/** @brief Evaluate row acceptance for a batch of complete O2 local 5D track states.
 * Input arrays are x, alpha, y, z, snp, tgl, q/pT; all are mandatory.
 * `absCharge=1` is currently used by the convenience constructor, as in the generic
 * state-at-radius batch interface. For multi-charge tracks use the scalar TrackPar API.
 */
template <typename Evaluator>
inline TPCSectorEdgeBatchResult localSectorEdgeBatch(
  const float* x, const float* alpha, const float* y, const float* z,
  const float* snp, const float* tgl, const float* qpt,
  std::size_t n, float bzKG, const TPCSectorEdgeConfig& cfg, Evaluator&& evaluate)
{
  TPCSectorEdgeBatchResult out;
  out.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    if (!x || !alpha || !y || !z || !snp || !tgl || !qpt ||
        !std::isfinite(x[i]) || !std::isfinite(alpha[i]) || !std::isfinite(y[i]) ||
        !std::isfinite(z[i]) || !std::isfinite(snp[i]) || !std::isfinite(tgl[i]) ||
        !std::isfinite(qpt[i]) || std::abs(qpt[i]) < 1.e-12f || std::abs(snp[i]) >= 1.f) {
      TPCSectorEdgeResult r;
      r.status = SectorEdgeStatus::InvalidInput;
      out.set(i, r);
      continue;
    }
    const auto track = o2::recohelpers::makeTrackParFromLocalParameters(
      x[i], alpha[i], y[i], z[i], snp[i], tgl[i], qpt[i]);
    out.set(i, evaluate(track, bzKG, cfg));
  }
  return out;
}

inline TPCSectorEdgeBatchResult localSectorEdgeHelixRowsBatch(
  const float* x, const float* alpha, const float* y, const float* z,
  const float* snp, const float* tgl, const float* qpt,
  std::size_t n, float bzKG, const TPCSectorEdgeConfig& cfg = {})
{
  return localSectorEdgeBatch(x, alpha, y, z, snp, tgl, qpt, n, bzKG, cfg,
                              [](const o2::track::TrackPar& t, float bz, const TPCSectorEdgeConfig& c) {
                                return sectorEdgeHelixRows(t, bz, c);
                              });
}

inline TPCSectorEdgeBatchResult localSectorEdgeStackLinearBatch(
  const float* x, const float* alpha, const float* y, const float* z,
  const float* snp, const float* tgl, const float* qpt,
  std::size_t n, float bzKG, const TPCSectorEdgeConfig& cfg = {})
{
  return localSectorEdgeBatch(x, alpha, y, z, snp, tgl, qpt, n, bzKG, cfg,
                              [](const o2::track::TrackPar& t, float bz, const TPCSectorEdgeConfig& c) {
                                return sectorEdgeStackLinear(t, bz, c);
                              });
}

inline TPCSectorEdgeBatchResult localSectorEdgeO2PropagatorRowsBatch(
  const float* x, const float* alpha, const float* y, const float* z,
  const float* snp, const float* tgl, const float* qpt,
  std::size_t n, float bzKG, const TPCSectorEdgeConfig& cfg = {})
{
  return localSectorEdgeBatch(x, alpha, y, z, snp, tgl, qpt, n, bzKG, cfg,
                              [](const o2::track::TrackPar& t, float bz, const TPCSectorEdgeConfig& c) {
                                return sectorEdgeO2PropagatorRows(t, bz, c);
                              });
}

///@}

} // namespace o2::tpc::recohelpers

#endif // ALICEO2_TRACKTOOLS_TRACKGEOMETRY_H_
