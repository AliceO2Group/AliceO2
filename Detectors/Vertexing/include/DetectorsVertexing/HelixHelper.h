// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file HelixHelper.h
/// \brief Helper classes for helical tracks manipulations
/// \author ruben.shahoyan@cern.ch

#ifndef _ALICEO2_HELIX_HELPER_
#define _ALICEO2_HELIX_HELPER_

#include "MathUtils/Primitive2D.h"
#include "ReconstructionDataFormats/Track.h"

namespace o2
{
namespace track
{

///__________________________________________________________________________
//< precalculated track radius, center, alpha sin,cos and their combinations
struct TrackAuxPar : public o2::math_utils::CircleXYf_t {
  using Track = o2::track::TrackPar;

  float c, s, cc, ss, cs; // cos ans sin of track alpha and their products

  TrackAuxPar() = default;
  TrackAuxPar(const Track& trc, float bz) { set(trc, bz); }
  float cosDif(const TrackAuxPar& t) const { return c * t.c + s * t.s; } // cos(alpha_this - alha_t)
  float sinDif(const TrackAuxPar& t) const { return s * t.c - c * t.s; } // sin(alpha_this - alha_t)
  void set(const Track& trc, float bz)
  {
    trc.getCircleParams(bz, *this, s, c);
    cc = c * c;
    ss = s * s;
    cs = c * s;
  }
  ClassDefNV(TrackAuxPar, 1);
};

//__________________________________________________________
//< crossing coordinates of 2 circles
struct CrossInfo {
  static constexpr float MaxDistXYDef = 10.;
  float xDCA[2] = {};
  float yDCA[2] = {};
  int nDCA = 0;

  int circlesCrossInfo(const TrackAuxPar& trax0, const TrackAuxPar& trax1, float maxDistXY = MaxDistXYDef)
  {
    const auto& trcA = trax0.rC > trax1.rC ? trax0 : trax1; // designate the largest circle as A
    const auto& trcB = trax0.rC > trax1.rC ? trax1 : trax0;
    float xDist = trcB.xC - trcA.xC, yDist = trcB.yC - trcA.yC;
    float dist2 = xDist * xDist + yDist * yDist, dist = std::sqrt(dist2), rsum = trcA.rC + trcB.rC;
    if (std::abs(dist) < 1e-12) {
      return nDCA; // circles are concentric?
    }
    if (dist > rsum) { // circles don't touch, chose a point in between
      // the parametric equation of lines connecting the centers is
      // x = x0 + t/dist * (x1-x0), y = y0 + t/dist * (y1-y0)
      if (dist - rsum > maxDistXY) { // too large distance
        return nDCA;
      }
      notTouchingXY(dist, xDist, yDist, trcA, trcB.rC);
    } else if (dist + trcB.rC < trcA.rC) { // the small circle is nestled into large one w/o touching
      // select the point of closest approach of 2 circles
      notTouchingXY(dist, xDist, yDist, trcA, -trcB.rC);
    } else { // 2 intersection points
      // to simplify calculations, we move to new frame x->x+Xc0, y->y+Yc0, so that
      // the 1st one is centered in origin
      if (std::abs(xDist) < std::abs(yDist)) {
        float a = (trcA.rC * trcA.rC - trcB.rC * trcB.rC + dist2) / (2. * yDist), b = -xDist / yDist, ab = a * b, bb = b * b;
        float det = ab * ab - (1. + bb) * (a * a - trcA.rC * trcA.rC);
        if (det > 0.) {
          det = std::sqrt(det);
          xDCA[0] = (-ab + det) / (1. + b * b);
          yDCA[0] = a + b * xDCA[0] + trcA.yC;
          xDCA[0] += trcA.xC;
          xDCA[1] = (-ab - det) / (1. + b * b);
          yDCA[1] = a + b * xDCA[1] + trcA.yC;
          xDCA[1] += trcA.xC;
          nDCA = 2;
        } else { // due to the finite precision the det<=0, i.e. the circles are barely touching, fall back to this special case
          notTouchingXY(dist, xDist, yDist, trcA, trcB.rC);
        }
      } else {
        float a = (trcA.rC * trcA.rC - trcB.rC * trcB.rC + dist2) / (2. * xDist), b = -yDist / xDist, ab = a * b, bb = b * b;
        float det = ab * ab - (1. + bb) * (a * a - trcA.rC * trcA.rC);
        if (det > 0.) {
          det = std::sqrt(det);
          yDCA[0] = (-ab + det) / (1. + bb);
          xDCA[0] = a + b * yDCA[0] + trcA.xC;
          yDCA[0] += trcA.yC;
          yDCA[1] = (-ab - det) / (1. + bb);
          xDCA[1] = a + b * yDCA[1] + trcA.xC;
          yDCA[1] += trcA.yC;
          nDCA = 2;
        } else { // due to the finite precision the det<=0, i.e. the circles are barely touching, fall back to this special case
          notTouchingXY(dist, xDist, yDist, trcA, trcB.rC);
        }
      }
    }
    return nDCA;
  }

  void notTouchingXY(float dist, float xDist, float yDist, const TrackAuxPar& trcA, float rBSign)
  {
    // fast method to calculate DCA between 2 circles, assuming that they don't touch each outer:
    // the parametric equation of lines connecting the centers is x = xA + t/dist * xDist, y = yA + t/dist * yDist
    // with xA,yY being the center of the circle A ( = trcA.xC, trcA.yC ), xDist = trcB.xC = trcA.xC ...
    // There are 2 special cases:
    // (a) small circle is inside the large one: provide rBSign as -trcB.rC
    // (b) circle are side by side: provide rBSign as trcB.rC
    nDCA = 1;
    auto t2d = (dist + trcA.rC - rBSign) / dist;
    xDCA[0] = trcA.xC + 0.5 * (xDist * t2d);
    yDCA[0] = trcA.yC + 0.5 * (yDist * t2d);
  }

  int linesCrossInfo(const TrackAuxPar& trax0, const TrackPar& tr0,
                     const TrackAuxPar& trax1, const TrackPar& tr1, float maxDistXY = MaxDistXYDef)
  {
    /// closest approach of 2 straight lines
    ///  TrackParam propagation can be parameterized in lab in a form
    ///  xLab(t) = (x*cosAlp - y*sinAlp) + t*(cosAlp - sinAlp* snp/csp) = xLab0 + t*(cosAlp - sinAlp* snp/csp)
    ///  yLab(t) = (x*sinAlp + y*cosAlp) + t*(sinAlp + cosAlp* snp/csp) = yLab0 + t*(sinAlp + cosAlp* snp/csp)
    ///  zLab(t) = z + t * tgl / csp = zLab0 + t * tgl / csp
    ///  where t is the x-step in the track alpha-frame, xLab,yLab,zLab are reference track coordinates in lab
    ///  frame (filled by TrackAuxPar for straight line tracks).
    ///
    ///  Therefore, for the parametric track equation in lab 3D we have (wrt tracking-X increment t)
    ///  xL(t) = xL + t Kx;  Kx = (cosAlp - sinAlp* snp/csp)
    ///  yL(t) = yL + t Ky;  Ky = (sinAlp + cosAlp* snp/csp)
    ///  zL(t) = zL + t Kz;  Kz = tgl / csp
    ///  Note that Kx^2 + Ky^2 + Kz^2 = (1+tgl^2) / csp^2

    float dx = trax1.xC - trax0.xC; // for straight line TrackAuxPar stores lab coordinates at referene point!!!
    float dy = trax1.yC - trax0.yC; //
    float dz = tr1.getZ() - tr0.getZ();
    auto csp0i2 = 1. / tr0.getCsp2(); // 1 / csp^2
    auto csp0i = std::sqrt(csp0i2);
    auto tgp0 = tr0.getSnp() * csp0i;
    float kx0 = trax0.c - trax0.s * tgp0;
    float ky0 = trax0.s + trax0.c * tgp0;
    float kz0 = tr0.getTgl() * csp0i;
    auto csp1i2 = 1. / tr1.getCsp2(); // 1 / csp^2
    auto csp1i = std::sqrt(csp1i2);
    auto tgp1 = tr1.getSnp() * std::sqrt(csp1i2);
    float kx1 = trax1.c - trax1.s * tgp1;
    float ky1 = trax1.s + trax1.c * tgp1;
    float kz1 = tr1.getTgl() * csp1i;
    /// Minimize |vecL1 - vecL0|^2 wrt t0 and t1: point of closest approach
    /// Leads to system
    /// A Dx = B with Dx = {dx0, dx1}
    /// with A =
    ///  |      kx0^2+ky0^2+kz0^2     -(kx0*kx1+ky0*ky1+kz0*kz1) | =  (1+tgl0^2) / csp0^2           ....
    ///  | -(kx0*kx1+ky0*ky1+kz0*kz1)     kx0^2+ky0^2+kz0^2      |     .....                   (1+tgl1^2) / csp1^2
    /// and B = {(dx Kx0 + dy Ky0 + dz Kz0), -(dx Kx1 + dy Ky1 + dz Kz1) }
    ///
    float a00 = (1.f + tr0.getTgl() * tr0.getTgl()) * csp0i2, a11 = (1.f + tr1.getTgl() * tr1.getTgl()) * csp1i2, a01 = -(kx0 * kx1 + ky0 * ky1 + kz0 * kz1);
    float b0 = dx * kx0 + dy * ky0 + dz * kz0, b1 = -(dx * kx1 + dy * ky1 + dz * kz1);
    float det = a00 * a11 - a01 * a01, det0 = b0 * a11 - b1 * a01, det1 = a00 * b1 - a01 * b0;
    if (std::abs(det) > o2::constants::math::Almost0) {
      auto detI = 1. / det;
      auto t0 = det0 * detI;
      auto t1 = det1 * detI;
      float addx0 = kx0 * t0, addy0 = ky0 * t0, addx1 = kx1 * t1, addy1 = ky1 * t1;
      dx += addx1 - addx0; // recalculate XY distance at DCA
      dy += addy1 - addy0;
      if (dx * dx + dy * dy > maxDistXY * maxDistXY) {
        return nDCA;
      }
      xDCA[0] = (trax0.xC + addx0 + trax1.xC + addx1) * 0.5;
      yDCA[0] = (trax0.yC + addy0 + trax1.yC + addy1) * 0.5;
      nDCA = 1;
    }
    return nDCA;
  }

  int circleLineCrossInfo(const TrackAuxPar& trax0, const TrackPar& tr0,
                          const TrackAuxPar& trax1, const TrackPar& tr1, float maxDistXY = MaxDistXYDef)
  {
    /// closest approach of line and circle
    ///  TrackParam propagation can be parameterized in lab in a form
    ///  xLab(t) = (x*cosAlp - y*sinAlp) + t*(cosAlp - sinAlp* snp/csp) = xLab0 + t*(cosAlp - sinAlp* snp/csp)
    ///  yLab(t) = (x*sinAlp + y*cosAlp) + t*(sinAlp + cosAlp* snp/csp) = yLab0 + t*(sinAlp + cosAlp* snp/csp)
    ///  zLab(t) = z + t * tgl / csp = zLab0 + t * tgl / csp
    ///  where t is the x-step in the track alpha-frame, xLab,yLab,zLab are reference track coordinates in lab
    ///  frame (filled by TrackAuxPar for straight line tracks).
    ///
    ///  Therefore, for the parametric track equation in lab 3D we have (wrt tracking-X increment t)
    ///  xL(t) = xL + t Kx;  Kx = (cosAlp - sinAlp* snp/csp)
    ///  yL(t) = yL + t Ky;  Ky = (sinAlp + cosAlp* snp/csp)
    ///  zL(t) = zL + t Kz;  Kz = tgl / csp
    ///  Note that Kx^2 + Ky^2  = 1 / csp^2

    const auto& traxH = trax0.rC > trax1.rC ? trax0 : trax1; // circle (for the line rC is set to 0)
    const auto& traxL = trax0.rC > trax1.rC ? trax1 : trax0; // line
    const auto& trcL = trax0.rC > trax1.rC ? tr1 : tr0;      // track of the line

    // solve quadratic equation of line crossing the circle
    float dx = traxL.xC - traxH.xC; // X distance between the line lab reference and circle center
    float dy = traxL.yC - traxH.yC; // Y...
    // t^2(kx^2+ky^2) + 2t(dx*kx+dy*ky) + dx^2 + dy^2 - r^2 = 0
    auto cspi2 = 1. / trcL.getCsp2(); // 1 / csp^2 == kx^2 +  ky^2
    auto cspi = std::sqrt(cspi2);
    auto tgp = trcL.getSnp() * cspi;
    float kx = traxL.c - traxL.s * tgp;
    float ky = traxL.s + traxL.c * tgp;
    double dk = dx * kx + dy * ky;
    double det = dk * dk - cspi2 * (dx * dx + dy * dy - traxH.rC * traxH.rC);
    if (det > 0) { // 2 crossings
      det = std::sqrt(det);
      float t0 = (-dk + det) * cspi2;
      float t1 = (-dk - det) * cspi2;
      xDCA[0] = traxL.xC + kx * t0;
      yDCA[0] = traxL.yC + ky * t0;
      xDCA[1] = traxL.xC + kx * t1;
      yDCA[1] = traxL.yC + ky * t1;
      nDCA = 2;
    } else {
      // there is no crossing, find the point of the closest approach on the line which is closest to the circle center
      float t = -dk * cspi2;
      float xL = traxL.xC + kx * t, yL = traxL.yC + ky * t;                                               // point on the line, need to average with point on the circle
      float dxc = xL - traxH.xC, dyc = yL - traxH.yC, dist = std::sqrt(dxc * dxc + dyc * dyc);
      if (dist - traxH.rC > maxDistXY) { // too large distance
        return nDCA;
      }
      float drcf = traxH.rC / dist; // radius / distance to circle center
      float xH = traxH.xC + dxc * drcf, yH = traxH.yC + dyc * drcf;
      xDCA[0] = (xL + xH) * 0.5;
      yDCA[0] = (yL + yH) * 0.5;
      nDCA = 1;
    }
    return nDCA;
  }

  int set(const TrackAuxPar& trax0, const TrackPar& tr0, const TrackAuxPar& trax1, const TrackPar& tr1, float maxDistXY = MaxDistXYDef)
  {
    // calculate up to 2 crossings between 2 circles
    nDCA = 0;
    if (trax0.rC > o2::constants::math::Almost0 && trax1.rC > o2::constants::math::Almost0) { // both are not straight lines
      nDCA = circlesCrossInfo(trax0, trax1, maxDistXY);
    } else if (trax0.rC < o2::constants::math::Almost0 && trax1.rC < o2::constants::math::Almost0) { // both are straigt lines
      nDCA = linesCrossInfo(trax0, tr0, trax1, tr1, maxDistXY);
    } else {
      nDCA = circleLineCrossInfo(trax0, tr0, trax1, tr1, maxDistXY);
    }
    //
    return nDCA;
  }

  CrossInfo() = default;

  CrossInfo(const TrackAuxPar& trax0, const TrackPar& tr0, const TrackAuxPar& trax1, const TrackPar& tr1, float maxDistXY = MaxDistXYDef)
  {
    set(trax0, tr0, trax1, tr1, maxDistXY);
  }
  ClassDefNV(CrossInfo, 1);
};

} // namespace track
} // namespace o2

#endif
