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
struct TrackAuxPar : public o2::utils::CircleXY {
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
};

//__________________________________________________________
//< crossing coordinates of 2 circles
struct CircleCrossInfo {
  float xDCA[2];
  float yDCA[2];
  int nDCA;

  CircleCrossInfo() = default;

  CircleCrossInfo(const TrackAuxPar& trc0, const TrackAuxPar& trc1) { set(trc0, trc1); }
  int set(const TrackAuxPar& trc0, const TrackAuxPar& trc1)
  {
    // calculate up to 2 crossings between 2 circles
    nDCA = 0;
    const auto& trcA = trc0.rC > trc1.rC ? trc0 : trc1; // designate the largest circle as A
    const auto& trcB = trc0.rC > trc1.rC ? trc1 : trc0;
    float xDist = trcB.xC - trcA.xC, yDist = trcB.yC - trcA.yC;
    float dist2 = xDist * xDist + yDist * yDist, dist = std::sqrt(dist2), rsum = trcA.rC + trcB.rC;
    if (std::abs(dist) < 1e-12) {
      return nDCA; // circles are concentric?
    }
    if (dist > rsum) { // circles don't touch, chose a point in between
      // the parametric equation of lines connecting the centers is
      // x = x0 + t/dist * (x1-x0), y = y0 + t/dist * (y1-y0)
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
};

} // namespace track
} // namespace o2

#endif
