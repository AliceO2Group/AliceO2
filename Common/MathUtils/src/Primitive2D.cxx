// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Primitive.cxx
/// \brief Implementation of 2D primitives: straight line (XY interval) and circle
/// \author ruben.shahoyan@cern.ch

#include "MathUtils/Primitive2D.h"
#include <cmath>

using namespace o2::utils;

//_____________________________________________________________________
bool IntervalXY::seenByCircle(const CircleXY& circle, float eps) const
{
  // check if XY interval is seen by the circle.
  // The tolerance parameter eps is interpreted as a fraction of the interval
  // lenght to be added to the edges along the interval (i.e. while the points
  // within the interval are spanning
  // x = xc + dx*t
  // y = yc + dy*t
  // with 0<t<1., we acctually check the interval for -eps<t<1+eps
  auto dx0 = xP - circle.xC, dy0 = yP - circle.yC, dx1 = dx0 + dxP, dy1 = dy0 + dyP;
  if (eps > 0.) {
    auto dxeps = eps * dxP, dyeps = eps * dyP;
    dx0 -= dxeps;
    dx1 += dxeps;
    dy0 -= dyeps;
    dy1 += dyeps;
  }
  auto d02 = dx0 * dx0 + dy0 * dy0, d12 = dx1 * dx1 + dy1 * dy1, rC2 = circle.rC * circle.rC; // distance^2 from circle center to edges
  return (d02 - rC2) * (d12 - rC2) < 0;
}

//_____________________________________________________________________
bool IntervalXY::seenByLine(const IntervalXY& other, float eps) const
{
  // check if XY interval is seen by the line defined by other interval
  // The tolerance parameter eps is interpreted as a fraction of the interval
  // lenght to be added to the edges along the interval (i.e. while the points
  // within the interval are spanning
  // x = xc + dx*t
  // y = yc + dy*t
  // with 0<t<1., we acctually check the interval for -eps<t<1+eps
  float a, b, c, x0, y0, x1, y1; // find equation of the line a*x+b*y+c = 0
  other.getLineCoefs(a, b, c);
  eval(-eps, x0, y0);
  eval(1.f + eps, x1, y1);
  return (a * x0 + b * y0 + c) * (a * x1 + b * y1 + c) < 0;
}

//_____________________________________________________________________
bool IntervalXY::lineCrossParam(const IntervalXY& other, float& t) const
{
  // get crossing parameter of 2 intervals
  float dx = other.xP - xP, dy = other.yP - yP;
  float det = -dxP * other.dyP + dyP * other.dxP;
  if (fabs(det) < 1.e-9) {
    return false; // parallel
  }
  t = (-dx * other.dyP + dy * other.dxP) / det;
  return true;
}

//_____________________________________________________________________
bool IntervalXY::circleCrossParam(const CircleXY& circle, float& t) const
{
  // get crossing point of circle with the interval
  // solution of the system (wrt t)
  // (x-xC)^2+(y-yC)^2 = rC2
  // x = xEdge + xProj * t;
  // y = yEdge + yProj * t;
  float dx = xP - circle.xC, dy = yP - circle.yC, del2I = 1. / (dxP * dxP + dyP * dyP),
        b = (dx * dxP + dy * dyP) * del2I, c = del2I * (dx * dx + dy * dy - circle.rC * circle.rC);
  float det = b * b - c;
  if (det < 0.) {
    return false;
  }
  det = sqrtf(det);
  float t0 = -b - det, t1 = -b + det;
  // select the one closer to [0:1] interval
  t = (fabs(t0 - 0.5) < fabs(t1 - 0.5)) ? t0 : t1;
  return true;
}

//_____________________________________________________________________
void IntervalXY::getLineCoefs(float& a, float& b, float& c) const
{
  // convert to line parameters in canonical form: a*x+b*y+c = 0
  c = xP * dyP - yP * dxP;
  if (c) {
    a = -dyP;
    b = dxP;
  } else if (fabs(dxP) > fabs(dyP)) {
    a = 0.;
    b = -1.;
    c = yP;
  } else {
    a = -1.;
    b = 0.;
    c = xP;
  }
}
