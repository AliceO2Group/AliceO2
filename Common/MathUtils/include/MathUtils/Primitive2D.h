// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Primitive2D.h
/// \brief Declarations of 2D primitives: straight line (XY interval) and circle
/// \author ruben.shahoyan@cern.ch

#ifndef ALICEO2_COMMON_MATH_PRIMITIVE2D_H
#define ALICEO2_COMMON_MATH_PRIMITIVE2D_H

#include "GPUCommonRtypes.h"

namespace o2
{
namespace utils
{

struct CircleXY {
  float rC, xC, yC; // circle radius, x-center, y-center
  CircleXY(float r = 0., float x = 0., float y = 0.) : rC(r), xC(x), yC(y) {}
  float getCenterD2() const { return xC * xC + yC * yC; }
  ClassDefNV(CircleXY, 1);
};

struct IntervalXY {
  ///< 2D interval in lab frame defined by its one edge and signed projection lengths on X,Y axes
  float xP, yP;   ///< one of edges
  float dxP, dyP; ///< other edge minus provided
  IntervalXY(float x = 0, float y = 0, float dx = 0, float dy = 0) : xP(x), yP(y), dxP(dx), dyP(dy) {}
  float getX0() const { return xP; }
  float getY0() const { return yP; }
  float getX1() const { return xP + dxP; }
  float getY1() const { return yP + dyP; }
  float getDX() const { return dxP; }
  float getDY() const { return dyP; }
  void eval(float t, float& x, float& y) const
  {
    x = xP + t * dxP;
    y = yP + t * dyP;
  }
  void getLineCoefs(float& a, float& b, float& c) const;

  void setEdges(float x0, float y0, float x1, float y1)
  {
    xP = x0;
    yP = y0;
    dxP = x1 - x0;
    dyP = y1 - y0;
  }
  bool seenByCircle(const CircleXY& circle, float eps) const;
  bool circleCrossParam(const CircleXY& cicle, float& t) const;

  bool seenByLine(const IntervalXY& other, float eps) const;
  bool lineCrossParam(const IntervalXY& other, float& t) const;

  ClassDefNV(IntervalXY, 1);
};

} // namespace utils
} // namespace o2

#endif
