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

/// \file MathiesonOriginal.cxx
/// \brief Original implementation of the Mathieson function
///
/// \author Philippe Pillot, Subatech

#include "MCHBase/MathiesonOriginal.h"

//#define DEBUG_LUT 1

#ifdef DEBUG_LUT
#include <iostream>
#include <fmt/format.h>
#endif

#include <TMath.h>

namespace o2
{
namespace mch
{

//_________________________________________________________________________________________________
MathiesonOriginal::LUT::LUT(int size, double min, double max)
{
  init(size, min, max);
}

//_________________________________________________________________________________________________
void MathiesonOriginal::LUT::init(int size, double min, double max)
{
  mSize = size;
  mMin = min;
  mMax = max;

  if (mSize < 1) {
    mTable = nullptr;
    mStep = 0;
    mInverseStep = 0;
    mInverseWidth = 0;
    return;
  }

  if (mSize > 1 && mMax > mMin) {
    mStep = (mMax - mMin) / (mSize - 0);
    mInverseStep = 1. / mStep;
    mInverseWidth = 1. / (mMax - mMin);
  }

  //mTable.resize(mSize + 1);
  //std::fill(mTable.begin(), mTable.end(), std::vector<double>(mSize + 1));

  mTable = new double*[mSize + 1];
  mTable[0] = new double[(mSize + 1) * (mSize + 1)];
  for (int i = 1; i < (mSize + 1); i++) {
    mTable[i] = &(mTable[0][(mSize + 1) * i]);
  }
}

//_________________________________________________________________________________________________
MathiesonOriginal::LUT::~LUT()
{
  if (mTable) {
    if (mTable[0]) {
      delete mTable[0];
    }
    delete mTable;
  }
}

//_________________________________________________________________________________________________
bool MathiesonOriginal::LUT::get(double x, double y, double& val) const
{
  // bilinear interpolation (https://en.wikipedia.org/wiki/Bilinear_interpolation#On_the_unit_square)
  if (mSize < 1) {
    return false;
  }

  double xn = (x - mMin) * mInverseStep;
  int x0 = int(xn);
  int x1 = x0 + 1;
  double xr = xn - x0;

  double yn = (y - mMin) * mInverseStep;
  int y0 = int(yn);
  int y1 = y0 + 1;
  double yr = yn - y0;

  double f00 = mTable[y0][x0];
  double f10 = mTable[y0][x1];
  double f01 = mTable[y1][x0];
  double f11 = mTable[y1][x1];

  val = f00 * (1.0 - xr) * (1.0 - yr) + f10 * xr * (1.0 - yr) + f01 * (1.0 - xr) * yr + f11 * xr * yr;

#ifdef DEBUG_LUT
  std::cout << fmt::format("LUT::get({}, {}): \n", x, y)
            << fmt::format("  xn={}  yn={}\n", xn, yn)
            << fmt::format("  x0={}  y0={}\n", x0, y0)
            << fmt::format("  xr={}  yr={}\n", xr, yr)
            << fmt::format("  f00={}  f10={}  f01={}  f11={}\n", f00, f10, f01, f11)
            << fmt::format("  val={}", val) << std::endl;
#endif

  return true;
}

//_________________________________________________________________________________________________
void MathiesonOriginal::setSqrtKx3AndDeriveKx2Kx4(float sqrtKx3)
{
  /// set the Mathieson parameter sqrt(K3) in x direction, perpendicular to the wires,
  /// and derive the Mathieson parameters K2 and K4 in the same direction
  mSqrtKx3 = sqrtKx3;
  mKx2 = TMath::Pi() / 2. * (1. - 0.5 * mSqrtKx3);
  float cx1 = mKx2 * mSqrtKx3 / 4. / TMath::ATan(static_cast<double>(mSqrtKx3));
  mKx4 = cx1 / mKx2 / mSqrtKx3;
}

//_________________________________________________________________________________________________
void MathiesonOriginal::setSqrtKy3AndDeriveKy2Ky4(float sqrtKy3)
{
  /// set the Mathieson parameter sqrt(K3) in y direction, along the wires,
  /// and derive the Mathieson parameters K2 and K4 in the same direction
  mSqrtKy3 = sqrtKy3;
  mKy2 = TMath::Pi() / 2. * (1. - 0.5 * mSqrtKy3);
  float cy1 = mKy2 * mSqrtKy3 / 4. / TMath::ATan(static_cast<double>(mSqrtKy3));
  mKy4 = cy1 / mKy2 / mSqrtKy3;
}

//_________________________________________________________________________________________________
void MathiesonOriginal::init()
{
  if (!mFastIntegral) {
    return;
  }

  mLUT.init(2000, -20, 20);

  for (int j = 0; j < mLUT.mSize; j++) {
    float y = mLUT.getY(j);
    for (int i = 0; i < mLUT.mSize; i++) {
      float x = mLUT.getX(i);
      float integral = integrateAnalytic(mLUT.mMin, mLUT.mMin, x, y);
      mLUT.set(i, j, integral);
#ifdef DEBUG_LUT
      std::cout << "LUT[" << i << "," << j << "] -> (" << x << "," << y << ") = " << integral << std::endl;
#endif
    }
  }
}

//_________________________________________________________________________________________________
float MathiesonOriginal::integrateLUT(float xMin, float yMin, float xMax, float yMax) const
{
  /// integrate the Mathieson over x and y in the given area
  double i00{ 0 };
  double i10{ 0 };
  double i01{ 0 };
  double i11{ 0 };
  mLUT.get(xMin, yMin, i00);
  mLUT.get(xMax, yMin, i10);
  mLUT.get(xMin, yMax, i01);
  mLUT.get(xMax, yMax, i11);

  return static_cast<float>(i11 - i10 - i01 + i00);
}

//_________________________________________________________________________________________________
float MathiesonOriginal::integrateAnalytic(float xMin, float yMin, float xMax, float yMax) const
{
  /// integrate the Mathieson over x and y in the given area

  //
  // The Mathieson function
  double uxMin = mSqrtKx3 * TMath::TanH(mKx2 * xMin);
  double uxMax = mSqrtKx3 * TMath::TanH(mKx2 * xMax);

  double uyMin = mSqrtKy3 * TMath::TanH(mKy2 * yMin);
  double uyMax = mSqrtKy3 * TMath::TanH(mKy2 * yMax);

  return static_cast<float>(4. * mKx4 * (TMath::ATan(uxMax) - TMath::ATan(uxMin)) *
                            mKy4 * (TMath::ATan(uyMax) - TMath::ATan(uyMin)));
}

//_________________________________________________________________________________________________
float MathiesonOriginal::integrate(float xMin, float yMin, float xMax, float yMax) const
{
  /// integrate the Mathieson over x and y in the given area

  xMin *= mInversePitch;
  xMax *= mInversePitch;
  yMin *= mInversePitch;
  yMax *= mInversePitch;


  float integral{ 0 };
  if (mFastIntegral && mLUT.isIncluded(xMin, yMin) && mLUT.isIncluded(xMax, yMax)) {
    integral = integrateLUT(xMin, yMin, xMax, yMax);
#ifdef DEBUG_LUT
    float int2 = integrateAnalytic(xMin, yMin, xMax, yMax);
    std::cout << fmt::format("({}, {}) -> ({}, {}): analytic={} lut={}", xMin, yMin, xMax, yMax, int2, integral) << std::endl;
#endif
  } else {
    integral = integrateAnalytic(xMin, yMin, xMax, yMax);
  }

  return integral;
}

} // namespace mch
} // namespace o2
