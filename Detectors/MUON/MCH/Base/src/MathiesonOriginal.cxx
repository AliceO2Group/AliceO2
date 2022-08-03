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

//#ifdef DEBUG_LUT
#include <iostream>
#include <fmt/format.h>
//#endif

#include <emmintrin.h>
#include <immintrin.h>

#include <TMath.h>

namespace o2
{
namespace mch
{

//_________________________________________________________________________________________________
MathiesonOriginal::LUT::LUT(int size, float min, float max)
{
  init(size, min, max);
}

//_________________________________________________________________________________________________
void MathiesonOriginal::LUT::init(int size, float min, float max)
{
  mSize = size;
  mLUTSize = mSize + 2;
  mMin = min;
  mMax = max;

  if (size < 1) {
    mTable = nullptr;
    mStep = 0;
    mInverseStep = 0;
    mInverseWidth = 0;
    return;
  }

  if (mSize > 1 && mMax > mMin) {
    mStep = (mMax - mMin) / mSize;
    mInverseStep = 1. / mStep;
    mInverseWidth = 1. / (mMax - mMin);
  }

  //mTable.resize(mSize + 1);
  //std::fill(mTable.begin(), mTable.end(), std::vector<double>(mSize + 1));

  mTable = new double*[mLUTSize];
  mTable[0] = new double[mLUTSize * mLUTSize];
  for (int i = 1; i < mLUTSize; i++) {
    mTable[i] = &(mTable[0][mLUTSize * i]);
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
bool MathiesonOriginal::LUT::get(float x, float y, double& val) const
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

#ifdef DEBUG_LUT
  std::cout << fmt::format("LUT::get({}, {}): \n", x, y)
            << fmt::format("  xn={}  yn={}\n", xn, yn)
            << fmt::format("  x0={}  y0={}\n", x0, y0)
            << fmt::format("  xr={}  yr={}\n", xr, yr)
            << std::endl;
#endif

  double f00 = mTable[y0][x0];
  double f10 = mTable[y0][x1];
  double f01 = mTable[y1][x0];
  double f11 = mTable[y1][x1];

  val = f00 * (1.0 - xr) * (1.0 - yr) + f10 * xr * (1.0 - yr) + f01 * (1.0 - xr) * yr + f11 * xr * yr;

#ifdef DEBUG_LUT
  std::cout << fmt::format("  f00={}  f10={}  f01={}  f11={}\n", f00, f10, f01, f11)
            << fmt::format("  val={}", val) << std::endl;
#endif

  return true;
}

//_________________________________________________________________________________________________
bool MathiesonOriginal::LUT::get4points(float xMin, float yMin, float xMax, float yMax, std::array<double, 4>& val) const
{
  // bilinear interpolation (https://en.wikipedia.org/wiki/Bilinear_interpolation#On_the_unit_square)
  if (mSize < 1) {
    return false;
  }

  double minNorm = mMin * mInverseStep;

  double xyn[4];
  xyn[0] = xMin * mInverseStep - minNorm;
  xyn[1] = xMax * mInverseStep - minNorm;
  xyn[2] = yMin * mInverseStep - minNorm;
  xyn[3] = yMax * mInverseStep - minNorm;

  int xy0[4];
  xy0[0] = int(xyn[0]);
  xy0[1] = int(xyn[1]);
  xy0[2] = int(xyn[2]);
  xy0[3] = int(xyn[3]);

  int xy1[4];
  xy1[0] = xy0[0] + 1;
  xy1[1] = xy0[1] + 1;
  xy1[2] = xy0[2] + 1;
  xy1[3] = xy0[3] + 1;

  double xyr[4];
  xyr[0] = xyn[0] - xy0[0];
  xyr[1] = xyn[1] - xy0[1];
  xyr[2] = xyn[2] - xy0[2];
  xyr[3] = xyn[3] - xy0[3];

  double oneminusxyr[4];
  oneminusxyr[0] = 1.0d - xyr[0];
  oneminusxyr[1] = 1.0d - xyr[1];
  oneminusxyr[2] = 1.0d - xyr[2];
  oneminusxyr[3] = 1.0d - xyr[3];

  // 2D grid values at the four corners around the interpolation points:
  // 0: (xMin, yMin)
  // 1: (xMax, yMin)
  // 2: (xMin, yMax)
  // 3: (xMax, yMax)

  //double f00 = mTable[y0][x0];
  double f00[4];
  //double f10 = mTable[y0][x1];
  double f10[4];
  //double f01 = mTable[y1][x0];
  double f01[4];

  //double f11 = mTable[y1][x1];
  double f11[4];

  f00[0] = mTable[xy0[2]][xy0[0]]; // (xMin, yMin)
  f00[1] = mTable[xy0[2]][xy0[1]]; // (xMax, yMin)

  f10[0] = mTable[xy0[2]][xy1[0]]; // (xMin, yMin)
  f10[1] = mTable[xy0[2]][xy1[1]]; // (xMax, yMin)

  f00[2] = mTable[xy0[3]][xy0[0]]; // (xMin, yMax)
  f00[3] = mTable[xy0[3]][xy0[1]]; // (xMin, yMax)

  f10[2] = mTable[xy0[3]][xy1[0]]; // (xMin, yMax)
  f10[3] = mTable[xy0[3]][xy1[1]]; // (xMin, yMax)

  f01[0] = mTable[xy1[2]][xy0[0]]; // (xMin, yMin)
  f01[1] = mTable[xy1[2]][xy0[1]]; // (xMax, yMin)

  f11[0] = mTable[xy1[2]][xy1[0]]; // (xMin, yMin)
  f11[1] = mTable[xy1[2]][xy1[1]]; // (xMax, yMin)

  f01[2] = mTable[xy1[3]][xy0[0]]; // (xMin, yMax)
  f01[3] = mTable[xy1[3]][xy0[1]]; // (xMin, yMax)

  f11[2] = mTable[xy1[3]][xy1[0]]; // (xMin, yMax)
  f11[3] = mTable[xy1[3]][xy1[1]]; // (xMin, yMax)

  //val = f00 * (1.0 - xr) * (1.0 - yr) + f10 * xr * (1.0 - yr) + f01 * (1.0 - xr) * yr + f11 * xr * yr;
  val[0] = f00[0] * oneminusxyr[0] * oneminusxyr[2] +
           f10[0] * xyr[0] * oneminusxyr[2] +
           f01[0] * oneminusxyr[0] * xyr[2] +
           f11[0] * xyr[0] * xyr[2];
  val[1] = f00[1] * oneminusxyr[1] * oneminusxyr[2] +
           f10[1] * xyr[1] * oneminusxyr[2] +
           f01[1] * oneminusxyr[1] * xyr[2] +
           f11[1] * xyr[1] * xyr[2];
  val[2] = f00[2] * oneminusxyr[0] * oneminusxyr[3] +
           f10[2] * xyr[0] * oneminusxyr[3] +
           f01[2] * oneminusxyr[0] * xyr[3] +
           f11[2] * xyr[0] * xyr[3];
  val[3] = f00[3] * oneminusxyr[1] * oneminusxyr[3] +
           f10[3] * xyr[1] * oneminusxyr[3] +
           f01[3] * oneminusxyr[1] * xyr[3] +
           f11[3] * xyr[1] * xyr[3];

  return true;
}

#define _mm_extract_f32(v, i)       _mm_cvtss_f32(_mm_shuffle_ps(v, v, i))
//_________________________________________________________________________________________________
bool MathiesonOriginal::LUT::get4pointsAVX2(float xMin, float yMin, float xMax, float yMax, std::array<double, 4>& val) const
{
  // bilinear interpolation (https://en.wikipedia.org/wiki/Bilinear_interpolation#On_the_unit_square)
  if (mSize < 1) {
    return false;
  }

  double unit = 1.0;
  //__m256d unitVec = _mm256_broadcast_sd(&unit);
  __m256d unitVec = _mm256_set1_pd(unit);

  //__m256d inverseStepVec = _mm256_broadcast_sd(&mInverseStep);
  __m256d inverseStepVec = _mm256_set1_pd(mInverseStep);

  double minNorm = mMin * mInverseStep;
  //__m256d minNormVec = _mm256_broadcast_sd(&minNorm);
  __m256d minNormVec = _mm256_set1_pd(minNorm);

  alignas(32) double xy[4] = {xMin, xMax, yMin, yMax};
  __m256d xyVec = _mm256_load_pd(xy);
  //__m256d xyVec = _mm256_store_pd(xMin, xMax, yMin, yMax);

  //xyn[0] = xMin * mInverseStep - minNorm;
  //xyn[1] = xMax * mInverseStep - minNorm;
  //xyn[2] = yMin * mInverseStep - minNorm;
  //xyn[3] = yMax * mInverseStep - minNorm;
  __m256d xynVec = _mm256_fmsub_pd(xyVec, inverseStepVec, minNormVec);

  // get foor of normalized XY coordinates - this will be used to index the LUT table
  __m256d xy0Vec = _mm256_floor_pd(xynVec);

  // add one to the floor vector
  __m256d xy1Vec = _mm256_add_pd(xy0Vec, unitVec);

  __m256d xyrVec = _mm256_sub_pd(xynVec, xy0Vec);

  __m256d oneminusxyrVec = _mm256_sub_pd(unitVec, xyrVec);

  // convert to integer, to get the table indexes
  __m128i ixy0Vec = _mm256_cvtpd_epi32(xy0Vec);
  __m128i ixy1Vec = _mm256_cvtpd_epi32(xy1Vec);

  alignas(16) int ixy0[4]{
        _mm_extract_epi32(ixy0Vec, 0),
        _mm_extract_epi32(ixy0Vec, 1),
        _mm_extract_epi32(ixy0Vec, 2),
        _mm_extract_epi32(ixy0Vec, 3)
  };
  //_mm_stream_si128 ((__m128i*)(ixy0), ixy0Vec);

  alignas(16) int ixy1[4]{
    _mm_extract_epi32(ixy1Vec, 0),
    _mm_extract_epi32(ixy1Vec, 1),
    _mm_extract_epi32(ixy1Vec, 2),
    _mm_extract_epi32(ixy1Vec, 3)
};
  //_mm_stream_si128 ((__m128i*)(ixy1), ixy1Vec);

  // 2D grid values at the four corners around the interpolation points:
  // 0: (xMin, yMin)
  // 1: (xMax, yMin)
  // 2: (xMin, yMax)
  // 3: (xMax, yMax)

  //double f00 = mTable[y0][x0];
  alignas(32) double f00[4];
  f00[0] = mTable[ixy0[2]][ixy0[0]]; // (xMin, yMin)
  f00[1] = mTable[ixy0[2]][ixy0[1]]; // (xMax, yMin)
  f00[2] = mTable[ixy0[3]][ixy0[0]]; // (xMin, yMax)
  f00[3] = mTable[ixy0[3]][ixy0[1]]; // (xMin, yMax)
  __m256d f00Vec = _mm256_load_pd(f00);

  //double f10 = mTable[y0][x1];
  alignas(32) double f10[4];
  f10[0] = mTable[ixy0[2]][ixy1[0]]; // (xMin, yMin)
  f10[1] = mTable[ixy0[2]][ixy1[1]]; // (xMax, yMin)
  f10[2] = mTable[ixy0[3]][ixy1[0]]; // (xMin, yMax)
  f10[3] = mTable[ixy0[3]][ixy1[1]]; // (xMin, yMax)
  __m256d f10Vec = _mm256_load_pd(f10);

  //double f01 = mTable[y1][x0];
  alignas(32) double f01[4];
  f01[0] = mTable[ixy1[2]][ixy0[0]]; // (xMin, yMin)
  f01[1] = mTable[ixy1[2]][ixy0[1]]; // (xMax, yMin)
  f01[2] = mTable[ixy1[3]][ixy0[0]]; // (xMin, yMax)
  f01[3] = mTable[ixy1[3]][ixy0[1]]; // (xMin, yMax)
  __m256d f01Vec = _mm256_load_pd(f01);

  //double f11 = mTable[y1][x1];
  alignas(32) double f11[4];
  f11[0] = mTable[ixy1[2]][ixy1[0]]; // (xMin, yMin)
  f11[1] = mTable[ixy1[2]][ixy1[1]]; // (xMax, yMin)
  f11[2] = mTable[ixy1[3]][ixy1[0]]; // (xMin, yMax)
  f11[3] = mTable[ixy1[3]][ixy1[1]]; // (xMin, yMax)
  __m256d f11Vec = _mm256_load_pd(f11);

  /*
  val = f00 * (1.0 - xr) * (1.0 - yr) + f10 * xr * (1.0 - yr) + f01 * (1.0 - xr) * yr + f11 * xr * yr;
  val[0] = f00[0] * oneminusxyr[0] * oneminusxyr[2] +
           f10[0] * xyr[0] * oneminusxyr[2] +
           f01[0] * oneminusxyr[0] * xyr[2] +
           f11[0] * xyr[0] * xyr[2];
  val[1] = f00[1] * oneminusxyr[1] * oneminusxyr[2] +
           f10[1] * xyr[1] * oneminusxyr[2] +
           f01[1] * oneminusxyr[1] * xyr[2] +
           f11[1] * xyr[1] * xyr[2];
  val[2] = f00[2] * oneminusxyr[0] * oneminusxyr[3] +
           f10[2] * xyr[0] * oneminusxyr[3] +
           f01[2] * oneminusxyr[0] * xyr[3] +
           f11[2] * xyr[0] * xyr[3];
  val[3] = f00[3] * oneminusxyr[1] * oneminusxyr[3] +
           f10[3] * xyr[1] * oneminusxyr[3] +
           f01[3] * oneminusxyr[1] * xyr[3] +
           f11[3] * xyr[1] * xyr[3];
  */
  __m256d tmp;
  // f00
  __m256d c000 = _mm256_permute4x64_pd(oneminusxyrVec, 0b01000100); // v[0],v[1],v[2],v[3] -> v[0],v[1],v[0],v[1]
  tmp = _mm256_mul_pd(f00Vec, c000);
  __m256d c001 = _mm256_permute4x64_pd(oneminusxyrVec, 0b11111010); // v[0],v[1],v[2],v[3] -> v[2],v[2],v[3],v[3]
  __m256d c00 = _mm256_mul_pd(tmp, c001);

  // f10
  __m256d c100 = _mm256_permute4x64_pd(xyrVec, 0b01000100); // v[0],v[1],v[2],v[3] -> v[0],v[1],v[0],v[1]
  tmp = _mm256_mul_pd(f10Vec, c100);
  __m256d c101 = c001;                                      // v[0],v[1],v[2],v[3] -> v[2],v[2],v[3],v[3]
  __m256d c10 = _mm256_mul_pd(tmp, c101);

  // f01
  __m256d c010 = c000;                                      // v[0],v[1],v[2],v[3] -> v[0],v[1],v[0],v[1]
  tmp = _mm256_mul_pd(f01Vec, c010);
  __m256d c011 = _mm256_permute4x64_pd(xyrVec, 0b11111010); // v[0],v[1],v[2],v[3] -> v[2],v[2],v[3],v[3]
  __m256d c01 = _mm256_mul_pd(tmp, c011);

  // f11
  __m256d c110 = c100;                    // v[0],v[1],v[2],v[3] -> v[0],v[1],v[0],v[1]
  tmp = _mm256_mul_pd(f11Vec, c110);
  __m256d c111 = c011;                    // v[0],v[1],v[2],v[3] -> v[2],v[2],v[3],v[3]
  __m256d c11 = _mm256_mul_pd(tmp, c111);


  __m256d outVec = _mm256_add_pd(_mm256_add_pd(c00, c10), _mm256_add_pd(c01, c11));
  //alignas(32) double out[4];
  //_mm256_stream_pd(out, outVec);

  val[0] = outVec[0];
  val[1] = outVec[1];
  val[2] = outVec[2];
  val[3] = outVec[3];

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

  for (int j = 0; j < mLUT.mLUTSize; j++) {
    float y = mLUT.getY(j);
    for (int i = 0; i < mLUT.mLUTSize; i++) {
      float x = mLUT.getX(i);
      double integral = integrateAnalytic(mLUT.mMin, mLUT.mMin, x, y);
      mLUT.set(i, j, integral);
#ifdef DEBUG_LUT
      //std::cout << "LUT[" << i << "," << j << "] -> (" << x << "," << y << ") = " << integral << std::endl;
#endif
    }
  }
}

//_________________________________________________________________________________________________
double MathiesonOriginal::integrateLUT(float xMin, float yMin, float xMax, float yMax) const
{
  /// integrate the Mathieson over x and y in the given area
  /*double i00{ 0 };
  double i10{ 0 };
  double i01{ 0 };
  double i11{ 0 };
  mLUT.get(xMin, yMin, i00);
  mLUT.get(xMax, yMin, i10);
  mLUT.get(xMin, yMax, i01);
  mLUT.get(xMax, yMax, i11);

  return (i00 - i10 - i01 + i11);
  */

  std::array<double, 4> i;
  std::array<double, 4> i2;
  mLUT.get4pointsAVX2(xMin, yMin, xMax, yMax, i);
  //mLUT.get4points(xMin, yMin, xMax, yMax, i);

  if (false) {
  std::array<double, 4> delta;
  delta[0] = i[0] / 100;
  delta[1] = i[1] / 100;
  delta[2] = i[2] / 100;
  delta[3] = i[3] / 100;

  if ((std::fabs(i[0] - i2[0]) > delta[0]) ||
      (std::fabs(i[1] - i2[1]) > delta[1]) ||
      (std::fabs(i[2] - i2[2]) > delta[2]) ||
      (std::fabs(i[3] - i2[3]) > delta[3])) {
    std::cout << fmt::format("[get4points]: {}/{}  {}/{}  {}/{}  {}/{}",
        i[0], i2[0], i[1], i2[1], i[2], i2[2], i[3], i2[3]) << std::endl;
  }
  }

  return (i[0] - i[1] - i[2] + i[3]);
  //return (iAVX[0] - iAVX[1] - iAVX[2] + iAVX[3]);
}

//_________________________________________________________________________________________________
double MathiesonOriginal::integrateAnalytic(float xMin, float yMin, float xMax, float yMax) const
{
  /// integrate the Mathieson over x and y in the given area

  //
  // The Mathieson function
  double uxMin = mSqrtKx3 * TMath::TanH(mKx2 * xMin);
  double uxMax = mSqrtKx3 * TMath::TanH(mKx2 * xMax);

  double uyMin = mSqrtKy3 * TMath::TanH(mKy2 * yMin);
  double uyMax = mSqrtKy3 * TMath::TanH(mKy2 * yMax);

  return (4. * mKx4 * (TMath::ATan(uxMax) - TMath::ATan(uxMin)) *
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


  double integral{ 0 };
  if (mFastIntegral && mLUT.isIncluded(xMin, yMin) && mLUT.isIncluded(xMax, yMax)) {
    integral = integrateLUT(xMin, yMin, xMax, yMax);
#ifdef DEBUG_LUT
    double int2 = integrateAnalytic(xMin, yMin, xMax, yMax);
    std::cout << fmt::format("({}, {}) -> ({}, {}): analytic={} lut={}", xMin, yMin, xMax, yMax, int2, integral) << std::endl;
#endif
  } else {
    integral = integrateAnalytic(xMin, yMin, xMax, yMax);
  }

  return static_cast<float>(integral);
}

} // namespace mch
} // namespace o2
