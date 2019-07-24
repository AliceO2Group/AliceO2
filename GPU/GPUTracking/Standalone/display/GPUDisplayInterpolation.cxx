// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUDisplayInterpolation.cxx
/// \author David Rohr

#include <cstdio>
#include "GPUDisplay.h"
#ifdef GPUCA_BUILD_EVENT_DISPLAY

using namespace GPUCA_NAMESPACE::gpu;

void GPUDisplay::opengl_spline::create(const vecpod<float>& x, const vecpod<float>& y)
{
  ma.clear();
  mb.clear();
  mc.clear();
  md.clear();
  mx.clear();
  if (x.size() != y.size() || x.size() < 2) {
    return;
  }
  int k = x.size() - 1;
  if (mVerbose) {
    for (unsigned int i = 0; i < x.size(); i++) {
      printf("Point %u: %f --> %f\n", i, x[i], y[i]);
    }
  }
  ma.resize(k + 1);
  mb.resize(k + 1);
  mc.resize(k + 1);
  md.resize(k + 1);
  mx.resize(k + 1);
  vecpod<float> h(k + 1), alpha(k + 1), l(k + 1), mu(k + 1), z(k + 1);
  for (int i = 0; i <= k; i++) {
    ma[i] = y[i];
  }
  for (int i = 0; i < k; i++) {
    h[i] = x[i + 1] - x[i];
  }
  for (int i = 1; i < k; i++) {
    alpha[i] = 3.f / h[i] * (ma[i + 1] - ma[i]) - 3.f / h[i - 1] * (ma[i] - ma[i - 1]);
  }
  l[0] = l[k] = 1;
  mu[0] = z[0] = z[k] = mc[k] = 0;
  for (int i = 1; i < k; i++) {
    l[i] = 2.f * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1];
    mu[i] = h[i] / l[i];
    z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
  }
  for (int i = k - 1; i >= 0; i--) {
    mc[i] = z[i] - mu[i] * mc[i + 1];
    mb[i] = (ma[i + 1] - ma[i]) / h[i] - h[i] / 3.f * (mc[i + 1] + 2.f * mc[i]);
    md[i] = (mc[i + 1] - mc[i]) / (3.f * h[i]);
  }
  for (int i = 0; i <= k; i++) {
    mx[i] = x[i];
  }
}

float GPUDisplay::opengl_spline::evaluate(float x)
{
  int base = 0;
  const int k = mx.size() - 1;
  if (k < 0) {
    return (0);
  }
  while (base < k - 1 && x > mx[base + 1]) {
    base++;
  }
  float retVal = ma[base];
  x -= mx[base];
  const float xx = x;
  retVal += mb[base] * x;
  x *= xx;
  retVal += mc[base] * x;
  x *= xx;
  retVal += md[base] * x;
  if (mVerbose) {
    printf("Evaluate: %f --> %f (basepoint %d)\n", xx, retVal, base);
  }
  return (retVal);
}

#endif
