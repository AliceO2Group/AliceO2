// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUDisplayQuaternion.cpp
/// \author David Rohr

#include "GPUDisplay.h"
#ifdef GPUCA_BUILD_EVENT_DISPLAY

#include <cmath>
using namespace GPUCA_NAMESPACE::gpu;

void GPUDisplay::createQuaternionFromMatrix(float* v, const float* mat)
{
  if (mat[0] > mat[5] && mat[0] > mat[10]) {
    const float S = sqrt(std::max(0.f, 1.0f + mat[0] - mat[5] - mat[10])) * 2;
    v[0] = 0.25 * S;
    v[1] = (mat[4] + mat[1]) / S;
    v[2] = (mat[2] + mat[8]) / S;
    v[3] = (mat[9] - mat[6]) / S;
  } else if (mat[5] > mat[10]) {
    const float S = sqrt(std::max(0.f, 1.0f + mat[5] - mat[0] - mat[10])) * 2;
    v[1] = 0.25 * S;
    v[0] = (mat[4] + mat[1]) / S;
    v[2] = (mat[9] + mat[6]) / S;
    v[3] = (mat[2] - mat[8]) / S;
  } else {
    float S = sqrt(std::max(0.f, 1.0f + mat[10] - mat[0] - mat[5])) * 2;
    v[2] = 0.25 * S;
    if (fabsf(S) < 0.001) {
      S = 1;
    }
    v[0] = (mat[2] + mat[8]) / S;
    v[1] = (mat[9] + mat[6]) / S;
    v[3] = (mat[4] - mat[1]) / S;
  }
  if (v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3] < 0.0001) {
    v[3] = 1;
  }
}

#endif
