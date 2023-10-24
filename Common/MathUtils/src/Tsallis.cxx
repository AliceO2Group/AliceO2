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

#include "MathUtils/Tsallis.h"
#include <cmath>

namespace o2::math_utils
{

float Tsallis::tsallisCharged(float pt, float mass, float sqrts)
{
  const float a = 6.81;
  const float b = 59.24;
  const float c = 0.082;
  const float d = 0.151;
  const float mt = std::sqrt(mass * mass + pt * pt);
  const float n = a + b / sqrts;
  const float T = c + d / sqrts;
  const float p0 = n * T;
  return std::pow((1. + mt / p0), -n) * pt;
}

bool Tsallis::downsampleTsallisCharged(float pt, float factorPt, float sqrts, float& weight, float rnd, float mass)
{
  const float prob = tsallisCharged(pt, mass, sqrts);
  const float probNorm = tsallisCharged(1., mass, sqrts);
  weight = prob / probNorm;
  return (rnd * (weight * pt * pt)) < factorPt;
}

} // namespace o2::math_utils
