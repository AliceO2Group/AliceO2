// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "ITS3Align/AlignmentMath.h"

#include <cmath>

#include <TMath.h>

#include "ITS3Base/SpecsV2.h"
#include "MathUtils/Utils.h"

namespace o2::its3::align
{

double getSensorPhiWidth(int sensorID, double radius)
{
  const bool isTop = sensorID % 2 == 0;
  const double phiBorder1 = o2::math_utils::to02Pid(((isTop ? 0. : 1.) * TMath::Pi()) + std::asin(constants::equatorialGap / 2. / radius));
  const double phiBorder2 = o2::math_utils::to02Pid(((isTop ? 1. : 2.) * TMath::Pi()) - std::asin(constants::equatorialGap / 2. / radius));
  const double width = phiBorder2 - phiBorder1;
  return (width < 0.) ? width + TMath::TwoPi() : width;
}

std::pair<double, double> computeUV(double gloX, double gloY, double gloZ, int sensorID, double radius)
{
  const bool isTop = sensorID % 2 == 0;
  const double phi = o2::math_utils::to02Pid(std::atan2(gloY, gloX));
  const double phiBorder1 = o2::math_utils::to02Pid(((isTop ? 0. : 1.) * TMath::Pi()) + std::asin(constants::equatorialGap / 2. / radius));
  const double phiBorder2 = o2::math_utils::to02Pid(((isTop ? 1. : 2.) * TMath::Pi()) - std::asin(constants::equatorialGap / 2. / radius));
  const double u = (((phi - phiBorder1) * 2.) / (phiBorder2 - phiBorder1)) - 1.;
  const double v = ((2. * gloZ + constants::segment::lengthSensitive) / constants::segment::lengthSensitive) - 1.;
  return {u, v};
}

TrackSlopes computeTrackSlopes(double snp, double tgl)
{
  const double csci = 1. / std::sqrt(1. - (snp * snp));
  return {.dydx = snp * csci, .dzdx = tgl * csci};
}

std::vector<double> legendrePols(int order, double x)
{
  std::vector<double> p(order + 1);
  p[0] = 1.;
  if (order > 0) {
    p[1] = x;
  }
  for (int n = 1; n < order; ++n) {
    p[n + 1] = ((2 * n + 1) * x * p[n] - n * p[n - 1]) / (n + 1);
  }
  return p;
}

} // namespace o2::its3::align
