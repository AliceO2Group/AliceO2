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

/// \file MathConstants.h
/// \brief useful math constants
/// \author ruben.shahoyan@cern.ch

#ifndef ALICEO2_COMMON_MATH_CONSTANTS_
#define ALICEO2_COMMON_MATH_CONSTANTS_

namespace o2
{
namespace constants
{
namespace math
{
constexpr float Almost0 = 1.17549e-38;
constexpr float Almost1 = 1.f - 1.0e-6;
constexpr float VeryBig = 1.f / Almost0;

constexpr float PI = 3.14159274101257324e+00f;
constexpr float TwoPI = 2.f * PI;
constexpr float PIHalf = 0.5f * PI;
constexpr float PIThird = PI / 3.0f;
constexpr float PIQuarter = 0.25f * PI;
constexpr float Rad2Deg = 180.f / PI;
constexpr float Deg2Rad = PI / 180.f;

constexpr int NSectors = 18;
constexpr float SectorSpanDeg = 360. / NSectors;
constexpr float SectorSpanRad = SectorSpanDeg * Deg2Rad;

// conversion from B(kGaus) to curvature for 1GeV pt
constexpr float B2C = -0.299792458e-3;
} // namespace math
} // namespace constants
} // namespace o2
#endif
