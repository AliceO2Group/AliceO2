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

#include "GPUCommonDef.h"

namespace o2
{
namespace constants
{
namespace math
{
GPUglobalconstexpr() float Almost0 = 0x1.0p-126f;   // smallest non-denormal float
GPUglobalconstexpr() float Epsilon = 0x0.000002p0f; // smallest float such that 1 != 1 + Epsilon
GPUglobalconstexpr() float Almost1 = 1.f - 1.0e-6f;
GPUglobalconstexpr() float VeryBig = 1.f / Almost0;

GPUglobalconstexpr() float PI = 3.14159274101257324e+00f;
GPUglobalconstexpr() float TwoPI = 2.f * PI;
GPUglobalconstexpr() float PIHalf = 0.5f * PI;
GPUglobalconstexpr() float PIThird = PI / 3.0f;
GPUglobalconstexpr() float PIQuarter = 0.25f * PI;
GPUglobalconstexpr() float Rad2Deg = 180.f / PI;
GPUglobalconstexpr() float Deg2Rad = PI / 180.f;

GPUglobalconstexpr() int NSectors = 18;
GPUglobalconstexpr() float SectorSpanDeg = 360. / NSectors;
GPUglobalconstexpr() float SectorSpanRad = SectorSpanDeg * Deg2Rad;

// conversion from B(kGaus) to curvature for 1GeV pt
GPUglobalconstexpr() float B2C = -0.299792458e-3;
} // namespace math
} // namespace constants
} // namespace o2
#endif
