// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Utils.h
/// \brief General auxilliary methods
/// \author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch

#ifndef ALICEO2_COMMON_MATH_UTILS_
#define ALICEO2_COMMON_MATH_UTILS_

#include "MathUtils/detail/bitOps.h"
#include "MathUtils/detail/StatAccumulator.h"
#include "MathUtils/detail/trigonometric.h"
#include "MathUtils/detail/TypeTruncation.h"

namespace o2
{
namespace math_utils
{

GPUdi() float to02Pi(float phi)
{
  return detail::to02Pi<float>(phi);
}

GPUdi() double to02Pid(double phi)
{
  return detail::to02Pi<double>(phi);
}

GPUdi() void bringTo02Pi(float& phi)
{
  detail::bringTo02Pi<float>(phi);
}

GPUdi() void bringTo02Pid(double& phi)
{
  detail::bringTo02Pi<double>(phi);
}

inline float toPMPiGen(float phi)
{
  return detail::toPMPiGen<float>(phi);
}

inline double toPMPiGend(double phi)
{
  return detail::toPMPiGen<double>(phi);
}

inline void bringToPMPiGen(float& phi)
{
  detail::bringToPMPiGen<float>(phi);
}

inline void bringToPMPiGend(double& phi)
{
  detail::bringToPMPiGen<double>(phi);
}

inline float to02PiGen(float phi)
{
  return detail::to02PiGen<float>(phi);
}

inline double to02PiGend(double phi)
{
  return detail::to02PiGen<double>(phi);
}

inline void bringTo02PiGen(float& phi)
{
  detail::bringTo02PiGen<float>(phi);
}

inline void bringTo02PiGend(double& phi)
{
  detail::bringTo02PiGen<double>(phi);
}

inline float toPMPi(float phi)
{
  return detail::toPMPi<float>(phi);
}

inline double toPMPid(double phi)
{
  return detail::toPMPi<double>(phi);
}

inline void bringToPMPi(float& phi)
{
  return detail::bringToPMPi<float>(phi);
}

inline void bringToPMPid(double& phi)
{
  return detail::bringToPMPi<double>(phi);
}

GPUdi() void sincos(float ang, float& s, float& c)
{
  detail::sincos<float>(ang, s, c);
}
#ifndef __OPENCL__
GPUdi() void sincosd(double ang, double& s, double& c)
{
  detail::sincos<double>(ang, s, c);
}
#endif

GPUdi() void rotateZ(float xL, float yL, float& xG, float& yG, float snAlp, float csAlp)
{
  return detail::rotateZ<float>(xL, yL, xG, yG, snAlp, csAlp);
}

GPUdi() void rotateZd(double xL, double yL, double& xG, double& yG, double snAlp, double csAlp)
{
  return detail::rotateZ<double>(xL, yL, xG, yG, snAlp, csAlp);
}

#ifndef GPUCA_GPUCODE_DEVICE
inline void rotateZInv(float xG, float yG, float& xL, float& yL, float snAlp, float csAlp)
{
  detail::rotateZInv<float>(xG, yG, xL, yL, snAlp, csAlp);
}

inline void rotateZInvd(double xG, double yG, double& xL, double& yL, double snAlp, double csAlp)
{
  detail::rotateZInv<double>(xG, yG, xL, yL, snAlp, csAlp);
}

inline std::tuple<float, float> rotateZInv(float xG, float yG, float snAlp, float csAlp)
{
  return detail::rotateZInv<float>(xG, yG, snAlp, csAlp);
}

inline std::tuple<double, double> rotateZInvd(double xG, double yG, double snAlp, double csAlp)
{
  return detail::rotateZInv<double>(xG, yG, snAlp, csAlp);
}

GPUdi() std::tuple<float, float> sincos(float ang)
{
  return detail::sincos<float>(ang);
}

GPUdi() std::tuple<double, double> sincosd(double ang)
{
  return detail::sincos<double>(ang);
}

inline std::tuple<float, float> rotateZ(float xL, float yL, float snAlp, float csAlp)
{
  return detail::rotateZ<float>(xL, yL, snAlp, csAlp);
}

inline std::tuple<double, double> rotateZd(double xL, double yL, double snAlp, double csAlp)
{
  return detail::rotateZ<double>(xL, yL, snAlp, csAlp);
}

inline void rotateZ(std::array<float, 3>& xy, float alpha)
{
  detail::rotateZ<float>(xy, alpha);
}

inline void rotateZd(std::array<double, 3>& xy, double alpha)
{
  detail::rotateZ<double>(xy, alpha);
}
#endif

inline int angle2Sector(float phi)
{
  return detail::angle2Sector<float>(phi);
}

inline int angle2Sectord(double phi)
{
  return detail::angle2Sector<double>(phi);
}

inline float sector2Angle(int sect)
{
  return detail::sector2Angle<float>(sect);
}

inline double sector2Angled(int sect)
{
  return detail::sector2Angle<double>(sect);
}

inline float angle2Alpha(float phi)
{
  return detail::angle2Alpha<float>(phi);
}

inline double angle2Alphad(double phi)
{
  return detail::angle2Alpha<double>(phi);
}

GPUhdi() float fastATan2(float y, float x)
{
  return detail::fastATan2<float>(y, x);
}

GPUhdi() double fastATan2d(double y, double x)
{
  return detail::fastATan2<double>(y, x);
}

using detail::StatAccumulator;

using detail::bit2Mask;
using detail::numberOfBitsSet;
using detail::truncateFloatFraction;

} // namespace math_utils
} // namespace o2

#endif
