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
/// \author ruben.shahoyan@cern.ch

#ifndef ALICEO2_COMMON_MATH_UTILS_
#define ALICEO2_COMMON_MATH_UTILS_

#ifndef __OPENCL__
#include <array>
#include <cmath>
#endif
#include "GPUCommonDef.h"
#include "GPUCommonMath.h"
#include "CommonConstants/MathConstants.h"

namespace o2
{
// namespace common
//{
namespace utils
{
GPUdi() void BringTo02Pi(float& phi)
{
  // ensure angle in [0:2pi] for the input in [-pi:pi] or [0:pi]
  if (phi < 0.f) {
    phi += o2::constants::math::TwoPI;
  }
}

inline void BringTo02PiGen(float& phi)
{
  // ensure angle in [0:2pi] for the any input angle
  while (phi < 0.f) {
    phi += o2::constants::math::TwoPI;
  }
  while (phi > o2::constants::math::TwoPI) {
    phi -= o2::constants::math::TwoPI;
  }
}

inline void BringToPMPi(float& phi)
{
  // ensure angle in [-pi:pi] for the input in [-pi:pi] or [0:pi]
  if (phi > o2::constants::math::PI) {
    phi -= o2::constants::math::TwoPI;
  }
}

inline void BringToPMPiGen(float& phi)
{
  // ensure angle in [-pi:pi] for any input angle
  while (phi < -o2::constants::math::PI) {
    phi += o2::constants::math::TwoPI;
  }
  while (phi > o2::constants::math::PI) {
    phi -= o2::constants::math::TwoPI;
  }
}

inline void sincosf(float ang, float& s, float& c)
{
  // consider speedup for simultaneus calculation
  s = o2::gpu::CAMath::Sin(ang);
  c = o2::gpu::CAMath::Cos(ang);
}

#ifndef __OPENCL__
inline void RotateZ(std::array<float, 3>& xy, float alpha)
{
  // transforms vector in tracking frame alpha to global frame
  float sn, cs, x = xy[0];
  sincosf(alpha, sn, cs);
  xy[0] = x * cs - xy[1] * sn;
  xy[1] = x * sn + xy[1] * cs;
}
#endif

inline int Angle2Sector(float phi)
{
  // convert angle to sector ID, phi can be either in 0:2pi or -pi:pi convention
  int sect = phi * o2::constants::math::Rad2Deg / o2::constants::math::SectorSpanDeg;
  if (phi < 0.f) {
    sect += o2::constants::math::NSectors - 1;
  }
  return sect;
}

inline float Sector2Angle(int sect)
{
  // convert sector to its angle center, in -pi:pi convention
  float ang = o2::constants::math::SectorSpanRad * (0.5f + sect);
  BringToPMPi(ang);
  return ang;
}

inline float Angle2Alpha(float phi)
{
  // convert angle to its sector alpha
  return Sector2Angle(Angle2Sector(phi));
}

//-------------------------------------->>>
// recursive creation of bitmask
template <typename T>
constexpr int bit2Mask(T v)
{
  return 0x1 << v;
}

template <typename T, typename... Args>
constexpr int bit2Mask(T first, Args... args)
{
  return (0x1 << first) | bit2Mask(args...);
}
//--------------------------------------<<<

GPUdi() float FastATan2(float y, float x)
{
  // Fast atan2(y,x) for any angle [-Pi,Pi]
  // Average inaccuracy: 0.00048
  // Max inaccuracy: 0.00084
  // Speed: 6.2 times faster than atan2f()

  constexpr float kPi = 3.1415926535897f;

  auto atan = [](float a) -> float {
    // returns the arctan for the angular range [-Pi/4, Pi/4]
    // the polynomial coefficients are taken from:
    // https://stackoverflow.com/questions/42537957/fast-accurate-atan-arctan-approximation-algorithm
    constexpr float kA = 0.0776509570923569f;
    constexpr float kB = -0.287434475393028f;
    constexpr float kC = (kPi / 4 - kA - kB);
    float a2 = a * a;
    return ((kA * a2 + kB) * a2 + kC) * a;
  };

  auto atan2P = [atan](float yy, float xx) -> float {
    // fast atan2(yy,xx) for the angular range [0,+Pi]
    constexpr float kPi025 = 1 * kPi / 4;
    constexpr float kPi075 = 3 * kPi / 4;
    float x1 = xx + yy; //  point p1 (x1,y1) = (xx,yy) - Pi/4
    float y1 = yy - xx;
    float phi0, tan;
    if (xx < 0) { // p1 is in the range [Pi/4, 3*Pi/4]
      phi0 = kPi075;
      tan = -x1 / y1;
    } else { // p1 is in the range [-Pi/4, Pi/4]
      phi0 = kPi025;
      tan = y1 / x1;
    }
    return phi0 + atan(tan);
  };

  // fast atan2(y,x) for any angle [-Pi,Pi]
  return copysignf(atan2P(o2::gpu::CAMath::Abs(y), x), y);
}

} // namespace utils
//}
} // namespace o2

#endif
