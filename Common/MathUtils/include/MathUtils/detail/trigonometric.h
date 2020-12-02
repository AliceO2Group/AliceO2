// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file trigonometric.h
/// \brief
/// \author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch

#ifndef MATHUTILS_INCLUDE_MATHUTILS_DETAIL_TRIGONOMETRIC_H_
#define MATHUTILS_INCLUDE_MATHUTILS_DETAIL_TRIGONOMETRIC_H_

#ifndef GPUCA_GPUCODE_DEVICE
#include <array>
#include <cmath>
#include <tuple>
#endif
#include "GPUCommonDef.h"
#include "GPUCommonMath.h"
#include "CommonConstants/MathConstants.h"

namespace o2
{
namespace math_utils
{
namespace detail
{

template <typename T>
GPUdi() T to02Pi(T phi)
{
  T res = phi;
  // ensure angle in [0:2pi] for the input in [-pi:pi] or [0:pi]
  if (res < 0.f) {
    res += o2::constants::math::TwoPI;
  }
  return res;
}

template <typename T>
GPUdi() void bringTo02Pi(T& phi)
{
  phi = to02Pi<T>(phi);
}

template <typename T>
inline T to02PiGen(T phi)
{
  T res = phi;
  // ensure angle in [0:2pi] for the any input angle
  while (res < 0.f) {
    res += o2::constants::math::TwoPI;
  }
  while (res > o2::constants::math::TwoPI) {
    res -= o2::constants::math::TwoPI;
  }
  return res;
}

template <typename T>
inline void bringTo02PiGen(T& phi)
{
  phi = to02PiGen<T>(phi);
}

template <typename T>
inline T toPMPi(T phi)
{
  T res = phi;
  // ensure angle in [-pi:pi] for the input in [-pi:pi] or [0:pi]
  if (res > o2::constants::math::PI) {
    res -= o2::constants::math::TwoPI;
  } else if (res < -o2::constants::math::PI) {
    res += o2::constants::math::TwoPI;
  }
  return res;
}

template <typename T>
inline void bringToPMPi(T& phi)
{
  phi = toPMPi<T>(phi);
}

template <typename T>
inline T toPMPiGen(T phi)
{
  // ensure angle in [-pi:pi] for any input angle
  T res = phi;
  while (res < -o2::constants::math::PI) {
    res += o2::constants::math::TwoPI;
  }
  while (res > o2::constants::math::PI) {
    res -= o2::constants::math::TwoPI;
  }
  return res;
}

template <typename T>
inline void bringToPMPiGen(T& phi)
{
  phi = toPMPiGen<T>(phi);
}

template <typename T>
GPUdi() void sincos(T ang, GPUgeneric() T& s, GPUgeneric() T& c)
{
  return o2::gpu::GPUCommonMath::SinCos(ang, s, c);
}

template <>
GPUdi() void sincos(double ang, GPUgeneric() double& s, GPUgeneric() double& c)
{
  return o2::gpu::GPUCommonMath::SinCosd(ang, s, c);
}

#ifndef GPUCA_GPUCODE_DEVICE

template <typename T>
GPUhdi() std::tuple<T, T> sincos(T ang)
{
  T sin = 0;
  T cos = 0;
  sincos<T>(ang, sin, cos);
  return std::make_tuple(sin, cos);
}

template <typename T>
inline std::tuple<T, T> rotateZ(T xL, T yL, T snAlp, T csAlp)
{
  // 2D rotation of the point by angle alpha (local to global)
  return std::make_tuple(xL * csAlp - yL * snAlp, xL * snAlp + yL * csAlp);
}

template <typename T>
inline std::tuple<T, T> rotateZInv(T xG, T yG, T snAlp, T csAlp)
{
  // inverse 2D rotation of the point by angle alpha (global to local)
  return rotateZ<T>(xG, yG, -snAlp, csAlp);
}

template <typename T>
inline void rotateZ(T xL, T yL, T& xG, T& yG, T snAlp, T csAlp)
{
  std::tie(xG, yG) = rotateZ<T>(xL, yL, snAlp, csAlp);
}

template <typename T>
inline void rotateZ(std::array<T, 3>& xy, T alpha)
{
  // transforms vector in tracking frame alpha to global frame
  auto [sin, cos] = sincos<T>(alpha);
  const T x = xy[0];
  xy[0] = x * cos - xy[1] * sin;
  xy[1] = x * sin + xy[1] * cos;
}

template <typename T>
inline void rotateZInv(T xG, T yG, T& xL, T& yL, T snAlp, T csAlp)
{
  // inverse 2D rotation of the point by angle alpha (global to local)
  rotateZ<T>(xG, yG, xL, yL, -snAlp, csAlp);
}

#endif

template <typename T>
inline int angle2Sector(T phi)
{
  // convert angle to sector ID, phi can be either in 0:2pi or -pi:pi convention
  int sect = phi * o2::constants::math::Rad2Deg / o2::constants::math::SectorSpanDeg;
  if (phi < 0.f) {
    sect += o2::constants::math::NSectors - 1;
  }
  return sect;
}

template <typename T>
inline T sector2Angle(int sect)
{
  // convert sector to its angle center, in -pi:pi convention
  T ang = o2::constants::math::SectorSpanRad * (0.5f + sect);
  bringToPMPi<T>(ang);
  return ang;
}

template <typename T>
inline T angle2Alpha(T phi)
{
  // convert angle to its sector alpha
  return sector2Angle<T>(angle2Sector<T>(phi));
}

template <typename T>
GPUhdi() T copysign(T x, T y)
{
  return o2::gpu::GPUCommonMath::Copysign(x, y);
}

template <>
GPUhdi() double copysign(double x, double y)
{
  return o2::gpu::GPUCommonMath::Copysignd(x, y);
}

template <typename T>
GPUhdi() T fastATan2(T y, T x)
{
  // Fast atan2(y,x) for any angle [-Pi,Pi]
  // Average inaccuracy: 0.00048
  // Max inaccuracy: 0.00084
  // Speed: 6.2 times faster than atan2f()
  constexpr T Pi = 3.1415926535897932384626433832795;

  auto atan = [](T a) -> T {
    // returns the arctan for the angular range [-Pi/4, Pi/4]
    // the polynomial coefficients are taken from:
    // https://stackoverflow.com/questions/42537957/fast-accurate-atan-arctan-approximation-algorithm
    constexpr T A = 0.0776509570923569;
    constexpr T B = -0.287434475393028;
    constexpr T C = (Pi / 4 - A - B);
    const T a2 = a * a;
    return ((A * a2 + B) * a2 + C) * a;
  };

  auto atan2P = [atan](T yy, T xx) -> T {
    // fast atan2(yy,xx) for the angular range [0,+Pi]
    constexpr T Pi025 = 1 * Pi / 4;
    constexpr T Pi075 = 3 * Pi / 4;
    const T x1 = xx + yy; //  point p1 (x1,y1) = (xx,yy) - Pi/4
    const T y1 = yy - xx;
    T phi0 = 0;
    T tan = 0;
    if (xx < 0) { // p1 is in the range [Pi/4, 3*Pi/4]
      phi0 = Pi075;
      tan = -x1 / y1;
    } else { // p1 is in the range [-Pi/4, Pi/4]
      phi0 = Pi025;
      tan = y1 / x1;
    }
    return phi0 + atan(tan);
  };

  // fast atan2(y,x) for any angle [-Pi,Pi]
  return copysign<T>(atan2P(o2::gpu::GPUCommonMath::Abs(y), x), y);
}

} // namespace detail
} // namespace math_utils
} // namespace o2

#endif /* MATHUTILS_INCLUDE_MATHUTILS_DETAIL_TRIGONOMETRIC_H_ */
