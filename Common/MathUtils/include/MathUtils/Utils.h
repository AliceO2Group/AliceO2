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
} // namespace utils
//}
} // namespace o2

#endif
