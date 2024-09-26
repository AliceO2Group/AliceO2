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

/// \file Utils.h
/// \brief General auxilliary methods
/// \author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch

#ifndef ALICEO2_COMMON_MATH_UTILS_
#define ALICEO2_COMMON_MATH_UTILS_

#include "MathUtils/detail/bitOps.h"
#include "MathUtils/detail/StatAccumulator.h"
#include "MathUtils/detail/trigonometric.h"
#include "MathUtils/detail/TypeTruncation.h"
#include "MathUtils/detail/basicMath.h"

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

GPUdi() void rotateZInv(float xG, float yG, float& xL, float& yL, float snAlp, float csAlp)
{
  detail::rotateZInv<float>(xG, yG, xL, yL, snAlp, csAlp);
}

GPUdi() void rotateZInvd(double xG, double yG, double& xL, double& yL, double snAlp, double csAlp)
{
  detail::rotateZInv<double>(xG, yG, xL, yL, snAlp, csAlp);
}

#ifndef GPUCA_GPUCODE_DEVICE
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

template <class T>
GPUhdi() T min(const T x, const T y)
{
  return detail::min<T>(x, y);
};

GPUhdi() double mind(const double x, const double y)
{
  return detail::min<double>(x, y);
};

template <class T>
GPUhdi() T max(const T x, const T y)
{
  return detail::max<T>(x, y);
};

GPUhdi() double maxd(const double x, const double y)
{
  return detail::max<double>(x, y);
};

GPUhdi() float sqrt(float x)
{
  return detail::sqrt<float>(x);
};

GPUhdi() double sqrtd(double x)
{
  return detail::sqrt<double>(x);
};

GPUhdi() float abs(float x)
{
  return detail::abs<float>(x);
};

GPUhdi() double absd(double x)
{
  return detail::abs<double>(x);
};

GPUdi() float asin(float x)
{
  return detail::asin<float>(x);
};

GPUdi() double asind(double x)
{
  return detail::asin<double>(x);
};

GPUdi() float atan(float x)
{
  return detail::atan<float>(x);
};

GPUdi() double atand(double x)
{
  return detail::atan<double>(x);
};

GPUdi() float atan2(float y, float x)
{
  return detail::atan2<float>(y, x);
};

GPUdi() double atan2d(double y, double x)
{
  return detail::atan2<double>(y, x);
};

GPUdi() float sin(float x)
{
  return detail::sin<float>(x);
};

GPUdi() double sind(double x)
{
  return detail::sin<double>(x);
};

GPUdi() float cos(float x)
{
  return detail::cos<float>(x);
};

GPUdi() double cosd(double x)
{
  return detail::cos<double>(x);
};

GPUdi() float tan(float x)
{
  return detail::tan<float>(x);
};

GPUdi() double tand(double x)
{
  return detail::tan<double>(x);
};

GPUdi() float twoPi()
{
  return detail::twoPi<float>();
};

GPUdi() double twoPid()
{
  return detail::twoPi<double>();
};

GPUdi() float pi()
{
  return detail::pi<float>();
}

GPUdi() double pid()
{
  return detail::pi<double>();
}

GPUdi() int nint(float x)
{
  return detail::nint<float>(x);
};

GPUdi() int nintd(double x)
{
  return detail::nint<double>(x);
};

GPUdi() bool finite(float x)
{
  return detail::finite<float>(x);
}

GPUdi() bool finited(double x)
{
  return detail::finite<double>(x);
}

GPUdi() unsigned int clz(unsigned int val)
{
  return detail::clz(val);
};

GPUdi() unsigned int popcount(unsigned int val)
{
  return detail::popcount(val);
};

GPUdi() float log(float x)
{
  return detail::log<float>(x);
};

GPUdi() double logd(double x)
{
  return detail::log<double>(x);
};

using detail::StatAccumulator;

using detail::bit2Mask;
using detail::numberOfBitsSet;
using detail::truncateFloatFraction;

} // namespace math_utils
} // namespace o2

#endif
