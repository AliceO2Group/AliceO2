// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file basicMath.h
/// \brief
/// \author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch

#ifndef MATHUTILS_INCLUDE_MATHUTILS_DETAIL_BASICMATH_H_
#define MATHUTILS_INCLUDE_MATHUTILS_DETAIL_BASICMATH_H_

#ifndef GPUCA_GPUCODE_DEVICE
#include <cmath>
#include <tuple>
#endif
#include "GPUCommonArray.h"
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
GPUhdi() T copysign(T x, T y)
{
  return o2::gpu::GPUCommonMath::Copysign(x, y);
}

template <>
GPUhdi() double copysign(double x, double y)
{
  return std::copysign(x, y);
}

template <class T>
GPUhdi() T min(const T x, const T y)
{
  return o2::gpu::GPUCommonMath::Min(x, y);
};

template <class T>
GPUhdi() T max(const T x, const T y)
{
  return o2::gpu::GPUCommonMath::Max(x, y);
};

template <class T>
GPUhdi() T sqrt(T x)
{
  return o2::gpu::GPUCommonMath::Sqrt(x);
};

template <>
GPUhdi() double sqrt(double x)
{
  return std::sqrt(x);
};

template <class T>
GPUhdi() T abs(T x)
{
  return o2::gpu::GPUCommonMath::Abs(x);
};

GPUhdi() double abs(double x)
{
  return std::abs(x);
};

template <class T>
GPUdi() int nint(T x)
{
  return o2::gpu::GPUCommonMath::Nint(x);
};

template <>
GPUdi() int nint(double x)
{
  return std::nearbyint(x);
};

template <class T>
GPUdi() bool finite(T x)
{
  return o2::gpu::GPUCommonMath::Finite(x);
}

template <>
GPUdi() bool finite(double x)
{
  return std::isfinite(x);
}

GPUdi() unsigned int clz(unsigned int val)
{
  return o2::gpu::GPUCommonMath::Clz(val);
};

GPUdi() unsigned int popcount(unsigned int val)
{
  return o2::gpu::GPUCommonMath::Popcount(val);
};

template <class T>
GPUdi() T log(T x)
{
  return o2::gpu::GPUCommonMath::Log(x);
};

template <>
GPUdi() double log(double x)
{
  return std::log(x);
};

} // namespace detail
} // namespace math_utils
} // namespace o2

#endif /* MATHUTILS_INCLUDE_MATHUTILS_DETAIL_BASICMATH_H_ */