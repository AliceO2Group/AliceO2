// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUCommonMath.h
/// \author David Rohr, Sergey Gorbunov

#ifndef GPUCOMMONMATH_H
#define GPUCOMMONMATH_H

#include "GPUCommonDef.h"

#if defined(__CUDACC__) && !defined(__clang__)
#include <sm_20_atomic_functions.h>
#endif

#if !defined(__OPENCL__)
#include <cmath>
#include <algorithm>
#endif

#if !defined(__OPENCL__) || defined(__OPENCLCPP__)
namespace GPUCA_NAMESPACE
{
namespace gpu
{
#endif

class GPUCommonMath
{
 public:
  GPUhdni() static float2 MakeFloat2(float x, float y); // TODO: Find better appraoch that is constexpr

  template <class T>
  GPUhd() static T Min(const T x, const T y);
  template <class T>
  GPUhd() static T Max(const T x, const T y);
  template <class T, class S, class R>
  GPUhd() static T MinWithRef(T x, T y, S refX, S refY, R& r);
  template <class T, class S, class R>
  GPUhd() static T MaxWithRef(T x, T y, S refX, S refY, R& r);
  template <class T, class S, class R>
  GPUhd() static T MaxWithRef(T x, T y, T z, T w, S refX, S refY, S refZ, S refW, R& r);
  GPUhdni() static float Sqrt(float x);
  GPUhdni() static float FastInvSqrt(float x);
  template <class T>
  GPUhd() static T Abs(T x);
  GPUhdni() static float ASin(float x);
  GPUhdni() static float ATan(float x);
  GPUhdni() static float ATan2(float y, float x);
  GPUhdni() static float Sin(float x);
  GPUhdni() static float Cos(float x);
  GPUhdni() static void SinCos(float x, float& s, float& c);
  GPUhdni() static void SinCos(double x, double& s, double& c);
  GPUhdni() static float Tan(float x);
  GPUhdni() static float Copysign(float x, float y);
  GPUhdni() static float TwoPi() { return 6.28319f; }
  GPUhdni() static float Pi() { return 3.1415926535897f; }
  GPUhdni() static int Nint(float x);
  GPUhdni() static bool Finite(float x);
  GPUhdni() static unsigned int Clz(unsigned int val);
  GPUhdni() static unsigned int Popcount(unsigned int val);

  GPUhdni() static float Log(float x);
  template <class T>
  GPUdi() static T AtomicExch(GPUglobalref() GPUgeneric() GPUAtomic(T) * addr, T val)
  {
    return GPUCommonMath::AtomicExchInt(addr, val);
  }
  template <class T>
  GPUdi() static T AtomicAdd(GPUglobalref() GPUgeneric() GPUAtomic(T) * addr, T val)
  {
    return GPUCommonMath::AtomicAddInt(addr, val);
  }
  template <class T>
  GPUdi() static void AtomicMax(GPUglobalref() GPUgeneric() GPUAtomic(T) * addr, T val)
  {
    GPUCommonMath::AtomicMaxInt(addr, val);
  }
  template <class T>
  GPUdi() static void AtomicMin(GPUglobalref() GPUgeneric() GPUAtomic(T) * addr, T val)
  {
    GPUCommonMath::AtomicMinInt(addr, val);
  }
  template <class T>
  GPUdi() static T AtomicExchShared(GPUsharedref() GPUgeneric() GPUAtomic(T) * addr, T val)
  {
#ifdef GPUCA_GPUCODE_DEVICE
    return GPUCommonMath::AtomicExchInt(addr, val);
#else
    T retVal = *addr;
    *addr = val;
    return retVal;
#endif
  }
  template <class T>
  GPUdi() static T AtomicAddShared(GPUsharedref() GPUgeneric() GPUAtomic(T) * addr, T val)
  {
#ifdef GPUCA_GPUCODE_DEVICE
    return GPUCommonMath::AtomicAddInt(addr, val);
#else
    T retVal = *addr;
    *addr += val;
    return retVal;
#endif
  }
  template <class T>
  GPUdi() static void AtomicMaxShared(GPUsharedref() GPUgeneric() GPUAtomic(T) * addr, T val)
  {
#ifdef GPUCA_GPUCODE_DEVICE
    GPUCommonMath::AtomicMaxInt(addr, val);
#else
    *addr = std::max(*addr, val);
#endif
  }
  template <class T>
  GPUdi() static void AtomicMinShared(GPUsharedref() GPUgeneric() GPUAtomic(T) * addr, T val)
  {
#ifdef GPUCA_GPUCODE_DEVICE
    GPUCommonMath::AtomicMinInt(addr, val);
#else
    *addr = std::min(*addr, val);
#endif
  }
  GPUd() static int Mul24(int a, int b);
  GPUd() static float FMulRZ(float a, float b);

  template <int I, class T>
  GPUd() CONSTEXPR17 static T nextMultipleOf(T val);

 private:
  template <class S, class T>
  GPUd() static unsigned int AtomicExchInt(S* addr, T val);
  template <class S, class T>
  GPUd() static unsigned int AtomicAddInt(S* addr, T val);
  template <class S, class T>
  GPUd() static void AtomicMaxInt(S* addr, T val);
  template <class S, class T>
  GPUd() static void AtomicMinInt(S* addr, T val);
};

typedef GPUCommonMath CAMath;

#if defined(GPUCA_GPUCODE_DEVICE) && (defined(__CUDACC__) || defined(__HIPCC__)) // clang-format off
    #define CHOICE(c1, c2, c3) (c2) // Select second option for CUDA and HIP
#elif defined(GPUCA_GPUCODE_DEVICE) && defined (__OPENCL__)
    #define CHOICE(c1, c2, c3) (c3) // Select third option for OpenCL
#else
    #define CHOICE(c1, c2, c3) (c1) //Select first option for Host
#endif // clang-format on

template <int I, class T>
GPUdi() CONSTEXPR17 T GPUCommonMath::nextMultipleOf(T val)
{
  CONSTEXPRIF(I & (I - 1))
  {
    T tmp = val % I;
    if (tmp)
      val += I - tmp;
    return val;
  }
  else
  {
    return (val + I - 1) & ~(T)(I - 1);
  }
}

GPUhdi() float2 GPUCommonMath::MakeFloat2(float x, float y)
{
#if !defined(GPUCA_GPUCODE) || defined(__OPENCL__) || defined(__OPENCL_HOST__)
  float2 ret = {x, y};
  return ret;
#else
  return make_float2(x, y);
#endif // GPUCA_GPUCODE
}

GPUhdi() int GPUCommonMath::Nint(float x)
{
  int i;
  if (x >= 0) {
    i = int(x + 0.5f);
    if (x + 0.5f == float(i) && i & 1)
      i--;
  } else {
    i = int(x - 0.5f);
    if (x - 0.5f == float(i) && i & 1)
      i++;
  }
  return i;
}

GPUhdi() bool GPUCommonMath::Finite(float x) { return CHOICE(std::isfinite(x), true, true); }

GPUhdi() float GPUCommonMath::ATan(float x) { return CHOICE(atanf(x), atanf(x), atan(x)); }

GPUhdi() float GPUCommonMath::ATan2(float y, float x) { return CHOICE(atan2f(y, x), atan2f(y, x), atan2(y, x)); }

GPUhdi() float GPUCommonMath::Sin(float x) { return CHOICE(sinf(x), sinf(x), sin(x)); }

GPUhdi() float GPUCommonMath::Cos(float x) { return CHOICE(cosf(x), cosf(x), cos(x)); }

GPUhdi() void GPUCommonMath::SinCos(float x, float& s, float& c)
{
#if !defined(GPUCA_GPUCODE_DEVICE) && defined(__APPLE__)
  __sincosf(x, &s, &c);
#elif !defined(GPUCA_GPUCODE_DEVICE) && defined(__GNU_SOURCE__)
  sincosf(x, &s, &c);
#else
  CHOICE({s = sin(x); c = cos(x); }, sincosf(x, &s, &c), s = sincos(x, &c));
#endif
}

GPUhdi() void GPUCommonMath::SinCos(double x, double& s, double& c)
{
#if !defined(GPUCA_GPUCODE_DEVICE) && defined(__APPLE__)
  __sincos(x, &s, &c);
#elif !defined(GPUCA_GPUCODE_DEVICE) && defined(__GNU_SOURCE__)
  sincos(x, &s, &c);
#else
  CHOICE({s = sin(x); c = cos(x); }, sincos(x, &s, &c), s = sincos(x, &c));
#endif
}

GPUhdi() float GPUCommonMath::Tan(float x) { return CHOICE(tanf(x), tanf(x), tan(x)); }

GPUhdi() unsigned int GPUCommonMath::Clz(unsigned int x)
{
#if (defined(__GNUC__) || defined(__clang__) || defined(__CUDACC__) || defined(__HIPCC__)) && (!defined(__OPENCL__) || defined(__OPENCLCPP__))
  return x == 0 ? 32 : CHOICE(__builtin_clz(x), __clz(x), __builtin_clz(x)); // use builtin if available
#else
  for (int i = 31; i >= 0; i--) {
    if (x & (1 << i)) {
      return (31 - i);
    }
  }
  return 32;
#endif
}

GPUhdi() unsigned int GPUCommonMath::Popcount(unsigned int x)
{
#if (defined(__GNUC__) || defined(__clang__) || defined(__CUDACC__) || defined(__HIPCC__)) && (!defined(__OPENCL__) /*|| defined(__OPENCLCPP__)*/) // TODO: remove OPENCLCPP workaround when reported SPIR-V bug is fixed
  return CHOICE(__builtin_popcount(x), __popc(x), __builtin_popcount(x));                                                                          // use builtin if available
#else
  unsigned int retVal = 0;
  for (int i = 0; i < 32; i++) {
    if (x & (1 << i)) {
      retVal++;
    }
  }
  return retVal;
#endif
}

template <class T>
GPUhdi() T GPUCommonMath::Min(const T x, const T y)
{
  return CHOICE(std::min(x, y), std::min(x, y), (x < y ? x : y));
}

template <class T>
GPUhdi() T GPUCommonMath::Max(const T x, const T y)
{
  return CHOICE(std::max(x, y), std::max(x, y), (x > y ? x : y));
}

template <class T, class S, class R>
GPUhdi() T GPUCommonMath::MinWithRef(T x, T y, S refX, S refY, R& r)
{
  if (x < y) {
    r = refX;
    return x;
  }
  r = refY;
  return y;
}

template <class T, class S, class R>
GPUhdi() T GPUCommonMath::MaxWithRef(T x, T y, S refX, S refY, R& r)
{
  if (x > y) {
    r = refX;
    return x;
  }
  r = refY;
  return y;
}

template <class T, class S, class R>
GPUhdi() T GPUCommonMath::MaxWithRef(T x, T y, T z, T w, S refX, S refY, S refZ, S refW, R& r)
{
  T retVal = x;
  S retRef = refX;
  if (y > retVal) {
    retVal = y;
    retRef = refY;
  }
  if (z > retVal) {
    retVal = z;
    retRef = refZ;
  }
  if (w > retVal) {
    retVal = w;
    retRef = refW;
  }
  r = retRef;
  return retVal;
}

GPUhdi() float GPUCommonMath::Sqrt(float x) { return CHOICE(sqrtf(x), sqrtf(x), sqrt(x)); }

GPUhdi() float GPUCommonMath::FastInvSqrt(float _x)
{
  // the function calculates fast inverse sqrt
  union {
    float f;
    int i;
  } x = {_x};
  const float xhalf = 0.5f * x.f;
  x.i = 0x5f3759df - (x.i >> 1);
  x.f = x.f * (1.5f - xhalf * x.f * x.f);
  return x.f;
}

template <>
GPUhdi() float GPUCommonMath::Abs<float>(float x)
{
  return CHOICE(fabsf(x), fabsf(x), fabs(x));
}

#if !defined(__OPENCL__) || defined(cl_khr_fp64)
template <>
GPUhdi() double GPUCommonMath::Abs<double>(double x)
{
  return CHOICE(fabs(x), fabs(x), fabs(x));
}
#endif

template <>
GPUhdi() int GPUCommonMath::Abs<int>(int x)
{
  return CHOICE(abs(x), abs(x), abs(x));
}

GPUhdi() float GPUCommonMath::ASin(float x) { return CHOICE(asinf(x), asinf(x), asin(x)); }

GPUhdi() float GPUCommonMath::Log(float x) { return CHOICE(logf(x), logf(x), log(x)); }

GPUhdi() float GPUCommonMath::Copysign(float x, float y)
{
#if defined(__OPENCLCPP__)
  return copysign(x, y);
#elif defined(GPUCA_GPUCODE) && !defined(__OPENCL__)
  return copysignf(x, y);
#elif defined(__cplusplus) && __cplusplus >= 201103L
  return std::copysignf(x, y);
#else
  x = GPUCommonMath::Abs(x);
  return (y >= 0) ? x : -x;
#endif // GPUCA_GPUCODE
}

#ifndef GPUCA_GPUCODE
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-value" // GCC BUG in omp atomic capture gives false warning
#endif

template <class S, class T>
GPUdi() unsigned int GPUCommonMath::AtomicExchInt(S* addr, T val)
{
#if defined(GPUCA_GPUCODE) && defined(__OPENCLCPP__) && (!defined(__clang__) || defined(GPUCA_OPENCL_CPP_CLANG_C11_ATOMICS))
  return ::atomic_exchange(addr, val);
#elif defined(GPUCA_GPUCODE) && defined(__OPENCL__)
  return ::atomic_xchg(addr, val);
#elif defined(GPUCA_GPUCODE) && (defined(__CUDACC__) || defined(__HIPCC__))
  return ::atomicExch(addr, val);
#else
  unsigned int old;
#ifdef WITH_OPENMP
#pragma omp atomic capture
#endif
  {
    old = *addr;
    *addr = val;
  }
  return old;
#endif // GPUCA_GPUCODE
}

template <class S, class T>
GPUdi() unsigned int GPUCommonMath::AtomicAddInt(S* addr, T val)
{
#if defined(GPUCA_GPUCODE) && defined(__OPENCLCPP__) && (!defined(__clang__) || defined(GPUCA_OPENCL_CPP_CLANG_C11_ATOMICS))
  return ::atomic_fetch_add(addr, val);
#elif defined(GPUCA_GPUCODE) && defined(__OPENCL__)
  return ::atomic_add(addr, val);
#elif defined(GPUCA_GPUCODE) && (defined(__CUDACC__) || defined(__HIPCC__))
  return ::atomicAdd(addr, val);
#else
  unsigned int old;
#ifdef WITH_OPENMP
#pragma omp atomic capture
#endif
  {
    old = *addr;
    *addr += val;
  }
  return old;
#endif // GPUCA_GPUCODE
}

template <class S, class T>
GPUdi() void GPUCommonMath::AtomicMaxInt(S* addr, T val)
{
#if defined(GPUCA_GPUCODE) && defined(__OPENCLCPP__) && (!defined(__clang__) || defined(GPUCA_OPENCL_CPP_CLANG_C11_ATOMICS))
  ::atomic_fetch_max(addr, val);
#elif defined(GPUCA_GPUCODE) && defined(__OPENCL__)
  ::atomic_max(addr, val);
#elif defined(GPUCA_GPUCODE) && (defined(__CUDACC__) || defined(__HIPCC__))
  ::atomicMax(addr, val);
#elif defined(WITH_OPENMP)
  while (*addr < val) {
    AtomicExch(addr, val);
  }
#else
  if (*addr < val) {
    *addr = val;
  }
#endif // GPUCA_GPUCODE
}

template <class S, class T>
GPUdi() void GPUCommonMath::AtomicMinInt(S* addr, T val)
{
#if defined(GPUCA_GPUCODE) && defined(__OPENCLCPP__) && (!defined(__clang__) || defined(GPUCA_OPENCL_CPP_CLANG_C11_ATOMICS))
  ::atomic_fetch_min(addr, val);
#elif defined(GPUCA_GPUCODE) && defined(__OPENCL__)
  ::atomic_min(addr, val);
#elif defined(GPUCA_GPUCODE) && (defined(__CUDACC__) || defined(__HIPCC__))
  ::atomicMin(addr, val);
#elif defined(WITH_OPENMP)
  while (*addr > val) {
    AtomicExch(addr, val);
  }
#else
  if (*addr > val) {
    *addr = val;
  }
#endif // GPUCA_GPUCODE
}

#ifndef GPUCA_GPUCODE
#pragma GCC diagnostic pop
#endif

#undef CHOICE

#if !defined(__OPENCL__) || defined(__OPENCLCPP__)
}
}
#endif

#endif // GPUCOMMONMATH_H
