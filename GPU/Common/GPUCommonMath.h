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

/// \file GPUCommonMath.h
/// \author David Rohr, Sergey Gorbunov

#ifndef GPUCOMMONMATH_H
#define GPUCOMMONMATH_H

#include "GPUCommonDef.h"

#if defined(__CUDACC__) && !defined(__clang__) && !defined(GPUCA_GPUCODE_COMPILEKERNELS) && !defined(GPUCA_GPUCODE_HOSTONLY)
#include <sm_20_atomic_functions.h>
#endif

#if !defined(GPUCA_GPUCODE_DEVICE)
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
  GPUd() static float2 MakeFloat2(float x, float y); // TODO: Find better appraoch that is constexpr

  template <class T>
  GPUhd() static T Min(const T x, const T y);
  template <class T>
  GPUhd() static T Max(const T x, const T y);
  template <class T, class S, class R>
  GPUd() static T MinWithRef(T x, T y, S refX, S refY, R& r);
  template <class T, class S, class R>
  GPUd() static T MaxWithRef(T x, T y, S refX, S refY, R& r);
  template <class T, class S, class R>
  GPUd() static T MaxWithRef(T x, T y, T z, T w, S refX, S refY, S refZ, S refW, R& r);
  template <class T>
  GPUdi() static T Clamp(const T v, const T lo, const T hi)
  {
    return Max(lo, Min(v, hi));
  }
  GPUhdni() static float Sqrt(float x);
  GPUd() static float FastInvSqrt(float x);
  template <class T>
  GPUhd() static T Abs(T x);
  GPUd() static float ASin(float x);
  GPUd() static float ACos(float x);
  GPUd() static float ATan(float x);
  GPUhd() static float ATan2(float y, float x);
  GPUd() static float Sin(float x);
  GPUd() static float Cos(float x);
  GPUhdni() static void SinCos(float x, float& s, float& c);
  GPUhdni() static void SinCosd(double x, double& s, double& c);
  GPUd() static float Tan(float x);
  GPUd() static float Pow(float x, float y);
  GPUhdni() static float Copysign(float x, float y);
  GPUd() static float TwoPi() { return 6.2831853f; }
  GPUd() static float Pi() { return 3.1415927f; }
  GPUd() static int Nint(float x);
  GPUd() static float Modf(float x, float y);
  GPUd() static bool Finite(float x);
  GPUd() static unsigned int Clz(unsigned int val);
  GPUd() static unsigned int Popcount(unsigned int val);

  GPUhdni() static float Hypot(float x, float y);
  GPUhdni() static float Hypot(float x, float y, float z);
  GPUhdni() static float Hypot(float x, float y, float z, float w);

  GPUd() static float Log(float x);
  GPUd() static float Exp(float x);

  template <class T>
  GPUdi() static T AtomicExch(GPUglobalref() GPUgeneric() GPUAtomic(T) * addr, T val)
  {
    return GPUCommonMath::AtomicExchInt(addr, val);
  }

  template <class T>
  GPUdi() static T AtomicCAS(GPUglobalref() GPUgeneric() GPUAtomic(T) * addr, T cmp, T val)
  {
    return GPUCommonMath::AtomicCASInt(addr, cmp, val);
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
  GPUd() CONSTEXPR static T nextMultipleOf(T val);

#ifdef GPUCA_NOCOMPAT
  GPUdi() static float Sum2() // Needed for legacy C++, For >=17 the below if constexpr handles the case
  {
    return 0.f;
  }

  template <typename... Args>
  GPUdi() static float Sum2(float w, Args... args)
  {
    if CONSTEXPR (sizeof...(Args) == 0) {
      return w * w;
    } else {
      return w * w + Sum2(args...);
    }
    return 0;
  }
#endif

 private:
  template <class S, class T>
  GPUd() static unsigned int AtomicExchInt(S* addr, T val);
  template <class S, class T>
  GPUd() static T AtomicCASInt(S* addr, T cmp, T val);
  template <class S, class T>
  GPUd() static unsigned int AtomicAddInt(S* addr, T val);
  template <class S, class T>
  GPUd() static void AtomicMaxInt(S* addr, T val);
  template <class S, class T>
  GPUd() static void AtomicMinInt(S* addr, T val);
};

typedef GPUCommonMath CAMath;

// CHOICE Syntax: CHOISE(Host, CUDA&HIP, OpenCL)
#if defined(GPUCA_GPUCODE_DEVICE) && (defined(__CUDACC__) || defined(__HIPCC__)) // clang-format off
    #define CHOICE(c1, c2, c3) (c2) // Select second option for CUDA and HIP
#elif defined(GPUCA_GPUCODE_DEVICE) && defined (__OPENCL__)
    #define CHOICE(c1, c2, c3) (c3) // Select third option for OpenCL
#else
    #define CHOICE(c1, c2, c3) (c1) //Select first option for Host
#endif // clang-format on

template <int I, class T>
GPUdi() CONSTEXPR T GPUCommonMath::nextMultipleOf(T val)
{
  if CONSTEXPR (I & (I - 1)) {
    T tmp = val % I;
    if (tmp) {
      val += I - tmp;
    }
    return val;
  } else {
    return (val + I - 1) & ~(T)(I - 1);
  }
  return 0; // BUG: Cuda complains about missing return value with constexpr if
}

GPUdi() float2 GPUCommonMath::MakeFloat2(float x, float y)
{
#if !defined(GPUCA_GPUCODE) || defined(__OPENCL__) || defined(__OPENCL_HOST__)
  float2 ret = {x, y};
  return ret;
#else
  return make_float2(x, y);
#endif // GPUCA_GPUCODE
}

GPUdi() int GPUCommonMath::Nint(float x)
{
  int i;
  if (x >= 0) {
    i = int(x + 0.5f);
    if (x + 0.5f == float(i) && i & 1) {
      i--;
    }
  } else {
    i = int(x - 0.5f);
    if (x - 0.5f == float(i) && i & 1) {
      i++;
    }
  }
  return i;
}

GPUdi() float GPUCommonMath::Modf(float x, float y) { return CHOICE(fmodf(x, y), fmodf(x, y), fmod(x, y)); }

GPUdi() bool GPUCommonMath::Finite(float x) { return CHOICE(std::isfinite(x), true, true); }

GPUdi() float GPUCommonMath::ATan(float x) { return CHOICE(atanf(x), atanf(x), atan(x)); }

GPUhdi() float GPUCommonMath::ATan2(float y, float x) { return CHOICE(atan2f(y, x), atan2f(y, x), atan2(y, x)); }

GPUdi() float GPUCommonMath::Sin(float x) { return CHOICE(sinf(x), sinf(x), sin(x)); }

GPUdi() float GPUCommonMath::Cos(float x) { return CHOICE(cosf(x), cosf(x), cos(x)); }

GPUdi() float GPUCommonMath::Pow(float x, float y) { return CHOICE(powf(x, y), powf(x, y), pow(x, y)); }

GPUhdi() void GPUCommonMath::SinCos(float x, float& s, float& c)
{
#if !defined(GPUCA_GPUCODE_DEVICE) && defined(__APPLE__)
  __sincosf(x, &s, &c);
#elif !defined(GPUCA_GPUCODE_DEVICE) && (defined(__GNU_SOURCE__) || defined(_GNU_SOURCE) || defined(GPUCA_GPUCODE))
  sincosf(x, &s, &c);
#else
  CHOICE((void)((s = sinf(x)) + (c = cosf(x))), sincosf(x, &s, &c), s = sincos(x, &c));
#endif
}

GPUhdi() void GPUCommonMath::SinCosd(double x, double& s, double& c)
{
#if !defined(GPUCA_GPUCODE_DEVICE) && defined(__APPLE__)
  __sincos(x, &s, &c);
#elif !defined(GPUCA_GPUCODE_DEVICE) && (defined(__GNU_SOURCE__) || defined(_GNU_SOURCE) || defined(GPUCA_GPUCODE))
  sincos(x, &s, &c);
#else
  CHOICE((void)((s = sin(x)) + (c = cos(x))), sincos(x, &s, &c), s = sincos(x, &c));
#endif
}

GPUdi() float GPUCommonMath::Tan(float x) { return CHOICE(tanf(x), tanf(x), tan(x)); }

GPUdi() unsigned int GPUCommonMath::Clz(unsigned int x)
{
#if (defined(__GNUC__) || defined(__clang__) || defined(__CUDACC__) || defined(__HIPCC__)) && (!defined(__OPENCL__) || defined(__OPENCLCPP__))
  return x == 0 ? 32 : CHOICE(__builtin_clz(x), __clz(x), __builtin_clz(x)); // use builtin if available
#else
  for (int i = 31; i >= 0; i--) {
    if (x & (1u << i)) {
      return (31 - i);
    }
  }
  return 32;
#endif
}

GPUdi() unsigned int GPUCommonMath::Popcount(unsigned int x)
{
#if (defined(__GNUC__) || defined(__clang__) || defined(__CUDACC__) || defined(__HIPCC__)) && (!defined(__OPENCL__) /*|| defined(__OPENCLCPP__)*/) // TODO: remove OPENCLCPP workaround when reported SPIR-V bug is fixed
  // use builtin if available
  return CHOICE(__builtin_popcount(x), __popc(x), __builtin_popcount(x));
#else
  x = x - ((x >> 1) & 0x55555555);
  x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
  return (((x + (x >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
#endif
}

GPUhdi() float GPUCommonMath::Hypot(float x, float y)
{
  return Sqrt(x * x + y * y);
}

GPUhdi() float GPUCommonMath::Hypot(float x, float y, float z)
{
  return Sqrt(x * x + y * y + z * z);
}

GPUhdi() float GPUCommonMath::Hypot(float x, float y, float z, float w)
{
  return Sqrt(x * x + y * y + z * z + w * w);
}

template <class T>
GPUhdi() T GPUCommonMath::Min(const T x, const T y)
{
  return CHOICE(std::min(x, y), min(x, y), (x < y ? x : y));
}

template <class T>
GPUhdi() T GPUCommonMath::Max(const T x, const T y)
{
  return CHOICE(std::max(x, y), max(x, y), (x > y ? x : y));
}

template <class T, class S, class R>
GPUdi() T GPUCommonMath::MinWithRef(T x, T y, S refX, S refY, R& r)
{
  if (x < y) {
    r = refX;
    return x;
  }
  r = refY;
  return y;
}

template <class T, class S, class R>
GPUdi() T GPUCommonMath::MaxWithRef(T x, T y, S refX, S refY, R& r)
{
  if (x > y) {
    r = refX;
    return x;
  }
  r = refY;
  return y;
}

template <class T, class S, class R>
GPUdi() T GPUCommonMath::MaxWithRef(T x, T y, T z, T w, S refX, S refY, S refZ, S refW, R& r)
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

GPUdi() float GPUCommonMath::FastInvSqrt(float _x)
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

GPUdi() float GPUCommonMath::ASin(float x) { return CHOICE(asinf(x), asinf(x), asin(x)); }

GPUdi() float GPUCommonMath::ACos(float x) { return CHOICE(acosf(x), acosf(x), acos(x)); }

GPUdi() float GPUCommonMath::Log(float x) { return CHOICE(logf(x), logf(x), log(x)); }
GPUdi() float GPUCommonMath::Exp(float x) { return CHOICE(expf(x), expf(x), exp(x)); }

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

template <class S, class T>
GPUdi() unsigned int GPUCommonMath::AtomicExchInt(S* addr, T val)
{
#if defined(GPUCA_GPUCODE) && defined(__OPENCLCPP__) && (!defined(__clang__) || defined(GPUCA_OPENCL_CPP_CLANG_C11_ATOMICS))
  return ::atomic_exchange(addr, val);
#elif defined(GPUCA_GPUCODE) && defined(__OPENCL__)
  return ::atomic_xchg(addr, val);
#elif defined(GPUCA_GPUCODE) && (defined(__CUDACC__) || defined(__HIPCC__))
  return ::atomicExch(addr, val);
#elif defined(WITH_OPENMP)
  unsigned int old;
  __atomic_exchange(addr, &val, &old, __ATOMIC_SEQ_CST);
  return old;
#else
  unsigned int old = *addr;
  *addr = val;
  return old;
#endif
}

template <class S, class T>
GPUdi() T GPUCommonMath::AtomicCASInt(S* addr, T cmp, T val)
{
#if defined(GPUCA_GPUCODE) && defined(__OPENCLCPP__) && (!defined(__clang__) || defined(GPUCA_OPENCL_CPP_CLANG_C11_ATOMICS))
  return ::atomic_compare_exchange(addr, cmp, val);
#elif defined(GPUCA_GPUCODE) && defined(__OPENCL__)
  return ::atomic_cmpxchg(addr, cmp, val);
#elif defined(GPUCA_GPUCODE) && (defined(__CUDACC__) || defined(__HIPCC__))
  return ::atomicCAS(addr, cmp, val);
#elif defined(WITH_OPENMP)
  __atomic_compare_exchange(addr, &cmp, &val, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
  return cmp;
#else
  T old = *addr;
  *addr = (old == cmp) ? val : old;
  return old;
#endif
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
#elif defined(WITH_OPENMP)
  return __atomic_add_fetch(addr, val, __ATOMIC_SEQ_CST) - val;
#else
  unsigned int old = *addr;
  *addr += val;
  return old;
#endif
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

#if (defined(__CUDACC__) || defined(__HIPCC__)) && !defined(__ROOTCINT__) && !defined(G__ROOT)
#define GPUCA_HAVE_ATOMIC_MINMAX_FLOAT
template <>
GPUdii() void GPUCommonMath::AtomicMaxInt(GPUglobalref() GPUgeneric() GPUAtomic(float) * addr, float val)
{
  if (val == -0.f) {
    val = 0.f;
  }
  if (val >= 0) {
    AtomicMaxInt((GPUAtomic(int)*)addr, __float_as_int(val));
  } else {
    AtomicMinInt((GPUAtomic(unsigned int)*)addr, __float_as_uint(val));
  }
}
template <>
GPUdii() void GPUCommonMath::AtomicMinInt(GPUglobalref() GPUgeneric() GPUAtomic(float) * addr, float val)
{
  if (val == -0.f) {
    val = 0.f;
  }
  if (val >= 0) {
    AtomicMinInt((GPUAtomic(int)*)addr, __float_as_int(val));
  } else {
    AtomicMaxInt((GPUAtomic(unsigned int)*)addr, __float_as_uint(val));
  }
}
#endif

#undef CHOICE

#if !defined(__OPENCL__) || defined(__OPENCLCPP__)
}
}
#endif

#endif // GPUCOMMONMATH_H
