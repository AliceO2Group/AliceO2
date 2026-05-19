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
#include <atomic>
#include <limits>
#include <cstring>
#endif

#if !defined(GPUCA_GPUCODE_COMPILEKERNELS) && (!defined(GPUCA_GPUCODE_DEVICE) || defined(__CUDACC__) || defined(__HIPCC__))
#include <cstdint>
#endif

// GPUCA_CHOICE Syntax: GPUCA_CHOICE(Host, CUDA&HIP, OpenCL)
#if defined(GPUCA_GPUCODE_DEVICE) && (defined(__CUDACC__) || defined(__HIPCC__)) // clang-format off
    #define GPUCA_CHOICE(c1, c2, c3) (c2) // Select second option for CUDA and HIP
#elif defined(GPUCA_GPUCODE_DEVICE) && defined (__OPENCL__)
    #define GPUCA_CHOICE(c1, c2, c3) (c3) // Select third option for OpenCL
#else
    #define GPUCA_CHOICE(c1, c2, c3) (c1) // Select first option for Host
#endif // clang-format on

namespace o2::gpu
{

class GPUCommonMath
{
 public:
  GPUd() static float2 MakeFloat2(float x, float y); // TODO: Find better appraoch that is constexpr

  template <class T>
  GPUhd() constexpr static T Min(const T x, const T y)
  {
    return GPUCA_CHOICE(std::min(x, y), min(x, y), min(x, y));
  }
  template <class T>
  GPUhd() constexpr static T Max(const T x, const T y)
  {
    return GPUCA_CHOICE(std::max(x, y), max(x, y), max(x, y));
  }
  template <class T, class S, class R>
  GPUd() static T MinWithRef(T x, T y, S refX, S refY, R& r);
  template <class T, class S, class R>
  GPUd() static T MaxWithRef(T x, T y, S refX, S refY, R& r);
  template <class T, class S, class R>
  GPUd() static T MaxWithRef(T x, T y, T z, T w, S refX, S refY, S refZ, S refW, R& r);
  template <class T>
  GPUdi() constexpr static T Clamp(const T v, const T lo, const T hi)
  {
    return Max(lo, Min(v, hi));
  }
  GPUhdni() constexpr static float Sqrt(float x);
  GPUd() static float InvSqrt(float x);
  template <class T>
  GPUdi() constexpr static T Square(T x)
  {
    return x * x;
  }
  template <class T>
  GPUhd() constexpr static T Abs(T x);
  GPUd() constexpr static float ASin(float x);
  GPUd() constexpr static float ACos(float x);
  GPUd() constexpr static float ATan(float x);
  GPUhd() constexpr static float ATan2(float y, float x);
  GPUd() constexpr static float Sin(float x);
  GPUd() constexpr static float Cos(float x);
  GPUhdni() static void SinCos(float x, float& s, float& c);
  GPUhdni() static void SinCosd(double x, double& s, double& c);
  GPUd() constexpr static float Tan(float x);
  GPUd() constexpr static float Pow(float x, float y);
  GPUd() constexpr static float Log(float x);
  GPUd() constexpr static float Exp(float x);
  GPUhdni() constexpr static float Copysign(float x, float y) { return GPUCA_CHOICE(std::copysignf(x, y), copysignf(x, y), copysign(x, y)); }
  GPUd() constexpr static float TwoPi() { return 6.2831853f; }
  GPUd() constexpr static float Pi() { return 3.1415927f; }
  GPUd() constexpr static float Round(float x);
  GPUd() constexpr static float Floor(float x) { return GPUCA_CHOICE(floorf(x), floorf(x), floor(x)); }
  GPUd() static uint32_t Float2UIntReint(const float& x);
  GPUd() constexpr static uint32_t Float2UIntRn(float x) { return (uint32_t)(int32_t)(x + 0.5f); }
  GPUd() constexpr static int32_t Float2IntRn(float x);
  GPUd() constexpr static float Modf(float x, float y);
  GPUhdi() static float Remainderf(float x, float y);
  GPUd() constexpr static bool Finite(float x);
  GPUd() constexpr static bool IsNaN(float x);
#ifndef __FAST_MATH__
  GPUd() constexpr static float QuietNaN() { return GPUCA_CHOICE(std::numeric_limits<float>::quiet_NaN(), __builtin_nanf(""), nan(0u)); }
#endif
  GPUd() constexpr static uint32_t Clz(uint32_t val);
  GPUd() constexpr static uint32_t Ctz(uint32_t val);
  GPUd() constexpr static uint32_t Popcount(uint32_t val);

  GPUd() static void memcpy(void* dst, const void* src, size_t size);

  GPUhdi() constexpr static float Hypot(float x, float y) { return Sqrt(x * x + y * y); }
  GPUhdi() constexpr static float Hypot(float x, float y, float z) { return Sqrt(x * x + y * y + z * z); }
  GPUhdi() constexpr static float Hypot(float x, float y, float z, float w) { return Sqrt(x * x + y * y + z * z + w * w); }

  template <typename T>
  GPUhd() constexpr static void Swap(T& a, T& b);

  template <class T>
  GPUdi() static T AtomicExch(GPUglobalref() GPUgeneric() GPUAtomic(T) * addr, T val)
  {
    return GPUCommonMath::AtomicExchInternal(addr, val);
  }

  template <class T>
  GPUdi() static bool AtomicCAS(GPUglobalref() GPUgeneric() GPUAtomic(T) * addr, T cmp, T val)
  {
    return GPUCommonMath::AtomicCASInternal(addr, cmp, val);
  }

  template <class T>
  GPUdi() static T AtomicAdd(GPUglobalref() GPUgeneric() GPUAtomic(T) * addr, T val)
  {
    return GPUCommonMath::AtomicAddInternal(addr, val);
  }
  template <class T>
  GPUdi() static void AtomicMax(GPUglobalref() GPUgeneric() GPUAtomic(T) * addr, T val)
  {
    GPUCommonMath::AtomicMaxInternal(addr, val);
  }
  template <class T>
  GPUdi() static void AtomicMin(GPUglobalref() GPUgeneric() GPUAtomic(T) * addr, T val)
  {
    GPUCommonMath::AtomicMinInternal(addr, val);
  }
  template <class T>
  GPUdi() static T AtomicExchShared(GPUsharedref() GPUgeneric() GPUAtomic(T) * addr, T val)
  {
    return GPUCommonMath::AtomicExchInternal(addr, val);
  }
  template <class T>
  GPUdi() static T AtomicAddShared(GPUsharedref() GPUgeneric() GPUAtomic(T) * addr, T val)
  {
    return GPUCommonMath::AtomicAddInternal(addr, val);
  }
  template <class T>
  GPUdi() static void AtomicMaxShared(GPUsharedref() GPUgeneric() GPUAtomic(T) * addr, T val)
  {
    GPUCommonMath::AtomicMaxInternal(addr, val);
  }
  template <class T>
  GPUdi() static void AtomicMinShared(GPUsharedref() GPUgeneric() GPUAtomic(T) * addr, T val)
  {
    GPUCommonMath::AtomicMinInternal(addr, val);
  }
  GPUd() constexpr static int32_t Mul24(int32_t a, int32_t b);
  GPUd() constexpr static float FMulRZ(float a, float b);

  template <int32_t I, class T>
  GPUd() constexpr static T nextMultipleOf(T val);

  template <typename... Args>
  GPUhdni() constexpr static float Sum2(float w, Args... args);

 private:
  template <class S, class T>
  GPUd() static uint32_t AtomicExchInternal(S* addr, T val);
  template <class S, class T>
  GPUd() static bool AtomicCASInternal(S* addr, T cmp, T val);
  template <class S, class T>
  GPUd() static uint32_t AtomicAddInternal(S* addr, T val);
  template <class S, class T>
  GPUd() static void AtomicMaxInternal(S* addr, T val);
  template <class S, class T>
  GPUd() static void AtomicMinInternal(S* addr, T val);
};

typedef GPUCommonMath CAMath;

template <typename... Args>
GPUhdi() constexpr float GPUCommonMath::Sum2(float w, Args... args)
{
  if constexpr (sizeof...(Args) == 0) {
    return w * w;
  } else {
    return w * w + Sum2(args...);
  }
  return 0;
}

GPUdi() void GPUCommonMath::memcpy(void* dst, const void* src, size_t size)
{
#ifndef GPUCA_GPUCODE_DEVICE
  std::memcpy(dst, src, size);
#elif defined(__CUDACC__) || defined(__HIPCC__)
  ::memcpy(dst, src, size);
#elif defined(__clang__) || defined(__GNUC__) || defined(__GNUG__)
  __builtin_memcpy(dst, src, size);
#else
  char* d = (char*)dst;
  const char* s = (const char*)src;
  for (size_t i = 0; i < size; i++) {
    d[i] = s[i];
  }
#endif
}

template <int32_t I, class T>
GPUdi() constexpr T GPUCommonMath::nextMultipleOf(T val)
{
  if constexpr (I & (I - 1)) {
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

GPUdi() constexpr float GPUCommonMath::Modf(float x, float y) { return GPUCA_CHOICE(fmodf(x, y), fmodf(x, y), fmod(x, y)); }
GPUhdi() float GPUCommonMath::Remainderf(float x, float y) { return GPUCA_CHOICE(std::remainderf(x, y), remainderf(x, y), remainder(x, y)); }

GPUdi() uint32_t GPUCommonMath::Float2UIntReint(const float& x)
{
#if defined(GPUCA_GPUCODE_DEVICE) && (defined(__CUDACC__) || defined(__HIPCC__))
  return __float_as_uint(x);
#elif defined(GPUCA_GPUCODE_DEVICE) && defined(__OPENCL__)
  return as_uint(x);
#else
  return reinterpret_cast<const uint32_t&>(x);
#endif
}

GPUCA_DETERMINISTIC_CODE( // clang-format off
GPUdi() constexpr float GPUCommonMath::Round(float x) { return GPUCA_CHOICE(roundf(x), roundf(x), round(x)); }
GPUdi() constexpr int32_t GPUCommonMath::Float2IntRn(float x) { return (int32_t)Round(x); }
GPUhdi() constexpr float GPUCommonMath::Sqrt(float x) { return GPUCA_CHOICE(sqrtf(x), (float)sqrt((double)x), sqrt(x)); }
GPUdi() constexpr float GPUCommonMath::ATan(float x) { return GPUCA_CHOICE((float)atan((double)x), (float)atan((double)x), atan(x)); }
GPUhdi() constexpr float GPUCommonMath::ATan2(float y, float x) { return GPUCA_CHOICE((float)atan2((double)y, (double)x), (float)atan2((double)y, (double)x), atan2(y, x)); }
GPUdi() constexpr float GPUCommonMath::Sin(float x) { return GPUCA_CHOICE((float)sin((double)x), (float)sin((double)x), sin(x)); }
GPUdi() constexpr float GPUCommonMath::Cos(float x) { return GPUCA_CHOICE((float)cos((double)x), (float)cos((double)x), cos(x)); }
GPUdi() constexpr float GPUCommonMath::Tan(float x) { return GPUCA_CHOICE((float)tanf((double)x), (float)tanf((double)x), tan(x)); }
GPUdi() constexpr float GPUCommonMath::Pow(float x, float y) { return GPUCA_CHOICE((float)pow((double)x, (double)y), pow((double)x, (double)y), pow(x, y)); }
GPUdi() constexpr float GPUCommonMath::ASin(float x) { return GPUCA_CHOICE((float)asin((double)x), (float)asin((double)x), asin(x)); }
GPUdi() constexpr float GPUCommonMath::ACos(float x) { return GPUCA_CHOICE((float)acos((double)x), (float)acos((double)x), acos(x)); }
GPUdi() constexpr float GPUCommonMath::Log(float x) { return GPUCA_CHOICE((float)log((double)x), (float)log((double)x), log(x)); }
GPUdi() constexpr float GPUCommonMath::Exp(float x) { return GPUCA_CHOICE((float)exp((double)x), (float)exp((double)x), exp(x)); }
GPUdi() constexpr bool GPUCommonMath::Finite(float x) { return GPUCA_CHOICE(std::isfinite(x), isfinite(x), isfinite(x)); }
GPUdi() constexpr bool GPUCommonMath::IsNaN(float x) { return GPUCA_CHOICE(std::isnan(x), isnan(x), isnan(x)); }
, // !GPUCA_DETERMINISTIC_CODE
GPUdi() constexpr float GPUCommonMath::Round(float x) { return GPUCA_CHOICE(roundf(x), rintf(x), rint(x)); }
GPUdi() constexpr int32_t GPUCommonMath::Float2IntRn(float x) { return GPUCA_CHOICE((int32_t)Round(x), __float2int_rn(x), (int32_t)Round(x)); }
GPUhdi() constexpr float GPUCommonMath::Sqrt(float x) { return GPUCA_CHOICE(sqrtf(x), sqrtf(x), sqrt(x)); }
GPUdi() constexpr float GPUCommonMath::ATan(float x) { return GPUCA_CHOICE(atanf(x), atanf(x), atan(x)); }
GPUhdi() constexpr float GPUCommonMath::ATan2(float y, float x) { return GPUCA_CHOICE(atan2f(y, x), atan2f(y, x), atan2(y, x)); }
GPUdi() constexpr float GPUCommonMath::Sin(float x) { return GPUCA_CHOICE(sinf(x), sinf(x), sin(x)); }
GPUdi() constexpr float GPUCommonMath::Cos(float x) { return GPUCA_CHOICE(cosf(x), cosf(x), cos(x)); }
GPUdi() constexpr float GPUCommonMath::Tan(float x) { return GPUCA_CHOICE(tanf(x), tanf(x), tan(x)); }
GPUdi() constexpr float GPUCommonMath::Pow(float x, float y) { return GPUCA_CHOICE(powf(x, y), powf(x, y), pow(x, y)); }
GPUdi() constexpr float GPUCommonMath::ASin(float x) { return GPUCA_CHOICE(asinf(x), asinf(x), asin(x)); }
GPUdi() constexpr float GPUCommonMath::ACos(float x) { return GPUCA_CHOICE(acosf(x), acosf(x), acos(x)); }
GPUdi() constexpr float GPUCommonMath::Log(float x) { return GPUCA_CHOICE(logf(x), logf(x), log(x)); }
GPUdi() constexpr float GPUCommonMath::Exp(float x) { return GPUCA_CHOICE(expf(x), expf(x), exp(x)); }
GPUdi() constexpr bool GPUCommonMath::Finite(float x) { return true; }
GPUdi() constexpr bool GPUCommonMath::IsNaN(float x) { return false; }
) // clang-format on

GPUhdi() void GPUCommonMath::SinCos(float x, float& s, float& c)
{
  GPUCA_DETERMINISTIC_CODE( // clang-format off
    s = sin((double)x);
    c = cos((double)x);
  , // !GPUCA_DETERMINISTIC_CODE
#if !defined(GPUCA_GPUCODE_DEVICE) && defined(__APPLE__)
    __sincosf(x, &s, &c);
#elif !defined(GPUCA_GPUCODE_DEVICE) && (defined(__GNU_SOURCE__) || defined(_GNU_SOURCE) || defined(GPUCA_GPUCODE))
    sincosf(x, &s, &c);
#else
    GPUCA_CHOICE((void)((s = sinf(x)) + (c = cosf(x))), sincosf(x, &s, &c), s = sincos(x, &c));
#endif
  ) // clang-format on
}

GPUhdi() void GPUCommonMath::SinCosd(double x, double& s, double& c)
{
#if !defined(GPUCA_GPUCODE_DEVICE) && defined(__APPLE__)
  __sincos(x, &s, &c);
#elif !defined(GPUCA_GPUCODE_DEVICE) && (defined(__GNU_SOURCE__) || defined(_GNU_SOURCE) || defined(GPUCA_GPUCODE))
  sincos(x, &s, &c);
#else
  GPUCA_CHOICE((void)((s = sin(x)) + (c = cos(x))), sincos(x, &s, &c), s = sincos(x, &c));
#endif
}

GPUdi() constexpr uint32_t GPUCommonMath::Clz(uint32_t x)
{
#if (defined(__GNUC__) || defined(__clang__) || defined(__CUDACC__) || defined(__HIPCC__))
  return x == 0 ? 32 : GPUCA_CHOICE(__builtin_clz(x), __clz(x), __builtin_clz(x)); // use builtin if available
#else
  for (int32_t i = 31; i >= 0; i--) {
    if (x & (1u << i)) {
      return (31 - i);
    }
  }
  return 32;
#endif
}

GPUdi() constexpr uint32_t GPUCommonMath::Ctz(uint32_t x)
{
#if (defined(__GNUC__) || defined(__clang__) || defined(__CUDACC__) || defined(__HIPCC__))
  return x == 0 ? 32 : GPUCA_CHOICE(__builtin_ctz(x), __ffs(x) - 1, __builtin_ctz(x));
#else
  for (uint32_t i = 0; i < 32; ++i) {
    if (x & (1u << i)) {
      return i;
    }
  }
  return 32;
#endif
}

GPUdi() constexpr uint32_t GPUCommonMath::Popcount(uint32_t x)
{
#if (defined(__GNUC__) || defined(__clang__) || defined(__CUDACC__) || defined(__HIPCC__)) && !defined(__OPENCL__) // TODO: remove OPENCL when reported SPIR-V bug is fixed
  // use builtin if available
  return GPUCA_CHOICE(__builtin_popcount(x), __popc(x), __builtin_popcount(x));
#else
  x = x - ((x >> 1) & 0x55555555);
  x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
  return (((x + (x >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
#endif
}

template <typename T>
GPUhdi() constexpr void GPUCommonMath::Swap(T& a, T& b)
{
#ifndef GPUCA_GPUCODE_DEVICE
  std::swap(a, b);
#else
  T tmp = a;
  a = b;
  b = tmp;
#endif
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

GPUdi() float GPUCommonMath::InvSqrt(float _x)
{
  GPUCA_DETERMINISTIC_CODE( // clang-format off
    return 1.f / Sqrt(_x);
  , // !GPUCA_DETERMINISTIC_CODE
#if defined(__CUDACC__) || defined(__HIPCC__)
    return __frsqrt_rn(_x);
#elif defined(__OPENCL__) && defined(__clang__)
    return 1.f / sqrt(_x);
#elif !defined(__OPENCL__) && (defined(__FAST_MATH__) || defined(__clang__))
    return 1.f / sqrtf(_x);
#else
    union {
      float f;
      int32_t i;
    } x = {_x};
    const float xhalf = 0.5f * x.f;
    x.i = 0x5f3759df - (x.i >> 1);
    x.f = x.f * (1.5f - xhalf * x.f * x.f);
    return x.f;
#endif
  ) // clang-format on
}

template <>
GPUhdi() constexpr float GPUCommonMath::Abs<float>(float x)
{
  return GPUCA_CHOICE(fabsf(x), fabsf(x), fabs(x));
}

template <>
GPUhdi() constexpr double GPUCommonMath::Abs<double>(double x)
{
  return GPUCA_CHOICE(fabs(x), fabs(x), fabs(x));
}

template <>
GPUhdi() constexpr int32_t GPUCommonMath::Abs<int32_t>(int32_t x)
{
  return GPUCA_CHOICE(abs(x), abs(x), abs(x));
}

template <class S, class T>
GPUdi() uint32_t GPUCommonMath::AtomicExchInternal(S* addr, T val)
{
#if defined(GPUCA_GPUCODE) && defined(__OPENCL__) && (!defined(__clang__) || defined(GPUCA_OPENCL_CLANG_C11_ATOMICS))
  return ::atomic_exchange(addr, val);
#elif defined(GPUCA_GPUCODE) && defined(__OPENCL__)
  return ::atomic_xchg(addr, val);
#elif defined(GPUCA_GPUCODE) && (defined(__CUDACC__) || defined(__HIPCC__))
  return ::atomicExch(addr, val);
#elif defined(WITH_OPENMP)
  uint32_t old;
  __atomic_exchange(addr, &val, &old, __ATOMIC_SEQ_CST);
  return old;
#else
  return reinterpret_cast<std::atomic<T>*>(addr)->exchange(val);
#endif
}

template <class S, class T>
GPUdi() bool GPUCommonMath::AtomicCASInternal(S* addr, T cmp, T val)
{
#if defined(GPUCA_GPUCODE) && defined(__OPENCL__) && (!defined(__clang__) || defined(GPUCA_OPENCL_CLANG_C11_ATOMICS))
  return ::atomic_compare_exchange(addr, cmp, val) == cmp;
#elif defined(GPUCA_GPUCODE) && defined(__OPENCL__)
  return ::atomic_cmpxchg(addr, cmp, val) == cmp;
#elif defined(GPUCA_GPUCODE) && (defined(__CUDACC__) || defined(__HIPCC__))
  return ::atomicCAS(addr, cmp, val) == cmp;
#elif defined(WITH_OPENMP)
  return __atomic_compare_exchange(addr, &cmp, &val, true, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
#else
  return reinterpret_cast<std::atomic<T>*>(addr)->compare_exchange_strong(cmp, val);
#endif
}

template <class S, class T>
GPUdi() uint32_t GPUCommonMath::AtomicAddInternal(S* addr, T val)
{
#if defined(GPUCA_GPUCODE) && defined(__OPENCL__) && (!defined(__clang__) || defined(GPUCA_OPENCL_CLANG_C11_ATOMICS))
  return ::atomic_fetch_add(addr, val);
#elif defined(GPUCA_GPUCODE) && defined(__OPENCL__)
  return ::atomic_add(addr, val);
#elif defined(GPUCA_GPUCODE) && (defined(__CUDACC__) || defined(__HIPCC__))
  return ::atomicAdd(addr, val);
#elif defined(WITH_OPENMP)
  return __atomic_add_fetch(addr, val, __ATOMIC_SEQ_CST) - val;
#else
  return reinterpret_cast<std::atomic<T>*>(addr)->fetch_add(val);
#endif
}

template <class S, class T>
GPUdi() void GPUCommonMath::AtomicMaxInternal(S* addr, T val)
{
#if defined(GPUCA_GPUCODE) && defined(__OPENCL__) && (!defined(__clang__) || defined(GPUCA_OPENCL_CLANG_C11_ATOMICS))
  ::atomic_fetch_max(addr, val);
#elif defined(GPUCA_GPUCODE) && defined(__OPENCL__)
  ::atomic_max(addr, val);
#elif defined(GPUCA_GPUCODE) && (defined(__CUDACC__) || defined(__HIPCC__))
  ::atomicMax(addr, val);
#else
  S current;
  while ((current = *(volatile S*)addr) < val && !AtomicCASInternal(addr, current, val)) {
  }
#endif // GPUCA_GPUCODE
}

template <class S, class T>
GPUdi() void GPUCommonMath::AtomicMinInternal(S* addr, T val)
{
#if defined(GPUCA_GPUCODE) && defined(__OPENCL__) && (!defined(__clang__) || defined(GPUCA_OPENCL_CLANG_C11_ATOMICS))
  ::atomic_fetch_min(addr, val);
#elif defined(GPUCA_GPUCODE) && defined(__OPENCL__)
  ::atomic_min(addr, val);
#elif defined(GPUCA_GPUCODE) && (defined(__CUDACC__) || defined(__HIPCC__))
  ::atomicMin(addr, val);
#else
  S current;
  while ((current = *(volatile S*)addr) > val && !AtomicCASInternal(addr, current, val)) {
  }
#endif // GPUCA_GPUCODE
}

#if (defined(__CUDACC__) || defined(__HIPCC__)) && !defined(G__ROOT) && !defined(__CLING__)
#define GPUCA_HAVE_ATOMIC_MINMAX_FLOAT
template <>
GPUdii() void GPUCommonMath::AtomicMaxInternal(GPUglobalref() GPUgeneric() GPUAtomic(float) * addr, float val)
{
  if (val == -0.f) {
    val = 0.f;
  }
  if (val >= 0) {
    AtomicMaxInternal((GPUAtomic(int32_t)*)addr, __float_as_int(val));
  } else {
    AtomicMinInternal((GPUAtomic(uint32_t)*)addr, __float_as_uint(val));
  }
}
template <>
GPUdii() void GPUCommonMath::AtomicMinInternal(GPUglobalref() GPUgeneric() GPUAtomic(float) * addr, float val)
{
  if (val == -0.f) {
    val = 0.f;
  }
  if (val >= 0) {
    AtomicMinInternal((GPUAtomic(int32_t)*)addr, __float_as_int(val));
  } else {
    AtomicMaxInternal((GPUAtomic(uint32_t)*)addr, __float_as_uint(val));
  }
}
#endif

#undef GPUCA_CHOICE

} // namespace o2::gpu

#endif // GPUCOMMONMATH_H
