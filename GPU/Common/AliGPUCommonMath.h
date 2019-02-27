//-*- Mode: C++ -*-
// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************

#ifndef ALIGPUCOMMONMATH_H
#define ALIGPUCOMMONMATH_H

#include "AliGPUCommonDef.h"

#if !defined(__OPENCL__)
#include <cmath>
#include <algorithm>
#endif

class AliGPUCommonMath
{
  public:
	GPUhdni() static float2 MakeFloat2( float x, float y );

	template <class T> GPUhd() static T Min( T x, T y );
	template <class T> GPUhd() static T Max( T x, T y );
	GPUhdni() static float Sqrt( float x );
	template <class T> GPUhd() static T Abs( T x );
	GPUhdni() static float ASin( float x );
	GPUhdni() static float ATan( float x );
	GPUhdni() static float ATan2( float y, float x );
	GPUhdni() static float Sin( float x );
	GPUhdni() static float Cos( float x );
	GPUhdni() static float Tan( float x );
	GPUhdni() static float Copysign( float x, float y );
	GPUhdni() static float TwoPi() { return 6.28319f; }
	GPUhdni() static float Pi() { return 3.1415926535897f; }
	GPUhdni() static int Nint( float x );
	GPUhdni() static bool Finite( float x );

	GPUhdni() static float Log(float x);
	GPUd() static int AtomicExch( GPUglobalref() GPUAtomic(int) *addr, int val );
	GPUd() static int AtomicAdd ( GPUglobalref() GPUAtomic(int) *addr, int val );
	GPUd() static void AtomicMax ( GPUglobalref() GPUAtomic(int) *addr, int val );
	GPUd() static void AtomicMin ( GPUglobalref() GPUAtomic(int) *addr, int val );
	GPUd() static int AtomicExchShared( GPUsharedref() GPUAtomic(int) *addr, int val );
	GPUd() static int AtomicAddShared ( GPUsharedref() GPUAtomic(int) *addr, int val );
	GPUd() static void AtomicMaxShared ( GPUsharedref() GPUAtomic(int) *addr, int val );
	GPUd() static void AtomicMinShared ( GPUsharedref() GPUAtomic(int) *addr, int val );
	GPUd() static int Mul24( int a, int b );
	GPUd() static float FMulRZ( float a, float b );
};

typedef AliGPUCommonMath CAMath;

#if defined(GPUCA_GPUCODE_DEVICE) && (defined (__CUDACC__) || defined(__HIPCC_))
    #define CHOICE(c1, c2, c3) c2
#elif defined(GPUCA_GPUCODE_DEVICE) && defined (__OPENCL__)
    #define CHOICE(c1, c2, c3) c3
#else
    #define CHOICE(c1, c2, c3) c1
#endif

GPUhdi() float2 AliGPUCommonMath::MakeFloat2(float x, float y)
{
#if !defined(GPUCA_GPUCODE) || defined(__OPENCL__)
	float2 ret = {x, y};
	return ret;
#else
	return make_float2(x, y);
#endif //GPUCA_GPUCODE
}

GPUhdi() int AliGPUCommonMath::Nint(float x)
{
	int i;
	if (x >= 0)
	{
		i = int(x + 0.5f);
		if (x + 0.5f == float(i) && i & 1) i--;
	}
	else
	{
		i = int(x - 0.5f);
		if (x - 0.5f == float(i) && i & 1) i++;
	}
	return i;
}

GPUhdi() bool AliGPUCommonMath::Finite(float x)
{
	return CHOICE(std::isfinite(x), true, true);
}

GPUhdi() float AliGPUCommonMath::ATan(float x)
{
	return CHOICE(atanf(x), atanf(x), atan(x));
}

GPUhdi() float AliGPUCommonMath::ATan2(float y, float x)
{
	return CHOICE(atan2f(y, x), atan2f(y, x), atan2(y, x));
}

GPUhdi() float AliGPUCommonMath::Sin(float x)
{
	return CHOICE(sinf(x), sinf(x), sin(x));
}

GPUhdi() float AliGPUCommonMath::Cos(float x)
{
	return CHOICE(cosf(x), cosf(x), cos(x));
}

GPUhdi() float AliGPUCommonMath::Tan(float x)
{
	return CHOICE(tanf(x), tanf(x), tan(x));
}

template <class T> GPUhdi() T AliGPUCommonMath::Min(T x, T y)
{
	return CHOICE(std::min(x, y), std::min(x, y), (x < y ? x : y));
}

template <class T> GPUhdi() T AliGPUCommonMath::Max(T x, T y)
{
	return CHOICE(std::max(x, y), std::max(x, y), (x > y ? x : y));
}

GPUhdi() float AliGPUCommonMath::Sqrt(float x)
{
	return CHOICE(sqrtf(x), sqrtf(x), sqrt(x));
}

template <> GPUhdi() float AliGPUCommonMath::Abs<float>(float x)
{
	return CHOICE(fabsf(x), fabsf(x), fabs(x));
}

#if !defined(__OPENCL__) || defined(cl_khr_fp64)
template <> GPUhdi() double AliGPUCommonMath::Abs<double>(double x)
{
	return CHOICE(fabs(x), fabs(x), fabs(x));
}
#endif

template <> GPUhdi() int AliGPUCommonMath::Abs<int>(int x)
{
	return CHOICE(abs(x), abs(x), abs(x));
}

GPUhdi() float AliGPUCommonMath::ASin(float x)
{
	return CHOICE(asinf(x), asinf(x), asin(x));
}

GPUhdi() float AliGPUCommonMath::Log(float x)
{
	return CHOICE(logf(x), logf(x), log(x));
}

GPUhdi() float AliGPUCommonMath::Copysign(float x, float y)
{
#if defined(__OPENCLCPP__)
    return copysign(x, y);
#elif defined(GPUCA_GPUCODE) && !defined(__OPENCL__)
	return copysignf(x, y);
#elif defined(__cplusplus) && __cplusplus >= 201103L
    return std::copysignf(x, y);
#else
	x = AliGPUCommonMath::Abs(x);
	return (y >= 0) ? x : -x;
#endif //GPUCA_GPUCODE
}

#if defined(__OPENCL__) && !defined(__OPENCLCPP__)
GPUdi() int AliGPUCommonMath::AtomicExchShared( GPUsharedref() GPUAtomic(int) *addr, int val ) {return ::atomic_xchg( (volatile __local int*) addr, val );}
GPUdi() int AliGPUCommonMath::AtomicAddShared ( GPUsharedref() GPUAtomic(int) *addr, int val ) {return ::atomic_add( (volatile __local int*) addr, val );}
GPUdi() void AliGPUCommonMath::AtomicMaxShared ( GPUsharedref() GPUAtomic(int) *addr, int val ) {::atomic_max( (volatile __local int*) addr, val );}
GPUdi() void AliGPUCommonMath::AtomicMinShared ( GPUsharedref() GPUAtomic(int) *addr, int val ) {::atomic_min( (volatile __local int*) addr, val );}
#else
GPUdi() int AliGPUCommonMath::AtomicExchShared( GPUAtomic(int) *addr, int val ) {return AliGPUCommonMath::AtomicExch(addr, val);}
GPUdi() int AliGPUCommonMath::AtomicAddShared ( GPUAtomic(int) *addr, int val ) {return AliGPUCommonMath::AtomicAdd(addr, val);}
GPUdi() void AliGPUCommonMath::AtomicMaxShared ( GPUAtomic(int) *addr, int val ) {AliGPUCommonMath::AtomicMax(addr, val);}
GPUdi() void AliGPUCommonMath::AtomicMinShared ( GPUAtomic(int) *addr, int val ) {AliGPUCommonMath::AtomicMin(addr, val);}
#endif

#ifndef GPUCA_GPUCODE
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-value" //GCC BUG in omp atomic capture gives false warning
#endif

GPUdi() int AliGPUCommonMath::AtomicExch( GPUglobalref() GPUAtomic(int) *addr, int val )
{
#if defined(GPUCA_GPUCODE) && defined(__OPENCLCPP__) && !defined(__clang__)
    return atomic_exchange(addr, val);
#elif defined(GPUCA_GPUCODE) && defined(__OPENCL__)
	return ::atomic_xchg( (volatile __global int*) addr, val );
#elif defined(GPUCA_GPUCODE) && (defined (__CUDACC__) || defined(__HIPCC_))
	return ::atomicExch( addr, val );
#else
	int old;
#ifdef GPUCA_HAVE_OPENMP
#pragma omp atomic capture
#endif
	{
		old = *addr;
		*addr = val;
	}
	return old;
#endif //GPUCA_GPUCODE
}

GPUdi() int AliGPUCommonMath::AtomicAdd ( GPUglobalref() GPUAtomic(int) *addr, int val )
{
#if defined(GPUCA_GPUCODE) && defined(__OPENCLCPP__) && !defined(__clang__)
    return atomic_fetch_add(addr, val);
#elif defined(GPUCA_GPUCODE) && defined(__OPENCL__)
	return ::atomic_add( (volatile __global int*) addr, val );
#elif defined(GPUCA_GPUCODE) && (defined (__CUDACC__) || defined(__HIPCC_))
	return ::atomicAdd( addr, val );
#else
	int old;
#ifdef GPUCA_HAVE_OPENMP
#pragma omp atomic capture
#endif
	{
		old = *addr;
		*addr += val;
	}
	return old;
#endif //GPUCA_GPUCODE
}

GPUdi() void AliGPUCommonMath::AtomicMax ( GPUglobalref() GPUAtomic(int) *addr, int val )
{
#if defined(GPUCA_GPUCODE) && defined(__OPENCLCPP__) && !defined(__clang__)
    atomic_fetch_max(addr, val);
#elif defined(GPUCA_GPUCODE) && defined(__OPENCL__)
	::atomic_max( (volatile __global int*) addr, val );
#elif defined(GPUCA_GPUCODE) && (defined (__CUDACC__) || defined(__HIPCC_))
	::atomicMax( addr, val );
#else
#ifdef GPUCA_HAVE_OPENMP
	while (*addr < val) AtomicExch(addr, val);
#else
	if ( *addr < val ) *addr = val;
#endif
#endif //GPUCA_GPUCODE
}

GPUdi() void AliGPUCommonMath::AtomicMin ( GPUglobalref() GPUAtomic(int) *addr, int val )
{
#if defined(GPUCA_GPUCODE) && defined(__OPENCLCPP__) && !defined(__clang__)
    atomic_fetch_min(addr, val);
#elif defined(GPUCA_GPUCODE) && defined(__OPENCL__)
	::atomic_min( (volatile __global int*) addr, val );
#elif defined(GPUCA_GPUCODE) && (defined (__CUDACC__) || defined(__HIPCC_))
	::atomicMin( addr, val );
#else
#ifdef GPUCA_HAVE_OPENMP
	while (*addr > val) AtomicExch(addr, val);
#else
	if ( *addr > val ) *addr = val;
	#endif
#endif //GPUCA_GPUCODE
}

#ifndef GPUCA_GPUCODE
#pragma GCC diagnostic pop
#endif

#undef CHOICE

#endif //ALIGPUCOMMONMATH_H
