//-*- Mode: C++ -*-
// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************

#ifndef ALITPCCOMMONMATH_H
#define ALITPCCOMMONMATH_H

#include "AliTPCCommonDef.h"

#if !defined(__OPENCL__)
#include <cmath>
#endif

class AliTPCCommonMath
{
  public:
	GPUd() static float2 MakeFloat2( float x, float y );

	GPUhd() static float Min( float x, float y );
	GPUhd() static float Max( float x, float y );
	GPUhd() static int Min( int x, int y );
	GPUhd() static int Max( int x, int y );
	GPUhd() static int Min( unsigned int x, unsigned int y );
	GPUhd() static int Max( unsigned int x, unsigned int y );
	GPUd() static float Sqrt( float x );
	GPUd() static float Abs( float x );
	GPUd() static double Abs( double x );
	GPUd() static int Abs( int x );
	GPUd() static float ASin( float x );
	GPUd() static float ATan2( float y, float x );
	GPUd() static float Sin( float x );
	GPUd() static float Cos( float x );
	GPUd() static float Tan( float x );
	GPUd() static float Copysign( float x, float y );
	GPUd() static float TwoPi() { return 6.28319; }
	GPUd() static float Pi() { return 3.1415926535897; }
	GPUd() static int Nint( float x );
	GPUd() static bool Finite( float x );

	GPUd() static float Log(float x);
	GPUd() static int AtomicExch( GPUglobalref() int *addr, int val );
	GPUd() static int AtomicAdd ( GPUglobalref() int *addr, int val );
	GPUd() static int AtomicMax ( GPUglobalref() int *addr, int val );
	GPUd() static int AtomicMin ( GPUglobalref() int *addr, int val );
	GPUd() static int AtomicExchShared( GPUsharedref() int *addr, int val );
	GPUd() static int AtomicAddShared ( GPUsharedref() int *addr, int val );
	GPUd() static int AtomicMaxShared ( GPUsharedref() int *addr, int val );
	GPUd() static int AtomicMinShared ( GPUsharedref() int *addr, int val );
	GPUd() static int Mul24( int a, int b );
	GPUd() static float FMulRZ( float a, float b );
};

typedef AliTPCCommonMath CAMath;

#if defined( GPUCA_GPUCODE ) && defined (__CUDACC__)
	#define choice(c1,c2) c1
	#define choiceA choice
#elif defined( GPUCA_GPUCODE ) && defined (__OPENCL__)
	#define choice(c1,c2) c1
	#define choiceA(c1, c2) c2
#else //Host
	#define choice(c1,c2) c2
	#define choiceA choice
#endif //GPUCA_GPUCODE

GPUdi() float2 AliTPCCommonMath::MakeFloat2(float x, float y)
{
#if !defined(GPUCA_GPUCODE) || defined(__OPENCL__)
	float2 ret = {x, y};
	return ret;
#else
	return make_float2(x, y);
#endif //GPUCA_GPUCODE
}

GPUdi() int AliTPCCommonMath::Nint(float x)
{
	int i;
	if (x >= 0)
	{
		i = int(x + 0.5);
		if (x + 0.5 == float(i) && i & 1) i--;
	}
	else
	{
		i = int(x - 0.5);
		if (x - 0.5 == float(i) && i & 1) i++;
	}
	return i;
}

GPUdi() bool AliTPCCommonMath::Finite(float x)
{
	return choice(1, std::isfinite(x));
}

GPUdi() float AliTPCCommonMath::ATan2(float y, float x)
{
	return choiceA(atan2f(y, x), atan2(y, x));
}

GPUdi() float AliTPCCommonMath::Copysign(float x, float y)
{
#if defined(GPUCA_GPUCODE) && !defined(__OPENCL__)
	return copysignf(x, y);
#else
	x = CAMath::Abs(x);
	return (y >= 0) ? x : -x;
#endif //GPUCA_GPUCODE
}

GPUdi() float AliTPCCommonMath::Sin(float x)
{
	return choiceA(sinf(x), sin(x));
}

GPUdi() float AliTPCCommonMath::Cos(float x)
{
	return choiceA(cosf(x), cos(x));
}

GPUdi() float AliTPCCommonMath::Tan(float x)
{
	return choiceA(tanf(x), tan(x));
}

GPUhdi() float AliTPCCommonMath::Min(float x, float y)
{
	return choiceA(fminf(x, y), (x < y ? x : y));
}

GPUhdi() float AliTPCCommonMath::Max(float x, float y)
{
	return choiceA(fmaxf(x, y), (x > y ? x : y));
}

GPUhdi() int AliTPCCommonMath::Min(int x, int y)
{
	return choiceA(min(x, y), (x < y ? x : y));
}

GPUhdi() int AliTPCCommonMath::Max(int x, int y)
{
	return choiceA(max(x, y), (x > y ? x : y));
}

GPUhdi() int AliTPCCommonMath::Min(unsigned int x, unsigned int y)
{
	return choiceA(min(x, y), (x < y ? x : y));
}

GPUhdi() int AliTPCCommonMath::Max(unsigned int x, unsigned int y)
{
	return choiceA(max(x, y), (x > y ? x : y));
}

GPUdi() float AliTPCCommonMath::Sqrt(float x)
{
	return choiceA(sqrtf(x), sqrt(x));
}

GPUdi() float AliTPCCommonMath::Abs(float x)
{
	return choiceA(fabsf(x), fabs(x));
}

GPUdi() double AliTPCCommonMath::Abs(double x)
{
	return choice(fabs(x), fabs(x));
}

GPUdi() int AliTPCCommonMath::Abs(int x)
{
	return choice(abs(x), (x >= 0 ? x : -x));
}

GPUdi() float AliTPCCommonMath::ASin(float x)
{
	return choiceA(asinf(x), asin(x));
}

GPUdi() float AliTPCCommonMath::Log(float x)
{
	return choice(log(x), log(x));
}

#if defined(__OPENCL__)
GPUdi() int AliTPCCommonMath::AtomicExchShared( GPUsharedref() int *addr, int val ) {return ::atomic_xchg( (volatile __local int*) addr, val );}
GPUdi() int AliTPCCommonMath::AtomicAddShared ( GPUsharedref() int *addr, int val ) {return ::atomic_add( (volatile __local int*) addr, val );}
GPUdi() int AliTPCCommonMath::AtomicMaxShared ( GPUsharedref() int *addr, int val ) {return ::atomic_max( (volatile __local int*) addr, val );}
GPUdi() int AliTPCCommonMath::AtomicMinShared ( GPUsharedref() int *addr, int val ) {return ::atomic_min( (volatile __local int*) addr, val );}
#else
GPUdi() int AliTPCCommonMath::AtomicExchShared( int *addr, int val ) {return(AliTPCCommonMath::AtomicExch(addr, val));}
GPUdi() int AliTPCCommonMath::AtomicAddShared ( int *addr, int val ) {return(AliTPCCommonMath::AtomicAdd(addr, val));}
GPUdi() int AliTPCCommonMath::AtomicMaxShared ( int *addr, int val ) {return(AliTPCCommonMath::AtomicMax(addr, val));}
GPUdi() int AliTPCCommonMath::AtomicMinShared ( int *addr, int val ) {return(AliTPCCommonMath::AtomicMin(addr, val));}
#endif

GPUdi() int AliTPCCommonMath::AtomicExch( GPUglobalref() int *addr, int val )
{
#if defined(GPUCA_GPUCODE) && defined(__OPENCL__)
	return ::atomic_xchg( (volatile __global int*) addr, val );
#elif defined(GPUCA_GPUCODE) && defined(__CUDACC__)
	return ::atomicExch( addr, val );
#else
	int old = *addr;
	*addr = val;
	return old;
#endif //GPUCA_GPUCODE
}

GPUdi() int AliTPCCommonMath::AtomicAdd ( GPUglobalref() int *addr, int val )
{
#if defined(GPUCA_GPUCODE) && defined(__OPENCL__)
	return ::atomic_add( (volatile __global int*) addr, val );
#elif defined(GPUCA_GPUCODE) && defined(__CUDACC__)
	return ::atomicAdd( addr, val );
#else
	int old = *addr;
	*addr += val;
	return old;
#endif //GPUCA_GPUCODE
}

GPUdi() int AliTPCCommonMath::AtomicMax ( GPUglobalref() int *addr, int val )
{
#if defined(GPUCA_GPUCODE) && defined(__OPENCL__)
	return ::atomic_max( (volatile __global int*) addr, val );
#elif defined(GPUCA_GPUCODE) && defined(__CUDACC__)
	return ::atomicMax( addr, val );
#else
	int old = *addr;
	if ( *addr < val ) *addr = val;
	return old;
#endif //GPUCA_GPUCODE
}

GPUdi() int AliTPCCommonMath::AtomicMin ( GPUglobalref() int *addr, int val )
{
#if defined(GPUCA_GPUCODE) && defined(__OPENCL__)
	return ::atomic_min( (volatile __global int*) addr, val );
#elif defined(GPUCA_GPUCODE) && defined(__CUDACC__)
	return ::atomicMin( addr, val );
#else
	int old = *addr;
	if ( *addr > val ) *addr = val;
	return old;
#endif //GPUCA_GPUCODE
}

#undef CHOICE

#endif //AliTPCCommonMath_H
