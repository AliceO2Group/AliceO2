//-*- Mode: C++ -*-
// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************

#ifndef ALIHLTTPCCAMATH_H
#define ALIHLTTPCCAMATH_H

#include "AliHLTTPCCADef.h"

#if !defined(__OPENCL__)
#include <cmath>
#endif

/**
 * @class ALIHLTTPCCAMath
 *
 *
 */
class AliHLTTPCCAMath
{
  public:
    GPUd() static float2 MakeFloat2( float x, float y );

    GPUhd() static float Min( float x, float y );
    GPUhd() static float Max( float x, float y );
    GPUhd() static int Min( int x, int y );
    GPUhd() static int Max( int x, int y );
    GPUhd() static float Sqrt( float x );
    GPUhd() static float Abs( float x );
    GPUhd() static double Abs( double x );
    GPUhd() static int Abs( int x );
    GPUhd() static float ASin( float x );
    GPUd() static float ATan2( float y, float x );
    GPUhd() static float Sin( float x );
    GPUhd() static float Cos( float x );
    GPUhd() static float Tan( float x );
    GPUd() static float Copysign( float x, float y );
    GPUd() static float TwoPi() { return 6.28319; }
    GPUd() static float Pi() { return 3.1415926535897; }
    GPUd() static int Nint( float x );
    GPUd() static bool Finite( float x );

    GPUhd() static float Log(float x);
    GPUd()  static int AtomicExch( register GPUglobalref() int *addr, int val );
    GPUd()  static int AtomicAdd (register GPUglobalref() int *addr, int val );
    GPUd()  static int AtomicMax (register GPUglobalref() int *addr, int val );
    GPUd()  static int AtomicMin (register GPUglobalref() int *addr, int val );
    GPUd()  static int AtomicExchShared(register GPUsharedref() int *addr, int val );
    GPUd()  static int AtomicAddShared (register GPUsharedref() int *addr, int val );
    GPUd()  static int AtomicMaxShared (register GPUsharedref() int *addr, int val );
    GPUd()  static int AtomicMinShared (register GPUsharedref() int *addr, int val );
    GPUd()  static int Mul24( int a, int b );
    GPUd()  static float FMulRZ( float a, float b );
};

typedef AliHLTTPCCAMath CAMath;

#if defined( HLTCA_GPUCODE ) && defined (__CUDACC__)
    #define choice(c1,c2) c1
    #define choiceA choice
#elif defined( HLTCA_GPUCODE ) && defined (__OPENCL__)
    #define choice(c1,c2) c1
    #define choiceA(c1, c2) c2
#else //Host
    #define choice(c1,c2) c2
    #define choiceA choice
#endif //HLTCA_GPUCODE

GPUd() inline float2 AliHLTTPCCAMath::MakeFloat2( float x, float y )
{
#if !defined( HLTCA_GPUCODE ) || defined(__OPENCL__)
  float2 ret = {x, y};
  return ret;
#else
  return make_float2( x, y );
#endif //HLTCA_GPUCODE
}


GPUd() inline int AliHLTTPCCAMath::Nint( float x )
{
  int i;
  if ( x >= 0 ) {
    i = int( x + 0.5 );
    if ( x + 0.5 == float( i ) && i & 1 ) i--;
  } else {
    i = int( x - 0.5 );
    if ( x - 0.5 == float( i ) && i & 1 ) i++;
  }
  return i;
}

GPUd() inline bool AliHLTTPCCAMath::Finite( float x )
{
  return choice( 1, std::isfinite( x ) );
}

GPUd() inline float AliHLTTPCCAMath::ATan2( float y, float x )
{
  return choiceA( atan2f( y, x ), atan2( y, x ) );
}


GPUd() inline float AliHLTTPCCAMath::Copysign( float x, float y )
{
#if defined( HLTCA_GPUCODE ) && !defined(__OPENCL__)
  return copysignf( x, y );
#else
  x = CAMath::Abs( x );
  return ( y >= 0 ) ? x : -x;
#endif //HLTCA_GPUCODE
}


GPUhd() inline float AliHLTTPCCAMath::Sin( float x )
{
  return choiceA( sinf( x ), sin( x ) );
}

GPUhd() inline float AliHLTTPCCAMath::Cos( float x )
{
  return choiceA( cosf( x ), cos( x ) );
}

GPUhd() inline float AliHLTTPCCAMath::Tan( float x )
{
  return choiceA( tanf( x ), tan( x ) );
}

GPUhd() inline float AliHLTTPCCAMath::Min( float x, float y )
{
  return choiceA( fminf( x, y ), ( x < y ? x : y ) );
}

GPUhd() inline float AliHLTTPCCAMath::Max( float x, float y )
{
  return choiceA( fmaxf( x, y ),  ( x > y ? x : y ) );
}

GPUhd() inline int AliHLTTPCCAMath::Min( int x, int y )
{
  return choiceA( min( x, y ),  ( x < y ? x : y ) );
}

GPUhd() inline int AliHLTTPCCAMath::Max( int x, int y )
{
  return choiceA( max( x, y ),  ( x > y ? x : y ) );
}

GPUhd() inline float AliHLTTPCCAMath::Sqrt( float x )
{
  return choiceA( sqrtf( x ), sqrt( x ) );
}

GPUhd() inline float AliHLTTPCCAMath::Abs( float x )
{
  return choiceA( fabsf( x ), fabs( x ) );
}

GPUhd() inline double AliHLTTPCCAMath::Abs( double x )
{
  return choice( fabs( x ), fabs( x ) );
}

GPUhd() inline int AliHLTTPCCAMath::Abs( int x )
{
  return choice( abs( x ), ( x >= 0 ? x : -x ) );
}

GPUhd() inline float AliHLTTPCCAMath::ASin( float x )
{
  return choiceA( asinf( x ), asin( x ) );
}

GPUhd() inline float AliHLTTPCCAMath::Log(float x)
{
	return choice( log(x), log(x) );
}

#if defined(__OPENCL__)
GPUd()  inline int AliHLTTPCCAMath::AtomicExchShared(register GPUsharedref() int *addr, int val ) {return ::atomic_xchg( (volatile __local int*) addr, val );}
GPUd()  inline int AliHLTTPCCAMath::AtomicAddShared (register GPUsharedref() int *addr, int val ) {return ::atomic_add( (volatile __local int*) addr, val );}
GPUd()  inline int AliHLTTPCCAMath::AtomicMaxShared (register GPUsharedref() int *addr, int val ) {return ::atomic_max( (volatile __local int*) addr, val );}
GPUd()  inline int AliHLTTPCCAMath::AtomicMinShared (register GPUsharedref() int *addr, int val ) {return ::atomic_min( (volatile __local int*) addr, val );}
#else
GPUd()  inline int AliHLTTPCCAMath::AtomicExchShared( int *addr, int val ) {return(AliHLTTPCCAMath::AtomicExch(addr, val));}
GPUd()  inline int AliHLTTPCCAMath::AtomicAddShared ( int *addr, int val ) {return(AliHLTTPCCAMath::AtomicAdd(addr, val));}
GPUd()  inline int AliHLTTPCCAMath::AtomicMaxShared ( int *addr, int val ) {return(AliHLTTPCCAMath::AtomicMax(addr, val));}
GPUd()  inline int AliHLTTPCCAMath::AtomicMinShared ( int *addr, int val ) {return(AliHLTTPCCAMath::AtomicMin(addr, val));}
#endif

GPUd()  inline int AliHLTTPCCAMath::AtomicExch(register GPUglobalref() int *addr, int val )
{
#if defined(HLTCA_GPUCODE) && defined(__OPENCL__)
	return ::atomic_xchg( (volatile __global int*) addr, val );
#elif defined(HLTCA_GPUCODE) && defined(__CUDACC__)
  return ::atomicExch( addr, val );
#else
  int old = *addr;
  *addr = val;
  return old;
#endif //HLTCA_GPUCODE
}

GPUd()  inline int AliHLTTPCCAMath::AtomicAdd (register GPUglobalref() int *addr, int val )
{
#if defined(HLTCA_GPUCODE) && defined(__OPENCL__)
  return ::atomic_add( (volatile __global int*) addr, val );
#elif defined(HLTCA_GPUCODE) && defined(__CUDACC__)
  return ::atomicAdd( addr, val );
#else
  int old = *addr;
  *addr += val;
  return old;
#endif //HLTCA_GPUCODE
}

GPUd()  inline int AliHLTTPCCAMath::AtomicMax (register GPUglobalref() int *addr, int val )
{
#if defined(HLTCA_GPUCODE) && defined(__OPENCL__)
  return ::atomic_max( (volatile __global int*) addr, val );
#elif defined(HLTCA_GPUCODE) && defined(__CUDACC__)
  return ::atomicMax( addr, val );
#else
  int old = *addr;
  if ( *addr < val ) *addr = val;
  return old;
#endif //HLTCA_GPUCODE
}

GPUd()  inline int AliHLTTPCCAMath::AtomicMin (register GPUglobalref() int *addr, int val )
{
#if defined(HLTCA_GPUCODE) && defined(__OPENCL__)
  return ::atomic_min( (volatile __global int*) addr, val );
#elif defined(HLTCA_GPUCODE) && defined(__CUDACC__)
  return ::atomicMin( addr, val );
#else
  int old = *addr;
  if ( *addr > val ) *addr = val;
  return old;
#endif //HLTCA_GPUCODE
}

#undef CHOICE

#endif //ALIHLTTPCCAMATH_H
