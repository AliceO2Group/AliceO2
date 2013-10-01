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

#if defined(HLTCA_STANDALONE) || defined(HLTCA_GPUCODE)
#if !defined(__OPENCL__) || defined(HLTCA_HOSTCODE)
#include <math.h>
#endif
#else
#include "TMath.h"
#endif //HLTCA_STANDALONE | HLTCA_GPUCODE

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

    GPUd()  static int AtomicExch( int *addr, int val );
    GPUd()  static int AtomicAdd ( int *addr, int val );
    GPUd()  static int AtomicMax ( int *addr, int val );
    GPUd()  static int AtomicMin ( int *addr, int val );
    GPUd()  static int AtomicExchShared( int *addr, int val );
    GPUd()  static int AtomicAddShared ( int *addr, int val );
    GPUd()  static int AtomicMaxShared ( int *addr, int val );
    GPUd()  static int AtomicMinShared ( int *addr, int val );
    GPUd()  static int Mul24( int a, int b );
    GPUd()  static float FMulRZ( float a, float b );
};

typedef AliHLTTPCCAMath CAMath;


#if defined( HLTCA_GPUCODE )
#define choice(c1,c2,c3) c1

#if defined( __OPENCL__ )
#if defined( HLTCA_HOSTCODE)
#if defined( HLTCA_STANDALONE )
#define choiceA(c1,c2,c3) c2
#else //HLTCA_STANDALONE
#define choiceA(c1,c2,c3) c3
#endif //HLTCA_STANDALONE
#else //HLTCA_HOSTCODE
#define choiceA(c1, c2, c3) c2
#endif //HLTCA_HOSTCODE
#else //__OPENCL

#define choiceA choice
#endif //__OPENCL__
#elif defined( HLTCA_STANDALONE )
#define choice(c1,c2,c3) c2
#define choiceA choice
#else
#define choice(c1,c2,c3) c3
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
#if defined(HLTCA_STANDALONE) || defined( HLTCA_GPUCODE )
  int i;
  if ( x >= 0 ) {
    i = int( x + 0.5 );
    if ( x + 0.5 == float( i ) && i & 1 ) i--;
  } else {
    i = int( x - 0.5 );
    if ( x - 0.5 == float( i ) && i & 1 ) i++;
  }
  return i;
#else
  return TMath::Nint( x );
#endif //HLTCA_STANDALONE | HLTCA_GPUCODE
}

GPUd() inline bool AliHLTTPCCAMath::Finite( float x )
{
  return choice( 1 /*isfinite( x )*/, finite( x ), finite( x ) );
}

GPUd() inline float AliHLTTPCCAMath::ATan2( float y, float x )
{
  return choiceA( atan2f( y, x ), atan2( y, x ), TMath::ATan2( y, x ) );
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
  return choiceA( sinf( x ), sin( x ), TMath::Sin( x ) );
}

GPUhd() inline float AliHLTTPCCAMath::Cos( float x )
{
  return choiceA( cosf( x ), cos( x ), TMath::Cos( x ) );
}

GPUhd() inline float AliHLTTPCCAMath::Tan( float x )
{
  return choiceA( tanf( x ), tan( x ), TMath::Tan( x ) );
}

GPUhd() inline float AliHLTTPCCAMath::Min( float x, float y )
{
  return choiceA( fminf( x, y ),  ( x < y ? x : y ), TMath::Min( x, y ) );
}

GPUhd() inline float AliHLTTPCCAMath::Max( float x, float y )
{
  return choiceA( fmaxf( x, y ),  ( x > y ? x : y ), TMath::Max( x, y ) );
}

GPUhd() inline int AliHLTTPCCAMath::Min( int x, int y )
{
  return choiceA( min( x, y ),  ( x < y ? x : y ), TMath::Min( x, y ) );
}

GPUhd() inline int AliHLTTPCCAMath::Max( int x, int y )
{
  return choiceA( max( x, y ),  ( x > y ? x : y ), TMath::Max( x, y ) );
}

GPUhd() inline float AliHLTTPCCAMath::Sqrt( float x )
{
  return choiceA( sqrtf( x ), sqrt( x ), TMath::Sqrt( x ) );
}

GPUhd() inline float AliHLTTPCCAMath::Abs( float x )
{
  return choiceA( fabsf( x ), fabs( x ), TMath::Abs( x ) );
}

GPUhd() inline double AliHLTTPCCAMath::Abs( double x )
{
  return choice( fabs( x ), fabs( x ), TMath::Abs( x ) );
}

GPUhd() inline int AliHLTTPCCAMath::Abs( int x )
{
  return choice( abs( x ), ( x >= 0 ? x : -x ), TMath::Abs( x ) );
}

GPUhd() inline float AliHLTTPCCAMath::ASin( float x )
{
  return choiceA( asinf( x ), asin( x ), TMath::ASin( x ) );
}


GPUd() inline int AliHLTTPCCAMath::Mul24( int a, int b )
{
#if defined(FERMI) || defined(__OPENCL__)
  return(a * b);
#else
  return choice( __mul24( a, b ), a*b, a*b );
#endif
}

GPUd() inline float AliHLTTPCCAMath::FMulRZ( float a, float b )
{
  return choiceA( __fmul_rz( a, b ), a*b, a*b );
}

GPUhd() inline float AliHLTTPCCAMath::Log(float x)
{
	return choice( Log(x), Log(x), TMath::Log(x));
}

#if defined(__OPENCL__) && !defined(HLTCA_HOSTCODE)
GPUd()  inline int AliHLTTPCCAMath::AtomicExchShared( int *addr, int val ) {return ::atomic_xchg( (volatile __local int*) addr, val );}
GPUd()  inline int AliHLTTPCCAMath::AtomicAddShared ( int *addr, int val ) {return ::atomic_add( (volatile __local int*) addr, val );}
GPUd()  inline int AliHLTTPCCAMath::AtomicMaxShared ( int *addr, int val ) {return ::atomic_max( (volatile __local int*) addr, val );}
GPUd()  inline int AliHLTTPCCAMath::AtomicMinShared ( int *addr, int val ) {return ::atomic_min( (volatile __local int*) addr, val );}

#else
GPUd()  inline int AliHLTTPCCAMath::AtomicExchShared( int *addr, int val ) {return(AliHLTTPCCAMath::AtomicExch(addr, val));}
GPUd()  inline int AliHLTTPCCAMath::AtomicAddShared ( int *addr, int val ) {return(AliHLTTPCCAMath::AtomicAdd(addr, val));}
GPUd()  inline int AliHLTTPCCAMath::AtomicMaxShared ( int *addr, int val ) {return(AliHLTTPCCAMath::AtomicMax(addr, val));}
GPUd()  inline int AliHLTTPCCAMath::AtomicMinShared ( int *addr, int val ) {return(AliHLTTPCCAMath::AtomicMin(addr, val));}
#endif


GPUd()  inline int AliHLTTPCCAMath::AtomicExch( int *addr, int val )
{
#if defined( HLTCA_GPUCODE ) & !defined(HLTCA_HOSTCODE)
#ifdef __OPENCL__
	return ::atomic_xchg( (volatile __global int*) addr, val );
#else
  return ::atomicExch( addr, val );
#endif
#else
  int old = *addr;
  *addr = val;
  return old;
#endif //HLTCA_GPUCODE
}

GPUd()  inline int AliHLTTPCCAMath::AtomicAdd ( int *addr, int val )
{
#if defined( HLTCA_GPUCODE ) & !defined(HLTCA_HOSTCODE)
#ifdef __OPENCL__
  return ::atomic_add( (volatile __global int*) addr, val );
#else
  return ::atomicAdd( addr, val );
#endif
#else
  int old = *addr;
  *addr += val;
  return old;
#endif //HLTCA_GPUCODE
}

GPUd()  inline int AliHLTTPCCAMath::AtomicMax ( int *addr, int val )
{
#if defined( HLTCA_GPUCODE ) & !defined(HLTCA_HOSTCODE)
#ifdef __OPENCL__
  return ::atomic_max( (volatile __global int*) addr, val );
#else
  return ::atomicMax( addr, val );
#endif
#else
  int old = *addr;
  if ( *addr < val ) *addr = val;
  return old;
#endif //HLTCA_GPUCODE
}

GPUd()  inline int AliHLTTPCCAMath::AtomicMin ( int *addr, int val )
{
#if defined( HLTCA_GPUCODE ) & !defined(HLTCA_HOSTCODE)
#ifdef __OPENCL__
  return ::atomic_min( (volatile __global int*) addr, val );
#else
  return ::atomicMin( addr, val );
#endif
#else
  int old = *addr;
  if ( *addr > val ) *addr = val;
  return old;
#endif //HLTCA_GPUCODE
}

#undef CHOICE

#endif //ALIHLTTPCCAMATH_H
