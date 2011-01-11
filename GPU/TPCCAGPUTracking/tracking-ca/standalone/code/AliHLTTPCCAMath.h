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
#include <math.h>
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

    GPUd() static float Min( float x, float y );
    GPUd() static float Max( float x, float y );
    GPUd() static int Min( int x, int y );
    GPUd() static int Max( int x, int y );
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

    GPUd()  static int AtomicExch( int *addr, int val );
    GPUd()  static int AtomicAdd ( int *addr, int val );
    GPUd()  static int AtomicMax ( int *addr, int val );
    GPUd()  static int AtomicMin ( int *addr, int val );
    GPUd()  static int Mul24( int a, int b );
    GPUd()  static float FMulRZ( float a, float b );
};

typedef AliHLTTPCCAMath CAMath;


#if defined( HLTCA_GPUCODE )
#define choice(c1,c2,c3) c1
#elif defined( HLTCA_STANDALONE )
#define choice(c1,c2,c3) c2
#else
#define choice(c1,c2,c3) c3
#endif //HLTCA_GPUCODE

GPUd() inline float2 AliHLTTPCCAMath::MakeFloat2( float x, float y )
{
#if !defined( HLTCA_GPUCODE )
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
  return choice( 1, finite( x ), finite( x ) );
}

GPUd() inline float AliHLTTPCCAMath::ATan2( float y, float x )
{
  return choice( atan2f( y, x ), atan2( y, x ), TMath::ATan2( y, x ) );
}


GPUd() inline float AliHLTTPCCAMath::Copysign( float x, float y )
{
#if defined( HLTCA_GPUCODE )
  return copysignf( x, y );
#else
  x = CAMath::Abs( x );
  return ( y >= 0 ) ? x : -x;
#endif //HLTCA_GPUCODE
}


GPUhd() inline float AliHLTTPCCAMath::Sin( float x )
{
  return choice( sinf( x ), sin( x ), TMath::Sin( x ) );
}

GPUhd() inline float AliHLTTPCCAMath::Cos( float x )
{
  return choice( cosf( x ), cos( x ), TMath::Cos( x ) );
}

GPUhd() inline float AliHLTTPCCAMath::Tan( float x )
{
  return choice( tanf( x ), tan( x ), TMath::Tan( x ) );
}

GPUhd() inline float AliHLTTPCCAMath::Min( float x, float y )
{
  return choice( fminf( x, y ),  ( x < y ? x : y ), TMath::Min( x, y ) );
}

GPUhd() inline float AliHLTTPCCAMath::Max( float x, float y )
{
  return choice( fmaxf( x, y ),  ( x > y ? x : y ), TMath::Max( x, y ) );
}

GPUhd() inline int AliHLTTPCCAMath::Min( int x, int y )
{
  return choice( min( x, y ),  ( x < y ? x : y ), TMath::Min( x, y ) );
}

GPUhd() inline int AliHLTTPCCAMath::Max( int x, int y )
{
  return choice( max( x, y ),  ( x > y ? x : y ), TMath::Max( x, y ) );
}

GPUhd() inline float AliHLTTPCCAMath::Sqrt( float x )
{
  return choice( sqrtf( x ), sqrt( x ), TMath::Sqrt( x ) );
}

GPUhd() inline float AliHLTTPCCAMath::Abs( float x )
{
  return choice( fabsf( x ), fabs( x ), TMath::Abs( x ) );
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
  return choice( asinf( x ), asin( x ), TMath::ASin( x ) );
}


GPUd() inline int AliHLTTPCCAMath::Mul24( int a, int b )
{
#ifdef FERMI
  return(a * b);
#else
  return choice( __mul24( a, b ), a*b, a*b );
#endif
}

GPUd() inline float AliHLTTPCCAMath::FMulRZ( float a, float b )
{
  return choice( __fmul_rz( a, b ), a*b, a*b );
}

GPUhd() inline float AliHLTTPCCAMath::Log(float x)
{
	return choice( Log(x), Log(x), TMath::Log(x));
}


GPUd()  inline int AliHLTTPCCAMath::AtomicExch( int *addr, int val )
{
#if defined( HLTCA_GPUCODE )
  return ::atomicExch( addr, val );
#else
  int old = *addr;
  *addr = val;
  return old;
#endif //HLTCA_GPUCODE
}

GPUd()  inline int AliHLTTPCCAMath::AtomicAdd ( int *addr, int val )
{
#if defined( HLTCA_GPUCODE )
  return ::atomicAdd( addr, val );
#else
  int old = *addr;
  *addr += val;
  return old;
#endif //HLTCA_GPUCODE
}

GPUd()  inline int AliHLTTPCCAMath::AtomicMax ( int *addr, int val )
{
#if defined( HLTCA_GPUCODE )
  return ::atomicMax( addr, val );
#else
  int old = *addr;
  if ( *addr < val ) *addr = val;
  return old;
#endif //HLTCA_GPUCODE
}

GPUd()  inline int AliHLTTPCCAMath::AtomicMin ( int *addr, int val )
{
#if defined( HLTCA_GPUCODE )
  return ::atomicMin( addr, val );
#else
  int old = *addr;
  if ( *addr > val ) *addr = val;
  return old;
#endif //HLTCA_GPUCODE
}

#undef CHOICE

#endif //ALIHLTTPCCAMATH_H
