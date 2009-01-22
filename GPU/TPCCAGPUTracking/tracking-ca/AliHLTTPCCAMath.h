//-*- Mode: C++ -*-
// @(#) $Id: AliHLTTPCCARow.h 27042 2008-07-02 12:06:02Z richterm $

//* This file is property of and copyright by the ALICE HLT Project        * 
//* ALICE Experiment at CERN, All rights reserved.                         *
//* See cxx source for full Copyright notice                               *

#ifndef ALIHLTTPCCAMATH_H
#define ALIHLTTPCCAMATH_H

#include "AliHLTTPCCADef.h"

#if defined(HLTCA_STANDALONE) || defined(HLTCA_GPUCODE)
#include <math.h>
#else
#include "TMath.h"
#endif

/**
 * @class ALIHLTTPCCAMath
 *
 *
 */
class AliHLTTPCCAMath
{
public:
  GPUd() static Float_t Min(Float_t x, Float_t y);
  GPUd() static Float_t Max(Float_t x, Float_t y);
  GPUd() static Int_t Min(Int_t x, Int_t y);
  GPUd() static Int_t Max(Int_t x, Int_t y);
  GPUd() static Float_t Sqrt(Float_t x );
  GPUd() static Float_t Abs(Float_t x );
  GPUd() static Double_t Abs(Double_t x );
  GPUd() static Int_t Abs(Int_t x );
  GPUd() static Float_t ASin(Float_t x );
  GPUd() static Float_t ATan2( Float_t y, Float_t x );
  GPUd() static Float_t Sin( Float_t x );
  GPUd() static Float_t Cos( Float_t x );
  GPUd() static Float_t Tan( Float_t x );
  GPUd() static Float_t Copysign( Float_t x, Float_t y );  
  GPUd() static Float_t TwoPi(){ return 6.28319; }
  GPUd() static Float_t Pi(){ return 3.1415926535897; }
  GPUd() static Int_t Nint(Float_t x );
  GPUd() static Bool_t Finite(Float_t x );

  GPUd()  static Int_t atomicExch( Int_t *addr, Int_t val);
  GPUd()  static Int_t atomicAdd ( Int_t *addr, Int_t val);
  GPUd()  static Int_t atomicMax ( Int_t *addr, Int_t val);
  GPUd()  static Int_t atomicMin ( Int_t *addr, Int_t val);
  GPUd()  static Int_t mul24( Int_t a, Int_t b );
  GPUd()  static Float_t fmul_rz( Float_t a, Float_t b );
};

typedef AliHLTTPCCAMath CAMath;


#if defined( HLTCA_GPUCODE )
#define choice(c1,c2,c3) c1
#elif defined( HLTCA_STANDALONE )
#define choice(c1,c2,c3) c2
#else
#define choice(c1,c2,c3) c3
#endif


GPUd() inline Int_t AliHLTTPCCAMath::Nint(Float_t x)
{  
#if defined(HLTCA_STANDALONE) || defined( HLTCA_GPUCODE )
  Int_t i;
  if (x >= 0) {
    i = int(x + 0.5);
    if (x + 0.5 == Float_t(i) && i & 1) i--;
  } else {
    i = int(x - 0.5);
    if (x - 0.5 == Float_t(i) && i & 1) i++;    
  }
  return i;
#else
  return TMath::Nint(x);
#endif
}

GPUd() inline Bool_t AliHLTTPCCAMath::Finite(Float_t x)
{  
  return choice( 1, finite(x), finite(x) );
}

GPUd() inline Float_t AliHLTTPCCAMath::ATan2(Float_t y, Float_t x)
{ 
  return choice(atan2f(y,x), atan2(y,x), TMath::ATan2(y,x) );
}


GPUd() inline Float_t AliHLTTPCCAMath::Copysign(Float_t x, Float_t y)
{ 
#if defined( HLTCA_GPUCODE )
  return copysignf(x,y);
#else
  x = CAMath::Abs(x);
  return (y>=0) ?x : -x;
#endif
}


GPUd() inline Float_t AliHLTTPCCAMath::Sin(Float_t x)
{ 
  return choice( sinf(x), sin(x), TMath::Sin(x) );
}

GPUd() inline Float_t AliHLTTPCCAMath::Cos(Float_t x)
{ 
  return choice( cosf(x), cos(x), TMath::Cos(x) );
}

GPUd() inline Float_t AliHLTTPCCAMath::Tan(Float_t x)
{ 
  return choice( tanf(x), tan(x), TMath::Tan(x) );
}

GPUd() inline Float_t AliHLTTPCCAMath::Min(Float_t x, Float_t y)
{ 
  return choice( fminf(x,y),  (x<y ?x :y), TMath::Min(x,y) );
}

GPUd() inline Float_t AliHLTTPCCAMath::Max(Float_t x, Float_t y)
{ 
  return choice( fmaxf(x,y),  (x>y ?x :y), TMath::Max(x,y) );
}

GPUd() inline Int_t AliHLTTPCCAMath::Min(Int_t x, Int_t y)
{ 
  return choice( min(x,y),  (x<y ?x :y), TMath::Min(x,y) );
}

GPUd() inline Int_t AliHLTTPCCAMath::Max(Int_t x, Int_t y)
{ 
  return choice( max(x,y),  (x>y ?x :y), TMath::Max(x,y) );
}

GPUd() inline Float_t AliHLTTPCCAMath::Sqrt(Float_t x )
{ 
  return choice( sqrtf(x), sqrt(x), TMath::Sqrt(x) );
}

GPUd() inline Float_t AliHLTTPCCAMath::Abs(Float_t x )
{ 
  return choice( fabsf(x), fabs(x), TMath::Abs(x) );
}

GPUd() inline Double_t AliHLTTPCCAMath::Abs(Double_t x )
{ 
  return choice( fabs(x), fabs(x), TMath::Abs(x) );
}

GPUd() inline Int_t AliHLTTPCCAMath::Abs(Int_t x )
{ 
  return choice( abs(x), (x>=0 ?x :-x), TMath::Abs(x) );
}

GPUd() inline Float_t AliHLTTPCCAMath::ASin(Float_t x )
{ 
  return choice( asinf(x), asin(x), TMath::ASin(x) );
}


GPUd() inline Int_t AliHLTTPCCAMath::mul24( Int_t a, Int_t b )
{
  return choice( __mul24(a,b), a*b, a*b );
}

GPUd() inline Float_t AliHLTTPCCAMath::fmul_rz( Float_t a, Float_t b )
{
  return choice( __fmul_rz(a,b), a*b, a*b );
}


GPUd()  inline Int_t AliHLTTPCCAMath::atomicExch( Int_t *addr, Int_t val)
{
#if defined( HLTCA_GPUCODE )
  return ::atomicExch(addr, val );
#else  
  Int_t old = *addr;
  *addr = val;
  return old;
#endif
}

GPUd()  inline Int_t AliHLTTPCCAMath::atomicAdd ( Int_t *addr, Int_t val)
{
#if defined( HLTCA_GPUCODE )
  return ::atomicAdd(addr, val );
#else  
  Int_t old = *addr;
  *addr += val;
  return old;
#endif
}

GPUd()  inline Int_t AliHLTTPCCAMath::atomicMax ( Int_t *addr, Int_t val)
{
#if defined( HLTCA_GPUCODE )
  return ::atomicMax(addr, val );
#else  
  Int_t old = *addr;
  if( *addr< val ) *addr = val;
  return old;
#endif
}

GPUd()  inline Int_t AliHLTTPCCAMath::atomicMin ( Int_t *addr, Int_t val)
{
#if defined( HLTCA_GPUCODE )
  return ::atomicMin(addr, val );
#else  
  Int_t old = *addr;
  if( *addr> val ) *addr = val;
  return old;
#endif
}

#undef CHOICE

#endif
