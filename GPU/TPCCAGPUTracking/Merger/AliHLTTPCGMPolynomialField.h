//-*- Mode: C++ -*-
//*************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************


#ifndef AliHLTTPCGMPolynomialField_H
#define AliHLTTPCGMPolynomialField_H

#include "AliHLTTPCCADef.h"

/**
 * @class AliHLTTPCGMPolynomialField
 *
 */


class AliHLTTPCGMPolynomialField
{
public:

  AliHLTTPCGMPolynomialField() : fNominalBz(0.) {
    Reset();
  }

  void Reset();

  void Set( float nominalBz, const float *Bx, const float *By, const float *Bz );

  GPUd() float GetNominalBz() const { return fNominalBz; }

  GPUd() void  GetField( float x, float y, float z, float B[3] ) const;

  GPUd() float GetFieldBz( float x, float y, float z ) const;

  void Print() const;

  static const int fkM = 10; // number of coefficients

  static void GetPolynoms( float x, float y, float z, float f[fkM] );

private:

  float fNominalBz; // nominal constant field value in [kG * 2.99792458E-4 GeV/c/cm]
  float fBx[fkM]; // polynomial coefficients
  float fBy[fkM];
  float fBz[fkM];
};


inline void AliHLTTPCGMPolynomialField::Reset()
{
  fNominalBz = 0.f;
  for( int i=0; i<fkM; i++){
    fBx[i] = 0.f;
    fBy[i] = 0.f;
    fBz[i] = 0.f;
  }
}

inline void AliHLTTPCGMPolynomialField::Set( float nominalBz, const float *Bx, const float *By, const float *Bz )
{
  if( !Bx || !By || !Bz ){ Reset(); return; }
  fNominalBz = nominalBz;  
  for( int i=0; i<fkM; i++){
    fBx[i] = Bx[i];
    fBy[i] = By[i];
    fBz[i] = Bz[i];
  }
}

inline void AliHLTTPCGMPolynomialField::GetPolynoms( float x, float y, float z, float f[fkM] )
{
  f[0]=1.f;
  f[1]=x;   f[2]=y;   f[3]=z;
  f[4]=x*x; f[5]=x*y; f[6]=x*z; f[7]=y*y; f[8]=y*z; f[9]=z*z;
}

GPUd() inline void AliHLTTPCGMPolynomialField::GetField( float x, float y, float z, float B[3] ) const
{
  const float f[fkM] = { 1.f, x, y, z, x*x, x*y, x*z, y*y, y*z, z*z };
  float bx = 0.f, by = 0.f, bz = 0.f;
  for( int i=0; i<fkM; i++){
    bx += fBx[i]*f[i];
    by += fBy[i]*f[i];
    bz += fBz[i]*f[i];
  }
  B[0] = bx;
  B[1] = by;
  B[2] = bz;
}

GPUd() inline float AliHLTTPCGMPolynomialField::GetFieldBz( float x, float y, float z ) const
{
  const float f[fkM] = { 1.f, x, y, z, x*x, x*y, x*z, y*y, y*z, z*z };
  float bz = 0.f;
  for( int i=0; i<fkM; i++){
    bz += fBz[i]*f[i];
  }
  return bz;
}

#endif 
