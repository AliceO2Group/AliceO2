//-*- Mode: C++ -*-
// $Id: AliHLTTPCGMFastField.h 39008 2010-02-18 17:33:32Z sgorbuno $
// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************


#ifndef AliHLTTPCGMFastField_H
#define AliHLTTPCGMFastField_H

#include "AliHLTTPCCAMath.h"

/**
 * @class AliHLTTPCGMFastField
 *
 */

class AliHLTTPCGMFastField
{
public:

  AliHLTTPCGMFastField(){ Init(); }
  
  void Init();

  float GetBx( float x, float y, float z ) const;
  float GetBy( float x, float y, float z ) const;
  float GetBz( float x, float y, float z ) const;

  void DumpField( const char *fileName="field.root" ) const;

private:
  
  float fBx[6];
  float fBy[6];
  float fBz[6];
};

inline void AliHLTTPCGMFastField::Init()
{
  const float cBx[6] = { 0, 0, 0, 0, 0, 0};
  const float cBy[6] = { 0, 0, 0, 0, 0, 0};
  const float cBz[6] = { 0.999286, -4.54386e-7, 2.32950e-5, -2.99912e-7, -2.03442e-8, 9.71402e-8 };
  for( int i=0; i<6; i++ ){
    fBx[i] = cBx[i];
    fBy[i] = cBy[i];
    fBz[i] = cBz[i];
  }
}

inline float AliHLTTPCGMFastField::GetBx( float x, float y, float z ) const
{
  float r2 = x * x + y * y;
  float r  = sqrt( r2 );
  const float *c = fBx;
  return ( c[0] + c[1]*z  + c[2]*r  + c[3]*z*z + c[4]*z*r + c[5]*r2 );
}

inline float AliHLTTPCGMFastField::GetBy( float x, float y, float z ) const
{
  float r2 = x * x + y * y;
  float r  = sqrt( r2 );
  const float *c = fBy;
  return ( c[0] + c[1]*z  + c[2]*r  + c[3]*z*z + c[4]*z*r + c[5]*r2 );
}

inline float AliHLTTPCGMFastField::GetBz( float x, float y, float z ) const
{
  float r2 = x * x + y * y;
  float r  = sqrt( r2 );
  const float *c = fBz;
  return ( c[0] + c[1]*z  + c[2]*r  + c[3]*z*z + c[4]*z*r + c[5]*r2 );
}

#endif 
