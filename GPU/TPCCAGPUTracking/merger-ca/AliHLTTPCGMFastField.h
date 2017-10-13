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

/**
 * @class AliHLTTPCGMFastField
 *
 */

class AliHLTTPCGMFastField
{
public:

  AliHLTTPCGMFastField(){ Init(); }
  
  void Init();

  void GetField( float x, float y, float z, float B[] ) const;
  void DumpField( const char *fileName="field.root" ) const;

private:

  static const int fkM = 9;
  float fBx0;
  float fBy0;
  float fBz0; 
  float fBx[fkM];
  float fBy[fkM];
  float fBz[fkM];
};

inline void AliHLTTPCGMFastField::Init()
{
  const float cBx[fkM+1] = { -2.58322252193e-05, 2.25564940592e-06, -4.14718357433e-08, -2.75251750281e-06,
			   -8.72029382037e-09, 1.72417402577e-09, 3.19352068345e-07, -3.28086002810e-09, 5.64790381130e-10, 8.92192542068e-09 };

  const float cBy[fkM+1] = { 6.37950097371e-06, -4.46194050596e-08, 9.01212274584e-07, 8.26001087262e-06,
			   7.99017740860e-10, -7.45108241773e-09, 4.81764572680e-10, 8.35443714209e-10, 3.14677095048e-07, -1.18421328299e-09 };
  
  const float cBz[fkM+1] = { 9.99663949013e-01, -3.54553162651e-06, 7.73496958573e-06, -2.90551361104e-06,
			   1.69738939348e-07, 5.00871899511e-10, 2.10037196524e-08, 1.66827078374e-07, -2.64136179595e-09, -3.02637317873e-07 };
  fBx0 = cBx[0];
  fBy0 = cBy[0];
  fBz0 = cBz[0];
  for( int i=0; i<fkM; i++ ) fBx[i] = cBx[i+1];
  for( int i=0; i<fkM; i++ ) fBy[i] = cBy[i+1];
  for( int i=0; i<fkM; i++ ) fBz[i] = cBz[i+1];
}

inline void AliHLTTPCGMFastField::GetField( float x, float y, float z, float B[] ) const
{
  float f[fkM] = { x, y, z, x*x, x*y, x*z, y*y, y*z, z*z };
  float bx = 0.f, by = 0.f, bz = 0.f;
  for( int i=0; i<fkM; i++){
    bx+= fBx[i]*f[i];
    by+= fBy[i]*f[i];
    bz+= fBz[i]*f[i];
  }
  B[0] = fBx0 + bx;
  B[1] = fBy0 + by;
  B[2] = fBz0 + bz;
}

#endif 
