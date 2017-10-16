//-*- Mode: C++ -*-
// $Id: AliHLTTPCGMPolynomialField.h 39008 2010-02-18 17:33:32Z sgorbuno $
// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************


#ifndef AliHLTTPCGMPolynomialField_H
#define AliHLTTPCGMPolynomialField_H

/**
 * @class AliHLTTPCGMPolynomialField
 *
 */

class AliHLTTPCGMPolynomialField
{
public:

  AliHLTTPCGMPolynomialField(): fNominalBzkG(.001), fBx0(0.f), fBy0(0.f), fBz0(0.f) {
    Init(fNominalBzkG);
  }
  
  void Init( float NominalBzkG );  
  
  float GetNominalBzkG() const { return fNominalBzkG;}
  
  void GetField( float x, float y, float z, float B[] ) const;
  float GetFieldBz( float x, float y, float z ) const;

  void DumpField( const char *fileName="field.root" ) const;

private:

  static const int fkM = 10;

  float fNominalBzkG; // Bz field constant in kGaus  
  float fBx0;
  float fBy0;
  float fBz0; 
  float fBx[fkM-1];
  float fBy[fkM-1];
  float fBz[fkM-1];
};

inline void AliHLTTPCGMPolynomialField::Init( float NominalBzkG )
{
  const float cBx[fkM] = { -2.58322252193e-05,
			    2.25564940592e-06, -4.14718357433e-08, -2.75251750281e-06,
			   -8.72029382037e-09,  1.72417402577e-09,  3.19352068345e-07, -3.28086002810e-09,  5.64790381130e-10,  8.92192542068e-09 };

  const float cBy[fkM] = {  6.37950097371e-06,
			   -4.46194050596e-08,  9.01212274584e-07,  8.26001087262e-06,
			    7.99017740860e-10, -7.45108241773e-09,  4.81764572680e-10,  8.35443714209e-10,  3.14677095048e-07, -1.18421328299e-09 };
  
  const float cBz[fkM] = {  9.99663949013e-01, -3.54553162651e-06,  7.73496958573e-06, -2.90551361104e-06,
			    1.69738939348e-07,  5.00871899511e-10,  2.10037196524e-08,  1.66827078374e-07, -2.64136179595e-09, -3.02637317873e-07 };

  const double kCLight = 0.000299792458;

  fNominalBzkG = NominalBzkG; 

  double constBz = fNominalBzkG * kCLight;
  
  fBx0 = constBz*cBx[0];
  fBy0 = constBz*cBy[0];
  fBz0 = constBz*cBz[0];
  for( int i=1; i<fkM; i++ ) fBx[i-1] = constBz*cBx[i];
  for( int i=1; i<fkM; i++ ) fBy[i-1] = constBz*cBy[i];
  for( int i=1; i<fkM; i++ ) fBz[i-1] = constBz*cBz[i];
}

inline void AliHLTTPCGMPolynomialField::GetField( float x, float y, float z, float B[] ) const
{
  float f[fkM-1] = { x, y, z, x*x, x*y, x*z, y*y, y*z, z*z };
  float bx = 0.f, by = 0.f, bz = 0.f;
  for( int i=0; i<fkM-1; i++){
    bx+= fBx[i]*f[i];
    by+= fBy[i]*f[i];
    bz+= fBz[i]*f[i];
  }
  B[0] = fBx0 + bx;
  B[1] = fBy0 + by;
  B[2] = fBz0 + bz;
}

inline float AliHLTTPCGMPolynomialField::GetFieldBz( float x, float y, float z ) const
{
  float f[fkM-1] = { x, y, z, x*x, x*y, x*z, y*y, y*z, z*z };
  float bz = 0.f;
  for( int i=0; i<fkM-1; i++){
    bz+= fBz[i]*f[i];
  }
  return fBz0 + bz;
}

#endif 
