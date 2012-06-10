#ifndef ALIHLTTPCSPLINE2D3D_H
#define ALIHLTTPCSPLINE2D3D_H

//* This file is property of and copyright by the ALICE HLT Project        * 
//* ALICE Experiment at CERN, All rights reserved.                         *
//* See cxx source for full Copyright notice                               *

/** @file   AliHLTTPCSpline2D3D.h
    @author Sergey Gorbunov
    @date   
    @brief
*/

// see below for class documentation
// or
// refer to README to build package
// or
// visit http://web.ift.uib.no/~kjeks/doc/alice-hlt

#include"Rtypes.h"

class AliTPCParam;
class AliTPCTransform;

/**
 * @class AliHLTTPCSpline2D3D
 *
 * The class presents spline interpolation for 2D->3D function (a,b)->(x,y,z)
 * 
 * @ingroup alihlt_tpc_components
 */

class AliHLTTPCSpline2D3D{
 public:
  /** standard constructor */    
  AliHLTTPCSpline2D3D();           

  /** constructor */    
  AliHLTTPCSpline2D3D(Float_t minA, Float_t  maxA, Int_t  nBinsA, Float_t  minB, Float_t  maxB,  Int_t  nBinsB);

  /** destructor */
  ~AliHLTTPCSpline2D3D();
 
  /** initialisation */
  void Init(Float_t minA, Float_t  maxA, Int_t  nBinsA, Float_t  minB, Float_t  maxB,  Int_t  nBinsB);
  
  /**  Filling of points */
  void Fill(void (*func)(Float_t a, Float_t b, Float_t xyz[] ) );
  /**  Filling of points */
  void Fill(Int_t ind, Float_t x, Float_t y, Float_t z);    
  /**  Filling of points */
  void Fill(Int_t ind, Float_t XYZ[]);    
  /**  Filling of points */
  void Fill(Int_t ind, Double_t  XYZ[]);    

  /**  Get A,B by the point index */
  void GetAB(Int_t ind, Float_t &A, Float_t &B) const ;
  
  /**  Get Interpolated value at A,B */
  void GetValue(Float_t A, Float_t B, Float_t XYZ[]) const ;  
 
  /**  Get Interpolated value at A,B */
  void GetValue(Float_t A, Float_t B, Double_t XYZ[]) const ;  

  /**  Get size of the grid */
  Int_t  GetMapSize() const ;

  /**  Get N of point on the grid */
  Int_t GetNPoints() const ;

 private:

  /** copy constructor prohibited */
  AliHLTTPCSpline2D3D(const AliHLTTPCSpline2D3D&);
  /** assignment operator prohibited */
  AliHLTTPCSpline2D3D& operator=(const AliHLTTPCSpline2D3D&);

  /** spline 3-st order,  4 points, da = a - point 1 */
  static  Float_t GetSpline3(Float_t *v, Float_t da);

 /** spline 2-nd order, 3 points, da = a - point 1 */
  static  Float_t GetSpline2(Float_t *v, Float_t da);

  Int_t fNA; // N points A axis
  Int_t fNB; // N points A axis
  Int_t fN;  // N points total
  Float_t fMinA; // min A axis
  Float_t fMinB; // min B axis
  Float_t fStepA; // step between points A axis
  Float_t fStepB; // step between points B axis
  Float_t fScaleA; // scale A axis
  Float_t fScaleB; // scale B axis
  Float_t *fX; // array of points, X values
  Float_t *fY; // array of points, Y values
  Float_t *fZ; // array of points, Z values
};

inline AliHLTTPCSpline2D3D::AliHLTTPCSpline2D3D()
: fNA(0), fNB(0), fN(0), fMinA(0), fMinB(0), fStepA(0), fStepB(0), fScaleA(0), fScaleB(0), fX(0), fY(0),fZ(0)
{
}

inline AliHLTTPCSpline2D3D::AliHLTTPCSpline2D3D(Float_t minA, Float_t  maxA, Int_t  nBinsA, Float_t  minB, Float_t  maxB,  Int_t  nBinsB)
: fNA(0), fNB(0), fN(0), fMinA(0), fMinB(0), fStepA(0), fStepB(0), fScaleA(0), fScaleB(0), fX(0), fY(0),fZ(0)
{
  Init(minA, maxA, nBinsA, minB, maxB, nBinsB);
}

inline AliHLTTPCSpline2D3D::~AliHLTTPCSpline2D3D()
{
  delete[] fX;
  delete[] fY;
  delete[] fZ;
}

inline void AliHLTTPCSpline2D3D::Fill(Int_t ind, Float_t x, Float_t y, Float_t z)
{
  fX[ind] = x;
  fY[ind] = y;
  fZ[ind] = z;
}

inline void AliHLTTPCSpline2D3D::Fill(Int_t ind, Float_t XYZ[] )
{
  Fill( ind, XYZ[0], XYZ[1], XYZ[2] );
}

inline void AliHLTTPCSpline2D3D::Fill(Int_t ind, Double_t  XYZ[] )
{
  Fill( ind, XYZ[0], XYZ[1], XYZ[2] );
}

inline void AliHLTTPCSpline2D3D::GetAB(Int_t ind, Float_t &A, Float_t &B) const 
{
  A = fMinA + (ind / fNB)*fStepA;
  B = fMinB + (ind % fNB)*fStepB;
}

inline Int_t AliHLTTPCSpline2D3D::GetMapSize() const 
{
  return 3*sizeof(float)*fN; 
}

inline Int_t AliHLTTPCSpline2D3D::GetNPoints() const 
{ 
  return fN; 
}

inline Float_t AliHLTTPCSpline2D3D::GetSpline3(Float_t *v, Float_t x)
{
  Float_t dv = v[2]-v[1];  
  Float_t z0 = 0.5*(v[2]-v[0]);
  Float_t z1 = 0.5*(v[3]-v[1]);
  return x*x*( (z1-dv + z0-dv)*(x-1) - (z0-dv) ) + z0*x + v[1]; 
}

inline Float_t AliHLTTPCSpline2D3D::GetSpline2(Float_t *v, Float_t x)
{
  return 0.5*x*( ( v[0]+v[2] -v[1] -v[1] )*x + v[2]-v[0]) + v[1]; 
}

inline void AliHLTTPCSpline2D3D::GetValue(Float_t A, Float_t B, Double_t XYZ[]) const 
{
  float fxyz[3];
  GetValue(A,B,fxyz);
  for( Int_t i=0; i<3; i++ ) XYZ[i] = fxyz[i];
}

#endif
