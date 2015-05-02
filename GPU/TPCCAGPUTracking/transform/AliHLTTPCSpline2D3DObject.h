#ifndef ALIHLTTPCSPLINE2D3DOBJECT_H
#define ALIHLTTPCSPLINE2D3DOBJECT_H

//* This file is property of and copyright by the ALICE HLT Project        * 
//* ALICE Experiment at CERN, All rights reserved.                         *
//* See cxx source for full Copyright notice                               *

/** @file   AliHLTTPCSpline2D3DObject.h
    @author Sergey Gorbunov
    @date   
    @brief
*/

// see below for class documentation
// or
// refer to README to build package
// or
// visit http://web.ift.uib.no/~kjeks/doc/alice-hlt

#include "TObject.h"
#include "TArrayF.h"

/**
 * @class AliHLTTPCSpline2D3DObject
 *
 * The class presents spline interpolation for 2D->3D function (a,b)->(x,y,z) to be stored in ROOT file
 * 
 * @ingroup alihlt_tpc_components
 */

class AliHLTTPCSpline2D3DObject: public TObject{
 public:
  /** standard constructor */    
  AliHLTTPCSpline2D3DObject();

  /** constructor */    
  AliHLTTPCSpline2D3DObject( Float_t minA, Int_t  nBinsA, Float_t  stepA, Float_t  minB, Int_t  nBinsB, Float_t  stepB );

  /** destructor */
  ~AliHLTTPCSpline2D3DObject();
 
  /** initialisation */
  void Init( Float_t minA, Int_t  nBinsA, Float_t  stepA, Float_t  minB, Int_t  nBinsB, Float_t  stepB );
  
  /**  Filling of values */
  void Set( Int_t ind, Float_t x, Float_t y, Float_t z );
  void Set( Int_t ind, const Float_t XYZ[] );

  /** Getters **/

  Int_t GetNPointsA() const { return fNA; }
  Int_t GetNPointsB() const { return fNB; }
  Int_t GetNPoints() const { return fNA*fNB; }

  Float_t GetMinA() const { return fMinA; }
  Float_t GetStepA() const { return fStepA; }

  Float_t GetMinB() const { return fMinB; }
  Float_t GetStepB() const { return fStepB; }

  void Get( Int_t ind, Float_t &x, Float_t &y, Float_t &z ) const;
  void Get( Int_t ind, Float_t XYZ[] ) const;

 private:

  Int_t fNA; // N points A axis
  Int_t fNB; // N points A axis
  Float_t fMinA; // min A axis
  Float_t fMinB; // min B axis
  Float_t fStepA; // step between points A axis
  Float_t fStepB; // step between points B axis
  TArrayF fXYZ; // array of points, {X,Y,Z,0} values
 public:
  ClassDef(AliHLTTPCSpline2D3DObject, 1)
};

inline AliHLTTPCSpline2D3DObject::AliHLTTPCSpline2D3DObject()
				 : fNA(0), fNB(0), fMinA(0), fMinB(0), fStepA(0), fStepB(0), fXYZ()
{
}

inline AliHLTTPCSpline2D3DObject::AliHLTTPCSpline2D3DObject( Float_t minA, Int_t  nBinsA, Float_t  stepA, Float_t  minB, Int_t  nBinsB, Float_t  stepB )
				 : fNA(0), fNB(0), fMinA(0), fMinB(0), fStepA(0), fStepB(0), fXYZ()
{
  Init(minA, nBinsA, stepA, minB, nBinsB, stepB);
}


inline void AliHLTTPCSpline2D3DObject::Set(Int_t ind, Float_t x, Float_t y, Float_t z)
{
  Int_t ind3 = ind*3;
  fXYZ[ind3] = x;
  fXYZ[ind3+1] = y;
  fXYZ[ind3+2] = z;
}

inline void AliHLTTPCSpline2D3DObject::Set( Int_t ind, const Float_t XYZ[] )
{
  Set( ind, XYZ[0], XYZ[1], XYZ[2] );
}

inline void AliHLTTPCSpline2D3DObject::Get( Int_t ind, Float_t &x, Float_t &y, Float_t &z ) const 
{
  Int_t ind3 = ind*3;
  x = fXYZ[ind3];
  y = fXYZ[ind3+1];
  z = fXYZ[ind3+2];
}

inline void AliHLTTPCSpline2D3DObject::Get( Int_t ind, Float_t XYZ[] ) const
{
  Get( ind, XYZ[0], XYZ[1], XYZ[2] ); 
}


#endif
