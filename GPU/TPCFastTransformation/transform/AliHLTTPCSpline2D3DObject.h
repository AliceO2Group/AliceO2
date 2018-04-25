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
  AliHLTTPCSpline2D3DObject( const AliHLTTPCSpline2D3DObject &);

  /** assignment operator */    
  AliHLTTPCSpline2D3DObject& operator=( const AliHLTTPCSpline2D3DObject &);

  /** destructor */
  ~AliHLTTPCSpline2D3DObject(){}
 
  /** deinitialization */
  void  Reset();

  /** initialisation */
  void InitGrid( Int_t nPointsA, Int_t nPointsB );

  /** setters */
  void SetMinA( Float_t v ){ fMinA = v; }
  void SetStepA( Float_t v ){ fStepA = v; }
  void SetMinB( Float_t v ){ fMinB = v; }
  void SetStepB( Float_t v ){ fStepB = v; }
  
  /**  fill the XYZ values to the grid*/
  
  void SetGridValue( Int_t iBin, Float_t x, Float_t y, Float_t z );
  void SetGridValue( Int_t iBin, const Float_t XYZ[] );

  /** getters **/

  Int_t GetNPointsA() const { return fNA; }
  Int_t GetNPointsB() const { return fNB; }
  Int_t GetNPoints() const { return fNA*fNB; }

  Float_t GetMinA() const { return fMinA; }
  Float_t GetStepA() const { return fStepA; }

  Float_t GetMinB() const { return fMinB; }
  Float_t GetStepB() const { return fStepB; }

  void GetGridValue( Int_t iBin, Float_t &x, Float_t &y, Float_t &z ) const;
  void GetGridValue( Int_t iBin, Float_t XYZ[] ) const;

 private:

  Int_t fNA; // N points A axis
  Int_t fNB; // N points B axis
  Int_t fN; // N points (== fNA*fNB )
  Float_t fMinA; // min A axis
  Float_t fMinB; // min B axis
  Float_t fStepA; // step between points A axis
  Float_t fStepB; // step between points B axis
  TArrayF fXYZ; // array of points, {X,Y,Z,0} values

 public:

  ClassDef(AliHLTTPCSpline2D3DObject, 1)
};


inline AliHLTTPCSpline2D3DObject::AliHLTTPCSpline2D3DObject()
				 : TObject(), fNA(0), fNB(0), fN(0), fMinA(0), fMinB(0), fStepA(0), fStepB(0), fXYZ()
{
}

inline AliHLTTPCSpline2D3DObject::AliHLTTPCSpline2D3DObject( const AliHLTTPCSpline2D3DObject &v)
				 : TObject(v), fNA(v.fNA), fNB(v.fNB), fN(v.fN), fMinA(v.fMinA), fMinB(v.fMinB), fStepA(v.fStepA), fStepB(v.fStepB), fXYZ(v.fXYZ)
{
  // constructor  
}

inline AliHLTTPCSpline2D3DObject& AliHLTTPCSpline2D3DObject::operator=( const AliHLTTPCSpline2D3DObject &v)
{
  // assignment operator 
  new (this) AliHLTTPCSpline2D3DObject( v );
  return *this;
}
  
inline void AliHLTTPCSpline2D3DObject::Reset()
{
  // reset
  InitGrid(0,0);
  fMinA = fMinB = fStepA = fStepB = 0.f;
}

inline void AliHLTTPCSpline2D3DObject::InitGrid( Int_t  nBinsA, Int_t  nBinsB )
{
  // initialisation
  fNA = nBinsA;
  fNB = nBinsB;
  fN = fNA*fNB;
  fXYZ.Set(fN*3);
}

inline void AliHLTTPCSpline2D3DObject::SetGridValue(Int_t iBin, Float_t x, Float_t y, Float_t z)
{
  if( iBin<0 || iBin>=fN ) return;
  Int_t ind3 = iBin*3;
  fXYZ[ind3] = x;
  fXYZ[ind3+1] = y;
  fXYZ[ind3+2] = z;
}

inline void AliHLTTPCSpline2D3DObject::SetGridValue( Int_t iBin, const Float_t XYZ[] )
{
  SetGridValue( iBin, XYZ[0], XYZ[1], XYZ[2] );
}

inline void AliHLTTPCSpline2D3DObject::GetGridValue( Int_t iBin, Float_t &x, Float_t &y, Float_t &z ) const 
{
  if( iBin<0 || iBin>=fN ) return;
  Int_t ind3 = iBin*3;
  x = fXYZ[ind3];
  y = fXYZ[ind3+1];
  z = fXYZ[ind3+2];
}

inline void AliHLTTPCSpline2D3DObject::GetGridValue( Int_t iBin, Float_t XYZ[] ) const
{
  GetGridValue( iBin, XYZ[0], XYZ[1], XYZ[2] ); 
}


#endif
