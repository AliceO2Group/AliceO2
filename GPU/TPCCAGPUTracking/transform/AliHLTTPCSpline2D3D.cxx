//**************************************************************************
//* This file is property of and copyright by the ALICE HLT Project        *
//* ALICE Experiment at CERN, All rights reserved.                         *
//*                                                                        *
//* Primary Authors: Sergey Gorbunov <sergey.gorbunov@cern.ch>             *
//*                  for The ALICE HLT Project.                            *
//*                                                                        *
//* Permission to use, copy, modify and distribute this software and its   *
//* documentation strictly for non-commercial purposes is hereby granted   *
//* without fee, provided that the above copyright notice appears in all   *
//* copies and that both the copyright notice and this permission notice   *
//* appear in the supporting documentation. The authors make no claims     *
//* about the suitability of this software for any purpose. It is          *
//* provided "as is" without express or implied warranty.                  *
//**************************************************************************

/** @file   AliHLTTPCSpline2D3D.cxx
    @author Sergey Gorbubnov
    @date   
    @brief 
*/


#include "AliHLTTPCFastTransform.h"
#include "AliTPCTransform.h"
#include "AliTPCParam.h"
#include "AliTPCcalibDB.h"
 
#include <iostream>
#include <iomanip>

using namespace std;





void AliHLTTPCSpline2D3D::Init(Float_t minA,Float_t  maxA, Int_t  nBinsA, Float_t  minB,Float_t  maxB, Int_t  nBinsB)
{
  //
  // Initialisation
  //

  if( maxA<= minA ) maxA = minA+1;
  if( maxB<= minB ) maxB = minB+1;
  if( nBinsA <3 ) nBinsA = 3;
  if( nBinsB <3 ) nBinsB = 3;

  fNA = nBinsA;
  fNB = nBinsB;
  fN = fNA*fNB;
  fMinA = minA;
  fMinB = minB;

  fStepA = (maxA-minA)/(nBinsA-1);
  fStepB = (maxB-minB)/(nBinsB-1);
  fScaleA = 1./fStepA;
  fScaleB = 1./fStepB;

  delete[] fX;
  delete[] fY;
  delete[] fZ;
  fX = new Float_t [fN];
  fY = new Float_t [fN];
  fZ = new Float_t [fN];
  memset ( fX, 0, fN*sizeof(Float_t) );
  memset ( fY, 0, fN*sizeof(Float_t) );
  memset ( fZ, 0, fN*sizeof(Float_t) );
}

void AliHLTTPCSpline2D3D::Fill(void (*func)(Float_t a, Float_t b, Float_t xyz[]) )
{
  //
  // Filling
  //

  for( Int_t i=0; i<GetNPoints(); i++){
    Float_t a, b, xyz[3];
    GetAB(i,a,b);
    (*func)(a,b,xyz);
    Fill(i,xyz);
  }
}





void AliHLTTPCSpline2D3D::GetValue(Float_t A, Float_t B, Float_t XYZ[] ) const
{
  //
  //  Get Interpolated value at A,B 
  //

  Float_t lA = (A-fMinA)*fScaleA;
  Int_t iA = ((int) lA)-1;
  bool splineA3 = 0;
  if( iA<0 ) iA=0;
  else if( iA>fNA-4 ) iA = fNA-3;
  else splineA3 = 1;

  Float_t lB = (B-fMinB)*fScaleB;
  Int_t iB = ((int) lB)-1;
  bool splineB3 = 0;
  if( iB<0 ) iB=0;
  else if( iB>fNB-4 ) iB = fNB-3;
  else splineB3 = 1;

  Float_t da = lA-iA-1;
  Float_t db = lB-iB-1;

  
  Float_t v[3][4];
  Int_t ind = iA*fNB  + iB;
  for( Int_t i=0; i<3+splineA3; i++ ){
    if( splineB3 ){
      v[0][i] = GetSpline3(fX+ind,db); 
      v[1][i] = GetSpline3(fY+ind,db); 
      v[2][i] = GetSpline3(fZ+ind,db); 
    } else {
      v[0][i] = GetSpline2(fX+ind,db); 
      v[1][i] = GetSpline2(fY+ind,db); 
      v[2][i] = GetSpline2(fZ+ind,db); 
    }
    ind+=fNB;
  } 
  if( splineA3 ){
    XYZ[0] =  GetSpline3(v[0],da);
    XYZ[1] =  GetSpline3(v[1],da);
    XYZ[2] =  GetSpline3(v[2],da);
  } else {
    XYZ[0] =  GetSpline2(v[0],da);
    XYZ[1] =  GetSpline2(v[1],da);
    XYZ[2] =  GetSpline2(v[2],da);
  }
}
