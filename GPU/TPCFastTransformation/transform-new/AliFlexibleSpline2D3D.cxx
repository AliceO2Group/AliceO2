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


#include "AliFlexibleSpline2D3D.h"
#include "Vc/Vc"


void AliFlexibleSpline2D3D::GetSplineVec( const float *data, float u, float v, float &x, float &y, float &z ) const
{
   // cout<<Vc::float_v::Size<<endl;
  const AliFlexibleSpline1D &gridU = GetGridU();
  const AliFlexibleSpline1D &gridV = GetGridV();
  int nu = gridU.GetNKnots();
  int iu = gridU.GetKnotIndex( u );
  int iv = gridV.GetKnotIndex( v );
  const AliFlexibleSpline1D::TKnot &knotU =  gridU.GetKnot( iu );
  const AliFlexibleSpline1D::TKnot &knotV =  gridV.GetKnot( iv );
  
  const float *dataV0 = data + (nu*(iv-1)+iu-1)*3;
  const float *dataV1 = dataV0 + 3*nu;
  const float *dataV2 = dataV0 + 6*nu;
  const float *dataV3 = dataV0 + 9*nu;

  Vc::float_v dataV[3+1];
  for( int i=0, i4=0; i<3; i++,i4+=4){
    Vc::float_v dt0( dataV0 + i4 );
    Vc::float_v dt1( dataV1 + i4 );
    Vc::float_v dt2( dataV2 + i4 );
    Vc::float_v dt3( dataV3 + i4 );
    dataV[i] = gridV.GetSpline( v, knotV, dt0, dt1, dt2, dt3);
  }
  
  Vc::float_v dataU0( reinterpret_cast< const float *>(dataV) + 0 );
  Vc::float_v dataU1( reinterpret_cast< const float *>(dataV) + 3 );
  Vc::float_v dataU2( reinterpret_cast< const float *>(dataV) + 6 );
  Vc::float_v dataU3( reinterpret_cast< const float *>(dataV) + 9 );


  Vc::float_v res = gridU.GetSpline( u, knotU, dataU0, dataU1, dataU2, dataU3 );

  x = res[0];
  y = res[1];
  z = res[2];
  
}

