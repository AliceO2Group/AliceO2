// $Id: AliHLTTPCGMSliceTrack.cxx 41769 2010-06-16 13:58:00Z sgorbuno $
// **************************************************************************
// This file is property of and copyright by the ALICE HLT Project          *
// ALICE Experiment at CERN, All rights reserved.                           *
//                                                                          *
// Primary Authors: Sergey Gorbunov <sergey.gorbunov@kip.uni-heidelberg.de> *
//                  for The ALICE HLT Project.                              *
//                                                                          *
// Permission to use, copy, modify and distribute this software and its     *
// documentation strictly for non-commercial purposes is hereby granted     *
// without fee, provided that the above copyright notice appears in all     *
// copies and that both the copyright notice and this permission notice     *
// appear in the supporting documentation. The authors make no claims       *
// about the suitability of this software for any purpose. It is            *
// provided "as is" without express or implied warranty.                    *
//                                                                          *
//***************************************************************************

#include "AliHLTTPCGMSliceTrack.h"
#include "AliHLTTPCCAMath.h"
#include "AliHLTTPCGMBorderTrack.h"
#include "AliHLTTPCGMTrackLinearisation.h"
#include "Riostream.h"
#include "AliHLTTPCCAParam.h"
#include <cmath>



bool AliHLTTPCGMSliceTrack::FilterErrors( AliHLTTPCCAParam &param, float maxSinPhi )
{
  float lastX = fOrigTrack->Cluster(fOrigTrack->NClusters()-1 ).GetX();

  const int N = 3;
  const float *cy = param.GetParamS0Par(0,0);
  const float *cz = param.GetParamS0Par(1,0);

  float bz = -param.ConstBz();
  const float kZLength = 250.f - 0.275f;

  float trDzDs2 = fDzDs*fDzDs;
  float k  = fQPt*bz;               
  float dx = .33*(lastX - fX);  
  float kdx = k*dx;
  float dxBz = dx * bz;
  float kdx205 = 2.f+kdx*kdx*0.5f;

  {
    float secPhi2 = fSecPhi*fSecPhi;
    float zz = fabs( kZLength - fabs(fZ) );	
    float zz2 = zz*zz;
    float angleY2 = secPhi2 - 1.f; 
    float angleZ2 = trDzDs2 * secPhi2 ;
    
    float cy0 = cy[0] + cy[1]*zz + cy[3]*zz2;
    float cy1 = cy[2] + cy[5]*zz;
    float cy2 = cy[4];
    float cz0 = cz[0] + cz[1]*zz + cz[3]*zz2;
    float cz1 = cz[2] + cz[5]*zz;
    float cz2 = cz[4];
    
    fC0 = fabs( cy0 + angleY2 * ( cy1 + angleY2*cy2 ) );
    fC2 = fabs( cz0 + angleZ2 * ( cz1 + angleZ2*cz2 ) );    

    fC3 = 0;
    fC5 = 1;
    fC7 = 0;
    fC9 = 1;
    fC10 = 0;
    fC12 = 0;
    fC14 = 10;
  }

  for( int iStep=0; iStep<N; iStep++ ){    

    float err2Y, err2Z;

    { // transport block
 
      float ex = fCosPhi; 
      float ey = fSinPhi;
      float ey1 = kdx + ey;
      if( fabs( ey1 ) > maxSinPhi ) return 0;

      float ss = ey + ey1;      
      float ex1 = sqrt(1.f - ey1*ey1);
          
      float cc = ex + ex1;  
      float dxcci = dx / cc;
      
      float dy = dxcci * ss;
      float norm2 = 1.f + ey*ey1 + ex*ex1;
      float dl = dxcci * sqrt( norm2 + norm2 );
     
      float dS;   
      {
	float dSin = 0.5f*k*dl;
	float a = dSin*dSin;
	const float k2 = 1.f/6.f;
	const float k4 = 3.f/40.f;
	dS = dl + dl*a*(k2 + a*(k4 ));//+ k6*a) );
      }
 
      float dz = dS * fDzDs;      
      float ex1i =1.f/ex1;
      float z = fZ+ dz;
      {	
	float secPhi2 = ex1i*ex1i;
	float zz = fabs( kZLength - fabs(z) );	
	float zz2 = zz*zz;
	float angleY2 = secPhi2 - 1.f; 
	float angleZ2 = trDzDs2 * secPhi2 ;

	float cy0 = cy[0] + cy[1]*zz + cy[3]*zz2;
	float cy1 = cy[2] + cy[5]*zz;
	float cy2 = cy[4];
	float cz0 = cz[0] + cz[1]*zz + cz[3]*zz2;
	float cz1 = cz[2] + cz[5]*zz;
	float cz2 = cz[4];
	
	err2Y = fabs( cy0 + angleY2 * ( cy1 + angleY2*cy2 ) );
	err2Z = fabs( cz0 + angleZ2 * ( cz1 + angleZ2*cz2 ) );
      }

      float hh = kdx205 * dxcci*ex1i; 
      float h2 = hh * fSecPhi;

      fX+=dx;      
      fY+= dy;
      fZ+= dz;
      fSinPhi = ey1;
      fCosPhi = ex1;
      fSecPhi = ex1i;
    
      float h4 = bz*dxcci*hh;
      
      float c20 = fC3;
      float c22 = fC5;
      float c31 = fC7;
      float c33 = fC9;
      float c40 = fC10;
      float c42 = fC12;
      float c44 = fC14;
      
      float c20ph4c42 =  c20 + h4*c42;
      float h2c22 = h2*c22;
      float h4c44 = h4*c44;
      float n7 = c31 + dS*c33;
      float n10 = c40 + h2*c42 + h4c44;
      float n12 = c42 + dxBz*c44;
      
      
      fC0+= h2*h2c22 + h4*h4c44 + 2.f*( h2*c20ph4c42  + h4*c40 );
      
      fC3 = c20ph4c42 + h2c22  + dxBz*n10;
      fC10 = n10;
      
      fC5 = c22 + dxBz*( c42 + n12 );
      fC12 = n12;
      
      fC2+= dS*(c31 + n7);
      fC7 = n7; 
      
   } // end transport block 


    // Filter block

    float 
      c00 = fC0,
      c11 = fC2,
      c20 = fC3,
      c31 = fC7,
      c40 = fC10;
                    
    float mS0 = 1.f/(err2Y + c00);    
    float mS2 = 1.f/(err2Z + c11);
            
    // K = CHtS
    
    float k00, k11, k20, k31, k40;
    
    k00 = c00 * mS0;
    k20 = c20 * mS0;
    k40 = c40 * mS0;
    
    fC0 -= k00 * c00 ;
    fC5 -= k20 * c20 ;
    fC10 -= k00 * c40 ;
    fC12 -= k40 * c20 ;
    fC3 -= k20 * c00 ;
    fC14 -= k40 * c40 ;
        
    k11 = c11 * mS2;
    k31 = c31 * mS2;
    
    fC7 -= k31 * c11;
    fC2 -= k11 * c11;
    fC9 -= k31 * c31;   
  }

  //* Check that the track parameters and covariance matrix are reasonable

  bool ok = 1;
  
  const float *c = &fX;
  for ( int i = 0; i < 17; i++ ) ok = ok && finite( c[i] );

  if ( fC0 <= 0.f || fC2 <= 0.f || fC5 <= 0.f || fC9 <= 0.f || fC14 <= 0.f 
       || fC0 > 5.f || fC2 > 5.f || fC5 > 2.f || fC9 > 2.f             ) ok = 0;

  if( ok ){
    ok = ok 
      && ( fC3*fC3<=fC5*fC0 )
      && ( fC7*fC7<=fC9*fC2 )
      && ( fC10*fC10<=fC14*fC0 )
      && ( fC12*fC12<=fC14*fC5 );
  }
 
  return ok;
}



bool AliHLTTPCGMSliceTrack::TransportToX( float x, float Bz, AliHLTTPCGMBorderTrack &b, float maxSinPhi, bool doCov ) const 
{
  Bz = -Bz;
  float ex = fCosPhi;
  float ey = fSinPhi;
  float k  = fQPt*Bz;
  float dx = x - fX;
  float ey1 = k*dx + ey;
  
  if( fabs( ey1 ) > maxSinPhi ) return 0;

  float ex1 = sqrt( 1.f - ey1 * ey1 );
  float dxBz = dx * Bz;
    
  float ss = ey + ey1;
  float cc = ex + ex1;  
  float dxcci = dx / cc;
  float norm2 = 1.f + ey*ey1 + ex*ex1;

  float dy = dxcci * ss;

  float dS;    
  {
    float dl = dxcci * sqrt( norm2 + norm2 );
    float dSin = 0.5f*k*dl;
    float a = dSin*dSin;
    const float k2 = 1.f/6.f;
    const float k4 = 3.f/40.f;
    //const float k6 = 5.f/112.f;
    dS = dl + dl*a*(k2 + a*(k4 ));//+ k6*a) );
  }
  
  float dz = dS * fDzDs;

  b.SetPar(0, fY + dy );
  b.SetPar(1, fZ + dz );
  b.SetPar(2, ey1 );
  b.SetPar(3, fDzDs);
  b.SetPar(4, fQPt);

  if (!doCov) return(1);

  float ex1i = 1.f/ex1;
  float hh = dxcci*ex1i*norm2; 
  float h2 = hh *fSecPhi;
  float h4 = Bz*dxcci*hh;
  
  float c20 = fC3;
  float c22 = fC5;  
  float c31 = fC7;  
  float c33 = fC9;
  float c40 = fC10;  
  float c42 = fC12;
  float c44 = fC14;

  float c20ph4c42 =  c20 + h4*c42;
  float h2c22 = h2*c22;
  float h4c44 = h4*c44;
  float n7 = c31 + dS*c33;
  
  b.SetCov(0, fC0 + h2*h2c22 + h4*h4c44 + 2.f*( h2*c20ph4c42  + h4*c40 ));
  b.SetCov(1, fC2+ dS*(c31 + n7) );
  b.SetCov(2, c22 + dxBz*( c42 + c42 + dxBz*c44 ));
  b.SetCov(3, c33);
  b.SetCov(4, c44);
  b.SetCovD(0, c20ph4c42 + h2c22  + dxBz*(c40 + h2*c42 + h4c44) );
  b.SetCovD(1, n7 );   
  return 1;
}



bool AliHLTTPCGMSliceTrack::TransportToXAlpha( float newX, float sinAlpha, float cosAlpha, float Bz, AliHLTTPCGMBorderTrack &b, float maxSinPhi ) const 
{
  //* 

  float c00 = fC0;
  float c11 = fC2;
  float c20 = fC3;
  float c22 = fC5;  
  float c31 = fC7;  
  float c33 = fC9;
  float c40 = fC10;  
  float c42 = fC12;
  float c44 = fC14;

  float x,y;
  float z = fZ;
  float sinPhi = fSinPhi;
  float cosPhi = fCosPhi;
  float secPhi = fSecPhi;
  float dzds = fDzDs;
  float qpt = fQPt;

  // Rotate the coordinate system in XY on the angle alpha
  {
    float sP = sinPhi, cP = cosPhi;
    cosPhi =  cP * cosAlpha + sP * sinAlpha;
    sinPhi = -cP * sinAlpha + sP * cosAlpha;
    
    if ( CAMath::Abs( sinPhi ) > .999 || CAMath::Abs( cP ) < 1.e-2  ) return 0;
    
    secPhi = 1./cosPhi;
    float j0 = cP *secPhi;
    float j2 = cosPhi / cP;    
    x =   fX*cosAlpha +  fY*sinAlpha ;
    y =  -fX*sinAlpha +  fY*cosAlpha ;    
    
    c00 *= j0 * j0;
    c40 *= j0;
    
    c22 *= j2 * j2;    
    c42 *= j2;    
    if( cosPhi < 0.f ){ // rotate to 180'
      cosPhi = -cosPhi;
      secPhi = -secPhi;
      sinPhi = -sinPhi;
      dzds = -dzds;
      qpt = -qpt;      
      c20 = -c20;
      c31 = -c31;
      c40 = -c40;
   }
  }

  Bz = -Bz;
  float ex = cosPhi;
  float ey = sinPhi;
  float k  = qpt*Bz;
  float dx = newX - x;
  float ey1 = k*dx + ey;
  
  if( fabs( ey1 ) > maxSinPhi ) return 0;

  float ex1 = sqrt( 1.f - ey1 * ey1 );

  float dxBz = dx * Bz;
    
  float ss = ey + ey1;
  float cc = ex + ex1;  
  float dxcci = dx / cc;
  float norm2 = 1.f + ey*ey1 + ex*ex1;

  float dy = dxcci * ss;

  float dS;    
  {
    float dl = dxcci * sqrt( norm2 + norm2 );
    float dSin = 0.5f*k*dl;
    float a = dSin*dSin;
    const float k2 = 1.f/6.f;
    const float k4 = 3.f/40.f;
    //const float k6 = 5.f/112.f;
    dS = dl + dl*a*(k2 + a*(k4 ));//+ k6*a) );
  }
  
  float ex1i = 1.f/ex1;
  float dz = dS * dzds;

  float hh = dxcci*ex1i*norm2; 
  float h2 = hh * secPhi;
  float h4 = Bz*dxcci*hh;  

  float c20ph4c42 =  c20 + h4*c42;
  float h2c22 = h2*c22;
  float h4c44 = h4*c44;
  float n7 = c31 + dS*c33;
  
  b.SetPar(0, y + dy );
  b.SetPar(1, z + dz );
  b.SetPar(2, ey1 );
  b.SetPar(3, dzds);
  b.SetPar(4, qpt);
  
  b.SetCov(0, c00 + h2*h2c22 + h4*h4c44 + 2.f*( h2*c20ph4c42  + h4*c40 ));
  b.SetCov(1, c11 + dS*(c31 + n7) );
  b.SetCov(2, c22 + dxBz*( c42 + c42 + dxBz*c44 ));
  b.SetCov(3, c33);
  b.SetCov(4, c44);
  b.SetCovD(0, c20ph4c42 + h2c22  + dxBz*(c40 + h2*c42 + h4c44) );
  b.SetCovD(1, n7 ); 

  return 1;
}
