// @(#) $Id$
// **************************************************************************
// This file is property of and copyright by the ALICE HLT Project          *
// ALICE Experiment at CERN, All rights reserved.                           *
//                                                                          *
// Primary Authors: Sergey Gorbunov <sergey.gorbunov@kip.uni-heidelberg.de> *
//                  Ivan Kisel <kisel@kip.uni-heidelberg.de>                *
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


#include "AliHLTTPCCAParam.h"
#include "AliHLTTPCCAMath.h"

#if !defined(HLTCA_GPUCODE) & !defined(HLTCA_STANDALONE)
#include "AliTPCClusterParam.h"
#include "AliTPCcalibDB.h"
#include "Riostream.h"
#endif

#if !defined(HLTCA_GPUCODE)

GPUdi() AliHLTTPCCAParam::AliHLTTPCCAParam()
    : fISlice( 0 ), fNRows( 63 ), fAlpha( 0.174533 ), fDAlpha( 0.349066 ),
    fCosAlpha( 0 ), fSinAlpha( 0 ), fAngleMin( 0 ), fAngleMax( 0 ), fRMin( 83.65 ), fRMax( 133.3 ),
    fZMin( 0.0529937 ), fZMax( 249.778 ), fErrX( 0 ), fErrY( 0 ), fErrZ( 0.228808 ), fPadPitch( 0.4 ), fBzkG( -5.00668 ),
    fConstBz( -5.00668*0.000299792458 ), fHitPickUpFactor( 1. ),
      fMaxTrackMatchDRow( 4 ), fNeighboursSearchArea(3.), fTrackConnectionFactor( 3.5 ), fTrackChiCut( 3.5 ), fTrackChi2Cut( 10 ), fClusterError2CorrectionY(1.), fClusterError2CorrectionZ(1.),
  fMinNTrackClusters( -1 ), fMaxTrackQPt(1./MIN_TRACK_PT_DEFAULT), fNWays(1), fNWaysOuter(0), fAssumeConstantBz(false), fToyMCEventsFlag(false), fContinuousTracking(false), fSearchWindowDZDR(0.), fTrackReferenceX(1000.)
{
  // constructor

  float const kParamS0Par[2][3][6]=
    { 
      {  { 6.45913474727e-04, 2.51547407970e-05, 1.57551113516e-02, 1.99872811635e-08, -5.86769729853e-03, 9.16301505640e-05 }, 
	 { 9.71546804067e-04, 1.70938055817e-05, 2.17084009200e-02, 3.90275758377e-08, -1.68631039560e-03, 8.40498323669e-05 }, 
	 { 7.27469159756e-05, 2.63869314949e-05, 3.29690799117e-02, -2.19274429725e-08, 1.77378822118e-02, 3.26595727529e-05 }
      }, 
      {  { 1.46874145139e-03, 6.36232061879e-06, 1.28665426746e-02, 1.19409449439e-07, 1.15883778781e-02, 1.32179644424e-04 }, 
	 { 1.15970033221e-03, 1.30452335725e-05, 1.87015570700e-02, 5.39766737973e-08, 1.64790824056e-02, 1.44115634612e-04 }, 
	 { 6.27940462437e-04, 1.78520094778e-05, 2.83537860960e-02, 1.16867742150e-08, 5.02607785165e-02, 1.88510020962e-04 }
      } 
    }; 
  
  float const kParamRMS0[2][3][4] =
    {
      {  { 4.17516864836e-02, 1.87623649254e-04, 5.63788712025e-02, 5.38373768330e-01,  }, 
	 { 8.29434990883e-02, 2.03291710932e-04, 6.81538805366e-02, 9.70965325832e-01,  }, 
	 { 8.67543518543e-02, 2.10733342101e-04, 1.38366967440e-01, 2.55089461803e-01,  }
      }, 
      {  { 5.96254616976e-02, 8.62886518007e-05, 3.61776389182e-02, 4.79704320431e-01,  }, 
	 { 6.12571723759e-02, 7.23929333617e-05, 3.93057651818e-02, 9.29222583771e-01,  }, 
	 { 6.58465921879e-02, 1.03639606095e-04, 6.07583411038e-02, 9.90289509296e-01,  }
      }
    }; 

  for( int i=0; i<2; i++){
    for( int j=0; j<3; j++){  
      for( int k=0; k<6; k++){
	fParamS0Par[i][j][k] = kParamS0Par[i][j][k];
      }
    }
  }
 
  for( int i=0; i<2; i++){
    for( int j=0; j<3; j++){  
      for( int k=0; k<4; k++){
	fParamRMS0[i][j][k] = kParamRMS0[i][j][k];
      }
    }
  }

  for( int i=0; i<HLTCA_ROW_COUNT; i++ ) fRowX[i] = 0;

  Update();
}

void AliHLTTPCCAParam::Initialize( int iSlice,
    int nRows, float rowX[],
    float alpha, float dAlpha,
    float rMin, float rMax,
    float zMin, float zMax,
    float padPitch, float zSigma,
    float bz
                                        )
{
  // initialization
  fISlice = iSlice;
  fAlpha = alpha;
  fDAlpha = dAlpha;
  fRMin = rMin;
  fRMax = rMax;
  fZMin = zMin;
  fZMax = zMax;
  fPadPitch = padPitch;
  fErrY = 1.; // not in use
  fErrZ = zSigma;
  fBzkG = bz;
  fNRows = nRows;
  for ( int irow = 0; irow < nRows; irow++ ) {
    fRowX[irow] = rowX[irow];
    //std::cout << " row " << irow << " x= " << rowX[irow] << std::endl;
  }

  Update();
}

void AliHLTTPCCAParam::Update()
{
  // update of calculated values

  const double kCLight = 0.000299792458;
  fConstBz = fBzkG * kCLight;

  fCosAlpha = CAMath::Cos( fAlpha );
  fSinAlpha = CAMath::Sin( fAlpha );
  fAngleMin = fAlpha - fDAlpha / 2.f;
  fAngleMax = fAlpha + fDAlpha / 2.f;
  fErrX = fPadPitch / CAMath::Sqrt( 12. );
  fTrackChi2Cut = fTrackChiCut * fTrackChiCut;
}


void AliHLTTPCCAParam::LoadClusterErrors( bool Print )
{
  // update of calculated values
#if !defined(HLTCA_STANDALONE)

  const AliTPCClusterParam *clparam =  AliTPCcalibDB::Instance()->GetClusterParam();
 if( !clparam ){
    cout<<"Error: AliHLTTPCCAParam::LoadClusterErrors():: No AliTPCClusterParam instance found !!!! "<<endl;
    return;
  }

 for( int i=0; i<2; i++ ){
   for( int j=0; j<3; j++){
     for( int k=0; k<6; k++){
       fParamS0Par[i][j][k] = clparam->GetParamS0Par(i,j,k);
     }
   }
 }
   
 for( int i=0; i<2; i++ ){
   for( int j=0; j<3; j++){
     for( int k=0; k<4; k++){
       fParamRMS0[i][j][k] = clparam->GetParamRMS0(i,j,k);
     }
   }
 }
  
 if( Print ){
   typedef std::numeric_limits< float > flt;
   cout<<std::scientific;
#if __cplusplus >= 201103L  
   cout<<std::setprecision( flt::max_digits10+2 );
#endif
   cout<<"fParamS0Par[2][3][7]="<<endl;
   cout<<" { "<<endl;
   for( int i=0; i<2; i++ ){
     cout<<"   { "<<endl;   
     for( int j=0; j<3; j++){
       cout<<" { ";   
       for( int k=0; k<6; k++){
	 cout<<fParamS0Par[i][j][k]<<", "; 
       }
       cout<<" }, "<<endl;   
     }
     cout<<"   }, "<<endl;
   }
   cout<<" }; "<<endl;

  cout<<"fParamRMS0[2][3][4]="<<endl;
  cout<<" { "<<endl;
  for( int i=0; i<2; i++ ){
    cout<<"   { "<<endl;   
    for( int j=0; j<3; j++){
      cout<<" { ";   
      for( int k=0; k<4; k++){
	cout<<fParamRMS0[i][j][k]<<", "; 
      }
      cout<<" }, "<<endl;   
    }
    cout<<"   }, "<<endl;
  }
  cout<<" }; "<<endl;
  

  const THnBase *waveMap = clparam->GetWaveCorrectionMap();
  const THnBase *resYMap = clparam->GetResolutionYMap();
  cout<<"waveMap = "<<(void*)waveMap<<endl;
  cout<<"resYMap = "<<(void*)resYMap<<endl;

 }

#endif
}

#endif


MEM_CLASS_PRE() GPUdi() void MEM_LG(AliHLTTPCCAParam)::Slice2Global( float x, float y,  float z,
    float *X, float *Y,  float *Z ) const
{
  // conversion of coorinates sector->global
  *X = x * fCosAlpha - y * fSinAlpha;
  *Y = y * fCosAlpha + x * fSinAlpha;
  *Z = z;
}

MEM_CLASS_PRE() GPUdi() void MEM_LG(AliHLTTPCCAParam)::Global2Slice( float X, float Y,  float Z,
    float *x, float *y,  float *z ) const
{
  // conversion of coorinates global->sector
  *x = X * fCosAlpha + Y * fSinAlpha;
  *y = Y * fCosAlpha - X * fSinAlpha;
  *z = Z;
}

MEM_CLASS_PRE() GPUdi() float MEM_LG(AliHLTTPCCAParam)::GetClusterRMS( int yz, int type, float z, float angle2 ) const
{
  //* recalculate the cluster error wih respect to the track slope

  MakeType(const float*) c = fParamRMS0[yz][type];
  float v = c[0] + c[1]*z + c[2]*angle2;
  v = fabs(v);
  return v;
}

MEM_CLASS_PRE() GPUdi() void MEM_LG(AliHLTTPCCAParam)::GetClusterRMS2( int iRow, float z, float sinPhi, float DzDs, float &ErrY2, float &ErrZ2 ) const
{
  int    rowType = ( iRow < 63 ) ? 0 : ( ( iRow > 126 ) ? 1 : 2 );
  z = CAMath::Abs( ( 250. - 0.275 ) - CAMath::Abs( z ) );  
  float s2 = sinPhi*sinPhi;
  if( s2>0.95f*0.95f ) s2 = 0.95f*0.95f;
  float sec2 = 1.f/(1.f-s2);
  float angleY2 = s2 * sec2; // dy/dx
  float angleZ2 = DzDs * DzDs * sec2; // dz/dx
  
  ErrY2 = GetClusterRMS( 0, rowType, z, angleY2 );
  ErrZ2 = GetClusterRMS( 1, rowType, z, angleZ2 );
  ErrY2 *= ErrY2;
  ErrZ2 *= ErrZ2;
}

MEM_CLASS_PRE() GPUdi() float MEM_LG(AliHLTTPCCAParam)::GetClusterError2( int yz, int type, float z, float angle2 ) const
{
  //* recalculate the cluster error wih respect to the track slope

  MakeType(const float*) c = fParamS0Par[yz][type];
  float v = c[0] + c[1]*z + c[2]*angle2 + c[3]*z*z
    +c[4]*angle2*angle2 + c[5]*z*angle2;
  v = fabs(v);
  if (v<0.01) v = 0.01;
  v *= yz ? fClusterError2CorrectionZ : fClusterError2CorrectionY;
  return v;
}

MEM_CLASS_PRE() GPUdi() void MEM_LG(AliHLTTPCCAParam)::GetClusterErrors2( int iRow, float z, float sinPhi, float DzDs, float &ErrY2, float &ErrZ2 ) const
{
  // Calibrated cluster error from OCDB for Y and Z
  int    rowType = ( iRow < 63 ) ? 0 : ( ( iRow > 126 ) ? 1 : 2 );
  z = CAMath::Abs( ( 250. - 0.275 ) - CAMath::Abs( z ) );  
  float s2 = sinPhi*sinPhi;
  if( s2>0.95f*0.95f ) s2 = 0.95f*0.95f;
  float sec2 = 1.f/(1.f-s2);
  float angleY2 = s2 * sec2; // dy/dx
  float angleZ2 = DzDs * DzDs * sec2; // dz/dx
  
  ErrY2 = GetClusterError2( 0, rowType, z, angleY2 );
  ErrZ2 = GetClusterError2( 1, rowType, z, angleZ2 );
}

#ifndef HLTCA_GPUCODE
GPUh() void AliHLTTPCCAParam::WriteSettings( std::ostream &out ) const
{
  // write settings to the file
  out << fISlice << std::endl;
  out << fNRows << std::endl;
  out << fAlpha << std::endl;
  out << fDAlpha << std::endl;
  out << fCosAlpha << std::endl;
  out << fSinAlpha << std::endl;
  out << fAngleMin << std::endl;
  out << fAngleMax << std::endl;
  out << fRMin << std::endl;
  out << fRMax << std::endl;
  out << fZMin << std::endl;
  out << fZMax << std::endl;
  out << fErrX << std::endl;
  out << fErrY << std::endl;
  out << fErrZ << std::endl;
  out << fPadPitch << std::endl;
  out << fBzkG << std::endl;
  out << fHitPickUpFactor << std::endl;
  out << fMaxTrackMatchDRow << std::endl;
  out << fTrackConnectionFactor << std::endl;
  out << fTrackChiCut << std::endl;
  out << fTrackChi2Cut << std::endl;
  for ( int iRow = 0; iRow < fNRows; iRow++ ) {
    out << fRowX[iRow] << std::endl;
  }
  out << std::endl;
  for ( int i = 0; i < 2; i++ )
    for ( int j = 0; j < 3; j++ )
      for ( int k = 0; k < 4; k++ )
        out << fParamRMS0[i][j][k] << std::endl;
  out << std::endl;
}

GPUh() void AliHLTTPCCAParam::ReadSettings( std::istream &in )
{
  // Read settings from the file

  in >> fISlice;
  in >> fNRows;
  in >> fAlpha;
  in >> fDAlpha;
  in >> fCosAlpha;
  in >> fSinAlpha;
  in >> fAngleMin;
  in >> fAngleMax;
  in >> fRMin;
  in >> fRMax;
  in >> fZMin;
  in >> fZMax;
  in >> fErrX;
  in >> fErrY;
  in >> fErrZ;
  in >> fPadPitch;
  in >> fBzkG;
  in >> fHitPickUpFactor;
  in >> fMaxTrackMatchDRow;
  in >> fTrackConnectionFactor;
  in >> fTrackChiCut;
  in >> fTrackChi2Cut;

  if( fNRows<0 || fNRows > HLTCA_ROW_COUNT ) fNRows = 0;

  for ( int iRow = 0; iRow < fNRows; iRow++ ) {
    in >> fRowX[iRow];
  }
  for ( int i = 0; i < 2; i++ )
    for ( int j = 0; j < 3; j++ )
      for ( int k = 0; k < 4; k++ )
        in >> fParamRMS0[i][j][k];
}
#endif
