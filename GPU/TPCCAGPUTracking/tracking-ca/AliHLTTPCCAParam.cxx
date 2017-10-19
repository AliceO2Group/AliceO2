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
      fMinNTrackClusters( -1 ), fMaxTrackQPt(1./0.015), fHighQPtForward(1.e10), fNWays(1), fSearchWindowDZDR(0.), fContinuousTracking(false)
{
  // constructor

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

  float const kParamS0Par[2][3][7]=
    { 
      {  { 6.45913474727e-04, 2.51547407970e-05, 1.57551113516e-02, 1.99872811635e-08, -5.86769729853e-03, 9.16301505640e-05, 1.01167142391e+00,  }, 
	 { 9.71546804067e-04, 1.70938055817e-05, 2.17084009200e-02, 3.90275758377e-08, -1.68631039560e-03, 8.40498323669e-05, 9.55379426479e-01,  }, 
	 { 7.27469159756e-05, 2.63869314949e-05, 3.29690799117e-02, -2.19274429725e-08, 1.77378822118e-02, 3.26595727529e-05, 1.17259633541e+00,  }
      }, 
      {  { 1.46874145139e-03, 6.36232061879e-06, 1.28665426746e-02, 1.19409449439e-07, 1.15883778781e-02, 1.32179644424e-04, 1.32442188263e+00,  }, 
	 { 1.15970033221e-03, 1.30452335725e-05, 1.87015570700e-02, 5.39766737973e-08, 1.64790824056e-02, 1.44115634612e-04, 1.24038755894e+00,  }, 
	 { 6.27940462437e-04, 1.78520094778e-05, 2.83537860960e-02, 1.16867742150e-08, 5.02607785165e-02, 1.88510020962e-04, 8.44087302685e-01,  }
      } 
    }; 

  for( int i=0; i<2; i++){
    for( int j=0; j<3; j++){  
      for( int k=0; k<4; k++){
	fParamRMS0[i][j][k] = kParamRMS0[i][j][k];
      }
    }
  }

  for( int i=0; i<2; i++){
    for( int j=0; j<3; j++){  
      for( int k=0; k<7; k++){
	fParamS0Par[i][j][k] = kParamS0Par[i][j][k];
      }
    }
  }

  // old values
  
  fParamS0Par[0][0][0] = 0.00047013;
  fParamS0Par[0][0][1] = 2.00135e-05;
  fParamS0Par[0][0][2] = 0.0106533;
  fParamS0Par[0][0][3] = 5.27104e-08;
  fParamS0Par[0][0][4] = 0.012829;
  fParamS0Par[0][0][5] = 0.000147125;
  fParamS0Par[0][0][6] = 4.99432;
  fParamS0Par[0][1][0] = 0.000883342;
  fParamS0Par[0][1][1] = 1.07011e-05;
  fParamS0Par[0][1][2] = 0.0103187;
  fParamS0Par[0][1][3] = 4.25141e-08;
  fParamS0Par[0][1][4] = 0.0224292;
  fParamS0Par[0][1][5] = 8.27274e-05;
  fParamS0Par[0][1][6] = 4.17233;
  fParamS0Par[0][2][0] = 0.000745399;
  fParamS0Par[0][2][1] = 5.62408e-06;
  fParamS0Par[0][2][2] = 0.0151562;
  fParamS0Par[0][2][3] = 5.08757e-08;
  fParamS0Par[0][2][4] = 0.0601004;
  fParamS0Par[0][2][5] = 7.97129e-05;
  fParamS0Par[0][2][6] = 4.84913;
  fParamS0Par[1][0][0] = 0.00215126;
  fParamS0Par[1][0][1] = 6.82233e-05;
  fParamS0Par[1][0][2] = 0.0221867;
  fParamS0Par[1][0][3] = -6.27825e-09;
  fParamS0Par[1][0][4] = -0.00745378;
  fParamS0Par[1][0][5] = 0.000172629;
  fParamS0Par[1][0][6] = 6.24987;
  fParamS0Par[1][1][0] = 0.00181667;
  fParamS0Par[1][1][1] = -4.17772e-06;
  fParamS0Par[1][1][2] = 0.0253429;
  fParamS0Par[1][1][3] = 1.3011e-07;
  fParamS0Par[1][1][4] = -0.00362827;
  fParamS0Par[1][1][5] = 0.00030406;
  fParamS0Par[1][1][6] = 17.7775;
  fParamS0Par[1][2][0] = 0.00158251;
  fParamS0Par[1][2][1] = -3.55911e-06;
  fParamS0Par[1][2][2] = 0.0247899;
  fParamS0Par[1][2][3] = 7.20604e-08;
  fParamS0Par[1][2][4] = 0.0179946;
  fParamS0Par[1][2][5] = 0.000425504;
  fParamS0Par[1][2][6] = 20.9294;
  
  for( int i=0; i<200; i++ ) fRowX[i] = 0;

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

  fPolinomialFieldBz[0] = fConstBz * (  0.999286   );
  fPolinomialFieldBz[1] = fConstBz * ( -4.54386e-7 );
  fPolinomialFieldBz[2] = fConstBz * (  2.32950e-5 );
  fPolinomialFieldBz[3] = fConstBz * ( -2.99912e-7 );
  fPolinomialFieldBz[4] = fConstBz * ( -2.03442e-8 );
  fPolinomialFieldBz[5] = fConstBz * (  9.71402e-8 );

  fCosAlpha = CAMath::Cos( fAlpha );
  fSinAlpha = CAMath::Sin( fAlpha );
  fAngleMin = fAlpha - fDAlpha / 2.f;
  fAngleMax = fAlpha + fDAlpha / 2.f;
  fErrX = fPadPitch / CAMath::Sqrt( 12. );
  fTrackChi2Cut = fTrackChiCut * fTrackChiCut;
}


void AliHLTTPCCAParam::LoadClusterErrors()
{
  // update of calculated values
#if !defined(HLTCA_STANDALONE)

  const AliTPCClusterParam *clparam =  AliTPCcalibDB::Instance()->GetClusterParam();
 if( !clparam ){
    cout<<"Error: AliHLTTPCCAParam::LoadClusterErrors():: No AliTPCClusterParam instance found !!!! "<<endl;
    return;
  }
  typedef std::numeric_limits< float > flt;
  cout<<std::scientific;
  cout<<std::setprecision( flt::max_digits10+2 );

  cout<<"fParamRMS0[2][3][4]="<<endl;
  cout<<" { "<<endl;
  for( int i=0; i<2; i++ ){
    cout<<"   { "<<endl;   
    for( int j=0; j<3; j++){
      cout<<" { ";   
      for( int k=0; k<4; k++){      
	cout<<clparam->GetParamRMS0(i,j,k)<<", "; 
      }
      cout<<" }, "<<endl;   
    }
    cout<<"   }, "<<endl;
  }
  cout<<" }; "<<endl;

  cout<<"fParamS0Par[2][3][7]="<<endl;
  cout<<" { "<<endl;
  for( int i=0; i<2; i++ ){
    cout<<"   { "<<endl;   
    for( int j=0; j<3; j++){
      cout<<" { ";   
      for( int k=0; k<7; k++){      
	cout<<clparam->GetParamS0Par(i,j,k)<<", "; 
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

MEM_CLASS_PRE() GPUdi() float MEM_LG(AliHLTTPCCAParam)::GetClusterError2( int yz, int type, float z, float angle2 ) const
{
  //* recalculate the cluster error wih respect to the track slope
  
  MakeType(const float*) c = fParamS0Par[yz][type];
  float v = c[0] + z * ( c[1] + c[3] * z ) + angle2 * ( c[2] + angle2 * c[4] + c[5] * z );
  return CAMath::Abs( v );  
}


MEM_CLASS_PRE() GPUdi() float MEM_LG(AliHLTTPCCAParam)::GetClusterError2New( int yz, int type, float z, float angle2 ) const
{
  //* recalculate the cluster error wih respect to the track slope

  // new parameterisation
  MakeType(const float*) c = fParamRMS0[yz][type];
  float v = c[0] + c[1]*z + c[2]*angle2;
  return CAMath::Abs( v );
}

MEM_CLASS_PRE() GPUdi() void MEM_LG(AliHLTTPCCAParam)::GetClusterErrors2New( int rowType, float z, float sinPhi, float cosPhi, float DzDs, float &Err2Y, float &Err2Z ) const
{
  //
  // Use calibrated cluster error from OCDB
  //

  z = CAMath::Abs( ( 250. - 0.275 ) - CAMath::Abs( z ) );
  float s2 = sinPhi*sinPhi;
  if( s2>0.95f*0.95f ) s2 = 0.95f*0.95f;
  float sec2 = 1.f/(1.f-s2);
  float angleY2 = s2 * sec2; // dy/dx
  float angleZ2 = DzDs * DzDs * sec2; // dz/dx
  Err2Y = GetClusterError2New( 0, rowType, z, angleY2 );
  Err2Z = GetClusterError2New( 1, rowType, z, angleZ2 );
}

MEM_CLASS_PRE() GPUdi() void MEM_LG(AliHLTTPCCAParam)::GetClusterErrors2( int iRow, float z, float sinPhi, float cosPhi, float DzDs, float &Err2Y, float &Err2Z ) const
{
  // Use calibrated cluster error from OCDB
  int    type = ( iRow < 63 ) ? 0 : ( ( iRow > 126 ) ? 1 : 2 );
  GetClusterErrors2v1(type, z, sinPhi, cosPhi, DzDs, Err2Y, Err2Z);
}

MEM_CLASS_PRE() GPUdi() void MEM_LG(AliHLTTPCCAParam)::GetClusterErrors2v1( int rowType, float z, float sinPhi, float cosPhi, float DzDs, float &Err2Y, float &Err2Z ) const
{
  //
  // Use calibrated cluster error from OCDB
  //

  z = CAMath::Abs( ( 250. - 0.275 ) - CAMath::Abs( z ) );  
  float s2 = sinPhi*sinPhi;
  if( s2>0.95f*0.95f ) s2 = 0.95f*0.95f;
  float sec2 = 1.f/(1.f-s2);
  float angleY2 = s2 * sec2; // dy/dx
  float angleZ2 = DzDs * DzDs * sec2; // dz/dx
  Err2Y = GetClusterError2( 0, rowType, z, angleY2 );
  Err2Z = GetClusterError2( 1, rowType, z, angleZ2 );
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
      for ( int k = 0; k < 7; k++ )
        out << fParamS0Par[i][j][k] << std::endl;
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

  if( fNRows<0 || fNRows > 200 ) fNRows = 0;

  for ( int iRow = 0; iRow < fNRows; iRow++ ) {
    in >> fRowX[iRow];
  }
  for ( int i = 0; i < 2; i++ )
    for ( int j = 0; j < 3; j++ )
      for ( int k = 0; k < 7; k++ )
        in >> fParamS0Par[i][j][k];
}
#endif
