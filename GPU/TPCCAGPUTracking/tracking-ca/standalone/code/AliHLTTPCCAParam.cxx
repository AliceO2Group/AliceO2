// @(#) $Id: AliHLTTPCCAParam.cxx 45665 2010-11-24 15:54:05Z sgorbuno $
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


#if !defined(HLTCA_GPUCODE)

GPUdi() AliHLTTPCCAParam::AliHLTTPCCAParam()
    : fISlice( 0 ), fNRows( 63 ), fAlpha( 0.174533 ), fDAlpha( 0.349066 ),
    fCosAlpha( 0 ), fSinAlpha( 0 ), fAngleMin( 0 ), fAngleMax( 0 ), fRMin( 83.65 ), fRMax( 133.3 ),
    fZMin( 0.0529937 ), fZMax( 249.778 ), fErrX( 0 ), fErrY( 0 ), fErrZ( 0.228808 ), fPadPitch( 0.4 ), fBzkG( -5.00668 ),
    fConstBz( -5.00668*0.000299792458 ), fHitPickUpFactor( 1. ),
      fMaxTrackMatchDRow( 4 ), fNeighboursSearchArea(3.), fTrackConnectionFactor( 3.5 ), fTrackChiCut( 3.5 ), fTrackChi2Cut( 10 ), fClusterError2CorrectionY(1.), fClusterError2CorrectionZ(1.),
      fMinNTrackClusters( 30 ),
      fMaxTrackQPt(1./0.1)
{
  // constructor
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

#endif


GPUdi() void AliHLTTPCCAParam::Slice2Global( float x, float y,  float z,
    float *X, float *Y,  float *Z ) const
{
  // conversion of coorinates sector->global
  *X = x * fCosAlpha - y * fSinAlpha;
  *Y = y * fCosAlpha + x * fSinAlpha;
  *Z = z;
}

GPUdi() void AliHLTTPCCAParam::Global2Slice( float X, float Y,  float Z,
    float *x, float *y,  float *z ) const
{
  // conversion of coorinates global->sector
  *x = X * fCosAlpha + Y * fSinAlpha;
  *y = Y * fCosAlpha - X * fSinAlpha;
  *z = Z;
}

GPUdi() float AliHLTTPCCAParam::GetClusterError2( int yz, int type, float z, float angle ) const
{
  //* recalculate the cluster error wih respect to the track slope

  float angle2 = angle * angle;
  const float *c = fParamS0Par[yz][type];
  float v = c[0] + z * ( c[1] + c[3] * z ) + angle2 * ( c[2] + angle2 * c[4] + c[5] * z );
  return CAMath::Abs( v );
}

GPUdi() void AliHLTTPCCAParam::GetClusterErrors2( int iRow, float z, float sinPhi, float cosPhi, float DzDs, float &Err2Y, float &Err2Z ) const
{
  //
  // Use calibrated cluster error from OCDB
  //

  z = CAMath::Abs( ( 250. - 0.275 ) - CAMath::Abs( z ) );
  int    type = ( iRow < 63 ) ? 0 : ( ( iRow > 126 ) ? 1 : 2 );
  float cosPhiInv = CAMath::Abs( cosPhi ) > 1.e-2 ? 1. / cosPhi : 0;
  float angleY = sinPhi * cosPhiInv ; // dy/dx
  float angleZ = DzDs * cosPhiInv ; // dz/dx

  Err2Y = GetClusterError2( 0, type, z, angleY );
  Err2Z = GetClusterError2( 1, type, z, angleZ );
}

GPUdi() void AliHLTTPCCAParam::GetClusterErrors2v1( int rowType, float z, float sinPhi, float cosPhi, float DzDs, float &Err2Y, float &Err2Z ) const
{
  //
  // Use calibrated cluster error from OCDB
  //

  z = CAMath::Abs( ( 250. - 0.275 ) - CAMath::Abs( z ) );
  float cosPhiInv = CAMath::Abs( cosPhi ) > 1.e-2 ? 1. / cosPhi : 0;
  float angleY = sinPhi * cosPhiInv ; // dy/dx
  float angleZ = DzDs * cosPhiInv ; // dz/dx

  Err2Y = GetClusterError2( 0, rowType, z, angleY );
  Err2Z = GetClusterError2( 1, rowType, z, angleZ );
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
  for ( int iRow = 0; iRow < fNRows; iRow++ ) {
    in >> fRowX[iRow];
  }
  for ( int i = 0; i < 2; i++ )
    for ( int j = 0; j < 3; j++ )
      for ( int k = 0; k < 7; k++ )
        in >> fParamS0Par[i][j][k];
}
#endif
