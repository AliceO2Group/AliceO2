// @(#) $Id$
//***************************************************************************
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
//***************************************************************************

#include "AliHLTTPCCAParam.h"
#include "AliHLTTPCCAMath.h"


#if !defined(HLTCA_GPUCODE)  

GPUd() AliHLTTPCCAParam::AliHLTTPCCAParam()
  : fISlice(0),fNRows(63),fAlpha(0.174533), fDAlpha(0.349066),
    fCosAlpha(0), fSinAlpha(0), fAngleMin(0), fAngleMax(0), fRMin(83.65), fRMax(133.3),
    fZMin(0.0529937), fZMax(249.778), fErrX(0), fErrY(0), fErrZ(0.228808),fPadPitch(0.4),fBz(-5.), 
    fHitPickUpFactor(2.),
    fMaxTrackMatchDRow(4), fTrackConnectionFactor(3.5), fTrackChiCut(3.5), fTrackChi2Cut(10)
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
#endif

GPUd() void AliHLTTPCCAParam::Initialize( Int_t iSlice, 
				   Int_t nRows, Float_t rowX[],
				   Float_t alpha, Float_t dAlpha,
				   Float_t rMin, Float_t rMax,
				   Float_t zMin, Float_t zMax,
				   Float_t padPitch, Float_t zSigma,
				   Float_t bz
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
  fBz = bz;
  fNRows = nRows;
  for( Int_t irow=0; irow<nRows; irow++ ){
    fRowX[irow] = rowX[irow];
  }

  Update();
}

GPUd() void AliHLTTPCCAParam::Update()
{
  // update of calculated values
  fCosAlpha = CAMath::Cos(fAlpha);
  fSinAlpha = CAMath::Sin(fAlpha);
  fAngleMin = fAlpha - fDAlpha/2.;
  fAngleMax = fAlpha + fDAlpha/2.;
  fErrX = fPadPitch/CAMath::Sqrt(12.);
  fTrackChi2Cut = fTrackChiCut * fTrackChiCut;
}

GPUd() void AliHLTTPCCAParam::Slice2Global( Float_t x, Float_t y,  Float_t z, 
				     Float_t *X, Float_t *Y,  Float_t *Z ) const
{  
  // conversion of coorinates sector->global
  *X = x*fCosAlpha - y*fSinAlpha;
  *Y = y*fCosAlpha + x*fSinAlpha;
  *Z = z;
}
 
GPUd() void AliHLTTPCCAParam::Global2Slice( Float_t X, Float_t Y,  Float_t Z, 
				     Float_t *x, Float_t *y,  Float_t *z ) const
{
  // conversion of coorinates global->sector
  *x = X*fCosAlpha + Y*fSinAlpha;
  *y = Y*fCosAlpha - X*fSinAlpha;
  *z = Z;
}

GPUd() Float_t AliHLTTPCCAParam::GetClusterError2( Int_t yz, Int_t type, Float_t z, Float_t angle ) const
{
  //* recalculate the cluster error wih respect to the track slope
  Float_t angle2 = angle*angle;
  const Float_t *c = fParamS0Par[yz][type];
  Float_t v = c[0] + z*(c[1] + c[3]*z) + angle2*(c[2] + angle2*c[4] + c[5]*z );
  return CAMath::Abs(v); 
}

GPUh() void AliHLTTPCCAParam::WriteSettings( std::ostream &out ) const 
{
  // write settings to the file
  out << fISlice<<std::endl;
  out << fNRows<<std::endl;
  out << fAlpha<<std::endl;
  out << fDAlpha<<std::endl;
  out << fCosAlpha<<std::endl;
  out << fSinAlpha<<std::endl;
  out << fAngleMin<<std::endl;
  out << fAngleMax<<std::endl;
  out << fRMin<<std::endl;
  out << fRMax<<std::endl;
  out << fZMin<<std::endl;
  out << fZMax<<std::endl;
  out << fErrX<<std::endl;
  out << fErrY<<std::endl;
  out << fErrZ<<std::endl;
  out << fPadPitch<<std::endl;
  out << fBz<<std::endl;
  out << fHitPickUpFactor<<std::endl;
  out << fMaxTrackMatchDRow<<std::endl;
  out << fTrackConnectionFactor<<std::endl;
  out << fTrackChiCut<<std::endl;
  out << fTrackChi2Cut<<std::endl;
  for( Int_t iRow = 0; iRow<fNRows; iRow++ ){
    out << fRowX[iRow]<<std::endl;
  }
  out<<std::endl;
  for( Int_t i=0; i<2; i++ )
    for( Int_t j=0; j<3; j++ )
      for( Int_t k=0; k<7; k++ )
	out << fParamS0Par[i][j][k]<<std::endl;
  out<<std::endl;
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
  in >> fBz;
  in >> fHitPickUpFactor;
  in >> fMaxTrackMatchDRow;
  in >> fTrackConnectionFactor;
  in >> fTrackChiCut;
  in >> fTrackChi2Cut;
  for( Int_t iRow = 0; iRow<fNRows; iRow++ ){
    in >> fRowX[iRow];
  }
  for( Int_t i=0; i<2; i++ )
    for( Int_t j=0; j<3; j++ )
      for( Int_t k=0; k<7; k++ )
	in >> fParamS0Par[i][j][k];
}
