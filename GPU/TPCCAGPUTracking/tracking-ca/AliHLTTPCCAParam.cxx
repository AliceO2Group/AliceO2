// @(#) $Id$
//*************************************************************************
// This file is property of and copyright by the ALICE HLT Project        * 
// ALICE Experiment at CERN, All rights reserved.                         *
//                                                                        *
// Primary Authors: Jochen Thaeder <thaeder@kip.uni-heidelberg.de>        *
//                  Ivan Kisel <kisel@kip.uni-heidelberg.de>              *
//                  for The ALICE HLT Project.                            *
//                                                                        *
// Permission to use, copy, modify and distribute this software and its   *
// documentation strictly for non-commercial purposes is hereby granted   *
// without fee, provided that the above copyright notice appears in all   *
// copies and that both the copyright notice and this permission notice   *
// appear in the supporting documentation. The authors make no claims     *
// about the suitability of this software for any purpose. It is          *
// provided "as is" without express or implied warranty.                  *
//*************************************************************************

#include "AliHLTTPCCAParam.h"
#include "TMath.h"


ClassImp(AliHLTTPCCAParam)

AliHLTTPCCAParam::AliHLTTPCCAParam()
  : fISec(0),fNRows(63),fAlpha(0.174533), fDAlpha(0.349066),
    fCosAlpha(0), fSinAlpha(0), fAngleMin(0), fAngleMax(0), fRMin(83.65), fRMax(133.3),
    fZMin(0.0529937), fZMax(249.778), fErrZ(0.228808), fErrX(0), fErrY(0),fPadPitch(0.4),fBz(-5.), 
    fCellConnectionFactor(3), fTrackConnectionFactor(5), fTrackChiCut(6), fTrackChi2Cut(0), fMaxTrackMatchDRow(4),
    fYErrorCorrection(0.33), fZErrorCorrection(0.45)
{
  Update();
}

void AliHLTTPCCAParam::Initialize( Int_t ParISec, 
				   Int_t ParNRows, Double_t ParRowX[],
				   Double_t ParAlpha, Double_t ParDAlpha,
				   Double_t ParRMin, Double_t ParRMax,
				   Double_t ParZMin, Double_t ParZMax,
				   Double_t ParPadPitch, Double_t ParZSigma,
				   Double_t ParBz
				   )
{
  // initialization 
  fISec = ParISec;
  fAlpha = ParAlpha;
  fDAlpha = ParDAlpha;
  fRMin = ParRMin;
  fRMax = ParRMax;
  fZMin = ParZMin;
  fZMax = ParZMax;
  fPadPitch = ParPadPitch;
  fErrY = 1.; // not in use
  fErrZ = ParZSigma;
  fBz = ParBz;
  fNRows = ParNRows;
  for( Int_t irow=0; irow<ParNRows; irow++ ){
    fRowX[irow] = ParRowX[irow];
  }

  Update();
}

void AliHLTTPCCAParam::Update()
{
  // update of calculated values
  fCosAlpha = TMath::Cos(fAlpha);
  fSinAlpha = TMath::Sin(fAlpha);
  fAngleMin = fAlpha - fDAlpha/2.;
  fAngleMax = fAlpha + fDAlpha/2.;
  fErrX = fPadPitch/TMath::Sqrt(12.);
  fTrackChi2Cut = fTrackChiCut * fTrackChiCut;
}

void AliHLTTPCCAParam::Sec2Global(   Double_t x, Double_t y,  Double_t z, 
				     Double_t *X, Double_t *Y,  Double_t *Z ) const
{  
  // conversion of coorinates sector->global
  *X = x*fCosAlpha - y*fSinAlpha;
  *Y = y*fCosAlpha + x*fSinAlpha;
  *Z = z;
}
 
void AliHLTTPCCAParam::Global2Sec( Double_t X, Double_t Y,  Double_t Z, 
				   Double_t *x, Double_t *y,  Double_t *z ) const
{
  // conversion of coorinates global->sector
  *x = X*fCosAlpha + Y*fSinAlpha;
  *y = Y*fCosAlpha - X*fSinAlpha;
  *z = Z;
}
