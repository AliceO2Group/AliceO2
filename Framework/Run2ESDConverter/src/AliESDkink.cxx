/**************************************************************************
 * Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 *                                                                        *
 * Author: The ALICE Off-line Project.                                    *
 * Contributors are mentioned in the code where appropriate.              *
 *                                                                        *
 * Permission to use, copy, modify and distribute this software and its   *
 * documentation strictly for non-commercial purposes is hereby granted   *
 * without fee, provided that the above copyright notice appears in all   *
 * copies and that both the copyright notice and this permission notice   *
 * appear in the supporting documentation. The authors make no claims     *
 * about the suitability of this software for any purpose. It is          *
 * provided "as is" without express or implied warranty.                  *
 **************************************************************************/

/* $Id$ */

//-------------------------------------------------------------------------
//    Origin: Marian Ivanov marian.ivanov@cern.ch
//-------------------------------------------------------------------------

#include <Riostream.h>
#include <TMath.h>
#include <TPDGCode.h>
#include "AliESDkink.h"


ClassImp(AliESDkink)

//____________________________________________________________________
AliESDkink::AliESDkink() :
  TObject(),
  fParamDaughter(),
  fParamMother(),
  fDist1(-1),
  fDist2(-1),
  fRr(0),
  fShapeFactor(0),
  fID(0),
  fRow0(0)
{
  //
  //Default constructor
  //
  for (Int_t i=0;i<12;i++) fStatus[i]=0;
  for (Int_t i=0;i<2;i++)
    for (Int_t j=0;j<2;j++){
      fTPCdensity[i][j]=0;
    }
  fTPCncls[0]=fTPCncls[1]=0;

  for (Int_t i=0; i<3; i++) {
    fPdr[i] = 0;
    fXr[i] = 0;
    fPm[i] = 0;
    fAngle[i] = 0;
  }
  fLab[0]=fLab[1]=0;
  fIndex[0]=fIndex[1]=-1;
  fMultiple[0]=fMultiple[1]=0;
}

AliESDkink::AliESDkink(const AliESDkink &source):
  TObject(source),
  fParamDaughter(source.fParamDaughter),
  fParamMother(source.fParamMother),
  fDist1(source.fDist1),
  fDist2(source.fDist1),
  fRr(source.fRr),
  fShapeFactor(source.fShapeFactor),
  fID(source.fID),
  fRow0(source.fRow0)
{
  //
  //Copy constructor
  //
  for (Int_t i=0;i<12;i++) fStatus[i]=source.fStatus[i];
  for (Int_t i=0;i<2;i++){
    fTPCncls[i] = source.fTPCncls[i];
    fLab[i]     = source.fLab[i];
    fIndex[i]   = source.fIndex[i];
    fMultiple[i]= source.fMultiple[i];
    for (Int_t j=0;j<2;j++){
      fTPCdensity[i][j] = source.fTPCdensity[i][j];
    }
  }
  for (Int_t i=0; i<3; i++) {
    fPdr[i] = source.fPdr[i];
    fXr[i] = source.fXr[i];
    fPm[i] = source.fPm[i];
    fAngle[i] = source.fAngle[i];
  }
}

//_____________________________________________________________________________
AliESDkink& AliESDkink::operator=(const AliESDkink &source)
{
  //
  // assignment operator
  //
  
  if (this!=&source) {
    TObject::operator=(source);
    fParamDaughter = source.fParamDaughter;
    fParamMother = source.fParamMother;
    fDist1 = source.fDist1;
    fDist2 = source.fDist1;
    fRr = source.fRr;
    fShapeFactor = source.fShapeFactor;
    fID = source.fID;
    fRow0 = source.fRow0;
    for (Int_t i=0;i<12;i++) fStatus[i]=source.fStatus[i];
    for (Int_t i=0;i<2;i++){
      fTPCncls[i] = source.fTPCncls[i];
      fLab[i]     = source.fLab[i];
      fIndex[i]   = source.fIndex[i];
      fMultiple[i]= source.fMultiple[i];
      for (Int_t j=0;j<2;j++){
	fTPCdensity[i][j] = source.fTPCdensity[i][j];
      }
    }
    for (Int_t i=0; i<3; i++) {
      fPdr[i] = source.fPdr[i];
      fXr[i] = source.fXr[i];
      fPm[i] = source.fPm[i];
      fAngle[i] = source.fAngle[i];
    }
  }

  return *this;
}

void AliESDkink::Copy(TObject &obj) const {
  
  // this overwrites the virtual TOBject::Copy()
  // to allow run time copying without casting
  // in AliESDEvent

  if(this==&obj)return;
  AliESDkink *robj = dynamic_cast<AliESDkink*>(&obj);
  if(!robj)return; // not an AliESDkink
  *robj = *this;

}


void AliESDkink::SetMother(const AliExternalTrackParam & pmother)  {
  //
  // set mother
  //
  fParamMother   = pmother;
}

void AliESDkink::SetDaughter(const AliExternalTrackParam & pdaughter){
  //
  //set daughter
  //
  fParamDaughter = pdaughter;

}
  
Double_t AliESDkink::GetTPCDensityFactor() const
{
  //
  //
  return fTPCdensity[0][0]+fTPCdensity[1][1]-TMath::Max(fTPCdensity[0][1],Double_t(0.0))-TMath::Max(fTPCdensity[1][0],Double_t(0.0)); 
}

Float_t AliESDkink::GetQt() const
{
  Float_t dmomentum = TMath::Sqrt(fPdr[0]*fPdr[0]+fPdr[1]*fPdr[1]+fPdr[2]*fPdr[2]);
  return TMath::Sin(fAngle[2])*dmomentum;
}
