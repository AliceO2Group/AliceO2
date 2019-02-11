#ifndef ALIESDKINK_H
#define ALIESDKINK_H

/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

/* $Id$ */

//-------------------------------------------------------------------------
//                          ESD V0 Vertex Class
//          This class is part of the Event Summary Data set of classes
//    Origin: Marian Ivanov marian.ivanov@cern.ch
//-------------------------------------------------------------------------

#include <TObject.h>
#include "AliExternalTrackParam.h"
#include <TPDGCode.h>

class AliESDtrack;

class AliESDkink : public TObject {
public:
  AliESDkink();             //constructor
  AliESDkink(const AliESDkink &source);             //constructor
  AliESDkink& operator=(const AliESDkink &source);
  virtual void Copy(TObject &obj) const;
  //
  void SetID(Short_t id){fID=id;}
  Short_t GetID(){return fID;}
  void SetMother(const AliExternalTrackParam & pmother); 
  void SetDaughter(const AliExternalTrackParam & pdaughter);
  Double_t GetTPCDensityFactor() const;
  Float_t GetQt() const;    
  //
  Double_t GetR() const {return fRr;}
  Double_t GetDistance() const {return fDist2;}
  UChar_t   GetTPCRow0() const {return fRow0;}
  Double_t GetAngle(Int_t i) const {return fAngle[i];}
  const Double_t *GetPosition() const   {return fXr;}
  const Double_t *GetMotherP()  const   {return fPm;}
  const Double_t *GetDaughterP()  const {return fPdr;}
  void SetTPCRow0(Int_t row0){fRow0 = row0;}
  Int_t GetLabel(Int_t i) const {return fLab[i];}
  void SetLabel(Int_t label, Int_t pos) {fLab[pos]=label;}
  Int_t GetIndex(Int_t i) const {return fIndex[i];}
  void SetIndex(Int_t index, Int_t pos){fIndex[pos]=index;}
  void SetStatus(Char_t status, Int_t pos){fStatus[pos]=status;}
  Char_t GetStatus(Int_t pos) const {return fStatus[pos];}
  void SetTPCncls(UChar_t ncls,Int_t pos) {fTPCncls[pos]=ncls;}
  const UChar_t *GetTPCncls() const {return fTPCncls;} 
  void  SetTPCDensity(Float_t dens, Int_t pos0,Int_t pos1){fTPCdensity[pos0][pos1]=dens;}
  Double_t GetTPCDensity(Int_t pos0,Int_t pos1) const {return fTPCdensity[pos0][pos1];}
  Double_t GetShapeFactor() const {return fShapeFactor;}
  void    SetShapeFactor(Float_t factor){fShapeFactor = factor;}
  void  SetMultiple(UChar_t mult,Int_t pos){fMultiple[pos]=mult;}
  const UChar_t * GetMultiple() const {return fMultiple;}
  //  
  const AliExternalTrackParam& RefParamDaughter() {return fParamDaughter;}
  const AliExternalTrackParam& RefParamMother()   {return fParamMother;}
 protected:

  AliExternalTrackParam fParamDaughter;
  AliExternalTrackParam fParamMother;

  Double32_t       fDist1;    //info about closest distance according closest MC - linear DCA
  Double32_t       fDist2;    //info about closest distance parabolic DCA
  //
  Double32_t       fPdr[3];    //momentum at vertex daughter  - according approx at DCA
  Double32_t       fXr[3];     //rec. position according helix
  //
  Double32_t       fPm[3];    //momentum at the vertex mother
  Double32_t       fRr;       // rec position of the vertex 

  Double32_t       fShapeFactor;       // tpc clusters shape factor
  Double32_t       fTPCdensity[2][2];  //[0,1,16]tpc cluster density before and after kink
  Double32_t       fAngle[3]; //[-2*pi,2*pi,16]three angles

  Int_t            fLab[2];   //MC label of the partecle
  Int_t            fIndex[2]; //reconstructed labels of the tracks

  Short_t          fID;       // kink ID

  UChar_t          fRow0;              // critical pad row number
  UChar_t          fMultiple[2];       //how many times the track's were used
  UChar_t          fTPCncls[2];     //number of clusters for mother particle

  Char_t           fStatus[12];       //status of kink - first 4 mother (ITS,TPC,TRD,TOF)  other daughter


  ClassDef(AliESDkink, 5)      // ESD V0 vertex
};

#endif


