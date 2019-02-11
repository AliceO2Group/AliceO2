#ifndef ALIESDTOFMATCH_H
#define ALIESDTOFMATCH_H

/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

/* $Id: $ */

//----------------------------------------------------------------------//
//                                                                      //
// AliESDTOFMatch Class                                                 //
//                                                                      //
//----------------------------------------------------------------------//

#include "AliVTOFMatch.h"

class AliESDTOFMatch : public AliVTOFMatch
{
 public:
  AliESDTOFMatch();
  AliESDTOFMatch(Int_t i,Double_t inttimes[AliPID::kSPECIESC],Double_t dx,Double_t dy,Double_t dz,Double_t l);
  AliESDTOFMatch(AliESDTOFMatch &source);
  virtual ~AliESDTOFMatch() {}
  AliESDTOFMatch &operator=(const AliESDTOFMatch& source);
  virtual Float_t GetDx() const {return fDx;}
  virtual Float_t GetDy() const {return fDy;}
  virtual Float_t GetDz() const {return fDz;}
  virtual Float_t GetTrackLength() const {return fTrackLength;}
  virtual void SetDx(Double_t delta) {fDx = delta;}
  virtual void SetDy(Double_t delta) {fDy = delta;}
  virtual void SetDz(Double_t delta) {fDz = delta;}
  virtual void SetTrackLength(Double_t length) {fTrackLength = length;}
  //
  virtual Double_t GetIntegratedTimes(Int_t i) const {return fIntegratedTimes[i];}
  virtual void     GetIntegratedTimes(Double_t* t) const {for (int i=AliPID::kSPECIESC;i--;) t[i]=fIntegratedTimes[i];}
  virtual void     SetIntegratedTimes(Double_t* t)       {for (int i=AliPID::kSPECIESC;i--;) fIntegratedTimes[i]=t[i];}
  //
  virtual Int_t GetTrackIndex()           const {return GetUniqueID();}
  virtual void  SetTrackIndex(Int_t id)         {SetUniqueID(id);}
  void Print(const Option_t *opt=0) const;
  
 protected:
  Double32_t fDx;                // DeltaX residual
  Double32_t fDy;                //! DeltaY residual
  Double32_t fDz;                // DeltaZ residual
  Double32_t fTrackLength;       // track Length
  Double32_t fIntegratedTimes[AliPID::kSPECIESC]; // int timex
  //
  ClassDef(AliESDTOFMatch, 1) // TOF matchable hit
    //
};
#endif
