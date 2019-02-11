#ifndef ALIITSPIDRESPONSE_H
#define ALIITSPIDRESPONSE_H
/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

/* $Id$ */

//-------------------------------------------------------
//                    ITS PID response class
//
//
//-------------------------------------------------------
//#include <Rtypes.h>
#include <TObject.h>
#include "AliPID.h"

class AliVTrack;
class AliITSPidParams;

class AliITSPIDResponse : public TObject {

public:
  AliITSPIDResponse(Bool_t isMC=kFALSE);
  //AliITSPIDResponse(Double_t *param);
 ~AliITSPIDResponse() {}

 void SetBetheBlochParamsITSTPC(Double_t* param){
   for(Int_t iPar=0; iPar<5; iPar++) fBBtpcits[iPar]=param[iPar];
 }
 void SetBetheBlochParamsITSTPCDeuteron(Double_t* param){
   for(Int_t iPar=0; iPar<5; iPar++) fBBdeu[iPar]=param[iPar];
 }
 void SetBetheBlochParamsITSTPCTriton(Double_t* param){
   for(Int_t iPar=0; iPar<5; iPar++) fBBtri[iPar]=param[iPar];
 }
 void SetBetheBlochParamsITSsa(Double_t* param){
   for(Int_t iPar=0; iPar<5; iPar++) fBBsa[iPar]=param[iPar];
 }
 void SetBetheBlochHybridParamsITSsa(Double_t* param){
   for(Int_t iPar=0; iPar<9; iPar++) fBBsaHybrid[iPar]=param[iPar];
 }
 void SetElectronBetheBlochParamsITSsa(Double_t* param){
   for(Int_t iPar=0; iPar<5; iPar++) fBBsaElectron[iPar]=param[iPar];
 }

 Double_t BetheAleph(Double_t p,Double_t mass) const;
 Double_t Bethe(Double_t p, Double_t mass, Bool_t isSA=kFALSE) const;
 Double_t Bethe(Double_t p, AliPID::EParticleType species, Bool_t isSA=kFALSE) const;
 Double_t BetheITSsaHybrid(Double_t p, Double_t mass) const;
 Double_t GetResolution(Double_t bethe, Int_t nPtsForPid=4, Bool_t isSA=kFALSE,Double_t p=0., AliPID::EParticleType type=AliPID::kPion) const;
  void GetITSProbabilities(Float_t mom, Double_t qclu[4], Double_t condprobfun[AliPID::kSPECIES], Bool_t isMC = kFALSE) const;
  void GetITSProbabilities(Float_t mom, Double_t qclu[4], Double_t condprobfun[AliPID::kSPECIES], AliITSPidParams *pars) const;
  
  Double_t GetNumberOfSigmas( const AliVTrack* track, AliPID::EParticleType species) const;
  
  Double_t GetSignalDelta( const AliVTrack* track, AliPID::EParticleType species, Bool_t ratio=kFALSE) const;
  
  Float_t GetNumberOfSigmas(Float_t mom, Float_t signal, AliPID::EParticleType type, Int_t nPtsForPid=4, Bool_t isSA=kFALSE) const {
    if(type==AliPID::kDeuteron && mom<0.4) return -999.;
    if(type==AliPID::kTriton && mom<0.55) return -999.;
    const Double_t chargeFactor = TMath::Power(AliPID::ParticleCharge(type),2.);
    Float_t bethe = Bethe(mom,type,isSA)*chargeFactor;
    return (signal - bethe)/GetResolution(bethe,nPtsForPid,isSA,mom,type);
  }
  Int_t GetParticleIdFromdEdxVsP(Float_t mom, Float_t signal, Bool_t isSA=kFALSE) const;
  
private:
  
  
  // Data members for truncated mean method
  Float_t  fRes;             // relative dEdx resolution
  Double_t fKp1;             // ALEPH BB param 1
  Double_t fKp2;             // ALEPH BB param 2
  Double_t fKp3;             // ALEPH BB param 3
  Double_t fKp4;             // ALEPH BB param 4
  Double_t fKp5;             // ALEPH BB param 
  Double_t  fBBsa[5];        // parameters of BB for SA tracks
  Double_t  fBBsaHybrid[9];  // parameters of Hybrid BB for SA tracks, PHOB + Polinomial al low beta*gamma
  Double_t  fBBsaElectron[5];// parameters of electron BB for SA tracks
  Double_t  fBBtpcits[5];     // parameters of BB for TPC+ITS tracks
  Double_t fBBdeu[5]; // parameters of deuteron BB for TPC+ITS tracks
  Double_t fBBtri[5]; // parameters of triton BB for TPC+ITS tracks
  Float_t  fResolSA[5];      // resolutions vs. n. of SDD/SSD points
  Float_t  fResolTPCITS[5];  // resolutions vs. n. of SDD/SSD points
  Double_t fResolTPCITSDeu3[3]; // deuteron resolutions vs. p for tracks with 3 SDD/SSD points
  Double_t fResolTPCITSDeu4[3]; // deuteron resolutions vs. p for tracks with 4 SDD/SSD points
  Double_t fResolTPCITSTri3[3]; // triton resolutions vs. p for tracks with 3 SDD/SSD points
  Double_t fResolTPCITSTri4[3]; // triton resolutions vs. p for tracks with 4 SDD/SSD points

  Double_t Bethe(Double_t bg, const Double_t * const par, Bool_t isNuclei) const;
  ClassDef(AliITSPIDResponse,5)   // ITS PID class
};

#endif


