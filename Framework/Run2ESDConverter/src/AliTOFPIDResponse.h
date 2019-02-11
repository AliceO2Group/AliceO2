#ifndef ALITOFPIDRESPONSE_H
#define ALITOFPIDRESPONSE_H

/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

//-------------------------------------------------------
//                    TOF PID class
//   Origin: Iouri Belikov, CERN, Jouri.Belikov@cern.ch 
//-------------------------------------------------------

#include <TObject.h>
#include <TF1.h>
#include "AliPID.h"
#include "AliVParticle.h"
#include "AliVTrack.h"

class AliTOFPIDParams;
class TH1F;
class TH1D;

class AliTOFPIDResponse : public TObject {
public:

  AliTOFPIDResponse();
  AliTOFPIDResponse(Double_t *param);
  ~AliTOFPIDResponse(){}

  void     SetTimeResolution(Float_t res) { fSigma = res; }
  void     SetTimeZero(Double_t t0) { fTime0=t0; }
  Double_t GetTimeZero() const { return fTime0; }
  Float_t  GetTimeResolution() const { return fSigma; }

  void     SetMaxMismatchProbability(Double_t p) {fPmax=p;}
  Double_t GetMaxMismatchProbability() const {return fPmax;}

  Double_t GetExpectedSigma(Float_t mom, Float_t tof, Float_t massZ) const;
  Double_t GetExpectedSigma(Float_t mom, Float_t tof, AliPID::EParticleType type) const;
  Double_t GetExpectedSignal(const AliVTrack *track, AliPID::EParticleType type) const;

  Double_t GetMismatchProbability(Double_t time,Double_t eta) const;

  static Double_t GetTailRandomValue(Float_t pt=1.0,Float_t eta=0.0,Float_t time=0.0,Float_t addmism=0.0); // generate a random value to add a tail to TOF time (for MC analyses), addmism = additional mismatch in percentile
  static Double_t GetMismatchRandomValue(Float_t eta); // generate a random value for mismatched tracks (for MC analyses)

  void     SetT0event(Float_t *t0event){for(Int_t i=0;i < fNmomBins;i++) fT0event[i] = t0event[i];};
  void     SetT0resolution(Float_t *t0resolution){for(Int_t i=0;i < fNmomBins;i++) fT0resolution[i] = t0resolution[i];};
  void     ResetT0info(){ for(Int_t i=0;i < fNmomBins;i++){ fT0event[i] = 0.0; fT0resolution[i] = 0.0; fMaskT0[i] = 0;} };
  void     SetMomBoundary();
  Int_t    GetMomBin(Float_t p) const;
  Int_t    GetNmomBins(){return fNmomBins;};
  Float_t  GetMinMom(Int_t ibin) const {if(ibin >=0 && ibin < fNmomBins) return fPCutMin[ibin]; else return 0.0;}; // overrun static array - coverity
  Float_t  GetMaxMom(Int_t ibin) const {if(ibin >=0 && ibin < fNmomBins) return fPCutMin[ibin+1]; else return 0.0;}; // overrun static array - coverity
  void     SetT0bin(Int_t ibin,Float_t t0bin){if(ibin >=0 && ibin < fNmomBins) fT0event[ibin] = t0bin;}; // overrun static array - coverity
  void     SetT0binRes(Int_t ibin,Float_t t0binRes){if(ibin >=0 && ibin < fNmomBins) fT0resolution[ibin] = t0binRes;}; // overrun static array - coverity
  void     SetT0binMask(Int_t ibin,Int_t t0binMask){if(ibin >=0 && ibin < fNmomBins) fMaskT0[ibin] = t0binMask;}; // overrun static array - coverity
  Float_t  GetT0bin(Int_t ibin) const {if(ibin >=0 && ibin < fNmomBins) return fT0event[ibin]; else return 0.0;}; // overrun static array - coverity
  Float_t  GetT0binRes(Int_t ibin) const {if(ibin >=0 && ibin < fNmomBins) return fT0resolution[ibin]; else return 0.0;}; // overrun static array - coverity
  Int_t    GetT0binMask(Int_t ibin) const {if(ibin >=0 && ibin < fNmomBins) return fMaskT0[ibin]; else return 0;}; // overrun static array - coverity

  // Get Start Time for a track
  Float_t  GetStartTime(Float_t mom) const;
  Float_t  GetStartTimeRes(Float_t mom) const;
  Int_t    GetStartTimeMask(Float_t mom) const;

  // Tracking resolution for expected times
  void SetTrackParameter(Int_t ip,Float_t value){if(ip>=0 && ip < 4) fPar[ip] = value;};
  Float_t GetTrackParameter(Int_t ip){if(ip>=0 && ip < 4) return fPar[ip]; else return -1.0;};
  Int_t GetTOFchannel(AliVParticle *trk) const;

  Float_t GetTOFtail() {if(fTOFtailResponse) return fTOFtailResponse->GetParameter(3);else return -1;};
  void    SetTOFtail(Float_t tail);
  void    SetTOFtailAllPara(Float_t mean,Float_t tail);

 private:
  Int_t LoadTOFtailHisto();

  Double_t fSigma;        // intrinsic TOF resolution

  // obsolete
  Double_t fPmax;         // "maximal" probability of mismathing (at ~0.5 GeV/c)
  Double_t fTime0;        // time zero
  //--------------

  // About event time (t0) info
  static const Int_t fNmomBins = 10; // number of momentum bin 
  Float_t fT0event[fNmomBins];    // t0 (best, T0, T0-TOF, ...) of the event as a function of p 
  Float_t fT0resolution[fNmomBins]; // t0 (best, T0, T0-TOF, ...) resolution as a function of p 
  Float_t fPCutMin[fNmomBins+1]; // min values for p bins
  Int_t fMaskT0[fNmomBins]; // mask withthe T0 used (0x1=T0-TOF,0x2=T0A,0x3=TOC) for p bins
  Float_t fPar[4]; // parameter for expected times resolution

  static TF1 *fTOFtailResponse; // function to generate a TOF tail
  static TH1F *fHmismTOF; // TOF mismatch distribution
  static TH1D *fHchannelTOFdistr;// TOF channel distance distribution
  static TH1D *fHTOFtailResponse;// histogram to generate a TOF tail

  ClassDef(AliTOFPIDResponse,6)   // TOF PID class
};

#endif
