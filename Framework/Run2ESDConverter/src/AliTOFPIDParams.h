#ifndef ALITOFPIDPARAMS_H
#define ALITOFPIDPARAMS_H
/* Copyright(c) 1998-2010, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

//***********************************************************
// Class AliTODPIDparams
// class to store PID parameters for TOF in OADB
// Author: P. Antonioli, pietro.antonioli@to.infn.it
//***********************************************************

#include <TObject.h>
#include <TNamed.h>
#include "AliPIDResponse.h"

class AliTOFPIDParams : public TNamed {

 public:
  AliTOFPIDParams();
  AliTOFPIDParams(Char_t * name);
  virtual ~AliTOFPIDParams();

  enum {kSigPparams = 4};

  Float_t GetTOFresolution(void) const {return fTOFresolution;}
  AliPIDResponse::EStartTimeType_t GetStartTimeMethod(void) const {return fStartTime;}
  Float_t GetSigParams(Int_t i) const {
    return ((i >= 0)  && (i<kSigPparams)) ? fSigPparams[i] : 0;}    
  Float_t GetTOFtail(void) const {return fTOFtail;}
  Float_t GetTOFmatchingLossMC(void) const {return fTOFmatchingLossMC;}
  Float_t GetTOFadditionalMismForMC(void) const {return fTOFadditionalMismForMC;}
  Float_t GetTOFtimeOffset(void) const {return fTOFtimeOffset;}
  const char *GetOADBentryTag() const {return fOADBentryTag.Data();}

  void SetOADBentryTag(const char* entry) {fOADBentryTag=entry;}
  void SetTOFresolution(Float_t res){fTOFresolution = res;}
  void SetStartTimeMethod(AliPIDResponse::EStartTimeType_t method){fStartTime=method;}
  void SetSigPparams(Float_t *params);
  void SetTOFtail(Float_t tail){fTOFtail = tail;}
  void SetTOFmatchingLossMC(Float_t lossMC){fTOFmatchingLossMC = lossMC;}
  void SetTOFadditionalMismForMC(Float_t misMC){fTOFadditionalMismForMC = misMC;}
  void SetTOFtimeOffset(Float_t timeOffset){fTOFtimeOffset = timeOffset;}

 private:
  AliPIDResponse::EStartTimeType_t fStartTime;      // startTime method
  Float_t fTOFresolution;                           // TOF MRPC intrinsic resolution
  Float_t fSigPparams[kSigPparams];                 // parameterisation of sigma(p) dependency 
  Float_t fTOFtail;                                 // fraction of tracks with TOF signal within gaussian behaviour
  Float_t fTOFmatchingLossMC;                       // fraction of tracks (%) MC has to loose to follow reconstructed data performance
  Float_t fTOFadditionalMismForMC;                  // fraction of tracks (%) MC has to add to match mismatch percentage in data
  Float_t fTOFtimeOffset;                           // overall offset to be added to startTime to handle intercalibration issues
  TString fOADBentryTag;                            // code name (period-pass) for the entry

  ClassDef(AliTOFPIDParams,3);

};

#endif

