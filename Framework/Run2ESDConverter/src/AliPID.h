#ifndef ALIPID_H
#define ALIPID_H

/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

///////////////////////////////////////////////////////////////////////////////
///                                                                          //
/// particle id probability densities                                        //
///                                                                          //
///////////////////////////////////////////////////////////////////////////////

/* $Id$ */


#include <TObject.h>
#include <TMath.h>

class AliPID : public TObject {
 public:
  enum {
    kSPECIES = 5,     // Number of default particle species recognized by the PID
    kSPECIESC = 9,    // Number of default particles + light nuclei recognized by the PID
    kSPECIESCN = 14,  // Number of charged+neutral particle species recognized by the PHOS/EMCAL PID
    kCharges = 3      // Number of possible charge options for TRD Parameters
  };
  enum EParticleType {
    kElectron = 0, 
    kMuon = 1, 
    kPion = 2, 
    kKaon = 3, 
    kProton = 4, 

    kDeuteron = 5,
    kTriton = 6,
    kHe3 = 7,
    kAlpha = 8,
    
    kPhoton = 9,
    kPi0 = 10, 
    kNeutron = 11, 
    kKaon0 = 12, 
    kEleCon = 13,
    
    kUnknown = 14
  };
  enum eTRDparticleCharge {
    kNoCharge = 0,
    kPosCharge = 1,
    kNegCharge = 2
  };
  static Int_t         ParticleCharge(Int_t iType) {
     if(!fgkParticleMass[0]) Init(); 
     return fgkParticleCharge[iType];
  }
  static Float_t       ParticleMass(Int_t iType) {
     if(!fgkParticleMass[0]) Init(); 
     return fgkParticleMass[iType];
  }
  static Float_t       ParticleMassZ(Int_t iType) {
     if(!fgkParticleMass[0]) Init(); 
     return fgkParticleMassZ[iType];
  }
  static const char*   ParticleName(Int_t iType) 
    {return fgkParticleName[iType];};
  static const char*   ParticleShortName(Int_t iType) 
    {return fgkParticleShortName[iType];};
  static const char*   ParticleLatexName(Int_t iType) 
    {return fgkParticleLatexName[iType];};
  static Int_t         ParticleCode(Int_t iType) 
    {return fgkParticleCode[iType];};

  AliPID();
  AliPID(const Double_t* probDensity, Bool_t charged = kTRUE);
  AliPID(const Float_t* probDensity, Bool_t charged = kTRUE);
  AliPID(const AliPID& pid);
  AliPID& operator = (const AliPID& pid);

  Double_t             GetProbability(EParticleType iType,
				      const Double_t* prior) const;
  Double_t             GetProbability(EParticleType iType) const;
  void                 GetProbabilities(Double_t* probabilities,
					const Double_t* prior) const;
  void                 GetProbabilities(Double_t* probabilities) const;
  EParticleType        GetMostProbable(const Double_t* prior) const;
  EParticleType        GetMostProbable() const;
  
  void                 SetProbabilities(const Double_t* probabilities,
                                        Bool_t charged = kTRUE);

  static void          SetPriors(const Double_t* prior,
				 Bool_t charged = kTRUE);
  static void          SetPrior(EParticleType iType, Double_t prior);

  AliPID&              operator *= (const AliPID& pid);

 private:

  static void          Init();

  Bool_t               fCharged;                           // flag for charged/neutral
  Double_t             fProbDensity[kSPECIESCN];           // probability densities
  static Double_t      fgPrior[kSPECIESCN];                // a priori probabilities

  static /*const*/ Float_t fgkParticleMass[kSPECIESCN+1];  // particle masses
  static /*const*/ Float_t fgkParticleMassZ[kSPECIESCN+1]; // particle masses/charge
  static /*const*/ Char_t  fgkParticleCharge[kSPECIESCN+1]; // particle charge (in e units!)
  static const char*   fgkParticleName[kSPECIESCN+1];      // particle names
  static const char*   fgkParticleShortName[kSPECIESCN+1]; // particle names
  static const char*   fgkParticleLatexName[kSPECIESCN+1]; // particle names
  static const Int_t   fgkParticleCode[kSPECIESCN+1];      // particle codes

  ClassDef(AliPID, 5)                                      // particle id probability densities
};


AliPID operator * (const AliPID& pid1, const AliPID& pid2);


#endif
