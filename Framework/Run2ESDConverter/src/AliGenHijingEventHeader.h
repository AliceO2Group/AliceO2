#ifndef ALIGENHIJINGEVENTHEADER_H
#define ALIGENHIJINGEVENTHEADER_H

/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

/* $Id$ */

#include <TLorentzVector.h>

#include "AliGenEventHeader.h"
#include "AliCollisionGeometry.h"

class AliGenHijingEventHeader : public AliGenEventHeader, public AliCollisionGeometry
{
 public:
    AliGenHijingEventHeader(const char* name);
  AliGenHijingEventHeader();
  virtual ~AliGenHijingEventHeader() {}
  // Getters
  Float_t TotalEnergy() const {return fTotalEnergy;} 
  Int_t   Trials() const {return fTrials;}
  Int_t   GetTrueNPart() const {return fNPart;}
  Bool_t  GetSpectatorsInTheStack() const {return fAreSpectatorsInTheStack;}
  Bool_t  GetFragmentationFromData() const {return fIsDataFragmentationSet;}
  Int_t   GetFreeProjSpecn() const {return  fFreeProjSpecn;}
  Int_t   GetFreeProjSpecp() const {return  fFreeProjSpecp;}
  Int_t   GetFreeTargSpecn() const {return  fFreeTargSpecn;}
  Int_t   GetFreeTargSpecp() const {return  fFreeTargSpecp;}
 	  
  // Setters
  void SetTotalEnergy(Float_t energy)  {fTotalEnergy=energy;}
  void SetJets(const TLorentzVector* jet1, const TLorentzVector* jet2,
	       const TLorentzVector* jet3, const TLorentzVector* jet4)
      {fJet1 = *jet1; fJet2 = *jet2; fJetFsr1 = *jet3; fJetFsr2 = *jet4;}
  void GetJets(TLorentzVector& jet1, TLorentzVector& jet2,
	       TLorentzVector& jet3, TLorentzVector& jet4) const  
      {jet1 = fJet1; jet2 = fJet2; jet3 = fJetFsr1; jet4 = fJetFsr2;}
  void SetTrials(Int_t trials) {fTrials = trials;}
  void SetTrueNPart(Int_t npart) {fNPart = npart;} 
  void SetSpectatorsInTheStack(Bool_t what) {fAreSpectatorsInTheStack=what;}
  void SetDataFromFragmentation(Bool_t what) {fIsDataFragmentationSet=what;}
  void SetFreeSpectators(Int_t specnproj, Int_t specpproj, Int_t specntarg, Int_t specptarg) 
       {fFreeProjSpecn=specnproj; fFreeProjSpecp=specpproj; fFreeTargSpecn=specntarg; fFreeTargSpecp=specptarg;}
 
protected:
  Float_t fTotalEnergy;              // Total energy of produced particles
  Int_t   fTrials;                   // Number of trials to fulfill trigger condition
  Int_t   fNPart;                    // True number of participants 
  TLorentzVector  fJet1;             // 4-Momentum-Vector of first   triggered jet  
  TLorentzVector  fJet2;             // 4-Momentum-Vector of second  triggered jet     
  TLorentzVector  fJetFsr1;          // 4-Momentum-Vector of first   triggered jet  
  TLorentzVector  fJetFsr2;          // 4-Momentum-Vector of second  triggered jet     
  // Added by Chiara O. for spectator generation
  Bool_t  fAreSpectatorsInTheStack;  // check whether spectators are in the stack
  Bool_t  fIsDataFragmentationSet;   // check if the data driven correction is switched on
  Int_t       fFreeProjSpecn;        // Num. of spectator neutrons from projectile nucleus
  Int_t       fFreeProjSpecp;        // Num. of spectator protons from projectile nucleus
  Int_t       fFreeTargSpecn;	     // Num. of spectator neutrons from target nucleus
  Int_t       fFreeTargSpecp;	     // Num. of spectator protons from target nucleus
  
  ClassDef(AliGenHijingEventHeader,6) // Event header for hijing event
};

#endif
