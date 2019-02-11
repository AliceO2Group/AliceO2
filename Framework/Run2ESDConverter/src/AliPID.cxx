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

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// particle id probability densities                                         //
//                                                                           //
// The AliPID class stores the probability densities for the different       //
// particle type hypotheses electron, muon, pion, kaon, proton, photon,      //
// pi0, neutron, K0 and electron conversion. These probability densities     //
// are determined from the detector response functions.                      //
// The * and *= operators are overloaded for AliPID to combine the PIDs      //
// from different detectors.                                                 //
//                                                                           //
// The Bayesian probability to be a particle of a given type can be          //
// calculated from the probability densities, if the a priori probabilities  //
// (or abundences, concentrations) of particle species are known. These      //
// priors can be given as argument to the GetProbability or GetMostProbable  //
// method or they can be set globally by calling the static method           //
// SetPriors().                                                              //
//                                                                           //
// The implementation of this class is based on the note ...                 //
// by Iouri Belikov and Karel Safarik.                                       //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include <TClass.h>
#include <TDatabasePDG.h>
#include <TPDGCode.h>

#include "AliLog.h"
#include "AliPDG.h"
#include "AliPID.h"

#define M(PID) TDatabasePDG::Instance()->GetParticle(fgkParticleCode[(PID)])->Mass()

ClassImp(AliPID)

const char* AliPID::fgkParticleName[AliPID::kSPECIESCN+1] = {
  "electron",
  "muon",
  "pion",
  "kaon",  
  "proton",
  
  "deuteron",
  "triton",
  "helium-3",
  "alpha",
  
  "photon",
  "pi0",
  "neutron",
  "kaon0",
  "eleCon",
  
  "unknown"
};

const char* AliPID::fgkParticleShortName[AliPID::kSPECIESCN+1] = {
  "e",
  "mu",
  "pi",
  "K",
  "p",

  "d",
  "t",
  "he3",
  "alpha",

  "photon",
  "pi0",
  "n",
  "K0",
  "eleCon",
  
  "unknown"
};

const char* AliPID::fgkParticleLatexName[AliPID::kSPECIESCN+1] = {
  "e",
  "#mu",
  "#pi",
  "K",
  "p",

  "d",
  "t",
  "^{3}He",
  "#alpha",

  "#gamma",
  "#pi_{0}",
  "n",
  "K_{0}",
  "eleCon",

  "unknown"
};

const Int_t AliPID::fgkParticleCode[AliPID::kSPECIESCN+1] = {
  ::kElectron, 
  ::kMuonMinus, 
  ::kPiPlus, 
  ::kKPlus, 
  ::kProton,
  
  1000010020,
  1000010030,
  1000020030,
  1000020040,

  ::kGamma,
  ::kPi0,
  ::kNeutron,
  ::kK0,
  ::kElectron,
  0
};

/*const*/ Float_t AliPID::fgkParticleMass[AliPID::kSPECIESCN+1] = {
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
  /*
  M(kElectron),  // electron
  M(kMuon), // muon
  M(kPion),    // pion
  M(kKaon),     // kaon
  M(kProton),    // proton
  M(kPhoton),     // photon
  M(kPi0),       // pi0
  M(kNeutron),   // neutron
  M(kKaon0),        // kaon0
  M(kEleCon),     // electron conversion
  M(kDeuteron), // deuteron
  M(kTriton),   // triton
  M(kHe3),      // he3
  M(kAlpha),    // alpha
  0.00000        // unknown
  */
};
/*const*/ Float_t AliPID::fgkParticleMassZ[AliPID::kSPECIESCN+1] = {
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
  /*
  M(kElectron),  // electron
  M(kMuon), // muon
  M(kPion),    // pion
  M(kKaon),     // kaon
  M(kProton),    // proton
  M(kPhoton),     // photon
  M(kPi0),       // pi0
  M(kNeutron),   // neutron
  M(kKaon0),        // kaon0
  M(kEleCon),     // electron conversion
  M(kDeuteron), // deuteron
  M(kTriton),   // triton
  M(kHe3)/2,      // he3
  M(kAlpha)/2,    // alpha
  0.00000        // unknown
  */
};

Char_t AliPID::fgkParticleCharge[AliPID::kSPECIESCN+1] = { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 };

Double_t AliPID::fgPrior[kSPECIESCN] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};


//_______________________________________________________________________
AliPID::AliPID() :
  TObject(),
  fCharged(0)
{
  //
  // Default constructor
  //
  Init();
  // set default values (= equal probabilities)
  for (Int_t i = 0; i < kSPECIESCN; i++)
    fProbDensity[i] = 1./kSPECIESCN;
}

//_______________________________________________________________________
AliPID::AliPID(const Double_t* probDensity, Bool_t charged) : 
  TObject(),
  fCharged(charged)
{
  //
  // Standard constructor
  //
  Init();
  // set given probability densities
  for (Int_t i = 0; i < kSPECIESC; i++)
    fProbDensity[i] = probDensity[i];

  for (Int_t i = kSPECIESC; i < kSPECIESCN; i++)
    fProbDensity[i] = ((charged) ? 0 : probDensity[i]);
}

//_______________________________________________________________________
AliPID::AliPID(const Float_t* probDensity, Bool_t charged) :
  TObject(),
  fCharged(charged)
{
  //
  // Standard constructor
  //
  Init();
  // set given probability densities
  for (Int_t i = 0; i < kSPECIESC; i++) 
    fProbDensity[i] = probDensity[i];

  for (Int_t i = kSPECIESC; i < kSPECIESCN; i++) 
    fProbDensity[i] = ((charged) ? 0 : probDensity[i]);
}

//_______________________________________________________________________
AliPID::AliPID(const AliPID& pid) : 
  TObject(pid),
  fCharged(pid.fCharged)
{
  //
  // copy constructor
  //
  // We do not call init here, MUST already be done
  for (Int_t i = 0; i < kSPECIESCN; i++) 
    fProbDensity[i] = pid.fProbDensity[i];
}

//_______________________________________________________________________
void AliPID::SetProbabilities(const Double_t* probDensity, Bool_t charged) 
{
  //
  // Set the probability densities
  //
  for (Int_t i = 0; i < kSPECIESC; i++) 
    fProbDensity[i] = probDensity[i];

  for (Int_t i = kSPECIESC; i < kSPECIESCN; i++) 
    fProbDensity[i] = ((charged) ? 0 : probDensity[i]);
}

//_______________________________________________________________________
AliPID& AliPID::operator = (const AliPID& pid)
{
// assignment operator

  if(this != &pid) {
    fCharged = pid.fCharged;
    for (Int_t i = 0; i < kSPECIESCN; i++) {
      fProbDensity[i] = pid.fProbDensity[i];
    }
  }
  return *this;
}

//_______________________________________________________________________
void AliPID::Init() 
{
  //
  // Initialise the masses, charges
  //
  // Initialise only once... 
  if(!fgkParticleMass[0]) {
    AliPDG::AddParticlesToPdgDataBase();
    for (Int_t i = 0; i < kSPECIESC; i++) {
      fgkParticleMass[i] = M(i);
      if (i == kHe3 || i == kAlpha) {
	fgkParticleMassZ[i] = M(i)/2.;
	fgkParticleCharge[i] = 2;
      }
      else {
	fgkParticleMassZ[i]=M(i);
	fgkParticleCharge[i]=1;
      }
    }
  }
}

//_____________________________________________________________________________
Double_t AliPID::GetProbability(EParticleType iType,
				const Double_t* prior) const
{
  //
  // Get the probability to be a particle of type "iType"
  // assuming the a priori probabilities "prior"
  //
  Double_t sum = 0.;
  Int_t nSpecies = ((fCharged) ? kSPECIESC : kSPECIESCN);
  for (Int_t i = 0; i < nSpecies; i++) {
    sum += fProbDensity[i] * prior[i];
  }
  if (sum <= 0) {
    AliError("Invalid probability densities or priors");
    return -1;
  }
  return fProbDensity[iType] * prior[iType] / sum;
}

//_____________________________________________________________________________
Double_t AliPID::GetProbability(EParticleType iType) const
{
// get the probability to be a particle of type "iType"
// assuming the globaly set a priori probabilities

  return GetProbability(iType, fgPrior);
}

//_____________________________________________________________________________
void AliPID::GetProbabilities(Double_t* probabilities,
			      const Double_t* prior) const
{
// get the probabilities to be a particle of given type
// assuming the a priori probabilities "prior"

  Double_t sum = 0.;
  Int_t nSpecies = ((fCharged) ? kSPECIESC : kSPECIESCN);
  for (Int_t i = 0; i < nSpecies; i++) {
    sum += fProbDensity[i] * prior[i];
  }
  if (sum <= 0) {
    AliError("Invalid probability densities or priors");
    for (Int_t i = 0; i < nSpecies; i++) probabilities[i] = -1;
    return;
  }
  for (Int_t i = 0; i < nSpecies; i++) {
    probabilities[i] = fProbDensity[i] * prior[i] / sum;
  }
}

//_____________________________________________________________________________
void AliPID::GetProbabilities(Double_t* probabilities) const
{
// get the probabilities to be a particle of given type
// assuming the globaly set a priori probabilities

  GetProbabilities(probabilities, fgPrior);
}

//_____________________________________________________________________________
AliPID::EParticleType AliPID::GetMostProbable(const Double_t* prior) const
{
// get the most probable particle id hypothesis
// assuming the a priori probabilities "prior"

  Double_t max = 0.;
  EParticleType id = kPion;
  Int_t nSpecies = ((fCharged) ? kSPECIESC : kSPECIESCN);
  for (Int_t i = 0; i < nSpecies; i++) {
    Double_t prob = fProbDensity[i] * prior[i];
    if (prob > max) {
      max = prob;
      id = EParticleType(i);
    }
  }
  if (max == 0) {
    AliError("Invalid probability densities or priors");
  }
  return id;
}

//_____________________________________________________________________________
AliPID::EParticleType AliPID::GetMostProbable() const
{
// get the most probable particle id hypothesis
// assuming the globaly set a priori probabilities

  return GetMostProbable(fgPrior);
}


//_____________________________________________________________________________
void AliPID::SetPriors(const Double_t* prior, Bool_t charged)
{
// use the given priors as global a priori probabilities

  Double_t sum = 0;
  for (Int_t i = 0; i < kSPECIESCN; i++) {
    if (charged && (i >= kSPECIESC)) {
      fgPrior[i] = 0;      
    } else {
      if (prior[i] < 0) {
	AliWarningClass(Form("negative prior (%g) for %ss. "
			     "Using 0 instead.", prior[i], 
			     fgkParticleName[i]));
	fgPrior[i] = 0;
      } else {
	fgPrior[i] = prior[i];
      }
    }
    sum += prior[i];
  }
  if (sum == 0) {
    AliWarningClass("all priors are zero.");
  }
}

//_____________________________________________________________________________
void AliPID::SetPrior(EParticleType iType, Double_t prior)
{
// use the given prior as global a priori probability for particles
// of type "iType"

  if (prior < 0) {
    AliWarningClass(Form("negative prior (%g) for %ss. Using 0 instead.", 
			 prior, fgkParticleName[iType]));
    prior = 0;
  }
  fgPrior[iType] = prior;
}


//_____________________________________________________________________________
AliPID& AliPID::operator *= (const AliPID& pid)
{
// combine this probability densities with the one of "pid"

  for (Int_t i = 0; i < kSPECIESCN; i++) {
    fProbDensity[i] *= pid.fProbDensity[i];
  }
  return *this;
}

//_____________________________________________________________________________
AliPID operator * (const AliPID& pid1, const AliPID& pid2)
{
// combine the two probability densities

  AliPID result;
  result *= pid1;
  result *= pid2;
  return result;
}
