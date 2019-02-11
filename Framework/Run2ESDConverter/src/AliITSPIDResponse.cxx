/**************************************************************************
 * Copyright(c) 2005-2007, ALICE Experiment at CERN, All rights reserved. *
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

//-----------------------------------------------------------------
// ITS PID method # 1
//           Implementation of the ITS PID class
// Very naive one... Should be made better by the detector experts...
//      Origin: Iouri Belikov, CERN, Jouri.Belikov@cern.ch
//-----------------------------------------------------------------
#include "TMath.h"
#include "AliVTrack.h"
#include "AliITSPIDResponse.h"
#include "AliITSPidParams.h"
#include "AliExternalTrackParam.h"

ClassImp(AliITSPIDResponse)

AliITSPIDResponse::AliITSPIDResponse(Bool_t isMC):
  fRes(0.13),
  fKp1(15.77),
  fKp2(4.95),
  fKp3(0.312),
  fKp4(2.14),
  fKp5(0.82)
{
  if(!isMC){
    fBBtpcits[0]=0.73;
    fBBtpcits[1]=14.68;
    fBBtpcits[2]=0.905;
    fBBtpcits[3]=1.2;
    fBBtpcits[4]=6.6;
    fBBdeu[0]=76.43; // parameters for the deuteron - tpcits - value from PbPb 2010 run (S.Trogolo - July 2014)
    fBBdeu[1]=-34.21;
    fBBdeu[2]=113.2;
    fBBdeu[3]=-18.12;
    fBBdeu[4]=0.6019;
    fBBtri[0]=13.34; // parameters for the triton - tpcits - value from PbPb 2010 run (S.Trogolo - July 2014)
    fBBtri[1]=55.17;
    fBBtri[2]=66.41;
    fBBtri[3]=-6.601;
    fBBtri[4]=-0.4134;
    fBBsa[0]=2.73198E7; //pure PHOBOS parameterization
    fBBsa[1]=6.92389;
    fBBsa[2]=1.90088E-6;
    fBBsa[3]=1.90088E-6;
    fBBsa[4]=3.40644E-7;
    fBBsaHybrid[0]=1.43505E7;  //PHOBOS+Polinomial parameterization
    fBBsaHybrid[1]=49.3402;
    fBBsaHybrid[2]=1.77741E-7;
    fBBsaHybrid[3]=1.77741E-7;
    fBBsaHybrid[4]=1.01311E-7;
    fBBsaHybrid[5]=77.2777;
    fBBsaHybrid[6]=33.4099;
    fBBsaHybrid[7]=46.0089;
    fBBsaHybrid[8]=-2.26583;
    fBBsaElectron[0]=4.05799E6;  //electrons in the ITS
    fBBsaElectron[1]=38.5713;
    fBBsaElectron[2]=1.46462E-7;
    fBBsaElectron[3]=1.46462E-7;
    fBBsaElectron[4]=4.40284E-7;
    fResolSA[0]=1.;   // 0 cluster tracks should not be used
    fResolSA[1]=0.25;  // rough values for tracks with 1
    fResolSA[2]=0.131;   // value from pp 2010 run (L. Milano, 16-Jun-11)
    fResolSA[3]=0.113; // value from pp 2010 run
    fResolSA[4]=0.104; // value from pp 2010 run
    for(Int_t i=0; i<5;i++) fResolTPCITS[i]=0.13;
    fResolTPCITSDeu3[0]=0.06918; // deuteron resolution vs p
    fResolTPCITSDeu3[1]=0.02498; // 3 ITS clusters for PId
    fResolTPCITSDeu3[2]=1.1; // value from PbPb 2010 run (July 2014)
    fResolTPCITSDeu4[0]=0.06756;// deuteron resolution vs p
    fResolTPCITSDeu4[1]=0.02078; // 4 ITS clusters for PId
    fResolTPCITSDeu4[2]=1.05; // value from PbPb 2010 run (July 2014)
    fResolTPCITSTri3[0]=0.07239; // triton resolution vs p
    fResolTPCITSTri3[1]=0.0192; // 3 ITS clusters for PId
    fResolTPCITSTri3[2]=1.1; // value from PbPb 2010 run (July 2014)
    fResolTPCITSTri4[0]=0.06083; // triton resolution
    fResolTPCITSTri4[1]=0.02579; // 4 ITS clusters for PId
    fResolTPCITSTri4[2]=1.15; // value from PbPb 2010 run (July 2014)
  }else{
    fBBtpcits[0]=1.04;
    fBBtpcits[1]=27.14;
    fBBtpcits[2]=1.00;
    fBBtpcits[3]=0.964;
    fBBtpcits[4]=2.59;
    fBBdeu[0]=88.22; // parameters for the deuteron - MC (LHC14a6)
    fBBdeu[1]=-40.74;
    fBBdeu[2]=107.2;
    fBBdeu[3]=-8.962;
    fBBdeu[4]=-0.766;
    fBBtri[0]=100.7; //parameters for the triton - MC (LHC14a6)
    fBBtri[1]=-68.56;
    fBBtri[2]=128.2;
    fBBtri[3]=-15.5;
    fBBtri[4]=0.1833;
    fBBsa[0]=2.02078E7; //pure PHOBOS parameterization
    fBBsa[1]=14.0724;
    fBBsa[2]=3.84454E-7;
    fBBsa[3]=3.84454E-7;
    fBBsa[4]=2.43913E-7;
    fBBsaHybrid[0]=1.05381E7; //PHOBOS+Polinomial parameterization
    fBBsaHybrid[1]=89.3933;
    fBBsaHybrid[2]=2.4831E-7;
    fBBsaHybrid[3]=2.4831E-7;
    fBBsaHybrid[4]=7.80591E-8;
    fBBsaHybrid[5]=62.9214;
    fBBsaHybrid[6]=32.347;
    fBBsaHybrid[7]=58.7661;
    fBBsaHybrid[8]=-3.39869;
    fBBsaElectron[0]=2.26807E6; //electrons in the ITS
    fBBsaElectron[1]=99.985;
    fBBsaElectron[2]=0.000714841;
    fBBsaElectron[3]=0.000259585;
    fBBsaElectron[4]=1.39412E-7;
    fResolSA[0]=1.;   // 0 cluster tracks should not be used
    fResolSA[1]=0.25;  // rough values for tracks with 1
    fResolSA[2]=0.126;   // value from pp 2010 simulations (L. Milano, 16-Jun-11)
    fResolSA[3]=0.109; // value from pp 2010 simulations
    fResolSA[4]=0.097; // value from pp 2010 simulations
    for(Int_t i=0; i<5;i++) fResolTPCITS[i]=0.13;
    fResolTPCITSDeu3[0]=0.06853; // deuteron resolution vs p
    fResolTPCITSDeu3[1]=0.01607; // 3 ITS clusters for PId
    fResolTPCITSDeu3[2]=1.08; // value from PbPb 2010 run (July 2014)
    fResolTPCITSDeu4[0]=0.06853;
    fResolTPCITSDeu4[1]=0.01607;
    fResolTPCITSDeu4[2]=1.08;
    fResolTPCITSTri3[0]=0.07239; // triton resolution vs p
    fResolTPCITSTri3[1]=0.0192; // 3 ITS clusters for PId
    fResolTPCITSTri3[2]=1.12; // value from PbPb 2010 run (July 2014)
    fResolTPCITSTri4[0]=0.07239; // triton resolution vs p
    fResolTPCITSTri4[1]=0.0192; // 3 ITS clusters for PId
    fResolTPCITSTri4[2]=1.12;
  }
}

/*
//_________________________________________________________________________
AliITSPIDResponse::AliITSPIDResponse(Double_t *param):
  fRes(param[0]),
  fKp1(15.77),
  fKp2(4.95),
  fKp3(0.312),
  fKp4(2.14),
  fKp5(0.82)
{
  //
  //  The main constructor
  //
  for (Int_t i=0; i<5;i++) {
      fBBsa[i]=0.;
      fBBtpcits[i]=0.;
      fResolSA[i]=0.;
      fResolTPCITS[i]=0.;
  }
}
*/

//_________________________________________________________________________
Double_t AliITSPIDResponse::BetheAleph(Double_t p, Double_t mass) const {
  //
  // returns AliExternalTrackParam::BetheBloch normalized to
  // fgMIP at the minimum
  //

  Double_t bb=
    AliExternalTrackParam::BetheBlochAleph(p/mass,fKp1,fKp2,fKp3,fKp4,fKp5);
  return bb;
}

//_________________________________________________________________________
Double_t AliITSPIDResponse::Bethe(Double_t bg, const Double_t * const par, Bool_t isNuclei) const
{

  const Double_t beta = bg/TMath::Sqrt(1.+ bg*bg);
  const Double_t gamma=bg/beta;
  Double_t bb=1.;

  Double_t eff=1.0;
  if(bg<par[2])
    eff=(bg-par[3])*(bg-par[3])+par[4];
  else
    eff=(par[2]-par[3])*(par[2]-par[3])+par[4];

  if(gamma>=0. && beta>0.){
    if(isNuclei){
      //Parameterization for deuteron between 0.4 - 1.5 GeV/c; triton between 0.58 - 1.65 GeV/c
      bb=par[0] + par[1]/bg + par[2]/(bg*bg) + par[3]/(bg*bg*bg) + par[4]/(bg*bg*bg*bg);
    }else{ //Parameterization for pion, kaon, proton, electron
      bb=(par[1]+2.0*TMath::Log(gamma)-beta*beta)*(par[0]/(beta*beta))*eff;
    }
  }

  return bb;
}

//_________________________________________________________________________
Double_t AliITSPIDResponse::Bethe(Double_t p, Double_t mass, Bool_t isSA) const {

  //OLD - Mantained for backward compatibility
  //from the MASS check --> Set the Particle Type
  //at the end use the method Bethe(Double_t p, AliPID::EParticleType species, Bool_t isSA) const to set the right parameter

  //
  // returns AliExternalTrackParam::BetheBloch normalized to
  // fgMIP at the minimum
  //

  // NEW: Parameterization for Deuteron and Triton energy loss, reproduced with a polynomial in fixed p range
  // fBBdeu --> parameters for deuteron
  // fBBtri --> parameters for triton

  //NOTE
  //NOTE: if changes are made here, please also check the alternative function below
  //NOTE

  AliPID::EParticleType species = AliPID::kPion;
  Bool_t foundMatchingSpecies = kFALSE;
  for (Int_t spec = 0; spec < AliPID::kSPECIESC; spec++) {
    if (TMath::AreEqualAbs(mass,AliPID::ParticleMassZ(spec),0.001)){
      species = (AliPID::EParticleType)spec;
      foundMatchingSpecies = kTRUE;
      break;
    }
  }
  if (!foundMatchingSpecies)
    printf("Error AliITSPIDResponse::Bethe: Mass does not match any species. Assuming pion! Note that this function is deprecated!\n");

  return Bethe(p,species,isSA);
}

//_________________________________________________________________________
Double_t AliITSPIDResponse::Bethe(Double_t p, AliPID::EParticleType species, Bool_t isSA) const
{
  // NEW - to be used
  // **** ATTENTION: the second parameter must be the PARTICLE TYPE you want to identify ****
  // Alternative bethe function assuming a particle type not a mass
  // should be slightly faster
  //

  const Double_t m=AliPID::ParticleMassZ(species);
  const Double_t bg=p/m;
  Bool_t isNuclei=kFALSE;

  //NOTE
  //NOTE: if changes are made here, please also check the alternative function above
  //NOTE
  const Double_t *par=fBBtpcits;
  if(isSA){
    if(species == AliPID::kElectron){
      //if is an electron use a specific BB parameterization
      //To be used only between 100 and 160 MeV/c
      par=fBBsaElectron;
    }else{
      par=fBBsa;
    }
  }else{
    if(species == AliPID::kDeuteron) {
      par=fBBdeu;
      isNuclei=kTRUE;
    }
    if(species == AliPID::kTriton  ) {
      par=fBBtri;
      isNuclei=kTRUE;
    }
  }

  return Bethe(bg, par, isNuclei);
}

//_________________________________________________________________________
Double_t AliITSPIDResponse::BetheITSsaHybrid(Double_t p, Double_t mass) const {
  //
  // returns AliExternalTrackParam::BetheBloch normalized to
  // fgMIP at the minimum. The PHOBOS parameterization is used for beta*gamma>0.76.
  // For beta*gamma<0.76 a polinomial function is used

  Double_t bg=p/mass;
  Double_t beta = bg/TMath::Sqrt(1.+ bg*bg);
  Double_t gamma=bg/beta;
  Double_t bb=1.;

  Double_t par[9];
  //parameters for pi, K, p
  for(Int_t ip=0; ip<9;ip++) par[ip]=fBBsaHybrid[ip];
  //if it is an electron the PHOBOS part of the parameterization is tuned for e
  //in the range used for identification beta*gamma is >0.76 for electrons
  //To be used only between 100 and 160 MeV/c
  if(mass>0.0005 && mass<0.00052)for(Int_t ip=0; ip<5;ip++) par[ip]=fBBsaElectron[ip];

  if(gamma>=0. && beta>0. && bg>0.1){
    if(bg>0.76){//PHOBOS
      Double_t eff=1.0;
      if(bg<par[2])
	eff=(bg-par[3])*(bg-par[3])+par[4];
      else
	eff=(par[2]-par[3])*(par[2]-par[3])+par[4];

      bb=(par[1]+2.0*TMath::Log(gamma)-beta*beta)*(par[0]/(beta*beta))*eff;
    }else{//Polinomial
      bb=par[5] + par[6]/bg + par[7]/(bg*bg) + par[8]/(bg*bg*bg);
    }
  }
  return bb;
}

//_________________________________________________________________________
Double_t AliITSPIDResponse::GetResolution(Double_t bethe,
					  Int_t nPtsForPid,
                                         Bool_t isSA,
                                         Double_t p,
                                         AliPID::EParticleType type) const {
  //
  // Calculate expected resolution for truncated mean
  //
  // NEW: Added new variables which are Double_t p and AliPID::EParticleType type
  // AliPID::EParticleType type is used to set the correct resolution for the different particles
  // default -> AliPID::EParticleType type = AliPID::kPion
  // Double_t p is used for the resolution of deuteron and triton, because they are function of the momentum
  // default -> Double_t p=0.

  Float_t r=0.f;
  Double_t c=1.; //this is a correction factor used for the nuclei resolution, while for pion/kaon/proton/electron is 1.

  if(isSA) r=fResolSA[nPtsForPid];
  else{
    const Double_t *par=0x0;
    if(type==AliPID::kDeuteron){
      if(nPtsForPid==4) par = fResolTPCITSDeu4;
      else par = fResolTPCITSDeu3;
      c=par[2];
      r=par[0]+par[1]*p;
    } else if(type==AliPID::kTriton){
      if(nPtsForPid==4) par = fResolTPCITSTri4;
      else par = fResolTPCITSTri3;
      c=par[2];
      r=par[0]+par[1]*p;
    } else{
      r=fResolTPCITS[nPtsForPid];
    }
  }

  return r*bethe*c;
}


//_________________________________________________________________________
void AliITSPIDResponse::GetITSProbabilities(Float_t mom, Double_t qclu[4], Double_t condprobfun[AliPID::kSPECIES], Bool_t isMC) const {
  //
  // Method to calculate PID probabilities for a single track
  // using the likelihood method
  //
  const Int_t nLay = 4;
  const Int_t nPart= 4;

  static AliITSPidParams pars(isMC);  // Pid parametrisation parameters

  Double_t itsProb[nPart] = {1,1,1,1}; // e, p, K, pi

  for (Int_t iLay = 0; iLay < nLay; iLay++) {
    if (qclu[iLay] <= 50.)
      continue;

    Float_t dedx = qclu[iLay];
    Float_t layProb = pars.GetLandauGausNorm(dedx,AliPID::kProton,mom,iLay+3);
    itsProb[0] *= layProb;

    layProb = pars.GetLandauGausNorm(dedx,AliPID::kKaon,mom,iLay+3);
    itsProb[1] *= layProb;

    layProb = pars.GetLandauGausNorm(dedx,AliPID::kPion,mom,iLay+3);
    itsProb[2] *= layProb;

    layProb = pars.GetLandauGausNorm(dedx,AliPID::kElectron,mom,iLay+3);
    itsProb[3] *= layProb;
  }

  // Normalise probabilities
  Double_t sumProb = 0;
  for (Int_t iPart = 0; iPart < nPart; iPart++) {
    sumProb += itsProb[iPart];
  }
  sumProb += itsProb[2]; // muon cannot be distinguished from pions

  for (Int_t iPart = 0; iPart < nPart; iPart++) {
    itsProb[iPart]/=sumProb;
  }
  condprobfun[AliPID::kElectron] = itsProb[3];
  condprobfun[AliPID::kMuon] = itsProb[2];
  condprobfun[AliPID::kPion] = itsProb[2];
  condprobfun[AliPID::kKaon] = itsProb[1];
  condprobfun[AliPID::kProton] = itsProb[0];
  return;
}

//_________________________________________________________________________
void AliITSPIDResponse::GetITSProbabilities(Float_t mom, Double_t qclu[4], Double_t condprobfun[AliPID::kSPECIES], AliITSPidParams *pars) const
{
  //
  // Method to calculate PID probabilities for a single track
  // using the likelihood method
  //
  const Int_t nLay = 4;
  const Int_t nPart= 4;
  
  Double_t itsProb[nPart] = {1,1,1,1}; // e, p, K, pi
  
  for (Int_t iLay = 0; iLay < nLay; iLay++) {
    if (qclu[iLay] <= 50.)
      continue;
    
    Float_t dedx = qclu[iLay];
    Float_t layProb = pars->GetLandauGausNorm(dedx,AliPID::kProton,mom,iLay+3);
    itsProb[0] *= layProb;
    
    layProb = pars->GetLandauGausNorm(dedx,AliPID::kKaon,mom,iLay+3);
    itsProb[1] *= layProb;
    
    layProb = pars->GetLandauGausNorm(dedx,AliPID::kPion,mom,iLay+3);
    itsProb[2] *= layProb;
    
    layProb = pars->GetLandauGausNorm(dedx,AliPID::kElectron,mom,iLay+3);
    itsProb[3] *= layProb;
  }
  
  // Normalise probabilities
  Double_t sumProb = 0;
  for (Int_t iPart = 0; iPart < nPart; iPart++) {
    sumProb += itsProb[iPart];
  }
  sumProb += itsProb[2]; // muon cannot be distinguished from pions
  
  for (Int_t iPart = 0; iPart < nPart; iPart++) {
    itsProb[iPart]/=sumProb;
  }
  condprobfun[AliPID::kElectron] = itsProb[3];
  condprobfun[AliPID::kMuon] = itsProb[2];
  condprobfun[AliPID::kPion] = itsProb[2];
  condprobfun[AliPID::kKaon] = itsProb[1];
  condprobfun[AliPID::kProton] = itsProb[0];
  return;
}

//_________________________________________________________________________
Double_t AliITSPIDResponse::GetNumberOfSigmas( const AliVTrack* track, AliPID::EParticleType type) const
{
  //
  // number of sigmas
  //
  UChar_t clumap=track->GetITSClusterMap();
  Int_t nPointsForPid=0;
  for(Int_t i=2; i<6; i++){
    if(clumap&(1<<i)) ++nPointsForPid;
  }
  Float_t mom=track->P();

  //check for ITS standalone tracks
  Bool_t isSA=kTRUE;
  if( track->GetStatus() & AliVTrack::kTPCin ) isSA=kFALSE;

  Float_t dEdx=track->GetITSsignal();
  if (track->GetITSsignalTunedOnData()>0) dEdx = track->GetITSsignalTunedOnData();

  //TODO: in case of the electron, use the SA parametrisation,
  //      this needs to be changed if ITS provides a parametrisation
  //      for electrons also for ITS+TPC tracks
  return GetNumberOfSigmas(mom,dEdx,type,nPointsForPid,isSA || (type==AliPID::kElectron));
}

//_________________________________________________________________________
Double_t AliITSPIDResponse::GetSignalDelta( const AliVTrack* track, AliPID::EParticleType type, Bool_t ratio/*=kFALSE*/) const
{
  //
  // Signal - expected
  //
  const Float_t mom=track->P();
  const Double_t chargeFactor = TMath::Power(AliPID::ParticleCharge(type),2.);
  Bool_t isSA=kTRUE;
  if( track->GetStatus() & AliVTrack::kTPCin ) isSA=kFALSE;

  Float_t dEdx=track->GetITSsignal();
  if (track->GetITSsignalTunedOnData()>0) dEdx = track->GetITSsignalTunedOnData();


  //TODO: in case of the electron, use the SA parametrisation,
  //      this needs to be changed if ITS provides a parametrisation
  //      for electrons also for ITS+TPC tracks

  const Float_t bethe = Bethe(mom,type, isSA || (type==AliPID::kElectron))*chargeFactor;

  Double_t delta=-9999.;
  if (!ratio) delta=dEdx-bethe;
  else if (bethe>1.e-20) delta=dEdx/bethe;

  return delta;
}

//_________________________________________________________________________
Int_t AliITSPIDResponse::GetParticleIdFromdEdxVsP(Float_t mom, Float_t signal, Bool_t isSA) const{
  // method to get particle identity with simple cuts on dE/dx vs. momentum

  Double_t massp=AliPID::ParticleMass(AliPID::kProton);
  Double_t massk=AliPID::ParticleMass(AliPID::kKaon);
  Double_t bethep=Bethe(mom,massp,isSA);
  Double_t bethek=Bethe(mom,massk,isSA);
  if(signal>(0.5*(bethep+bethek))) return AliPID::kProton;
  Double_t masspi=AliPID::ParticleMass(AliPID::kPion);
  Double_t bethepi=Bethe(mom,masspi,isSA);
  if(signal>(0.5*(bethepi+bethek))) return AliPID::kKaon;
  return AliPID::kPion;

}
