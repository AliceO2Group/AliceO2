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
//                Implementation of the AliKalmanTrack class
//   that is the base for AliTPCtrack, AliITStrackV2 and AliTRDtrack
//        Origin: Iouri Belikov, CERN, Jouri.Belikov@cern.ch
//-------------------------------------------------------------------------
#include <TGeoManager.h>

#include "AliKalmanTrack.h"
#include "AliMiscConstants.h"

ClassImp(AliKalmanTrack)

//_______________________________________________________________________
  AliKalmanTrack::AliKalmanTrack():AliExternalTrackParam(),
  fFakeRatio(0),
  fChi2(0),
  fMass(AliPID::ParticleMass(AliPID::kPion)),
  fLab(-gkDummyLabel),
  fN(0),
  fStartTimeIntegral(kFALSE),
  fIntegratedLength(0)
{
  //
  // Default constructor
  //

  for(Int_t i=0; i<AliPID::kSPECIESC; i++) fIntegratedTime[i] = 0;
}

AliKalmanTrack::AliKalmanTrack(const AliKalmanTrack &t):
  AliExternalTrackParam(t),
  fFakeRatio(t.fFakeRatio),
  fChi2(t.fChi2),
  fMass(t.fMass),
  fLab(t.fLab),
  fN(t.fN),
  fStartTimeIntegral(t.fStartTimeIntegral),
  fIntegratedLength(t.fIntegratedLength)
{
  //
  // Copy constructor
  //
  
  for (Int_t i=0; i<AliPID::kSPECIESC; i++)
      fIntegratedTime[i] = t.fIntegratedTime[i];
}

AliKalmanTrack& AliKalmanTrack::operator=(const AliKalmanTrack&o){
  if(this!=&o){
    AliExternalTrackParam::operator=(o);
    fLab = o.fLab;
    fFakeRatio = o.fFakeRatio;
    fChi2 = o.fChi2;
    fMass = o.fMass;
    fN = o.fN;
    fStartTimeIntegral = o.fStartTimeIntegral;
    for(Int_t i = 0;i<AliPID::kSPECIESC;++i)fIntegratedTime[i] = o.fIntegratedTime[i];
    fIntegratedLength = o.fIntegratedLength;
  }
  return *this;
}

//_______________________________________________________________________
void AliKalmanTrack::StartTimeIntegral() 
{
  // Sylwester Radomski, GSI
  // S.Radomski@gsi.de
  //
  // Start time integration
  // To be called at Vertex by ITS tracker
  //
  
  //if (fStartTimeIntegral) 
  //  AliWarning("Reseting Recorded Time.");

  fStartTimeIntegral = kTRUE;
  for(Int_t i=0; i<AliPID::kSPECIESC; i++) fIntegratedTime[i] = 0;  
  fIntegratedLength = 0;
}

//_______________________________________________________________________
void AliKalmanTrack:: AddTimeStep(Double_t length) 
{
  // 
  // Add step to integrated time
  // this method should be called by a sublasses at the end
  // of the PropagateTo function or by a tracker
  // each time step is made.
  //
  // If integration not started function does nothing
  //
  // Formula
  // dt = dl * sqrt(p^2 + m^2) / p
  // p = pT * (1 + tg^2 (lambda) )
  //
  // pt = 1/external parameter [4]
  // tg lambda = external parameter [3]
  //
  //
  // Sylwester Radomski, GSI
  // S.Radomski@gsi.de
  // 
  
  static const Double_t kcc = 2.99792458e-2;

  if (!fStartTimeIntegral) return;
  
  fIntegratedLength += length;

  Double_t xr, param[5];
  
  GetExternalParameters(xr, param);
  double tgl = param[3];

  Double_t p2inv = param[4]*param[4]/(1+tgl*tgl);

  //  if (length > 100) return;

  for (Int_t i=0; i<AliPID::kSPECIESC; i++) {
    
    Double_t massz = AliPID::ParticleMassZ(i);
    Double_t correction = TMath::Sqrt( 1. + massz*massz*p2inv ); // 1/beta
    Double_t time = length * correction / kcc;

    fIntegratedTime[i] += time;
  }
}

//_______________________________________________________________________
Double_t AliKalmanTrack::GetIntegratedTime(Int_t pdg) const 
{
  // Sylwester Radomski, GSI
  // S.Radomski@gsi.de
  //
  // Return integrated time hypothesis for a given particle
  // type assumption.
  //
  // Input parameter:
  // pdg - Pdg code of a particle type
  //


  if (!fStartTimeIntegral) {
    AliWarning("Time integration not started");
    return 0.;
  }

  for (Int_t i=0; i<AliPID::kSPECIESC; i++)
    if (AliPID::ParticleCode(i) == TMath::Abs(pdg)) return fIntegratedTime[i];

  AliWarning(Form("Particle type [%d] not found", pdg));
  return 0;
}

void AliKalmanTrack::GetIntegratedTimes(Double_t *times, Int_t nspec) const {
  for (Int_t i=nspec; i--;) times[i]=fIntegratedTime[i];
}

void AliKalmanTrack::SetIntegratedTimes(const Double_t *times) {
  for (Int_t i=AliPID::kSPECIESC; i--;) fIntegratedTime[i]=times[i];
}

