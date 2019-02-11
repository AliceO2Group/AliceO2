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


///////////////////////////////////////////////////////////////////////////
//                       Detector PID                                    //
//                                                                       //
//                                                                       //
/*

This class is supposed to store the detector pid values for all detectors
  and all particle species.
It is meant to be used to buffer the PID values as a transient object in
  AliESDtrack and AliAODTrack, respectively.
The calculation filling and association to the track is done in
  AliAnalysisTaskPID response.
The idea of this object is to save computing time in an analysis train with
  many analyses where access to pid is done often



*/
//                                                                       //
///////////////////////////////////////////////////////////////////////////

#include "AliPIDValues.h"

#include "AliDetectorPID.h"

ClassImp(AliDetectorPID)


AliDetectorPID::AliDetectorPID() :
  TObject(),
  fArrNsigmas("AliPIDValues",AliPIDResponse::kNdetectors),
  fArrRawProbabilities("AliPIDValues",AliPIDResponse::kNdetectors)
{
  //
  // default constructor
  //
  
}

//_______________________________________________________________________
AliDetectorPID::AliDetectorPID(const AliDetectorPID &pid) :
  TObject(pid),
  fArrNsigmas(pid.fArrNsigmas),
  fArrRawProbabilities(pid.fArrRawProbabilities)
{
  //
  // copy constructor
  //
  
}

//_______________________________________________________________________
AliDetectorPID::~AliDetectorPID()
{
  //
  // destructor
  //
  fArrNsigmas.Delete();
  fArrRawProbabilities.Delete();
}

//_______________________________________________________________________
AliDetectorPID& AliDetectorPID::operator= (const AliDetectorPID &pid)
{
  //
  // assignment operator
  //
  
  if (this==&pid) return *this;

  TObject::operator=(pid);
  
  fArrNsigmas.Clear();
  fArrRawProbabilities.Clear();
  
  AliPIDValues *val=0x0;
  for (Int_t idet=0; idet<(Int_t)AliPIDResponse::kNdetectors; ++idet){
    val=static_cast<AliPIDValues*>(pid.fArrNsigmas.UncheckedAt(idet));
    if (val) new (fArrNsigmas[idet]) AliPIDValues(*val);

    val=static_cast<AliPIDValues*>(pid.fArrRawProbabilities.UncheckedAt(idet));
    if (val) new (fArrRawProbabilities[idet]) AliPIDValues(*val);
  }

  return *this;
}

//_______________________________________________________________________
void AliDetectorPID::SetRawProbability(AliPIDResponse::EDetector det, const Double_t prob[],
                                       Int_t nspecies, AliPIDResponse::EDetPidStatus status)
{
  //
  // set raw probabilities for nspecies for 'det'ector
  //
  
  AliPIDValues *val=static_cast<AliPIDValues*>(fArrRawProbabilities.UncheckedAt(det));
  if (!val)
    val=new (fArrRawProbabilities[(Int_t)det]) AliPIDValues;

  val->SetValues(prob,nspecies,status);
}

//_______________________________________________________________________
void AliDetectorPID::SetNumberOfSigmas(AliPIDResponse::EDetector det, const Double_t nsig[], Int_t nspecies,
                                       AliPIDResponse::EDetPidStatus status)
{
  //
  // set number of sigmas for nspecies for 'det'ector
  //
  
  AliPIDValues *val=static_cast<AliPIDValues*>(fArrNsigmas.UncheckedAt(det));
  if (!val)
    val=new (fArrNsigmas[(Int_t)det]) AliPIDValues;

  val->SetValues(nsig,nspecies);
  val->SetPIDStatus(status);
}

//_______________________________________________________________________
AliPIDResponse::EDetPidStatus AliDetectorPID::GetRawProbability(AliPIDResponse::EDetector det, Double_t prob[], Int_t nspecies) const
{
  //
  // get raw probabilities for nspecies for 'det'ector
  //
  
  AliPIDValues *val=static_cast<AliPIDValues*>(fArrRawProbabilities.UncheckedAt((Int_t)det));
  if (!val) {
    for (Int_t i=0; i<nspecies; ++i) prob[i]=1.; //TODO: Is '1' the correct values or better 1/nspecies
    return AliPIDResponse::kDetNoSignal;
  }

  return val->GetValues(prob,nspecies);
}

//_______________________________________________________________________
AliPIDResponse::EDetPidStatus AliDetectorPID::GetNumberOfSigmas(AliPIDResponse::EDetector det, Double_t nsig[], Int_t nspecies) const
{
  AliPIDValues *val=static_cast<AliPIDValues*>(fArrNsigmas.UncheckedAt((Int_t)det));
  if (!val) {
    for (Int_t i=0; i<nspecies; ++i) nsig[i]=-999.;
    return AliPIDResponse::kDetNoSignal;
  }
  
  return val->GetValues(nsig,nspecies);
}

//_______________________________________________________________________
Double_t AliDetectorPID::GetRawProbability(AliPIDResponse::EDetector det, AliPID::EParticleType type) const
{
  //
  // get 'det'ector raw probability for particle 'type'
  //
  
  AliPIDValues *val=static_cast<AliPIDValues*>(fArrRawProbabilities.UncheckedAt((Int_t)det));
  if (!val) {
    return 0.; //TODO: Is '0' the correct value?
  }
  
  return val->GetValue(type);
}

//_______________________________________________________________________
Double_t AliDetectorPID::GetNumberOfSigmas(AliPIDResponse::EDetector det, AliPID::EParticleType type) const
{
  //
  // get 'det'ector number of sigmas for particle 'type'
  //
  AliPIDValues *val=static_cast<AliPIDValues*>(fArrNsigmas.UncheckedAt((Int_t)det));
  if (!val) {
    return -999.; //TODO: Is '-999.' the correct value?
  }
  
  return val->GetValue(type);
}

//_______________________________________________________________________
AliPIDResponse::EDetPidStatus AliDetectorPID::GetRawProbability(AliPIDResponse::EDetector det, AliPID::EParticleType type, Double_t &prob) const
{
  //
  // get 'det'ector raw probability for particle 'type'
  //
  
  AliPIDValues *val=static_cast<AliPIDValues*>(fArrRawProbabilities.UncheckedAt((Int_t)det));
  if (!val) {
    prob=0.;
    return AliPIDResponse::kDetNoSignal; 
  }
  
  prob=val->GetValue(type);
  return val->GetPIDStatus();
}

//_______________________________________________________________________
AliPIDResponse::EDetPidStatus AliDetectorPID::GetNumberOfSigmas(AliPIDResponse::EDetector det, AliPID::EParticleType type, Double_t &nsig) const
{
  //
  // get 'det'ector number of sigmas for particle 'type'
  //
  AliPIDValues *val=static_cast<AliPIDValues*>(fArrNsigmas.UncheckedAt((Int_t)det));
  if (!val) {
    nsig=-999.;
    return AliPIDResponse::kDetNoSignal; 
  }
  
  nsig=val->GetValue(type);
  return val->GetPIDStatus();
}


//_______________________________________________________________________
AliPIDResponse::EDetPidStatus AliDetectorPID::GetPIDStatus(AliPIDResponse::EDetector det) const
{
  //
  // return the detector PID status
  //
  
  AliPIDValues *val=static_cast<AliPIDValues*>(fArrRawProbabilities.UncheckedAt((Int_t)det));
  if (!val) val=static_cast<AliPIDValues*>(fArrNsigmas.UncheckedAt((Int_t)det));
  if (val) return val->GetPIDStatus();

  return AliPIDResponse::kDetNoSignal;
}

