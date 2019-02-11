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

/* $Id: AliESD.cxx 54579 2012-02-14 12:22:34Z shahoian $ */

//-----------------------------------------------------------------
//           Implementation of the ESD class
//   This is the class to deal with during the phisical analysis of data
//   This class is generated directly by the reconstruction methods
//      Origin: Iouri Belikov, CERN, Jouri.Belikov@cern.ch
//-----------------------------------------------------------------

#include "AliESD.h"
#include "AliESDfriend.h"

ClassImp(AliESD)

//______________________________________________________________________________
AliESD::AliESD():
  fEventNumberInFile(0),
  fBunchCrossNumber(0),
  fOrbitNumber(0),
  fPeriodNumber(0),
  fRunNumber(0),
  fTimeStamp(0),
  fEventType(0),
  fTriggerMask(0),
  fTriggerMaskNext50(0),
  fTriggerCluster(0),
  fRecoVersion(0),
  fMagneticField(0),
  fZDCN1Energy(0),
  fZDCP1Energy(0),
  fZDCN2Energy(0),
  fZDCP2Energy(0),
  fZDCEMEnergy(0),
  fZDCParticipants(0),
  fT0zVertex(0),
  fSPDVertex(),
  fPrimaryVertex(),
  fSPDMult(),
  fT0clock(0),
  fT0timeStart(0),
  fT0trig(0),
  fTracks("AliESDtrack",15000),
  fHLTConfMapTracks("AliESDHLTtrack",25000),
  fHLTHoughTracks("AliESDHLTtrack",15000),
  fMuonTracks("AliESDMuonTrack",30),
  fPmdTracks("AliESDPmdTrack",3000),
  fTrdTracks("AliESDTrdTrack",300),
  fV0s("AliESDv0",200),  
  fCascades("AliESDcascade",20),
  fKinks("AliESDkink",4000),
  fCaloClusters("AliESDCaloCluster",10000),
  fEMCALClusters(0), 
  fFirstEMCALCluster(-1),
  fEMCALTriggerPosition(0x0),
  fEMCALTriggerAmplitudes(0x0),
  fPHOSClusters(0), 
  fFirstPHOSCluster(-1),
  fPHOSTriggerPosition(0x0),
  fPHOSTriggerAmplitudes(0x0),
  fESDFMD(0x0),
  fESDVZERO(0x0),
  fESDACORDE(0x0),
  fESDAD(0x0),
  fErrorLogs("AliRawDataErrorLog",5)
{
  // 
  // Standar constructor
  //

   for (Int_t i=0; i<3; i++) fT0TOF[i] = 0;
 for (Int_t i=0; i<24; i++) {
    fT0time[i] = 0;
    fT0amplitude[i] = 0;
  }
  for (Int_t i=0; i<2; i++) fDiamondXY[i]=0.;
  for (Int_t i=0; i<3; i++) fDiamondCovXY[i]=0.;
}

//______________________________________________________________________________
AliESD::AliESD(const AliESD& esd):
  TObject(esd),
  fEventNumberInFile(esd.fEventNumberInFile),
  fBunchCrossNumber(esd.fBunchCrossNumber),
  fOrbitNumber(esd.fOrbitNumber),
  fPeriodNumber(esd.fPeriodNumber),
  fRunNumber(esd.fRunNumber),
  fTimeStamp(esd.fTimeStamp),
  fEventType(esd.fEventType),
  fTriggerMask(esd.fTriggerMask),
  fTriggerMaskNext50(esd.fTriggerMaskNext50),
  fTriggerCluster(esd.fTriggerCluster),
  fRecoVersion(esd.fRecoVersion),
  fMagneticField(esd.fMagneticField),
  fZDCN1Energy(esd.fZDCN1Energy),
  fZDCP1Energy(esd.fZDCP1Energy),
  fZDCN2Energy(esd.fZDCN2Energy),
  fZDCP2Energy(esd.fZDCP2Energy),
  fZDCEMEnergy(esd.fZDCEMEnergy),
  fZDCParticipants(esd.fZDCParticipants),
  fT0zVertex(esd.fT0zVertex),
  fSPDVertex(esd.fSPDVertex),
  fPrimaryVertex(esd.fPrimaryVertex),
  fSPDMult(esd.fSPDMult),
  fT0clock(esd.fT0clock),
  fT0timeStart(esd.fT0timeStart),
  fT0trig(esd.fT0trig),
  fTracks(*((TClonesArray*)esd.fTracks.Clone())),
  fHLTConfMapTracks(*((TClonesArray*)esd.fHLTConfMapTracks.Clone())),
  fHLTHoughTracks(*((TClonesArray*)esd.fHLTHoughTracks.Clone())),
  fMuonTracks(*((TClonesArray*)esd.fMuonTracks.Clone())),
  fPmdTracks(*((TClonesArray*)esd.fPmdTracks.Clone())),
  fTrdTracks(*((TClonesArray*)esd.fTrdTracks.Clone())),
  fV0s(*((TClonesArray*)esd.fV0s.Clone())),  
  fCascades(*((TClonesArray*)esd.fCascades.Clone())),
  fKinks(*((TClonesArray*)esd.fKinks.Clone())),
  fCaloClusters(*((TClonesArray*)esd.fCaloClusters.Clone())),
  fEMCALClusters(esd.fEMCALClusters), 
  fFirstEMCALCluster(esd.fFirstEMCALCluster),
  fEMCALTriggerPosition(esd. fEMCALTriggerPosition),
  fEMCALTriggerAmplitudes(esd.fEMCALTriggerAmplitudes),
  fPHOSClusters(esd.fPHOSClusters), 
  fFirstPHOSCluster(esd.fFirstPHOSCluster),
  fPHOSTriggerPosition(esd.fPHOSTriggerPosition),
  fPHOSTriggerAmplitudes(esd.fPHOSTriggerAmplitudes),
  fESDFMD(esd.fESDFMD),
  fESDVZERO(esd.fESDVZERO),
  fESDACORDE(esd.fESDACORDE),
  fESDAD(esd.fESDAD),
  fErrorLogs(*((TClonesArray*)esd.fErrorLogs.Clone()))
{
  // 
  // copy constructor
  //
  for (Int_t i=0; i<3; i++)fT0TOF[i] = esd.fT0TOF[i];
  for (Int_t i=0; i<24; i++) {
    fT0time[i] = esd.fT0time[i];
    fT0amplitude[i] = esd.fT0amplitude[i];
  }
  for (Int_t i=0; i<2; i++) fDiamondXY[i]=esd.fDiamondXY[i];
  for (Int_t i=0; i<3; i++) fDiamondCovXY[i]=esd.fDiamondCovXY[i];
}

//______________________________________________________________________________
AliESD::~AliESD()
{
  //
  // Standard destructor
  //
  fTracks.Delete();
  fHLTConfMapTracks.Delete();
  fHLTHoughTracks.Delete();
  fMuonTracks.Delete();
  fPmdTracks.Delete();
  fTrdTracks.Delete();
  fV0s.Delete();
  fCascades.Delete();
  fKinks.Delete();
  fCaloClusters.Delete();
  delete fESDFMD;
  delete fESDVZERO;
  delete fEMCALTriggerPosition;
  delete fEMCALTriggerAmplitudes;
  delete fPHOSTriggerPosition;
  delete fPHOSTriggerAmplitudes;
  delete fESDACORDE;
  delete fESDAD;
  fErrorLogs.Delete();

}

//______________________________________________________________________________
void AliESD::Reset()
{
  // 
  // Reset the contents and delete the entries in TClonesArrays
  //

  fEventNumberInFile=0;
  fBunchCrossNumber=0;
  fOrbitNumber=0;
  fPeriodNumber=0;
  fRunNumber=0;
  fTimeStamp = 0;
  fEventType = 0;
  fTriggerMask=0;
  fTriggerMaskNext50=0;
  fTriggerCluster=0;
  fRecoVersion=0;
  fMagneticField=0;
  fZDCN1Energy=0;
  fZDCP1Energy=0;
  fZDCN2Energy=0;
  fZDCP2Energy=0;
  fZDCEMEnergy=0;
  fZDCParticipants=0;
  fT0zVertex=0;
  
  for (Int_t i=0; i<2; i++) fDiamondXY[i]=0.;
  for (Int_t i=0; i<3; i++) fDiamondCovXY[i]=0.;

  for (Int_t i=0; i<24; i++) {
    fT0time[i] = 0;
    fT0amplitude[i] = 0;
  }
  fT0timeStart = 0;
  fT0clock = 0;
  for (Int_t i=0; i<3; i++) fT0TOF[i] = 0;
//
  fSPDMult.~AliMultiplicity();
  new (&fSPDMult) AliMultiplicity();
  fSPDVertex.~AliESDVertex();
  new (&fSPDVertex) AliESDVertex();
  fPrimaryVertex.~AliESDVertex();
  new (&fPrimaryVertex) AliESDVertex();
//
  fTracks.Delete();
  fHLTConfMapTracks.Delete();
  fHLTHoughTracks.Delete();
  fMuonTracks.Delete();
  fPmdTracks.Delete();
  fTrdTracks.Delete();
  fV0s.Delete();
  fCascades.Delete();
  fKinks.Delete();
  fCaloClusters.Delete();
//
  fEMCALClusters=0; 
  fFirstEMCALCluster=-1; 
  fPHOSClusters=0; 
  fFirstPHOSCluster=-1; 
//
  if (fEMCALTriggerPosition)   fEMCALTriggerPosition  ->Reset();
  if (fEMCALTriggerAmplitudes) fEMCALTriggerAmplitudes->Reset();
  if (fPHOSTriggerPosition)    fPHOSTriggerPosition   ->Reset();
  if (fPHOSTriggerAmplitudes)  fPHOSTriggerAmplitudes ->Reset();
//
  if (fESDFMD) fESDFMD->Clear();
//
  if (fESDVZERO){
      fESDVZERO->~AliESDVZERO();
      new (fESDVZERO) AliESDVZERO();
  } 
//
  if (fESDACORDE){
      fESDACORDE->~AliESDACORDE();
      new (fESDACORDE) AliESDACORDE();	
  }	

 if (fESDAD){
      fESDAD->~AliESDAD();
      new (fESDAD) AliESDAD();	
  }	

//
//
  fErrorLogs.Delete();
}


Bool_t  AliESD::RemoveKink(Int_t rm) {
  // ---------------------------------------------------------
  // Remove a kink candidate and references to it from ESD,
  // if this candidate does not come from a reconstructed decay
  // Not yet implemented...
  // ---------------------------------------------------------
  Int_t last=GetNumberOfKinks()-1;
  if ((rm<0)||(rm>last)) return kFALSE;

  return kTRUE;
}

Bool_t  AliESD::RemoveV0(Int_t rm) {
  // ---------------------------------------------------------
  // Remove a V0 candidate and references to it from ESD,
  // if this candidate does not come from a reconstructed decay
  // ---------------------------------------------------------
  Int_t last=GetNumberOfV0s()-1;
  if ((rm<0)||(rm>last)) return kFALSE;

  AliESDv0 *v0=GetV0(rm);
  Int_t idxP=v0->GetPindex(), idxN=v0->GetNindex();

  v0=GetV0(last);
  Int_t lastIdxP=v0->GetPindex(), lastIdxN=v0->GetNindex();

  Int_t used=0;

  // Check if this V0 comes from a reconstructed decay
  Int_t ncs=GetNumberOfCascades();
  for (Int_t n=0; n<ncs; n++) {
    AliESDcascade *cs=GetCascade(n);

    Int_t csIdxP=cs->GetPindex();
    Int_t csIdxN=cs->GetNindex();

    if (idxP==csIdxP)
       if (idxN==csIdxN) return kFALSE;

    if (csIdxP==lastIdxP)
       if (csIdxN==lastIdxN) used++;
  }

  //Replace the removed V0 with the last V0 
  TClonesArray &a=fV0s;
  delete a.RemoveAt(rm);

  if (rm==last) return kTRUE;

  //v0 is pointing to the last V0 candidate... 
  new (a[rm]) AliESDv0(*v0);
  delete a.RemoveAt(last);

  if (!used) return kTRUE;
  

  // Remap the indices of the daughters of reconstructed decays
  for (Int_t n=0; n<ncs; n++) {
    AliESDcascade *cs=GetCascade(n);


    Int_t csIdxP=cs->GetPindex();
    Int_t csIdxN=cs->GetNindex();

    if (csIdxP==lastIdxP)
      if (csIdxN==lastIdxN) {
         cs->AliESDv0::SetIndex(1,idxP);
         cs->AliESDv0::SetIndex(0,idxN);
         used--;
         if (!used) return kTRUE;
      }
  }

  return kTRUE;
}

Bool_t  AliESD::RemoveTrack(Int_t rm) {
  // ---------------------------------------------------------
  // Remove a track and references to it from ESD,
  // if this track does not come from a reconstructed decay
  // ---------------------------------------------------------
  Int_t last=GetNumberOfTracks()-1;
  if ((rm<0)||(rm>last)) return kFALSE;

  Int_t used=0;

  // Check if this track comes from the reconstructed primary vertex
  if (fPrimaryVertex.GetStatus()) {
     UShort_t *primIdx=fPrimaryVertex.GetIndices();
     Int_t n=fPrimaryVertex.GetNIndices();
     while (n--) {
       Int_t idx=Int_t(primIdx[n]);
       if (rm==idx) return kFALSE;
       if (idx==last) used++; 
     }
  }
  
  // Check if this track comes from a reconstructed decay
  Int_t nv0=GetNumberOfV0s();
  for (Int_t n=0; n<nv0; n++) {
    AliESDv0 *v0=GetV0(n);

    Int_t idx=v0->GetNindex();
    if (rm==idx) return kFALSE;
    if (idx==last) used++;

    idx=v0->GetPindex();
    if (rm==idx) return kFALSE;
    if (idx==last) used++;
  }

  Int_t ncs=GetNumberOfCascades();
  for (Int_t n=0; n<ncs; n++) {
    AliESDcascade *cs=GetCascade(n);

    Int_t idx=cs->GetIndex();
    if (rm==idx) return kFALSE;
    if (idx==last) used++;
  }

  Int_t nkn=GetNumberOfKinks();
  for (Int_t n=0; n<nkn; n++) {
    AliESDkink *kn=GetKink(n);

    Int_t idx=kn->GetIndex(0);
    if (rm==idx) return kFALSE;
    if (idx==last) used++;

    idx=kn->GetIndex(1);
    if (rm==idx) return kFALSE;
    if (idx==last) used++;
  }


  //Replace the removed track with the last track 
  TClonesArray &a=fTracks;
  delete a.RemoveAt(rm);

  if (rm==last) return kTRUE;

  AliESDtrack *t=GetTrack(last);
  if (!t) {AliFatal(Form("NULL pointer for ESD track %d",last)); return kFALSE;}
  t->SetID(rm);
  new (a[rm]) AliESDtrack(*t);
  delete a.RemoveAt(last);


  if (!used) return kTRUE;
  

  // Remap the indices of the tracks used for the primary vertex reconstruction
  if (fPrimaryVertex.GetStatus()) {
     UShort_t *primIdx=fPrimaryVertex.GetIndices();
     Int_t n=fPrimaryVertex.GetNIndices();
     while (n--) {
       Int_t idx=Int_t(primIdx[n]);
       if (idx==last) {
          primIdx[n]=Short_t(rm); 
          used--;
          if (!used) return kTRUE;
       }
     }
  }
  
  // Remap the indices of the daughters of reconstructed decays
  for (Int_t n=0; n<nv0; n++) {
    AliESDv0 *v0=GetV0(n);
    if (v0->GetIndex(0)==last) {
       v0->SetIndex(0,rm);
       used--;
       if (!used) return kTRUE;
    }
    if (v0->GetIndex(1)==last) {
       v0->SetIndex(1,rm);
       used--;
       if (!used) return kTRUE;
    }
  }

  for (Int_t n=0; n<ncs; n++) {
    AliESDcascade *cs=GetCascade(n);
    if (cs->GetIndex()==last) {
       cs->SetIndex(rm);
       used--;
       if (!used) return kTRUE;
    }
  }

  for (Int_t n=0; n<nkn; n++) {
    AliESDkink *kn=GetKink(n);
    if (kn->GetIndex(0)==last) {
       kn->SetIndex(rm,0);
       used--;
       if (!used) return kTRUE;
    }
    if (kn->GetIndex(1)==last) {
       kn->SetIndex(rm,1);
       used--;
       if (!used) return kTRUE;
    }
  }

  return kTRUE;
}


Bool_t AliESD::Clean(Float_t *cleanPars) {
  //
  // Remove the data which are not needed for the physics analysis.
  //
  // 1) Cleaning the V0 candidates
  //    ---------------------------
  //    If the cosine of the V0 pointing angle "csp" and 
  //    the DCA between the daughter tracks "dca" does not satisfy 
  //    the conditions 
  //
  //     csp > cleanPars[1] + dca/cleanPars[0]*(1.- cleanPars[1])
  //
  //    an attempt to remove this V0 candidate from ESD is made.
  //
  //    The V0 candidate gets removed if it does not belong to any 
  //    recosntructed cascade decay
  //
  //    12.11.2007, optimal values: cleanPars[0]=0.5, cleanPars[1]=0.999
  //
  // 2) Cleaning the tracks
  //    ----------------------
  //    If track's transverse parameter is larger than cleanPars[2]
  //                       OR
  //    track's longitudinal parameter is larger than cleanPars[3]
  //    an attempt to remove this track from ESD is made.
  //
  //    The track gets removed if it does not come 
  //    from a reconstructed decay
  //
  Bool_t rc=kFALSE;

  Float_t dcaMax=cleanPars[0];
  Float_t cspMin=cleanPars[1];

  Int_t nV0s=GetNumberOfV0s();
  for (Int_t i=nV0s-1; i>=0; i--) {
    AliESDv0 *v0=GetV0(i);

    Float_t dca=v0->GetDcaV0Daughters();
    Float_t csp=v0->GetV0CosineOfPointingAngle();
    Float_t cspcut=cspMin + dca/dcaMax*(1.-cspMin);
    if (csp > cspcut) continue;

    if (RemoveV0(i)) rc=kTRUE;
  }


  Float_t dmax=cleanPars[2], zmax=cleanPars[3];

  const AliESDVertex *vertex=GetVertex();
  Bool_t vtxOK=vertex->GetStatus();
  
  Int_t nTracks=GetNumberOfTracks();
  for (Int_t i=nTracks-1; i>=0; i--) {
    AliESDtrack *track=GetTrack(i);
    if (!track) {AliFatal(Form("NULL pointer for ESD track %d",i)); return kFALSE;}
    Float_t xy,z; track->GetImpactParameters(xy,z);
    if ((TMath::Abs(xy) > dmax) || (vtxOK && (TMath::Abs(z) > zmax))) {
      if (RemoveTrack(i)) rc=kTRUE;
    }
  }

  return rc;
}

Int_t AliESD::AddV0(const AliESDv0 *v) {
  //
  // Add V0
  //
    Int_t idx=fV0s.GetEntriesFast();
    new(fV0s[idx]) AliESDv0(*v);
    return idx;
}  

//______________________________________________________________________________
void AliESD::Print(Option_t *) const 
{
  //
  // Print header information of the event
  //
  printf("ESD run information\n");
  printf("Event # in file %d Bunch crossing # %d Orbit # %d Period # %d Run # %d Trigger %lld %lld Magnetic field %f \n",
	 GetEventNumberInFile(),
	 GetBunchCrossNumber(),
	 GetOrbitNumber(),
	 GetPeriodNumber(),
	 GetRunNumber(),
	 GetTriggerMask(),
	 GetTriggerMaskNext50(),
	 GetMagneticField() );
    printf("Vertex: (%.4f +- %.4f, %.4f +- %.4f, %.4f +- %.4f) cm\n",
	   fPrimaryVertex.GetX(), fPrimaryVertex.GetXRes(),
	   fPrimaryVertex.GetY(), fPrimaryVertex.GetYRes(),
	   fPrimaryVertex.GetZ(), fPrimaryVertex.GetZRes());
    printf("Mean vertex in RUN: X=%.4f Y=%.4f cm\n",
	   GetDiamondX(),GetDiamondY());
    printf("SPD Multiplicity. Number of tracklets %d \n",
           fSPDMult.GetNumberOfTracklets());
  printf("Event from reconstruction version %d \n",fRecoVersion);
  printf("Number of tracks: \n");
  printf("                 charged   %d\n", GetNumberOfTracks());
  printf("                 hlt CF    %d\n", GetNumberOfHLTConfMapTracks());
  printf("                 hlt HT    %d\n", GetNumberOfHLTHoughTracks());
  printf("                 muon      %d\n", GetNumberOfMuonTracks());
  printf("                 pmd       %d\n", GetNumberOfPmdTracks());
  printf("                 trd       %d\n", GetNumberOfTrdTracks());
  printf("                 v0        %d\n", GetNumberOfV0s());
  printf("                 cascades  %d\n", GetNumberOfCascades());
  printf("                 kinks     %d\n", GetNumberOfKinks());
  printf("                 CaloClusters %d\n", GetNumberOfCaloClusters());
  printf("                 phos      %d\n", GetNumberOfPHOSClusters());
  printf("                 emcal     %d\n", GetNumberOfEMCALClusters());
  printf("                 FMD       %s\n", (fESDFMD ? "yes" : "no"));
  printf("                 VZERO     %s\n", (fESDVZERO ? "yes" : "no"));
}

void AliESD::SetESDfriend(const AliESDfriend *ev) {
  //
  // Attaches the complementary info to the ESD
  //
  if (!ev) return;

  Int_t ntrk=ev->GetNumberOfTracks();

  for (Int_t i=0; i<ntrk; i++) {
    const AliESDfriendTrack *f=ev->GetTrack(i);
    if (!f) {AliFatal(Form("NULL pointer for ESD track %d",i));}
    GetTrack(i)->SetFriendTrack(f);
  }
}

void AliESD::GetESDfriend(AliESDfriend *ev) const {
  //
  // Extracts the complementary info from the ESD
  //
  if (!ev) return;

  Int_t ntrk=GetNumberOfTracks();

  for (Int_t i=0; i<ntrk; i++) {
    AliESDtrack *t=GetTrack(i);
    if (!t) {AliFatal(Form("NULL pointer for ESD track %d",i));}
    const AliESDfriendTrack *f=t->GetFriendTrack();
    ev->AddTrack(f);

    t->ReleaseESDfriendTrack();// Not to have two copies of "friendTrack"

  }
}

void AliESD::SetDiamond(const AliESDVertex *vertex)
{
  //
  // Set the interaction diamond
  //  
    fDiamondXY[0]=vertex->GetX();
    fDiamondXY[1]=vertex->GetY();
    Double_t cov[6];
    vertex->GetCovMatrix(cov);
    fDiamondCovXY[0]=cov[0];
    fDiamondCovXY[1]=cov[1];
    fDiamondCovXY[2]=cov[2];
  }
