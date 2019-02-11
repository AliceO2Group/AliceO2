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

//====================================================================================================================================================
//
//      ESD description of an ALICE muon forward track, combining the information of the Muon Spectrometer and the Muon Forward Tracker
//
//      Contact author: antonio.uras@cern.ch
//
//====================================================================================================================================================

#include "AliESDMuonGlobalTrack.h"
#include "AliESDEvent.h"

#include "TClonesArray.h"
#include "TLorentzVector.h"
#include "TMath.h"
#include "TDatabasePDG.h"

ClassImp(AliESDMuonGlobalTrack)

//====================================================================================================================================================

AliESDMuonGlobalTrack::AliESDMuonGlobalTrack():
  AliVParticle(),
  fCharge(0),
  fMatchTrigger(0),
  fNMFTClusters(0),
  fNWrongMFTClustersMC(-1),
  fMFTClusterPattern(0),
  fPx(0), 
  fPy(0), 
  fPz(0), 
  fPt(0), 
  fP(0), 
  fEta(0), 
  fRapidity(0),
  fFirstTrackingPointX(0),
  fFirstTrackingPointY(0),
  fFirstTrackingPointZ(0),
  fXAtVertex(0),
  fYAtVertex(0),
  fRAtAbsorberEnd(0),
  fCovariances(0),
  fChi2OverNdf(0),
  fChi2MatchTrigger(0),
  fLabel(-1),
  fMuonClusterMap(0),
  fHitsPatternInTrigCh(0),
  fHitsPatternInTrigChTrk(0),
  fLoCircuit(0),
  fIsConnected(kFALSE),
  fESDEvent(0)
{

  //  Default constructor

  fProdVertexXYZ[0]=0;
  fProdVertexXYZ[1]=0;
  fProdVertexXYZ[2]=0;

}

//====================================================================================================================================================

AliESDMuonGlobalTrack::AliESDMuonGlobalTrack(Double_t px, Double_t py, Double_t pz):
  AliVParticle(),
  fCharge(0),
  fMatchTrigger(0),
  fNMFTClusters(0),
  fNWrongMFTClustersMC(-1),
  fMFTClusterPattern(0),
  fPx(0), 
  fPy(0), 
  fPz(0), 
  fPt(0), 
  fP(0), 
  fEta(0), 
  fRapidity(0),
  fFirstTrackingPointX(0),
  fFirstTrackingPointY(0),
  fFirstTrackingPointZ(0),
  fXAtVertex(0),
  fYAtVertex(0),
  fRAtAbsorberEnd(0),
  fCovariances(0),
  fChi2OverNdf(0),
  fChi2MatchTrigger(0),
  fLabel(-1),
  fMuonClusterMap(0),
  fHitsPatternInTrigCh(0),
  fHitsPatternInTrigChTrk(0),
  fLoCircuit(0),
  fIsConnected(kFALSE),
  fESDEvent(0)
{

  //  Constructor with kinematics

  SetPxPyPz(px, py, pz);

  fProdVertexXYZ[0]=0;
  fProdVertexXYZ[1]=0;
  fProdVertexXYZ[2]=0;

}

//====================================================================================================================================================

AliESDMuonGlobalTrack::AliESDMuonGlobalTrack(const AliESDMuonGlobalTrack& muonTrack):
  AliVParticle(muonTrack),
  fCharge(muonTrack.fCharge),
  fMatchTrigger(muonTrack.fMatchTrigger),
  fNMFTClusters(muonTrack.fNMFTClusters),
  fNWrongMFTClustersMC(muonTrack.fNWrongMFTClustersMC),
  fMFTClusterPattern(muonTrack.fMFTClusterPattern),
  fPx(muonTrack.fPx), 
  fPy(muonTrack.fPy), 
  fPz(muonTrack.fPz), 
  fPt(muonTrack.fPt), 
  fP(muonTrack.fP), 
  fEta(muonTrack.fEta), 
  fRapidity(muonTrack.fRapidity),
  fFirstTrackingPointX(muonTrack.fFirstTrackingPointX),
  fFirstTrackingPointY(muonTrack.fFirstTrackingPointY),
  fFirstTrackingPointZ(muonTrack.fFirstTrackingPointZ),
  fXAtVertex(muonTrack.fXAtVertex),
  fYAtVertex(muonTrack.fYAtVertex),
  fRAtAbsorberEnd(muonTrack.fRAtAbsorberEnd),
  fCovariances(0),
  fChi2OverNdf(muonTrack.fChi2OverNdf),
  fChi2MatchTrigger(muonTrack.fChi2MatchTrigger),
  fLabel(muonTrack.fLabel),
  fMuonClusterMap(muonTrack.fMuonClusterMap),
  fHitsPatternInTrigCh(muonTrack.fHitsPatternInTrigCh),
  fHitsPatternInTrigChTrk(muonTrack.fHitsPatternInTrigChTrk),
  fLoCircuit(muonTrack.fLoCircuit),
  fIsConnected(muonTrack.fIsConnected),
  fESDEvent(muonTrack.fESDEvent)
{

  // Copy constructor
  
  fProdVertexXYZ[0]=muonTrack.fProdVertexXYZ[0];
  fProdVertexXYZ[1]=muonTrack.fProdVertexXYZ[1];
  fProdVertexXYZ[2]=muonTrack.fProdVertexXYZ[2];

  if (muonTrack.fCovariances) fCovariances = new TMatrixD(*(muonTrack.fCovariances));

}

//====================================================================================================================================================

AliESDMuonGlobalTrack& AliESDMuonGlobalTrack::operator=(const AliESDMuonGlobalTrack& muonTrack) {

  // Assignment operator

  if (this == &muonTrack) return *this;

  // Base class assignement
  AliVParticle::operator=(muonTrack);

  fCharge                 = muonTrack.fCharge;
  fMatchTrigger           = muonTrack.fMatchTrigger;
  fNMFTClusters           = muonTrack.fNMFTClusters;
  fNWrongMFTClustersMC    = muonTrack.fNWrongMFTClustersMC;
  fMFTClusterPattern      = muonTrack.fMFTClusterPattern;
  fPx                     = muonTrack.fPx; 
  fPy                     = muonTrack.fPy; 
  fPz                     = muonTrack.fPz; 
  fPt                     = muonTrack.fPt; 
  fP                      = muonTrack.fP;
  fEta                    = muonTrack.fEta;
  fRapidity               = muonTrack.fRapidity;
  fFirstTrackingPointX    = muonTrack.fFirstTrackingPointX;
  fFirstTrackingPointY    = muonTrack.fFirstTrackingPointY;
  fFirstTrackingPointZ    = muonTrack.fFirstTrackingPointZ;
  fXAtVertex              = muonTrack.fXAtVertex;
  fYAtVertex              = muonTrack.fYAtVertex;
  fRAtAbsorberEnd         = muonTrack.fRAtAbsorberEnd;
  fChi2OverNdf            = muonTrack.fChi2OverNdf;
  fChi2MatchTrigger       = muonTrack.fChi2MatchTrigger;
  fLabel                  = muonTrack.fLabel;
  fMuonClusterMap         = muonTrack.fMuonClusterMap;
  fHitsPatternInTrigCh    = muonTrack.fHitsPatternInTrigCh;
  fHitsPatternInTrigChTrk = muonTrack.fHitsPatternInTrigChTrk;
  fLoCircuit              = muonTrack.fLoCircuit;
  fIsConnected            = muonTrack.fIsConnected;
  fESDEvent               = muonTrack.fESDEvent;

  fProdVertexXYZ[0]=muonTrack.fProdVertexXYZ[0];
  fProdVertexXYZ[1]=muonTrack.fProdVertexXYZ[1];
  fProdVertexXYZ[2]=muonTrack.fProdVertexXYZ[2];

  if (muonTrack.fCovariances) {
    if (fCovariances) *fCovariances = *(muonTrack.fCovariances);
    else fCovariances = new TMatrixD(*(muonTrack.fCovariances));
  } 
  else {
    delete fCovariances;
    fCovariances = 0x0;
  }
  
  return *this;

}

//====================================================================================================================================================

void AliESDMuonGlobalTrack::Copy(TObject &obj) const {
  
  // This overwrites the virtual TObject::Copy()
  // to allow run time copying without casting
  // in AliESDEvent

  if (this==&obj) return;
  AliESDMuonGlobalTrack *robj = dynamic_cast<AliESDMuonGlobalTrack*>(&obj);
  if (!robj) return; // not an AliESDMuonGlobalTrack
  *robj = *this;

}

//====================================================================================================================================================

void AliESDMuonGlobalTrack::SetPxPyPz(Double_t px, Double_t py, Double_t pz) {

  Double_t mMu = TDatabasePDG::Instance()->GetParticle("mu-")->Mass();
  Double_t eMu = TMath::Sqrt(mMu*mMu + px*px + py*py + pz*pz);

  TLorentzVector kinem(px, py, pz, eMu);

  fPx       =  kinem.Px();
  fPy       =  kinem.Py();
  fPz       =  kinem.Pz();
  fP        =  kinem.P();
  fPt       =  kinem.Pt();
  fEta      =  kinem.Eta();
  fRapidity =  kinem.Rapidity(); 

}

//====================================================================================================================================================

const TMatrixD& AliESDMuonGlobalTrack::GetCovariances() const {

  // Return the covariance matrix (create it before if needed)

  if (!fCovariances) {
    fCovariances = new TMatrixD(5,5);
    fCovariances->Zero();
  }
  return *fCovariances;

}

//====================================================================================================================================================

void AliESDMuonGlobalTrack::SetCovariances(const TMatrixD& covariances) {

  // Set the covariance matrix

  if (fCovariances) *fCovariances = covariances;
  else fCovariances = new TMatrixD(covariances);

}

//====================================================================================================================================================
