#ifndef AliESDMuonGlobalTrack_H
#define AliESDMuonGlobalTrack_H

/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

//====================================================================================================================================================
//
//      ESD description of an ALICE muon forward track, combining the information of the Muon Spectrometer and the Muon Forward Tracker
//
//      Contact author: antonio.uras@cern.ch
//
//====================================================================================================================================================

#include "TMath.h"
#include "TMatrixD.h"
#include "TDatabasePDG.h"
#include "TArrayI.h"
#include "TLorentzVector.h"
#include "AliESDVertex.h"
#include "TRef.h"

#include "AliVParticle.h"

class AliESDEvent;
class TClonesArray;

//====================================================================================================================================================

class AliESDMuonGlobalTrack : public AliVParticle {

public:

  AliESDMuonGlobalTrack();
  AliESDMuonGlobalTrack(Double_t px, Double_t py, Double_t pz);
  virtual ~AliESDMuonGlobalTrack() {;}
  AliESDMuonGlobalTrack(const AliESDMuonGlobalTrack& esdTrack);
  AliESDMuonGlobalTrack& operator=(const AliESDMuonGlobalTrack& esdTrack);
  virtual void Copy(TObject &obj) const;

  void  SetCharge(Int_t charge) { fCharge = charge; } 
  Short_t GetCharge() const { return fCharge; }

  /* Double_t GetOffset(Double_t x, Double_t y, Double_t z); */
  /* Double_t GetOffsetX(Double_t x, Double_t z); */
  /* Double_t GetOffsetY(Double_t y, Double_t z); */

  // Set and Get methods for kinematics at primary vertex
  void SetPxPyPz(Double_t px, Double_t py, Double_t pz);

  // Get and Set methods for global tracking info
  Double_t GetChi2OverNdf() const { return fChi2OverNdf; }            // chi2/ndf
  void     SetChi2OverNdf(Double_t chi2) { fChi2OverNdf = chi2; }     // chi2/ndf

  Double_t GetChi2MatchTrigger() const { return fChi2MatchTrigger; }
  void     SetChi2MatchTrigger(Double_t chi2MatchTrigger) { fChi2MatchTrigger = chi2MatchTrigger; }

  // Get and Set methods for various info copied and pasted from the MUON track
  UShort_t GetHitsPatternInTrigCh() const {return fHitsPatternInTrigCh;}
  void     SetHitsPatternInTrigCh(UShort_t hitsPatternInTrigCh) {fHitsPatternInTrigCh = hitsPatternInTrigCh;}
  UInt_t   GetHitsPatternInTrigChTrk() const {return fHitsPatternInTrigChTrk;}
  void     SetHitsPatternInTrigChTrk(UInt_t hitsPatternInTrigChTrk) {fHitsPatternInTrigChTrk = hitsPatternInTrigChTrk;}
  UInt_t   GetMuonClusterMap() const {return fMuonClusterMap;}
  void     SetMuonClusterMap(UInt_t muonClusterMap) {fMuonClusterMap = muonClusterMap;}
  Int_t    GetLoCircuit() const { return fLoCircuit; }
  void     SetLoCircuit(Int_t loCircuit) { fLoCircuit = loCircuit; }
  Bool_t   IsConnected() const { return fIsConnected; }
  void     Connected(Bool_t flag) { fIsConnected = flag; }

  // Get and Set methods for trigger matching
  void  SetMatchTrigger(Int_t matchTrigger) { fMatchTrigger = matchTrigger; }
  Int_t GetMatchTrigger() { return fMatchTrigger; }

  void SetNMFTClusters(Int_t nMFTClusters) { fNMFTClusters = nMFTClusters; }
  Short_t GetNMFTClusters() { return fNMFTClusters; }

  void SetNWrongMFTClustersMC(Int_t nWrongMFTClustersMC) { fNWrongMFTClustersMC = nWrongMFTClustersMC; }
  Short_t GetNWrongMFTClustersMC() { return fNWrongMFTClustersMC; }

  void SetMFTClusterPattern(ULong_t mftClusterPattern) { fMFTClusterPattern = mftClusterPattern; }
  ULong_t GetMFTClusterPattern() { return fMFTClusterPattern; }

  // Kinematics
  Double_t Pt()       const { return fPt;  }
  Double_t Eta()      const { return fEta; }
  Double_t Rapidity() const { return fRapidity; }
  Double_t Px()       const { return fPx; }
  Double_t Py()       const { return fPy; }
  Double_t Pz()       const { return fPz; }
  Double_t P()        const { return fP;  }

  Bool_t   PxPyPz(Double_t p[3]) const { p[0] = Px(); p[1] = Py(); p[2] = Pz(); return kTRUE; }

  void SetFirstTrackingPoint(Double_t x, Double_t y, Double_t z) {fFirstTrackingPointX = x; fFirstTrackingPointY = y; fFirstTrackingPointZ = z; }
  void GetFirstTrackingPoint(Double_t x[3]) { x[0] = fFirstTrackingPointX; x[1] = fFirstTrackingPointY; x[2] = fFirstTrackingPointZ; }

  void SetXYAtVertex(Double_t x, Double_t y) { fXAtVertex = x; fYAtVertex = y; }
  void GetXYAtVertex(Double_t x[2]) { x[0] = fXAtVertex; x[1] = fYAtVertex; }

  Double_t GetRAtAbsorberEnd() { return fRAtAbsorberEnd; }
  void SetRAtAbsorberEnd(Double_t r) { fRAtAbsorberEnd = r; }

  // Additional methods to comply with AliVParticle
  Double_t Xv() const {return -999.;} // put reasonable values here
  Double_t Yv() const {return -999.;} //
  Double_t Zv() const {return -999.;} //
  Bool_t   XvYvZv(Double_t x[3]) const { x[0] = Xv(); x[1] = Yv(); x[2] = Zv(); return kTRUE; }  
  Double_t OneOverPt() const { return (Pt() != 0.) ? 1./Pt() : FLT_MAX; }
  Double_t Phi() const { return TMath::Pi()+TMath::ATan2(-Py(), -Px()); }
  Double_t Theta() const { return TMath::ATan2(Pt(), Pz()); }
  Double_t E() const { return TMath::Sqrt(M()*M() + P()*P()); }
  Double_t M() const { return TDatabasePDG::Instance()->GetParticle("mu-")->Mass(); }
  Double_t Y() const { return Rapidity(); }
  Short_t  Charge() const { return fCharge; }

  // Return kTRUE if the track contain tracker data
  Bool_t ContainTrackerData() const {return (fMuonClusterMap>0) ? kTRUE : kFALSE;}

  // Dummy
  const Double_t *PID() const { return (Double_t*)0x0; }
  Int_t PdgCode() const { return 0; }
  
  // Set the corresponding MC track number
  void  SetLabel(Int_t label) { fLabel = label; }
  // Return the corresponding MC track number
  Int_t GetLabel() const { return fLabel; }

  void SetProdVertexXYZ(Double_t x, Double_t y, Double_t z) { fProdVertexXYZ[0]=x; fProdVertexXYZ[1]=y; fProdVertexXYZ[2]=z; }
  void GetProdVertexXYZ(Double_t *vertex) { vertex[0]=fProdVertexXYZ[0]; vertex[1]=fProdVertexXYZ[1]; vertex[2]=fProdVertexXYZ[2]; }

  const TMatrixD& GetCovariances() const;
  void            SetCovariances(const TMatrixD& covariances);

  AliESDEvent* GetESDEvent() const { return fESDEvent; }
  void         SetESDEvent(AliESDEvent* evt) { fESDEvent = evt; }  
  
protected:

  Short_t fCharge, fMatchTrigger, fNMFTClusters, fNWrongMFTClustersMC;
  ULong_t fMFTClusterPattern;  // Tells us which MFT clusters are contained in the track, and which one is a good one (if MC)

  // kinematics at vertex
  Double_t fPx, fPy, fPz, fPt, fP, fEta, fRapidity;

  // coordinates of the first tracking point
  Double_t fFirstTrackingPointX, fFirstTrackingPointY, fFirstTrackingPointZ;

  // transverse coordinates at DCA to the primary vertex (offset)
  Double_t fXAtVertex, fYAtVertex;

  Double_t fRAtAbsorberEnd;

  mutable TMatrixD *fCovariances; // Covariance matrix of track parameters (see AliMUONTrackParam)

  // global tracking info
  Double_t fChi2OverNdf;            //  chi2/ndf in the MUON+MFT track fit
  Double_t fChi2MatchTrigger;       //  chi2 of trigger/track matching

  Int_t fLabel;                     //  point to the corresponding MC track

  UInt_t   fMuonClusterMap;         // Map of clusters in MUON tracking chambers
  UShort_t fHitsPatternInTrigCh;    // Word containing info on the hits left in trigger chambers
  UInt_t   fHitsPatternInTrigChTrk; // Trigger hit map from tracker track extrapolation
  Int_t    fLoCircuit;
  Bool_t   fIsConnected;
  
  Double_t fProdVertexXYZ[3];       // vertex of origin

  AliESDEvent *fESDEvent;           //! Pointer back to event to which the track belongs
  
  ClassDef(AliESDMuonGlobalTrack,4) // MUON+MFT ESD track class 

};

//====================================================================================================================================================

#endif 
