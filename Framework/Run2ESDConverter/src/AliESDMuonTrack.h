#ifndef ALIESDMUONTRACK_H
#define ALIESDMUONTRACK_H

/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
* See cxx source for full Copyright notice                               */

/* $Id$ */

/// \class AliESDMuonTrack
/// \brief Class to describe the MUON tracks in the Event Summary Data class
//  Author: G.Martinez


#include <TMath.h>
#include <TMatrixD.h>
#include <TDatabasePDG.h>
#include <TArrayI.h>

#include "AliVParticle.h"

class AliESDEvent;
class TClonesArray;
class TLorentzVector;

class AliESDMuonTrack : public AliVParticle {
public:
  AliESDMuonTrack(); //Constructor
  virtual ~AliESDMuonTrack(); // Destructor
  AliESDMuonTrack(const AliESDMuonTrack& esdm);
  AliESDMuonTrack& operator=(const AliESDMuonTrack& esdm);
  virtual void Copy(TObject &obj) const;

  virtual void Clear(Option_t* opt = "");
  
  void Reset();
  
  // Return kTRUE if the track contain tracker data
  Bool_t ContainTrackerData() const {return (fMuonClusterMap>0) ? kTRUE : kFALSE;}
  // Return kTRUE if the track contain trigger data
  Bool_t ContainTriggerData() const {return (LoCircuit()>0) ? kTRUE : kFALSE;}
  
  // Get and Set methods for data at vertex
  Double_t GetInverseBendingMomentum(void) const {return fInverseBendingMomentum;}
  void     SetInverseBendingMomentum(Double_t InverseBendingMomentum) 
		{fInverseBendingMomentum = InverseBendingMomentum;}
  Double_t GetThetaX(void) const {return fThetaX;}
  void     SetThetaX(Double_t ThetaX) {fThetaX = ThetaX;}
  Double_t GetThetaY(void) const {return fThetaY;}
  void     SetThetaY(Double_t ThetaY) {fThetaY = ThetaY;}
  Double_t GetZ(void) const {return fZ;}
  void     SetZ(Double_t Z) {fZ = Z;}
  Double_t GetBendingCoor(void) const {return fBendingCoor;}
  void     SetBendingCoor(Double_t BendingCoor) {fBendingCoor = BendingCoor;}
  Double_t GetNonBendingCoor(void) const {return fNonBendingCoor;}
  void     SetNonBendingCoor(Double_t NonBendingCoor) {fNonBendingCoor = NonBendingCoor;}
  
  // Get and Set methods for data at Distance of Closest Approach in the vertex plane
  Double_t GetInverseBendingMomentumAtDCA(void) const {return fInverseBendingMomentumAtDCA;}
  void     SetInverseBendingMomentumAtDCA(Double_t InverseBendingMomentum) 
		{fInverseBendingMomentumAtDCA = InverseBendingMomentum;}
  Double_t GetThetaXAtDCA(void) const {return fThetaXAtDCA;}
  void     SetThetaXAtDCA(Double_t ThetaX) {fThetaXAtDCA = ThetaX;}
  Double_t GetThetaYAtDCA(void) const {return fThetaYAtDCA;}
  void     SetThetaYAtDCA(Double_t ThetaY) {fThetaYAtDCA = ThetaY;}
  Double_t GetBendingCoorAtDCA(void) const {return fBendingCoorAtDCA;}
  void     SetBendingCoorAtDCA(Double_t BendingCoor) {fBendingCoorAtDCA = BendingCoor;}
  Double_t GetNonBendingCoorAtDCA(void) const {return fNonBendingCoorAtDCA;}
  void     SetNonBendingCoorAtDCA(Double_t NonBendingCoor) {fNonBendingCoorAtDCA = NonBendingCoor;}
  Double_t GetDCA(void) const {return TMath::Sqrt(fNonBendingCoorAtDCA*fNonBendingCoorAtDCA +
						  fBendingCoorAtDCA*fBendingCoorAtDCA);}
  
  // Get and Set methods for data at first station
  Double_t GetInverseBendingMomentumUncorrected(void) const {return fInverseBendingMomentumUncorrected;}
  void     SetInverseBendingMomentumUncorrected(Double_t InverseBendingMomentum) 
		{fInverseBendingMomentumUncorrected = InverseBendingMomentum;}
  Double_t GetThetaXUncorrected(void) const {return fThetaXUncorrected;}
  void     SetThetaXUncorrected(Double_t ThetaX) {fThetaXUncorrected = ThetaX;}
  Double_t GetThetaYUncorrected(void) const {return fThetaYUncorrected;}
  void     SetThetaYUncorrected(Double_t ThetaY) {fThetaYUncorrected = ThetaY;}
  Double_t GetZUncorrected(void) const {return fZUncorrected;}
  void     SetZUncorrected(Double_t Z) {fZUncorrected = Z;}
  Double_t GetBendingCoorUncorrected(void) const {return fBendingCoorUncorrected;}
  void     SetBendingCoorUncorrected(Double_t BendingCoor) {fBendingCoorUncorrected = BendingCoor;}
  Double_t GetNonBendingCoorUncorrected(void) const {return fNonBendingCoorUncorrected;}
  void     SetNonBendingCoorUncorrected(Double_t NonBendingCoor) {fNonBendingCoorUncorrected = NonBendingCoor;}
  
  // Get and Set methods for covariance matrix of data at first station
  void     GetCovariances(TMatrixD& cov) const;
  void     SetCovariances(const TMatrixD& cov);
  void     GetCovarianceXYZPxPyPz(Double_t cov[21]) const;
  
  // Get and Set methods for the transverse position r of the track at the end of the absorber
  Double_t GetRAtAbsorberEnd() const { return fRAtAbsorberEnd; }
  void     SetRAtAbsorberEnd(Double_t r) { fRAtAbsorberEnd = r; }
  
  // Get and Set methods for global tracking info
  Double_t GetChi2(void) const {return fChi2;}
  void     SetChi2(Double_t Chi2) {fChi2 = Chi2;}
  UChar_t  GetNHit(void) const {return fNHit;}
  Int_t    GetNDF() const;
  Double_t GetNormalizedChi2() const;
  
  // Get and Set methods for trigger matching
  Int_t    GetMatchTrigger() const;
  Bool_t   MatchTriggerDigits(Bool_t fromTrack) const;
  Double_t GetChi2MatchTrigger() const {return fChi2MatchTrigger;}
  void     SetChi2MatchTrigger(Double_t Chi2MatchTrigger) {fChi2MatchTrigger = Chi2MatchTrigger;}
  UShort_t GetHitsPatternInTrigCh() const {return fHitsPatternInTrigCh;}
  void     SetHitsPatternInTrigCh(UShort_t hitsPatternInTrigCh) {fHitsPatternInTrigCh = hitsPatternInTrigCh;}
  UInt_t   GetHitsPatternInTrigChTrk() const {return fHitsPatternInTrigChTrk;}
  void     SetHitsPatternInTrigChTrk(UInt_t hitsPatternInTrigChTrk) {fHitsPatternInTrigChTrk = hitsPatternInTrigChTrk;}
  void     SetLocalTrigger(Int_t locTrig) { fLocalTrigger = locTrig; }
  Int_t    LoCircuit(void) const { return fLocalTrigger & 0xFF;       }
  Int_t    LoStripX(void) const  { return fLocalTrigger >>  8 & 0x1F; }
  Int_t    LoStripY(void) const  { return fLocalTrigger >> 13 & 0x0F; }
  Int_t    LoDev(void)    const  { return fLocalTrigger >> 17 & 0x1F; }
  Int_t    LoLpt(void)    const  { return fLocalTrigger >> 22 & 0x03; }
  Int_t    LoHpt(void)    const  { return fLocalTrigger >> 24 & 0x03; }
  Int_t    GetTriggerWithoutChamber(void) const { return fLocalTrigger >> 26 & 0xF; }
  Bool_t TriggerFiredWithoutChamber(Int_t ich) const { return GetTriggerWithoutChamber() >> (3 - ich) & 0x1; }
  Int_t GetMuonTrigDevSign() const;

  // Get and Set methods for the hit strips pattern in the trigger chambers
  UShort_t GetTriggerX1Pattern() const { return fX1Pattern; }
  UShort_t GetTriggerY1Pattern() const { return fY1Pattern; }
  UShort_t GetTriggerX2Pattern() const { return fX2Pattern; }
  UShort_t GetTriggerY2Pattern() const { return fY2Pattern; }
  UShort_t GetTriggerX3Pattern() const { return fX3Pattern; }
  UShort_t GetTriggerY3Pattern() const { return fY3Pattern; }
  UShort_t GetTriggerX4Pattern() const { return fX4Pattern; }
  UShort_t GetTriggerY4Pattern() const { return fY4Pattern; }
  void     SetTriggerX1Pattern(UShort_t pat) { fX1Pattern = pat; }
  void     SetTriggerY1Pattern(UShort_t pat) { fY1Pattern = pat; }
  void     SetTriggerX2Pattern(UShort_t pat) { fX2Pattern = pat; }
  void     SetTriggerY2Pattern(UShort_t pat) { fY2Pattern = pat; }
  void     SetTriggerX3Pattern(UShort_t pat) { fX3Pattern = pat; }
  void     SetTriggerY3Pattern(UShort_t pat) { fY3Pattern = pat; }
  void     SetTriggerX4Pattern(UShort_t pat) { fX4Pattern = pat; }
  void     SetTriggerY4Pattern(UShort_t pat) { fY4Pattern = pat; }

  // Get and Set methods for muon cluster map
  UInt_t   GetMuonClusterMap() const {return fMuonClusterMap;}
  void     SetMuonClusterMap(UInt_t muonClusterMap) {fMuonClusterMap = muonClusterMap;}
  void     AddInMuonClusterMap(Int_t chamber) {fMuonClusterMap |= BIT(chamber);}
  Bool_t   IsInMuonClusterMap(Int_t chamber) const {return (Bool_t) ((fMuonClusterMap & BIT(chamber)) != 0);}
  
  // Identify the tracks sharing cluster(s) with another (use the last bit of fMuonClusterMap)
  void     Connected(Bool_t flag = kTRUE) {flag ? SETBIT(fMuonClusterMap,31) : CLRBIT(fMuonClusterMap,31);}
  Bool_t   IsConnected() const {return TESTBIT(fMuonClusterMap,31);}
  
  // Methods to fill and get the Id of associated clusters
  void     AddClusterId(UInt_t clusterId);
  Int_t    GetNClusters() const {return static_cast<Int_t>(fNHit);}
  UInt_t   GetClusterId(Int_t i) const {return (fClustersId && i >= 0 && i < fNHit) ? static_cast<UInt_t>(fClustersId->At(i)) : 0;}
  
  // Method to transfer clusters to the new ESD structure
  Bool_t   IsOldTrack() {return (fClusters);}
  void     MoveClustersToESD(AliESDEvent &esd);
  
  // Methods to compute track momentum
  Double_t Px() const;
  Double_t Py() const;
  Double_t Pz() const;
  Double_t P() const;
  Bool_t   PxPyPz(Double_t p[3]) const { p[0] = Px(); p[1] = Py(); p[2] = Pz(); return kTRUE; }
  void     LorentzP(TLorentzVector& vP) const;
  Double_t PxAtDCA() const;
  Double_t PyAtDCA() const;
  Double_t PzAtDCA() const;
  Double_t PAtDCA() const;
  Bool_t   PxPyPzAtDCA(Double_t p[3]) const { p[0] = Px(); p[1] = Py(); p[2] = Pz(); return kTRUE; }
  void     LorentzPAtDCA(TLorentzVector& vP) const;
  Double_t PxUncorrected() const;
  Double_t PyUncorrected() const;
  Double_t PzUncorrected() const;
  Double_t PUncorrected() const;
  Bool_t   PxPyPzUncorrected(Double_t p[3]) const { p[0] = Px(); p[1] = Py(); p[2] = Pz(); return kTRUE; }
  void     LorentzPUncorrected(TLorentzVector& vP) const;
  
  // additional methods to comply with AliVParticle
  Double_t Xv() const {return -999.;} // put reasonable values here
  Double_t Yv() const {return -999.;} //
  Double_t Zv() const {return -999.;} //
  Bool_t   XvYvZv(Double_t x[3]) const { x[0] = Xv(); x[1] = Yv(); x[2] = Zv(); return kTRUE; }  
  Double_t Pt() const { return TMath::Sqrt(Px()*Px() + Py()*Py()); }
  Double_t OneOverPt() const { return (Pt() != 0.) ? 1./Pt() : FLT_MAX; }
  Double_t Phi() const { return TMath::Pi()+TMath::ATan2(-Py(), -Px()); }
  Double_t Theta() const { return TMath::ATan2(Pt(), Pz()); }
  Double_t E() const { return TMath::Sqrt(M()*M() + P()*P()); }
  Double_t M() const { return TDatabasePDG::Instance()->GetParticle("mu-")->Mass(); }
  Double_t Eta() const { return -TMath::Log(TMath::Tan(0.5 * Theta()));}
  Double_t Y() const { return (Pz()/E() != 1.) ? TMath::ATanH(Pz()/E()) : FLT_MAX; }
  Short_t  Charge() const { return (Short_t)TMath::Sign(1., GetInverseBendingMomentum()); }


  // Dummy
  const Double_t *PID() const { return (Double_t*)0x0; }
  Int_t    PdgCode() const {return 0;}
  
  /// Set the corresponding MC track number
  void  SetLabel(Int_t label) {fLabel = label;}
  /// Return the corresponding MC track number
  Int_t GetLabel() const {return fLabel;}

  /// Additional methods to decode hit pattern
  /// The hit pattern is a UShort_t with:
  /// <pre>
  ///   0    |   1 0 0 0 1  |  1 1  |   1  1  0  1  |   1  1  0  1   
  ///        |              |       |               | 
  /// unused |  RPC (0-17)  |  flag |  Bend plane   | Non-bend plane
  ///        |      or      |       | Match chamber | Match chamber
  ///        | further info |       |  11 12 13 14  |  11 12 13 14
  ///        |    (20-24)   |       |               |
  /// </pre>
  enum EAliTriggerChPatternFlag {
    kNoEff,    ///< Track is not good for chamber efficiency evaluation
    kChEff,    ///< Track crosses different RPCs
    kSlatEff,  ///< Track crosses the same RPC in all planes
    kBoardEff  ///< Track crosses the same board in all planes
  };
  enum EAliTriggerChPatternInfo {
    kCrossDifferentSlats  = 20, ///< The RPC cannot be univoquely determined
    kTrackMatchesManyPads = 21, ///< Track not good for effciency calculation since it matches many pads
    kTrackMatchesFewPads  = 22, ///< Track not good for effciency calculation since it matches pads in less than 3/4 chambers
    kTrackOutsideGeometry = 23, ///< Problems in pattern determination since track extrapolation is outside trigger chambers
    kTrackerTrackPattern  = 24, ///< The pattern was calculated from a tracker track not matching trigger track
    kTrackMatchesMasks    = 25  ///< Track not good for effciency calculation since it matches masked pads
  };
  /// Set hits pattern
  static void SetFiredChamber(UInt_t& pattern, Int_t cathode, Int_t chamber);
  /// Add efficiency flag and crossed RPC or info on rejected track
  static void AddEffInfo(UInt_t& pattern, Int_t slatOrInfo, Int_t board = 0, EAliTriggerChPatternFlag effType = kNoEff);
  /// Chamber was hit
  static Bool_t IsChamberHit(UInt_t pattern, Int_t cathode, Int_t chamber);
  /// Get Efficiency flag
  static Int_t GetEffFlag(UInt_t pattern);
  /// Getting crossed slat or info
  static Int_t GetSlatOrInfo(UInt_t pattern);
  /// Getting crossed board
  static Int_t GetCrossedBoard(UInt_t pattern);

  void AddMuonTrigDevSignInfo(UInt_t& pattern) const;

  AliESDEvent* GetESDEvent() const {return fESDEvent;}
  void         SetESDEvent(AliESDEvent* evt) {fESDEvent = evt;}  
  
protected:
  // parameters at vertex
  Double32_t fInverseBendingMomentum; ///< Inverse bending momentum (GeV/c ** -1) times the charge 
  Double32_t fThetaX;                 ///< Angle of track at vertex in X direction (rad)
  Double32_t fThetaY;                 ///< Angle of track at vertex in Y direction (rad)
  Double32_t fZ;                      ///< Z coordinate (cm)
  Double32_t fBendingCoor;            ///< bending coordinate (cm)
  Double32_t fNonBendingCoor;         ///< non bending coordinate (cm)
  
  // parameters at Distance of Closest Approach in the vertex plane
  Double32_t fInverseBendingMomentumAtDCA; ///< Inverse bending momentum (GeV/c ** -1) times the charge 
  Double32_t fThetaXAtDCA;                 ///< Angle of track at vertex in X direction (rad)
  Double32_t fThetaYAtDCA;                 ///< Angle of track at vertex in Y direction (rad)
  Double32_t fBendingCoorAtDCA;            ///< bending coordinate (cm)
  Double32_t fNonBendingCoorAtDCA;         ///< non bending coordinate (cm)
  
  // parameters at first tracking station
  Double32_t fInverseBendingMomentumUncorrected; ///< Inverse bending momentum (GeV/c ** -1) times the charge 
  Double32_t fThetaXUncorrected;                 ///< Angle of track at vertex in X direction (rad)
  Double32_t fThetaYUncorrected;                 ///< Angle of track at vertex in Y direction (rad)
  Double32_t fZUncorrected;                      ///< Z coordinate (cm)
  Double32_t fBendingCoorUncorrected;            ///< bending coordinate (cm)
  Double32_t fNonBendingCoorUncorrected;         ///< non bending coordinate (cm)
  
  /// reduced covariance matrix of UNCORRECTED track parameters, ordered as follow:      <pre>
  /// [0] =  <X,X>
  /// [1] =<X,ThetaX>  [2] =<ThetaX,ThetaX>
  /// [3] =  <X,Y>     [4] =  <Y,ThetaX>     [5] =  <Y,Y>
  /// [6] =<X,ThetaY>  [7] =<ThetaX,ThetaY>  [8] =<Y,ThetaY>  [9] =<ThetaY,ThetaY>
  /// [10]=<X,InvP_yz> [11]=<ThetaX,InvP_yz> [12]=<Y,InvP_yz> [13]=<ThetaY,InvP_yz> [14]=<InvP_yz,InvP_yz>  </pre>
  Double32_t fCovariances[15]; ///< \brief reduced covariance matrix of parameters AT FIRST CHAMBER
  
  Double32_t fRAtAbsorberEnd; ///< transverse position r of the track at the end of the absorber
  
  // global tracking info
  Double32_t fChi2;                ///< chi2 in the MUON track fit
  Double32_t fChi2MatchTrigger;    ///< chi2 of trigger/track matching
  Int_t      fLocalTrigger;        ///< packed local trigger information

  // hit strips pattern in the trigger chambers
  UShort_t fX1Pattern;             ///< x-strips pattern in st6/ch1
  UShort_t fY1Pattern;             ///< y-strips pattern in st6/ch1
  UShort_t fX2Pattern;             ///< x-strips pattern in st6/ch2
  UShort_t fY2Pattern;             ///< y-strips pattern in st6/ch2
  UShort_t fX3Pattern;             ///< x-strips pattern in st7/ch1
  UShort_t fY3Pattern;             ///< y-strips pattern in st7/ch1
  UShort_t fX4Pattern;             ///< x-strips pattern in st7/ch2
  UShort_t fY4Pattern;             ///< y-strips pattern in st7/ch2
  
  UInt_t   fMuonClusterMap;        ///< Map of clusters in tracking chambers
  UShort_t fHitsPatternInTrigCh;   ///< Word containing info on the hits left in trigger chambers
  UInt_t  fHitsPatternInTrigChTrk; ///< Trigger hit map from tracker track extrapolation
  UChar_t  fNHit;                  ///< number of clusters attached to the track
  
  mutable TClonesArray* fClusters; ///< Array of clusters attached to the track -- deprecated
  
  TArrayI* fClustersId;            ///< Array of clusters'Id attached to the track
  
  Int_t fLabel;                    ///< point to the corresponding MC track

  AliESDEvent*   fESDEvent; //!Pointer back to event to which the track belongs
  
  ClassDef(AliESDMuonTrack,15) // MUON ESD track class 
};

#endif 
