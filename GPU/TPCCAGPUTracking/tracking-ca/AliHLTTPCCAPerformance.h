//-*- Mode: C++ -*-
// $Id$
// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        * 
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************

#ifndef ALIHLTTPCCAPERFORMANCE_H
#define ALIHLTTPCCAPERFORMANCE_H

#include "AliHLTTPCCADef.h"
#include "Riostream.h"

class TObject;
class TParticle;
class AliHLTTPCCAMCTrack;
class AliHLTTPCCAMCPoint;
class AliHLTTPCCAGBTracker;
class TDirectory;
class TH1D;
class TH2D;
class TProfile;

/**
 * @class AliHLTTPCCAPerformance
 * 
 * Does performance evaluation of the HLT Cellular Automaton-based tracker
 * It checks performance for AliHLTTPCCATracker slice tracker 
 * and for AliHLTTPCCAGBTracker global tracker
 *
 */
class AliHLTTPCCAPerformance
{

 public:

  struct AliHLTTPCCAHitLabel{
    Int_t fLab[3]; //* array of 3 MC labels
  };

  AliHLTTPCCAPerformance();
  AliHLTTPCCAPerformance(const AliHLTTPCCAPerformance&);
  const AliHLTTPCCAPerformance &operator=(const AliHLTTPCCAPerformance&) const;

  virtual ~AliHLTTPCCAPerformance();

  static AliHLTTPCCAPerformance &Instance();

  void SetTracker( AliHLTTPCCAGBTracker * const Tracker );
  void StartEvent();
  void SetNHits( Int_t NHits );
  void SetNMCTracks( Int_t NMCTracks );
  void SetNMCPoints( Int_t NMCPoints );

  void ReadHitLabel( Int_t HitID, 
		     Int_t lab0, Int_t lab1, Int_t lab2 );
  void ReadMCTrack( Int_t index, const TParticle *part );
  void ReadMCTPCTrack( Int_t index, Float_t X, Float_t Y, Float_t Z, 
		       Float_t Px, Float_t Py, Float_t Pz ); 

  void ReadMCPoint( Int_t TrackID, Float_t X, Float_t Y, Float_t Z, Float_t Time, Int_t iSlice );

  void CreateHistos();
  void WriteHistos();
  void SlicePerformance( Int_t iSlice, Bool_t PrintFlag  );
  void SliceTrackletPerformance( Int_t iSlice, Bool_t PrintFlag );
  void SliceTrackCandPerformance( Int_t iSlice, Bool_t PrintFlag );

  void Performance( fstream *StatFile = 0);

  void WriteMCEvent( ostream &out ) const;
  void ReadMCEvent( istream &in );
  void WriteMCPoints( ostream &out ) const;
  void ReadMCPoints( istream &in );
  Bool_t DoClusterPulls() const { return fDoClusterPulls; }
  void SetDoClusterPulls( Bool_t v ) { fDoClusterPulls = v; }
  AliHLTTPCCAHitLabel *HitLabels() const { return fHitLabels;}
  AliHLTTPCCAMCTrack *MCTracks() const { return fMCTracks; }
  Int_t NMCTracks() const { return fNMCTracks; }

  TH1D *HNHitsPerSeed() const { return fhNHitsPerSeed;}
  TH1D *HNHitsPerTrackCand() const { return fhNHitsPerTrackCand; }

  TH1D *LinkChiRight( int i ) const { return fhLinkChiRight[i]; }
  TH1D *LinkChiWrong( int i ) const { return fhLinkChiWrong[i]; }

  void LinkPerformance( Int_t iSlice );

protected:

  AliHLTTPCCAGBTracker *fTracker; //* pointer to the tracker
  

  AliHLTTPCCAHitLabel *fHitLabels; //* array of hit MC labels
  Int_t fNHits;                    //* number of hits
  AliHLTTPCCAMCTrack *fMCTracks;   //* array of MC tracks
  Int_t fNMCTracks;                //* number of MC tracks
  AliHLTTPCCAMCPoint *fMCPoints;   //* array of MC points
  Int_t fNMCPoints;                //* number of MC points
  Bool_t fDoClusterPulls;          //* do cluster pulls (very slow)
  Int_t fStatNEvents; //* n of events proceed
  Double_t fStatTime; //* reco time;

  Int_t fStatSeedNRecTot; //* total n of reconstructed tracks 
  Int_t fStatSeedNRecOut; //* n of reconstructed tracks in Out set
  Int_t fStatSeedNGhost;//* n of reconstructed tracks in Ghost set
  Int_t fStatSeedNMCAll;//* n of MC tracks 
  Int_t fStatSeedNRecAll; //* n of reconstructed tracks in All set
  Int_t fStatSeedNClonesAll;//* total n of reconstructed tracks in Clone set
  Int_t fStatSeedNMCRef; //* n of MC reference tracks 
  Int_t fStatSeedNRecRef; //* n of reconstructed tracks in Ref set
  Int_t fStatSeedNClonesRef; //* n of reconstructed clones in Ref set

  Int_t fStatCandNRecTot; //* total n of reconstructed tracks 
  Int_t fStatCandNRecOut; //* n of reconstructed tracks in Out set
  Int_t fStatCandNGhost;//* n of reconstructed tracks in Ghost set
  Int_t fStatCandNMCAll;//* n of MC tracks 
  Int_t fStatCandNRecAll; //* n of reconstructed tracks in All set
  Int_t fStatCandNClonesAll;//* total n of reconstructed tracks in Clone set
  Int_t fStatCandNMCRef; //* n of MC reference tracks 
  Int_t fStatCandNRecRef; //* n of reconstructed tracks in Ref set
  Int_t fStatCandNClonesRef; //* n of reconstructed clones in Ref set

  Int_t fStatNRecTot; //* total n of reconstructed tracks 
  Int_t fStatNRecOut; //* n of reconstructed tracks in Out set
  Int_t fStatNGhost;//* n of reconstructed tracks in Ghost set
  Int_t fStatNMCAll;//* n of MC tracks 
  Int_t fStatNRecAll; //* n of reconstructed tracks in All set
  Int_t fStatNClonesAll;//* total n of reconstructed tracks in Clone set
  Int_t fStatNMCRef; //* n of MC reference tracks 
  Int_t fStatNRecRef; //* n of reconstructed tracks in Ref set
  Int_t fStatNClonesRef; //* n of reconstructed clones in Ref set

  Int_t fStatGBNRecTot; //* global tracker: total n of reconstructed tracks 
  Int_t fStatGBNRecOut; //* global tracker: n of reconstructed tracks in Out set
  Int_t fStatGBNGhost;//* global tracker: n of reconstructed tracks in Ghost set
  Int_t fStatGBNMCAll;//* global tracker: n of MC tracks 
  Int_t fStatGBNRecAll; //* global tracker: n of reconstructed tracks in All set
  Int_t fStatGBNClonesAll;//* global tracker: total n of reconstructed tracks in Clone set
  Int_t fStatGBNMCRef; //* global tracker: n of MC reference tracks 
  Int_t fStatGBNRecRef; //* global tracker: n of reconstructed tracks in Ref set
  Int_t fStatGBNClonesRef; //* global tracker: n of reconstructed clones in Ref set

  TDirectory *fHistoDir; //* ROOT directory with histogramms

  TH1D 
  *fhResY,       //* track Y resolution at the TPC entrance
  *fhResZ,       //* track Z resolution at the TPC entrance
  *fhResSinPhi,  //* track SinPhi resolution at the TPC entrance
  *fhResDzDs,    //* track DzDs resolution at the TPC entrance
  *fhResPt,      //* track Pt relative resolution at the TPC entrance
  *fhPullY,      //* track Y pull at the TPC entrance
  *fhPullZ,      //* track Z pull at the TPC entrance
  *fhPullSinPhi, //* track SinPhi pull at the TPC entrance
  *fhPullDzDs, //* track DzDs pull at the TPC entrance
  *fhPullQPt,    //* track Q/Pt pull at the TPC entrance
  *fhPullYS,       //* sqrt(chi2/ndf) deviation of the track parameters Y and SinPhi at the TPC entrance
  *fhPullZT;      //* sqrt(chi2/ndf) deviation of the track parameters Z and DzDs at the TPC entrance

  TH1D 
  *fhHitErrY, //* hit error in Y
    *fhHitErrZ,//* hit error in Z    
    *fhHitResY,//* hit resolution Y
    *fhHitResZ,//* hit resolution Z
    *fhHitPullY,//* hit  pull Y
    *fhHitPullZ;//* hit  pull Z
  TProfile *fhHitShared; //* ratio of the shared clusters

  TH1D 
    *fhHitResY1,//* hit resolution Y, pt>1GeV
    *fhHitResZ1,//* hit resolution Z, pt>1GeV
    *fhHitPullY1,//* hit  pull Y, pt>1GeV
    *fhHitPullZ1;//* hit  pull Z, pt>1GeV

  TH1D
    *fhCellPurity,//* cell purity
    *fhCellNHits//* cell n hits
    ;

  TProfile 
    *fhCellPurityVsN, //* cell purity vs N hits
    *fhCellPurityVsPt,//* cell purity vs MC Pt
    *fhEffVsP, //* reconstruction efficiency vs P plot
    *fhSeedEffVsP, //* reconstruction efficiency vs P plot
    *fhCandEffVsP, //* reconstruction efficiency vs P plot
    *fhGBEffVsP, //* global reconstruction efficiency vs P plot
    *fhGBEffVsPt, //* global reconstruction efficiency vs P plot
    *fhNeighQuality, // quality for neighbours finder 
    *fhNeighEff,// efficiency for neighbours finder
    *fhNeighQualityVsPt,// quality for neighbours finder vs track Pt
    *fhNeighEffVsPt;// efficiency for neighbours finder vs track Pt
  TH1D 
  *fhNeighDy, // dy for neighbours
    *fhNeighDz,// dz for neighbours
    *fhNeighChi;// chi2^0.5 for neighbours
  TH2D
    *fhNeighDyVsPt, // dy for neighbours vs track Pt
    *fhNeighDzVsPt,// dz for neighbours vs track Pt
    *fhNeighChiVsPt, // chi2^0.5 for neighbours vs track Pt
    *fhNeighNCombVsArea; // N neighbours in the search area

  TH1D 
    *fhNHitsPerSeed, // n hits per track seed
    *fhNHitsPerTrackCand; // n hits per track candidate

  TH1D 
    *fhTrackLengthRef, // reconstructed track length, %
    *fhRefRecoX,// parameters of non-reconstructed ref. mc track
    *fhRefRecoY,// parameters of non-reconstructed ref. mc track
    *fhRefRecoZ,// parameters of non-reconstructed ref. mc track
    *fhRefRecoP, // parameters of non-reconstructed ref. mc track
    *fhRefRecoPt,// parameters of non-reconstructed ref. mc track
    *fhRefRecoAngleY,// parameters of non-reconstructed ref. mc track
    *fhRefRecoAngleZ,// parameters of non-reconstructed ref. mc track
    *fhRefRecoNHits,// parameters of non-reconstructed ref. mc track
    *fhRefNotRecoX,// parameters of non-reconstructed ref. mc track
    *fhRefNotRecoY,// parameters of non-reconstructed ref. mc track
    *fhRefNotRecoZ,// parameters of non-reconstructed ref. mc track
    *fhRefNotRecoP, // parameters of non-reconstructed ref. mc track
    *fhRefNotRecoPt,// parameters of non-reconstructed ref. mc track
    *fhRefNotRecoAngleY,// parameters of non-reconstructed ref. mc track
    *fhRefNotRecoAngleZ,// parameters of non-reconstructed ref. mc track
    *fhRefNotRecoNHits;// parameters of non-reconstructed ref. mc track

  TProfile * fhLinkEff[4]; // link efficiency
  TH1D *fhLinkAreaY[4]; // area in Y for the link finder
  TH1D *fhLinkAreaZ[4]; // area in Z for the link finder
  TH1D *fhLinkChiRight[4]; // sqrt(chi^2) for right neighbours
  TH1D *fhLinkChiWrong[4]; // sqrt(chi^2) for wrong neighbours

  static void WriteDir2Current( TObject *obj );
  
};

#endif
