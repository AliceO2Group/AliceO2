//-*- Mode: C++ -*-
// $Id$

//* This file is property of and copyright by the ALICE HLT Project        * 
//* ALICE Experiment at CERN, All rights reserved.                         *
//* See cxx source for full Copyright notice                               *

#ifndef ALIHLTTPCCAPERFORMANCE_H
#define ALIHLTTPCCAPERFORMANCE_H

#include "AliHLTTPCCADef.h"

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

  AliHLTTPCCAPerformance();
  AliHLTTPCCAPerformance(const AliHLTTPCCAPerformance&);
  AliHLTTPCCAPerformance &operator=(const AliHLTTPCCAPerformance&);

  virtual ~AliHLTTPCCAPerformance();

  void SetTracker( AliHLTTPCCAGBTracker *Tracker );
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
  void Performance();

  void WriteMCEvent( ostream &out ) const;
  void ReadMCEvent( istream &in );
  void WriteMCPoints( ostream &out ) const;
  void ReadMCPoints( istream &in );
  Bool_t& DoClusterPulls(){ return fDoClusterPulls; }

protected:

  AliHLTTPCCAGBTracker *fTracker; //* pointer to the tracker
  
  struct AliHLTTPCCAHitLabel{
    Int_t fLab[3]; //* array of 3 MC labels
  };

  AliHLTTPCCAHitLabel *fHitLabels; //* array of hit MC labels
  Int_t fNHits;                    //* number of hits
  AliHLTTPCCAMCTrack *fMCTracks;   //* array of MC tracks
  Int_t fNMCTracks;                //* number of MC tracks
  AliHLTTPCCAMCPoint *fMCPoints;   //* array of MC points
  Int_t fNMCPoints;                //* number of MC points
  Bool_t fDoClusterPulls;          //* do cluster pulls (very slow)
  Int_t fStatNEvents; //* n of events proceed
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
  *fhPullQPt;    //* track Q/Pt pull at the TPC entrance

  TH1D 
  *fhHitErrY, //* hit error in Y
    *fhHitErrZ,//* hit error in Z    
    *fhHitResY,//* hit resolution Y
    *fhHitResZ,//* hit resolution Z
    *fhHitPullY,//* hit  pull Y
    *fhHitPullZ;//* hit  pull Z

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
    *fhGBEffVsP, //* global reconstruction efficiency vs P plot
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

  static void WriteDir2Current( TObject *obj );
  
};

#endif
