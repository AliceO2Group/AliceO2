//-*- Mode: C++ -*-
// $Id$

//* This file is property of and copyright by the ALICE HLT Project        * 
//* ALICE Experiment at CERN, All rights reserved.                         *
//* See cxx source for full Copyright notice                               *

#ifndef ALIHLTTPCCAPERFORMANCE_H
#define ALIHLTTPCCAPERFORMANCE_H


#include "TObject.h"

class TParticle;
class AliHLTTPCCAMCTrack;
class AliHLTTPCCAGBTracker;
class TDirectory;
class TH1D;
class TProfile;

/**
 * @class AliHLTTPCCAPerformance
 * 
 * Does performance evaluation of the HLT Cellular Automaton-based tracker
 * It checks performance for AliHLTTPCCATracker slice tracker 
 * and for AliHLTTPCCAGBTracker global tracker
 *
 */
class AliHLTTPCCAPerformance:public TObject
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
  void ReadHitLabel( Int_t HitID, 
		     Int_t lab0, Int_t lab1, Int_t lab2 );
  void ReadMCTrack( Int_t index, const TParticle *part );

  void CreateHistos();
  void WriteHistos();
  void SlicePerformance( Int_t iSlice, Bool_t PrintFlag  );
  void Performance();

protected:

  AliHLTTPCCAGBTracker *fTracker; //* pointer to the tracker
  
  struct AliHLTTPCCAHitLabel{
    Int_t fLab[3]; //* array of 3 MC labels
  };
  AliHLTTPCCAHitLabel *fHitLabels; //* array of hit MC labels
  Int_t fNHits; //* number of hits
  AliHLTTPCCAMCTrack *fMCTracks; //* array of Mc tracks
  Int_t fNMCTracks; //* number of MC tracks
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

  TDirectory *fHistoDir; //* ROOT directory with histogramms

  TH1D 
  *fhHitErrY, //* hit error in Y
    *fhHitErrZ,//* hit error in Z
    *fhHitResX,//* hit resolution X
    *fhHitResY,//* hit resolution Y
    *fhHitResZ,//* hit resolution Z
    *fhHitPullX,//* hit  pull X
    *fhHitPullY,//* hit  pull Y
    *fhHitPullZ,//* hit  pull Z
    *fhCellPurity,//* cell purity
    *fhCellNHits//* cell n hits
    ;

    TProfile 
  *fhCellPurityVsN, //* cell purity vs N hits
  *fhCellPurityVsPt,//* cell purity vs MC Pt
  *fhEffVsP; //* reconstruction efficiency vs P plot

  static void WriteDir2Current( TObject *obj );
  
  ClassDef(AliHLTTPCCAPerformance,1) 
};

#endif
