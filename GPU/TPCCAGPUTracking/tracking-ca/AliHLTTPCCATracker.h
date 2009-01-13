//-*- Mode: C++ -*-
// @(#) $Id$

//* This file is property of and copyright by the ALICE HLT Project        * 
//* ALICE Experiment at CERN, All rights reserved.                         *
//* See cxx source for full Copyright notice                               *

#ifndef ALIHLTTPCCATRACKER_H
#define ALIHLTTPCCATRACKER_H


#include "AliHLTTPCCADef.h"
#include "AliHLTTPCCAParam.h"
#include "AliHLTTPCCARow.h"
#include "AliHLTTPCCAHit.h"

class AliHLTTPCCATrack;
class AliHLTTPCCAOutTrack;
class AliHLTTPCCATrackParam;
class AliHLTTPCCATrackParam1;



/**
 * @class AliHLTTPCCATracker
 * 
 * Slice tracker for ALICE HLT. 
 * The class reconstructs tracks in one slice of TPC.
 * The reconstruction algorithm is based on the Cellular Automaton method
 *
 * The CA tracker is designed stand-alone. 
 * It is integrated to the HLT framework via AliHLTTPCCATrackerComponent interface.
 * The class is under construction.
 *
 */
class AliHLTTPCCATracker
{
 public:

#if !defined(HLTCA_GPUCODE)  
  AliHLTTPCCATracker();
  AliHLTTPCCATracker( const AliHLTTPCCATracker& );
  AliHLTTPCCATracker &operator=( const AliHLTTPCCATracker& );

  GPUd() ~AliHLTTPCCATracker();
#endif

  GPUd() void Initialize( AliHLTTPCCAParam &param );

  GPUd() void StartEvent();

  GPUd() void ReadEvent( Int_t *RowFirstHit, Int_t *RowNHits, Float_t *Y, Float_t *Z, Int_t NHits );

  void Reconstruct();

  void WriteOutput();

  GPUd() void GetErrors2( Int_t iRow,  const AliHLTTPCCATrackParam &t, Float_t &Err2Y, Float_t &Err2Z ) const;
  GPUd() void GetErrors2( Int_t iRow,  const AliHLTTPCCATrackParam1 &t, Float_t &Err2Y, Float_t &Err2Z ) const;

  GPUhd() AliHLTTPCCAParam &Param(){ return fParam; }
  GPUhd() AliHLTTPCCARow *Rows(){ return fRows; }

  Int_t * HitsID(){ return fHitsID; }

  Int_t *OutTrackHits(){ return  fOutTrackHits; }
  Int_t NOutTrackHits() const { return  fNOutTrackHits; }
  GPUd() AliHLTTPCCAOutTrack *OutTracks(){ return  fOutTracks; }
  GPUd() Int_t NOutTracks() const { return  fNOutTracks; }
  GPUhd() Int_t *TrackHits(){ return fTrackHits; }

  GPUhd() AliHLTTPCCATrack *Tracks(){ return  fTracks; }
  GPUhd() Int_t &NTracks()  { return *fNTracks; }

  Double_t *Timers(){ return fTimers; }

  GPUhd() static Int_t IRowIHit2ID( Int_t iRow, Int_t iHit ){ 
    return (iHit<<8)+iRow; 
  }
  GPUhd() static Int_t ID2IRow( Int_t HitID ){ 
    return ( HitID%256 ); 
  }
  GPUhd() static Int_t ID2IHit( Int_t HitID ){ 
    return ( HitID>>8 ); 
  }  

  GPUhd() AliHLTTPCCAHit &ID2Hit( Int_t HitID ) {
    return fHits[fRows[HitID%256].FirstHit() + (HitID>>8)];
  }
  GPUhd() AliHLTTPCCARow &ID2Row( Int_t HitID ) {
    return fRows[HitID%256];
  }
  
  void FitTrack( AliHLTTPCCATrack &track, Float_t *t0 = 0 ) const;
  void FitTrackFull( AliHLTTPCCATrack &track, Float_t *t0 = 0 ) const;
  GPUhd() void SetPointers();

  GPUhd() Short_t *HitLinkUp(){ return fHitLinkUp;}
  GPUhd() Short_t *HitLinkDown(){ return fHitLinkDown;}
  GPUhd() Int_t  *StartHits(){ return fStartHits;}
  GPUhd() Int_t  *Tracklets(){ return fTracklets;}
  GPUhd() Int_t  *HitIsUsed(){ return fHitIsUsed;}
  GPUhd() AliHLTTPCCAHit *Hits(){ return fHits;}
  GPUhd() Int_t &NHitsTotal(){ return fNHitsTotal;}
  GPUhd() uint4 *&TexHitsFullData(){ return fTexHitsFullData;}
  GPUhd() Int_t &TexHitsFullSize(){ return fTexHitsFullSize;}

  GPUd() UChar_t GetGridContent( UInt_t i ) const; 
  GPUd() AliHLTTPCCAHit GetHit( UInt_t i ) const;

  private:
 
  //  

  AliHLTTPCCAParam fParam; // parameters
  AliHLTTPCCARow fRows[200];// array of hit rows
  Double_t fTimers[10]; // running CPU time for different parts of the algorithm
  
  // event
  Int_t fNHitsTotal;// total number of hits in event
  Int_t fGridSizeTotal; // total grid size
  Int_t fGrid1SizeTotal;// total grid1 size

  AliHLTTPCCAHit *fHits; // hits
  ushort2 *fHits1; // hits1
  
  UChar_t *fGridContents; // grid content
  UInt_t *fGrid1Contents; // grid1 content
  Int_t *fHitsID; // hit ID's
 
  // temporary information
  
  Short_t *fHitLinkUp; // array of up links
  Short_t *fHitLinkDown;// array of down links
  Int_t *fHitIsUsed; // array of used flags
  Int_t  *fStartHits;   // array of start hits
  Int_t *fTracklets; // array of tracklets

  Int_t *fNTracks;// number of reconstructed tracks
  AliHLTTPCCATrack *fTracks;   // reconstructed tracks
  Int_t *fTrackHits; // array of track hit numbers

  // output

  Int_t fNOutTracks; // number of tracks in fOutTracks array
  Int_t fNOutTrackHits;  // number of hits in fOutTrackHits array
  AliHLTTPCCAOutTrack *fOutTracks; // output array of the reconstructed tracks
  Int_t *fOutTrackHits;  // output array of ID's of the reconstructed hits

  char *fEventMemory; // common event memory
  UInt_t fEventMemSize; // size of the event memory

  uint4 *fTexHitsFullData; // CUDA texture for hits
  Int_t fTexHitsFullSize; // size of the CUDA texture
};


#endif
