//-*- Mode: C++ -*-
// @(#) $Id$
// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        * 
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************

#ifndef ALIHLTTPCCATRACKER_H
#define ALIHLTTPCCATRACKER_H


#include "AliHLTTPCCADef.h"
#include "AliHLTTPCCAParam.h"
#include "AliHLTTPCCARow.h"
#include "AliHLTTPCCAHit.h"
#include <iostream>

class AliHLTTPCCATrack;
class AliHLTTPCCAOutTrack;
class AliHLTTPCCATrackParam;
class AliHLTTPCCATracklet;
class AliHLTTPCCASliceOutput;

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

  GPUd() void Initialize( const AliHLTTPCCAParam &param );

  GPUd() void StartEvent();

  GPUd() void ReadEvent( const Int_t *RowFirstHit, const Int_t *RowNHits, const Float_t *X, const Float_t *Y, const Float_t *Z, Int_t NHits );

  GPUd() void SetupRowData();

  void Reconstruct();
  void WriteOutput();

  GPUd() void GetErrors2( Int_t iRow,  const AliHLTTPCCATrackParam &t, Float_t &Err2Y, Float_t &Err2Z ) const;
  GPUd() void GetErrors2( Int_t iRow, Float_t z, Float_t sinPhi, Float_t cosPhi, Float_t DzDs, Float_t &Err2Y, Float_t &Err2Z ) const;

  GPUhd() static Int_t IRowIHit2ID( Int_t iRow, Int_t iHit ){ 
    return (iHit<<8)+iRow; 
  }
  GPUhd() static Int_t ID2IRow( Int_t HitID ){ 
    return ( HitID%256 ); 
  }
  GPUhd() static Int_t ID2IHit( Int_t HitID ){ 
    return ( HitID>>8 ); 
  }  

  //GPUhd() AliHLTTPCCAHit &ID2Hit( Int_t HitID ) {
  //return fHits[fRows[HitID%256].FirstHit() + (HitID>>8)];
  //}
  GPUhd() const AliHLTTPCCARow &ID2Row( Int_t HitID ) const {
    return fRows[HitID%256];
  }
  
  void FitTrack( AliHLTTPCCATrack &track, Float_t *t0 = 0 ) const;
  void FitTrackFull( AliHLTTPCCATrack &track, Float_t *t0 = 0 ) const;
  GPUhd() void SetPointersCommon();
  GPUhd() void SetPointersHits( Int_t MaxNHits );
  GPUhd() void SetPointersTracks( Int_t MaxNTracks, Int_t MaxNHits );

#if !defined(HLTCA_GPUCODE)  
  GPUh() void WriteEvent( std::ostream &out );
  GPUh() void ReadEvent( std::istream &in );
  GPUh() void WriteTracks( std::ostream &out ) ;
  GPUh() void ReadTracks( std::istream &in );
#endif

  GPUhd() const AliHLTTPCCAParam &Param() const { return fParam; }
  GPUhd() void SetParam( const AliHLTTPCCAParam &v ){ fParam = v; }

  GPUhd() const AliHLTTPCCARow &Row(Int_t i) const { return fRows[i]; }
  GPUhd() Double_t Timer(Int_t i) const { return fTimers[i]; }
  GPUhd() void SetTimer(Int_t i, Double_t v ){ fTimers[i] = v; }

  GPUhd() Int_t NHitsTotal() const { return fNHitsTotal;}

  GPUhd() const Char_t *InputEvent()    const { return fInputEvent; }
  GPUhd() Int_t  InputEventSize() const { return fInputEventSize; }

  GPUhd() const uint4  *RowData()    const   { return fRowData; }
  GPUhd() Int_t  RowDataSize()  const { return fRowDataSize; }
 
  GPUhd() Int_t *HitInputIDs() const { return fHitInputIDs; }
  GPUhd() Int_t  *HitWeights() const { return fHitWeights; }  
  
  GPUhd() Int_t  *NTracklets() const { return fNTracklets; }
  GPUhd() Int_t  *TrackletStartHits() const { return fTrackletStartHits; }
  GPUhd() AliHLTTPCCATracklet  *Tracklets() const { return fTracklets;}
  
  GPUhd() Int_t *NTracks()  const { return fNTracks; }
  GPUhd() AliHLTTPCCATrack *Tracks() const { return  fTracks; }
  GPUhd() Int_t *NTrackHits()  const { return fNTrackHits; }
  GPUhd() Int_t *TrackHits() const { return fTrackHits; }

  GPUhd() const AliHLTTPCCASliceOutput * Output() const { return fOutput; }

  GPUhd()  Int_t *NOutTracks() const { return  fNOutTracks; }
  GPUhd()  AliHLTTPCCAOutTrack *OutTracks() const { return  fOutTracks; }
  GPUhd()  Int_t *NOutTrackHits() const { return  fNOutTrackHits; }
  GPUhd()  Int_t *OutTrackHits() const { return  fOutTrackHits; }
 
  GPUh() void SetCommonMemory( Char_t * const mem ){ fCommonMemory = mem; }

  private:  

  AliHLTTPCCAParam fParam; // parameters
  AliHLTTPCCARow fRows[200];// array of hit rows
  Double_t fTimers[10]; // running CPU time for different parts of the algorithm
  
  // event

  Int_t fNHitsTotal;// total number of hits in event
 
  Char_t *fCommonMemory; // common event memory
  Int_t   fCommonMemorySize; // size of the event memory [bytes]

  Char_t *fHitMemory; // event memory for hits
  Int_t   fHitMemorySize; // size of the event memory [bytes]

  Char_t *fTrackMemory; // event memory for tracks
  Int_t   fTrackMemorySize; // size of the event memory [bytes]

  Char_t *fInputEvent;     // input event
  Int_t   fInputEventSize; // size of the input event [bytes]

  uint4  *fRowData;     // TPC rows: clusters, grid, links to neighbours
  Int_t   fRowDataSize; // size of the row data
 
  Int_t *fHitInputIDs; // cluster index in InputEvent  
  Int_t *fHitWeights;  // the weight of the longest tracklet crossed the cluster
  
  Int_t *fNTracklets;     // number of tracklets 
  Int_t *fTrackletStartHits;   // start hits for the tracklets
  AliHLTTPCCATracklet *fTracklets; // tracklets

  // 
  Int_t *fNTracks;            // number of reconstructed tracks
  AliHLTTPCCATrack *fTracks;  // reconstructed tracks
  Int_t *fNTrackHits;           // number of track hits
  Int_t *fTrackHits;          // array of track hit numbers

  // output

  AliHLTTPCCASliceOutput *fOutput;

  // obsolete output

  Int_t *fNOutTracks; // number of tracks in fOutTracks array
  AliHLTTPCCAOutTrack *fOutTracks; // output array of the reconstructed tracks
  Int_t *fNOutTrackHits;  // number of hits in fOutTrackHits array
  Int_t *fOutTrackHits;  // output array of ID's of the reconstructed hits
  
  //temporary
  
  Int_t *fTmpHitInputIDs; // temporary step

};


#endif
