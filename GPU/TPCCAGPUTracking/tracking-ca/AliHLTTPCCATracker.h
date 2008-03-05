//-*- Mode: C++ -*-
// @(#) $Id$

//* This file is property of and copyright by the ALICE HLT Project        * 
//* ALICE Experiment at CERN, All rights reserved.                         *
//* See cxx source for full Copyright notice                               *

#ifndef ALIHLTTPCCATRACKER_H
#define ALIHLTTPCCATRACKER_H


#include "Rtypes.h"
#include "AliHLTTPCCAParam.h"
#include "AliHLTTPCCARow.h"
#include "AliHLTTPCCATrack.h"

class AliHLTTPCCAHit;
class AliHLTTPCCACell;
class AliHLTTPCCAOutTrack;


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

  AliHLTTPCCATracker();
  AliHLTTPCCATracker( const AliHLTTPCCATracker& );
  AliHLTTPCCATracker &operator=( const AliHLTTPCCATracker& );

  virtual ~AliHLTTPCCATracker();

  void Initialize( AliHLTTPCCAParam &param );

  void StartEvent();

  void ReadHitRow( Int_t iRow, AliHLTTPCCAHit *Row, Int_t NHits );

  void Reconstruct();

  void FindCells();
  void FindTracks();
  void FitTrack( AliHLTTPCCATrack &track, Int_t nIter=2 );

  AliHLTTPCCAParam &Param(){ return fParam; }
  AliHLTTPCCARow *Rows(){ return fRows; }

  Int_t *OutTrackHits(){ return  fOutTrackHits; }
  Int_t NOutTrackHits() const { return  fNOutTrackHits; }
  AliHLTTPCCAOutTrack *OutTracks(){ return  fOutTracks; }
  Int_t NOutTracks() const { return  fNOutTracks; }

  AliHLTTPCCATrack *Tracks(){ return  fTracks; }
  Int_t NTracks() const { return fNTracks; }

  Int_t *TrackCells(){ return  fTrackCells; }

  Double_t *Timers(){ return fTimers; }

  AliHLTTPCCACell &GetTrackCell( AliHLTTPCCATrack &t, Int_t i ) const {
    Int_t ind = fTrackCells[t.IFirstCell()+i];
    AliHLTTPCCARow &row = fRows[ind%256];
    return row.Cells()[ind>>8];
  }
  AliHLTTPCCARow &GetTrackCellRow( AliHLTTPCCATrack &t, Int_t i ) const {
    Int_t ind = fTrackCells[t.IFirstCell()+i];
    return fRows[ind%256];    
  }
  Int_t GetTrackCellIRow( AliHLTTPCCATrack &t, Int_t i ) const {
    Int_t ind = fTrackCells[t.IFirstCell()+i];
    return ind%256;    
  }
 
 protected:
  
  AliHLTTPCCAParam fParam; // parameters

  AliHLTTPCCARow *fRows;// array of hit rows

  Int_t *fOutTrackHits;  // output array of ID's of the reconstructed hits
  Int_t fNOutTrackHits;  // number of hits in fOutTrackHits array
  AliHLTTPCCAOutTrack *fOutTracks; // output array of the reconstructed tracks
  Int_t fNOutTracks; // number of tracks in fOutTracks array
  Int_t *fTrackCells; // indices of cells for reconstructed tracks
  Int_t fNHitsTotal;// total number of hits in event
  AliHLTTPCCATrack *fTracks;   // reconstructed tracks
  Int_t fNTracks;// number of reconstructed tracks
  Int_t *fCellHitPointers;// global array of cell->hit pointers

  Double_t fTimers[10]; // running CPU time for different parts of the algorithm

  ClassDef(AliHLTTPCCATracker,1);
};

#endif
