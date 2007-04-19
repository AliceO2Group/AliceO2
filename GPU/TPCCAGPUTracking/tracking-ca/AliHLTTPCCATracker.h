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

class AliHLTTPCCATrack;
class AliHLTTPCCAHit;
class AliHLTTPCCACell;
class AliHLTTPCCAOutTrack;
class AliHLTTPCCATrackParam;
class AliHLTTPCCAEndPoint;

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
  void MergeCells();
  void FindTracks();

  AliHLTTPCCAParam &Param(){ return fParam; }
  AliHLTTPCCARow *Rows(){ return fRows; }

  Int_t *OutTrackHits(){ return  fOutTrackHits; }
  Int_t NOutTrackHits() const { return  fNOutTrackHits; }
  AliHLTTPCCAOutTrack *OutTracks(){ return  fOutTracks; }
  Int_t NOutTracks() const { return  fNOutTracks; }

  AliHLTTPCCATrack *Tracks(){ return  fTracks; }
  Int_t NTracks() const { return fNTracks; }

  Double_t *Timers(){ return fTimers; }

  static Int_t IRowICell2ID( Int_t iRow, Int_t iCell ){ 
    return (iCell<<8)+iRow; 
  }
  static Int_t ID2IRow( Int_t CellID ){ 
    return ( CellID%256 ); 
  }
  static Int_t ID2ICell( Int_t CellID ){ 
    return ( CellID>>8 ); 
  }  
  AliHLTTPCCACell &ID2Cell( Int_t CellID ) const{
    return fRows[CellID%256].Cells()[CellID>>8];
  }
  AliHLTTPCCARow &ID2Row( Int_t CellID ) const{
    return fRows[CellID%256];
  }
  
  AliHLTTPCCAEndPoint &ID2Point( Int_t PointID ) const{
    return fRows[PointID%256].EndPoints()[PointID>>8];
  }

  void FitTrack( AliHLTTPCCATrack &track, Float_t *t0 = 0 ) const;

 protected:
  
  AliHLTTPCCAParam fParam; // parameters

  AliHLTTPCCARow *fRows;// array of hit rows

  Int_t *fOutTrackHits;  // output array of ID's of the reconstructed hits
  Int_t fNOutTrackHits;  // number of hits in fOutTrackHits array
  AliHLTTPCCAOutTrack *fOutTracks; // output array of the reconstructed tracks
  Int_t fNOutTracks; // number of tracks in fOutTracks array

  Int_t fNHitsTotal;// total number of hits in event
  AliHLTTPCCATrack *fTracks;   // reconstructed tracks
  Int_t fNTracks;// number of reconstructed tracks
  Int_t *fCellHitPointers;// global array of cell->hit pointers
  AliHLTTPCCACell *fCells;// global array of cells
  AliHLTTPCCAEndPoint *fEndPoints;// global array of endpoints
  Double_t fTimers[10]; // running CPU time for different parts of the algorithm

  ClassDef(AliHLTTPCCATracker,1);
};

#endif
