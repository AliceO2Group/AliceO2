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

    GPUd() void ReadEvent( const int *RowFirstHit, const int *RowNHits, const float *X, const float *Y, const float *Z, int NHits );

    GPUd() void SetupRowData();

    void Reconstruct();
    void WriteOutput();

    GPUd() void GetErrors2( int iRow,  const AliHLTTPCCATrackParam &t, float &Err2Y, float &Err2Z ) const;
    GPUd() void GetErrors2( int iRow, float z, float sinPhi, float cosPhi, float DzDs, float &Err2Y, float &Err2Z ) const;

    GPUhd() static int IRowIHit2ID( int iRow, int iHit ) {
      return ( iHit << 8 ) + iRow;
    }
    GPUhd() static int ID2IRow( int HitID ) {
      return ( HitID % 256 );
    }
    GPUhd() static int ID2IHit( int HitID ) {
      return ( HitID >> 8 );
    }

    //GPUhd() AliHLTTPCCAHit &ID2Hit( int HitID ) {
    //return fHits[fRows[HitID%256].FirstHit() + (HitID>>8)];
    //}
    GPUhd() const AliHLTTPCCARow &ID2Row( int HitID ) const {
      return fRows[HitID%256];
    }

    void FitTrack( AliHLTTPCCATrack &track, float *t0 = 0 ) const;
    void FitTrackFull( AliHLTTPCCATrack &track, float *t0 = 0 ) const;
    GPUhd() void SetPointersCommon();
    GPUhd() void SetPointersHits( int MaxNHits );
    GPUhd() void SetPointersTracks( int MaxNTracks, int MaxNHits );

#if !defined(HLTCA_GPUCODE)
    GPUh() void WriteEvent( std::ostream &out );
    GPUh() void ReadEvent( std::istream &in );
    GPUh() void WriteTracks( std::ostream &out ) ;
    GPUh() void ReadTracks( std::istream &in );
#endif

    GPUhd() const AliHLTTPCCAParam &Param() const { return fParam; }
    GPUhd() void SetParam( const AliHLTTPCCAParam &v ) { fParam = v; }

    GPUhd() const AliHLTTPCCARow &Row( int i ) const { return fRows[i]; }
    GPUhd() double Timer( int i ) const { return fTimers[i]; }
    GPUhd() void SetTimer( int i, double v ) { fTimers[i] = v; }

    GPUhd() int NHitsTotal() const { return fNHitsTotal;}

    GPUhd() const char *InputEvent()    const { return fInputEvent; }
    GPUhd() int  InputEventSize() const { return fInputEventSize; }

    GPUhd() const uint4  *RowData()    const   { return fRowData; }
    GPUhd() int  RowDataSize()  const { return fRowDataSize; }

    GPUhd() int *HitInputIDs() const { return fHitInputIDs; }
    GPUhd() int  *HitWeights() const { return fHitWeights; }

    GPUhd() int  *NTracklets() const { return fNTracklets; }
    GPUhd() int  *TrackletStartHits() const { return fTrackletStartHits; }
    GPUhd() AliHLTTPCCATracklet  *Tracklets() const { return fTracklets;}

    GPUhd() int *NTracks()  const { return fNTracks; }
    GPUhd() AliHLTTPCCATrack *Tracks() const { return  fTracks; }
    GPUhd() int *NTrackHits()  const { return fNTrackHits; }
    GPUhd() int *TrackHits() const { return fTrackHits; }

    GPUhd() const AliHLTTPCCASliceOutput * Output() const { return fOutput; }

    GPUhd()  int *NOutTracks() const { return  fNOutTracks; }
    GPUhd()  AliHLTTPCCAOutTrack *OutTracks() const { return  fOutTracks; }
    GPUhd()  int *NOutTrackHits() const { return  fNOutTrackHits; }
    GPUhd()  int *OutTrackHits() const { return  fOutTrackHits; }

    GPUh() void SetCommonMemory( char * const mem ) { fCommonMemory = mem; }

  private:

    AliHLTTPCCAParam fParam; // parameters
    AliHLTTPCCARow fRows[200];// array of hit rows
    double fTimers[10]; // running CPU time for different parts of the algorithm

    // event

    int fNHitsTotal;// total number of hits in event

    char *fCommonMemory; // common event memory
    int   fCommonMemorySize; // size of the event memory [bytes]

    char *fHitMemory; // event memory for hits
    int   fHitMemorySize; // size of the event memory [bytes]

    char *fTrackMemory; // event memory for tracks
    int   fTrackMemorySize; // size of the event memory [bytes]

    char *fInputEvent;     // input event
    int   fInputEventSize; // size of the input event [bytes]

    uint4  *fRowData;     // TPC rows: clusters, grid, links to neighbours
    int   fRowDataSize; // size of the row data

    int *fHitInputIDs; // cluster index in InputEvent
    int *fHitWeights;  // the weight of the longest tracklet crossed the cluster

    int *fNTracklets;     // number of tracklets
    int *fTrackletStartHits;   // start hits for the tracklets
    AliHLTTPCCATracklet *fTracklets; // tracklets

    //
    int *fNTracks;            // number of reconstructed tracks
    AliHLTTPCCATrack *fTracks;  // reconstructed tracks
    int *fNTrackHits;           // number of track hits
    int *fTrackHits;          // array of track hit numbers

    // output

    AliHLTTPCCASliceOutput *fOutput;

    // obsolete output

    int *fNOutTracks; // number of tracks in fOutTracks array
    AliHLTTPCCAOutTrack *fOutTracks; // output array of the reconstructed tracks
    int *fNOutTrackHits;  // number of hits in fOutTrackHits array
    int *fOutTrackHits;  // output array of ID's of the reconstructed hits

    //temporary

    int *fTmpHitInputIDs; // temporary step

};


#endif
