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
#include "AliHLTArray.h"
#include "AliHLTTPCCAHitId.h"
#include "AliHLTTPCCASliceData.h"
#include "AliHLTTPCCATracklet.h"
#include "AliHLTTPCCAOutTrack.h"

class AliHLTTPCCATrack;
class AliHLTTPCCATrackParam;
class AliHLTTPCCAClusterData;
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

    GPUd() ~AliHLTTPCCATracker();
#endif

    GPUd() void Initialize( const AliHLTTPCCAParam &param );

    GPUd() void StartEvent();

    void ReadEvent( AliHLTTPCCAClusterData *clusterData );

    void Reconstruct();
    void WriteOutput();

    GPUd() void GetErrors2( int iRow,  const AliHLTTPCCATrackParam &t, float &Err2Y, float &Err2Z ) const;
    GPUd() void GetErrors2( int iRow, float z, float sinPhi, float cosPhi, float DzDs, float &Err2Y, float &Err2Z ) const;

    void FitTrack( const AliHLTTPCCATrack &track, float *t0 = 0 ) const;
    void FitTrackFull( const AliHLTTPCCATrack &track, float *t0 = 0 ) const;

    GPUhd() void SetPointersCommon();
    GPUhd() void SetPointersHits( int MaxNHits );
    GPUhd() void SetPointersTracks( int MaxNTracks, int MaxNHits );

#if !defined(HLTCA_GPUCODE)
    GPUh() void WriteEvent( std::ostream &out );
    GPUh() void WriteTracks( std::ostream &out ) ;
    GPUh() void ReadTracks( std::istream &in );
#endif

    GPUhd() const AliHLTTPCCAParam &Param() const { return fParam; }
    GPUhd() void SetParam( const AliHLTTPCCAParam &v ) { fParam = v; }

    GPUhd() const AliHLTTPCCAClusterData *ClusterData() const { return fClusterData; }
    GPUhd() const AliHLTTPCCASliceData &Data() const { return fData; }
    GPUhd() const AliHLTTPCCARow &Row( int rowIndex ) const { return fData.Row( rowIndex ); }
    GPUhd() const AliHLTTPCCARow &Row( const AliHLTTPCCAHitId &HitId ) const { return fData.Row( HitId.RowIndex() ); }

    GPUhd() double Timer( int i ) const { return fTimers[i]; }
    GPUhd() void SetTimer( int i, double v ) { fTimers[i] = v; }

    GPUhd() int NHitsTotal() const { return fData.NumberOfHits(); }

   void SetHitLinkUpData( const AliHLTTPCCARow &row, int hitIndex, short v ) { fData.SetHitLinkUpData( row, hitIndex, v ); }
     void SetHitLinkDownData( const AliHLTTPCCARow &row, int hitIndex, short v ) { fData.SetHitLinkDownData( row, hitIndex, v ); }
     short HitLinkUpData( const AliHLTTPCCARow &row, int hitIndex ) const { return fData.HitLinkUpData( row, hitIndex ); }
    short HitLinkDownData( const AliHLTTPCCARow &row, int hitIndex ) const { return fData.HitLinkDownData( row, hitIndex ); }

     int FirstHitInBin( const AliHLTTPCCARow &row, int binIndex ) const { return fData.FirstHitInBin( row, binIndex ); }

     unsigned short HitDataY( const AliHLTTPCCARow &row, int hitIndex ) const {
      return fData.HitDataY( row, hitIndex );
    }
     unsigned short HitDataZ( const AliHLTTPCCARow &row, int hitIndex ) const {
      return fData.HitDataZ( row, hitIndex );
    }

    int HitInputID( const AliHLTTPCCARow &row, int hitIndex ) const { return fData.ClusterDataIndex( row, hitIndex ); }

    /**
     * The hit weight is used to determine whether a hit belongs to a certain tracklet or another one
     * competing for the same hit. The tracklet that has a higher weight wins. Comparison is done
     * using the the number of hits in the tracklet (the more hits it has the more it keeps). If
     * tracklets have the same number of hits then it doesn't matter who gets it, but it should be
     * only one. So a unique number (row index is good) is added in the least significant part of
     * the weight
     */
    static int CalculateHitWeight( int NHits, int unique ) {
      return ( NHits << 16 ) + unique;
    }
     void MaximizeHitWeight( const AliHLTTPCCARow &row, int hitIndex, int weight ) {
      fData.MaximizeHitWeight( row, hitIndex, weight );
    }
    int HitWeight( const AliHLTTPCCARow &row, int hitIndex ) const {
      return fData.HitWeight( row, hitIndex );
    }

  GPUhd() int NTracklets() const { return *fNTracklets; }
    GPUhd() int  *NTracklets() { return fNTracklets; }

    GPUhd() const AliHLTTPCCAHitId &TrackletStartHit( int i ) const { return fTrackletStartHits[i]; }
    GPUhd() AliHLTTPCCAHitId *TrackletStartHits() const { return fTrackletStartHits; }
    GPUhd() const AliHLTTPCCATracklet &Tracklet( int i ) const { return fTracklets[i]; }
    GPUhd() AliHLTTPCCATracklet  *Tracklets() const { return fTracklets;}

    GPUhd() int *NTracks()  const { return fNTracks; }
    GPUhd() AliHLTTPCCATrack *Tracks() const { return  fTracks; }
    GPUhd() int *NTrackHits()  const { return fNTrackHits; }
    GPUhd() AliHLTTPCCAHitId *TrackHits() const { return fTrackHits; }

    GPUhd() const AliHLTTPCCASliceOutput * Output() const { return fOutput; }

    GPUhd()  int *NOutTracks() const { return  fNOutTracks; }
    GPUhd()  AliHLTTPCCAOutTrack *OutTracks() const { return  fOutTracks; }
    GPUhd()  const AliHLTTPCCAOutTrack &OutTrack( int index ) const { return fOutTracks[index]; }
    GPUhd()  int *NOutTrackHits() const { return  fNOutTrackHits; }
    GPUhd()  int *OutTrackHits() const { return  fOutTrackHits; }
    GPUhd()  int OutTrackHit( int i ) const { return  fOutTrackHits[i]; }


  private:
    void SetupCommonMemory();

    AliHLTTPCCAParam fParam; // parameters
    double fTimers[10]; // running CPU time for different parts of the algorithm

    /** A pointer to the ClusterData object that the SliceData was created from. This can be used to
     * merge clusters from inside the SliceTracker code and recreate the SliceData. */
    AliHLTTPCCAClusterData *fClusterData; // ^
    AliHLTTPCCASliceData fData; // The SliceData object. It is used to encapsulate the storage in memory from the access

    // event

    char *fCommonMemory; // common event memory
    int   fCommonMemorySize; // size of the event memory [bytes]

    char *fHitMemory; // event memory for hits
    int   fHitMemorySize; // size of the event memory [bytes]

    char *fTrackMemory; // event memory for tracks
    int   fTrackMemorySize; // size of the event memory [bytes]


    int *fNTracklets;     // number of tracklets
    AliHLTTPCCAHitId *fTrackletStartHits;   // start hits for the tracklets
    AliHLTTPCCATracklet *fTracklets; // tracklets

    //
    int *fNTracks;            // number of reconstructed tracks
    AliHLTTPCCATrack *fTracks;  // reconstructed tracks
    int *fNTrackHits;           // number of track hits
    AliHLTTPCCAHitId *fTrackHits;          // array of track hit numbers

    // output

    AliHLTTPCCASliceOutput *fOutput;

    // obsolete output

    int *fNOutTracks; // number of tracks in fOutTracks array
    AliHLTTPCCAOutTrack *fOutTracks; // output array of the reconstructed tracks
    int *fNOutTrackHits;  // number of hits in fOutTrackHits array
    int *fOutTrackHits;  // output array of ID's of the reconstructed hits

    // disable copy
    AliHLTTPCCATracker( const AliHLTTPCCATracker& );
    AliHLTTPCCATracker &operator=( const AliHLTTPCCATracker& );
};


#endif
