//-*- Mode: C++ -*-
// @(#) $Id: AliHLTTPCCATracker.h 33907 2009-07-23 13:52:49Z sgorbuno $
// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************

#ifndef ALIHLTTPCCATRACKER_H
#define ALIHLTTPCCATRACKER_H


#include "AliHLTTPCCADef.h"
#include "AliHLTTPCCAGPUConfig.h"
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

class AliHLTTPCCAClusterData;

class AliHLTTPCCATracker
{
	//friend class AliHLTTPCCAGPUTracker;
  public:

	AliHLTTPCCATracker()
		:
		fParam(),
		fClusterData( 0 ),
		fData(),
		fIsGPUTracker( false ),
		fGPUDebugLevel( 0 ),
		fGPUDebugOut( 0 ),
		fCommonMemory( 0 ),
		fCommonMemorySize( 0 ),
		fHitMemory( 0 ),
		fHitMemorySize( 0 ),
		fTrackMemory( 0 ),
		fTrackMemorySize( 0 ),
		fNTracklets( 0 ),
		fTrackletStartHits( 0 ),
		fTracklets( 0 ),
		fNTracks( 0 ),
		fTracks( 0 ),
		fNTrackHits( 0 ),
		fTrackHits( 0 ),
		fOutput( 0 ),
		fNOutTracks( 0 ),
		fOutTracks( 0 ),
		fNOutTrackHits( 0 ),
		fOutTrackHits( 0 )
	{
	  // constructor
	}
    GPUd() ~AliHLTTPCCATracker();

    void Initialize( const AliHLTTPCCAParam &param );

    void StartEvent();

	int CheckEmptySlice();
	void WriteOutput();

#if !defined(HLTCA_GPUCODE)
    void Reconstruct();
#endif

	//Make Reconstruction steps directly callable (Used for GPU debugging)
	void RunNeighboursFinder();
	void RunNeighboursCleaner();
	void RunStartHitsFinder();
	void RunTrackletConstructor();
	void RunTrackletSelector();

	//GPU Tracker Interface
	void SetGPUTracker();
	void SetGPUDebugLevel(int Level, std::ostream *NewDebugOut = NULL) {fGPUDebugLevel = Level;if (NewDebugOut) fGPUDebugOut = NewDebugOut;}

	char* SetGPUTrackerCommonMemory(char* pGPUMemory);
	char* SetGPUTrackerHitsMemory(char* pGPUMemory, int MaxNHits );
	char* SetGPUTrackerTracksMemory(char* pGPUMemory, int MaxNTracks, int MaxNHits );

	char* SetGPUSliceDataMemory(char* pGPUMemory, const AliHLTTPCCAClusterData *data) {return(fData.SetGPUSliceDataMemory(pGPUMemory, data));}

	//Debugging Stuff
	void DumpLinks(std::ostream &out);		//Dump all links to file (for comparison after NeighboursFinder/Cleaner)
	void DumpStartHits(std::ostream &out);	//Same for Start Hits
	void DumpTrackHits(std::ostream &out);	//Same for Track Hits
	void DumpTrackletHits(std::ostream &out);	//Same for Track Hits

    GPUd() void GetErrors2( int iRow,  const AliHLTTPCCATrackParam &t, float &Err2Y, float &Err2Z ) const;
    GPUd() void GetErrors2( int iRow, float z, float sinPhi, float cosPhi, float DzDs, float &Err2Y, float &Err2Z ) const;

    void FitTrack( const AliHLTTPCCATrack &track, float *t0 = 0 ) const;
    void FitTrackFull( const AliHLTTPCCATrack &track, float *t0 = 0 ) const;

	void SetPointersCommon();
    void SetPointersHits( int MaxNHits );
    void SetPointersTracks( int MaxNTracks, int MaxNHits );

#if !defined(HLTCA_GPUCODE)
    void ReadEvent( AliHLTTPCCAClusterData *clusterData );

    GPUh() void WriteEvent( std::ostream &out );
    GPUh() void WriteTracks( std::ostream &out ) ;
    GPUh() void ReadTracks( std::istream &in );
#endif

    GPUhd() const AliHLTTPCCAParam &Param() const { return fParam; }
    GPUhd() void SetParam( const AliHLTTPCCAParam &v ) { fParam = v; }

    GPUhd() const AliHLTTPCCAClusterData *ClusterData() const { return fClusterData; }
    GPUhd() const AliHLTTPCCASliceData &Data() const { return fData; }

	GPUh() void ClearSliceDataHitWeights() {fData.ClearHitWeights();}

    GPUhd() const AliHLTTPCCARow &Row( int rowIndex ) const { return fData.Row( rowIndex ); }
    GPUh() const AliHLTTPCCARow &Row( const AliHLTTPCCAHitId &HitId ) const { return fData.Row( HitId.RowIndex() ); }

    GPUhd() double Timer( int i ) const { return fTimers[i]; }
    GPUhd() void SetTimer( int i, double v ) { fTimers[i] = v; }

    GPUhd() int NHitsTotal() const { return fData.NumberOfHits(); }

    GPUd() void SetHitLinkUpData( const AliHLTTPCCARow &row, int hitIndex, short v ) { fData.SetHitLinkUpData( row, hitIndex, v ); }
    GPUd() void SetHitLinkDownData( const AliHLTTPCCARow &row, int hitIndex, short v ) { fData.SetHitLinkDownData( row, hitIndex, v ); }
    GPUd() short HitLinkUpData( const AliHLTTPCCARow &row, int hitIndex ) const { return fData.HitLinkUpData( row, hitIndex ); }
    GPUd() short HitLinkDownData( const AliHLTTPCCARow &row, int hitIndex ) const { return fData.HitLinkDownData( row, hitIndex ); }

	GPUd() const ushort_v *HitDataY( const AliHLTTPCCARow &row ) const { return fData.HitDataY(row); }
	GPUd() const ushort_v *HitDataZ( const AliHLTTPCCARow &row ) const { return fData.HitDataZ(row); }
	GPUd() const short_v *HitLinkUpData  ( const AliHLTTPCCARow &row ) const { return fData.HitLinkUpData(row); }
	GPUd() const short_v *HitLinkDownData( const AliHLTTPCCARow &row ) const { return fData.HitLinkDownData(row); }
	GPUd() const ushort_v *FirstHitInBin( const AliHLTTPCCARow &row ) const { return fData.FirstHitInBin(row); }
	
	GPUd() int FirstHitInBin( const AliHLTTPCCARow &row, int binIndex ) const { return fData.FirstHitInBin( row, binIndex ); }

    GPUd() unsigned short HitDataY( const AliHLTTPCCARow &row, int hitIndex ) const {
      return fData.HitDataY( row, hitIndex );
    }
    GPUd() unsigned short HitDataZ( const AliHLTTPCCARow &row, int hitIndex ) const {
      return fData.HitDataZ( row, hitIndex );
    }

    GPUhd() int HitInputID( const AliHLTTPCCARow &row, int hitIndex ) const { return fData.ClusterDataIndex( row, hitIndex ); }

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
    GPUd() void MaximizeHitWeight( const AliHLTTPCCARow &row, int hitIndex, int weight ) {
      fData.MaximizeHitWeight( row, hitIndex, weight );
    }
    GPUd() int HitWeight( const AliHLTTPCCARow &row, int hitIndex ) const {
      return fData.HitWeight( row, hitIndex );
    }

    GPUhd() int NTracklets() const { return *fNTracklets; }
    GPUhd() int  *NTracklets() { return fNTracklets; }
	GPUhd() int  *NextTracklet() { return fNextTracklet; }

    GPUhd() const AliHLTTPCCAHitId &TrackletStartHit( int i ) const { return fTrackletStartHits[i]; }
    GPUhd() AliHLTTPCCAHitId *TrackletStartHits() const { return fTrackletStartHits; }
    GPUhd() AliHLTTPCCAHitId *TrackletTmpStartHits() const { return fTrackletTmpStartHits; }
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

	GPUh() char *CommonMemory() {return(fCommonMemory); }
	GPUh() size_t CommonMemorySize() const {return(fCommonMemorySize); }
	GPUh() char *HitMemory() {return(fHitMemory); }
	GPUh() size_t HitMemorySize() const {return(fHitMemorySize); }
	GPUh() char* &TrackMemory() {return(fTrackMemory); }
	GPUh() size_t TrackMemorySize() const {return(fTrackMemorySize); }
	GPUh() char *SliceDataMemory() {return(fData.Memory()); }
	GPUh() size_t SliceDataMemorySize() const {return(fData.MemorySize()); }
	GPUh() int* SliceDataHitWeights() {return(fData.HitWeights()); }

	GPUhd() uint2* RowStartHitCountOffset() {return(fRowStartHitCountOffset);}

	GPUh() unsigned long long int* PerfTimer(unsigned int i) {return &fPerfTimers[i]; }
	void StandaloneQueryTime(unsigned long long int *i);
	void StandaloneQueryFreq(unsigned long long int *i);

#ifdef HLTCA_GPU_TRACKLET_CONSTRUCTOR_DO_PROFILE
	int* fStageAtSync;				//Pointer to array storing current stage for every thread at every sync point
	int* fThreadTimes;
#endif

#ifndef CUDA_DEVICE_EMULATION
  private:
#endif

    void SetupCommonMemory();

    AliHLTTPCCAParam fParam; // parameters
	double fTimers[10];
    unsigned long long int fPerfTimers[10]; // running CPU time for different parts of the algorithm
	void StandalonePerfTime(int i);

    /** A pointer to the ClusterData object that the SliceData was created from. This can be used to
     * merge clusters from inside the SliceTracker code and recreate the SliceData. */
    AliHLTTPCCAClusterData *fClusterData; // ^
    AliHLTTPCCASliceData fData; // The SliceData object. It is used to encapsulate the storage in memory from the access

  //Will this tracker run on GPU?
  bool fIsGPUTracker; // is it GPU tracker
  int fGPUDebugLevel; // debug level
  std::ostream *fGPUDebugOut; // debug stream

    // event

    char *fCommonMemory; // common event memory
    size_t   fCommonMemorySize; // size of the event memory [bytes]

    char *fHitMemory; // event memory for hits
    size_t   fHitMemorySize; // size of the event memory [bytes]

    char *fTrackMemory; // event memory for tracks
    size_t   fTrackMemorySize; // size of the event memory [bytes]

	//GPU Temp Arrays
	uint2* fRowStartHitCountOffset;				//Offset and length of start hits in row
	AliHLTTPCCAHitId *fTrackletTmpStartHits;	//Unsorted start hits

	int *fNextTracklet;
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
