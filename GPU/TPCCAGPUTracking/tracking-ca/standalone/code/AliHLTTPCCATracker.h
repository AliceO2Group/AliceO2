//-*- Mode: C++ -*-
// @(#) $Id: AliHLTTPCCATracker.h 35348 2009-10-08 12:04:48Z sgorbuno $
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
#include "AliHLTTPCCASliceOutput.h"
#include "AliHLTTPCCATrackletConstructor.h"

class AliHLTTPCCATrack;
class AliHLTTPCCATrackParam;
class AliHLTTPCCAClusterData;

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
  public:

	AliHLTTPCCATracker()
		: fParam(),
		fOutputControl(),
		fClusterData( 0 ),
		fData(),
		fIsGPUTracker( false ),
		fGPUDebugLevel( 0 ),
		fGPUDebugOut( 0 ),
		fRowStartHitCountOffset( NULL ),
		fTrackletTmpStartHits( NULL ),
		fGPUTrackletTemp( NULL ),
		fRowBlockTracklets( NULL ),
		fRowBlockPos( NULL ),
		fBlockStartingTracklet( NULL ),
		fGPUParametersConst(),
		fCommonMem( 0 ),
		fHitMemory( 0 ),
		fHitMemorySize( 0 ),
		fTrackletMemory( 0 ),
		fTrackletMemorySize( 0 ),
		fTrackMemory( 0 ),
		fTrackMemorySize( 0 ),
		fTrackletStartHits( 0 ),
		fTracklets( 0 ),
		fTrackletRowHits( NULL ),
		fTracks( 0 ),
		fTrackHits( 0 ),
		fOutput( 0 )
	{
	  // constructor
	}
    ~AliHLTTPCCATracker();

	struct StructGPUParameters
	{
		StructGPUParameters() : fScheduleFirstDynamicTracklet( 0 ), fGPUError( 0 ) {}
		int fScheduleFirstDynamicTracklet;		//Last Tracklet with fixed position in sheduling
		int fGPUError;							//Signalizes error on GPU during GPU Reconstruction, kind of return value
	};

	struct StructGPUParametersConst
	{
		StructGPUParametersConst() : fGPUFixedBlockCount( 0 ), fGPUiSlice( 0 ), fGPUnSlices( 0 ) {}
		int fGPUFixedBlockCount;				//Count of blocks that is used for this tracker in fixed schedule situations
		int fGPUiSlice;
		int fGPUnSlices;
	};

	struct commonMemoryStruct
	{
		commonMemoryStruct() : fNTracklets( 0 ), fNTracks( 0 ), fNTrackHits( 0 ), fGPUParameters() {}
	    int fNTracklets;     // number of tracklets
	    int fNTracks;            // number of reconstructed tracks
	    int fNTrackHits;           // number of track hits
		StructGPUParameters fGPUParameters;
	};

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
	char* SetGPUTrackerHitsMemory(char* pGPUMemory, int MaxNHits);
	char* SetGPUTrackerTrackletsMemory(char* pGPUMemory, int MaxNTracklets);
	char* SetGPUTrackerTracksMemory(char* pGPUMemory, int MaxNTracks, int MaxNHits );

	//Debugging Stuff
	void DumpSliceData(std::ostream &out);	//Dump Input Slice Data
	void DumpLinks(std::ostream &out);		//Dump all links to file (for comparison after NeighboursFinder/Cleaner)
	void DumpStartHits(std::ostream &out);	//Same for Start Hits
	void DumpHitWeights(std::ostream &out); //....
	void DumpTrackHits(std::ostream &out);	//Same for Track Hits
	void DumpTrackletHits(std::ostream &out);	//Same for Track Hits

    GPUd() void GetErrors2( int iRow,  const AliHLTTPCCATrackParam &t, float &Err2Y, float &Err2Z ) const;
    GPUd() void GetErrors2( int iRow, float z, float sinPhi, float cosPhi, float DzDs, float &Err2Y, float &Err2Z ) const;

    void FitTrack( const AliHLTTPCCATrack &track, float *t0 = 0 ) const;
    void FitTrackFull( const AliHLTTPCCATrack &track, float *t0 = 0 ) const;

    void SetupCommonMemory();
    void SetPointersHits( int MaxNHits );
	void SetPointersTracklets ( int MaxNTracklets );
    void SetPointersTracks( int MaxNTracks, int MaxNHits );

	void SetOutput( AliHLTTPCCASliceOutput** out ) { fOutput = out; }

    void ReadEvent( AliHLTTPCCAClusterData *clusterData );

#if !defined(HLTCA_GPUCODE)
    GPUh() void WriteEvent( std::ostream &out );
    GPUh() void WriteTracks( std::ostream &out ) ;
    GPUh() void ReadTracks( std::istream &in );
#endif

    GPUhd() const AliHLTTPCCAParam &Param() const { return fParam; }
    GPUhd() void SetParam( const AliHLTTPCCAParam &v ) { fParam = v; }

	GPUhd() const AliHLTTPCCASliceOutput::outputControlStruct* OutputControl() const { return fOutputControl; }
	GPUh() void SetOutputControl( AliHLTTPCCASliceOutput::outputControlStruct* val)	{ fOutputControl = val;	}

    GPUhd() AliHLTTPCCAClusterData *ClusterData() const { return fClusterData; }
    GPUhd() const AliHLTTPCCASliceData &Data() const { return fData; }
	GPUhd() AliHLTTPCCASliceData *pData() {return &fData; }

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

	GPUd() const ushort2 *HitData( const AliHLTTPCCARow &row ) const { return fData.HitData(row); }
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
    GPUd() ushort2 HitData( const AliHLTTPCCARow &row, int hitIndex ) const {
      return fData.HitData( row, hitIndex );
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

    GPUhd() int NTracklets() const { return fCommonMem->fNTracklets; }
    GPUhd() int  *NTracklets() { return &fCommonMem->fNTracklets; }

    GPUhd() const AliHLTTPCCAHitId &TrackletStartHit( int i ) const { return fTrackletStartHits[i]; }
    GPUhd() AliHLTTPCCAHitId *TrackletStartHits() const { return fTrackletStartHits; }
    GPUhd() AliHLTTPCCAHitId *TrackletTmpStartHits() const { return fTrackletTmpStartHits; }
    GPUhd() const AliHLTTPCCATracklet &Tracklet( int i ) const { return fTracklets[i]; }
    GPUhd() AliHLTTPCCATracklet  *Tracklets() const { return fTracklets;}
	GPUhd() int* TrackletRowHits() { return fTrackletRowHits; }

    GPUhd() int *NTracks()  const { return &fCommonMem->fNTracks; }
    GPUhd() AliHLTTPCCATrack *Tracks() const { return  fTracks; }
    GPUhd() int *NTrackHits()  const { return &fCommonMem->fNTrackHits; }
    GPUhd() AliHLTTPCCAHitId *TrackHits() const { return fTrackHits; }

    GPUhd() AliHLTTPCCASliceOutput** Output() const { return fOutput; }

	GPUh() commonMemoryStruct *CommonMemory() {return(fCommonMem); }
	static GPUh() size_t CommonMemorySize() { return(sizeof(AliHLTTPCCATracker::commonMemoryStruct)); }
	GPUh() char* &HitMemory() {return(fHitMemory); }
	GPUh() size_t HitMemorySize() const {return(fHitMemorySize); }
	GPUh() char* &TrackletMemory() {return(fTrackletMemory); }
	GPUh() size_t TrackletMemorySize() const {return(fTrackletMemorySize); }
	GPUh() char* &TrackMemory() {return(fTrackMemory); }
	GPUh() size_t TrackMemorySize() const {return(fTrackMemorySize); }
	GPUhd() AliHLTTPCCARow* SliceDataRows() {return(fData.Rows()); }

	GPUhd() uint3* RowStartHitCountOffset() const {return(fRowStartHitCountOffset);}
	GPUhd() AliHLTTPCCATrackletConstructor::AliHLTTPCCAGPUTempMemory* GPUTrackletTemp() const {return(fGPUTrackletTemp);}
	GPUhd() int* RowBlockTracklets(int reverse, int iRowBlock) const {return(&fRowBlockTracklets[(reverse * ((fParam.NRows() / HLTCA_GPU_SCHED_ROW_STEP) + 1) + iRowBlock) * fCommonMem->fNTracklets]);}
	GPUhd() int* RowBlockTracklets() const {return(fRowBlockTracklets);}
	GPUhd() int4* RowBlockPos(int reverse, int iRowBlock) const {return(&fRowBlockPos[reverse * ((fParam.NRows() / HLTCA_GPU_SCHED_ROW_STEP) + 1) + iRowBlock]);}
	GPUhd() int4* RowBlockPos() const {return(fRowBlockPos);}
	GPUhd() uint2* BlockStartingTracklet() const {return(fBlockStartingTracklet);}
	GPUhd() StructGPUParameters* GPUParameters() const {return(&fCommonMem->fGPUParameters);}
	GPUhd() StructGPUParametersConst* GPUParametersConst() {return(&fGPUParametersConst);}
	
	GPUh() unsigned long long int* PerfTimer(unsigned int i) {return &fPerfTimers[i]; }

#ifdef HLTCA_GPU_TRACKLET_CONSTRUCTOR_DO_PROFILE
	char* fStageAtSync;				//Pointer to array storing current stage for every thread at every sync point
	int* fThreadTimes;
#endif

  private:
    AliHLTTPCCAParam fParam; // parameters
	double fTimers[10];
    unsigned long long int fPerfTimers[16]; // running CPU time for different parts of the algorithm
	void StandalonePerfTime(int i);

	AliHLTTPCCASliceOutput::outputControlStruct* fOutputControl;

    /** A pointer to the ClusterData object that the SliceData was created from. This can be used to
     * merge clusters from inside the SliceTracker code and recreate the SliceData. */
    AliHLTTPCCAClusterData *fClusterData; // ^
    AliHLTTPCCASliceData fData; // The SliceData object. It is used to encapsulate the storage in memory from the access

	bool fIsGPUTracker; // is it GPU tracker object
	int fGPUDebugLevel; // debug level
	std::ostream *fGPUDebugOut; // debug stream

	//GPU Temp Arrays
	uint3* fRowStartHitCountOffset;				//Offset, length and new offset of start hits in row
	AliHLTTPCCAHitId *fTrackletTmpStartHits;	//Unsorted start hits
	AliHLTTPCCATrackletConstructor::AliHLTTPCCAGPUTempMemory *fGPUTrackletTemp;	//Temp Memory for GPU Tracklet Constructor
	int* fRowBlockTracklets;					//Reference which tracklet is processed in which rowblock next
	int4* fRowBlockPos;							//x is last tracklet to be processed, y is last tracklet already processed, z is last tracklet to be processed in next iteration, w is initial x value to check if tracklet must be initialized
	uint2* fBlockStartingTracklet;

	StructGPUParametersConst fGPUParametersConst;

	// event

	commonMemoryStruct *fCommonMem; // common event memory

    char *fHitMemory; // event memory for hits
    size_t   fHitMemorySize; // size of the event memory [bytes]

	char *fTrackletMemory;
	size_t fTrackletMemorySize;

    char *fTrackMemory; // event memory for tracks
    size_t   fTrackMemorySize; // size of the event memory [bytes]

    AliHLTTPCCAHitId *fTrackletStartHits;   // start hits for the tracklets
    AliHLTTPCCATracklet *fTracklets; // tracklets
	int *fTrackletRowHits;

    //
    AliHLTTPCCATrack *fTracks;  // reconstructed tracks
    AliHLTTPCCAHitId *fTrackHits;          // array of track hit numbers

    // output

    AliHLTTPCCASliceOutput **fOutput;

    // disable copy
    AliHLTTPCCATracker( const AliHLTTPCCATracker& );
    AliHLTTPCCATracker &operator=( const AliHLTTPCCATracker& );

	static int starthitSortComparison(const void*a, const void* b);
};


#endif
