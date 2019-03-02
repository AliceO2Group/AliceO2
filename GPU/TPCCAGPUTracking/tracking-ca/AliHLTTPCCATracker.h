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
#include "AliHLTTPCCAGPUConfig.h"

#if !defined(__OPENCL__) || defined(HLTCA_HOSTCODE)
#include <iostream>
#endif

#include "AliHLTTPCCAParam.h"
#include "AliHLTTPCCAHitId.h"
#include "AliHLTTPCCASliceData.h"
#include "AliHLTTPCCASliceOutput.h"
#include "AliHLTTPCCATrackletConstructor.h"
#include "AliHLTTPCCATracklet.h"

MEM_CLASS_PRE() class AliHLTTPCCATrack;
MEM_CLASS_PRE() class AliHLTTPCCATrackParam;
class AliHLTTPCCAClusterData;
MEM_CLASS_PRE() class AliHLTTPCCARow;

#ifdef HLTCA_STANDALONE
#ifdef HLTCA_GPUCODE
#define GPUCODE
#endif
#include "../cmodules/timer.h"
#ifdef HLTCA_GPUCODE
#undef GPUCODE
#endif
#endif

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

MEM_CLASS_PRE() class AliHLTTPCCATracker
{
  public:

  AliHLTTPCCATracker()
    :
#ifdef HLTCA_STANDALONE
#ifdef HLTCA_GPU_TRACKLET_CONSTRUCTOR_DO_PROFILE
      fStageAtSync( NULL ),
#endif
      fLinkTmpMemory( NULL ),
#endif
      fParam(),
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
  }
  ~AliHLTTPCCATracker();
  
  struct StructGPUParameters
  {
    StructGPUParameters() : fNextTracklet(0), fScheduleFirstDynamicTracklet( 0 ), fGPUError( 0 ) {}
    int fNextTracklet;						//Next Tracklet to process
    int fScheduleFirstDynamicTracklet;		//Last Tracklet with fixed position in sheduling
    int fGPUError;							//Signalizes error on GPU during GPU Reconstruction, kind of return value
  };
  
  MEM_CLASS_PRE2() struct StructGPUParametersConst
  {
    StructGPUParametersConst() : fGPUFixedBlockCount( 0 ), fGPUiSlice( 0 ), fGPUnSlices( 0 ), fGPUMem( NULL ) {}
    int fGPUFixedBlockCount;				//Count of blocks that is used for this tracker in fixed schedule situations
    int fGPUiSlice;							// slice number processed by running GPU MP
    int fGPUnSlices;						// n of slices to be processed in parallel
    GPUglobalref() char* fGPUMem;			//Base pointer to GPU memory (Needed for OpenCL for verification)
  };
  
  struct commonMemoryStruct
  {
    commonMemoryStruct() : fNTracklets( 0 ), fNTracks( 0 ), fNLocalTracks( 0 ), fNTrackHits( 0 ), fNLocalTrackHits( 0 ), fGPUParameters() {}
    int fNTracklets;     // number of tracklets
    int fNTracks;            // number of reconstructed tracks
    int fNLocalTracks;	 //number of reconstructed tracks before global tracking
    int fNTrackHits;           // number of track hits
    int fNLocalTrackHits; //see above
    StructGPUParameters fGPUParameters; // GPU parameters
  };
  
  MEM_CLASS_PRE2() void Initialize( const MEM_LG2(AliHLTTPCCAParam) &param );
  
  void StartEvent();
  
  int CheckEmptySlice() const;
  void WriteOutputPrepare();
  void WriteOutput();
  
#if !defined(HLTCA_GPUCODE)
  void Reconstruct();
  void ReconstructOutput();
#endif //!HLTCA_GPUCODE
  void DoTracking();
  
  //Make Reconstruction steps directly callable (Used for GPU debugging)
  void RunNeighboursFinder();
  void RunNeighboursCleaner();
  void RunStartHitsFinder();
  void RunTrackletConstructor();
  void RunTrackletSelector();
  
  //GPU Tracker Interface
  void SetGPUTracker();
#if !defined(__OPENCL__) || defined(HLTCA_HOSTCODE)
  void SetGPUDebugLevel(int Level, std::ostream *NewDebugOut = NULL) {fGPUDebugLevel = Level;if (NewDebugOut) fGPUDebugOut = NewDebugOut;}
  char* SetGPUTrackerCommonMemory(char* const pGPUMemory);
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
  void DumpOutput(FILE* out);	//Similar for output

  void SetOutput( AliHLTTPCCASliceOutput** out ) { fOutput = out; }
  int ReadEvent( AliHLTTPCCAClusterData *clusterData );

  GPUhd() const AliHLTTPCCASliceOutput::outputControlStruct* OutputControl() const { return fOutputControl; }  
  GPUh() void SetOutputControl( AliHLTTPCCASliceOutput::outputControlStruct* const val)	{ fOutputControl = val;	}

  GPUhd() AliHLTTPCCAClusterData *ClusterData() const { return fClusterData; }

  GPUh() void ClearSliceDataHitWeights() {fData.ClearHitWeights();}
  GPUh() MakeType(const MEM_LG(AliHLTTPCCARow)&) Row( const AliHLTTPCCAHitId &HitId ) const { return fData.Row( HitId.RowIndex() ); }

  GPUhd() AliHLTTPCCASliceOutput** Output() const { return fOutput; }

  GPUh() GPUglobalref() commonMemoryStruct *CommonMemory() const {return(fCommonMem); }
  GPUh() static size_t CommonMemorySize() { return(sizeof(AliHLTTPCCATracker::commonMemoryStruct)); }
  GPUh() GPUglobalref() char* HitMemory() const {return(fHitMemory); }
  GPUh() size_t HitMemorySize() const {return(fHitMemorySize); }
  GPUh() char* TrackletMemory() {return(fTrackletMemory); }
  GPUh() size_t TrackletMemorySize() const {return(fTrackletMemorySize); }
  GPUh() char* TrackMemory() {return(fTrackMemory); }
  GPUh() size_t TrackMemorySize() const {return(fTrackMemorySize); }

  GPUh() void SetGPUSliceDataMemory(void* const pSliceMemory, void* const pRowMemory) { fData.SetGPUSliceDataMemory(pSliceMemory, pRowMemory); }

#endif  
  
  MEM_CLASS_PRE2() GPUd() void GetErrors2( int iRow,  const MEM_LG2(AliHLTTPCCATrackParam) &t, float &ErrY2, float &ErrZ2 ) const
  {
    //fParam.GetClusterErrors2( iRow, fParam.GetContinuousTracking() != 0. ? 125. : t.Z(), t.SinPhi(), t.DzDs(), ErrY2, ErrZ2 );
    fParam.GetClusterRMS2( iRow, fParam.GetContinuousTracking() != 0. ? 125. : t.Z(), t.SinPhi(), t.DzDs(), ErrY2, ErrZ2 );
  }
  GPUd() void GetErrors2( int iRow, float z, float sinPhi, float DzDs, float &ErrY2, float &ErrZ2 ) const
  {
    //fParam.GetClusterErrors2( iRow, fParam.GetContinuousTracking() != 0. ? 125. : z, sinPhi, DzDs, ErrY2, ErrZ2 );
    fParam.GetClusterRMS2( iRow, fParam.GetContinuousTracking() != 0. ? 125. : z, sinPhi, DzDs, ErrY2, ErrZ2 );
  }
  
  
  void SetupCommonMemory();
  void SetPointersHits( int MaxNHits );
  void SetPointersTracklets ( int MaxNTracklets );
  void SetPointersTracks( int MaxNTracks, int MaxNHits );
  size_t SetPointersSliceData(const AliHLTTPCCAClusterData *data, bool allocate = false) { return(fData.SetPointers(data, allocate)); }
 
#if !defined(HLTCA_GPUCODE)
  GPUh() void WriteEvent( std::ostream &out );
  GPUh() void WriteTracks( std::ostream &out ) ;
  GPUh() void ReadTracks( std::istream &in );
#endif //!HLTCA_GPUCODE
  
  GPUhd() MakeType(const MEM_LG(AliHLTTPCCAParam)&) Param() const { return fParam; }
  GPUhd() MakeType(const MEM_LG(AliHLTTPCCAParam)*) pParam() const { return &fParam; }
  MEM_CLASS_PRE2() GPUhd() void SetParam( const MEM_LG2(AliHLTTPCCAParam) &v ) { fParam = v; }
  
  GPUhd() MakeType(const MEM_LG(AliHLTTPCCASliceData)&) Data() const { return fData; }
  
  GPUhd() GPUglobalref() const MEM_GLOBAL(AliHLTTPCCARow)& Row( int rowIndex ) const { return fData.Row( rowIndex ); }
  
  GPUhd() int NHitsTotal() const { return fData.NumberOfHits(); }
  
  MEM_TEMPLATE() GPUd() void SetHitLinkUpData( const MEM_TYPE( AliHLTTPCCARow)&row, int hitIndex, calink v ) { fData.SetHitLinkUpData( row, hitIndex, v ); }
  MEM_TEMPLATE() GPUd() void SetHitLinkDownData( const MEM_TYPE( AliHLTTPCCARow)&row, int hitIndex, calink v ) { fData.SetHitLinkDownData( row, hitIndex, v ); }
  MEM_TEMPLATE() GPUd() calink HitLinkUpData( const MEM_TYPE( AliHLTTPCCARow)&row, int hitIndex ) const { return fData.HitLinkUpData( row, hitIndex ); }
  MEM_TEMPLATE() GPUd() calink HitLinkDownData( const MEM_TYPE( AliHLTTPCCARow)&row, int hitIndex ) const { return fData.HitLinkDownData( row, hitIndex ); }
  
  MEM_TEMPLATE() GPUd() GPUglobalref() const cahit2 *HitData( const MEM_TYPE( AliHLTTPCCARow)& row ) const { return fData.HitData(row); }
  MEM_TEMPLATE() GPUd() GPUglobalref() const calink *HitLinkUpData  ( const MEM_TYPE( AliHLTTPCCARow)&row ) const { return fData.HitLinkUpData(row); }
  MEM_TEMPLATE() GPUd() GPUglobalref() const calink *HitLinkDownData( const MEM_TYPE( AliHLTTPCCARow)&row ) const { return fData.HitLinkDownData(row); }
  MEM_TEMPLATE() GPUd() GPUglobalref() const calink *FirstHitInBin( const MEM_TYPE( AliHLTTPCCARow)&row ) const { return fData.FirstHitInBin(row); }
  
  MEM_TEMPLATE() GPUd() int FirstHitInBin( const MEM_TYPE( AliHLTTPCCARow)&row, int binIndex ) const { return fData.FirstHitInBin( row, binIndex ); }
  
  MEM_TEMPLATE() GPUd() cahit HitDataY( const MEM_TYPE( AliHLTTPCCARow)&row, int hitIndex ) const {
    return fData.HitDataY( row, hitIndex );
  }
  MEM_TEMPLATE() GPUd() cahit HitDataZ( const MEM_TYPE( AliHLTTPCCARow)&row, int hitIndex ) const {
    return fData.HitDataZ( row, hitIndex );
  }
  MEM_TEMPLATE() GPUd() cahit2 HitData( const MEM_TYPE( AliHLTTPCCARow)&row, int hitIndex ) const {
    return fData.HitData( row, hitIndex );
  }
  
  MEM_TEMPLATE() GPUhd() int HitInputID( const MEM_TYPE( AliHLTTPCCARow)&row, int hitIndex ) const { return fData.ClusterDataIndex( row, hitIndex ); }
  
  /**
   * The hit weight is used to determine whether a hit belongs to a certain tracklet or another one
   * competing for the same hit. The tracklet that has a higher weight wins. Comparison is done
   * using the the number of hits in the tracklet (the more hits it has the more it keeps). If
   * tracklets have the same number of hits then it doesn't matter who gets it, but it should be
   * only one. So a unique number (row index is good) is added in the least significant part of
   * the weight
   */
  GPUd() static int CalculateHitWeight( int NHits, float chi2, int ) {
    const float chi2_suppress = 6.f;
    float weight = (((float) NHits * (chi2_suppress - chi2 / 500.f)) * (1e9 / chi2_suppress / 160.));
    if (weight < 0. || weight > 2e9) return 0;
    return ( (int) weight );
    //return( (NHits << 16) + num);
  }
  MEM_TEMPLATE() GPUd() void MaximizeHitWeight( const MEM_TYPE( AliHLTTPCCARow)&row, int hitIndex, int weight ) {
    fData.MaximizeHitWeight( row, hitIndex, weight );
  }
  MEM_TEMPLATE() GPUd() void SetHitWeight( const MEM_TYPE( AliHLTTPCCARow)&row, int hitIndex, int weight ) {
	fData.SetHitWeight( row, hitIndex, weight );
  }
  MEM_TEMPLATE() GPUd() int HitWeight( const MEM_TYPE( AliHLTTPCCARow)&row, int hitIndex ) const {
    return fData.HitWeight( row, hitIndex );
  }
  
  GPUhd() GPUglobalref() int *NTracklets() const { return &fCommonMem->fNTracklets; }
  
  GPUhd() const AliHLTTPCCAHitId &TrackletStartHit( int i ) const { return fTrackletStartHits[i]; }
  GPUhd() GPUglobalref() AliHLTTPCCAHitId *TrackletStartHits() const { return fTrackletStartHits; }
  GPUhd() GPUglobalref() AliHLTTPCCAHitId *TrackletTmpStartHits() const { return fTrackletTmpStartHits; }
  MEM_CLASS_PRE2() GPUhd() const MEM_LG2(AliHLTTPCCATracklet) &Tracklet( int i ) const { return fTracklets[i]; }
  GPUhd() GPUglobalref() MEM_GLOBAL(AliHLTTPCCATracklet) *Tracklets() const { return fTracklets;}
  GPUhd() GPUglobalref() calink* TrackletRowHits() const { return fTrackletRowHits; }

  GPUhd() GPUglobalref() int *NTracks()  const { return &fCommonMem->fNTracks; }
  GPUhd() GPUglobalref() MEM_GLOBAL(AliHLTTPCCATrack) *Tracks() const { return fTracks; }
  GPUhd() GPUglobalref() int *NTrackHits()  const { return &fCommonMem->fNTrackHits; }
  GPUhd() GPUglobalref() AliHLTTPCCAHitId *TrackHits() const { return fTrackHits; }
  
 
  GPUhd() GPUglobalref() MEM_GLOBAL(AliHLTTPCCARow)* SliceDataRows() const {return(fData.Rows()); }
  
  GPUhd() GPUglobalref() uint3* RowStartHitCountOffset() const {return(fRowStartHitCountOffset);}
  GPUhd() GPUglobalref() StructGPUParameters* GPUParameters() const {return(&fCommonMem->fGPUParameters);}
  GPUhd() MakeType(MEM_LG(StructGPUParametersConst)*) GPUParametersConst() {return(&fGPUParametersConst);}
  GPUhd() void SetGPUTextureBase(char* val) { fData.SetGPUTextureBase(val); }

  struct trackSortData
  {
	int fTtrack;		//Track ID
	float fSortVal;		//Value to sort for
  };

  void PerformGlobalTracking(AliHLTTPCCATracker& sliceLeft, AliHLTTPCCATracker& sliceRight, int MaxTracks);

#ifdef HLTCA_STANDALONE  
  void StartTimer(int i) {if (fGPUDebugLevel) fTimers[i].Start();}
  void StopTimer(int i) {if (fGPUDebugLevel) fTimers[i].Stop();}
  double GetTimer(int i) {return fTimers[i].GetElapsedTime();}
  void ResetTimer(int i) {fTimers[i].Reset();}
#else
  void StartTimer(int i) {}
  void StopTimer(int i) {}
  double GetTimer(int i) {return 0;}
  void ResetTimer(int i) {}
#endif

private:
#if !defined(__OPENCL__) || defined(HLTCA_HOSTCODE)
  GPUh() int PerformGlobalTrackingRun(AliHLTTPCCATracker& sliceNeighbour, int iTrack, int rowIndex, float angle, int direction);
#endif

	//Temporary Variables for Standalone measurements
#ifdef HLTCA_STANDALONE
public:
#ifdef  HLTCA_GPU_TRACKLET_CONSTRUCTOR_DO_PROFILE
  char* fStageAtSync;				//Pointer to array storing current stage for every thread at every sync point
#endif
  char *fLinkTmpMemory;				//tmp memory for hits after neighbours finder
private:
#endif
  
  MEM_LG(AliHLTTPCCAParam) fParam; // parameters
#ifdef HLTCA_STANDALONE
  HighResTimer fTimers[10];
#endif
  
  AliHLTTPCCASliceOutput::outputControlStruct* fOutputControl; // output control
  
  /** A pointer to the ClusterData object that the SliceData was created from. This can be used to
   * merge clusters from inside the SliceTracker code and recreate the SliceData. */
  GPUglobalref() AliHLTTPCCAClusterData *fClusterData; // ^
  MEM_LG(AliHLTTPCCASliceData) fData; // The SliceData object. It is used to encapsulate the storage in memory from the access
  
  char fIsGPUTracker; // is it GPU tracker object
  int fGPUDebugLevel; // debug level

#if !defined(__OPENCL__) || defined(HLTCA_HOSTCODE)
  std::ostream *fGPUDebugOut; // debug stream
#else
  void* fGPUDebugOut; //No this is a hack, but I have no better idea.
#endif
  
  //GPU Temp Arrays
  GPUglobalref() uint3* fRowStartHitCountOffset;				//Offset, length and new offset of start hits in row
  GPUglobalref() AliHLTTPCCAHitId *fTrackletTmpStartHits;	//Unsorted start hits
  GPUglobalref() char* fGPUTrackletTemp;					//Temp Memory for GPU Tracklet Constructor
  GPUglobalref() int* fRowBlockTracklets;					//Reference which tracklet is processed in which rowblock next
  GPUglobalref() int4* fRowBlockPos;							//x is last tracklet to be processed, y is last tracklet already processed, z is last tracklet to be processed in next iteration, w is initial x value to check if tracklet must be initialized  
  GPUglobalref() uint2* fBlockStartingTracklet;			// First Tracklet that is to be processed by current GPU MP

  MEM_LG(StructGPUParametersConst) fGPUParametersConst; // Parameters for GPU if this is a GPU tracker

  // event
  
  GPUglobalref() commonMemoryStruct *fCommonMem; // common event memory
  
  GPUglobalref() char *fHitMemory; // event memory for hits
  size_t   fHitMemorySize; // size of the event memory for hits [bytes]

  GPUglobalref() char *fTrackletMemory;	//event memory for tracklets
  size_t fTrackletMemorySize; //size of the event memory for tracklets

  GPUglobalref() char *fTrackMemory; // event memory for tracks
  size_t   fTrackMemorySize; // size of the event memory for tracks [bytes]

  GPUglobalref() AliHLTTPCCAHitId *fTrackletStartHits;   // start hits for the tracklets
  GPUglobalref() MEM_GLOBAL(AliHLTTPCCATracklet) *fTracklets; // tracklets
  GPUglobalref() calink *fTrackletRowHits;			//Hits for each Tracklet in each row

  //
  GPUglobalref() MEM_GLOBAL(AliHLTTPCCATrack) *fTracks;  // reconstructed tracks
  GPUglobalref() AliHLTTPCCAHitId *fTrackHits;          // array of track hit numbers
  
  // output
  
  GPUglobalref() AliHLTTPCCASliceOutput **fOutput;		//address of pointer pointing to SliceOutput Object
  
  // disable copy
  AliHLTTPCCATracker( const AliHLTTPCCATracker& );
  AliHLTTPCCATracker &operator=( const AliHLTTPCCATracker& );
  
  static int StarthitSortComparison(const void*a, const void* b);
};

#endif //ALIHLTTPCCATRACKER_H
