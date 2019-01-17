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


#include "AliGPUTPCDef.h"
#include "AliGPUTPCGPUConfig.h"

#if !defined(GPUCA_GPUCODE)
#include <iostream>
#endif

#include "AliGPUTPCHitId.h"
#include "AliGPUTPCSliceData.h"
#include "AliGPUTPCSliceOutput.h"
#include "AliGPUTPCTrackletConstructor.h"
#include "AliGPUTPCTracklet.h"
#include "AliGPUProcessor.h"

MEM_CLASS_PRE() class AliGPUCAParam;

MEM_CLASS_PRE() class AliGPUTPCTrack;
MEM_CLASS_PRE() class AliGPUTPCTrackParam;
class AliGPUTPCClusterData;
MEM_CLASS_PRE() class AliGPUTPCRow;

#ifdef GPUCA_GPUCODE
	#define GPUCODE
#endif
#include "cmodules/timer.h"
#ifdef GPUCA_GPUCODE
	#undef GPUCODE
#endif

class AliGPUTPCClusterData;

MEM_CLASS_PRE()
class AliGPUTPCTracker : public AliGPUProcessor
{
  public:
#ifndef __OPENCL__
	AliGPUTPCTracker();
	~AliGPUTPCTracker();
#endif

	struct StructGPUParameters
	{
		StructGPUParameters() : fNextTracklet(0), fScheduleFirstDynamicTracklet(0), fGPUError(0) {}
		int fNextTracklet;                 //Next Tracklet to process
		int fScheduleFirstDynamicTracklet; //Last Tracklet with fixed position in sheduling
		int fGPUError;                     //Signalizes error on GPU during GPU Reconstruction, kind of return value
	};

	MEM_CLASS_PRE2() struct StructGPUParametersConst
	{
		StructGPUParametersConst() : fGPUFixedBlockCount(0), fGPUiSlice(0), fGPUMem(NULL) {}
		int fGPUFixedBlockCount;      //Count of blocks that is used for this tracker in fixed schedule situations
		int fGPUiSlice;               // slice number processed by running GPU MP
		GPUglobalref() char *fGPUMem; //Base pointer to GPU memory (Needed for OpenCL for verification)
	};

	struct commonMemoryStruct
	{
		commonMemoryStruct() : fNTracklets(0), fNTracks(0), fNLocalTracks(0), fNTrackHits(0), fNLocalTrackHits(0), fGPUParameters() {}
		int fNTracklets;                    // number of tracklets
		int fNTracks;                       // number of reconstructed tracks
		int fNLocalTracks;                  //number of reconstructed tracks before global tracking
		int fNTrackHits;                    // number of track hits
		int fNLocalTrackHits;               //see above
		StructGPUParameters fGPUParameters; // GPU parameters
	};
  
	MEM_CLASS_PRE2() void Initialize( int iSlice );
	MEM_CLASS_PRE2() void InitializeRows( const MEM_LG2(AliGPUCAParam) *param ) { fData.InitializeRows(*param); }
  
	int CheckEmptySlice();
	void WriteOutputPrepare();
	void WriteOutput();
  
#if !defined(GPUCA_GPUCODE)
	void Reconstruct();
	void ReconstructOutput();
#endif //!GPUCA_GPUCODE
	void DoTracking();
  
	//Make Reconstruction steps directly callable (Used for GPU debugging)
	void RunNeighboursFinder();
	void RunNeighboursCleaner();
	void RunStartHitsFinder();
	void RunTrackletConstructor();
	void RunTrackletSelector();
  
	//GPU Tracker Interface
#if !defined(__OPENCL__)
	//Debugging Stuff
	void DumpSliceData(std::ostream &out);	//Dump Input Slice Data
	void DumpLinks(std::ostream &out);		//Dump all links to file (for comparison after NeighboursFinder/Cleaner)
	void DumpStartHits(std::ostream &out);	//Same for Start Hits
	void DumpHitWeights(std::ostream &out); //....
	void DumpTrackHits(std::ostream &out);	//Same for Track Hits
	void DumpTrackletHits(std::ostream &out);	//Same for Track Hits
	void DumpOutput(FILE* out);	//Similar for output

	void SetOutput( AliGPUTPCSliceOutput** out ) { fOutput = out; }
	int ReadEvent();

	GPUh() const AliGPUTPCClusterData *ClusterData() const { return fData.ClusterData(); }

	GPUh() void ClearSliceDataHitWeights() {fData.ClearHitWeights();}
	GPUh() MakeType(const MEM_LG(AliGPUTPCRow)&) Row( const AliGPUTPCHitId &HitId ) const { return fData.Row(HitId.RowIndex()); }

	GPUhd() AliGPUTPCSliceOutput** Output() const { return fOutput; }

	GPUh() GPUglobalref() commonMemoryStruct *CommonMemory() const {return(fCommonMem); }

#endif
  
	MEM_CLASS_PRE2() GPUd() void GetErrors2( int iRow,  const MEM_LG2(AliGPUTPCTrackParam) &t, float &ErrY2, float &ErrZ2 ) const
	{
		//mCAParam.GetClusterErrors2( iRow, mCAParam.GetContinuousTracking() != 0. ? 125. : t.Z(), t.SinPhi(), t.DzDs(), ErrY2, ErrZ2 );
		mCAParam->GetClusterRMS2( iRow, mCAParam->ContinuousTracking != 0. ? 125. : t.Z(), t.SinPhi(), t.DzDs(), ErrY2, ErrZ2 );
	}
	GPUd() void GetErrors2( int iRow, float z, float sinPhi, float DzDs, float &ErrY2, float &ErrZ2 ) const
	{
		//mCAParam.GetClusterErrors2( iRow, mCAParam.GetContinuousTracking() != 0. ? 125. : z, sinPhi, DzDs, ErrY2, ErrZ2 );
		mCAParam->GetClusterRMS2( iRow, mCAParam->ContinuousTracking != 0. ? 125. : z, sinPhi, DzDs, ErrY2, ErrZ2 );
	}
  
	void SetupCommonMemory();
	
	void* SetPointersScratch(void* mem);
	void* SetPointersScratchHost(void* mem);
	void* SetPointersCommon(void* mem);
	void* SetPointersTracklets(void* mem);
	void* SetPointersTracks(void* mem);
	void* SetPointersTrackHits(void* mem);
	void RegisterMemoryAllocation();
	
	short MemoryResScratch() {return mMemoryResScratch;}
	short MemoryResScratchHost() {return mMemoryResScratchHost;}
	short MemoryResCommon() {return mMemoryResCommon;}
	short MemoryResTracklets() {return mMemoryResTracklets;}
	short MemoryResTracks() {return mMemoryResTracks;}
	short MemoryResTrackHits() {return mMemoryResTrackHits;}

	void SetMaxData();
 
	GPUhd() MakeType(const MEM_LG(AliGPUCAParam)&) Param() const { return *mCAParam; }
	GPUhd() MakeType(const MEM_LG(AliGPUCAParam)*) pParam() const { return mCAParam; }
	GPUhd() int ISlice() const { return fISlice; }
  
	GPUhd() MakeType(const MEM_LG(AliGPUTPCSliceData)&) Data() const { return fData; }
	GPUhd() MakeType(MEM_LG(AliGPUTPCSliceData)&) Data() { return fData; }
  
	GPUhd() GPUglobalref() const MEM_GLOBAL(AliGPUTPCRow)& Row( int rowIndex ) const { return fData.Row( rowIndex ); }
  
	GPUhd() int NHitsTotal() const { return fData.NumberOfHits(); }
	GPUhd() int NMaxTracks() const { return fNMaxTracks; }
  
	MEM_TEMPLATE() GPUd() void SetHitLinkUpData(const MEM_TYPE(AliGPUTPCRow) &row, int hitIndex, calink v) { fData.SetHitLinkUpData(row, hitIndex, v); }
	MEM_TEMPLATE() GPUd() void SetHitLinkDownData(const MEM_TYPE(AliGPUTPCRow) &row, int hitIndex, calink v) { fData.SetHitLinkDownData(row, hitIndex, v); }
	MEM_TEMPLATE() GPUd() calink HitLinkUpData(const MEM_TYPE(AliGPUTPCRow) &row, int hitIndex) const { return fData.HitLinkUpData(row, hitIndex); }
	MEM_TEMPLATE() GPUd() calink HitLinkDownData(const MEM_TYPE(AliGPUTPCRow) &row, int hitIndex) const { return fData.HitLinkDownData(row, hitIndex); }
  
	MEM_TEMPLATE() GPUd() GPUglobalref() const cahit2 *HitData(const MEM_TYPE(AliGPUTPCRow) &row) const { return fData.HitData(row); }
	MEM_TEMPLATE() GPUd() GPUglobalref() const calink *HitLinkUpData(const MEM_TYPE(AliGPUTPCRow) &row) const { return fData.HitLinkUpData(row); }
	MEM_TEMPLATE() GPUd() GPUglobalref() const calink *HitLinkDownData(const MEM_TYPE(AliGPUTPCRow) &row) const { return fData.HitLinkDownData(row); }
	MEM_TEMPLATE() GPUd() GPUglobalref() const calink *FirstHitInBin(const MEM_TYPE(AliGPUTPCRow) &row) const { return fData.FirstHitInBin(row); }
  
	MEM_TEMPLATE() GPUd() int FirstHitInBin( const MEM_TYPE( AliGPUTPCRow) &row, int binIndex ) const { return fData.FirstHitInBin(row, binIndex); }
  
	MEM_TEMPLATE() GPUd() cahit HitDataY( const MEM_TYPE(AliGPUTPCRow) &row, int hitIndex) const { return fData.HitDataY(row, hitIndex); }
	MEM_TEMPLATE() GPUd() cahit HitDataZ( const MEM_TYPE(AliGPUTPCRow) &row, int hitIndex) const { return fData.HitDataZ(row, hitIndex); }
	MEM_TEMPLATE() GPUd() cahit2 HitData( const MEM_TYPE(AliGPUTPCRow) &row, int hitIndex) const { return fData.HitData(row, hitIndex); }
  
	MEM_TEMPLATE() GPUhd() int HitInputID( const MEM_TYPE(AliGPUTPCRow) &row, int hitIndex) const { return fData.ClusterDataIndex(row, hitIndex); }
  
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
	MEM_TEMPLATE() GPUd() void MaximizeHitWeight(const MEM_TYPE(AliGPUTPCRow) &row, int hitIndex, int weight)
	{
		fData.MaximizeHitWeight( row, hitIndex, weight );
	}
	MEM_TEMPLATE() GPUd() void SetHitWeight(const MEM_TYPE(AliGPUTPCRow) &row, int hitIndex, int weight)
	{
		fData.SetHitWeight( row, hitIndex, weight );
	}
	MEM_TEMPLATE() GPUd() int HitWeight(const MEM_TYPE(AliGPUTPCRow) &row, int hitIndex) const
	{
		return fData.HitWeight(row, hitIndex);
	}
  
	GPUhd() GPUglobalref() int *NTracklets() const { return &fCommonMem->fNTracklets; }
  
	GPUhd() const AliGPUTPCHitId &TrackletStartHit( int i ) const { return fTrackletStartHits[i]; }
	GPUhd() GPUglobalref() AliGPUTPCHitId *TrackletStartHits() const { return fTrackletStartHits; }
	GPUhd() GPUglobalref() AliGPUTPCHitId *TrackletTmpStartHits() const { return fTrackletTmpStartHits; }
	MEM_CLASS_PRE2() GPUhd() const MEM_LG2(AliGPUTPCTracklet) &Tracklet( int i ) const { return fTracklets[i]; }
	GPUhd() GPUglobalref() MEM_GLOBAL(AliGPUTPCTracklet) *Tracklets() const { return fTracklets;}
	GPUhd() GPUglobalref() calink* TrackletRowHits() const { return fTrackletRowHits; }

	GPUhd() GPUglobalref() int *NTracks()  const { return &fCommonMem->fNTracks; }
	GPUhd() GPUglobalref() MEM_GLOBAL(AliGPUTPCTrack) *Tracks() const { return fTracks; }
	GPUhd() GPUglobalref() int *NTrackHits()  const { return &fCommonMem->fNTrackHits; }
	GPUhd() GPUglobalref() AliGPUTPCHitId *TrackHits() const { return fTrackHits; }
  
	GPUhd() GPUglobalref() MEM_GLOBAL(AliGPUTPCRow)* SliceDataRows() const {return(fData.Rows()); }
  
	GPUhd() GPUglobalref() int* RowStartHitCountOffset() const {return(fRowStartHitCountOffset);}
	GPUhd() GPUglobalref() StructGPUParameters* GPUParameters() const {return(&fCommonMem->fGPUParameters);}
	GPUhd() MakeType(MEM_LG(StructGPUParametersConst)*) GPUParametersConst() {return(&fGPUParametersConst);}
	GPUhd() MakeType(MEM_LG(const StructGPUParametersConst)*) GetGPUParametersConst() const {return(&fGPUParametersConst);}
	GPUhd() void SetGPUTextureBase(void* val) { fData.SetGPUTextureBase(val); }

	struct trackSortData
	{
		int fTtrack;		//Track ID
		float fSortVal;		//Value to sort for
	};

	void PerformGlobalTracking(AliGPUTPCTracker& sliceLeft, AliGPUTPCTracker& sliceRight, int MaxTracksLeft, int MaxTracksRight);
	
	void StartTimer(int i) {if (mCAParam->debugLevel) fTimers[i].Start();}
	void StopTimer(int i) {if (mCAParam->debugLevel) fTimers[i].Stop();}
	double GetTimer(int i) {return fTimers[i].GetElapsedTime();}
	void ResetTimer(int i) {fTimers[i].Reset();}
	void* LinkTmpMemory() {return fLinkTmpMemory;}

#if !defined(GPUCA_GPUCODE)
	GPUh() int PerformGlobalTrackingRun(AliGPUTPCTracker& sliceNeighbour, int iTrack, int rowIndex, float angle, int direction);
	void SetGPUDebugOutput(std::ostream *file) {fGPUDebugOut = file;}
#endif

  private:
	char* fStageAtSync;					//Temporary performance variable: Pointer to array storing current stage for every thread at every sync point
	char* fLinkTmpMemory;				//tmp memory for hits after neighbours finder

	int fISlice; //Number of slice
	HighResTimer fTimers[10];
  
	/** A pointer to the ClusterData object that the SliceData was created from. This can be used to
	* merge clusters from inside the SliceTracker code and recreate the SliceData. */
	MEM_LG(AliGPUTPCSliceData) fData; // The SliceData object. It is used to encapsulate the storage in memory from the access
  
#if !defined(GPUCA_GPUCODE)
	std::ostream *fGPUDebugOut; // debug stream
#else
	void* fGPUDebugOut; //No this is a hack, but I have no better idea.
#endif
	int fNMaxStartHits;
	int fNMaxTracklets;
	int fNMaxTracks;
	int fNMaxTrackHits;
	short mMemoryResScratch;
	short mMemoryResScratchHost;
	short mMemoryResCommon;
	short mMemoryResTracklets;
	short mMemoryResTracks;
	short mMemoryResTrackHits;

	//GPU Temp Arrays
	GPUglobalref() int* fRowStartHitCountOffset;				//Offset, length and new offset of start hits in row
	GPUglobalref() AliGPUTPCHitId *fTrackletTmpStartHits;	//Unsorted start hits
	GPUglobalref() char* fGPUTrackletTemp;					//Temp Memory for GPU Tracklet Constructor

	MEM_LG(StructGPUParametersConst) fGPUParametersConst; // Parameters for GPU if this is a GPU tracker

	// event
	GPUglobalref() commonMemoryStruct *fCommonMem; // common event memory
	GPUglobalref() AliGPUTPCHitId *fTrackletStartHits;   // start hits for the tracklets
	GPUglobalref() MEM_GLOBAL(AliGPUTPCTracklet) *fTracklets; // tracklets
	GPUglobalref() calink *fTrackletRowHits;			//Hits for each Tracklet in each row
	GPUglobalref() MEM_GLOBAL(AliGPUTPCTrack) *fTracks;	// reconstructed tracks
	GPUglobalref() AliGPUTPCHitId *fTrackHits;			// array of track hit numbers
	
	// output
	GPUglobalref() AliGPUTPCSliceOutput **fOutput;		//address of pointer pointing to SliceOutput Object
	void* fOutputMemory;									//Pointer to output memory if stored internally
  
	// disable copy
	AliGPUTPCTracker( const AliGPUTPCTracker& );
	AliGPUTPCTracker &operator=( const AliGPUTPCTracker& );
  
	static int StarthitSortComparison(const void*a, const void* b);
};

#endif //ALIHLTTPCCATRACKER_H
