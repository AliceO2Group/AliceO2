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

#include "AliGPUCAParam.h"
#include "AliGPUTPCHitId.h"
#include "AliGPUTPCSliceData.h"
#include "AliGPUTPCSliceOutput.h"
#include "AliGPUTPCTrackletConstructor.h"
#include "AliGPUTPCTracklet.h"
class AliGPUReconstruction;

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
class AliGPUTPCTracker
{
  public:
	AliGPUTPCTracker();
	~AliGPUTPCTracker();

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
  
	MEM_CLASS_PRE2() void Initialize( const MEM_LG2(AliGPUCAParam) *param, int iSlice );
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
	void SetGPUTracker();
#if !defined(__OPENCL__)
	char* SetGPUTrackerCommonMemory(char* const pGPUMemory);
	char* SetGPUTrackerHitsMemory(char* pGPUMemory, int MaxNHits);
	char* SetGPUTrackerTrackletsMemory(char* pGPUMemory, int MaxNTracklets);
	char* SetGPUTrackerTracksMemory(char* pGPUMemory, int MaxNTracks, int MaxNHits);
  
	//Debugging Stuff
	void DumpSliceData(std::ostream &out);	//Dump Input Slice Data
	void DumpLinks(std::ostream &out);		//Dump all links to file (for comparison after NeighboursFinder/Cleaner)
	void DumpStartHits(std::ostream &out);	//Same for Start Hits
	void DumpHitWeights(std::ostream &out); //....
	void DumpTrackHits(std::ostream &out);	//Same for Track Hits
	void DumpTrackletHits(std::ostream &out);	//Same for Track Hits
	void DumpOutput(FILE* out);	//Similar for output

	void SetOutput( AliGPUTPCSliceOutput** out ) { fOutput = out; }
	int ReadEvent( const AliGPUTPCClusterData *clusterData );

	GPUh() void SetAliGPUReconstruction( AliGPUReconstruction* val) { fGPUReconstruction = val; }

	GPUhd() const AliGPUTPCClusterData *ClusterData() const { return fClusterData; }

	GPUh() void ClearSliceDataHitWeights() {fData.ClearHitWeights();}
	GPUh() MakeType(const MEM_LG(AliGPUTPCRow)&) Row( const AliGPUTPCHitId &HitId ) const { return fData.Row(HitId.RowIndex()); }

	GPUhd() AliGPUTPCSliceOutput** Output() const { return fOutput; }

	GPUh() GPUglobalref() commonMemoryStruct *CommonMemory() const {return(fCommonMem); }
	GPUh() static size_t CommonMemorySize() { return(sizeof(AliGPUTPCTracker::commonMemoryStruct)); }
	GPUh() GPUglobalref() char* HitMemory() const {return(fHitMemory); }
	GPUh() size_t HitMemorySize() const {return(fHitMemorySize); }
	GPUh() char* TrackletMemory() {return(fTrackletMemory); }
	GPUh() size_t TrackletMemorySize() const {return(fTrackletMemorySize); }
	GPUh() char* TrackMemory() {return(fTrackMemory); }
	GPUh() size_t TrackMemorySize() const {return(fTrackMemorySize); }

	GPUh() void SetGPUSliceDataMemory(void* const pSliceMemory, void* const pRowMemory) { fData.SetGPUSliceDataMemory(pSliceMemory, pRowMemory); }

#endif
  
	MEM_CLASS_PRE2() GPUd() void GetErrors2( int iRow,  const MEM_LG2(AliGPUTPCTrackParam) &t, float &ErrY2, float &ErrZ2 ) const
	{
		//fParam.GetClusterErrors2( iRow, fParam.GetContinuousTracking() != 0. ? 125. : t.Z(), t.SinPhi(), t.DzDs(), ErrY2, ErrZ2 );
		fParam->GetClusterRMS2( iRow, fParam->ContinuousTracking != 0. ? 125. : t.Z(), t.SinPhi(), t.DzDs(), ErrY2, ErrZ2 );
	}
	GPUd() void GetErrors2( int iRow, float z, float sinPhi, float DzDs, float &ErrY2, float &ErrZ2 ) const
	{
		//fParam.GetClusterErrors2( iRow, fParam.GetContinuousTracking() != 0. ? 125. : z, sinPhi, DzDs, ErrY2, ErrZ2 );
		fParam->GetClusterRMS2( iRow, fParam->ContinuousTracking != 0. ? 125. : z, sinPhi, DzDs, ErrY2, ErrZ2 );
	}
  
	void SetupCommonMemory();
	void SetPointersHits(int MaxNHits);
	void SetPointersTracklets(int MaxNTracklets);
	void SetPointersTracks(int MaxNTracks, int MaxNHits);
	size_t SetPointersSliceData(const AliGPUTPCClusterData *data, bool allocate = false) { return(fData.SetPointers(data, allocate)); }
 
#if !defined(GPUCA_GPUCODE)
	GPUh() void WriteEvent(std::ostream &out);
	GPUh() void WriteTracks(std::ostream &out);
	GPUh() void ReadTracks(std::istream &in);
#endif //!GPUCA_GPUCODE
  
	GPUhd() MakeType(const MEM_LG(AliGPUCAParam)&) Param() const { return *fParam; }
	GPUhd() MakeType(const MEM_LG(AliGPUCAParam)*) pParam() const { return fParam; }
	GPUhd() void SetParam(const MEM_LG(AliGPUCAParam)* p) { fParam = p; }
	GPUhd() int ISlice() const { return fISlice; }
  
	GPUhd() MakeType(const MEM_LG(AliGPUTPCSliceData)&) Data() const { return fData; }
  
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
	GPUhd() void SetGPUTextureBase(char* val) { fData.SetGPUTextureBase(val); }

	struct trackSortData
	{
		int fTtrack;		//Track ID
		float fSortVal;		//Value to sort for
	};

	void PerformGlobalTracking(AliGPUTPCTracker& sliceLeft, AliGPUTPCTracker& sliceRight, int MaxTracksLeft, int MaxTracksRight);
	
	void StartTimer(int i) {if (fParam->debugLevel) fTimers[i].Start();}
	void StopTimer(int i) {if (fParam->debugLevel) fTimers[i].Stop();}
	double GetTimer(int i) {return fTimers[i].GetElapsedTime();}
	void ResetTimer(int i) {fTimers[i].Reset();}

#if !defined(GPUCA_GPUCODE)
	GPUh() int PerformGlobalTrackingRun(AliGPUTPCTracker& sliceNeighbour, int iTrack, int rowIndex, float angle, int direction);
	void SetGPUDebugOutput(std::ostream *file) {fGPUDebugOut = file;}
#endif

	//Temporary Variables for Standalone measurements
#ifdef  GPUCA_GPU_TRACKLET_CONSTRUCTOR_DO_PROFILE
	char* fStageAtSync;				//Pointer to array storing current stage for every thread at every sync point
#endif
	char *fLinkTmpMemory;				//tmp memory for hits after neighbours finder
  
  private:
	GPUglobalref() const MEM_GLOBAL(AliGPUCAParam) *fParam; // parameters
	int fISlice; //Number of slice
	HighResTimer fTimers[10];
  
	AliGPUReconstruction *fGPUReconstruction;
  
	/** A pointer to the ClusterData object that the SliceData was created from. This can be used to
	* merge clusters from inside the SliceTracker code and recreate the SliceData. */
	GPUglobalref() const AliGPUTPCClusterData *fClusterData; // ^
	MEM_LG(AliGPUTPCSliceData) fData; // The SliceData object. It is used to encapsulate the storage in memory from the access
  
	char fIsGPUTracker; // is it GPU tracker object
#if !defined(GPUCA_GPUCODE)
	std::ostream *fGPUDebugOut; // debug stream
#else
	void* fGPUDebugOut; //No this is a hack, but I have no better idea.
#endif
	int fNMaxTracks;

	//GPU Temp Arrays
	GPUglobalref() int* fRowStartHitCountOffset;				//Offset, length and new offset of start hits in row
	GPUglobalref() AliGPUTPCHitId *fTrackletTmpStartHits;	//Unsorted start hits
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

	GPUglobalref() AliGPUTPCHitId *fTrackletStartHits;   // start hits for the tracklets
	GPUglobalref() MEM_GLOBAL(AliGPUTPCTracklet) *fTracklets; // tracklets
	GPUglobalref() calink *fTrackletRowHits;			//Hits for each Tracklet in each row

	//
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
