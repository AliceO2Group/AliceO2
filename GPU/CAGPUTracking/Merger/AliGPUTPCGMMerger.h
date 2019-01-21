//-*- Mode: C++ -*-
// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************

#ifndef ALIHLTTPCGMMERGER_H
#define ALIHLTTPCGMMERGER_H

#include "AliGPUCAParam.h"
#include "AliGPUTPCDef.h"
#include "AliGPUTPCGMBorderTrack.h"
#include "AliGPUTPCGMMergedTrack.h"
#include "AliGPUTPCGMPolynomialField.h"
#include "AliGPUTPCGMSliceTrack.h"
#include "AliTPCCommonDef.h"
#include "AliGPUProcessor.h"

#if !defined(GPUCA_GPUCODE)
#include <cmath>
#include <iostream>
#endif //GPUCA_GPUCODE

class AliGPUTPCSliceTrack;
class AliGPUTPCSliceOutput;
class AliGPUTPCGMCluster;
class AliGPUTPCGMTrackParam;
class AliGPUTPCTracker;

/**
 * @class AliGPUTPCGMMerger
 *
 */
class AliGPUTPCGMMerger : public AliGPUProcessor
{

  public:
	AliGPUTPCGMMerger();
	~AliGPUTPCGMMerger();

	void InitializeProcessor();
	void RegisterMemoryAllocation();
	void SetMaxData();
	void* SetPointersHostOnly(void* mem);
	void* SetPointersGPURefit(void* mem);
    
	void OverrideSliceTracker(AliGPUTPCTracker* trk) { fSliceTrackers = trk; }

	void SetSliceData(int index, const AliGPUTPCSliceOutput *SliceData);
	bool Reconstruct();
    void Clear();

	int NOutputTracks() const { return fNOutputTracks; }
	const AliGPUTPCGMMergedTrack *OutputTracks() const { return fOutputTracks; }
	AliGPUTPCGMMergedTrack *OutputTracks() 	{ return fOutputTracks; }

	GPUhd() const AliGPUCAParam &SliceParam() const { return *mCAParam; }

	GPUd() const AliGPUTPCGMPolynomialField &Field() const { return fField; }
	GPUhd() const AliGPUTPCGMPolynomialField *pField() const { return &fField; }
	void SetField(AliGPUTPCGMPolynomialField *field) { fField = *field; }

	int NClusters() const { return (fNClusters); }
	int NOutputTrackClusters() const { return (fNOutputTrackClusters); }
	const AliGPUTPCGMMergedTrackHit *Clusters() const { return (fClusters); }
	AliGPUTPCGMMergedTrackHit *Clusters() { return (fClusters); }
	const int *GlobalClusterIDs() const { return (fGlobalClusterIDs); }
	const AliGPUTPCTracker *SliceTrackers() const { return (fSliceTrackers); }
	int *ClusterAttachment() const { return (fClusterAttachment); }
	int MaxId() const { return (fMaxID); }
	unsigned int *TrackOrder() const { return (fTrackOrder); }

	enum attachTypes {attachAttached = 0x40000000, attachGood = 0x20000000, attachGoodLeg = 0x10000000, attachTube = 0x08000000, attachHighIncl = 0x04000000, attachTrackMask = 0x03FFFFFF, attachFlagMask = 0xFC000000};
	
	short MemoryResMerger() {return mMemoryResMerger;}
	short MemoryResRefit() {return mMemoryResRefit;}

  private:
	AliGPUTPCGMMerger(const AliGPUTPCGMMerger &) CON_DELETE;
	const AliGPUTPCGMMerger &operator=(const AliGPUTPCGMMerger &) const CON_DELETE;

	void MakeBorderTracks(int iSlice, int iBorder, AliGPUTPCGMBorderTrack B[], int &nB, bool fromOrig = false);

	void MergeBorderTracks(int iSlice1, AliGPUTPCGMBorderTrack B1[], int N1, int iSlice2, AliGPUTPCGMBorderTrack B2[], int N2, int crossCE = 0);

	void UnpackSlices();
	void MergeCEInit();
	void MergeCEFill(const AliGPUTPCGMSliceTrack *track, const AliGPUTPCGMMergedTrackHit &cls, int itr);
	void MergeCE();
	void MergeWithingSlices();
	void MergeSlices();
	void ResolveMergeSlices(bool fromOrig, bool mergeAll);
	void MergeSlicesStep(int border0, int border1, bool fromOrig);
	void PrepareClustersForFit();
	void CollectMergedTracks();
	void Refit(bool resetTimers);
	void Finalize();
	void ClearTrackLinks(int n);

	void PrintMergeGraph(AliGPUTPCGMSliceTrack *trk);
	void CheckMergedTracks();
	int GetTrackLabel(AliGPUTPCGMBorderTrack &trk);

	int SliceTrackInfoFirst(int iSlice) { return fSliceTrackInfoIndex[iSlice]; }
	int SliceTrackInfoLast(int iSlice) { return fSliceTrackInfoIndex[iSlice + 1]; }
	int SliceTrackInfoGlobalFirst(int iSlice) { return fSliceTrackInfoIndex[fgkNSlices + iSlice]; }
	int SliceTrackInfoGlobalLast(int iSlice) { return fSliceTrackInfoIndex[fgkNSlices + iSlice + 1]; }
	int SliceTrackInfoLocalTotal() { return fSliceTrackInfoIndex[fgkNSlices]; }
	int SliceTrackInfoTotal() { return fSliceTrackInfoIndex[2 * fgkNSlices]; }

	static const int fgkNSlices = 36; //* N slices
	int fNextSliceInd[fgkNSlices];
	int fPrevSliceInd[fgkNSlices];

	AliGPUTPCGMPolynomialField fField;

	const AliGPUTPCSliceOutput *fkSlices[fgkNSlices]; //* array of input slice tracks

	int *fTrackLinks;
	
	int fNMaxSliceTracks;
	int fNMaxTracks;
	int fNMaxSingleSliceTracks; // max N tracks in one slice
	int fNMaxOutputTrackClusters;
	
	short mMemoryResMerger;
	short mMemoryResRefit;

	int fMaxID;
	int fNClusters; //Total number of incoming clusters
	int fNOutputTracks;
	int fNOutputTrackClusters;
	AliGPUTPCGMMergedTrack *fOutputTracks; //* array of output merged tracks

	AliGPUTPCGMSliceTrack *fSliceTrackInfos; //* additional information for slice tracks
	int fSliceTrackInfoIndex[fgkNSlices * 2 + 1];
	AliGPUTPCGMMergedTrackHit *fClusters;
	int *fGlobalClusterIDs;
	int *fClusterAttachment;
	unsigned int *fTrackOrder;
	AliGPUTPCGMBorderTrack *fBorderMemory; // memory for border tracks
	AliGPUTPCGMBorderTrack *fBorder[fgkNSlices];
	AliGPUTPCGMBorderTrack::Range *fBorderRangeMemory;       // memory for border tracks
	AliGPUTPCGMBorderTrack::Range *fBorderRange[fgkNSlices]; // memory for border tracks
	int fBorderCETracks[2][fgkNSlices];

	const AliGPUTPCTracker *fSliceTrackers;
};

#endif //ALIHLTTPCGMMERGER_H
