// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCGMMerger.h
/// \author Sergey Gorbunov, David Rohr

#ifndef GPUTPCGMMERGER_H
#define GPUTPCGMMERGER_H

#include "GPUParam.h"
#include "GPUTPCDef.h"
#include "GPUTPCGMBorderTrack.h"
#include "GPUTPCGMMergedTrack.h"
#include "GPUTPCGMPolynomialField.h"
#include "GPUTPCGMSliceTrack.h"
#include "GPUCommonDef.h"
#include "GPUProcessor.h"

#if !defined(GPUCA_GPUCODE)
#include <cmath>
#include <iostream>
#endif //GPUCA_GPUCODE

class GPUTPCSliceTrack;
class GPUTPCSliceOutput;
class GPUTPCGMCluster;
class GPUTPCGMTrackParam;
class GPUTPCTracker;
class GPUChainTracking;

/**
 * @class GPUTPCGMMerger
 *
 */
class GPUTPCGMMerger : public GPUProcessor
{

public:
	GPUTPCGMMerger();
	~GPUTPCGMMerger() CON_DEFAULT;

	void InitializeProcessor();
	void RegisterMemoryAllocation();
	void SetMaxData();
	void* SetPointersHostOnly(void* mem);
	void* SetPointersGPURefit(void* mem);
    
	void OverrideSliceTracker(GPUTPCTracker* trk) { fSliceTrackers = trk; }
	void SetTrackingChain(GPUChainTracking* c) {fChainTracking = c;}

	void SetSliceData(int index, const GPUTPCSliceOutput *SliceData);
	int CheckSlices();

	GPUhd() int NOutputTracks() const { return fNOutputTracks; }
	GPUhd() const GPUTPCGMMergedTrack *OutputTracks() const { return fOutputTracks; }
	GPUhd() GPUTPCGMMergedTrack *OutputTracks() 	{ return fOutputTracks; }

	GPUhd() const GPUParam &SliceParam() const { return *mCAParam; }

	GPUd() const GPUTPCGMPolynomialField &Field() const { return fField; }
	GPUhd() const GPUTPCGMPolynomialField *pField() const { return &fField; }
	void SetField(GPUTPCGMPolynomialField *field) { fField = *field; }

	GPUhd() int NClusters() const { return (fNClusters); }
	GPUhd() int NOutputTrackClusters() const { return (fNOutputTrackClusters); }
	GPUhd() const GPUTPCGMMergedTrackHit *Clusters() const { return (fClusters); }
	GPUhd() GPUTPCGMMergedTrackHit *Clusters() { return (fClusters); }
	GPUhd() const GPUTPCTracker *SliceTrackers() const { return (fSliceTrackers); }
	GPUhd() GPUAtomic(int) *ClusterAttachment() const { return (fClusterAttachment); }
	GPUhd() int MaxId() const { return (fMaxID); }
	GPUhd() unsigned int *TrackOrder() const { return (fTrackOrder); }

	enum attachTypes {attachAttached = 0x40000000, attachGood = 0x20000000, attachGoodLeg = 0x10000000, attachTube = 0x08000000, attachHighIncl = 0x04000000, attachTrackMask = 0x03FFFFFF, attachFlagMask = 0xFC000000};
	
	short MemoryResMerger() {return mMemoryResMerger;}
	short MemoryResRefit() {return mMemoryResRefit;}

	void UnpackSlices();
	void MergeCEInit();
	void MergeCE();
	void MergeWithingSlices();
	void MergeSlices();
	void PrepareClustersForFit();
	void CollectMergedTracks();
	void Finalize();

  private:
	GPUTPCGMMerger(const GPUTPCGMMerger &) CON_DELETE;
	const GPUTPCGMMerger &operator=(const GPUTPCGMMerger &) const CON_DELETE;

	void MakeBorderTracks(int iSlice, int iBorder, GPUTPCGMBorderTrack B[], int &nB, bool fromOrig = false);
	void MergeBorderTracks(int iSlice1, GPUTPCGMBorderTrack B1[], int N1, int iSlice2, GPUTPCGMBorderTrack B2[], int N2, int crossCE = 0);

	void MergeCEFill(const GPUTPCGMSliceTrack *track, const GPUTPCGMMergedTrackHit &cls, int itr);
	void ResolveMergeSlices(bool fromOrig, bool mergeAll);
	void MergeSlicesStep(int border0, int border1, bool fromOrig);
	void ClearTrackLinks(int n);

	void PrintMergeGraph(GPUTPCGMSliceTrack *trk);
	void CheckMergedTracks();
	int GetTrackLabel(GPUTPCGMBorderTrack &trk);

	int SliceTrackInfoFirst(int iSlice) { return fSliceTrackInfoIndex[iSlice]; }
	int SliceTrackInfoLast(int iSlice) { return fSliceTrackInfoIndex[iSlice + 1]; }
	int SliceTrackInfoGlobalFirst(int iSlice) { return fSliceTrackInfoIndex[fgkNSlices + iSlice]; }
	int SliceTrackInfoGlobalLast(int iSlice) { return fSliceTrackInfoIndex[fgkNSlices + iSlice + 1]; }
	int SliceTrackInfoLocalTotal() { return fSliceTrackInfoIndex[fgkNSlices]; }
	int SliceTrackInfoTotal() { return fSliceTrackInfoIndex[2 * fgkNSlices]; }

	static CONSTEXPR int fgkNSlices = GPUCA_NSLICES; //* N slices
	int fNextSliceInd[fgkNSlices];
	int fPrevSliceInd[fgkNSlices];

	GPUTPCGMPolynomialField fField;

	const GPUTPCSliceOutput *fkSlices[fgkNSlices]; //* array of input slice tracks

	int *fTrackLinks;
	
	unsigned int fNMaxSliceTracks; //maximum number of incoming slice tracks
	unsigned int fNMaxTracks; //maximum number of output tracks
	unsigned int fNMaxSingleSliceTracks; // max N tracks in one slice
	unsigned int fNMaxOutputTrackClusters; //max number of clusters in output tracks (double-counting shared clusters)
	unsigned int fNMaxClusters; //max total unique clusters (in event)
	
	short mMemoryResMerger;
	short mMemoryResRefit;

	int fMaxID;
	int fNClusters; //Total number of incoming clusters
	int fNOutputTracks;
	int fNOutputTrackClusters;
	GPUTPCGMMergedTrack *fOutputTracks; //* array of output merged tracks

	GPUTPCGMSliceTrack *fSliceTrackInfos; //* additional information for slice tracks
	int fSliceTrackInfoIndex[fgkNSlices * 2 + 1];
	GPUTPCGMMergedTrackHit *fClusters;
	int *fGlobalClusterIDs;
	GPUAtomic(int) *fClusterAttachment;
	unsigned int *fTrackOrder;
	char* fTmpMem;
	GPUTPCGMBorderTrack *fBorderMemory; // memory for border tracks
	GPUTPCGMBorderTrack *fBorder[fgkNSlices];
	GPUTPCGMBorderTrack::Range *fBorderRangeMemory;       // memory for border tracks
	GPUTPCGMBorderTrack::Range *fBorderRange[fgkNSlices]; // memory for border tracks
	int fBorderCETracks[2][fgkNSlices];

	const GPUTPCTracker *fSliceTrackers;
	GPUChainTracking* fChainTracking;        // Tracking chain with access to input data / parameters
};

#endif //GPUTPCGMMERGER_H
