// $Id: GPUTPCGMMerger.cxx 30732 2009-01-22 23:02:02Z sgorbuno $
// **************************************************************************
// This file is property of and copyright by the ALICE HLT Project          *
// ALICE Experiment at CERN, All rights reserved.                           *
//                                                                          *
// Primary Authors: Sergey Gorbunov <sergey.gorbunov@kip.uni-heidelberg.de> *
//                  Ivan Kisel <kisel@kip.uni-heidelberg.de>                *
//                  for The ALICE HLT Project.                              *
//                                                                          *
// Permission to use, copy, modify and distribute this software and its     *
// documentation strictly for non-commercial purposes is hereby granted     *
// without fee, provided that the above copyright notice appears in all     *
// copies and that both the copyright notice and this permission notice     *
// appear in the supporting documentation. The authors make no claims       *
// about the suitability of this software for any purpose. It is            *
// provided "as is" without express or implied warranty.                    *
//                                                                          *
//***************************************************************************

#include <cstdio>
#include <cstring>
#include "GPUTPCSliceOutTrack.h"
#include "GPUTPCTracker.h"
#include "GPUTPCClusterData.h"
#include "GPUTPCTrackParam.h"
#include "GPUTPCGMCluster.h"
#include "GPUTPCGMPolynomialField.h"
#include "GPUTPCGMPolynomialFieldManager.h"
#include "GPUTPCGMMerger.h"
#include "GPUReconstruction.h"
#include "GPUChainTracking.h"
#include "GPUQA.h"

#include "GPUCommonMath.h"

#include "GPUTPCTrackParam.h"
#include "GPUTPCSliceOutput.h"
#include "GPUTPCGMMergedTrack.h"
#include "GPUParam.h"
#include "GPUTPCTrackLinearisation.h"

#include "GPUTPCGMTrackParam.h"
#include "GPUTPCGMSliceTrack.h"
#include "GPUTPCGMBorderTrack.h"
#include <cmath>

#include <algorithm>

#include "GPUTPCGPUConfig.h"

#define DEBUG 0

static constexpr int kMaxParts = 400;
static constexpr int kMaxClusters = 1000;

//#define OFFLINE_FITTER

#if !defined(GPUCA_ALIROOT_LIB) || defined(GPUCA_GPUCODE)
#undef OFFLINE_FITTER
#endif

GPUTPCGMMerger::GPUTPCGMMerger() :
	fField(),
	fTrackLinks(nullptr),
	fNMaxSliceTracks(0),
	fNMaxTracks(0),
	fNMaxSingleSliceTracks(0),
	fNMaxOutputTrackClusters( 0 ),
	fNMaxClusters( 0 ),
	mMemoryResMerger(-1),
	mMemoryResRefit(-1),
	fMaxID(0),
	fNClusters(0),
	fNOutputTracks( 0 ),
	fNOutputTrackClusters( 0 ),
	fOutputTracks( 0 ),
	fSliceTrackInfos( 0 ),
	fClusters(nullptr),
	fGlobalClusterIDs(nullptr),
	fClusterAttachment(nullptr),
	fTrackOrder(nullptr),
	fTmpMem(0),
	fBorderMemory(0),
	fBorderRangeMemory(0),
	fSliceTrackers(nullptr),
	fChainTracking(nullptr)
{
	//* constructor

	for ( int iSlice = 0; iSlice < fgkNSlices; iSlice++ ) {
		fNextSliceInd[iSlice] = iSlice + 1;
		fPrevSliceInd[iSlice] = iSlice - 1;
	}
	int mid = fgkNSlices / 2 - 1 ;
	int last = fgkNSlices - 1 ;
	fNextSliceInd[ mid ] = 0;
	fPrevSliceInd[ 0 ] = mid;
	fNextSliceInd[ last ] = fgkNSlices / 2;
	fPrevSliceInd[ fgkNSlices/2 ] = last;

	fField.Reset(); // set very wrong initial value in order to see if the field was not properly initialised
	for (int i = 0; i < fgkNSlices; i++) fkSlices[i] = nullptr;
}

//DEBUG CODE
#if defined(GPUCA_MERGER_BY_MC_LABEL) || DEBUG == 1
void GPUTPCGMMerger::CheckMergedTracks()
{
	std::vector<bool> trkUsed(SliceTrackInfoLocalTotal());
	for (int i = 0;i < SliceTrackInfoLocalTotal();i++) trkUsed[i] = false;

	for ( int itr = 0; itr < SliceTrackInfoLocalTotal(); itr++ )
	{
		GPUTPCGMSliceTrack &track = fSliceTrackInfos[itr];
		if ( track.PrevSegmentNeighbour() >= 0 ) continue;
		if ( track.PrevNeighbour() >= 0 ) continue;
		int leg = 0;
		GPUTPCGMSliceTrack *trbase = &track, *tr = &track;
		tr->SetPrevSegmentNeighbour(1000000000);
		while (true)
		{
			int iTrk = tr - fSliceTrackInfos;
			if (trkUsed[iTrk])
			{
				printf("FAILURE: double use\n");
			}
			trkUsed[iTrk] = true;

			int jtr = tr->NextSegmentNeighbour();
			if( jtr >= 0 )
			{
				tr = &(fSliceTrackInfos[jtr]);
				tr->SetPrevSegmentNeighbour(1000000002);
					continue;
			}
			jtr = trbase->NextNeighbour();
			if( jtr>=0 )
			{
				trbase = &(fSliceTrackInfos[jtr]);
				tr = trbase;
				if( tr->PrevSegmentNeighbour() >= 0 ) break;
				tr->SetPrevSegmentNeighbour(1000000001);
				leg++;
				continue;
			}
			break;
		}
	}
	for (int i = 0;i < SliceTrackInfoLocalTotal();i++)
	{
		if (trkUsed[i] == false)
		{
			printf("FAILURE: trk missed\n");
		}
	}
}

int GPUTPCGMMerger::GetTrackLabel(GPUTPCGMBorderTrack &trk)
{
	GPUTPCGMSliceTrack *track = &fSliceTrackInfos[trk.TrackID()];
	const GPUTPCSliceOutCluster *clusters = track->OrigTrack()->Clusters();
	int nClusters = track->OrigTrack()->NClusters();
	std::vector<int> labels;
	GPUTPCStandaloneFramework &hlt = GPUTPCStandaloneFramework::Instance();
	for (int i = 0; i < nClusters; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			int label = hlt.GetMCLabels()[clusters[i].GetId()].fClusterID[j].fMCID;
			if (label >= 0) labels.push_back(label);
		}
	}
	if (labels.size() == 0) return (-1);
	labels.push_back(-1);
	std::sort(labels.begin(), labels.end());
	int bestLabel = -1, bestLabelCount = 0;
	int curLabel = labels[0], curCount = 1;
	for (unsigned int i = 1; i < labels.size(); i++)
	{
		if (labels[i] == curLabel)
		{
			curCount++;
		}
		else
		{
			if (curCount > bestLabelCount)
			{
				bestLabelCount = curCount;
				bestLabel = curLabel;
			}
			curLabel = labels[i];
			curCount = 1;
		}
	}
	return bestLabel;
}
#endif
//END DEBUG CODE

void GPUTPCGMMerger::InitializeProcessor()
{
	fSliceTrackers = fChainTracking->GetTPCSliceTrackers();
	if (mCAParam->AssumeConstantBz) GPUTPCGMPolynomialFieldManager::GetPolynomialField(GPUTPCGMPolynomialFieldManager::kUniform, mCAParam->BzkG, fField);
	else GPUTPCGMPolynomialFieldManager::GetPolynomialField(mCAParam->BzkG, fField);
}

void* GPUTPCGMMerger::SetPointersHostOnly(void* mem)
{
	computePointerWithAlignment(mem, fSliceTrackInfos, fNMaxSliceTracks);
	if (mCAParam->rec.NonConsecutiveIDs) computePointerWithAlignment(mem, fGlobalClusterIDs, fNMaxOutputTrackClusters);
	computePointerWithAlignment(mem, fBorderMemory, fNMaxSliceTracks);
	computePointerWithAlignment(mem, fBorderRangeMemory, 2 * fNMaxSliceTracks);
	computePointerWithAlignment(mem, fTrackLinks, fNMaxSliceTracks);
	size_t tmpSize = CAMath::Max(fNMaxSingleSliceTracks * fgkNSlices * sizeof(int), fNMaxTracks * sizeof(int) + fNMaxClusters * sizeof(char));
	computePointerWithAlignment(mem, fTmpMem, tmpSize);
	
	int nTracks = 0;
	for (int iSlice = 0; iSlice < fgkNSlices; iSlice++)
	{
		fBorder[iSlice] = fBorderMemory + nTracks;
		fBorderRange[iSlice] = fBorderRangeMemory + 2 * nTracks;
		nTracks += fkSlices[iSlice]->NTracks();
	}
	return mem;
}

void* GPUTPCGMMerger::SetPointersGPURefit(void* mem)
{
	computePointerWithAlignment(mem, fOutputTracks, fNMaxTracks);
	computePointerWithAlignment(mem, fClusters, fNMaxOutputTrackClusters);
	computePointerWithAlignment(mem, fTrackOrder, fNMaxTracks);
	computePointerWithAlignment(mem, fClusterAttachment, fNMaxClusters);

	return mem;
}

void GPUTPCGMMerger::RegisterMemoryAllocation()
{
	AllocateAndInitializeLate();
	mMemoryResMerger = mRec->RegisterMemoryAllocation(this, &GPUTPCGMMerger::SetPointersHostOnly, GPUMemoryResource::MEMORY_SCRATCH | GPUMemoryResource::MEMORY_HOST, "Merger");
	mMemoryResRefit = mRec->RegisterMemoryAllocation(this, &GPUTPCGMMerger::SetPointersGPURefit, GPUMemoryResource::MEMORY_INOUT, "Refit");
}

void GPUTPCGMMerger::SetMaxData()
{
	fNMaxSliceTracks = 0;
	fNClusters = 0;
	fNMaxSingleSliceTracks = 0;
	for (int iSlice = 0; iSlice < fgkNSlices; iSlice++)
	{
		if (!fkSlices[iSlice]) continue;
		fNMaxSliceTracks += fkSlices[iSlice]->NTracks();
		fNClusters += fkSlices[iSlice]->NTrackClusters();
		if (fNMaxSingleSliceTracks < fkSlices[iSlice]->NTracks()) fNMaxSingleSliceTracks = fkSlices[iSlice]->NTracks();
	}
	fNMaxOutputTrackClusters = fNClusters * 1.1f + 1000;
	fNMaxTracks = fNMaxSliceTracks;
	fNMaxClusters = 0;
	if (fSliceTrackers) for (int i = 0;i < fgkNSlices;i++) fNMaxClusters += fSliceTrackers[i].NHitsTotal();
	else fNMaxClusters = fNClusters;
}

void GPUTPCGMMerger::SetSliceData(int index, const GPUTPCSliceOutput *sliceData)
{
	fkSlices[index] = sliceData;
}

void GPUTPCGMMerger::ClearTrackLinks(int n)
{
	for (int i = 0; i < n; i++) fTrackLinks[i] = -1;
}

void GPUTPCGMMerger::UnpackSlices()
{
	//* unpack the cluster information from the slice tracks and initialize track info array

	int nTracksCurrent = 0;

	const GPUTPCSliceOutTrack *firstGlobalTracks[fgkNSlices];

	unsigned int maxSliceTracks = 0;
	for (int i = 0; i < fgkNSlices; i++)
	{
		firstGlobalTracks[i] = 0;
		if (fkSlices[i]->NLocalTracks() > maxSliceTracks) maxSliceTracks = fkSlices[i]->NLocalTracks();
	}
	
	if (maxSliceTracks > fNMaxSingleSliceTracks) throw std::runtime_error("fNMaxSingleSliceTracks too small");

	int *TrackIds = (int*) fTmpMem;
	for (unsigned int i = 0; i < maxSliceTracks * fgkNSlices; i++) TrackIds[i] = -1;

	for (int iSlice = 0; iSlice < fgkNSlices; iSlice++)
	{

		fSliceTrackInfoIndex[iSlice] = nTracksCurrent;

		float alpha = mCAParam->Alpha(iSlice);
		const GPUTPCSliceOutput &slice = *(fkSlices[iSlice]);
		const GPUTPCSliceOutTrack *sliceTr = slice.GetFirstTrack();

		for (unsigned int itr = 0; itr < slice.NLocalTracks(); itr++, sliceTr = sliceTr->GetNextTrack())
		{
			GPUTPCGMSliceTrack &track = fSliceTrackInfos[nTracksCurrent];
			track.Set(sliceTr, alpha, iSlice);
			if (!track.FilterErrors(*mCAParam, GPUCA_MAX_SIN_PHI, 0.1f)) continue;
			if (DEBUG) printf("INPUT Slice %d, Track %d, QPt %f DzDs %f\n", iSlice, itr, track.QPt(), track.DzDs());
			track.SetPrevNeighbour(-1);
			track.SetNextNeighbour(-1);
			track.SetNextSegmentNeighbour(-1);
			track.SetPrevSegmentNeighbour(-1);
			track.SetGlobalTrackId(0, -1);
			track.SetGlobalTrackId(1, -1);
			TrackIds[iSlice * maxSliceTracks + sliceTr->LocalTrackId()] = nTracksCurrent;
			nTracksCurrent++;
		}
		firstGlobalTracks[iSlice] = sliceTr;
	}
	for (int iSlice = 0; iSlice < fgkNSlices; iSlice++)
	{
		fSliceTrackInfoIndex[fgkNSlices + iSlice] = nTracksCurrent;

		float alpha = mCAParam->Alpha(iSlice);
		const GPUTPCSliceOutput &slice = *(fkSlices[iSlice]);
		const GPUTPCSliceOutTrack *sliceTr = firstGlobalTracks[iSlice];
		for (unsigned int itr = slice.NLocalTracks(); itr < slice.NTracks(); itr++, sliceTr = sliceTr->GetNextTrack())
		{
			int localId = TrackIds[(sliceTr->LocalTrackId() >> 24) * maxSliceTracks + (sliceTr->LocalTrackId() & 0xFFFFFF)];
			if (localId == -1) continue;
			GPUTPCGMSliceTrack &track = fSliceTrackInfos[nTracksCurrent];
			track.Set(sliceTr, alpha, iSlice);
			track.SetGlobalSectorTrackCov();
			track.SetPrevNeighbour(-1);
			track.SetNextNeighbour(-1);
			track.SetNextSegmentNeighbour(-1);
			track.SetPrevSegmentNeighbour(-1);
			track.SetLocalTrackId(localId);
			nTracksCurrent++;
		}
	}
	fSliceTrackInfoIndex[2 * fgkNSlices] = nTracksCurrent;
}

void GPUTPCGMMerger::MakeBorderTracks(int iSlice, int iBorder, GPUTPCGMBorderTrack B[], int &nB, bool fromOrig)
{
	//* prepare slice tracks for merging with next/previous/same sector
	//* each track transported to the border line

	float fieldBz = mCAParam->ConstBz;

	nB = 0;

	float dAlpha = mCAParam->DAlpha / 2;
	float x0 = 0;

	if ( iBorder == 0 ) { // transport to the left edge of the sector and rotate horizontally
		dAlpha = dAlpha - CAMath::Pi() / 2 ;
	} else if ( iBorder == 1 ) { // transport to the right edge of the sector and rotate horizontally
		dAlpha = -dAlpha - CAMath::Pi() / 2 ;
	} else if ( iBorder == 2 ) { // transport to the middle of the sector and rotate vertically to the border on the left
		x0 = mCAParam->RowX[ 63 ];
	} else if ( iBorder == 3 ) { // transport to the middle of the sector and rotate vertically to the border on the right
		dAlpha = -dAlpha;
		x0 = mCAParam->RowX[ 63 ];
	} else if ( iBorder == 4 ) { // transport to the middle of the sßector, w/o rotation
		dAlpha = 0;
		x0 = mCAParam->RowX[ 63 ];
	}

	const float maxSin = CAMath::Sin(60. / 180. * CAMath::Pi());
	float cosAlpha = CAMath::Cos(dAlpha);
	float sinAlpha = CAMath::Sin(dAlpha);

	GPUTPCGMSliceTrack trackTmp;
	for (int itr = SliceTrackInfoFirst(iSlice); itr < SliceTrackInfoLast(iSlice); itr++)
	{

		const GPUTPCGMSliceTrack *track = &fSliceTrackInfos[itr];

		if (track->PrevSegmentNeighbour() >= 0 && track->Slice() == fSliceTrackInfos[track->PrevSegmentNeighbour()].Slice()) continue;
		if (fromOrig)
		{
			if (fabsf(track->QPt()) < MERGE_LOOPER_QPT_LIMIT) continue;
			const GPUTPCGMSliceTrack *trackMin = track;
			while (track->NextSegmentNeighbour() >= 0 && track->Slice() == fSliceTrackInfos[track->NextSegmentNeighbour()].Slice())
			{
				track = &fSliceTrackInfos[track->NextSegmentNeighbour()];
				if (track->OrigTrack()->Param().X() < trackMin->OrigTrack()->Param().X()) trackMin = track;
			}
			trackTmp = *trackMin;
			track = &trackTmp;
			trackTmp.Set(trackMin->OrigTrack(), trackMin->Alpha(), trackMin->Slice());
		}
		else
		{
			if (fabsf(track->QPt()) < MERGE_HORIZONTAL_DOUBLE_QPT_LIMIT)
			{
				if (iBorder == 0 && track->NextNeighbour() >= 0) continue;
				if (iBorder == 1 && track->PrevNeighbour() >= 0) continue;
			}
		}
		GPUTPCGMBorderTrack &b = B[nB];

		if (track->TransportToXAlpha(x0, sinAlpha, cosAlpha, fieldBz, b, maxSin))
		{
			b.SetTrackID(itr);
			b.SetNClusters(track->NClusters());
			for (int i = 0; i < 4; i++)
				if (fabsf(b.Cov()[i]) >= 5.0) b.SetCov(i, 5.0);
			if (fabsf(b.Cov()[4]) >= 0.5) b.SetCov(4, 0.5);
			nB++;
		}
	}
}

void GPUTPCGMMerger::MergeBorderTracks(int iSlice1, GPUTPCGMBorderTrack B1[], int N1, int iSlice2, GPUTPCGMBorderTrack B2[], int N2, int crossCE)
{
	//* merge two sets of tracks
	if (N1 == 0 || N2 == 0) return;

	if (DEBUG) printf("\nMERGING Slices %d %d NTracks %d %d CROSS %d\n", iSlice1, iSlice2, N1, N2, crossCE);
	int statAll = 0, statMerged = 0;
	float factor2ys = 1.5; //1.5;//SG!!!
	float factor2zt = 1.5; //1.5;//SG!!!
	float factor2k = 2.0;  //2.2;

	factor2k = 3.5 * 3.5 * factor2k * factor2k;
	factor2ys = 3.5 * 3.5 * factor2ys * factor2ys;
	factor2zt = 3.5 * 3.5 * factor2zt * factor2zt;

	int minNPartHits = 10; //SG!!!
	int minNTotalHits = 20;

	GPUTPCGMBorderTrack::Range *range1 = fBorderRange[iSlice1];
	GPUTPCGMBorderTrack::Range *range2 = fBorderRange[iSlice2] + N2;

	bool sameSlice = (iSlice1 == iSlice2);
	{
		for ( int itr = 0; itr < N1; itr++ ){
			GPUTPCGMBorderTrack &b = B1[itr];
			float d = CAMath::Max(0.5f, 3.5f * sqrtf(b.Cov()[1]));
			if (fabsf(b.Par()[4]) >= 20) d *= 2;
			else if (d > 3) d = 3;
			if (DEBUG) {printf("  Input Slice 1 %d Track %d: ", iSlice1, itr); for (int i = 0;i < 5;i++) {printf("%8.3f ", b.Par()[i]);} printf(" - "); for (int i = 0;i < 5;i++) {printf("%8.3f ", b.Cov()[i]);} printf(" - D %8.3f\n", d);}
			range1[itr].fId = itr;
			range1[itr].fMin = b.Par()[1] + b.ZOffset() - d;
			range1[itr].fMax = b.Par()[1] + b.ZOffset() + d;
		}
		std::sort(range1,range1+N1,GPUTPCGMBorderTrack::Range::CompMin);
		if(sameSlice)
		{
			for(int i=0; i<N1; i++) range2[i]= range1[i];
			std::sort(range2,range2+N1,GPUTPCGMBorderTrack::Range::CompMax);
			N2 = N1;
			B2 = B1;
		}
		else
		{
			for ( int itr = 0;itr < N2;itr++)
			{
				GPUTPCGMBorderTrack &b = B2[itr];
				float d = CAMath::Max(0.5f, 3.5f * sqrtf(b.Cov()[1]));
				if (fabsf(b.Par()[4]) >= 20) d *= 2;
				else if (d > 3) d = 3;
				if (DEBUG) {printf("  Input Slice 2 %d Track %d: ", iSlice2, itr);for (int i = 0;i < 5;i++) {printf("%8.3f ", b.Par()[i]);}printf(" - ");for (int i = 0;i < 5;i++) {printf("%8.3f ", b.Cov()[i]);}printf(" - D %8.3f\n", d);}
				range2[itr].fId = itr;
				range2[itr].fMin = b.Par()[1] + b.ZOffset() - d;
				range2[itr].fMax = b.Par()[1] + b.ZOffset() + d;
			}
			std::sort(range2,range2+N2,GPUTPCGMBorderTrack::Range::CompMax);
		}
	}

	int i2 = 0;
	for (int i1 = 0;i1 < N1;i1++)
	{

		GPUTPCGMBorderTrack::Range r1 = range1[i1];
		while( i2<N2 && range2[i2].fMax< r1.fMin ) i2++;

		GPUTPCGMBorderTrack &b1 = B1[r1.fId];
		if ( b1.NClusters() < minNPartHits ) continue;
		int iBest2 = -1;
		int lBest2 = 0;
		statAll++;
		for(int k2 = i2;k2<N2;k2++)
		{
			GPUTPCGMBorderTrack::Range r2 = range2[k2];
			if( r2.fMin > r1.fMax ) break;
			if( sameSlice && (r1.fId >= r2.fId) ) continue;
			// do check

			GPUTPCGMBorderTrack &b2 = B2[r2.fId];
#ifdef GPUCA_MERGER_BY_MC_LABEL
			if (GetTrackLabel(b1) != GetTrackLabel(b2)) //DEBUG CODE, match by MC label
#endif
			{
				if (DEBUG) {printf("Comparing track %3d to %3d: ", r1.fId, r2.fId);for (int i = 0;i < 5;i++) {printf("%8.3f ", b1.Par()[i]);}printf(" - ");for (int i = 0;i < 5;i++) {printf("%8.3f ", b1.Cov()[i]);}printf("\n%28s", "");
					for (int i = 0;i < 5;i++) {printf("%8.3f ", b2.Par()[i]);}printf(" - ");for (int i = 0;i < 5;i++) {printf("%8.3f ", b2.Cov()[i]);}printf("   -   %5s   -   ", GetTrackLabel(b1) == GetTrackLabel(b2) ? "CLONE" : "FAKE");}
				if ( b2.NClusters() < lBest2 ) {if (DEBUG) {printf("!NCl1\n");}continue;}
				if (crossCE >= 2 && abs(b1.Row() - b2.Row()) > 1) {if (DEBUG) {printf("!ROW\n");}continue;}
				if( !b1.CheckChi2Y(b2, factor2ys ) ) {if (DEBUG) {printf("!Y\n");}continue;}
				//if( !b1.CheckChi2Z(b2, factor2zt ) ) {if (DEBUG) {printf("!NCl1\n");}continue;}
				if( !b1.CheckChi2QPt(b2, factor2k ) ) {if (DEBUG) {printf("!QPt\n");}continue;}
				float fys = fabsf(b1.Par()[4]) < 20 ? factor2ys : (2. * factor2ys);
				float fzt = fabsf(b1.Par()[4]) < 20 ? factor2zt : (2. * factor2zt);
				if( !b1.CheckChi2YS(b2, fys ) ) {if (DEBUG) {printf("!YS\n");}continue;}
				if( !b1.CheckChi2ZT(b2, fzt ) ) {if (DEBUG) {printf("!ZT\n");}continue;}
				if (fabsf(b1.Par()[4]) < 20)
				{
					if ( b2.NClusters() < minNPartHits ) {if (DEBUG) {printf("!NCl2\n");}continue;}
					if ( b1.NClusters() + b2.NClusters() < minNTotalHits ) {if (DEBUG) {printf("!NCl3\n");}continue;}
				}
				if (DEBUG) printf("OK: dZ %8.3f D1 %8.3f D2 %8.3f\n", fabsf(b1.Par()[1] - b2.Par()[1]), 3.5*sqrt(b1.Cov()[1]), 3.5*sqrt(b2.Cov()[1]));
			} //DEBUG CODE, match by MC label
			lBest2 = b2.NClusters();
			iBest2 = b2.TrackID();
		}

		if (iBest2 < 0) continue;
		statMerged++;

		if (DEBUG) printf("Found match %d %d\n", b1.TrackID(), iBest2);

		fTrackLinks[b1.TrackID()] = iBest2;
	}
	//printf("STAT: slices %d, %d: all %d merged %d\n", iSlice1, iSlice2, statAll, statMerged);
}

void GPUTPCGMMerger::MergeWithingSlices()
{
	float x0 = mCAParam->RowX[63];
	const float maxSin = CAMath::Sin(60. / 180.*CAMath::Pi());

	ClearTrackLinks(SliceTrackInfoLocalTotal());
	for (int iSlice = 0;iSlice < fgkNSlices;iSlice++)
	{
		int nBord = 0;
		for ( int itr = SliceTrackInfoFirst(iSlice); itr < SliceTrackInfoLast(iSlice); itr++ )
		{
			GPUTPCGMSliceTrack &track = fSliceTrackInfos[itr];
			GPUTPCGMBorderTrack &b = fBorder[iSlice][nBord];
			if (track.TransportToX(x0, mCAParam->ConstBz, b, maxSin))
			{
				b.SetTrackID( itr );
				if (DEBUG) {printf("WITHIN SLICE %d Track %d - ", iSlice, itr);for (int i = 0;i < 5;i++) {printf("%8.3f ", b.Par()[i]);} printf(" - ");for (int i = 0;i < 5;i++) {printf("%8.3f ", b.Cov()[i]);} printf("\n");}
				b.SetNClusters( track.NClusters() );
				nBord++;
			}
		}

		MergeBorderTracks( iSlice, fBorder[iSlice], nBord, iSlice, fBorder[iSlice], nBord );
	}

	ResolveMergeSlices(false, true);
}

void GPUTPCGMMerger::MergeSlices()
{
	MergeSlicesStep(2, 3, false);
	MergeSlicesStep(0, 1, false);
	MergeSlicesStep(0, 1, true);
}

void GPUTPCGMMerger::MergeSlicesStep(int border0, int border1, bool fromOrig)
{
	ClearTrackLinks(SliceTrackInfoLocalTotal());
	for (int iSlice = 0; iSlice < fgkNSlices; iSlice++)
	{
		int jSlice = fNextSliceInd[iSlice];
		GPUTPCGMBorderTrack *bCurr = fBorder[iSlice], *bNext = fBorder[jSlice];
		int nCurr = 0, nNext = 0;
		MakeBorderTracks(iSlice, border0, bCurr, nCurr, fromOrig);
		MakeBorderTracks(jSlice, border1, bNext, nNext, fromOrig);
		MergeBorderTracks(iSlice, bCurr, nCurr, jSlice, bNext, nNext, fromOrig ? -1 : 0);
	}
	ResolveMergeSlices(fromOrig, false);
}

void GPUTPCGMMerger::PrintMergeGraph(GPUTPCGMSliceTrack* trk)
{
	GPUTPCGMSliceTrack* orgTrack = trk;
	while (trk->PrevSegmentNeighbour() >= 0) trk = &fSliceTrackInfos[trk->PrevSegmentNeighbour()];
	GPUTPCGMSliceTrack* orgTower = trk;
	while (trk->PrevNeighbour() >= 0) trk = &fSliceTrackInfos[trk->PrevNeighbour()];

	int nextId = trk - fSliceTrackInfos;
	printf("Graph of track %d\n", (int) (orgTrack - fSliceTrackInfos));
	while (nextId >= 0)
	{
		trk = &fSliceTrackInfos[nextId];
		if (trk->PrevSegmentNeighbour() >= 0) printf("TRACK TREE INVALID!!! %d --> %d\n", trk->PrevSegmentNeighbour(), nextId);
		printf(trk == orgTower ? "--" : "  ");
		while (nextId >= 0)
		{
			GPUTPCGMSliceTrack *trk2 = &fSliceTrackInfos[nextId];
			if (trk != trk2 && (trk2->PrevNeighbour() >= 0 || trk2->NextNeighbour() >= 0))
			{
				printf("   (TRACK TREE INVALID!!! %d <-- %d --> %d)   ", trk2->PrevNeighbour(), nextId, trk2->NextNeighbour());
			}
			printf(" %s%5d(%5.2f)", trk2 == orgTrack ? "!" : " ", nextId, trk2->QPt());
			nextId = trk2->NextSegmentNeighbour();
		}
		printf("\n");
		nextId = trk->NextNeighbour();
	}
}

void GPUTPCGMMerger::ResolveMergeSlices(bool fromOrig, bool mergeAll)
{
	if (!mergeAll)
	{
		/*int neighborType = fromOrig ? 1 : 0;

		int old1 = newTrack2.PrevNeighbour(0);
		int old2 = newTrack1.NextNeighbour(0);
		if (old1 < 0 && old2 < 0) neighborType = 0;
		if (old1 == itr) continue;
		if (neighborType) old1 = newTrack2.PrevNeighbour(1);
		if ( old1 >= 0 )
		{
			GPUTPCGMSliceTrack &oldTrack1 = fSliceTrackInfos[old1];
			if ( oldTrack1.NClusters() < newTrack1.NClusters() ) {
				newTrack2.SetPrevNeighbour( -1, neighborType );
				oldTrack1.SetNextNeighbour( -1, neighborType );
			} else continue;
		}

		if (old2 == itr2) continue;
		if (neighborType) old2 = newTrack1.NextNeighbour(1);
		if ( old2 >= 0 )
		{
			GPUTPCGMSliceTrack &oldTrack2 = fSliceTrackInfos[old2];
			if ( oldTrack2.NClusters() < newTrack2.NClusters() )
			{
			oldTrack2.SetPrevNeighbour( -1, neighborType );
			} else continue;
		}
		newTrack1.SetNextNeighbour( itr2, neighborType );
		newTrack2.SetPrevNeighbour( itr, neighborType );*/
	}

	for (int itr = 0; itr < SliceTrackInfoLocalTotal(); itr++)
	{
		int itr2 = fTrackLinks[itr];
		if (itr2 < 0) continue;
		GPUTPCGMSliceTrack *track1 = &fSliceTrackInfos[itr];
		GPUTPCGMSliceTrack *track2 = &fSliceTrackInfos[itr2];
		GPUTPCGMSliceTrack *track1Base = track1;
		GPUTPCGMSliceTrack *track2Base = track2;

		bool sameSegment = fabsf(track1->NClusters() > track2->NClusters() ? track1->QPt() : track2->QPt()) < 2 || track1->QPt() * track2->QPt() > 0;
		//printf("\nMerge %d with %d - same segment %d\n", itr, itr2, (int) sameSegment);
		//PrintMergeGraph(track1);
		//PrintMergeGraph(track2);

		while (track2->PrevSegmentNeighbour() >= 0) track2 = &fSliceTrackInfos[track2->PrevSegmentNeighbour()];
		if (sameSegment)
		{
			if (track1 == track2) continue;
			while (track1->PrevSegmentNeighbour() >= 0)
			{
				track1 = &fSliceTrackInfos[track1->PrevSegmentNeighbour()];
				if (track1 == track2) goto NextTrack;
			}
			std::swap(track1, track1Base);
			for (int k = 0; k < 2; k++)
			{
				GPUTPCGMSliceTrack *tmp = track1Base;
				while (tmp->Neighbour(k) >= 0)
				{
					tmp = &fSliceTrackInfos[tmp->Neighbour(k)];
					if (tmp == track2) goto NextTrack;
				}
			}

			while (track1->NextSegmentNeighbour() >= 0)
			{
				track1 = &fSliceTrackInfos[track1->NextSegmentNeighbour()];
				if (track1 == track2) goto NextTrack;
			}
		}
		else
		{
			while (track1->PrevSegmentNeighbour() >= 0) track1 = &fSliceTrackInfos[track1->PrevSegmentNeighbour()];

			if (track1 == track2) continue;
			for (int k = 0; k < 2; k++)
			{
				GPUTPCGMSliceTrack *tmp = track1;
				while (tmp->Neighbour(k) >= 0)
				{
					tmp = &fSliceTrackInfos[tmp->Neighbour(k)];
					if (tmp == track2) goto NextTrack;
				}
			}

			float z1min = track1->MinClusterZ(), z1max = track1->MaxClusterZ();
			float z2min = track2->MinClusterZ(), z2max = track2->MaxClusterZ();
			if (track1 != track1Base) {z1min = CAMath::Min(z1min, track1Base->MinClusterZ()); z1max = CAMath::Max(z1max, track1Base->MaxClusterZ());}
			if (track2 != track2Base) {z2min = CAMath::Min(z2min, track2Base->MinClusterZ()); z2max = CAMath::Max(z2max, track2Base->MaxClusterZ());}

			bool goUp = z2max - z1min > z1max - z2min;

			if (track1->Neighbour(goUp) < 0 && track2->Neighbour(!goUp) < 0)
			{
				track1->SetNeighbor(track2 - fSliceTrackInfos, goUp);
				track2->SetNeighbor(track1 - fSliceTrackInfos, !goUp);
				//printf("Result (simple neighbor)\n");
				//PrintMergeGraph(track1);
				continue;
			}
			else if (track1->Neighbour(goUp) < 0)
			{
				track2 = &fSliceTrackInfos[track2->Neighbour(!goUp)];
				std::swap(track1, track2);
			}
			else if (track2->Neighbour(!goUp) < 0)
			{
				track1 = &fSliceTrackInfos[track1->Neighbour(goUp)];
			}
			else
			{ //Both would work, but we use the simpler one
				track1 = &fSliceTrackInfos[track1->Neighbour(goUp)];
			}
			track1Base = track1;
		}

		track2Base = track2;
		if (!sameSegment) while (track1->NextSegmentNeighbour() >= 0) track1 = &fSliceTrackInfos[track1->NextSegmentNeighbour()];
		track1->SetNextSegmentNeighbour(track2 - fSliceTrackInfos);
		track2->SetPrevSegmentNeighbour(track1 - fSliceTrackInfos);
		for (int k = 0;k < 2;k++)
		{
			track1 = track1Base;
			track2 = track2Base;
			while (track2->Neighbour(k) >= 0)
			{
				if (track1->Neighbour(k) >= 0)
				{
					GPUTPCGMSliceTrack *track1new = &fSliceTrackInfos[track1->Neighbour(k)];
					GPUTPCGMSliceTrack *track2new = &fSliceTrackInfos[track2->Neighbour(k)];
					track2->SetNeighbor(-1, k);
					track2new->SetNeighbor(-1, k ^ 1);
					track1 = track1new;
					while (track1->NextSegmentNeighbour() >= 0) track1 = &fSliceTrackInfos[track1->NextSegmentNeighbour()];
					track1->SetNextSegmentNeighbour(track2new - fSliceTrackInfos);
					track2new->SetPrevSegmentNeighbour(track1 - fSliceTrackInfos);
					track1 = track1new;
					track2 = track2new;
				}
				else
				{
					GPUTPCGMSliceTrack *track2new = &fSliceTrackInfos[track2->Neighbour(k)];
					track1->SetNeighbor(track2->Neighbour(k), k);
					track2->SetNeighbor(-1, k);
					track2new->SetNeighbor(track1 - fSliceTrackInfos, k ^ 1);
				}
			}
		}
		//printf("Result\n");
		//PrintMergeGraph(track1);
NextTrack:;
	}
}

void GPUTPCGMMerger::MergeCEInit()
{
	for (int k = 0; k < 2; k++)
	{
		for (int i = 0; i < fgkNSlices; i++)
		{
			fBorderCETracks[k][i] = 0;
		}
	}
}

void GPUTPCGMMerger::MergeCEFill(const GPUTPCGMSliceTrack *track, const GPUTPCGMMergedTrackHit &cls, int itr)
{
	if (mCAParam->rec.NonConsecutiveIDs) return;
#ifdef MERGE_CE_ROWLIMIT
	if (cls.fRow < MERGE_CE_ROWLIMIT || cls.fRow >= GPUCA_ROW_COUNT - MERGE_CE_ROWLIMIT) return;
#endif
	if (!mCAParam->ContinuousTracking && fabsf(cls.fZ) > 10) return;
	int slice = track->Slice();
	for (int attempt = 0; attempt < 2; attempt++)
	{
		GPUTPCGMBorderTrack &b = attempt == 0 ? fBorder[slice][fBorderCETracks[0][slice]] : fBorder[slice][fkSlices[slice]->NTracks() - 1 - fBorderCETracks[1][slice]];
		const float x0 = attempt == 0 ? mCAParam->RowX[63] : cls.fX;
		if(track->TransportToX(x0, mCAParam->ConstBz, b, GPUCA_MAX_SIN_PHI_LOW))
		{
			b.SetTrackID(itr);
			b.SetNClusters(fOutputTracks[itr].NClusters());
			for (int i = 0;i < 4;i++) if (fabsf(b.Cov()[i]) >= 5.0) b.SetCov(i, 5.0);
			if (fabsf(b.Cov()[4]) >= 0.5) b.SetCov(4, 0.5);
			if (track->CSide())
			{
				b.SetPar(1, b.Par()[1] - 2 * (cls.fZ - b.ZOffset()));
				b.SetZOffset(-b.ZOffset());
			}
			if (attempt) b.SetRow(cls.fRow);
			fBorderCETracks[attempt][slice]++;
			break;
		}
	}
}

void GPUTPCGMMerger::MergeCE()
{
	ClearTrackLinks(fNOutputTracks);
	for (int iSlice = 0;iSlice < fgkNSlices / 2;iSlice++)
	{
		int jSlice = iSlice + fgkNSlices / 2;
		MergeBorderTracks(iSlice, fBorder[iSlice], fBorderCETracks[0][iSlice], jSlice, fBorder[jSlice], fBorderCETracks[0][jSlice], 1);
		MergeBorderTracks(iSlice, fBorder[iSlice] + fkSlices[iSlice]->NTracks() - fBorderCETracks[1][iSlice], fBorderCETracks[1][iSlice], jSlice, fBorder[jSlice] + fkSlices[jSlice]->NTracks() - fBorderCETracks[1][jSlice], fBorderCETracks[1][jSlice], 2);
	}
	for (int i = 0;i < fNOutputTracks;i++)
	{
		if (fTrackLinks[i] >= 0)
		{
			GPUTPCGMMergedTrack* trk[2] = {&fOutputTracks[i], &fOutputTracks[fTrackLinks[i]]};
			
			if (!trk[1]->OK() || trk[1]->CCE()) continue;

			if (fNOutputTrackClusters + trk[0]->NClusters() + trk[1]->NClusters() >= fNMaxOutputTrackClusters)
			{
				printf("Insufficient cluster memory for merging CE tracks (OutputClusters %d, total clusters %d)\n", fNOutputTrackClusters, fNMaxOutputTrackClusters);
				return;
			}
			
			bool looper = trk[0]->Looper() || trk[1]->Looper() || (trk[0]->GetParam().GetQPt() > 1 && trk[0]->GetParam().GetQPt() * trk[1]->GetParam().GetQPt() < 0);
			bool needswap = false;
			if (looper)
			{
				const float z0max = CAMath::Max(fabsf(fClusters[trk[0]->FirstClusterRef()].fZ), fabsf(fClusters[trk[0]->FirstClusterRef() + trk[0]->NClusters() - 1].fZ));
				const float z1max = CAMath::Max(fabsf(fClusters[trk[1]->FirstClusterRef()].fZ), fabsf(fClusters[trk[1]->FirstClusterRef() + trk[1]->NClusters() - 1].fZ));
				if (z1max < z0max) needswap = true;
			}
			else
			{
				if (fClusters[trk[0]->FirstClusterRef()].fX > fClusters[trk[1]->FirstClusterRef()].fX) needswap = true;
			}
			if (needswap)
			{
				std::swap(trk[0], trk[1]);
			}

			bool reverse[2] = {false, false};
			if (looper)
			{
				reverse[0] = (fClusters[trk[0]->FirstClusterRef()].fZ > fClusters[trk[0]->FirstClusterRef() + trk[0]->NClusters() - 1].fZ) ^ (trk[0]->CSide() > 0);
				reverse[1] = (fClusters[trk[1]->FirstClusterRef()].fZ < fClusters[trk[1]->FirstClusterRef() + trk[1]->NClusters() - 1].fZ) ^ (trk[1]->CSide() > 0);
			}

			if (mCAParam->ContinuousTracking)
			{
				const float z0 = trk[0]->CSide() ? CAMath::Max(fClusters[trk[0]->FirstClusterRef()].fZ, fClusters[trk[0]->FirstClusterRef() + trk[0]->NClusters() - 1].fZ) :
					CAMath::Min(fClusters[trk[0]->FirstClusterRef()].fZ, fClusters[trk[0]->FirstClusterRef() + trk[0]->NClusters() - 1].fZ);
				const float z1 = trk[1]->CSide() ? CAMath::Max(fClusters[trk[1]->FirstClusterRef()].fZ, fClusters[trk[1]->FirstClusterRef() + trk[1]->NClusters() - 1].fZ) :
					CAMath::Min(fClusters[trk[1]->FirstClusterRef()].fZ, fClusters[trk[1]->FirstClusterRef() + trk[1]->NClusters() - 1].fZ);
				float offset = fabsf(z1) > fabsf(z0) ? -z0 : z1;
				trk[1]->Param().Z() += trk[1]->Param().ZOffset() - offset;
				trk[1]->Param().ZOffset() = offset;
			}

			int newRef = fNOutputTrackClusters;
			for (int k = 1;k >= 0;k--)
			{
				if (reverse[k]) for (int j = trk[k]->NClusters() - 1;j >= 0;j--) fClusters[fNOutputTrackClusters++] = fClusters[trk[k]->FirstClusterRef() + j];
				else for (unsigned int j = 0;j < trk[k]->NClusters();j++) fClusters[fNOutputTrackClusters++] = fClusters[trk[k]->FirstClusterRef() + j];
			}
			trk[1]->SetFirstClusterRef(newRef);
			trk[1]->SetNClusters(trk[0]->NClusters() + trk[1]->NClusters());
			trk[1]->SetCCE(true);
			trk[0]->SetNClusters(0);
			trk[0]->SetOK(false);
		}
	}

	//for (int i = 0;i < fNOutputTracks;i++) {if (fOutputTracks[i].CCE() == false) {fOutputTracks[i].SetNClusters(0);fOutputTracks[i].SetOK(false);}} //Remove all non-CE tracks
}

struct GPUTPCGMMerger_CompareClusterIdsLooper
{
	struct clcomparestruct {unsigned char leg;};

	const unsigned char fLeg;
	const bool fOutwards;
	const GPUTPCSliceOutCluster* const fCmp1;
	const clcomparestruct* const fCmp2;
	GPUTPCGMMerger_CompareClusterIdsLooper(unsigned char leg, bool outwards, const GPUTPCSliceOutCluster* cmp1, const clcomparestruct* cmp2) : fLeg(leg), fOutwards(outwards), fCmp1(cmp1), fCmp2(cmp2) {}
	bool operator()(const int aa, const int bb)
	{
		const clcomparestruct& a = fCmp2[aa];
		const clcomparestruct& b = fCmp2[bb];
		const GPUTPCSliceOutCluster& a1 = fCmp1[aa];
		const GPUTPCSliceOutCluster& b1 = fCmp1[bb];
		if (a.leg != b.leg) return ((fLeg > 0) ^ (a.leg > b.leg));
		if (a1.GetX() != b1.GetX()) return((a1.GetX() > b1.GetX()) ^ ((a.leg - fLeg) & 1) ^ fOutwards);
		return false;
	}
};

struct GPUTPCGMMerger_CompareClusterIds
{
	const GPUTPCSliceOutCluster* const fCmp;
	GPUTPCGMMerger_CompareClusterIds(const GPUTPCSliceOutCluster* cmp) : fCmp(cmp) {}
	bool operator()(const int aa, const int bb)
	{
		const GPUTPCSliceOutCluster& a = fCmp[aa];
		const GPUTPCSliceOutCluster& b = fCmp[bb];
		return(a.GetX() > b.GetX());
	}
};

struct GPUTPCGMMerger_CompareTracks
{
	const GPUTPCGMMergedTrack* const fCmp;
	GPUTPCGMMerger_CompareTracks(GPUTPCGMMergedTrack* cmp) : fCmp(cmp) {}
	bool operator()(const int aa, const int bb)
	{
		const GPUTPCGMMergedTrack& a = fCmp[aa];
		const GPUTPCGMMergedTrack& b = fCmp[bb];
		return(fabsf(a.GetParam().GetQPt()) > fabsf(b.GetParam().GetQPt()));
	}
};

bool GPUTPCGMMerger_CompareParts(const GPUTPCGMSliceTrack* a, const GPUTPCGMSliceTrack* b)
{
  return(a->X() > b->X());
}

void GPUTPCGMMerger::CollectMergedTracks()
{
	//Resolve connections for global tracks first
	for (int iSlice = 0; iSlice < fgkNSlices; iSlice++)
	{
		for (int itr = SliceTrackInfoGlobalFirst(iSlice); itr < SliceTrackInfoGlobalLast(iSlice); itr++)
		{
			GPUTPCGMSliceTrack &globalTrack = fSliceTrackInfos[itr];
			GPUTPCGMSliceTrack &localTrack = fSliceTrackInfos[globalTrack.LocalTrackId()];
			localTrack.SetGlobalTrackId(localTrack.GlobalTrackId(0) != -1, itr);
		}
	}

	//CheckMergedTracks();

	//Now collect the merged tracks
	fNOutputTracks = 0;
	int nOutTrackClusters = 0;

	GPUTPCGMSliceTrack *trackParts[kMaxParts];

	for (int itr = 0; itr < SliceTrackInfoLocalTotal(); itr++)
	{

		GPUTPCGMSliceTrack &track = fSliceTrackInfos[itr];

		if (track.PrevSegmentNeighbour() >= 0) continue;
		if (track.PrevNeighbour() >= 0) continue;
		int nParts = 0;
		int nHits = 0;
		int leg = 0;
		GPUTPCGMSliceTrack *trbase = &track, *tr = &track;
		tr->SetPrevSegmentNeighbour(1000000000);
		while (true)
		{
			if (nParts >= kMaxParts) break;
			if (nHits + tr->NClusters() > kMaxClusters) break;
			nHits += tr->NClusters();

		tr->SetLeg(leg);
		trackParts[nParts++] = tr;
		for (int i = 0; i < 2; i++)
		{
			if (tr->GlobalTrackId(i) != -1)
			{
				if (nParts >= kMaxParts) break;
				if (nHits + fSliceTrackInfos[tr->GlobalTrackId(i)].NClusters() > kMaxClusters) break;
				trackParts[nParts] = &fSliceTrackInfos[tr->GlobalTrackId(i)];
				trackParts[nParts++]->SetLeg(leg);
				nHits += fSliceTrackInfos[tr->GlobalTrackId(i)].NClusters();
			}
		}
		int jtr = tr->NextSegmentNeighbour();
		if (jtr >= 0)
		{
			tr = &(fSliceTrackInfos[jtr]);
			tr->SetPrevSegmentNeighbour(1000000002);
			continue;
		}
		jtr = trbase->NextNeighbour();
		if (jtr >= 0)
		{
			trbase = &(fSliceTrackInfos[jtr]);
			tr = trbase;
			if (tr->PrevSegmentNeighbour() >= 0) break;
			tr->SetPrevSegmentNeighbour(1000000001);
			leg++;
			continue;
		}
		break;
		}

		// unpack and sort clusters
		if (nParts > 1 && leg == 0)
		{
			std::sort(trackParts, trackParts + nParts, GPUTPCGMMerger_CompareParts);
		}

		GPUTPCSliceOutCluster trackClusters[kMaxClusters];
		uchar2 clA[kMaxClusters];
		nHits = 0;
		for( int ipart=0; ipart<nParts; ipart++ )
		{
			const GPUTPCGMSliceTrack *t = trackParts[ipart];
			if (DEBUG) printf("Collect Track %d Part %d QPt %f DzDs %f\n", fNOutputTracks, ipart, t->QPt(), t->DzDs());
			int nTrackHits = t->NClusters();
			const GPUTPCSliceOutCluster *c= t->OrigTrack()->Clusters();
			GPUTPCSliceOutCluster *c2 = trackClusters + nHits + nTrackHits-1;
			for( int i=0; i<nTrackHits; i++, c++, c2-- )
			{
			  *c2 = *c;
			  clA[nHits].x = t->Slice();
			  clA[nHits++].y = t->Leg();
			}
		}
		if ( nHits < TRACKLET_SELECTOR_MIN_HITS(track.QPt()) ) continue;

		int ordered = leg == 0;
		if (ordered)
		{
			for( int i=1; i<nHits; i++ )
			{
				if ( trackClusters[i].GetX() > trackClusters[i-1].GetX() || trackClusters[i].GetId() == trackClusters[i - 1].GetId())
				{
					ordered = 0;
					break;
				}
			}
		}
		int firstTrackIndex = 0;
		int lastTrackIndex = nParts - 1;
		if (ordered == 0)
		{
			int nTmpHits = 0;
			GPUTPCSliceOutCluster trackClustersUnsorted[kMaxClusters];
			uchar2 clAUnsorted[kMaxClusters];
			int clusterIndices[kMaxClusters];
			for (int i = 0;i < nHits;i++)
			{
				trackClustersUnsorted[i] = trackClusters[i];
				clAUnsorted[i] = clA[i];
				clusterIndices[i] = i;
			}

			if (leg > 0)
			{
				//Find QPt and DzDs for the segment closest to the vertex, if low/mid Pt
				float baseZ = 1e9;
				unsigned char baseLeg = 0;
				const float factor = trackParts[0]->CSide() ? -1.f : 1.f;
				for (int i = 0;i < nParts;i++)
				{
				  if(trackParts[i]->Leg() == 0 || trackParts[i]->Leg() == leg)
				  {
					float z = CAMath::Min(trackParts[i]->OrigTrack()->Clusters()[0].GetZ() * factor, trackParts[i]->OrigTrack()->Clusters()[trackParts[i]->OrigTrack()->NClusters() - 1].GetZ() * factor);
					if (z < baseZ)
					{
						baseZ = z;
						baseLeg = trackParts[i]->Leg();
					}
				  }
				}
				int iLongest = 1e9;
				int length = 0;
				for (int i = (baseLeg ? (nParts - 1) : 0);baseLeg ? (i >= 0) : (i < nParts);baseLeg ? i-- : i++)
				{
					if (trackParts[i]->Leg() != baseLeg) break;
					if (trackParts[i]->OrigTrack()->NClusters() > length)
					{
						iLongest = i;
						length = trackParts[i]->OrigTrack()->NClusters();
					}
				}
				bool outwards = (trackParts[iLongest]->OrigTrack()->Clusters()[0].GetZ() > trackParts[iLongest]->OrigTrack()->Clusters()[trackParts[iLongest]->OrigTrack()->NClusters() - 1].GetZ()) ^ trackParts[iLongest]->CSide();

				GPUTPCGMMerger_CompareClusterIdsLooper::clcomparestruct clusterSort[kMaxClusters];
				for (int iPart = 0;iPart < nParts;iPart++)
				{
					const GPUTPCGMSliceTrack *t = trackParts[iPart];
					int nTrackHits = t->NClusters();
					for (int j = 0;j < nTrackHits;j++)
					{
						int i = nTmpHits + j;
						clusterSort[i].leg = t->Leg();
					}
					nTmpHits += nTrackHits;
				}

			std::sort(clusterIndices, clusterIndices + nHits, GPUTPCGMMerger_CompareClusterIdsLooper(baseLeg, outwards, trackClusters, clusterSort));
			}
			else
			{
				std::sort(clusterIndices, clusterIndices + nHits, GPUTPCGMMerger_CompareClusterIds(trackClusters));
			}
			nTmpHits = 0;
			firstTrackIndex = lastTrackIndex = -1;
			for (int i = 0;i < nParts;i++)
			{
				nTmpHits += trackParts[i]->NClusters();
				if (nTmpHits > clusterIndices[0] && firstTrackIndex == -1) firstTrackIndex = i;
				if (nTmpHits > clusterIndices[nHits - 1] && lastTrackIndex == -1) lastTrackIndex = i;
			}

			int nFilteredHits = 0;
			int indPrev = -1;
			for (int i = 0;i < nHits;i++)
			{
				int ind = clusterIndices[i];
				if(indPrev >= 0 && trackClustersUnsorted[ind].GetId() == trackClustersUnsorted[indPrev].GetId()) continue;
				indPrev = ind;
				trackClusters[nFilteredHits] = trackClustersUnsorted[ind];
				clA[nFilteredHits] = clAUnsorted[ind];
				nFilteredHits++;
			}
			nHits = nFilteredHits;
		}

		GPUTPCGMMergedTrackHit *cl = fClusters + nOutTrackClusters;
		int* clid = fGlobalClusterIDs + nOutTrackClusters;
		for( int i=0; i<nHits; i++ )
		{
			cl[i].fX = trackClusters[i].GetX();
			cl[i].fY = trackClusters[i].GetY();
			cl[i].fZ = trackClusters[i].GetZ();
			cl[i].fRow = trackClusters[i].GetRow();
			if (!mCAParam->rec.NonConsecutiveIDs) //We already have global consecutive numbers from the slice tracker, and we need to keep them for late cluster attachment
			{
				cl[i].fNum = trackClusters[i].GetId();
			}
			else //Produce consecutive numbers for shared cluster flagging
			{
				cl[i].fNum = nOutTrackClusters + i;
				clid[i] = trackClusters[i].GetId();
			}
			cl[i].fAmp = trackClusters[i].GetAmp();
			cl[i].fState = trackClusters[i].GetFlags() & GPUTPCGMMergedTrackHit::hwcfFlags; //Only allow edge and deconvoluted flags
			cl[i].fSlice = clA[i].x;
			cl[i].fLeg = clA[i].y;
#ifdef GMPropagatePadRowTime
			cl[i].fPad = trackClusters[i].fPad;
			cl[i].fTime = trackClusters[i].fTime;
#endif
		}

		GPUTPCGMMergedTrack &mergedTrack = fOutputTracks[fNOutputTracks];
		mergedTrack.SetFlags(0);
		mergedTrack.SetOK(1);
		mergedTrack.SetLooper(leg > 0);
		mergedTrack.SetNClusters( nHits );
		mergedTrack.SetFirstClusterRef( nOutTrackClusters );
		GPUTPCGMTrackParam &p1 = mergedTrack.Param();
		const GPUTPCGMSliceTrack &p2 = *trackParts[firstTrackIndex];
		mergedTrack.SetCSide(p2.CSide());

		GPUTPCGMBorderTrack b;
		if (p2.TransportToX(cl[0].fX, mCAParam->ConstBz, b, GPUCA_MAX_SIN_PHI, false))
		{
			p1.X() = cl[0].fX;
			p1.Y() = b.Par()[0];
			p1.Z() = b.Par()[1];
			p1.SinPhi() = b.Par()[2];
		}
		else
		{
			p1.X() = p2.X();
			p1.Y() = p2.Y();
			p1.Z() = p2.Z();
			p1.SinPhi() = p2.SinPhi();
		}
		p1.ZOffset() = p2.ZOffset();
		p1.DzDs()  = p2.DzDs();
		p1.QPt()  = p2.QPt();
		mergedTrack.SetAlpha( p2.Alpha() );

		//if (nParts > 1) printf("Merged %d: QPt %f %d parts %d hits\n", fNOutputTracks, p1.QPt(), nParts, nHits);

		/*if (GPUQA::QAAvailable() && mRec->GetQA() && mRec->GetQA()->SuppressTrack(fNOutputTracks))
		{
			mergedTrack.SetOK(0);
			mergedTrack.SetNClusters(0);
		}*/

		bool CEside = (mergedTrack.CSide() != 0) ^ (cl[0].fZ > cl[nHits - 1].fZ);
		if (mergedTrack.NClusters() && mergedTrack.OK()) MergeCEFill(trackParts[CEside ? lastTrackIndex : firstTrackIndex], cl[CEside ? (nHits - 1) : 0], fNOutputTracks);
		fNOutputTracks++;
		nOutTrackClusters += nHits;
	}
	fNOutputTrackClusters = nOutTrackClusters;
}

void GPUTPCGMMerger::PrepareClustersForFit()
{
	unsigned int maxId = 0;
	if (mCAParam->rec.NonConsecutiveIDs)
	{
		maxId = fNOutputTrackClusters;
	}
	else
	{
		for (int i = 0;i < fgkNSlices;i++)
		{
			maxId += fSliceTrackers[i].NHitsTotal();
		}
	}
	if (maxId > fNMaxClusters) throw std::runtime_error("fNMaxClusters too small");
	fMaxID = maxId;

	unsigned int* trackSort = (unsigned int*) fTmpMem;
	unsigned char* sharedCount = (unsigned char*) (trackSort + fNOutputTracks);

	if (!mCAParam->rec.NonConsecutiveIDs)
	{
		for (int i = 0;i < fNOutputTracks;i++) trackSort[i] = i;
		std::sort(trackSort, trackSort + fNOutputTracks, GPUTPCGMMerger_CompareTracks(fOutputTracks));
		memset(fClusterAttachment, 0, maxId * sizeof(fClusterAttachment[0]));
		for (int i = 0;i < fNOutputTracks;i++) fTrackOrder[trackSort[i]] = i;
		for (int i = 0;i < fNOutputTrackClusters;i++) fClusterAttachment[fClusters[i].fNum] = attachAttached | attachGood;
		for (unsigned int k = 0;k < maxId;k++) sharedCount[k] = 0;
		for (int k = 0;k < fNOutputTrackClusters;k++)
		{
			sharedCount[fClusters[k].fNum] = (sharedCount[fClusters[k].fNum] << 1) | 1;
		}
		for (int k = 0;k < fNOutputTrackClusters;k++)
		{
			if (sharedCount[fClusters[k].fNum] > 1) fClusters[k].fState |= GPUTPCGMMergedTrackHit::flagShared;
		}
	}
}

int GPUTPCGMMerger::CheckSlices()
{
	for (int i = 0; i < fgkNSlices; i++)
	{
		if (fkSlices[i] == NULL)
		{
			printf("Slice %d missing\n", i);
			return 1;
		}
	}
	return 0;
}

void GPUTPCGMMerger::Finalize()
{
	if (mCAParam->rec.NonConsecutiveIDs)
	{
		for (int i = 0;i < fNOutputTrackClusters;i++)
		{
			fClusters[i].fNum = fGlobalClusterIDs[i];
		}
	}
	else
	{
		int* trkOrderReverse = (int*) fTmpMem;
		for (int i = 0;i < fNOutputTracks;i++) trkOrderReverse[fTrackOrder[i]] = i;
		for (int i = 0;i < fNOutputTrackClusters;i++) fClusterAttachment[fClusters[i].fNum] = 0; //Reset adjacent attachment for attached clusters, set correctly below
		for (int i = 0;i < fNOutputTracks;i++)
		{
			const GPUTPCGMMergedTrack& trk = fOutputTracks[i];
			if (!trk.OK() || trk.NClusters() == 0) continue;
			char goodLeg = fClusters[trk.FirstClusterRef() + trk.NClusters() - 1].fLeg;
			for (unsigned int j = 0;j < trk.NClusters();j++)
			{
				int id = fClusters[trk.FirstClusterRef() + j].fNum;
				int weight = fTrackOrder[i] | attachAttached;
				unsigned char clusterState = fClusters[trk.FirstClusterRef() + j].fState;
				if (!(clusterState & GPUTPCGMMergedTrackHit::flagReject)) weight |= attachGood;
				else if (clusterState & GPUTPCGMMergedTrackHit::flagNotFit) weight |= attachHighIncl;
				if (fClusters[trk.FirstClusterRef() + j].fLeg == goodLeg) weight |= attachGoodLeg;
				CAMath::AtomicMax(&fClusterAttachment[id], weight);
			}
		}
		for (int i = 0;i < fMaxID;i++) if (fClusterAttachment[i] != 0)
		{
			fClusterAttachment[i] = (fClusterAttachment[i] & attachFlagMask) | trkOrderReverse[fClusterAttachment[i] & attachTrackMask];
		}
	}
	fTrackOrder = NULL;
}
