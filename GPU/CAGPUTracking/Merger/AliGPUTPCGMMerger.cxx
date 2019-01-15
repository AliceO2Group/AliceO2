// $Id: AliGPUTPCGMMerger.cxx 30732 2009-01-22 23:02:02Z sgorbuno $
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

#include <stdio.h>
#include <string.h>
#include "AliGPUTPCSliceOutTrack.h"
#include "AliGPUTPCTracker.h"
#include "AliGPUTPCClusterData.h"
#include "AliGPUTPCTrackParam.h"
#include "AliGPUTPCGMCluster.h"
#include "AliGPUTPCGMPolynomialField.h"
#include "AliGPUTPCGMPolynomialFieldManager.h"
#include "AliGPUTPCGMMerger.h"
#include "AliGPUReconstruction.h"
#include "AliGPUReconstructionDeviceBase.h"
#include "AliGPUCAQA.h"

#include "AliTPCCommonMath.h"

#include "AliGPUTPCTrackParam.h"
#include "AliGPUTPCSliceOutput.h"
#include "AliGPUTPCGMMergedTrack.h"
#include "AliGPUCAParam.h"
#include "AliGPUTPCTrackLinearisation.h"

#include "AliGPUTPCGMTrackParam.h"
#include "AliGPUTPCGMSliceTrack.h"
#include "AliGPUTPCGMBorderTrack.h"
#include <cmath>

#include <algorithm>

#include "AliGPUTPCGPUConfig.h"
#include "MemoryAssignmentHelpers.h"

#define DEBUG 0

static constexpr int kMaxParts = 400;
static constexpr int kMaxClusters = 1000;

//#define OFFLINE_FITTER

#if !defined(GPUCA_ALIROOT_LIB) || defined(GPUCA_GPUCODE)
#undef OFFLINE_FITTER
#endif

#if defined(OFFLINE_FITTER)
#include "AliGPUTPCGMOfflineFitter.h"
AliGPUTPCGMOfflineFitter gOfflineFitter;
#endif

AliGPUTPCGMMerger::AliGPUTPCGMMerger() :
	fField(),
	fSliceParam(NULL),
	fTrackLinks(NULL),
	fNOutputTracks( 0 ),
	fNOutputTrackClusters( 0 ),
	fNMaxOutputTrackClusters( 0 ),
	fOutputTracks( 0 ),
	fSliceTrackInfos( 0 ),
	fMaxSliceTracks(0),
	fClusters(NULL),
	fGlobalClusterIDs(NULL),
	fClusterAttachment(NULL),
	fMaxID(0),
	fTrackOrder(NULL),
	fBorderMemory(0),
	fBorderRangeMemory(0),
	fSliceTrackers(NULL),
	fNClusters(0)
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
	for (int i = 0; i < fgkNSlices; i++) fkSlices[i] = NULL;
}

AliGPUTPCGMMerger::~AliGPUTPCGMMerger()
{
	//* destructor
	ClearMemory();
}

//DEBUG CODE
#if defined(GPUCA_MERGER_BY_MC_LABEL) || DEBUG == 1
void AliGPUTPCGMMerger::CheckMergedTracks()
{
	std::vector<bool> trkUsed(SliceTrackInfoLocalTotal());
	for (int i = 0;i < SliceTrackInfoLocalTotal();i++) trkUsed[i] = false;

	for ( int itr = 0; itr < SliceTrackInfoLocalTotal(); itr++ )
	{
		AliGPUTPCGMSliceTrack &track = fSliceTrackInfos[itr];
		if ( track.PrevSegmentNeighbour() >= 0 ) continue;
		if ( track.PrevNeighbour() >= 0 ) continue;
		int leg = 0;
		AliGPUTPCGMSliceTrack *trbase = &track, *tr = &track;
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

int AliGPUTPCGMMerger::GetTrackLabel(AliGPUTPCGMBorderTrack &trk)
{
	AliGPUTPCGMSliceTrack *track = &fSliceTrackInfos[trk.TrackID()];
	const AliGPUTPCSliceOutCluster *clusters = track->OrigTrack()->Clusters();
	int nClusters = track->OrigTrack()->NClusters();
	std::vector<int> labels;
	AliGPUTPCStandaloneFramework &hlt = AliGPUTPCStandaloneFramework::Instance();
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

void AliGPUTPCGMMerger::Initialize(AliGPUReconstruction* rec, AliGPUProcessor::ProcessorType type, long int TimeStamp, bool isMC)
{
	InitGPUProcessor(rec, type);
	fSliceParam = &mRec->GetParam();
	if (mRec->GetDeviceType() == AliGPUReconstruction::DeviceType::CUDA)
	{
		fSliceTrackers = nullptr;
	}
	else
	{
		fSliceTrackers = mRec->GetTPCSliceTrackers();
	}

	if (fSliceParam->AssumeConstantBz) AliGPUTPCGMPolynomialFieldManager::GetPolynomialField(AliGPUTPCGMPolynomialFieldManager::kUniform, fSliceParam->BzkG, fField);
	else AliGPUTPCGMPolynomialFieldManager::GetPolynomialField(fSliceParam->BzkG, fField);

#if (defined(OFFLINE_FITTER))
	#error NOT WORKING, TIMESTAMP NOT SET
	gOfflineFitter.Initialize(*fSliceParam, TimeStamp, isMC);
#else
	(void) (TimeStamp + isMC); //Suppress warning
#endif
}

void AliGPUTPCGMMerger::Clear()
{
	ClearMemory();
}

void AliGPUTPCGMMerger::ClearMemory()
{
	delete[] fTrackLinks;
	delete[] fSliceTrackInfos;
	if (mGPUProcessorType == PROCESSOR_TYPE_CPU)
	{
		delete[] fOutputTracks;
		delete[] fClusters;
	}
	delete[] fGlobalClusterIDs;
	delete[] fBorderMemory;
	delete[] fBorderRangeMemory;

	delete[] fTrackOrder;
	delete[] fClusterAttachment;

	fTrackLinks = NULL;
	fNOutputTracks = 0;
	fOutputTracks = NULL;
	fSliceTrackInfos = NULL;
	fMaxSliceTracks = 0;
	fClusters = NULL;
	fGlobalClusterIDs = NULL;
	fBorderMemory = NULL;
	fBorderRangeMemory = NULL;
	fTrackOrder = NULL;
	fClusterAttachment = NULL;
}

void AliGPUTPCGMMerger::SetSliceData(int index, const AliGPUTPCSliceOutput *sliceData)
{
	fkSlices[index] = sliceData;
}

bool AliGPUTPCGMMerger::Reconstruct()
{
	//* main merging routine
	for (int i = 0; i < fgkNSlices; i++)
	{
		if (fkSlices[i] == NULL)
		{
			printf("Slice %d missing\n", i);
			return false;
		}
	}
	int nIter = 1;
	HighResTimer timer;
	static double times[8] = {};
	static int nCount = 0;
	if (fSliceParam->resetTimers || !GPUCA_TIMING_SUM)
	{
		for (unsigned int k = 0; k < sizeof(times) / sizeof(times[0]); k++) times[k] = 0;
		nCount = 0;
	}
	//cout<<"Merger..."<<endl;
	for (int iter = 0; iter < nIter; iter++)
	{
		if (!AllocateMemory()) return false;
		timer.ResetStart();
		UnpackSlices();
		times[0] += timer.GetCurrentElapsedTime(true);
		MergeWithingSlices();
		times[1] += timer.GetCurrentElapsedTime(true);
		MergeSlices();
		times[2] += timer.GetCurrentElapsedTime(true);
		MergeCEInit();
		times[3] += timer.GetCurrentElapsedTime(true);
		CollectMergedTracks();
		times[4] += timer.GetCurrentElapsedTime(true);
		MergeCE();
		times[3] += timer.GetCurrentElapsedTime(true);
		PrepareClustersForFit();
		times[5] += timer.GetCurrentElapsedTime(true);
		Refit(fSliceParam->resetTimers);
		times[6] += timer.GetCurrentElapsedTime(true);
		Finalize();
		times[7] += timer.GetCurrentElapsedTime(true);
		nCount++;
		if (fSliceParam->debugLevel > 0)
		{
			printf("Merge Time:\tUnpack Slices:\t%1.0f us\n", times[0] * 1000000 / nCount);
			printf("\t\tMerge Within:\t%1.0f us\n", times[1] * 1000000 / nCount);
			printf("\t\tMerge Slices:\t%1.0f us\n", times[2] * 1000000 / nCount);
			printf("\t\tMerge CE:\t%1.0f us\n", times[3] * 1000000 / nCount);
			printf("\t\tCollect:\t%1.0f us\n", times[4] * 1000000 / nCount);
			printf("\t\tClusters:\t%1.0f us\n", times[5] * 1000000 / nCount);
			printf("\t\tRefit:\t\t%1.0f us\n", times[6] * 1000000 / nCount);
			printf("\t\tFinalize:\t%1.0f us\n", times[7] * 1000000 / nCount);
		}
	}
	return true;
}

bool AliGPUTPCGMMerger::AllocateMemory()
{
	//* memory allocation

	ClearMemory();

	int nTracks = 0;
	fNClusters = 0;
	fMaxSliceTracks = 0;

	for (int iSlice = 0; iSlice < fgkNSlices; iSlice++)
	{
		nTracks += fkSlices[iSlice]->NTracks();
		fNClusters += fkSlices[iSlice]->NTrackClusters();
		if (fMaxSliceTracks < fkSlices[iSlice]->NTracks()) fMaxSliceTracks = fkSlices[iSlice]->NTracks();
	}
	fNMaxOutputTrackClusters = fNClusters * 1.1f + 1000;

	//cout<<"\nMerger: input "<<nTracks<<" tracks, "<<nClusters<<" clusters"<<endl;

	fSliceTrackInfos = new AliGPUTPCGMSliceTrack[nTracks];
	if (mGPUProcessorType != PROCESSOR_TYPE_CPU)
	{
		char *hostBaseMem = dynamic_cast<const AliGPUReconstructionDeviceBase *>(mRec)->MergerHostMemory();
		char *basemem = hostBaseMem;
		AssignMemory(fClusters, basemem, fNMaxOutputTrackClusters);
		AssignMemory(fOutputTracks, basemem, nTracks);
		if ((size_t)(basemem - hostBaseMem) > GPUCA_GPU_MERGER_MEMORY)
		{
			printf("Insufficient memory for track merger %lld > %lld\n", (long long int) (basemem - hostBaseMem), (long long int) GPUCA_GPU_MERGER_MEMORY);
			return (false);
		}
	}
	else
	{
		fOutputTracks = new AliGPUTPCGMMergedTrack[nTracks];
		fClusters = new AliGPUTPCGMMergedTrackHit[fNMaxOutputTrackClusters];
	}
	if (!fSliceTrackers) fGlobalClusterIDs = new int[fNMaxOutputTrackClusters];
	fBorderMemory = new AliGPUTPCGMBorderTrack[nTracks];
	fBorderRangeMemory = new AliGPUTPCGMBorderTrack::Range[2 * nTracks];
	nTracks = 0;
	for (int iSlice = 0; iSlice < fgkNSlices; iSlice++)
	{
		fBorder[iSlice] = fBorderMemory + nTracks;
		fBorderRange[iSlice] = fBorderRangeMemory + 2 * nTracks;
		nTracks += fkSlices[iSlice]->NTracks();
	}
	fTrackLinks = new int[nTracks];
	return (fOutputTracks != NULL && fSliceTrackInfos != NULL && fClusters != NULL && fBorderMemory != NULL && fBorderRangeMemory != NULL && fTrackLinks != NULL);
}

void AliGPUTPCGMMerger::ClearTrackLinks(int n)
{
	for (int i = 0; i < n; i++) fTrackLinks[i] = -1;
}

void AliGPUTPCGMMerger::UnpackSlices()
{
	//* unpack the cluster information from the slice tracks and initialize track info array

	int nTracksCurrent = 0;

	const AliGPUTPCSliceOutTrack *firstGlobalTracks[fgkNSlices];

	int maxSliceTracks = 0;
	for (int i = 0; i < fgkNSlices; i++)
	{
		firstGlobalTracks[i] = 0;
		if (fkSlices[i]->NLocalTracks() > maxSliceTracks) maxSliceTracks = fkSlices[i]->NLocalTracks();
	}

	int *TrackIds = new int[maxSliceTracks * fgkNSlices];
	for (int i = 0; i < maxSliceTracks * fgkNSlices; i++) TrackIds[i] = -1;

	for (int iSlice = 0; iSlice < fgkNSlices; iSlice++)
	{

		fSliceTrackInfoIndex[iSlice] = nTracksCurrent;

		float alpha = fSliceParam->Alpha(iSlice);
		const AliGPUTPCSliceOutput &slice = *(fkSlices[iSlice]);
		const AliGPUTPCSliceOutTrack *sliceTr = slice.GetFirstTrack();

		for (int itr = 0; itr < slice.NLocalTracks(); itr++, sliceTr = sliceTr->GetNextTrack())
		{
			AliGPUTPCGMSliceTrack &track = fSliceTrackInfos[nTracksCurrent];
			track.Set(sliceTr, alpha, iSlice);
			if (!track.FilterErrors(*fSliceParam, GPUCA_MAX_SIN_PHI, 0.1f)) continue;
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

		float alpha = fSliceParam->Alpha(iSlice);
		const AliGPUTPCSliceOutput &slice = *(fkSlices[iSlice]);
		const AliGPUTPCSliceOutTrack *sliceTr = firstGlobalTracks[iSlice];
		for (int itr = slice.NLocalTracks(); itr < slice.NTracks(); itr++, sliceTr = sliceTr->GetNextTrack())
		{
			int localId = TrackIds[(sliceTr->LocalTrackId() >> 24) * maxSliceTracks + (sliceTr->LocalTrackId() & 0xFFFFFF)];
			if (localId == -1) continue;
			AliGPUTPCGMSliceTrack &track = fSliceTrackInfos[nTracksCurrent];
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

	delete[] TrackIds;
}

void AliGPUTPCGMMerger::MakeBorderTracks(int iSlice, int iBorder, AliGPUTPCGMBorderTrack B[], int &nB, bool fromOrig)
{
	//* prepare slice tracks for merging with next/previous/same sector
	//* each track transported to the border line

	float fieldBz = fSliceParam->ConstBz;

	nB = 0;

	float dAlpha = fSliceParam->DAlpha / 2;
	float x0 = 0;

	if ( iBorder == 0 ) { // transport to the left edge of the sector and rotate horisontally
		dAlpha = dAlpha - CAMath::Pi() / 2 ;
	} else if ( iBorder == 1 ) { // transport to the right edge of the sector and rotate horisontally
		dAlpha = -dAlpha - CAMath::Pi() / 2 ;
	} else if ( iBorder == 2 ) { // transport to the middle of the sector and rotate vertically to the border on the left
		x0 = fSliceParam->RowX[ 63 ];
	} else if ( iBorder == 3 ) { // transport to the middle of the sector and rotate vertically to the border on the right
		dAlpha = -dAlpha;
		x0 = fSliceParam->RowX[ 63 ];
	} else if ( iBorder == 4 ) { // transport to the middle of the sßector, w/o rotation
		dAlpha = 0;
		x0 = fSliceParam->RowX[ 63 ];
	}

	const float maxSin = CAMath::Sin(60. / 180. * CAMath::Pi());
	float cosAlpha = CAMath::Cos(dAlpha);
	float sinAlpha = CAMath::Sin(dAlpha);

	AliGPUTPCGMSliceTrack trackTmp;
	for (int itr = SliceTrackInfoFirst(iSlice); itr < SliceTrackInfoLast(iSlice); itr++)
	{

		const AliGPUTPCGMSliceTrack *track = &fSliceTrackInfos[itr];

		if (track->PrevSegmentNeighbour() >= 0 && track->Slice() == fSliceTrackInfos[track->PrevSegmentNeighbour()].Slice()) continue;
		if (fromOrig)
		{
			if (fabs(track->QPt()) < MERGE_LOOPER_QPT_LIMIT) continue;
			const AliGPUTPCGMSliceTrack *trackMin = track;
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
			if (fabs(track->QPt()) < MERGE_HORIZONTAL_DOUBLE_QPT_LIMIT)
			{
				if (iBorder == 0 && track->NextNeighbour() >= 0) continue;
				if (iBorder == 1 && track->PrevNeighbour() >= 0) continue;
			}
		}
		AliGPUTPCGMBorderTrack &b = B[nB];

		if (track->TransportToXAlpha(x0, sinAlpha, cosAlpha, fieldBz, b, maxSin))
		{
			b.SetTrackID(itr);
			b.SetNClusters(track->NClusters());
			for (int i = 0; i < 4; i++)
				if (fabs(b.Cov()[i]) >= 5.0) b.SetCov(i, 5.0);
			if (fabs(b.Cov()[4]) >= 0.5) b.SetCov(4, 0.5);
			nB++;
		}
	}
}

void AliGPUTPCGMMerger::MergeBorderTracks(int iSlice1, AliGPUTPCGMBorderTrack B1[], int N1, int iSlice2, AliGPUTPCGMBorderTrack B2[], int N2, int crossCE)
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

	AliGPUTPCGMBorderTrack::Range *range1 = fBorderRange[iSlice1];
	AliGPUTPCGMBorderTrack::Range *range2 = fBorderRange[iSlice2] + N2;

	bool sameSlice = (iSlice1 == iSlice2);
	{
		for ( int itr = 0; itr < N1; itr++ ){
			AliGPUTPCGMBorderTrack &b = B1[itr];
			float d = CAMath::Max(0.5f, 3.5*sqrt(b.Cov()[1]));
			if (fabs(b.Par()[4]) >= 20) d *= 2;
			else if (d > 3) d = 3;
			if (DEBUG) {printf("  Input Slice 1 %d Track %d: ", iSlice1, itr); for (int i = 0;i < 5;i++) {printf("%8.3f ", b.Par()[i]);} printf(" - "); for (int i = 0;i < 5;i++) {printf("%8.3f ", b.Cov()[i]);} printf(" - D %8.3f\n", d);}
			range1[itr].fId = itr;
			range1[itr].fMin = b.Par()[1] + b.ZOffset() - d;
			range1[itr].fMax = b.Par()[1] + b.ZOffset() + d;
		}
		std::sort(range1,range1+N1,AliGPUTPCGMBorderTrack::Range::CompMin);
		if(sameSlice)
		{
			for(int i=0; i<N1; i++) range2[i]= range1[i];
			std::sort(range2,range2+N1,AliGPUTPCGMBorderTrack::Range::CompMax);
			N2 = N1;
			B2 = B1;
		}
		else
		{
			for ( int itr = 0;itr < N2;itr++)
			{
				AliGPUTPCGMBorderTrack &b = B2[itr];
				float d = CAMath::Max(0.5f, 3.5*sqrt(b.Cov()[1]));
				if (fabs(b.Par()[4]) >= 20) d *= 2;
				else if (d > 3) d = 3;
				if (DEBUG) {printf("  Input Slice 2 %d Track %d: ", iSlice2, itr);for (int i = 0;i < 5;i++) {printf("%8.3f ", b.Par()[i]);}printf(" - ");for (int i = 0;i < 5;i++) {printf("%8.3f ", b.Cov()[i]);}printf(" - D %8.3f\n", d);}
				range2[itr].fId = itr;
				range2[itr].fMin = b.Par()[1] + b.ZOffset() - d;
				range2[itr].fMax = b.Par()[1] + b.ZOffset() + d;
			}
			std::sort(range2,range2+N2,AliGPUTPCGMBorderTrack::Range::CompMax);
		}
	}

	int i2 = 0;
	for (int i1 = 0;i1 < N1;i1++)
	{

		AliGPUTPCGMBorderTrack::Range r1 = range1[i1];
		while( i2<N2 && range2[i2].fMax< r1.fMin ) i2++;

		AliGPUTPCGMBorderTrack &b1 = B1[r1.fId];
		if ( b1.NClusters() < minNPartHits ) continue;
		int iBest2 = -1;
		int lBest2 = 0;
		statAll++;
		for(int k2 = i2;k2<N2;k2++)
		{
			AliGPUTPCGMBorderTrack::Range r2 = range2[k2];
			if( r2.fMin > r1.fMax ) break;
			if( sameSlice && (r1.fId >= r2.fId) ) continue;
			// do check

			AliGPUTPCGMBorderTrack &b2 = B2[r2.fId];
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
				float fys = fabs(b1.Par()[4]) < 20 ? factor2ys : (2. * factor2ys);
				float fzt = fabs(b1.Par()[4]) < 20 ? factor2zt : (2. * factor2zt);
				if( !b1.CheckChi2YS(b2, fys ) ) {if (DEBUG) {printf("!YS\n");}continue;}
				if( !b1.CheckChi2ZT(b2, fzt ) ) {if (DEBUG) {printf("!ZT\n");}continue;}
				if (fabs(b1.Par()[4]) < 20)
				{
					if ( b2.NClusters() < minNPartHits ) {if (DEBUG) {printf("!NCl2\n");}continue;}
					if ( b1.NClusters() + b2.NClusters() < minNTotalHits ) {if (DEBUG) {printf("!NCl3\n");}continue;}
				}
				if (DEBUG) printf("OK: dZ %8.3f D1 %8.3f D2 %8.3f\n", fabs(b1.Par()[1] - b2.Par()[1]), 3.5*sqrt(b1.Cov()[1]), 3.5*sqrt(b2.Cov()[1]));
			} //DEBUG CODE, match by MC label
			lBest2 = b2.NClusters();
			iBest2 = b2.TrackID();
		}

		if ( iBest2 < 0 ) continue;
		statMerged++;

		if (DEBUG) printf("Found match %d %d\n", b1.TrackID(), iBest2);

		fTrackLinks[b1.TrackID()] = iBest2;
	}
	//printf("STAT: slices %d, %d: all %d merged %d\n", iSlice1, iSlice2, statAll, statMerged);
}

void AliGPUTPCGMMerger::MergeWithingSlices()
{
	float x0 = fSliceParam->RowX[ 63 ];
	const float maxSin = CAMath::Sin( 60. / 180.*CAMath::Pi() );

	ClearTrackLinks(SliceTrackInfoLocalTotal());
	for (int iSlice = 0;iSlice < fgkNSlices;iSlice++)
	{
		int nBord = 0;
		for ( int itr = SliceTrackInfoFirst(iSlice); itr < SliceTrackInfoLast(iSlice); itr++ )
		{
			AliGPUTPCGMSliceTrack &track = fSliceTrackInfos[itr];
			AliGPUTPCGMBorderTrack &b = fBorder[iSlice][nBord];
			if (track.TransportToX(x0, fSliceParam->ConstBz, b, maxSin))
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

void AliGPUTPCGMMerger::MergeSlices()
{
	MergeSlicesStep(2, 3, false);
	MergeSlicesStep(0, 1, false);
	MergeSlicesStep(0, 1, true);
}

void AliGPUTPCGMMerger::MergeSlicesStep(int border0, int border1, bool fromOrig)
{
	ClearTrackLinks(SliceTrackInfoLocalTotal());
	for (int iSlice = 0; iSlice < fgkNSlices; iSlice++)
	{
		int jSlice = fNextSliceInd[iSlice];
		AliGPUTPCGMBorderTrack *bCurr = fBorder[iSlice], *bNext = fBorder[jSlice];
		int nCurr = 0, nNext = 0;
		MakeBorderTracks(iSlice, border0, bCurr, nCurr, fromOrig);
		MakeBorderTracks(jSlice, border1, bNext, nNext, fromOrig);
		MergeBorderTracks(iSlice, bCurr, nCurr, jSlice, bNext, nNext, fromOrig ? -1 : 0);
	}
	ResolveMergeSlices(fromOrig, false);
}

void AliGPUTPCGMMerger::PrintMergeGraph(AliGPUTPCGMSliceTrack* trk)
{
	AliGPUTPCGMSliceTrack* orgTrack = trk;
	while (trk->PrevSegmentNeighbour() >= 0) trk = &fSliceTrackInfos[trk->PrevSegmentNeighbour()];
	AliGPUTPCGMSliceTrack* orgTower = trk;
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
			AliGPUTPCGMSliceTrack *trk2 = &fSliceTrackInfos[nextId];
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

void AliGPUTPCGMMerger::ResolveMergeSlices(bool fromOrig, bool mergeAll)
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
			AliGPUTPCGMSliceTrack &oldTrack1 = fSliceTrackInfos[old1];
			if ( oldTrack1.NClusters() < newTrack1.NClusters() ) {
				newTrack2.SetPrevNeighbour( -1, neighborType );
				oldTrack1.SetNextNeighbour( -1, neighborType );
			} else continue;
		}

		if (old2 == itr2) continue;
		if (neighborType) old2 = newTrack1.NextNeighbour(1);
		if ( old2 >= 0 )
		{
			AliGPUTPCGMSliceTrack &oldTrack2 = fSliceTrackInfos[old2];
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
		AliGPUTPCGMSliceTrack *track1 = &fSliceTrackInfos[itr];
		AliGPUTPCGMSliceTrack *track2 = &fSliceTrackInfos[itr2];
		AliGPUTPCGMSliceTrack *track1Base = track1;
		AliGPUTPCGMSliceTrack *track2Base = track2;

		bool sameSegment = fabs(track1->NClusters() > track2->NClusters() ? track1->QPt() : track2->QPt()) < 2 || track1->QPt() * track2->QPt() > 0;
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
				AliGPUTPCGMSliceTrack *tmp = track1Base;
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
				AliGPUTPCGMSliceTrack *tmp = track1;
				while (tmp->Neighbour(k) >= 0)
				{
					tmp = &fSliceTrackInfos[tmp->Neighbour(k)];
					if (tmp == track2) goto NextTrack;
				}
			}

			float z1min = track1->MinClusterZ(), z1max = track1->MaxClusterZ();
			float z2min = track2->MinClusterZ(), z2max = track2->MaxClusterZ();
			if (track1 != track1Base) {z1min = std::min(z1min, track1Base->MinClusterZ()); z1max = std::max(z1max, track1Base->MaxClusterZ());}
			if (track2 != track2Base) {z2min = std::min(z2min, track2Base->MinClusterZ()); z2max = std::max(z2max, track2Base->MaxClusterZ());}

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
					AliGPUTPCGMSliceTrack *track1new = &fSliceTrackInfos[track1->Neighbour(k)];
					AliGPUTPCGMSliceTrack *track2new = &fSliceTrackInfos[track2->Neighbour(k)];
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
					AliGPUTPCGMSliceTrack *track2new = &fSliceTrackInfos[track2->Neighbour(k)];
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

void AliGPUTPCGMMerger::MergeCEInit()
{
	for (int k = 0; k < 2; k++)
	{
		for (int i = 0; i < fgkNSlices; i++)
		{
			fBorderCETracks[k][i] = 0;
		}
	}
}

void AliGPUTPCGMMerger::MergeCEFill(const AliGPUTPCGMSliceTrack *track, const AliGPUTPCGMMergedTrackHit &cls, int itr)
{
#if defined(GPUCA_STANDALONE) && !defined(GPUCA_GPUCODE)
#ifdef MERGE_CE_ROWLIMIT
	if (cls.fRow < MERGE_CE_ROWLIMIT || cls.fRow >= GPUCA_ROW_COUNT - MERGE_CE_ROWLIMIT) return;
#endif
	if (!fSliceParam->ContinuousTracking && fabs(cls.fZ) > 10) return;
	int slice = track->Slice();
	for (int attempt = 0; attempt < 2; attempt++)
	{
		AliGPUTPCGMBorderTrack &b = attempt == 0 ? fBorder[slice][fBorderCETracks[0][slice]] : fBorder[slice][fkSlices[slice]->NTracks() - 1 - fBorderCETracks[1][slice]];
		const float x0 = attempt == 0 ? fSliceParam->RowX[63] : cls.fX;
		if(track->TransportToX(x0, fSliceParam->ConstBz, b, GPUCA_MAX_SIN_PHI_LOW))
		{
			b.SetTrackID(itr);
			b.SetNClusters(fOutputTracks[itr].NClusters());
			for (int i = 0;i < 4;i++) if (fabs(b.Cov()[i]) >= 5.0) b.SetCov(i, 5.0);
			if (fabs(b.Cov()[4]) >= 0.5) b.SetCov(4, 0.5);
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
#endif
}

void AliGPUTPCGMMerger::MergeCE()
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
			AliGPUTPCGMMergedTrack* trk[2] = {&fOutputTracks[i], &fOutputTracks[fTrackLinks[i]]};
			
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
				const float z0max = std::max(fabs(fClusters[trk[0]->FirstClusterRef()].fZ), fabs(fClusters[trk[0]->FirstClusterRef() + trk[0]->NClusters() - 1].fZ));
				const float z1max = std::max(fabs(fClusters[trk[1]->FirstClusterRef()].fZ), fabs(fClusters[trk[1]->FirstClusterRef() + trk[1]->NClusters() - 1].fZ));
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

			if (fSliceParam->ContinuousTracking)
			{
				const float z0 = trk[0]->CSide() ? std::max(fClusters[trk[0]->FirstClusterRef()].fZ, fClusters[trk[0]->FirstClusterRef() + trk[0]->NClusters() - 1].fZ) :
					std::min(fClusters[trk[0]->FirstClusterRef()].fZ, fClusters[trk[0]->FirstClusterRef() + trk[0]->NClusters() - 1].fZ);
				const float z1 = trk[1]->CSide() ? std::max(fClusters[trk[1]->FirstClusterRef()].fZ, fClusters[trk[1]->FirstClusterRef() + trk[1]->NClusters() - 1].fZ) :
					std::min(fClusters[trk[1]->FirstClusterRef()].fZ, fClusters[trk[1]->FirstClusterRef() + trk[1]->NClusters() - 1].fZ);
				float offset = fabs(z1) > fabs(z0) ? -z0 : z1;
				trk[1]->Param().Z() += trk[1]->Param().ZOffset() - offset;
				trk[1]->Param().ZOffset() = offset;
			}

			int newRef = fNOutputTrackClusters;
			for (int k = 1;k >= 0;k--)
			{
				if (reverse[k]) for (int j = trk[k]->NClusters() - 1;j >= 0;j--) fClusters[fNOutputTrackClusters++] = fClusters[trk[k]->FirstClusterRef() + j];
				else for (int j = 0;j < trk[k]->NClusters();j++) fClusters[fNOutputTrackClusters++] = fClusters[trk[k]->FirstClusterRef() + j];
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

struct AliGPUTPCGMMerger_CompareClusterIdsLooper
{
	struct clcomparestruct {unsigned char leg;};

	const unsigned char fLeg;
	const bool fOutwards;
	const AliGPUTPCSliceOutCluster* const fCmp1;
	const clcomparestruct* const fCmp2;
	AliGPUTPCGMMerger_CompareClusterIdsLooper(unsigned char leg, bool outwards, const AliGPUTPCSliceOutCluster* cmp1, const clcomparestruct* cmp2) : fLeg(leg), fOutwards(outwards), fCmp1(cmp1), fCmp2(cmp2) {}
	bool operator()(const int aa, const int bb)
	{
		const clcomparestruct& a = fCmp2[aa];
		const clcomparestruct& b = fCmp2[bb];
		const AliGPUTPCSliceOutCluster& a1 = fCmp1[aa];
		const AliGPUTPCSliceOutCluster& b1 = fCmp1[bb];
		if (a.leg != b.leg) return ((fLeg > 0) ^ (a.leg > b.leg));
		if (a1.GetX() != b1.GetX()) return((a1.GetX() > b1.GetX()) ^ ((a.leg - fLeg) & 1) ^ fOutwards);
		return false;
	}
};

struct AliGPUTPCGMMerger_CompareClusterIds
{
	const AliGPUTPCSliceOutCluster* const fCmp;
	AliGPUTPCGMMerger_CompareClusterIds(const AliGPUTPCSliceOutCluster* cmp) : fCmp(cmp) {}
	bool operator()(const int aa, const int bb)
	{
		const AliGPUTPCSliceOutCluster& a = fCmp[aa];
		const AliGPUTPCSliceOutCluster& b = fCmp[bb];
		return(a.GetX() > b.GetX());
	}
};

struct AliGPUTPCGMMerger_CompareTracks
{
	const AliGPUTPCGMMergedTrack* const fCmp;
	AliGPUTPCGMMerger_CompareTracks(AliGPUTPCGMMergedTrack* cmp) : fCmp(cmp) {}
	bool operator()(const int aa, const int bb)
	{
		const AliGPUTPCGMMergedTrack& a = fCmp[aa];
		const AliGPUTPCGMMergedTrack& b = fCmp[bb];
		return(fabs(a.GetParam().GetQPt()) > fabs(b.GetParam().GetQPt()));
	}
};

bool AliGPUTPCGMMerger_CompareParts(const AliGPUTPCGMSliceTrack* a, const AliGPUTPCGMSliceTrack* b)
{
  return(a->X() > b->X());
}

void AliGPUTPCGMMerger::CollectMergedTracks()
{
	//Resolve connections for global tracks first
	for (int iSlice = 0; iSlice < fgkNSlices; iSlice++)
	{
		for (int itr = SliceTrackInfoGlobalFirst(iSlice); itr < SliceTrackInfoGlobalLast(iSlice); itr++)
		{
			AliGPUTPCGMSliceTrack &globalTrack = fSliceTrackInfos[itr];
			AliGPUTPCGMSliceTrack &localTrack = fSliceTrackInfos[globalTrack.LocalTrackId()];
			localTrack.SetGlobalTrackId(localTrack.GlobalTrackId(0) != -1, itr);
		}
	}

	//CheckMergedTracks();

	//Now collect the merged tracks
	fNOutputTracks = 0;
	int nOutTrackClusters = 0;

	AliGPUTPCGMSliceTrack *trackParts[kMaxParts];

	for (int itr = 0; itr < SliceTrackInfoLocalTotal(); itr++)
	{

		AliGPUTPCGMSliceTrack &track = fSliceTrackInfos[itr];

		if (track.PrevSegmentNeighbour() >= 0) continue;
		if (track.PrevNeighbour() >= 0) continue;
		int nParts = 0;
		int nHits = 0;
		int leg = 0;
		AliGPUTPCGMSliceTrack *trbase = &track, *tr = &track;
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
			std::sort(trackParts, trackParts + nParts, AliGPUTPCGMMerger_CompareParts);
		}

		AliGPUTPCSliceOutCluster trackClusters[kMaxClusters];
		uchar2 clA[kMaxClusters];
		nHits = 0;
		for( int ipart=0; ipart<nParts; ipart++ )
		{
			const AliGPUTPCGMSliceTrack *t = trackParts[ipart];
			if (DEBUG) printf("Collect Track %d Part %d QPt %f DzDs %f\n", fNOutputTracks, ipart, t->QPt(), t->DzDs());
			int nTrackHits = t->NClusters();
			const AliGPUTPCSliceOutCluster *c= t->OrigTrack()->Clusters();
			AliGPUTPCSliceOutCluster *c2 = trackClusters + nHits + nTrackHits-1;
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
			AliGPUTPCSliceOutCluster trackClustersUnsorted[kMaxClusters];
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
					float z = std::min(trackParts[i]->OrigTrack()->Clusters()[0].GetZ() * factor, trackParts[i]->OrigTrack()->Clusters()[trackParts[i]->OrigTrack()->NClusters() - 1].GetZ() * factor);
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

				AliGPUTPCGMMerger_CompareClusterIdsLooper::clcomparestruct clusterSort[kMaxClusters];
				for (int iPart = 0;iPart < nParts;iPart++)
				{
					const AliGPUTPCGMSliceTrack *t = trackParts[iPart];
					int nTrackHits = t->NClusters();
					for (int j = 0;j < nTrackHits;j++)
					{
						int i = nTmpHits + j;
						clusterSort[i].leg = t->Leg();
					}
					nTmpHits += nTrackHits;
				}

			std::sort(clusterIndices, clusterIndices + nHits, AliGPUTPCGMMerger_CompareClusterIdsLooper(baseLeg, outwards, trackClusters, clusterSort));
			}
			else
			{
				std::sort(clusterIndices, clusterIndices + nHits, AliGPUTPCGMMerger_CompareClusterIds(trackClusters));
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

		AliGPUTPCGMMergedTrackHit *cl = fClusters + nOutTrackClusters;
		int* clid = fGlobalClusterIDs + nOutTrackClusters;
		for( int i=0; i<nHits; i++ )
		{
			cl[i].fX = trackClusters[i].GetX();
			cl[i].fY = trackClusters[i].GetY();
			cl[i].fZ = trackClusters[i].GetZ();
			cl[i].fRow = trackClusters[i].GetRow();
			if (fSliceTrackers) //We already have global consecutive numbers from the slice tracker, and we need to keep them for late cluster attachment
			{
				cl[i].fNum = trackClusters[i].GetId();
			}
			else //Produce consecutive numbers for shared cluster flagging
			{
				cl[i].fNum = nOutTrackClusters + i;
				clid[i] = trackClusters[i].GetId();
			}
			cl[i].fAmp = trackClusters[i].GetAmp();
			cl[i].fState = trackClusters[i].GetFlags() & AliGPUTPCGMMergedTrackHit::hwcfFlags; //Only allow edge and deconvoluted flags
			cl[i].fSlice = clA[i].x;
			cl[i].fLeg = clA[i].y;
#ifdef GMPropagatePadRowTime
			cl[i].fPad = trackClusters[i].fPad;
			cl[i].fTime = trackClusters[i].fTime;
#endif
		}

		AliGPUTPCGMMergedTrack &mergedTrack = fOutputTracks[fNOutputTracks];
		mergedTrack.SetFlags(0);
		mergedTrack.SetOK(1);
		mergedTrack.SetLooper(leg > 0);
		mergedTrack.SetNClusters( nHits );
		mergedTrack.SetFirstClusterRef( nOutTrackClusters );
		AliGPUTPCGMTrackParam &p1 = mergedTrack.Param();
		const AliGPUTPCGMSliceTrack &p2 = *trackParts[firstTrackIndex];
		mergedTrack.SetCSide(p2.CSide());

		AliGPUTPCGMBorderTrack b;
		if (p2.TransportToX(cl[0].fX, fSliceParam->ConstBz, b, GPUCA_MAX_SIN_PHI, false))
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

		if (AliGPUCAQA::QAAvailable() && mRec->GetQA() && mRec->GetQA()->SuppressTrack(fNOutputTracks))
		{
			mergedTrack.SetOK(0);
			mergedTrack.SetNClusters(0);
		}

		bool CEside = (mergedTrack.CSide() != 0) ^ (cl[0].fZ > cl[nHits - 1].fZ);
		if (mergedTrack.NClusters() && mergedTrack.OK()) MergeCEFill(trackParts[CEside ? lastTrackIndex : firstTrackIndex], cl[CEside ? (nHits - 1) : 0], fNOutputTracks);
		fNOutputTracks++;
		nOutTrackClusters += nHits;
	}
	fNOutputTrackClusters = nOutTrackClusters;
}

void AliGPUTPCGMMerger::PrepareClustersForFit()
{
	unsigned int maxId = 0;
	if (fSliceTrackers)
	{
		for (int i = 0;i < fgkNSlices;i++)
		{
			for (int j = 0;j < fSliceTrackers[i].ClusterData()->NumberOfClusters();j++)
			{
				unsigned int id = fSliceTrackers[i].ClusterData()->Id(j);
				if (id > maxId) maxId = id;
			}
		}
	}
	else
	{
		maxId = fNOutputTrackClusters;
	}
	maxId++;
	unsigned char* sharedCount = new unsigned char[maxId];

#if defined(GPUCA_STANDALONE) && !defined(GPUCA_GPUCODE)
	if (mRec->GetDeviceType() != AliGPUReconstruction::DeviceType::CUDA)
	{
		unsigned int* trackSort = new unsigned int[fNOutputTracks];
		if (fTrackOrder) delete[] fTrackOrder;
		if (fClusterAttachment) delete[] fClusterAttachment;
		fTrackOrder = new unsigned int[fNOutputTracks];
		fClusterAttachment = new int[maxId];
		fMaxID = maxId;
		for (int i = 0;i < fNOutputTracks;i++) trackSort[i] = i;
		std::sort(trackSort, trackSort + fNOutputTracks, AliGPUTPCGMMerger_CompareTracks(fOutputTracks));
		memset(fClusterAttachment, 0, maxId * sizeof(fClusterAttachment[0]));
		for (int i = 0;i < fNOutputTracks;i++) fTrackOrder[trackSort[i]] = i;
		for (int i = 0;i < fNOutputTrackClusters;i++) fClusterAttachment[fClusters[i].fNum] = attachAttached | attachGood;
		delete[] trackSort;
	}
#endif

	for (unsigned int k = 0;k < maxId;k++) sharedCount[k] = 0;
	for (int k = 0;k < fNOutputTrackClusters;k++)
	{
		sharedCount[fClusters[k].fNum] = (sharedCount[fClusters[k].fNum] << 1) | 1;
	}
	for (int k = 0;k < fNOutputTrackClusters;k++)
	{
		if (sharedCount[fClusters[k].fNum] > 1) fClusters[k].fState |= AliGPUTPCGMMergedTrackHit::flagShared;
	}
	delete[] sharedCount;
}

void AliGPUTPCGMMerger::Refit(bool resetTimers)
{
	//* final refit
#ifdef GPUCA_GPU_MERGER
	if (mRec->GetDeviceType() == AliGPUReconstruction::DeviceType::CUDA)
	{
		dynamic_cast<const AliGPUReconstructionDeviceBase*>(mRec)->RefitMergedTracks(this, resetTimers);
	}
  else
#endif
	{
#ifdef GPUCA_HAVE_OPENMP
#pragma omp parallel for num_threads(mRec->GetDeviceProcessingSettings().nThreads)
#endif
		for ( int itr = 0; itr < fNOutputTracks; itr++ )
		{
			AliGPUTPCGMTrackParam::RefitTrack(fOutputTracks[itr], itr, this, fClusters);
#if defined(OFFLINE_FITTER)
			gOfflineFitter.RefitTrack(fOutputTracks[itr], &fField, fClusters);
#endif
		}
	}
}

void AliGPUTPCGMMerger::Finalize()
{
	if (mRec->GetDeviceType() == AliGPUReconstruction::DeviceType::CUDA) return;
#if defined(GPUCA_STANDALONE) && !defined(GPUCA_GPUCODE)
	int* trkOrderReverse = new int[fNOutputTracks];
	for (int i = 0;i < fNOutputTracks;i++) trkOrderReverse[fTrackOrder[i]] = i;
	for (int i = 0;i < fNOutputTrackClusters;i++) fClusterAttachment[fClusters[i].fNum] = 0; //Reset adjacent attachment for attached clusters, set correctly below
	for (int i = 0;i < fNOutputTracks;i++)
	{
		const AliGPUTPCGMMergedTrack& trk = fOutputTracks[i];
		if (!trk.OK() || trk.NClusters() == 0) continue;
		char goodLeg = fClusters[trk.FirstClusterRef() + trk.NClusters() - 1].fLeg;
		for (int j = 0;j < trk.NClusters();j++)
		{
			int id = fClusters[trk.FirstClusterRef() + j].fNum;
			int weight = fTrackOrder[i] | attachAttached;
			unsigned char clusterState = fClusters[trk.FirstClusterRef() + j].fState;
			if (!(clusterState & AliGPUTPCGMMergedTrackHit::flagReject)) weight |= attachGood;
			else if (clusterState & AliGPUTPCGMMergedTrackHit::flagNotFit) weight |= attachHighIncl;
			if (fClusters[trk.FirstClusterRef() + j].fLeg == goodLeg) weight |= attachGoodLeg;
			CAMath::AtomicMax(&fClusterAttachment[id], weight);
		}
	}
	for (int i = 0;i < fMaxID;i++) if (fClusterAttachment[i] != 0)
	{
		fClusterAttachment[i] = (fClusterAttachment[i] & attachFlagMask) | trkOrderReverse[fClusterAttachment[i] & attachTrackMask];
	}
	delete[] trkOrderReverse;
	delete[] fTrackOrder;
	fTrackOrder = NULL;
#endif
}
