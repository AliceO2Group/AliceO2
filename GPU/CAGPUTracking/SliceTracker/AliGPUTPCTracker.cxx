// @(#) $Id$
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

#include "AliGPUTPCTracker.h"
#include "AliGPUTPCRow.h"
#include "AliGPUTPCTrack.h"
#include "AliGPUTPCTracklet.h"
#include "AliTPCCommonMath.h"

#include "AliGPUTPCHitArea.h"
#include "AliGPUTPCNeighboursFinder.h"
#include "AliGPUTPCNeighboursCleaner.h"
#include "AliGPUTPCStartHitsFinder.h"
#include "AliGPUTPCTrackletConstructor.h"
#include "AliGPUTPCTrackLinearisation.h"
#include "AliGPUTPCTrackletSelector.h"
#include "AliGPUTPCProcess.h"
#include "AliGPUTPCClusterData.h"
#include "AliGPUCAOutputControl.h"

#include "AliGPUTPCTrackParam.h"

#include "AliGPUTPCGPUConfig.h"

#if !defined(GPUCA_GPUCODE)
#include <string.h>
#include <cmath>
#include <algorithm>

#include "AliGPUReconstruction.h"
#endif

ClassImp( AliGPUTPCTracker )

#if !defined(GPUCA_GPUCODE)

AliGPUTPCTracker::AliGPUTPCTracker() :
	AliGPUProcessor(),
	fStageAtSync( NULL ),
	fLinkTmpMemory( NULL ),
	fISlice(0),
	fData(),
	fGPUDebugOut( 0 ),
	fNMaxTracks( 0 ),
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
	fOutput( 0 ),
	fOutputMemory(NULL)
{}

AliGPUTPCTracker::~AliGPUTPCTracker()
{
	// destructor
	if (mGPUProcessorType == PROCESSOR_TYPE_CPU)
	{
		if (fCommonMem) delete fCommonMem;
		if (fHitMemory) delete[] fHitMemory;
		if (fTrackletMemory) delete[] fTrackletMemory;
		if (fTrackMemory) delete[] fTrackMemory;
		fCommonMem = NULL;
		fHitMemory = fTrackletMemory = fTrackMemory = NULL;
	}
	if (fLinkTmpMemory) delete[] fLinkTmpMemory;
	if (fOutputMemory) free(fOutputMemory);
}

// ----------------------------------------------------------------------------------
void AliGPUTPCTracker::Initialize( const AliGPUCAParam *param, int iSlice )
{
	mParam = param;
	fISlice = iSlice;
	InitializeRows(mParam);

	SetupCommonMemory();
}

char* AliGPUTPCTracker::SetGPUTrackerCommonMemory(char* const pGPUMemory)
{
	//Set up common Memory Pointer for GPU Tracker
	fCommonMem = (commonMemoryStruct*) pGPUMemory;
	return(pGPUMemory + sizeof(commonMemoryStruct));
}


char* AliGPUTPCTracker::SetGPUTrackerHitsMemory(char* pGPUMemory, int MaxNHits)
{
	//Set up Hits Memory Pointers for GPU Tracker
	fHitMemory = (char*) pGPUMemory;
	SetPointersHits(MaxNHits);
	pGPUMemory += fHitMemorySize;
	AliGPUReconstruction::computePointerWithAlignment(pGPUMemory, fTrackletTmpStartHits, GPUCA_ROW_COUNT * GPUCA_GPU_MAX_ROWSTARTHITS);
	AliGPUReconstruction::computePointerWithAlignment(pGPUMemory, fRowStartHitCountOffset, GPUCA_ROW_COUNT);

	return(pGPUMemory);
}

char* AliGPUTPCTracker::SetGPUTrackerTrackletsMemory(char* pGPUMemory, int MaxNTracks)
{
	//Set up Tracklet Memory Pointers for GPU Tracker
	fTrackletMemory = (char*) pGPUMemory;
	SetPointersTracklets(MaxNTracks);
	pGPUMemory += fTrackletMemorySize;
	return(pGPUMemory);
}

char* AliGPUTPCTracker::SetGPUTrackerTracksMemory(char* pGPUMemory, int MaxNTracks, int MaxNHits )
{
	//Set up Tracks Memory Pointer for GPU Tracker
	fTrackMemory = (char*) pGPUMemory;
	SetPointersTracks(MaxNTracks, MaxNHits);
	pGPUMemory += fTrackMemorySize;

	return(pGPUMemory);
}

void AliGPUTPCTracker::SetupCommonMemory()
{
	// set up common memory

	if (mGPUProcessorType == PROCESSOR_TYPE_CPU)
	{
		if ( !fCommonMem ) {
			fCommonMem = new commonMemoryStruct;
		}

		if (fHitMemory)	delete[] fHitMemory;
		if (fTrackletMemory) delete[] fTrackletMemory;
		if (fTrackMemory) delete[] fTrackMemory;
	}

	fHitMemory = fTrackletMemory = fTrackMemory = 0;

	if (fCommonMem)
	{
		fCommonMem->fNTracklets = 0;
		fCommonMem->fNTracks = 0 ;
		fCommonMem->fNTrackHits = 0;
	}
}

int AliGPUTPCTracker::ReadEvent()
{
	// read event

	StartTimer(0);
	
	SetupCommonMemory();

	//* Convert input hits, create grids, etc.
	if (fData.InitFromClusterData())
	{
		printf("Error initializing from cluster data\n");
		return 1;
	}
	if (fData.MaxZ() > 300 && !mParam->ContinuousTracking)
	{
		printf("Need to set continuous tracking mode for data outside of the TPC volume!\n");
		return 1;
	}
	if (mGPUProcessorType == PROCESSOR_TYPE_CPU)
	{
		SetPointersHits( fData.NumberOfHits() ); // to calculate the size
		fHitMemory = reinterpret_cast<char*> ( new uint4 [ fHitMemorySize/sizeof( uint4 ) + 100] );
	}
	SetPointersHits( fData.NumberOfHits() ); // set pointers for hits
	StopTimer(0);
	return 0;
}

GPUhd() void AliGPUTPCTracker::SetPointersHits( int MaxNHits )
{
	// set all pointers to the event memory

	char *mem = fHitMemory;

	// extra arrays for tpc clusters
	AliGPUReconstruction::computePointerWithAlignment( mem, fTrackletStartHits, MaxNHits);

	// calculate the size
	fHitMemorySize = mem - fHitMemory;
}

GPUhd() void AliGPUTPCTracker::SetPointersTracklets( int MaxNTracklets )
{
	// set all pointers to the tracklets memory
	char *mem = fTrackletMemory;

	// memory for tracklets

	AliGPUReconstruction::computePointerWithAlignment( mem, fTracklets, MaxNTracklets );
#ifdef EXTERN_ROW_HITS
	AliGPUReconstruction::computePointerWithAlignment( mem, fTrackletRowHits, MaxNTracklets * GPUCA_ROW_COUNT);
#endif

	fTrackletMemorySize = mem - fTrackletMemory;
}


GPUhd() void AliGPUTPCTracker::SetPointersTracks( int MaxNTracks, int MaxNHits )
{
	// set all pointers to the tracks memory
	char *mem = fTrackMemory;

	// memory for selected tracks

	AliGPUReconstruction::computePointerWithAlignment( mem, fTracks, MaxNTracks );
	AliGPUReconstruction::computePointerWithAlignment( mem, fTrackHits, 2 * MaxNHits );

	// calculate the size

	fTrackMemorySize = mem - fTrackMemory;
}

GPUh() int AliGPUTPCTracker::CheckEmptySlice()
{
	//Check if the Slice is empty, if so set the output apropriate and tell the reconstuct procesdure to terminate
	if ( NHitsTotal() < 1 )
	{
		fCommonMem->fNTracks = fCommonMem->fNTrackHits = 0;
		WriteOutputPrepare();
		AliGPUTPCSliceOutput* useOutput = *fOutput;
		if (useOutput == NULL) return(1);
		useOutput->SetNTracks( 0 );
		useOutput->SetNTrackClusters( 0 );
		return 1;
	}
	return 0;
}

void AliGPUTPCTracker::RunNeighboursFinder()
{
	//Run the CPU Neighbours Finder
	AliGPUTPCProcess<AliGPUTPCNeighboursFinder>( GPUCA_ROW_COUNT, 1, *this );
}

void AliGPUTPCTracker::RunNeighboursCleaner()
{
	//Run the CPU Neighbours Cleaner
	AliGPUTPCProcess<AliGPUTPCNeighboursCleaner>( GPUCA_ROW_COUNT - 2, 1, *this );
}

void AliGPUTPCTracker::RunStartHitsFinder()
{
	//Run the CPU Start Hits Finder
	AliGPUTPCProcess<AliGPUTPCStartHitsFinder>( GPUCA_ROW_COUNT - 4, 1, *this );
}

void AliGPUTPCTracker::RunTrackletConstructor()
{
	//Run CPU Tracklet Constructor
	AliGPUTPCTrackletConstructor::AliGPUTPCTrackletConstructorCPU(*this);
}

void AliGPUTPCTracker::RunTrackletSelector()
{
	//Run CPU Tracklet Selector
	AliGPUTPCProcess<AliGPUTPCTrackletSelector>( 1, fCommonMem->fNTracklets, *this );
}

GPUh() void AliGPUTPCTracker::DoTracking()
{
	fCommonMem->fNTracklets = fCommonMem->fNTracks = fCommonMem->fNTrackHits = 0;

	if (mParam->debugLevel >= 6)
	{
		if (!mRec->GetDeviceProcessingSettings().comparableDebutOutput)
		{
			*fGPUDebugOut << std::endl << std::endl << "Slice: " << fISlice << std::endl;
			*fGPUDebugOut << "Slice Data:" << std::endl;
		}
		DumpSliceData(*fGPUDebugOut);
	}

	StartTimer(1);
	RunNeighboursFinder();
	StopTimer(1);

	if (mRec->GetDeviceProcessingSettings().keepAllMemory)
	{
		if (fLinkTmpMemory) delete[] fLinkTmpMemory;
		fLinkTmpMemory = new char[fData.ScratchMemorySize()];
		memcpy(fLinkTmpMemory, fData.ScratchMemory(), fData.ScratchMemorySize());
	}

	if (mParam->debugLevel >= 6) DumpLinks(*fGPUDebugOut);

	StartTimer(2);
	RunNeighboursCleaner();
	StopTimer(2);

	if (mParam->debugLevel >= 6) DumpLinks(*fGPUDebugOut);

	StartTimer(3);
	RunStartHitsFinder();
	StopTimer(3);

	if (mParam->debugLevel >= 6) DumpStartHits(*fGPUDebugOut);

	StartTimer(5);
	fData.ClearHitWeights();
	StopTimer(5);

	if (mGPUProcessorType == PROCESSOR_TYPE_CPU)
	{
		SetPointersTracklets( fCommonMem->fNTracklets * 2 ); // to calculate the size
		fTrackletMemory = reinterpret_cast<char*> ( new uint4 [ fTrackletMemorySize/sizeof( uint4 ) + 100] );
		fNMaxTracks = fCommonMem->fNTracklets * 2 + 50;
		SetPointersTracks( fNMaxTracks, NHitsTotal() ); // to calculate the size
		fTrackMemory = reinterpret_cast<char*> ( new uint4 [ fTrackMemorySize/sizeof( uint4 ) + 100] );
	}

	SetPointersTracklets( fCommonMem->fNTracklets * 2 ); // set pointers for tracklets
	SetPointersTracks( fCommonMem->fNTracklets * 2 + 50, NHitsTotal() ); // set pointers for tracks

	StartTimer(6);
	RunTrackletConstructor();
	StopTimer(6);
	if (mParam->debugLevel >= 3) printf("Slice %d, Number of tracklets: %d\n", fISlice, *NTracklets());

	if (mParam->debugLevel >= 6) DumpTrackletHits(*fGPUDebugOut);
	if (mParam->debugLevel >= 6 && !mRec->GetDeviceProcessingSettings().comparableDebutOutput) DumpHitWeights(*fGPUDebugOut);

	StartTimer(7);
	RunTrackletSelector();
	StopTimer(7);
	if (mParam->debugLevel >= 3) printf("Slice %d, Number of tracks: %d\n", fISlice, *NTracks());

	if (mParam->debugLevel >= 6) DumpTrackHits(*fGPUDebugOut);

	//std::cout<<"Memory used for slice "<<mParam->ISlice()<<" : "<<fCommonMemorySize/1024./1024.<<" + "<<fHitMemorySize/1024./1024.<<" + "<<fTrackMemorySize/1024./1024.<<" = "<<( fCommonMemorySize+fHitMemorySize+fTrackMemorySize )/1024./1024.<<" Mb "<<std::endl;
}

GPUh() void AliGPUTPCTracker::Reconstruct()
{

	if (CheckEmptySlice()) return;
	DoTracking();
}

GPUh() void AliGPUTPCTracker::ReconstructOutput()
{
	WriteOutputPrepare();
	WriteOutput();
}

GPUh() void AliGPUTPCTracker::WriteOutputPrepare()
{
	StartTimer(9);
	AliGPUTPCSliceOutput::Allocate(*fOutput, fCommonMem->fNTracks, fCommonMem->fNTrackHits, &mRec->OutputControl(), fOutputMemory);
	StopTimer(9);
}

template <class T> static inline bool SortComparison(const T& a, const T& b)
{
	return(a.fSortVal < b.fSortVal);
}

GPUh() void AliGPUTPCTracker::WriteOutput()
{
	// write output
	AliGPUTPCSliceOutput* useOutput = *fOutput;

	if (useOutput == NULL) return;

	useOutput->SetNTracks( 0 );
	useOutput->SetNLocalTracks( 0 );
	useOutput->SetNTrackClusters( 0 );
	
	if (fCommonMem->fNTracks == 0) return;
	if (fCommonMem->fNTracks > MAX_SLICE_NTRACK)
	{
		printf("Maximum number of tracks exceeded, cannot store\n");
		return;
	}
	StartTimer(9);

	int nStoredHits = 0;
	int nStoredTracks = 0;
	int nStoredLocalTracks = 0;

	AliGPUTPCSliceOutTrack *out = useOutput->FirstTrack();
	
	trackSortData* trackOrder = new trackSortData[fCommonMem->fNTracks];
	for (int i = 0;i < fCommonMem->fNTracks;i++)
	{
		trackOrder[i].fTtrack = i;
		trackOrder[i].fSortVal = fTracks[trackOrder[i].fTtrack].NHits() / 1000.f + fTracks[trackOrder[i].fTtrack].Param().GetZ() * 100.f + fTracks[trackOrder[i].fTtrack].Param().GetY();
	}
	std::sort(trackOrder, trackOrder + fCommonMem->fNLocalTracks, SortComparison<trackSortData>);
	std::sort(trackOrder + fCommonMem->fNLocalTracks, trackOrder + fCommonMem->fNTracks, SortComparison<trackSortData>);
	
	for (int iTrTmp = 0;iTrTmp < fCommonMem->fNTracks;iTrTmp++)
	{
		const int iTr = trackOrder[iTrTmp].fTtrack;
		AliGPUTPCTrack &iTrack = fTracks[iTr];

		out->SetParam( iTrack.Param() );
		out->SetLocalTrackId( iTrack.LocalTrackId() );
		int nClu = 0;
		int iID = iTrack.FirstHitID();

		for (int ith = 0;ith < iTrack.NHits();ith++)
		{
			const AliGPUTPCHitId &ic = fTrackHits[iID + ith];
			int iRow = ic.RowIndex();
			int ih = ic.HitIndex();

			const AliGPUTPCRow &row = fData.Row( iRow );
#ifdef GPUCA_ARRAY_BOUNDS_CHECKS
			if (ih >= row.NHits() || ih < 0)
			{
				printf("Array out of bounds access (Sector Row) (Hit %d / %d - NumC %d): Sector %d Row %d Index %d\n", ith, iTrack.NHits(), NHitsTotal(), fISlice, iRow, ih);
				fflush(stdout);
				continue;
			}
#endif
			int clusterIndex = fData.ClusterDataIndex( row, ih );

#ifdef GPUCA_ARRAY_BOUNDS_CHECKS
			if (clusterIndex >= NHitsTotal() || clusterIndex < 0)
			{
				printf("Array out of bounds access (Cluster Data) (Hit %d / %d - NumC %d): Sector %d Row %d Hit %d, Clusterdata Index %d\n", ith, iTrack.NHits(), NHitsTotal(), fISlice, iRow, ih, clusterIndex);
				fflush(stdout);
				continue;
			}
#endif

			float origX = fData.ClusterData()->X( clusterIndex );
			float origY = fData.ClusterData()->Y( clusterIndex );
			float origZ = fData.ClusterData()->Z( clusterIndex );
			int id = fData.ClusterData()->Id( clusterIndex );
			unsigned char flags = fData.ClusterData()->Flags( clusterIndex );
			unsigned short amp = fData.ClusterData()->Amp( clusterIndex );
			AliGPUTPCSliceOutCluster c;
			c.Set( id, iRow, flags, amp, origX, origY, origZ );
#ifdef GMPropagatePadRowTime
			c.fPad = fData.ClusterData()->GetClusterData( clusterIndex )->fPad;
			c.fTime = fData.ClusterData()->GetClusterData( clusterIndex )->fTime;
#endif
			out->SetCluster( nClu, c );
			nClu++;
		}

		nStoredTracks++;
		if (iTr < fCommonMem->fNLocalTracks) nStoredLocalTracks++;
		nStoredHits+=nClu;
		out->SetNClusters( nClu );
		out = out->NextTrack();
	}
	delete[] trackOrder;

	useOutput->SetNTracks( nStoredTracks );
	useOutput->SetNLocalTracks( nStoredLocalTracks );
	useOutput->SetNTrackClusters( nStoredHits );
	if (mParam->debugLevel >= 3) printf("Slice %d, Output: Tracks %d, local tracks %d, hits %d\n", fISlice, nStoredTracks, nStoredLocalTracks, nStoredHits);

	StopTimer(9);
}

GPUh() int AliGPUTPCTracker::PerformGlobalTrackingRun(AliGPUTPCTracker& sliceNeighbour, int iTrack, int rowIndex, float angle, int direction)
{
	/*for (int j = 0;j < fTracks[j].NHits();j++)
	{
		printf("Hit %3d: Row %3d: X %3.7lf Y %3.7lf\n", j, fTrackHits[fTracks[iTrack].FirstHitID() + j].RowIndex(), Row(fTrackHits[fTracks[iTrack].FirstHitID() + j].RowIndex()).X(),
		(float) Data().HitDataY(Row(fTrackHits[fTracks[iTrack].FirstHitID() + j].RowIndex()), fTrackHits[fTracks[iTrack].FirstHitID() + j].HitIndex()) * Row(fTrackHits[fTracks[iTrack].FirstHitID() + j].RowIndex()).HstepY() + Row(fTrackHits[fTracks[iTrack].FirstHitID() + j].RowIndex()).Grid().YMin());
		}*/

	if (sliceNeighbour.fCommonMem->fNTracklets == 0) return(0);

	AliGPUTPCTrackParam tParam;
	tParam.InitParam();
	tParam.SetCov( 0, 0.05 );
	tParam.SetCov( 2, 0.05 );
	tParam.SetCov( 5, 0.001 );
	tParam.SetCov( 9, 0.001 );
	tParam.SetCov( 14, 0.05 );
	tParam.SetParam(fTracks[iTrack].Param());

	//printf("Parameters X %f Y %f Z %f SinPhi %f DzDs %f QPt %f SignCosPhi %f\n", tParam.X(), tParam.Y(), tParam.Z(), tParam.SinPhi(), tParam.DzDs(), tParam.QPt(), tParam.SignCosPhi());
	if (!tParam.Rotate(angle, GPUCA_MAX_SIN_PHI)) return(0);
	//printf("Rotated X %f Y %f Z %f SinPhi %f DzDs %f QPt %f SignCosPhi %f\n", tParam.X(), tParam.Y(), tParam.Z(), tParam.SinPhi(), tParam.DzDs(), tParam.QPt(), tParam.SignCosPhi());

	int maxRowGap = 10;
	AliGPUTPCTrackLinearisation t0( tParam );
	do
	{
		rowIndex += direction;
		if (!tParam.TransportToX(sliceNeighbour.Row(rowIndex).X(), t0, mParam->ConstBz, GPUCA_MAX_SIN_PHI)) return(0); //Reuse t0 linearization until we are in the next sector
		//printf("Transported X %f Y %f Z %f SinPhi %f DzDs %f QPt %f SignCosPhi %f (MaxY %f)\n", tParam.X(), tParam.Y(), tParam.Z(), tParam.SinPhi(), tParam.DzDs(), tParam.QPt(), tParam.SignCosPhi(), sliceNeighbour.Row(rowIndex).MaxY());
		if (--maxRowGap == 0) return(0);
	} while (fabs(tParam.Y()) > sliceNeighbour.Row(rowIndex).MaxY());

	float err2Y, err2Z;
	GetErrors2( rowIndex, tParam.Z(), tParam.SinPhi(), tParam.DzDs(), err2Y, err2Z );
	if (tParam.GetCov(0) < err2Y) tParam.SetCov(0, err2Y);
	if (tParam.GetCov(2) < err2Z) tParam.SetCov(2, err2Z);

	int nHits = AliGPUTPCTrackletConstructor::AliGPUTPCTrackletConstructorGlobalTracking(sliceNeighbour, tParam, rowIndex, direction, 0);
	if (nHits >= GLOBAL_TRACKING_MIN_HITS)
	{
		//printf("%d hits found\n", nHits);
		AliGPUTPCTrack& track = sliceNeighbour.fTracks[sliceNeighbour.fCommonMem->fNTracks];
		if (direction == 1)
		{
			int i = 0;
			while (i < nHits)
			{
#ifdef EXTERN_ROW_HITS
				const calink rowHit = sliceNeighbour.TrackletRowHits()[rowIndex * *sliceNeighbour.NTracklets()];
#else
				const calink rowHit = sliceNeighbour.Tracklet(0).RowHit(rowIndex);
#endif
				if (rowHit != CALINK_INVAL)
				{
					//printf("New track: entry %d, row %d, hitindex %d\n", i, rowIndex, sliceNeighbour.fTrackletRowHits[rowIndex * sliceNeighbour.fCommonMem->fNTracklets]);
					sliceNeighbour.fTrackHits[sliceNeighbour.fCommonMem->fNTrackHits + i].Set(rowIndex, rowHit);
					//if (i == 0) tParam.TransportToX(sliceNeighbour.Row(rowIndex).X(), mParam->ConstBz(), GPUCA_MAX_SIN_PHI); //Use transport with new linearisation, we have changed the track in between - NOT needed, fitting will always start at outer end of global track!
					i++;
				}
				rowIndex ++;
			}
		}
		else
		{
			int i = nHits - 1;
			while (i >= 0)
			{
#ifdef EXTERN_ROW_HITS
				const calink rowHit = sliceNeighbour.TrackletRowHits()[rowIndex * *sliceNeighbour.NTracklets()];
#else
				const calink rowHit = sliceNeighbour.Tracklet(0).RowHit(rowIndex);
#endif
				if (rowHit != CALINK_INVAL)
				{
					//printf("New track: entry %d, row %d, hitindex %d\n", i, rowIndex, sliceNeighbour.fTrackletRowHits[rowIndex * sliceNeighbour.fCommonMem->fNTracklets]);
					sliceNeighbour.fTrackHits[sliceNeighbour.fCommonMem->fNTrackHits + i].Set(rowIndex, rowHit);
					i--;
				}
				rowIndex--;
			}
		}
		track.SetAlive(1);
		track.SetParam(tParam.GetParam());
		track.SetNHits(nHits);
		track.SetFirstHitID(sliceNeighbour.fCommonMem->fNTrackHits);
		track.SetLocalTrackId((fISlice << 24) | fTracks[iTrack].LocalTrackId());
		sliceNeighbour.fCommonMem->fNTracks++;
		sliceNeighbour.fCommonMem->fNTrackHits += nHits;
	}

	return(nHits >= GLOBAL_TRACKING_MIN_HITS);
}

GPUh() void AliGPUTPCTracker::PerformGlobalTracking(AliGPUTPCTracker& sliceLeft, AliGPUTPCTracker& sliceRight, int MaxTracksLeft, int MaxTracksRight)
{
	StartTimer(8);
	int ul = 0, ur = 0, ll = 0, lr = 0;
	
	int nTrkLeft = sliceLeft.fCommonMem->fNTracklets, nTrkRight = sliceRight.fCommonMem->fNTracklets;
	sliceLeft.fCommonMem->fNTracklets = sliceRight.fCommonMem->fNTracklets = 1;
	AliGPUTPCTracklet *trkLeft = sliceLeft.fTracklets, *trkRight = sliceRight.fTracklets;
	sliceLeft.fTracklets = sliceRight.fTracklets = new AliGPUTPCTracklet;
#ifdef EXTERN_ROW_HITS
	calink *lnkLeft = sliceLeft.fTrackletRowHits, *lnkRight = sliceRight.fTrackletRowHits;
	sliceLeft.fTrackletRowHits = sliceRight.fTrackletRowHits = new calink[GPUCA_ROW_COUNT];
#endif

	for (int i = 0;i < fCommonMem->fNLocalTracks;i++)
	{
		{
			const int tmpHit = fTracks[i].FirstHitID();
			if (fTrackHits[tmpHit].RowIndex() >= GLOBAL_TRACKING_MIN_ROWS && fTrackHits[tmpHit].RowIndex() < GLOBAL_TRACKING_RANGE)
			{
				int rowIndex = fTrackHits[tmpHit].RowIndex();
				const AliGPUTPCRow& row = Row(rowIndex);
				float Y = (float) Data().HitDataY(row, fTrackHits[tmpHit].HitIndex()) * row.HstepY() + row.Grid().YMin();
				if (sliceLeft.NHitsTotal() < 1) {}
				else if (sliceLeft.fCommonMem->fNTracks >= MaxTracksLeft) {printf("Insufficient memory for global tracking (%d:l %d / %d)\n", fISlice, sliceLeft.fCommonMem->fNTracks, MaxTracksLeft);}
				else if (Y < -row.MaxY() * GLOBAL_TRACKING_Y_RANGE_LOWER_LEFT)
				{
					//printf("Track %d, lower row %d, left border (%f of %f)\n", i, fTrackHits[tmpHit].RowIndex(), Y, -row.MaxY());
					ll += PerformGlobalTrackingRun(sliceLeft, i, rowIndex, -mParam->DAlpha, -1);
				}
				if (sliceRight.NHitsTotal() < 1) {}
				else if (sliceRight.fCommonMem->fNTracks >= MaxTracksRight) {printf("Insufficient memory for global tracking (%d:r %d / %d)\n", fISlice, sliceRight.fCommonMem->fNTracks, MaxTracksRight);}
				else if (Y > row.MaxY() * GLOBAL_TRACKING_Y_RANGE_LOWER_RIGHT)
				{
					//printf("Track %d, lower row %d, right border (%f of %f)\n", i, fTrackHits[tmpHit].RowIndex(), Y, row.MaxY());
					lr += PerformGlobalTrackingRun(sliceRight, i, rowIndex, mParam->DAlpha, -1);
				}
			}
		}
		
		{
			const int tmpHit = fTracks[i].FirstHitID() + fTracks[i].NHits() - 1;
			if (fTrackHits[tmpHit].RowIndex() < GPUCA_ROW_COUNT - GLOBAL_TRACKING_MIN_ROWS && fTrackHits[tmpHit].RowIndex() >= GPUCA_ROW_COUNT - GLOBAL_TRACKING_RANGE)
			{
				int rowIndex = fTrackHits[tmpHit].RowIndex();
				const AliGPUTPCRow& row = Row(rowIndex);
				float Y = (float) Data().HitDataY(row, fTrackHits[tmpHit].HitIndex()) * row.HstepY() + row.Grid().YMin();
				if (sliceLeft.NHitsTotal() < 1) {}
				else if (sliceLeft.fCommonMem->fNTracks >= MaxTracksLeft) {printf("Insufficient memory for global tracking (%d:l %d / %d)\n", fISlice, sliceLeft.fCommonMem->fNTracks, MaxTracksLeft);}
				else if (Y < -row.MaxY() * GLOBAL_TRACKING_Y_RANGE_UPPER_LEFT)
				{
					//printf("Track %d, upper row %d, left border (%f of %f)\n", i, fTrackHits[tmpHit].RowIndex(), Y, -row.MaxY());
					ul += PerformGlobalTrackingRun(sliceLeft, i, rowIndex, -mParam->DAlpha, 1);
				}
				if (sliceRight.NHitsTotal() < 1) {}
				else if (sliceRight.fCommonMem->fNTracks >= MaxTracksRight) {printf("Insufficient memory for global tracking (%d:r %d / %d)\n", fISlice, sliceRight.fCommonMem->fNTracks, MaxTracksRight);}
				else if (Y > row.MaxY() * GLOBAL_TRACKING_Y_RANGE_UPPER_RIGHT)
				{
					//printf("Track %d, upper row %d, right border (%f of %f)\n", i, fTrackHits[tmpHit].RowIndex(), Y, row.MaxY());
					ur += PerformGlobalTrackingRun(sliceRight, i, rowIndex, mParam->DAlpha, 1);
				}
			}
		}
	}
	
	sliceLeft.fCommonMem->fNTracklets = nTrkLeft;sliceRight.fCommonMem->fNTracklets = nTrkRight;
	delete sliceLeft.fTracklets;
	sliceLeft.fTracklets = trkLeft;sliceRight.fTracklets = trkRight;
#ifdef EXTERN_ROW_HITS
	delete[] sliceLeft.fTrackletRowHits;
	sliceLeft.fTrackletRowHits = lnkLeft;sliceRight.fTrackletRowHits = lnkRight;
#endif
	StopTimer(8);
	//printf("Global Tracking Result: Slide %2d: LL %3d LR %3d UL %3d UR %3d\n", mParam->ISlice(), ll, lr, ul, ur);
}

#endif
