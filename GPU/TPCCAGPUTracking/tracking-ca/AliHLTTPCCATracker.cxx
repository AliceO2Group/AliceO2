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

#include "AliHLTTPCCATracker.h"
#include "AliHLTTPCCARow.h"
#include "AliHLTTPCCATrack.h"
#include "AliHLTTPCCATracklet.h"
#include "AliHLTTPCCAMath.h"
#include "MemoryAssignmentHelpers.h"

#include "AliHLTTPCCAHitArea.h"
#include "AliHLTTPCCANeighboursFinder.h"
#include "AliHLTTPCCANeighboursCleaner.h"
#include "AliHLTTPCCAStartHitsFinder.h"
#include "AliHLTTPCCATrackletConstructor.h"
#include "AliHLTTPCCATrackLinearisation.h"
#include "AliHLTTPCCATrackletSelector.h"
#include "AliHLTTPCCAProcess.h"
#include "AliHLTTPCCAClusterData.h"

#include "AliHLTTPCCATrackParam.h"

#include "AliHLTTPCCAGPUConfig.h"

#if !defined(HLTCA_GPUCODE)
#include <iostream>
#include <iomanip>
#include <string.h>
#include <cmath>
#include <algorithm>
#endif

//#define DRAW1

#ifdef DRAW1
#include "AliHLTTPCCADisplay.h"
#endif //DRAW1

#ifdef HLTCA_INTERNAL_PERFORMANCE
//#include "AliHLTTPCCAPerformance.h"
#endif

ClassImp( AliHLTTPCCATracker )

#if !defined(__OPENCL__) || defined(HLTCA_HOSTCODE)

#if !defined(HLTCA_GPUCODE)

AliHLTTPCCATracker::~AliHLTTPCCATracker()
{
	// destructor
	if (!fIsGPUTracker)
	{
		if (fCommonMem) delete fCommonMem;
		if (fHitMemory) delete[] fHitMemory;
		if (fTrackletMemory) delete[] fTrackletMemory;
		if (fTrackMemory) delete[] fTrackMemory;
		fCommonMem = NULL;
		fHitMemory = fTrackMemory = NULL;
	}
#ifdef HLTCA_STANDALONE
	if (fLinkTmpMemory) delete[] fLinkTmpMemory;
#endif
}

// ----------------------------------------------------------------------------------
void AliHLTTPCCATracker::Initialize( const AliHLTTPCCAParam &param )
{
	// initialisation
	fParam = param;
	fParam.Update();
	fData.InitializeRows( fParam );

	StartEvent();
}

void AliHLTTPCCATracker::StartEvent()
{
	// start new event and fresh the memory

	SetupCommonMemory();
}

void AliHLTTPCCATracker::SetGPUTracker()
{
	//Make this a GPU Tracker
	fIsGPUTracker = true;
	fData.SetGpuSliceData();
}

char* AliHLTTPCCATracker::SetGPUTrackerCommonMemory(char* const pGPUMemory)
{
	//Set up common Memory Pointer for GPU Tracker
	fCommonMem = (commonMemoryStruct*) pGPUMemory;
	return(pGPUMemory + sizeof(commonMemoryStruct));
}


char* AliHLTTPCCATracker::SetGPUTrackerHitsMemory(char* pGPUMemory, int MaxNHits)
{
	//Set up Hits Memory Pointers for GPU Tracker
	fHitMemory = (char*) pGPUMemory;
	SetPointersHits(MaxNHits);
	pGPUMemory += fHitMemorySize;
	AssignMemory(fTrackletTmpStartHits, pGPUMemory, NHitsTotal());
	AssignMemory(fRowStartHitCountOffset, pGPUMemory, Param().NRows());

	return(pGPUMemory);
}

char* AliHLTTPCCATracker::SetGPUTrackerTrackletsMemory(char* pGPUMemory, int MaxNTracks)
{
	//Set up Tracklet Memory Pointers for GPU Tracker
	fTrackletMemory = (char*) pGPUMemory;
	SetPointersTracklets(MaxNTracks);
	pGPUMemory += fTrackletMemorySize;
	return(pGPUMemory);
}

char* AliHLTTPCCATracker::SetGPUTrackerTracksMemory(char* pGPUMemory, int MaxNTracks, int MaxNHits )
{
	//Set up Tracks Memory Pointer for GPU Tracker
	fTrackMemory = (char*) pGPUMemory;
	SetPointersTracks(MaxNTracks, MaxNHits);
	pGPUMemory += fTrackMemorySize;

	return(pGPUMemory);
}

void AliHLTTPCCATracker::DumpOutput(FILE* out)
{
	fprintf(out, "Slice %d\n", fParam.ISlice());
	const AliHLTTPCCASliceOutTrack* track = (*(Output()))->GetFirstTrack();
	for (int j = 0;j < (*(Output()))->NTracks();j++)
	{
		fprintf(out, "Track %d (%d): ", j, track->NClusters());
		for (int k = 0;k < track->NClusters();k++)
		{
			fprintf(out, "(%2.3f,%2.3f,%2.4f) ", track->Cluster(k).GetX(), track->Cluster(k).GetY(), track->Cluster(k).GetZ());
		}
		fprintf(out, " - (%8.5f %8.5f %8.5f %8.5f %8.5f)", track->Param().Y(), track->Param().Z(), track->Param().SinPhi(), track->Param().DzDs(), track->Param().QPt());
		fprintf(out, "\n");
		track = track->GetNextTrack();
	}
}

void AliHLTTPCCATracker::DumpSliceData(std::ostream &out)
{
	//Dump Slice Input Data to File
	out << "Slice Data (Slice" << Param().ISlice() << "):" << std::endl;
	for (int i = 0;i < Param().NRows();i++)
	{
		if (Row(i).NHits() == 0) continue;
		out << "Row: " << i << std::endl;
		for (int j = 0;j < Row(i).NHits();j++)
		{
			if (j && j % 16 == 0) out << std::endl;
			out << j << '-' << Data().HitDataY(Row(i), j) << '-' << Data().HitDataZ(Row(i), j) << ", ";
		}
		out << std::endl;
	}
}

void AliHLTTPCCATracker::DumpLinks(std::ostream &out)
{
	//Dump Links (after Neighbours Finder / Cleaner) to file
	out << "Hit Links(Slice" << Param().ISlice() << "):" << std::endl;
	for (int i = 0;i < Param().NRows();i++)
	{
		if (Row(i).NHits() == 0) continue;
		out << "Row: " << i << std::endl;
		for (int j = 0;j < Row(i).NHits();j++)
		{
			if (j && j % 32 == 0) out << std::endl;
			out << HitLinkUpData(Row(i), j) << "/" << HitLinkDownData(Row(i), j) << ", ";
		}
		out << std::endl;
	}
}

void AliHLTTPCCATracker::DumpHitWeights(std::ostream &out)
{
	//dump hit weights to file
	out << "Hit Weights(Slice" << Param().ISlice() << "):" << std::endl;
	for (int i = 0;i < Param().NRows();i++)
	{
		if (Row(i).NHits() == 0) continue;
		out << "Row: " << i << ":" << std::endl;
		for (int j = 0;j < Row(i).NHits();j++)
		{
			if (j && j % 32 == 0) out << std::endl;
			out << HitWeight(Row(i), j) << ", ";
		}
		out << std::endl;
	}
}

int AliHLTTPCCATracker::StarthitSortComparison(const void*a, const void* b)
{
	//qsort helper function to sort start hits
	AliHLTTPCCAHitId* aa = (AliHLTTPCCAHitId*) a;
	AliHLTTPCCAHitId* bb = (AliHLTTPCCAHitId*) b;

	if (aa->RowIndex() != bb->RowIndex()) return(aa->RowIndex() - bb->RowIndex());
	return(aa->HitIndex() - bb->HitIndex());
}

void AliHLTTPCCATracker::DumpStartHits(std::ostream &out)
{
	//sort start hits and dump to file
	out << "Start Hits: (Slice" << Param().ISlice() << ") (" << *NTracklets() << ")" << std::endl;
#ifdef HLTCA_GPU_SORT_DUMPDATA
	qsort(TrackletStartHits(), *NTracklets(), sizeof(AliHLTTPCCAHitId), StarthitSortComparison);
#endif
	for (int i = 0;i < *NTracklets();i++)
	{
		out << TrackletStartHit(i).RowIndex() << "-" << TrackletStartHit(i).HitIndex() << std::endl;
	}
	out << std::endl;
}

void AliHLTTPCCATracker::DumpTrackHits(std::ostream &out)
{
	//dump tracks to file
	out << "Tracks: (Slice" << Param().ISlice() << ") (" << *NTracks() << ")" << std::endl;
#ifdef HLTCA_GPU_SORT_DUMPDATA
	for (int k = 0;k < Param().NRows();k++)
	{
		for (int l = 0;l < Row(k).NHits();l++)
		{
#endif
			for (int j = 0;j < *NTracks();j++)
			{
				if (Tracks()[j].NHits() == 0 || !Tracks()[j].Alive()) continue;
#ifdef HLTCA_GPU_SORT_DUMPDATA
				if (TrackHits()[Tracks()[j].FirstHitID()].RowIndex() == k && TrackHits()[Tracks()[j].FirstHitID()].HitIndex() == l)
				{
#endif
					for (int i = 0;i < Tracks()[j].NHits();i++)
					{
						out << TrackHits()[Tracks()[j].FirstHitID() + i].RowIndex() << "-" << TrackHits()[Tracks()[j].FirstHitID() + i].HitIndex() << ", ";
					}
#ifndef BITWISE_COMPATIBLE_DEBUG_OUTPUT
					out << "(Track: " << j << ")";
#endif
					out << std::endl;
#ifdef HLTCA_GPU_SORT_DUMPDATA
				}
			}	
#endif
		}	
#ifdef HLTCA_GPU_SORT_DUMPDATA
	}
#endif
}

void AliHLTTPCCATracker::DumpTrackletHits(std::ostream &out)
{
	//dump tracklets to file
	int nTracklets = *NTracklets();
	if( nTracklets<0 ) nTracklets = 0;
	if( nTracklets>HLTCA_GPU_MAX_TRACKLETS ) nTracklets = HLTCA_GPU_MAX_TRACKLETS;
	out << "Tracklets: (Slice" << Param().ISlice() << ") (" << nTracklets << ")" << std::endl;
#ifdef HLTCA_GPU_SORT_DUMPDATA
	AliHLTTPCCAHitId* tmpIds = new AliHLTTPCCAHitId[nTracklets];
	AliHLTTPCCATracklet* tmpTracklets = new AliHLTTPCCATracklet[nTracklets];
	memcpy(tmpIds, TrackletStartHits(), nTracklets * sizeof(AliHLTTPCCAHitId));
	memcpy(tmpTracklets, Tracklets(), nTracklets * sizeof(AliHLTTPCCATracklet));
#ifdef EXTERN_ROW_HITS
	calink* tmpHits = new calink[nTracklets * Param().NRows()];
	memcpy(tmpHits, TrackletRowHits(), nTracklets * Param().NRows() * sizeof(calink));
#endif
	qsort(TrackletStartHits(), nTracklets, sizeof(AliHLTTPCCAHitId), StarthitSortComparison);
	for (int i = 0;i < nTracklets; i++ ){
		for (int j = 0;j < nTracklets; j++ ){
			if (tmpIds[i].RowIndex() == TrackletStartHit(j).RowIndex() && tmpIds[i].HitIndex() == TrackletStartHit(j).HitIndex() ){
				memcpy(&Tracklets()[j], &tmpTracklets[i], sizeof(AliHLTTPCCATracklet));
#ifdef EXTERN_ROW_HITS
				if (tmpTracklets[i].NHits() ){
					for (int k = tmpTracklets[i].FirstRow();k <= tmpTracklets[i].LastRow();k++){
						const int pos = k * nTracklets + j;
						if (pos < 0 || pos >= HLTCA_GPU_MAX_TRACKLETS * fParam.NRows()){
							printf("internal error\n");	      
						} else {
							fTrackletRowHits[pos] = tmpHits[k * nTracklets + i];
						}
					}
				}
#endif
				break;
			}
		}
	}
	delete[] tmpIds;
	delete[] tmpTracklets;
#ifdef EXTERN_ROW_HITS
	delete[] tmpHits;
#endif
#endif
	for (int j = 0;j < nTracklets; j++ )
	{
		out << "Tracklet " << std::setw(4) << j << " (Hits: " << std::setw(3) << Tracklets()[j].NHits() << ", Start: " << std::setw(3) << TrackletStartHit(j).RowIndex() << "-" << std::setw(3) << TrackletStartHit(j).HitIndex() << ", Rows: " << (Tracklets()[j].NHits() ? Tracklets()[j].FirstRow() : -1) << " - " << (Tracklets()[j].NHits() ? Tracklets()[j].LastRow() : -1) << ") ";
		if (Tracklets()[j].NHits() == 0);
		else if (Tracklets()[j].LastRow() > Tracklets()[j].FirstRow() && (Tracklets()[j].FirstRow() >= Param().NRows() || Tracklets()[j].LastRow() >= Param().NRows()))
		{
#ifdef HLTCA_STANDALONE
			printf("\nError: Tracklet %d First %d Last %d Hits %d", j, Tracklets()[j].FirstRow(), Tracklets()[j].LastRow(), Tracklets()[j].NHits());
			out << " (Error: Tracklet " << j << " First " << Tracklets()[j].FirstRow() << " Last " << Tracklets()[j].LastRow() << " Hits " << Tracklets()[j].NHits() << ") ";
			for (int i = 0;i < Param().NRows();i++)
			{
				//if (Tracklets()[j].RowHit(i) != CALINK_INVAL)
#ifdef EXTERN_ROW_HITS
				out << i << "-" << fTrackletRowHits[i * fCommonMem->fNTracklets + j] << ", ";
#else
				out << i << "-" << Tracklets()[j].RowHit(i) << ", ";
#endif
			}
#endif
		}
		else if (Tracklets()[j].NHits() && Tracklets()[j].LastRow() >= Tracklets()[j].FirstRow())
		{
			int nHits = 0;;
			for (int i = Tracklets()[j].FirstRow();i <= Tracklets()[j].LastRow();i++)
			{
#ifdef EXTERN_ROW_HITS
				calink ih = fTrackletRowHits[i * fCommonMem->fNTracklets + j];
#else
				calink ih = Tracklets()[j].RowHit(i);
#endif
				if (ih != CALINK_INVAL)
				{
					nHits++;
				}
#ifdef EXTERN_ROW_HITS
				out << i << "-" << fTrackletRowHits[i * fCommonMem->fNTracklets + j] << ", ";
#else
				out << i << "-" << Tracklets()[j].RowHit(i) << ", ";
#endif
			}
			if (nHits != Tracklets()[j].NHits())
			{
				std::cout << std::endl << "Wrong NHits!: Expected " << Tracklets()[j].NHits() << ", found " << nHits;
				out << std::endl << "Wrong NHits!: Expected " << Tracklets()[j].NHits() << ", found " << nHits;
			}
		}
		out << std::endl;
	}
}


void AliHLTTPCCATracker::SetupCommonMemory()
{
	// set up common memory

	if (!fIsGPUTracker)
	{
		if ( !fCommonMem ) {
			// the 1600 extra bytes are not used unless fCommonMemorySize increases with a later event
			//fCommonMemory = reinterpret_cast<char*> ( new uint4 [ fCommonMemorySize/sizeof( uint4 ) + 100] );
			fCommonMem = new commonMemoryStruct;
		}

		if (fHitMemory)	delete[] fHitMemory;
		if (fTrackletMemory) delete[] fTrackletMemory;
		if (fTrackMemory) delete[] fTrackMemory;
	}

	fHitMemory = fTrackletMemory = fTrackMemory = 0;

	fData.Clear();
	fCommonMem->fNTracklets = 0;
	fCommonMem->fNTracks = 0 ;
	fCommonMem->fNTrackHits = 0;
}

int AliHLTTPCCATracker::ReadEvent( AliHLTTPCCAClusterData *clusterData )
{
	// read event

	StartTimer(0);
	fClusterData = clusterData;

	StartEvent();

	//* Convert input hits, create grids, etc.
	if (fData.InitFromClusterData( *clusterData )) return 1;
	if (fData.MaxZ() > 300 && !fParam.GetContinuousTracking())
	{
		printf("Need to set continuous tracking mode for data outside of the TPC volume!\n");
		return 1;
	}
	if (!fIsGPUTracker)
	{
		SetPointersHits( fData.NumberOfHits() ); // to calculate the size
		fHitMemory = reinterpret_cast<char*> ( new uint4 [ fHitMemorySize/sizeof( uint4 ) + 100] );
	}
	SetPointersHits( fData.NumberOfHits() ); // set pointers for hits
	StopTimer(0);
	return 0;
}

GPUhd() void  AliHLTTPCCATracker::SetPointersHits( int MaxNHits )
{
	// set all pointers to the event memory

	char *mem = fHitMemory;

	// extra arrays for tpc clusters

#ifdef HLTCA_GPU_SORT_STARTHITS_2
	AssignMemory( fTrackletStartHits, mem, MaxNHits + 32);
#else
	AssignMemory( fTrackletStartHits, mem, MaxNHits);
#endif

	// calculate the size

	fHitMemorySize = mem - fHitMemory;
}

GPUhd() void  AliHLTTPCCATracker::SetPointersTracklets( int MaxNTracklets )
{
	// set all pointers to the tracklets memory
	char *mem = fTrackletMemory;

	// memory for tracklets

	AssignMemory( fTracklets, mem, MaxNTracklets );
#ifdef EXTERN_ROW_HITS
	AssignMemory( fTrackletRowHits, mem, MaxNTracklets * Param().NRows());
#endif

	fTrackletMemorySize = mem - fTrackletMemory;
}


GPUhd() void  AliHLTTPCCATracker::SetPointersTracks( int MaxNTracks, int MaxNHits )
{
	// set all pointers to the tracks memory
	char *mem = fTrackMemory;

	// memory for selected tracks

	AssignMemory( fTracks, mem, MaxNTracks );
	AssignMemory( fTrackHits, mem, 2 * MaxNHits );

	// calculate the size

	fTrackMemorySize = mem - fTrackMemory;
}

GPUh() int AliHLTTPCCATracker::CheckEmptySlice() const
{
	//Check if the Slice is empty, if so set the output apropriate and tell the reconstuct procesdure to terminate
	if ( NHitsTotal() < 1 ) {
		{
			AliHLTTPCCASliceOutput::Allocate(*fOutput, 0, 0, fOutputControl);
			AliHLTTPCCASliceOutput* useOutput = *fOutput;
			if (useOutput == NULL) return(1);
			useOutput->SetNTracks( 0 );
			useOutput->SetNTrackClusters( 0 );
		}

		return 1;
	}
	return 0;
}

void AliHLTTPCCATracker::RunNeighboursFinder()
{
	//Run the CPU Neighbours Finder
	AliHLTTPCCAProcess<AliHLTTPCCANeighboursFinder>( Param().NRows(), 1, *this );
}

void AliHLTTPCCATracker::RunNeighboursCleaner()
{
	//Run the CPU Neighbours Cleaner
	AliHLTTPCCAProcess<AliHLTTPCCANeighboursCleaner>( Param().NRows() - 2, 1, *this );
}

void AliHLTTPCCATracker::RunStartHitsFinder()
{
	//Run the CPU Start Hits Finder
	AliHLTTPCCAProcess<AliHLTTPCCAStartHitsFinder>( Param().NRows() - 4, 1, *this );
}

void AliHLTTPCCATracker::RunTrackletConstructor()
{
	//Run CPU Tracklet Constructor
	AliHLTTPCCATrackletConstructor::AliHLTTPCCATrackletConstructorCPU(*this);
}

void AliHLTTPCCATracker::RunTrackletSelector()
{
	//Run CPU Tracklet Selector
	AliHLTTPCCAProcess<AliHLTTPCCATrackletSelector>( 1, fCommonMem->fNTracklets, *this );
}

GPUh() void AliHLTTPCCATracker::DoTracking()
{
	fCommonMem->fNTracklets = fCommonMem->fNTracks = fCommonMem->fNTrackHits = 0;

	if (fGPUDebugLevel >= 6)
	{
#ifndef BITWISE_COMPATIBLE_DEBUG_OUTPUT
		*fGPUDebugOut << std::endl << std::endl << "Slice: " << Param().ISlice() << std::endl;
		*fGPUDebugOut << "Slice Data:" << std::endl;
#endif
		DumpSliceData(*fGPUDebugOut);
	}

	StartTimer(1);
	RunNeighboursFinder();
	StopTimer(1);

#ifdef TRACKER_KEEP_TEMPDATA
	if (fLinkTmpMemory) delete[] fLinkTmpMemory;
	fLinkTmpMemory = new char[fData.MemorySize()];
	memcpy(fLinkTmpMemory, fData.Memory(), fData.MemorySize());
#endif

	if (fGPUDebugLevel >= 6) DumpLinks(*fGPUDebugOut);

#ifdef HLTCA_INTERNAL_PERFORMANCE
	//if( Param().ISlice()<=2 )
	//AliHLTTPCCAPerformance::Instance().LinkPerformance( Param().ISlice() );
#endif


#ifdef DRAW1
	if ( NHitsTotal() > 0 ) {
		AliHLTTPCCADisplay::Instance().DrawSliceLinks( -1, -1, 1 );
		AliHLTTPCCADisplay::Instance().Ask();
	}
#endif //DRAW1

	StartTimer(2);
	RunNeighboursCleaner();
	StopTimer(2);

	if (fGPUDebugLevel >= 6) DumpLinks(*fGPUDebugOut);

	StartTimer(3);
	RunStartHitsFinder();
	StopTimer(3);

	if (fGPUDebugLevel >= 6) DumpStartHits(*fGPUDebugOut);

	StartTimer(5);
	fData.ClearHitWeights();
	StopTimer(5);

	if (!fIsGPUTracker)
	{
		SetPointersTracklets( fCommonMem->fNTracklets * 2 ); // to calculate the size
		fTrackletMemory = reinterpret_cast<char*> ( new uint4 [ fTrackletMemorySize/sizeof( uint4 ) + 100] );
		SetPointersTracks( fCommonMem->fNTracklets * 2 + 50, NHitsTotal() ); // to calculate the size
		fTrackMemory = reinterpret_cast<char*> ( new uint4 [ fTrackMemorySize/sizeof( uint4 ) + 100] );
	}

	SetPointersTracklets( fCommonMem->fNTracklets * 2 ); // set pointers for hits
	SetPointersTracks( fCommonMem->fNTracklets * 2 + 50, NHitsTotal() ); // set pointers for hits

	StartTimer(6);
	RunTrackletConstructor();
	StopTimer(6);

	if (fGPUDebugLevel >= 6) DumpTrackletHits(*fGPUDebugOut);
#ifndef BITWISE_COMPATIBLE_DEBUG_OUTPUT
	if (fGPUDebugLevel >= 6) DumpHitWeights(*fGPUDebugOut);
#endif

	//std::cout<<"Slice "<<Param().ISlice()<<": NHits="<<NHitsTotal()<<", NTracklets="<<*NTracklets()<<std::endl;

	StartTimer(7);
	RunTrackletSelector();
	StopTimer(7);

	//std::cout<<"Slice "<<Param().ISlice()<<": N start hits/tracklets/tracks = "<<nStartHits<<" "<<nStartHits<<" "<<*fNTracks<<std::endl;

	if (fGPUDebugLevel >= 6) DumpTrackHits(*fGPUDebugOut);

	//std::cout<<"Memory used for slice "<<fParam.ISlice()<<" : "<<fCommonMemorySize/1024./1024.<<" + "<<fHitMemorySize/1024./1024.<<" + "<<fTrackMemorySize/1024./1024.<<" = "<<( fCommonMemorySize+fHitMemorySize+fTrackMemorySize )/1024./1024.<<" Mb "<<std::endl;
}

GPUh() void AliHLTTPCCATracker::Reconstruct()
{

	if (CheckEmptySlice()) return;
	DoTracking();
}

GPUh() void AliHLTTPCCATracker::ReconstructOutput()
{
	WriteOutputPrepare();
	WriteOutput();
}

GPUh() void AliHLTTPCCATracker::WriteOutputPrepare()
{
	StartTimer(9);
	if (fOutputControl == NULL)
	{
		fOutputControl = new AliHLTTPCCASliceOutput::outputControlStruct;
		memset(fOutputControl, 0, sizeof(*fOutputControl));
	}
	AliHLTTPCCASliceOutput::Allocate(*fOutput, fCommonMem->fNTracks, fCommonMem->fNTrackHits, fOutputControl);
	StopTimer(9);
}

template <class T> static inline bool SortComparison(const T& a, const T& b)
{
	return(a.fSortVal < b.fSortVal);
}

GPUh() void AliHLTTPCCATracker::WriteOutput()
{
	// write output
	AliHLTTPCCASliceOutput* useOutput = *fOutput;

	//cout<<"output: nTracks = "<<*fNTracks<<", nHitsTotal="<<NHitsTotal()<<std::endl;

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

	AliHLTTPCCASliceOutTrack *out = useOutput->FirstTrack();
	
	trackSortData* trackOrder = new trackSortData[fCommonMem->fNTracks];
	for (int i = 0;i < fCommonMem->fNTracks;i++)
	{
		trackOrder[i].fTtrack = i;
		trackOrder[i].fSortVal = fTracks[trackOrder[i].fTtrack].NHits() / 1000.f + fTracks[trackOrder[i].fTtrack].Param().GetZ() * 100.f + fTracks[trackOrder[i].fTtrack].Param().GetY();
	}
	//std::sort(trackOrder, trackOrder + fCommonMem->fNLocalTracks, SortComparison<trackSortData>);
	//std::sort(trackOrder + fCommonMem->fNLocalTracks, trackOrder + fCommonMem->fNTracks, SortComparison<trackSortData>);
	
	for (int iTrTmp = 0;iTrTmp < fCommonMem->fNTracks;iTrTmp++)
	{
		const int iTr = trackOrder[iTrTmp].fTtrack;
		AliHLTTPCCATrack &iTrack = fTracks[iTr];

		out->SetParam( iTrack.Param() );
		out->SetLocalTrackId( iTrack.LocalTrackId() );
		int nClu = 0;
		int iID = iTrack.FirstHitID();

		for (int ith = 0;ith < iTrack.NHits();ith++)
		{
			const AliHLTTPCCAHitId &ic = fTrackHits[iID + ith];
			int iRow = ic.RowIndex();
			int ih = ic.HitIndex();

			const AliHLTTPCCARow &row = fData.Row( iRow );
#ifdef HLTCA_ARRAY_BOUNDS_CHECKS
			if (ih >= row.NHits() || ih < 0)
			{
				printf("Array out of bounds access (Sector Row) (Hit %d / %d - NumC %d): Sector %d Row %d Index %d\n", ith, iTrack.NHits(), fClusterData->NumberOfClusters(), fParam.ISlice(), iRow, ih);
				fflush(stdout);
				continue;
			}
#endif
			int clusterIndex = fData.ClusterDataIndex( row, ih );

#ifdef HLTCA_ARRAY_BOUNDS_CHECKS
			if (clusterIndex >= fClusterData->NumberOfClusters() || clusterIndex < 0)
			{
				printf("Array out of bounds access (Cluster Data) (Hit %d / %d - NumC %d): Sector %d Row %d Hit %d, Clusterdata Index %d\n", ith, iTrack.NHits(), fClusterData->NumberOfClusters(), fParam.ISlice(), iRow, ih, clusterIndex);
				fflush(stdout);
				continue;
			}
#endif

			float origX = fClusterData->X( clusterIndex );
			float origY = fClusterData->Y( clusterIndex );
			float origZ = fClusterData->Z( clusterIndex );      
			int id = fClusterData->Id( clusterIndex );
			AliHLTTPCCASliceOutCluster c;
			c.Set( id, iRow, origX, origY, origZ );
			out->SetCluster( nClu, c );
			nClu++;
		}

		nStoredTracks++;
		if (iTr < fCommonMem->fNLocalTracks) nStoredLocalTracks++;
		nStoredHits+=nClu; 
		out->SetNClusters( nClu );
		out = out->NextTrack();
	}

	useOutput->SetNTracks( nStoredTracks );
	useOutput->SetNLocalTracks( nStoredLocalTracks );
	useOutput->SetNTrackClusters( nStoredHits );

	StopTimer(9);
}

#endif

#if !defined(HLTCA_GPUCODE)

GPUh() void AliHLTTPCCATracker::WriteEvent( std::ostream &out )
{
	// write event to the file
	for ( int iRow = 0; iRow < fParam.NRows(); iRow++ ) {
		out << fData.Row( iRow ).HitNumberOffset() << " " << fData.Row( iRow ).NHits() << std::endl;
	}
	out << NHitsTotal() << std::endl;

	AliHLTResizableArray<float> y( NHitsTotal() ), z( NHitsTotal() );

	for ( int iRow = 0; iRow < fParam.NRows(); iRow++ ) {
		const AliHLTTPCCARow &row = Row( iRow );
		float y0 = row.Grid().YMin();
		float z0 = row.Grid().ZMin();
		float stepY = row.HstepY();
		float stepZ = row.HstepZ();
		for ( int ih = 0; ih < fData.Row( iRow ).NHits(); ih++ ) {
			int id = HitInputID( row, ih );
			y[id] = y0 + HitDataY( row, ih ) * stepY;
			z[id] = z0 + HitDataZ( row, ih ) * stepZ;
		}
	}
	for ( int ih = 0; ih < NHitsTotal(); ih++ ) {
		out << y[ih] << " " << z[ih] << std::endl;
	}
}

GPUh() void AliHLTTPCCATracker::WriteTracks( std::ostream &/*out*/ )
{
	//* Write tracks to file --- dummy
}

GPUh() void AliHLTTPCCATracker::ReadTracks( std::istream &/*in*/ )
{
	//* Read tracks  from file -- dummy
}

GPUh() int AliHLTTPCCATracker::PerformGlobalTrackingRun(AliHLTTPCCATracker& sliceNeighbour, int iTrack, int rowIndex, float angle, int direction)
{
	/*for (int j = 0;j < fTracks[j].NHits();j++)
	{
		printf("Hit %3d: Row %3d: X %3.7lf Y %3.7lf\n", j, fTrackHits[fTracks[iTrack].FirstHitID() + j].RowIndex(), Row(fTrackHits[fTracks[iTrack].FirstHitID() + j].RowIndex()).X(),
		(float) Data().HitDataY(Row(fTrackHits[fTracks[iTrack].FirstHitID() + j].RowIndex()), fTrackHits[fTracks[iTrack].FirstHitID() + j].HitIndex()) * Row(fTrackHits[fTracks[iTrack].FirstHitID() + j].RowIndex()).HstepY() + Row(fTrackHits[fTracks[iTrack].FirstHitID() + j].RowIndex()).Grid().YMin());
		}*/

	if (sliceNeighbour.fCommonMem->fNTracklets == 0) return(0);

	AliHLTTPCCATrackParam tParam;
	tParam.InitParam();
	tParam.SetCov( 0, 0.05 );
	tParam.SetCov( 2, 0.05 );
	tParam.SetCov( 5, 0.001 );
	tParam.SetCov( 9, 0.001 );
	tParam.SetCov( 14, 0.05 );
	tParam.SetParam(fTracks[iTrack].Param());

	//printf("Parameters X %f Y %f Z %f SinPhi %f DzDs %f QPt %f SignCosPhi %f\n", tParam.X(), tParam.Y(), tParam.Z(), tParam.SinPhi(), tParam.DzDs(), tParam.QPt(), tParam.SignCosPhi());
	if (!tParam.Rotate(angle, HLTCA_MAX_SIN_PHI)) return(0);
	//printf("Rotated X %f Y %f Z %f SinPhi %f DzDs %f QPt %f SignCosPhi %f\n", tParam.X(), tParam.Y(), tParam.Z(), tParam.SinPhi(), tParam.DzDs(), tParam.QPt(), tParam.SignCosPhi());

	int maxRowGap = 10;
	AliHLTTPCCATrackLinearisation t0( tParam );
	do
	{
		rowIndex += direction;
		if (!tParam.TransportToX(sliceNeighbour.Row(rowIndex).X(), t0, fParam.ConstBz(), HLTCA_MAX_SIN_PHI)) return(0); //Reuse t0 linearization until we are in the next sector
		//printf("Transported X %f Y %f Z %f SinPhi %f DzDs %f QPt %f SignCosPhi %f (MaxY %f)\n", tParam.X(), tParam.Y(), tParam.Z(), tParam.SinPhi(), tParam.DzDs(), tParam.QPt(), tParam.SignCosPhi(), sliceNeighbour.Row(rowIndex).MaxY());
		if (--maxRowGap == 0) return(0);
	} while (fabs(tParam.Y()) > sliceNeighbour.Row(rowIndex).MaxY());

	float err2Y, err2Z;
	GetErrors2( rowIndex, *((MEM_LG2(AliHLTTPCCATrackParam)*) &tParam ), err2Y, err2Z );
	if (tParam.GetCov(0) < err2Y) tParam.SetCov(0, err2Y);
	if (tParam.GetCov(2) < err2Z) tParam.SetCov(2, err2Z);

#ifdef GLOBAL_TRACKING_MAINTAIN_TRACKLETS
	int tmphits[HLTCA_ROW_COUNT];
	for (int i = 0;i < HLTCA_ROW_COUNT;i++)
	{
#ifdef EXTERN_ROW_HITS
		tmphits[i] = sliceNeighbour.TrackletRowHits()[i * *sliceNeighbour.NTracklets()];
#else
		tmphits[i] = sliceNeighbour.Tracklet(0).RowHit(i);
#endif
	}
#endif

	int nHits = AliHLTTPCCATrackletConstructor::AliHLTTPCCATrackletConstructorGlobalTracking(sliceNeighbour, tParam, rowIndex, direction);
	if (nHits >= GLOBAL_TRACKING_MIN_HITS)
	{
		//printf("%d hits found\n", nHits);
		AliHLTTPCCATrack& track = sliceNeighbour.fTracks[sliceNeighbour.fCommonMem->fNTracks];
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
					//if (i == 0) tParam.TransportToX(sliceNeighbour.Row(rowIndex).X(), fParam.ConstBz(), HLTCA_MAX_SIN_PHI); //Use transport with new linearisation, we have changed the track in between - NOT needed, fitting will always start at outer end of global track!
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
		track.SetLocalTrackId((fParam.ISlice() << 24) | fTracks[iTrack].LocalTrackId());
		sliceNeighbour.fCommonMem->fNTracks++;
		sliceNeighbour.fCommonMem->fNTrackHits += nHits;
	}

#ifdef GLOBAL_TRACKING_MAINTAIN_TRACKLETS
	for (int i = 0;i < HLTCA_ROW_COUNT;i++)
	{
#ifdef EXTERN_ROW_HITS
		sliceNeighbour.TrackletRowHits()[i * *sliceNeighbour.NTracklets()] = tmphits[i];
#else
		sliceNeighbour.Tracklet(0).SetRowHit(i, tmphits[i]);
#endif
	}
#endif

	return(nHits >= GLOBAL_TRACKING_MIN_HITS);
}

GPUh() void AliHLTTPCCATracker::PerformGlobalTracking(AliHLTTPCCATracker& sliceLeft, AliHLTTPCCATracker& sliceRight, int MaxTracks)
{
	StartTimer(8);
	int ul = 0, ur = 0, ll = 0, lr = 0;
	for (int i = 0;i < fCommonMem->fNLocalTracks;i++)
	{
		if (sliceLeft.fCommonMem->fNTracks >= MaxTracks || sliceRight.fCommonMem->fNTracks >= MaxTracks)  {printf("Insufficient memory for global tracking\n"); return;}

		{
			const int tmpHit = fTracks[i].FirstHitID();
			if (fTrackHits[tmpHit].RowIndex() >= GLOBAL_TRACKING_MIN_ROWS && fTrackHits[tmpHit].RowIndex() < GLOBAL_TRACKING_RANGE)
			{
				int rowIndex = fTrackHits[tmpHit].RowIndex();
				const AliHLTTPCCARow& row = Row(rowIndex);
				float Y = (float) Data().HitDataY(row, fTrackHits[tmpHit].HitIndex()) * row.HstepY() + row.Grid().YMin();
				if (Y < -row.MaxY() * GLOBAL_TRACKING_Y_RANGE_LOWER_LEFT)
				{
					//printf("Track %d, lower row %d, left border (%f of %f)\n", i, fTrackHits[tmpHit].RowIndex(), Y, -row.MaxY());
					ll += PerformGlobalTrackingRun(sliceLeft, i, rowIndex, -fParam.DAlpha(), -1);
				}
				if (sliceRight.fCommonMem->fNTracks >= MaxTracks) {printf("Insufficient memory for global tracking\n"); return;}
				if (Y > row.MaxY() * GLOBAL_TRACKING_Y_RANGE_LOWER_RIGHT)
				{
					//printf("Track %d, lower row %d, right border (%f of %f)\n", i, fTrackHits[tmpHit].RowIndex(), Y, row.MaxY());
					lr += PerformGlobalTrackingRun(sliceRight, i, rowIndex, fParam.DAlpha(), -1);
				}
			}
		}
		
		{
			const int tmpHit = fTracks[i].FirstHitID() + fTracks[i].NHits() - 1;
			if (fTrackHits[tmpHit].RowIndex() < HLTCA_ROW_COUNT - GLOBAL_TRACKING_MIN_ROWS && fTrackHits[tmpHit].RowIndex() >= HLTCA_ROW_COUNT - GLOBAL_TRACKING_RANGE)
			{
				int rowIndex = fTrackHits[tmpHit].RowIndex();
				const AliHLTTPCCARow& row = Row(rowIndex);
				float Y = (float) Data().HitDataY(row, fTrackHits[tmpHit].HitIndex()) * row.HstepY() + row.Grid().YMin();
				if (Y < -row.MaxY() * GLOBAL_TRACKING_Y_RANGE_UPPER_LEFT)
				{
					//printf("Track %d, upper row %d, left border (%f of %f)\n", i, fTrackHits[tmpHit].RowIndex(), Y, -row.MaxY());
					ul += PerformGlobalTrackingRun(sliceLeft, i, rowIndex, -fParam.DAlpha(), 1);
				}
				if (sliceLeft.fCommonMem->fNTracks >= MaxTracks) {printf("Insufficient memory for global tracking\n"); return;}
				if (Y > row.MaxY() * GLOBAL_TRACKING_Y_RANGE_UPPER_RIGHT)
				{
					//printf("Track %d, upper row %d, right border (%f of %f)\n", i, fTrackHits[tmpHit].RowIndex(), Y, row.MaxY());
					ur += PerformGlobalTrackingRun(sliceRight, i, rowIndex, fParam.DAlpha(), 1);
				}
			}
		}
	}
	StopTimer(8);
	//printf("Global Tracking Result: Slide %2d: LL %3d LR %3d UL %3d UR %3d\n", fParam.ISlice(), ll, lr, ul, ur);
}

#endif
#endif
