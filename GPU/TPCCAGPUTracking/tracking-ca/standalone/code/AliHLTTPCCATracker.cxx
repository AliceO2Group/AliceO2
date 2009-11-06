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
#include "AliHLTTPCCAOutTrack.h"
#include "AliHLTTPCCARow.h"
#include "AliHLTTPCCATrack.h"
#include "AliHLTTPCCATracklet.h"
#include "AliHLTTPCCAMath.h"
#include "MemoryAssignmentHelpers.h"

#include "TStopwatch.h"
#include "AliHLTTPCCAHitArea.h"
#include "AliHLTTPCCANeighboursFinder.h"
#include "AliHLTTPCCANeighboursCleaner.h"
#include "AliHLTTPCCAStartHitsFinder.h"
#include "AliHLTTPCCATrackletConstructor.h"
#include "AliHLTTPCCATrackletSelector.h"
#include "AliHLTTPCCAProcess.h"
#include "AliHLTTPCCASliceTrack.h"
#include "AliHLTTPCCASliceOutput.h"
#include "AliHLTTPCCAClusterData.h"

#include "AliHLTTPCCATrackParam.h"

#include "AliHLTTPCCAGPUConfig.h"

#if !defined(HLTCA_GPUCODE)
#include <iostream>
#include <iomanip>
#include <string.h>
#endif

//#define DRAW1

#ifdef DRAW1
#include "AliHLTTPCCADisplay.h"
#endif //DRAW1

#ifdef HLTCA_INTERNAL_PERFORMANCE
//#include "AliHLTTPCCAPerformance.h"
#endif

#ifdef HLTCA_STANDALONE
#include "AliHLTTPCCAStandaloneFramework.h"
#endif

ClassImp( AliHLTTPCCATracker )


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
	AssignMemory(fGPUTrackletTemp, pGPUMemory, MaxNTracks);
	AssignMemory(fRowBlockTracklets, pGPUMemory, MaxNTracks * 2 * (Param().NRows() / HLTCA_GPU_SCHED_ROW_STEP + 1));
	AssignMemory(fRowBlockPos, pGPUMemory, 2 * (Param().NRows() / HLTCA_GPU_SCHED_ROW_STEP + 1));
	AssignMemory(fBlockStartingTracklet, pGPUMemory, HLTCA_GPU_BLOCK_COUNT);

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

void AliHLTTPCCATracker::DumpSliceData(std::ostream &out)
{
	//Dump Slice Input Data to File
	out << "Slice Data:" << std::endl;
	for (int i = 0;i < Param().NRows();i++)
	{
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
	out << "Hit Links:" << std::endl;
	for (int i = 0;i < Param().NRows();i++)
	{
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
    out << "Hit Weights:" << std::endl;
    for (int i = 0;i < Param().NRows();i++)
    {
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
	out << "Start Hits: (" << *NTracklets() << ")" << std::endl;
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
	out << "Tracks: (" << *NTracks() << ")" << std::endl;
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
					out << "(Track: " << j << ")" << std::endl;
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
	out << "Tracklets: (" << *NTracklets() << ")" << std::endl;
#ifdef HLTCA_GPU_SORT_DUMPDATA
	AliHLTTPCCAHitId* tmpIds = new AliHLTTPCCAHitId[*NTracklets()];
	AliHLTTPCCATracklet* tmpTracklets = new AliHLTTPCCATracklet[*NTracklets()];
	memcpy(tmpIds, TrackletStartHits(), *NTracklets() * sizeof(AliHLTTPCCAHitId));
	memcpy(tmpTracklets, Tracklets(), *NTracklets() * sizeof(AliHLTTPCCATracklet));
#ifdef EXTERN_ROW_HITS
	int* tmpHits = new int[*NTracklets() * Param().NRows()];
	memcpy(tmpHits, TrackletRowHits(), *NTracklets() * Param().NRows() * sizeof(int));
#endif
	qsort(TrackletStartHits(), *NTracklets(), sizeof(AliHLTTPCCAHitId), StarthitSortComparison);
	for (int i = 0;i < *NTracklets();i++)
	{
		for (int j = 0;j < *NTracklets();j++)
		{
			if (tmpIds[i].RowIndex() == TrackletStartHit(j).RowIndex() && tmpIds[i].HitIndex() == TrackletStartHit(j).HitIndex())
			{
				memcpy(&Tracklets()[j], &tmpTracklets[i], sizeof(AliHLTTPCCATracklet));
#ifdef EXTERN_ROW_HITS
				if (tmpTracklets[i].NHits())
				{
					for (int k = tmpTracklets[i].FirstRow();k <= tmpTracklets[i].LastRow();k++)
					{
						fTrackletRowHits[k * *NTracklets() + j] = tmpHits[k * *NTracklets() + i];
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
	for (int j = 0;j < *NTracklets();j++)
	{
		out << "Tracklet " << j << " (Hits: " << std::setw(3) << Tracklets()[j].NHits() << ", Start: " << std::setw(3) << TrackletStartHit(j).RowIndex() << "-" << std::setw(3) << TrackletStartHit(j).HitIndex() << ") ";
		if (Tracklets()[j].NHits() == 0);
		else if (Tracklets()[j].LastRow() > Tracklets()[j].FirstRow() && (Tracklets()[j].FirstRow() >= Param().NRows() || Tracklets()[j].LastRow() >= Param().NRows()))
		{
#ifdef HLTCA_STANDALONE
			printf("\nError: First %d Last %d Num %d", Tracklets()[j].FirstRow(), Tracklets()[j].LastRow(), Tracklets()[j].NHits());
#endif
		}
		else if (Tracklets()[j].NHits() && Tracklets()[j].LastRow() > Tracklets()[j].FirstRow())
		{
			for (int i = Tracklets()[j].FirstRow();i <= Tracklets()[j].LastRow();i++)
			{
				//if (Tracklets()[j].RowHit(i) != -1)
#ifdef EXTERN_ROW_HITS
					out << i << "-" << fTrackletRowHits[i * fCommonMem->fNTracklets + j] << ", ";
#else
					out << i << "-" << Tracklets()[j].RowHit(i) << ", ";
#endif
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

void AliHLTTPCCATracker::ReadEvent( AliHLTTPCCAClusterData *clusterData )
{
  // read event

  fClusterData = clusterData;

  StartEvent();

  //* Convert input hits, create grids, etc.
  fData.InitFromClusterData( *clusterData );
  {
    if (!fIsGPUTracker)
	{
		SetPointersHits( fData.NumberOfHits() ); // to calculate the size
		fHitMemory = reinterpret_cast<char*> ( new uint4 [ fHitMemorySize/sizeof( uint4 ) + 100] );
	}
    SetPointersHits( fData.NumberOfHits() ); // set pointers for hits
  }
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
      if (fOutput == NULL) return(1);
      useOutput->SetNTracks( 0 );
      useOutput->SetNTrackClusters( 0 );
      useOutput->SetNOutTracks(0);
      useOutput->SetNOutTrackHits(0);
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
  AliHLTTPCCATrackletConstructor::AliHLTTPCCATrackletConstructorNewCPU(*this);
}

void AliHLTTPCCATracker::RunTrackletSelector()
{
	//Run CPU Tracklet Selector
  AliHLTTPCCAProcess<AliHLTTPCCATrackletSelector>( 1, fCommonMem->fNTracklets, *this );
}

#ifdef HLTCA_STANDALONE
void AliHLTTPCCATracker::StandalonePerfTime(int i)
{
  //Query Performance Timer for Standalone Version of Tracker
  if (fGPUDebugLevel >= 1)
  {
    AliHLTTPCCAStandaloneFramework::StandaloneQueryTime(&fPerfTimers[i]);
  }
}
#else
void AliHLTTPCCATracker::StandalonePerfTime(int /*i*/) {}
#endif

GPUh() void AliHLTTPCCATracker::Reconstruct()
{
  //* reconstruction of event
  //std::cout<<"Reconstruct slice "<<fParam.ISlice()<<", nHits="<<NHitsTotal()<<std::endl;

  fTimers[0] = 0; // find neighbours
  fTimers[1] = 0; // construct tracklets
  fTimers[2] = 0; // fit tracklets
  fTimers[3] = 0; // prolongation of tracklets
  fTimers[4] = 0; // selection
  fTimers[5] = 0; // write output
  fTimers[6] = 0;
  fTimers[7] = 0;

  //if( fParam.ISlice()<1 ) return; //SG!!!

  TStopwatch timer0;

  StandalonePerfTime(0);

  if (CheckEmptySlice()) return;

#ifdef DRAW1
  //if( fParam.ISlice()==2 || fParam.ISlice()==3)
    {
  AliHLTTPCCADisplay::Instance().ClearView();
  AliHLTTPCCADisplay::Instance().SetSliceView();
  AliHLTTPCCADisplay::Instance().SetCurrentSlice( this );
  AliHLTTPCCADisplay::Instance().DrawSlice( this, 1 );
  if ( NHitsTotal() > 0 ) {
    AliHLTTPCCADisplay::Instance().DrawSliceHits( kRed, .5 );
    AliHLTTPCCADisplay::Instance().Ask();
  }
  }
#endif //DRAW1

	fCommonMem->fNTracklets = fCommonMem->fNTracks = fCommonMem->fNTrackHits = 0;

#if !defined(HLTCA_GPUCODE)

  if (fGPUDebugLevel >= 6)
  {
	  *fGPUDebugOut << std::endl << std::endl << "Slice: " << Param().ISlice() << std::endl;
	  *fGPUDebugOut << "Slice Data:" << std::endl;
	  DumpSliceData(*fGPUDebugOut);
  }

  StandalonePerfTime(1);

  RunNeighboursFinder();

  StandalonePerfTime(2);

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

  RunNeighboursCleaner();

  StandalonePerfTime(3);

  if (fGPUDebugLevel >= 6) DumpLinks(*fGPUDebugOut);

  RunStartHitsFinder();

  StandalonePerfTime(4);
  StandalonePerfTime(5);

  if (fGPUDebugLevel >= 6) DumpStartHits(*fGPUDebugOut);
  
  fData.ClearHitWeights();

  SetPointersTracklets( fCommonMem->fNTracklets * 2 ); // to calculate the size
  fTrackletMemory = reinterpret_cast<char*> ( new uint4 [ fTrackletMemorySize/sizeof( uint4 ) + 100] );
  SetPointersTracklets( fCommonMem->fNTracklets * 2 ); // set pointers for hits

  SetPointersTracks( fCommonMem->fNTracklets * 2, NHitsTotal() ); // to calculate the size
  fTrackMemory = reinterpret_cast<char*> ( new uint4 [ fTrackMemorySize/sizeof( uint4 ) + 100] );
  SetPointersTracks( fCommonMem->fNTracklets * 2, NHitsTotal() ); // set pointers for hits

  StandalonePerfTime(6);
  StandalonePerfTime(7);

  RunTrackletConstructor();

  StandalonePerfTime(8);

  if (fGPUDebugLevel >= 6) DumpTrackletHits(*fGPUDebugOut);
  if (fGPUDebugLevel >= 6) DumpHitWeights(*fGPUDebugOut);

  //std::cout<<"Slice "<<Param().ISlice()<<": NHits="<<NHitsTotal()<<", NTracklets="<<*NTracklets()<<std::endl;

  RunTrackletSelector();

  StandalonePerfTime(9);

  //std::cout<<"Slice "<<Param().ISlice()<<": N start hits/tracklets/tracks = "<<nStartHits<<" "<<nStartHits<<" "<<*fNTracks<<std::endl;

  if (fGPUDebugLevel >= 6) DumpTrackHits(*fGPUDebugOut);

  //std::cout<<"Memory used for slice "<<fParam.ISlice()<<" : "<<fCommonMemorySize/1024./1024.<<" + "<<fHitMemorySize/1024./1024.<<" + "<<fTrackMemorySize/1024./1024.<<" = "<<( fCommonMemorySize+fHitMemorySize+fTrackMemorySize )/1024./1024.<<" Mb "<<std::endl;

  WriteOutput();

  StandalonePerfTime(10);

#endif

#ifdef DRAW1
  {
    AliHLTTPCCADisplay &disp = AliHLTTPCCADisplay::Instance();
    AliHLTTPCCATracker &slice = *this;
    std::cout << "N out tracks = " << slice.NOutTracks() << std::endl;
    AliHLTTPCCADisplay::Instance().SetSliceView();
    AliHLTTPCCADisplay::Instance().SetCurrentSlice( this );
    AliHLTTPCCADisplay::Instance().DrawSlice( this, 1 );
    disp.DrawSliceHits( kRed, .5 );
    disp.Ask();
    for ( int itr = 0; itr < slice.NOutTracks(); itr++ ) {
      std::cout << "track N " << itr << ", nhits=" << slice.OutTracks()[itr].NHits() << std::endl;
      disp.DrawSliceOutTrack( itr, kBlue );      
      //disp.Ask();
      //int id = slice.OutTracks()[itr].OrigTrackID();
      //AliHLTTPCCATrack &tr = Tracks()[id];
      //for( int ih=0; ih<tr.NHits(); ih++ ){
      //int ic = (fTrackHits[tr.FirstHitID()+ih]);
      //std::cout<<ih<<" "<<ID2IRow(ic)<<" "<<ID2IHit(ic)<<std::endl;
      //}
      //disp.DrawSliceTrack( id, kBlue );
      //disp.Ask();
    }
    disp.Ask();
  }
#endif //DRAW1

  timer0.Stop();
  fTimers[0] = timer0.CpuTime() / 100.;

}

GPUh() void AliHLTTPCCATracker::WriteOutput()
{
  // write output

  TStopwatch timer;

  //cout<<"output: nTracks = "<<*fNTracks<<", nHitsTotal="<<NHitsTotal()<<std::endl;

  if (fOutputControl == NULL) fOutputControl = new AliHLTTPCCASliceOutput::outputControlStruct;
  AliHLTTPCCASliceOutput::Allocate(*fOutput, fCommonMem->fNTracks, fCommonMem->fNTrackHits, fOutputControl);
  AliHLTTPCCASliceOutput* useOutput = *fOutput;
  if (useOutput == NULL) return;

  if (fOutputControl->fDefaultOutput)
  {
	  useOutput->SetNTracks( fCommonMem->fNTracks );
	  useOutput->SetNTrackClusters( fCommonMem->fNTrackHits );

	  int nStoredHits = 0;

	  for ( int iTr = 0; iTr < fCommonMem->fNTracks; iTr++ ) {
		AliHLTTPCCATrack &iTrack = fTracks[iTr];

		AliHLTTPCCASliceTrack out;
		out.SetFirstClusterRef( nStoredHits );
		out.SetNClusters( iTrack.NHits() );
		out.SetParam( iTrack.Param() );

		useOutput->SetTrack( iTr, out );

		int iID = iTrack.FirstHitID();
		for ( int ith = 0; ith < iTrack.NHits(); ith++ ) {
		  const AliHLTTPCCAHitId &ic = fTrackHits[iID + ith];
		  int iRow = ic.RowIndex();
		  int ih = ic.HitIndex();

		  const AliHLTTPCCARow &row = fData.Row( iRow );

		  //float y0 = row.Grid().YMin();
		  //float z0 = row.Grid().ZMin();
		  //float stepY = row.HstepY();
		  //float stepZ = row.HstepZ();
		  //float x = row.X();

		  //const uint4 *tmpint4 = RowData() + row.FullOffset();
		  //const ushort2 *hits = reinterpret_cast<const ushort2*>(tmpint4);
		  //ushort2 hh = hits[ih];
		  //float y = y0 + hh.x*stepY;
		  //float z = z0 + hh.y*stepZ;

		  int clusterIndex = fData.ClusterDataIndex( row, ih );
		  int clusterRowIndex = clusterIndex - fClusterData->RowOffset( iRow );

		  if ( clusterIndex < 0 || clusterIndex >= fClusterData->NumberOfClusters() ) {
			//std::cout << inpIDtot << ", " << fClusterData->NumberOfClusters()
			//<< "; " << inpID << ", " << fClusterData->NumberOfClusters( iRow ) << std::endl;
			//abort();
			continue;
		  }
		  if ( clusterRowIndex < 0 || clusterRowIndex >= fClusterData->NumberOfClusters( iRow ) ) {
			//std::cout << inpIDtot << ", " << fClusterData->NumberOfClusters()
			//<< "; " << inpID << ", " << fClusterData->NumberOfClusters( iRow ) << std::endl;
			//abort();
			continue;
		  }

		  float origX = fClusterData->X( clusterIndex );
		  float origY = fClusterData->Y( clusterIndex );
		  float origZ = fClusterData->Z( clusterIndex );


		  int id = fClusterData->Id( clusterIndex );

		  float2 hUnpackedYZ;
		  hUnpackedYZ.x = origY;
		  hUnpackedYZ.y = origZ;
		  float hUnpackedX = origX;

		  useOutput->SetClusterId( nStoredHits, id  );
		  useOutput->SetClusterRow( nStoredHits, ( unsigned char ) iRow  );
		  useOutput->SetClusterUnpackedYZ( nStoredHits, hUnpackedYZ );
		  useOutput->SetClusterUnpackedX( nStoredHits, hUnpackedX );
		  nStoredHits++;
		}
	  }
  }


  // old stuff
  if (fOutputControl->fObsoleteOutput)
  {
	  useOutput->SetNOutTrackHits(0);
	  useOutput->SetNOutTracks(0);


	  for ( int iTr = 0; iTr < fCommonMem->fNTracks; iTr++ ) {

		const AliHLTTPCCATrack &iTrack = fTracks[iTr];

		//std::cout<<"iTr = "<<iTr<<", nHits="<<iTrack.NHits()<<std::endl;

		//if( !iTrack.Alive() ) continue;
		if ( iTrack.NHits() < 3 ) continue;
		AliHLTTPCCAOutTrack &out = useOutput->OutTracks()[useOutput->NOutTracks()];
		out.SetFirstHitRef( useOutput->NOutTrackHits() );
		out.SetNHits( 0 );
		out.SetOrigTrackID( iTr );
		AliHLTTPCCATrackParam tmpParam;
		tmpParam.InitParam();
		tmpParam.SetParam(iTrack.Param());
		out.SetStartPoint( tmpParam );
		out.SetEndPoint( tmpParam );

		int iID = iTrack.FirstHitID();
		int nOutTrackHitsOld = useOutput->NOutTrackHits();

		for ( int ith = 0; ith < iTrack.NHits(); ith++ ) {
		  const AliHLTTPCCAHitId &ic = fTrackHits[iID + ith];
		  const AliHLTTPCCARow &row = Row( ic );
		  int ih = ic.HitIndex();
		  useOutput->SetOutTrackHit(useOutput->NOutTrackHits(), HitInputID( row, ih ));
		  useOutput->SetNOutTrackHits(useOutput->NOutTrackHits() + 1 );
		  //std::cout<<"write i,row,hit,id="<<ith<<", "<<ID2IRow(ic)<<", "<<ih<<", "<<HitInputID( row, ih )<<std::endl;
		  if ( useOutput->NOutTrackHits() >= 10*NHitsTotal() ) {
			std::cout << "fNOutTrackHits>NHitsTotal()" << std::endl;
			//exit(0);
			return;//SG!!!
		  }
		  out.SetNHits( out.NHits() + 1 );
		}
		if ( out.NHits() >= 2 ) {
		  useOutput->SetNOutTracks(useOutput->NOutTracks() + 1);
		} else {
		  useOutput->SetNOutTrackHits(nOutTrackHitsOld);
		}
	  }
  }

  timer.Stop();
  fTimers[5] += timer.CpuTime();
}

#endif

GPUh() void AliHLTTPCCATracker::FitTrackFull( const AliHLTTPCCATrack &/**/, float * /**/ ) const
{
  // fit track with material
#ifdef XXX
  //* Fit the track
  FitTrack( iTrack, tt0 );
  if ( iTrack.NHits() <= 3 ) return;

  AliHLTTPCCATrackParam &t = iTrack.Param();
  AliHLTTPCCATrackParam t0 = t;

  t.Chi2() = 0;
  t.NDF() = -5;
  bool first = 1;

  int iID = iTrack.FirstHitID();
  for ( int ih = 0; ih < iTrack.NHits(); ih++, iID++ ) {
    const AliHLTTPCCAHitId &ic = fTrackHits[iID];
    int iRow = ic.rowIndex();
    const AliHLTTPCCARow &row = fData.Row( iRow );
    if ( !t0.TransportToX( row.X() ) ) continue;
    float dy, dz;
    const AliHLTTPCCAHit &h = ic.hitIndex();

    // check for wrong hits
    if ( 0 ) {
      dy = t0.GetY() - h.Y();
      dz = t0.GetZ() - h.Z();

      //if( dy*dy > 3.5*3.5*(/*t0.GetErr2Y() + */h.ErrY()*h.ErrY() ) ) continue;//SG!!!
      //if( dz*dz > 3.5*3.5*(/*t0.GetErr2Z() + */h.ErrZ()*h.ErrZ() ) ) continue;
    }

    if ( !t.TransportToX( row.X() ) ) continue;

    //* Update the track

    if ( first ) {
      t.Cov()[ 0] = .5 * .5;
      t.Cov()[ 1] = 0;
      t.Cov()[ 2] = .5 * .5;
      t.Cov()[ 3] = 0;
      t.Cov()[ 4] = 0;
      t.Cov()[ 5] = .2 * .2;
      t.Cov()[ 6] = 0;
      t.Cov()[ 7] = 0;
      t.Cov()[ 8] = 0;
      t.Cov()[ 9] = .2 * .2;
      t.Cov()[10] = 0;
      t.Cov()[11] = 0;
      t.Cov()[12] = 0;
      t.Cov()[13] = 0;
      t.Cov()[14] = .2 * .2;
      t.Chi2() = 0;
      t.NDF() = -5;
    }
    float err2Y, err2Z;
    GetErrors2( iRow, t, err2Y, err2Z );

    if ( !t.Filter2( h.Y(), h.Z(), err2Y, err2Z ) ) continue;

    first = 0;
  }
  /*
  float cosPhi = iTrack.Param().GetCosPhi();
  p0.Param().TransportToX(ID2Row( iTrack.PointID()[0] ).X());
  p2.Param().TransportToX(ID2Row( iTrack.PointID()[1] ).X());
  if( p0.Param().GetCosPhi()*cosPhi<0 ){ // change direction
  float *par = p0.Param().Par();
  float *cov = p0.Param().Cov();
  par[2] = -par[2]; // sin phi
  par[3] = -par[3]; // DzDs
    par[4] = -par[4]; // kappa
    cov[3] = -cov[3];
    cov[4] = -cov[4];
    cov[6] = -cov[6];
    cov[7] = -cov[7];
    cov[10] = -cov[10];
    cov[11] = -cov[11];
    p0.Param().CosPhi() = -p0.Param().GetCosPhi();
  }
  */
#endif
}

GPUh() void AliHLTTPCCATracker::FitTrack( const AliHLTTPCCATrack &/*track*/, float * /*t0[]*/ ) const
{
  //* Fit the track
#ifdef XXX
  AliHLTTPCCAEndPoint &p2 = ID2Point( track.PointID()[1] );
  const AliHLTTPCCAHit &c0 = ID2Hit( fTrackHits[p0.TrackHitID()].HitID() );
  const AliHLTTPCCAHit &c1 = ID2Hit( fTrackHits[track.HitID()[1]].HitID() );
  const AliHLTTPCCAHit &c2 = ID2Hit( fTrackHits[p2.TrackHitID()].HitID() );
  const AliHLTTPCCARow &row0 = ID2Row( fTrackHits[p0.TrackHitID()].HitID() );
  const AliHLTTPCCARow &row1 = ID2Row( fTrackHits[track.HitID()[1]].HitID() );
  const AliHLTTPCCARow &row2 = ID2Row( fTrackHits[p2.TrackHitID()].HitID() );
  float sp0[5] = {row0.X(), c0.Y(), c0.Z(), c0.ErrY(), c0.ErrZ() };
  float sp1[5] = {row1.X(), c1.Y(), c1.Z(), c1.ErrY(), c1.ErrZ() };
  float sp2[5] = {row2.X(), c2.Y(), c2.Z(), c2.ErrY(), c2.ErrZ() };
  //std::cout<<"Fit track, points ="<<sp0[0]<<" "<<sp0[1]<<" / "<<sp1[0]<<" "<<sp1[1]<<" / "<<sp2[0]<<" "<<sp2[1]<<std::endl;
  if ( track.NHits() >= 3 ) {
    p0.Param().ConstructXYZ3( sp0, sp1, sp2, p0.Param().CosPhi(), t0 );
    p2.Param().ConstructXYZ3( sp2, sp1, sp0, p2.Param().CosPhi(), t0 );
    //p2.Param() = p0.Param();
    //p2.Param().TransportToX(row2.X());
    //p2.Param().Par()[1] = -p2.Param().Par()[1];
    //p2.Param().Par()[4] = -p2.Param().Par()[4];
  } else {
    p0.Param().X() = row0.X();
    p0.Param().Y() = c0.Y();
    p0.Param().Z() = c0.Z();
    p0.Param().Err2Y() = c0.ErrY() * c0.ErrY();
    p0.Param().Err2Z() = c0.ErrZ() * c0.ErrZ();
    p2.Param().X() = row2.X();
    p2.Param().Y() = c2.Y();
    p2.Param().Z() = c2.Z();
    p2.Param().Err2Y() = c2.ErrY() * c2.ErrY();
    p2.Param().Err2Z() = c2.ErrZ() * c2.ErrZ();
  }
#endif
}


GPUd() void AliHLTTPCCATracker::GetErrors2( int iRow, float z, float sinPhi, float cosPhi, float DzDs, float &Err2Y, float &Err2Z ) const
{
  //
  // Use calibrated cluster error from OCDB
  //

  fParam.GetClusterErrors2( iRow, z, sinPhi, cosPhi, DzDs, Err2Y, Err2Z );
  Err2Y*=fParam.ClusterError2CorrectionY();
  Err2Z*=fParam.ClusterError2CorrectionZ();
}

GPUd() void AliHLTTPCCATracker::GetErrors2( int iRow, const AliHLTTPCCATrackParam &t, float &Err2Y, float &Err2Z ) const
{
  //
  // Use calibrated cluster error from OCDB
  //

  fParam.GetClusterErrors2( iRow, t.GetZ(), t.SinPhi(), t.GetCosPhi(), t.DzDs(), Err2Y, Err2Z );
}


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

GPUh() void AliHLTTPCCATracker::WriteTracks( std::ostream &out )
{
  //* Write tracks to file
  AliHLTTPCCASliceOutput* useOutput = *fOutput;

  out << fTimers[0] << std::endl;
  out << useOutput->NOutTrackHits() << std::endl;
  for ( int ih = 0; ih < useOutput->NOutTrackHits(); ih++ ) {
    out << useOutput->OutTrackHit(ih) << " ";
  }
  out << std::endl;

  out << useOutput->NOutTracks() << std::endl;

  for ( int itr = 0; itr < useOutput->NOutTracks(); itr++ ) {
    const AliHLTTPCCAOutTrack &t = useOutput->OutTrack(itr);
    AliHLTTPCCATrackParam p1 = t.StartPoint();
    AliHLTTPCCATrackParam p2 = t.EndPoint();
    out << t.NHits() << " ";
    out << t.FirstHitRef() << " ";
    out << t.OrigTrackID() << " ";
    out << std::endl;
    out << p1.X() << " ";
    out << p1.SignCosPhi() << " ";
    out << p1.Chi2() << " ";
    out << p1.NDF() << std::endl;
    for ( int i = 0; i < 5; i++ ) out << p1.Par()[i] << " ";
    out << std::endl;
    for ( int i = 0; i < 15; i++ ) out << p1.Cov()[i] << " ";
    out << std::endl;
    out << p2.X() << " ";
    out << p2.SignCosPhi() << " ";
    out << p2.Chi2() << " ";
    out << p2.NDF() << std::endl;
    for ( int i = 0; i < 5; i++ ) out << p2.Par()[i] << " ";
    out << std::endl;
    for ( int i = 0; i < 15; i++ ) out << p2.Cov()[i] << " ";
    out << std::endl;
  }
}

GPUh() void AliHLTTPCCATracker::ReadTracks( std::istream &in )
{
  //* Read tracks  from file
  AliHLTTPCCASliceOutput::Allocate(*fOutput, 4096, 16384, fOutputControl);//Just some max values
  AliHLTTPCCASliceOutput* useOutput = *fOutput;

  int tmpval;
  in >> fTimers[0];
  in >> tmpval;
  useOutput->SetNOutTrackHits(tmpval);

  for ( int ih = 0; ih < useOutput->NOutTrackHits(); ih++ ) {
    in >> tmpval;
	useOutput->SetOutTrackHit(ih, tmpval);
  }
  in >> tmpval;
  useOutput->SetNOutTracks(tmpval);

  for ( int itr = 0; itr < useOutput->NOutTracks(); itr++ ) {
    AliHLTTPCCAOutTrack &t = useOutput->OutTracks()[itr];
    AliHLTTPCCATrackParam p1, p2;
    int i;
    float f;
    in >> i; t.SetNHits( i );
    in >> i; t.SetFirstHitRef( i );
    in >> i; t.SetOrigTrackID( i );
    in >> f; p1.SetX( f );
    in >> f; p1.SetSignCosPhi( f );
    in >> f; p1.SetChi2( f );
    in >> i; p1.SetNDF( i );
    for ( int j = 0; j < 5; j++ ) { in >> f; p1.SetPar( j, f ); }
    for ( int j = 0; j < 15; j++ ) { in >> f; p1.SetCov( j, f ); }
    in >> f; p2.SetX( f );
    in >> f; p2.SetSignCosPhi( f );
    in >> f; p2.SetChi2( f );
    in >> i; p2.SetNDF( i );
    for ( int j = 0; j < 5; j++ ) { in >> f; p2.SetPar( j, f ); }
    for ( int j = 0; j < 15; j++ ) { in >> f; p2.SetCov( j, f ); }
    t.SetStartPoint( p1 );
    t.SetEndPoint( p2 );
  }
}
#endif
