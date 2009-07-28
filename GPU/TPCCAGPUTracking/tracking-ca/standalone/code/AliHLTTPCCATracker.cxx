// @(#) $Id: AliHLTTPCCATracker.cxx 33907 2009-07-23 13:52:49Z sgorbuno $
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
#include "AliHLTTPCCAGrid.h"
#include "AliHLTTPCCARow.h"
#include "AliHLTTPCCATrack.h"
#include "AliHLTTPCCATracklet.h"
#include "AliHLTTPCCAMath.h"
#include "AliHLTTPCCAHit.h"
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
#include "AliHLTTPCCADataCompressor.h"
#include "AliHLTTPCCAClusterData.h"

#include "AliHLTTPCCATrackParam.h"

#if !defined(HLTCA_GPUCODE)
#include <iostream>
#endif

//#define DRAW1

#ifdef DRAW1
#include "AliHLTTPCCADisplay.h"
#endif //DRAW

#ifdef HLTCA_INTERNAL_PERFORMANCE
#include "AliHLTTPCCAPerformance.h"
#endif


ClassImp( AliHLTTPCCATracker )

GPUd() AliHLTTPCCATracker::~AliHLTTPCCATracker()
{
  // destructor
	if (!fIsGPUTracker)
	{
		delete[] fCommonMemory;
		delete[] fHitMemory;
		delete[] fTrackMemory;
	}
}

#if !defined(HLTCA_GPUCODE)


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
	fIsGPUTracker = true;
}

char* AliHLTTPCCATracker::SetGPUTrackerCommonMemory(char* pGPUMemory)
{
	fCommonMemory = (char*) pGPUMemory;
	SetPointersCommon();
	return(pGPUMemory + fCommonMemorySize);
}


char* AliHLTTPCCATracker::SetGPUTrackerHitsMemory(char* pGPUMemory, int MaxNHits )
{
	fHitMemory = (char*) pGPUMemory;
	SetPointersHits(MaxNHits);
	return(pGPUMemory + fHitMemorySize);
}


char* AliHLTTPCCATracker::SetGPUTrackerTracksMemory(char* pGPUMemory, int MaxNTracks, int MaxNHits )
{
	fTrackMemory = (char*) pGPUMemory;
	SetPointersTracks(MaxNTracks, MaxNHits);
	return(pGPUMemory + fTrackMemorySize);
}

void AliHLTTPCCATracker::DumpLinks(std::ostream &out)
{
	for (int i = 0;i < Param().NRows();i++)
	{
		out << "Row: " << i << endl;
		for (int j = 0;j < Row(i).NHits();j++)
		{
			out << HitLinkUpData(Row(i), j) << ", ";
		}
		out << endl;
	}
}

void AliHLTTPCCATracker::DumpStartHits(std::ostream &out)
{
	for (int j = 0;j < Param().NRows();j++)
	{
		for (int i = 0;i < *NTracklets();i++)
		{
			if (TrackletStartHit(i).RowIndex() == j)
				out << TrackletStartHit(i).RowIndex() << "-" << TrackletStartHit(i).HitIndex() << endl;
		}
	}
	out << endl;
}

void AliHLTTPCCATracker::DumpTrackHits(std::ostream &out)
{
	for (int k = 0;k < Param().NRows();k++)
	{
		for (int j = 0;j < *NTracks();j++)
		{
			if (Tracks()[j].NHits() == 0 || !Tracks()[j].Alive()) continue;
			if (TrackHits()[Tracks()[j].FirstHitID()].RowIndex() == k)
			{
				for (int i = 0;i < Tracks()[j].NHits();i++)
				{
					out << TrackHits()[Tracks()[j].FirstHitID() + i].RowIndex() << "-" << TrackHits()[Tracks()[j].FirstHitID() + i].HitIndex() << ", ";
				}
				out << "(Track: " << j << ")" << endl;
			}
		}
	}
}

int trackletSortComparison(const void* a, const void* b)
{
	const AliHLTTPCCATracklet* aa = (AliHLTTPCCATracklet*) a;
	const AliHLTTPCCATracklet* bb = (AliHLTTPCCATracklet*) b;
	if (aa->NHits() == 0) return(-1);
	if (bb->NHits() == 0) return(1);
	if (aa->FirstRow() != bb->FirstRow())
	{
		return(aa->FirstRow() - bb->FirstRow());
	}
	for (int i = aa->FirstRow();i <= aa->LastRow();i++)
	{
		if (i >= bb->LastRow()) return(-1);
		if (aa->RowHit(i) != bb->RowHit(i))
		{
			return(aa->RowHit(i) - bb->RowHit(i));
		}
	}
	return(0);
}

void AliHLTTPCCATracker::DumpTrackletHits(std::ostream &out)
{
	qsort(Tracklets(), *NTracklets(), sizeof(AliHLTTPCCATracklet), trackletSortComparison);
	for (int k = 0;k < Param().NRows();k++)
	{
		for (int j = 0;j < *NTracklets();j++)
		{
			if (Tracklets()[j].NHits() == 0) continue;
			if (Tracklets()[j].LastRow() > Tracklets()[j].FirstRow() && (Tracklets()[j].FirstRow() >= Param().NRows() || Tracklets()[j].LastRow() >= Param().NRows()))
			{
				printf("\nError: First %d Last %d Num %d", Tracklets()[j].FirstRow(), Tracklets()[j].LastRow(), Tracklets()[j].NHits());
			}
			else if (Tracklets()[j].NHits() && Tracklets()[j].FirstRow() == k && Tracklets()[j].LastRow() > Tracklets()[j].FirstRow())
			{
				for (int i = Tracklets()[j].FirstRow();i <= Tracklets()[j].LastRow();i++)
				{
					if (Tracklets()[j].RowHit(i) != -1)
						out << i << "-" << Tracklets()[j].RowHit(i) << ", ";
				}
				out << endl;
			}
		}
	}
}


void  AliHLTTPCCATracker::SetupCommonMemory()
{
  // set up common memory

  if (!fIsGPUTracker)
  {
    if ( !fCommonMemory ) {
      SetPointersCommon(); // just to calculate the size
      // the 1600 extra bytes are not used unless fCommonMemorySize increases with a later event
      fCommonMemory = reinterpret_cast<char*> ( new uint4 [ fCommonMemorySize/sizeof( uint4 ) + 100] );
      SetPointersCommon();// set pointers
    }

    delete[] fHitMemory;
    delete[] fTrackMemory;
    fHitMemory = 0;
    fTrackMemory = 0;
  }

  fData.Clear();
  *fNTracklets = 0;
  *fNTracks = 0 ;
  *fNTrackHits = 0;
  *fNOutTracks = 0;
  *fNOutTrackHits = 0;
}

void AliHLTTPCCATracker::ReadEvent( AliHLTTPCCAClusterData *clusterData )
{
  // read event

  fClusterData = clusterData;

  StartEvent();

  //* Convert input hits, create grids, etc.
  fData.InitFromClusterData( *clusterData );

  {
    SetPointersHits( fData.NumberOfHits() ); // to calculate the size
    fHitMemory = reinterpret_cast<char*> ( new uint4 [ fHitMemorySize/sizeof( uint4 ) + 100] );
    SetPointersHits( fData.NumberOfHits() ); // set pointers for hits
    *fNTracklets = 0;
    *fNTracks = 0 ;
    *fNOutTracks = 0;
    *fNOutTrackHits = 0;
  }
}


GPUhd() void  AliHLTTPCCATracker::SetPointersCommon()
{
  // set all pointers to the event memory

  char *mem = fCommonMemory;
  AssignMemory( fNTracklets, mem, 1 );
  AssignMemory( fNTracks, mem, 1 );
  AssignMemory( fNTrackHits, mem, 1 );
  AssignMemory( fNOutTracks, mem, 1 );
  AssignMemory( fNOutTrackHits, mem, 1 );

  // calculate the size

  fCommonMemorySize = mem - fCommonMemory;
}


GPUhd() void  AliHLTTPCCATracker::SetPointersHits( int MaxNHits )
{
  // set all pointers to the event memory

  char *mem = fHitMemory;

  // extra arrays for tpc clusters

  AssignMemory( fTrackletStartHits, mem, MaxNHits );

  // arrays for track hits

  AssignMemory( fTrackHits, mem, 10 * MaxNHits );

  AssignMemory( fOutTrackHits, mem, 10 * MaxNHits );

  // calculate the size

  fHitMemorySize = mem - fHitMemory;
}


GPUhd() void  AliHLTTPCCATracker::SetPointersTracks( int MaxNTracks, int MaxNHits )
{
  // set all pointers to the tracks memory

  char *mem = fTrackMemory;

  // memory for tracklets

  AssignMemory( fTracklets, mem, MaxNTracks );

  // memory for selected tracks

  AssignMemory( fTracks, mem, MaxNTracks );

  // memory for output

  AlignTo < sizeof( void * ) > ( mem );
  fOutput = reinterpret_cast<AliHLTTPCCASliceOutput *>( mem );
  mem += AliHLTTPCCASliceOutput::EstimateSize( MaxNTracks, MaxNHits );

  // memory for output tracks

  AssignMemory( fOutTracks, mem, MaxNTracks );

  // calculate the size

  fTrackMemorySize = mem - fTrackMemory;
}

GPUh() int AliHLTTPCCATracker::CheckEmptySlice()
{
  if ( NHitsTotal() < 1 ) {
    {
      SetPointersTracks( 1, 1 ); // to calculate the size
      fTrackMemory = reinterpret_cast<char*> ( new uint4 [ fTrackMemorySize/sizeof( uint4 ) + 100] );
      SetPointersTracks( 1, 1 ); // set pointers for tracks
      fOutput->SetNTracks( 0 );
      fOutput->SetNTrackClusters( 0 );
    }

    return 1;
  }
  return 0;
}

void AliHLTTPCCATracker::RunNeighboursFinder()
{
	AliHLTTPCCAProcess<AliHLTTPCCANeighboursFinder>( Param().NRows(), 1, *this );
}

void AliHLTTPCCATracker::RunNeighboursCleaner()
{
	AliHLTTPCCAProcess<AliHLTTPCCANeighboursCleaner>( Param().NRows() - 2, 1, *this );
}

void AliHLTTPCCATracker::RunStartHitsFinder()
{
	AliHLTTPCCAProcess<AliHLTTPCCAStartHitsFinder>( Param().NRows() - 4, 1, *this );
}

void AliHLTTPCCATracker::RunTrackletConstructor()
{
  AliHLTTPCCAProcess1<AliHLTTPCCATrackletConstructor>( 1, TRACKLET_CONSTRUCTOR_NMEMTHREDS + *fNTracklets, *this );
}

void AliHLTTPCCATracker::RunTrackletSelector()
{
  AliHLTTPCCAProcess<AliHLTTPCCATrackletSelector>( 1, *fNTracklets, *this );
}

void AliHLTTPCCATracker::StandaloneQueryTime(unsigned long long int *i)
{
#ifdef HLTCA_STANDALONE
#ifdef R__WIN32
	  QueryPerformanceCounter((LARGE_INTEGER*) i);
#else
	  timespec t;
	  clock_gettime(CLOCK_REALTIME, &t);
	  *i = (unsigned long long int) t.tv_sec * (unsigned long long int) 1000000000 + (unsigned long long int) t.tv_nsec;
#endif
#endif
}

void AliHLTTPCCATracker::StandaloneQueryFreq(unsigned long long int *i)
{
#ifdef HLTCA_STANDALONE
#ifdef R__WIN32
	  QueryPerformanceFrequency((LARGE_INTEGER*) i);
#else
	*i = 1000000000;
#endif
#endif
}

void AliHLTTPCCATracker::StandalonePerfTime(int i)
{
#ifdef HLTCA_STANDALONE
  if (fGPUDebugLevel >= 2)
  {
	  StandaloneQueryTime(&fPerfTimers[i]);
  }
#endif
}

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
  //if( fParam.ISlice()==15){
  AliHLTTPCCADisplay::Instance().ClearView();
  AliHLTTPCCADisplay::Instance().SetSliceView();
  AliHLTTPCCADisplay::Instance().SetCurrentSlice( this );
  AliHLTTPCCADisplay::Instance().DrawSlice( this, 1 );
  if ( NHitsTotal() > 0 ) {
    AliHLTTPCCADisplay::Instance().DrawSliceHits( kRed, .5 );
    AliHLTTPCCADisplay::Instance().Ask();
  }
  //}
#endif

  *fNTracks = 0;
  *fNTracklets = 0;

#if !defined(HLTCA_GPUCODE)

  if (fGPUDebugLevel >= 3)
  {
	  *fGPUDebugOut << endl << endl << "Slice: " << Param().ISlice() << endl;
  }

  StandalonePerfTime(1);

  RunNeighboursFinder();

  StandalonePerfTime(2);

  if (fGPUDebugLevel >= 3)
  {
	  *fGPUDebugOut << "Neighbours Finder:" << endl;
	  DumpLinks(*fGPUDebugOut);
  }
#ifdef HLTCA_INTERNAL_PERFORMANCE
  //if( Param().ISlice()<=2 )
  //AliHLTTPCCAPerformance::Instance().LinkPerformance( Param().ISlice() );
#endif


#ifdef DRAW
  if ( NHitsTotal() > 0 ) {
    AliHLTTPCCADisplay::Instance().DrawSliceLinks( -1, -1, 1 );
    AliHLTTPCCADisplay::Instance().Ask();
  }
#endif

  RunNeighboursCleaner();

  StandalonePerfTime(3);

  if (fGPUDebugLevel >= 3)
  {
	  *fGPUDebugOut << "Neighbours Cleaner:" << endl;
	  DumpLinks(*fGPUDebugOut);
  }

  RunStartHitsFinder();

  StandalonePerfTime(4);

  if (fGPUDebugLevel >= 3)
  {
	  *fGPUDebugOut << "Start Hits: (" << *fNTracklets << ")" << endl;
	  DumpStartHits(*fGPUDebugOut);
  }
  
  if (fGPUDebugLevel >= 2) printf("%3d ", *fNTracklets);

  fData.ClearHitWeights();

  SetPointersTracks( *fNTracklets * 2, NHitsTotal() ); // to calculate the size
  fTrackMemory = reinterpret_cast<char*> ( new uint4 [ fTrackMemorySize/sizeof( uint4 ) + 100] );
  SetPointersTracks( *fNTracklets * 2, NHitsTotal() ); // set pointers for hits

  StandalonePerfTime(5);

  RunTrackletConstructor();

  StandalonePerfTime(6);

  if (fGPUDebugLevel >= 3)
  {
	  *fGPUDebugOut << "Tracklet Hits:" << endl;
	  DumpTrackletHits(*fGPUDebugOut);
  }

  //std::cout<<"Slice "<<Param().ISlice()<<": NHits="<<NHitsTotal()<<", NTracklets="<<*NTracklets()<<std::endl;

  RunTrackletSelector();

  StandalonePerfTime(7);

  //std::cout<<"Slice "<<Param().ISlice()<<": N start hits/tracklets/tracks = "<<nStartHits<<" "<<nStartHits<<" "<<*fNTracks<<std::endl;

  if (fGPUDebugLevel >= 3)
  {
	  *fGPUDebugOut << "Track Hits: (" << *NTracks() << ")" << endl;
	  DumpTrackHits(*fGPUDebugOut);
  }

  //std::cout<<"Memory used for slice "<<fParam.ISlice()<<" : "<<fCommonMemorySize/1024./1024.<<" + "<<fHitMemorySize/1024./1024.<<" + "<<fTrackMemorySize/1024./1024.<<" = "<<( fCommonMemorySize+fHitMemorySize+fTrackMemorySize )/1024./1024.<<" Mb "<<std::endl;

  WriteOutput();

  StandalonePerfTime(8);

#endif

#ifdef DRAW
  {
    AliHLTTPCCADisplay &disp = AliHLTTPCCADisplay::Instance();
    AliHLTTPCCATracker &slice = *this;
    std::cout << "N out tracks = " << *slice.NOutTracks() << std::endl;
    //disp.Ask();
    AliHLTTPCCADisplay::Instance().SetCurrentSlice( this );
    AliHLTTPCCADisplay::Instance().DrawSlice( this, 1 );
    disp.DrawSliceHits( -1, .5 );
    for ( int itr = 0; itr < *slice.NOutTracks(); itr++ ) {
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
#endif

  timer0.Stop();
  fTimers[0] = timer0.CpuTime() / 100.;

}

GPUh() void AliHLTTPCCATracker::WriteOutput()
{
  // write output

  TStopwatch timer;

  //cout<<"output: nTracks = "<<*fNTracks<<", nHitsTotal="<<NHitsTotal()<<std::endl;

  fOutput->SetNTracks( *fNTracks );
  fOutput->SetNTrackClusters( *fNTrackHits );
  fOutput->SetPointers();

  int nStoredHits = 0;

  for ( int iTr = 0; iTr < *fNTracks; iTr++ ) {
    AliHLTTPCCATrack &iTrack = fTracks[iTr];

    AliHLTTPCCASliceTrack out;
    out.SetFirstClusterRef( nStoredHits );
    out.SetNClusters( iTrack.NHits() );
    out.SetParam( iTrack.Param() );

    fOutput->SetTrack( iTr, out );

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

      unsigned short hPackedYZ = 0;
      UChar_t hPackedAmp = 0;
      float2 hUnpackedYZ;
      hUnpackedYZ.x = origY;
      hUnpackedYZ.y = origZ;
      float hUnpackedX = origX;

      fOutput->SetClusterId( nStoredHits, id  );
      fOutput->SetClusterRow( nStoredHits, ( unsigned char ) iRow  );
      fOutput->SetClusterPackedYZ( nStoredHits, hPackedYZ );
      fOutput->SetClusterPackedAmp( nStoredHits, hPackedAmp );
      fOutput->SetClusterUnpackedYZ( nStoredHits, hUnpackedYZ );
      fOutput->SetClusterUnpackedX( nStoredHits, hUnpackedX );
      nStoredHits++;
    }
  }


  // old stuff

  *fNOutTrackHits = 0;
  *fNOutTracks = 0;


  for ( int iTr = 0; iTr < *fNTracks; iTr++ ) {

    const AliHLTTPCCATrack &iTrack = fTracks[iTr];

    //std::cout<<"iTr = "<<iTr<<", nHits="<<iTrack.NHits()<<std::endl;

    //if( !iTrack.Alive() ) continue;
    if ( iTrack.NHits() < 3 ) continue;
    AliHLTTPCCAOutTrack &out = fOutTracks[*fNOutTracks];
    out.SetFirstHitRef( *fNOutTrackHits );
    out.SetNHits( 0 );
    out.SetOrigTrackID( iTr );
    out.SetStartPoint( iTrack.Param() );
    out.SetEndPoint( iTrack.Param() );

    int iID = iTrack.FirstHitID();
    int nOutTrackHitsOld = *fNOutTrackHits;

    for ( int ith = 0; ith < iTrack.NHits(); ith++ ) {
      const AliHLTTPCCAHitId &ic = fTrackHits[iID + ith];
      const AliHLTTPCCARow &row = Row( ic );
      int ih = ic.HitIndex();
      fOutTrackHits[*fNOutTrackHits] = HitInputID( row, ih );
      ( *fNOutTrackHits )++;
      //std::cout<<"write i,row,hit,id="<<ith<<", "<<ID2IRow(ic)<<", "<<ih<<", "<<HitInputID( row, ih )<<std::endl;
      if ( *fNOutTrackHits >= 10*NHitsTotal() ) {
        std::cout << "fNOutTrackHits>NHitsTotal()" << std::endl;
        //exit(0);
        return;//SG!!!
      }
      out.SetNHits( out.NHits() + 1 );
    }
    if ( out.NHits() >= 2 ) {
      ( *fNOutTracks )++;
    } else {
      ( *fNOutTrackHits ) = nOutTrackHitsOld;
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

  out << fTimers[0] << std::endl;
  out << *fNOutTrackHits << std::endl;
  for ( int ih = 0; ih < *fNOutTrackHits; ih++ ) {
    out << fOutTrackHits[ih] << " ";
  }
  out << std::endl;

  out << *fNOutTracks << std::endl;

  for ( int itr = 0; itr < *fNOutTracks; itr++ ) {
    AliHLTTPCCAOutTrack &t = fOutTracks[itr];
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
  in >> fTimers[0];
  in >> *fNOutTrackHits;

  for ( int ih = 0; ih < *fNOutTrackHits; ih++ ) {
    in >> fOutTrackHits[ih];
  }
  in >> *fNOutTracks;

  for ( int itr = 0; itr < *fNOutTracks; itr++ ) {
    AliHLTTPCCAOutTrack &t = fOutTracks[itr];
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
