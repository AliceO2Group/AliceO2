// $Id: AliHLTTPCGMMerger.cxx 30732 2009-01-22 23:02:02Z sgorbuno $
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
#include "AliHLTTPCCASliceOutTrack.h"
#include "AliHLTTPCCATracker.h"
#include "AliHLTTPCCATrackParam.h"
#include "AliHLTTPCGMCluster.h"
#include "AliHLTTPCGMPolynomialField.h"
#include "AliHLTTPCGMPolynomialFieldCreator.h"

#include "AliHLTTPCGMMerger.h"

#include "AliHLTTPCCAMath.h"

#include "AliHLTTPCCATrackParam.h"
#include "AliHLTTPCCASliceOutput.h"
#include "AliHLTTPCGMMergedTrack.h"
#include "AliHLTTPCCAParam.h"
#include "AliHLTTPCCATrackLinearisation.h"

#include "AliHLTTPCGMTrackParam.h"
#include "AliHLTTPCGMTrackLinearisation.h"
#include "AliHLTTPCGMSliceTrack.h"
#include "AliHLTTPCGMBorderTrack.h"
#include <cmath>

#include <algorithm>

#include "AliHLTTPCCAGPUConfig.h"
#include "MemoryAssignmentHelpers.h"

#define DEBUG 0

AliHLTTPCGMMerger::AliHLTTPCGMMerger()
  :
  fField(),
  fSliceParam(),
  fNOutputTracks( 0 ),
  fNOutputTrackClusters( 0 ),
  fOutputTracks( 0 ),
  fSliceTrackInfos( 0 ),
  fMaxSliceTracks(0),
  fClusters(NULL),
  fBorderMemory(0),
  fBorderRangeMemory(0),
  fGPUTracker(NULL),
  fDebugLevel(0),
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
  fPrevSliceInd[ 0 ] = mid;  fNextSliceInd[ last ] = fgkNSlices / 2;
  fPrevSliceInd[ fgkNSlices/2 ] = last;

  fField.Reset(); // set very wrong initial value in order to see if the field was not properly initialised    
  
  Clear();
}

AliHLTTPCGMMerger::AliHLTTPCGMMerger(const AliHLTTPCGMMerger&)
  :
  fField(),
  fSliceParam(),
  fNOutputTracks( 0 ),
  fNOutputTrackClusters( 0 ),
  fOutputTracks( 0 ),
  fSliceTrackInfos( 0 ),  
  fMaxSliceTracks(0),
  fClusters(NULL),
  fBorderMemory(0),
  fBorderRangeMemory(0),
  fGPUTracker(NULL),
  fDebugLevel(0),
  fNClusters(0)
{
  //* dummy
  for ( int iSlice = 0; iSlice < fgkNSlices; iSlice++ ) {
    fNextSliceInd[iSlice] = 0;
    fPrevSliceInd[iSlice] = 0;
  }
  fField.Reset();

  Clear();
}

const AliHLTTPCGMMerger &AliHLTTPCGMMerger::operator=(const AliHLTTPCGMMerger&) const
{
  //* dummy
  return *this;
}

AliHLTTPCGMMerger::~AliHLTTPCGMMerger()
{
  //* destructor
  ClearMemory();
}

void AliHLTTPCGMMerger::SetSliceParam( const AliHLTTPCCAParam &v )
{
  fSliceParam = v;
  if (fSliceParam.AssumeConstantBz()) AliHLTTPCGMPolynomialFieldCreator::GetPolynomialField( AliHLTTPCGMPolynomialFieldCreator::kUniform, v.BzkG(), fField );
  else AliHLTTPCGMPolynomialFieldCreator::GetPolynomialField( v.BzkG(), fField );
}

void AliHLTTPCGMMerger::Clear()
{
  for ( int i = 0; i < fgkNSlices; ++i ) {
    fkSlices[i] = 0;
    fSliceNTrackInfos[ i ] = 0;
    fSliceTrackInfoStart[ i ] = 0;
  }
  ClearMemory();
}
 
void AliHLTTPCGMMerger::ClearMemory()
{
  if (fSliceTrackInfos) delete[] fSliceTrackInfos;  
  if (!(fGPUTracker && fGPUTracker->IsInitialized()))
  {
	  if (fOutputTracks) delete[] fOutputTracks;
	  if (fClusters) delete[] fClusters;
  }
  if (fBorderMemory) delete[] fBorderMemory;
  if (fBorderRangeMemory) delete[] fBorderRangeMemory;

  fNOutputTracks = 0;
  fOutputTracks = 0;
  fSliceTrackInfos = 0;
  fMaxSliceTracks = 0;
  fClusters = NULL;
  fBorderMemory = 0;  
  fBorderRangeMemory = 0;
}

void AliHLTTPCGMMerger::SetSliceData( int index, const AliHLTTPCCASliceOutput *sliceData )
{
  fkSlices[index] = sliceData;
}

bool AliHLTTPCGMMerger::Reconstruct()
{
  //* main merging routine

  //fSliceParam.LoadClusterErrors();
  
  int nIter = 1;
#ifdef HLTCA_STANDALONE
  HighResTimer timer;
  static double times[5] = {};
  static int nCount = 0;
#endif
  //cout<<"Merger..."<<endl;
  for( int iter=0; iter<nIter; iter++ ){
    if( !AllocateMemory() ) return 0;
#ifdef HLTCA_STANDALONE
	timer.ResetStart();
#endif
    UnpackSlices();
#ifdef HLTCA_STANDALONE
	times[0] += timer.GetCurrentElapsedTime(true);
#endif
   MergeWithingSlices();
#ifdef HLTCA_STANDALONE
    times[1] += timer.GetCurrentElapsedTime(true);
#endif
    MergeSlices();
#ifdef HLTCA_STANDALONE
    times[2] += timer.GetCurrentElapsedTime(true);
#endif
    CollectMergedTracks();
#ifdef HLTCA_STANDALONE
    times[3] += timer.GetCurrentElapsedTime(true);
#endif
    Refit();
#ifdef HLTCA_STANDALONE
    times[4] += timer.GetCurrentElapsedTime();
    nCount++;
	if (fDebugLevel > 0)
	{
		printf("Merge Time:\tUnpack Slices:\t%1.0f us\n", times[0] * 1000000 / nCount);
		printf("\t\tMerge Within:\t%1.0f us\n", times[1] * 1000000 / nCount);
		printf("\t\tMerge Slices:\t%1.0f us\n", times[2] * 1000000 / nCount);
		printf("\t\tCollect:\t%1.0f us\n", times[3] * 1000000 / nCount);
		printf("\t\tRefit:\t\t%1.0f us\n", times[4] * 1000000 / nCount);
	}
    if (!HLTCA_TIMING_SUM)
    {
        for (int i = 0;i < 5;i++) times[i] = 0.;
        nCount = 0;
    }
    timer.Stop();  
#endif
  }  
  //cout<<"\nMerger time = "<<timer.CpuTime()*1.e3/nIter<<" ms\n"<<endl;

  return 1;
}

bool AliHLTTPCGMMerger::AllocateMemory()
{
  //* memory allocation
  
  ClearMemory();

  int nTracks = 0;
  fNClusters = 0;
  fMaxSliceTracks  = 0;
  
  for ( int iSlice = 0; iSlice < fgkNSlices; iSlice++ ) {
    if ( !fkSlices[iSlice] ) continue;
    nTracks += fkSlices[iSlice]->NTracks();
    fNClusters += fkSlices[iSlice]->NTrackClusters();
    if( fMaxSliceTracks < fkSlices[iSlice]->NTracks() ) fMaxSliceTracks = fkSlices[iSlice]->NTracks();
  }

  //cout<<"\nMerger: input "<<nTracks<<" tracks, "<<nClusters<<" clusters"<<endl;

  fSliceTrackInfos = new AliHLTTPCGMSliceTrack[nTracks];
  if (fGPUTracker && fGPUTracker->IsInitialized())
  {
	char* basemem = fGPUTracker->MergerHostMemory();
	AssignMemory(fClusters, basemem, fNClusters);
	AssignMemory(fOutputTracks, basemem, nTracks);
	if ((size_t) (basemem - fGPUTracker->MergerHostMemory()) > HLTCA_GPU_MERGER_MEMORY)
	{
		printf("Insufficient memory for track merger %lld > %lld\n", (long long int) (basemem - fGPUTracker->MergerHostMemory()), (long long int) HLTCA_GPU_MERGER_MEMORY);
		return(false);
	}
  }
  else
  {
	  fOutputTracks = new AliHLTTPCGMMergedTrack[nTracks];
	  fClusters = new AliHLTTPCGMMergedTrackHit[fNClusters];
  }
  fBorderMemory = new AliHLTTPCGMBorderTrack[fMaxSliceTracks*2];
  fBorderRangeMemory = new AliHLTTPCGMBorderTrack::Range[fMaxSliceTracks*2];
  
  return ( ( fOutputTracks!=NULL )
	   && ( fSliceTrackInfos!=NULL )
	   && ( fClusters!=NULL )
	   && ( fBorderMemory!=NULL )
	   && ( fBorderRangeMemory!=NULL )
	   );
}

void AliHLTTPCGMMerger::UnpackSlices()
{
  //* unpack the cluster information from the slice tracks and initialize track info array
  
  int nTracksCurrent = 0;

  const AliHLTTPCCASliceOutTrack* firstGlobalTracks[fgkNSlices];

  int maxSliceTracks = 0;
  for( int i=0; i<fgkNSlices; i++)
  {
      firstGlobalTracks[i] = 0;
      if (fkSlices[i] && fkSlices[i]->NLocalTracks() > maxSliceTracks) maxSliceTracks = fkSlices[i]->NLocalTracks();
  }

  int* TrackIds = new int[maxSliceTracks * fgkNSlices];
  for (int i = 0;i < maxSliceTracks * fgkNSlices;i++) TrackIds[i] = -1;

  for ( int iSlice = 0; iSlice < fgkNSlices; iSlice++ ) {

    fSliceTrackInfoStart[ iSlice ] = nTracksCurrent;

    fSliceNTrackInfos[ iSlice ] = 0;

    if ( !fkSlices[iSlice] ) continue;

    float alpha = fSliceParam.Alpha( iSlice );

    const AliHLTTPCCASliceOutput &slice = *( fkSlices[iSlice] );
    const AliHLTTPCCASliceOutTrack *sliceTr = slice.GetFirstTrack();    
    
    for ( int itr = 0; itr < slice.NLocalTracks(); itr++, sliceTr = sliceTr->GetNextTrack() ) {
      AliHLTTPCGMSliceTrack &track = fSliceTrackInfos[nTracksCurrent];
      track.Set( sliceTr, alpha, iSlice );
      if( !track.FilterErrors( fSliceParam, .999 ) ) continue;
      if (DEBUG) printf("Slice %d, Track %d, QPt %f DzDs %f\n", iSlice, itr, track.QPt(), track.DzDs());
      track.SetPrevNeighbour( -1 );
      track.SetNextNeighbour( -1 );
      track.SetSliceNeighbour( -1 );
      track.SetUsed( 0 );
      track.SetGlobalTrackId(0, -1);
      track.SetGlobalTrackId(1, -1);
      TrackIds[iSlice * maxSliceTracks + sliceTr->LocalTrackId()] = nTracksCurrent;
      nTracksCurrent++;
      fSliceNTrackInfos[ iSlice ]++;
    }
    firstGlobalTracks[iSlice] = sliceTr;
  }
  for (int iSlice = 0;iSlice < fgkNSlices;iSlice++)
  {
	  fSliceTrackGlobalInfoStart[iSlice] = nTracksCurrent;
	  fSliceNGlobalTrackInfos[iSlice] = 0;

	  if ( !fkSlices[iSlice] ) continue;
	  float alpha = fSliceParam.Alpha( iSlice );

	  const AliHLTTPCCASliceOutput &slice = *( fkSlices[iSlice] );
	  const AliHLTTPCCASliceOutTrack *sliceTr = firstGlobalTracks[iSlice];
	  for (int itr = slice.NLocalTracks();itr < slice.NTracks();itr++, sliceTr = sliceTr->GetNextTrack())
	  {
		  int localId = TrackIds[(sliceTr->LocalTrackId() >> 24) * maxSliceTracks + (sliceTr->LocalTrackId() & 0xFFFFFF)];
		  if (localId == -1) continue;
		  AliHLTTPCGMSliceTrack &track = fSliceTrackInfos[nTracksCurrent];
		  track.Set( sliceTr, alpha, iSlice );
		  track.SetGlobalSectorTrackCov();
		  track.SetPrevNeighbour( -1 );
		  track.SetNextNeighbour( -1 );
		  track.SetSliceNeighbour( -1 );
		  track.SetUsed( 0 );           
		  track.SetLocalTrackId(localId);
		  nTracksCurrent++;
		  fSliceNGlobalTrackInfos[ iSlice ]++;
	  }
  }

  delete[] TrackIds;
}

void AliHLTTPCGMMerger::MakeBorderTracks( int iSlice, int iBorder, AliHLTTPCGMBorderTrack B[], int &nB )
{
  //* prepare slice tracks for merging with next/previous/same sector
  //* each track transported to the border line

  float fieldBz = fSliceParam.ConstBz();
  
  nB = 0;
  
  float dAlpha = fSliceParam.DAlpha() / 2;
  float x0 = 0;

  if ( iBorder == 0 ) { // transport to the left age of the sector and rotate horisontally
    dAlpha = dAlpha - CAMath::Pi() / 2 ;
  } else if ( iBorder == 1 ) { //  transport to the right age of the sector and rotate horisontally
    dAlpha = -dAlpha - CAMath::Pi() / 2 ;
  } else if ( iBorder == 2 ) { // transport to the left age of the sector and rotate vertically
    //dAlpha = dAlpha; //causes compiler warning
    x0 = fSliceParam.RowX( 63 );
  } else if ( iBorder == 3 ) { // transport to the right age of the sector and rotate vertically
    dAlpha = -dAlpha;
    x0 =  fSliceParam.RowX( 63 );
  } else if ( iBorder == 4 ) { // transport to the middle of the sector, w/o rotation
    dAlpha = 0;
    x0 = fSliceParam.RowX( 63 );
  }

  const float maxSin = CAMath::Sin( 60. / 180.*CAMath::Pi() );
  float cosAlpha = AliHLTTPCCAMath::Cos( dAlpha );
  float sinAlpha = AliHLTTPCCAMath::Sin( dAlpha );

  for ( int itr = 0; itr < fSliceNTrackInfos[iSlice]; itr++ ) {

    AliHLTTPCGMSliceTrack &track = fSliceTrackInfos[ fSliceTrackInfoStart[iSlice] + itr ];

    if( track.Used() ) continue;
    AliHLTTPCGMBorderTrack &b = B[nB];

    if(  track.TransportToXAlpha( x0, sinAlpha, cosAlpha, fieldBz, b, maxSin)){
      b.SetTrackID( itr );
      b.SetNClusters( track.NClusters() );
	  for (int i = 0;i < 4;i++) if (fabs(b.Cov()[i]) >= 5.0) b.SetCov(i, 5.0);
	  if (fabs(b.Cov()[4]) >= 0.5) b.SetCov(4, 0.5);
      nB++; 
    }
  }
}


void AliHLTTPCGMMerger::MergeBorderTracks ( int iSlice1, AliHLTTPCGMBorderTrack B1[],  int N1,
					    int iSlice2, AliHLTTPCGMBorderTrack B2[],  int N2 )
{
  //* merge two sets of tracks
  if (N1 == 0 || N2 == 0) return;
  
  if (DEBUG) printf("\nMERGING Slices %d %d NTracks %d %d\n", iSlice1, iSlice2, N1, N2);
  int statAll=0, statMerged=0;
  float factor2ys = 1.5;//1.5;//SG!!!
  float factor2zt = 1.5;//1.5;//SG!!!
  float factor2k = 2.0;//2.2;

  factor2k  = 3.5 * 3.5 * factor2k * factor2k;
  factor2ys = 3.5 * 3.5 * factor2ys * factor2ys;
  factor2zt = 3.5 * 3.5 * factor2zt * factor2zt;
 
  int minNPartHits = 10;//SG!!!
  int minNTotalHits = 20;

  AliHLTTPCGMBorderTrack::Range *range1 = fBorderRangeMemory;
  AliHLTTPCGMBorderTrack::Range *range2 = fBorderRangeMemory + N1;

  bool sameSlice = (iSlice1 == iSlice2);
  {
    for ( int itr = 0; itr < N1; itr++ ){
      AliHLTTPCGMBorderTrack &b = B1[itr];
      float d = AliHLTTPCCAMath::Max(0.5f, 3.5*sqrt(b.Cov()[1]));
      if (fabs(b.Par()[4]) >= 20) d *= 2;
      else if (d > 3) d = 3;
      if (DEBUG) {printf("  Input Slice 1 %d Track %d: ", iSlice1, itr); for (int i = 0;i < 5;i++) {printf("%8.3f ", b.Par()[i]);} printf(" - "); for (int i = 0;i < 5;i++) {printf("%8.3f ", b.Cov()[i]);} printf(" - D %8.3f\n", d);}
      range1[itr].fId = itr;
      range1[itr].fMin = b.Par()[1] + b.ZOffset() - d;
      range1[itr].fMax = b.Par()[1] + b.ZOffset() + d;
    }
    std::sort(range1,range1+N1,AliHLTTPCGMBorderTrack::Range::CompMin);
    if( sameSlice ){
      for(int i=0; i<N1; i++) range2[i]= range1[i];
      std::sort(range2,range2+N1,AliHLTTPCGMBorderTrack::Range::CompMax);
      N2 = N1;
      B2 = B1;
    }else{
      for ( int itr = 0; itr < N2; itr++ ){
        AliHLTTPCGMBorderTrack &b = B2[itr];
        float d = AliHLTTPCCAMath::Max(0.5f, 3.5*sqrt(b.Cov()[1]));
        if (fabs(b.Par()[4]) >= 20) d *= 2;
        else if (d > 3) d = 3;
        if (DEBUG) {printf("  Input Slice 2 %d Track %d: ", iSlice2, itr);for (int i = 0;i < 5;i++) {printf("%8.3f ", b.Par()[i]);}printf(" - ");for (int i = 0;i < 5;i++) {printf("%8.3f ", b.Cov()[i]);}printf(" - D %8.3f\n", d);}
        range2[itr].fId = itr;
        range2[itr].fMin = b.Par()[1] + b.ZOffset() - d;
        range2[itr].fMax = b.Par()[1] + b.ZOffset() + d;
      }        
      std::sort(range2,range2+N2,AliHLTTPCGMBorderTrack::Range::CompMax);
    }
  }

  int i2 = 0;
  for ( int i1 = 0; i1 < N1; i1++ ) {

    AliHLTTPCGMBorderTrack::Range r1 = range1[i1];
    while( i2<N2 && range2[i2].fMax< r1.fMin ) i2++;
 
    AliHLTTPCGMBorderTrack &b1 = B1[r1.fId];   
    if ( b1.NClusters() < minNPartHits ) continue;
    int iBest2 = -1;
    int lBest2 = 0;
    statAll++;
    for( int k2 = i2; k2<N2; k2++){
      
      AliHLTTPCGMBorderTrack::Range r2 = range2[k2];
      if( r2.fMin > r1.fMax ) break;
      if( sameSlice && (r1.fId >= r2.fId) ) continue;
      // do check
      AliHLTTPCGMBorderTrack &b2 = B2[r2.fId];
      if (DEBUG) {printf("Comparing track %3d to %3d: ", r1.fId, r2.fId);for (int i = 0;i < 5;i++) {printf("%8.3f ", b1.Par()[i]);}printf(" - ");for (int i = 0;i < 5;i++) {printf("%8.3f ", b1.Cov()[i]);}printf("\n%28s", "");
        for (int i = 0;i < 5;i++) {printf("%8.3f ", b2.Par()[i]);}printf(" - ");for (int i = 0;i < 5;i++) {printf("%8.3f ", b2.Cov()[i]);}printf("   -   ");}

      if ( b2.NClusters() < lBest2 ) {if (DEBUG) {printf("!NCl1\n");}continue;}
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

      lBest2 = b2.NClusters();
      iBest2 = b2.TrackID();
    }

    if ( iBest2 < 0 ) continue;
    statMerged++;
    AliHLTTPCGMSliceTrack &newTrack1 = fSliceTrackInfos[fSliceTrackInfoStart[iSlice1] + b1.TrackID() ];
    AliHLTTPCGMSliceTrack &newTrack2 = fSliceTrackInfos[fSliceTrackInfoStart[iSlice2] + iBest2 ];

    int old1 = newTrack2.PrevNeighbour();

    if ( old1 >= 0 ) {
      AliHLTTPCGMSliceTrack &oldTrack1 = fSliceTrackInfos[fSliceTrackInfoStart[iSlice1] + old1];
      if ( oldTrack1.NClusters()  < newTrack1.NClusters() ) {
        newTrack2.SetPrevNeighbour( -1 );
        oldTrack1.SetNextNeighbour( -1 );
      } else continue;
    }
    int old2 = newTrack1.NextNeighbour();
    if ( old2 >= 0 ) {
      AliHLTTPCGMSliceTrack &oldTrack2 = fSliceTrackInfos[fSliceTrackInfoStart[iSlice2] + old2];
      if ( oldTrack2.NClusters() < newTrack2.NClusters() ) {
        oldTrack2.SetPrevNeighbour( -1 );
      } else continue;
    }
    newTrack1.SetNextNeighbour( iBest2 );
    newTrack2.SetPrevNeighbour( b1.TrackID() );
  }
  //cout<<"slices "<<iSlice1<<","<<iSlice2<<": all "<<statAll<<" merged "<<statMerged<<endl;
}


void AliHLTTPCGMMerger::MergeWithingSlices()
{
  //* merge track segments withing one slice

  float x0 = fSliceParam.RowX( 63 );  
  const float maxSin = CAMath::Sin( 60. / 180.*CAMath::Pi() );

  for ( int iSlice = 0; iSlice < fgkNSlices; iSlice++ ) {

    int nBord = 0;
    for ( int itr = 0; itr < fSliceNTrackInfos[iSlice]; itr++ ) {
      AliHLTTPCGMSliceTrack &track = fSliceTrackInfos[ fSliceTrackInfoStart[iSlice] + itr ];
      //track.SetPrevNeighbour( -1 );      
      //track.SetNextNeighbour( -1 );
      //track.SetSliceNeighbour( -1 );
      //track.SetUsed(0);
      
      AliHLTTPCGMBorderTrack &b = fBorderMemory[nBord];
      if( track.TransportToX( x0, fSliceParam.ConstBz(), b, maxSin) ){
	b.SetTrackID( itr );
        if (DEBUG) {printf("WITHIN SLICE %d Track %d - ", iSlice, itr);for (int i = 0;i < 5;i++) {printf("%8.3f ", b.Par()[i]);} printf(" - ");for (int i = 0;i < 5;i++) {printf("%8.3f ", b.Cov()[i]);} printf("\n");}
	b.SetNClusters( track.NClusters() );
	nBord++;
      }
    }  

    MergeBorderTracks( iSlice, fBorderMemory, nBord, iSlice, fBorderMemory, nBord );
    
    for ( int itr = 0; itr < fSliceNTrackInfos[iSlice]; itr++ ) {
      AliHLTTPCGMSliceTrack &track = fSliceTrackInfos[ fSliceTrackInfoStart[iSlice] + itr];
      if( track.PrevNeighbour()>=0 || track.Used() ) continue;
      int jtr = track.NextNeighbour();
      track.SetSliceNeighbour( jtr );
      track.SetNextNeighbour(-1);
      while( jtr>=0 ){
	AliHLTTPCGMSliceTrack &trackN = fSliceTrackInfos[ fSliceTrackInfoStart[iSlice] + jtr];
 	if( trackN.NClusters()>track.NClusters() ) track.CopyParamFrom(trackN);
	trackN.SetUsed(2);
	jtr = trackN.NextNeighbour();
	trackN.SetSliceNeighbour( jtr );
	trackN.SetNextNeighbour(-1);
	trackN.SetPrevNeighbour(-1);
      }
    }
  }
}

void AliHLTTPCGMMerger::MergeSlices()
{
  //* track merging between slices

  //for ( int iSlice = 0; iSlice < fgkNSlices; iSlice++ ) {
  //for ( int itr = 0; itr < fSliceNTrackInfos[iSlice]; itr++ ) {
  //AliHLTTPCGMSliceTrack &track = fSliceTrackInfos[ fSliceTrackInfoStart[iSlice] + itr ];
  //track.SetPrevNeighbour( -1 );
  //track.SetNextNeighbour( -1 );
  //}
  //}

  AliHLTTPCGMBorderTrack 
    *bCurr = fBorderMemory,
    *bNext = fBorderMemory + fMaxSliceTracks;
  
  for ( int iSlice = 0; iSlice < fgkNSlices; iSlice++ ) {    
    int jSlice = fNextSliceInd[iSlice];    
    int nCurr = 0, nNext = 0;
    MakeBorderTracks( iSlice, 2, bCurr, nCurr );
    MakeBorderTracks( jSlice, 3, bNext, nNext );
    MergeBorderTracks( iSlice, bCurr, nCurr, jSlice, bNext, nNext );
    MakeBorderTracks( iSlice, 0, bCurr, nCurr );
    MakeBorderTracks( jSlice, 1, bNext, nNext );
    MergeBorderTracks( iSlice, bCurr, nCurr, jSlice, bNext, nNext );
  }
}

struct clcomparestruct {int i; float x; float z; float q;};

struct AliHLTTPCGMMerger_CompareClusterIds
{
	float fQPt, fDzDs, fThresh;
	AliHLTTPCGMMerger_CompareClusterIds(float q, float z) : fQPt(q), fDzDs(z), fThresh(fabs(0.1f * 3.14f * 666.f * z / q)) {if (fThresh < 1.) fThresh = 1.; if (fThresh > 4.) fThresh = 4.;}
	bool operator()(const clcomparestruct& a, const clcomparestruct& b) { //a < b ?
		float dz = a.z - b.z;
		if (a.q * b.q < 0) return(dz * fDzDs > 0);
		if (fabs(dz) > fThresh) return(dz * fDzDs > 0);
		return((a.x - b.x) * a.q * fQPt > 0);
	}
};

void AliHLTTPCGMMerger::CollectMergedTracks()
{
  //* 

  //for ( int iSlice = 0; iSlice < fgkNSlices; iSlice++ ) {
  //for ( int itr = 0; itr < fSliceNTrackInfos[iSlice]; itr++ ) {
  //AliHLTTPCGMSliceTrack &track = fSliceTrackInfos[ fSliceTrackInfoStart[iSlice] + itr ];
  //if( track.Used()!=2 ) track.SetUsed(0);
  //}
  //}

  //Resolve connections for global tracks first
  for (int iSlice = 0;iSlice < fgkNSlices;iSlice++)
  {
    for (int itr = 0;itr < fSliceNGlobalTrackInfos[iSlice];itr++)
	{
		AliHLTTPCGMSliceTrack &globalTrack = fSliceTrackInfos[fSliceTrackGlobalInfoStart[iSlice] + itr];
		AliHLTTPCGMSliceTrack &localTrack = fSliceTrackInfos[globalTrack.LocalTrackId()];
		localTrack.SetGlobalTrackId(localTrack.GlobalTrackId(0) != -1, fSliceTrackGlobalInfoStart[iSlice] + itr);
	}
  }

  //Now collect the merged tracks
  fNOutputTracks = 0;
  int nOutTrackClusters = 0;
  const int kMaxParts = 400;
  const int kMaxClusters = 1000;

  const AliHLTTPCGMSliceTrack *trackParts[kMaxParts];

  for ( int iSlice = 0; iSlice < fgkNSlices; iSlice++ ) {

    for ( int itr = 0; itr < fSliceNTrackInfos[iSlice]; itr++ ) {

      AliHLTTPCGMSliceTrack &track = fSliceTrackInfos[fSliceTrackInfoStart[iSlice] + itr];

      if ( track.Used() ) continue;
      if ( track.PrevNeighbour() >= 0 ) continue;
      int nParts = 0;
      int jSlice = iSlice;
      AliHLTTPCGMSliceTrack *trbase = &track, *tr = &track;
      tr->SetUsed( 1 );
      do{
	    if( nParts >= kMaxParts ) break;
	    trackParts[nParts++] = tr;
	    for (int i = 0;i < 2;i++) if (tr->GlobalTrackId(i) != -1) trackParts[nParts++] = &fSliceTrackInfos[tr->GlobalTrackId(i)];

	    int jtr = tr->SliceNeighbour();
	    if( jtr >= 0 ) {
	      tr = &(fSliceTrackInfos[fSliceTrackInfoStart[jSlice] + jtr]);
	      tr->SetUsed( 2 );
	      continue;
	    }
	    jtr = trbase->NextNeighbour();
	    if( jtr>=0 ){
	      jSlice = fNextSliceInd[jSlice];
	      trbase = &(fSliceTrackInfos[fSliceTrackInfoStart[jSlice] + jtr]);
	      tr = trbase;
	      if( tr->Used() ) break;
	      tr->SetUsed( 1 );
	      continue;	  
	    }
	    break;
      }while(1);
	  
      // unpack and sort clusters
      
      AliHLTTPCCASliceOutCluster trackClusters[kMaxClusters];
      uchar2 clA[kMaxClusters];
      int nHits = 0;
      for( int ipart=0; ipart<nParts; ipart++ ){
	const AliHLTTPCGMSliceTrack *t = trackParts[ipart];
        if (DEBUG) printf("Collect Track %d Part %d QPt %f DzDs %f\n", fNOutputTracks, ipart, t->QPt(), t->DzDs());
	int nTrackHits = t->NClusters();
	if( nHits + nTrackHits > kMaxClusters ) break;
	const AliHLTTPCCASliceOutCluster *c= t->OrigTrack()->Clusters();
	AliHLTTPCCASliceOutCluster *c2 = trackClusters + nHits + nTrackHits-1;
	for( int i=0; i<nTrackHits; i++, c++, c2-- )
	{
		*c2 = *c;
		clA[nHits].x = t->Slice();
		clA[nHits++].y = (t->QPt() > 0);
	}
      }
      if ( nHits < TRACKLET_SELECTOR_MIN_HITS(track.QPt()) ) continue;

	int ordered = 1;
	for( int i=1; i<nHits; i++ )
	{
	  if ( trackClusters[i].GetX() > trackClusters[i-1].GetX() )
	  {
	    ordered = 0;
	    break;
	  }
	}

	int firstTrackIndex = 0;
	if (ordered == 0)
	{
	  int nTmpHits = 0;
	  
	  //Find QPt and DzDs for the segment closest to the vertex, if low/mid Pt
	  float baseQPt = trackParts[0]->QPt() > 0 ? 1.f : -1.0;
	  float baseZ = trackParts[0]->DzDs() > 0 ? 1.0 : -1.0;
	  if (fabs(trackParts[0]->QPt()) > 2)
	  {
		  float minZ = 1000.f;
		  for (int i = 0;i < nParts;i++)
		  {
			if (fabs(trackParts[i]->Z()) < minZ)
			{
				  baseQPt = trackParts[i]->QPt();
				  baseZ = trackParts[i]->DzDs();
				  minZ = fabs(trackParts[i]->Z());
			}
		  }
	  }

	  AliHLTTPCCASliceOutCluster trackClustersUnsorted[kMaxClusters];
	  uchar2 clAUnsorted[kMaxClusters];
	  clcomparestruct clusterIndices[kMaxClusters];
	  for (int iPart = 0;iPart < nParts;iPart++)
	  {
		const AliHLTTPCGMSliceTrack *t = trackParts[iPart];
		int nTrackHits = t->NClusters();
		if (nTmpHits + nTrackHits > kMaxClusters) break;
		for (int j = 0;j < nTrackHits;j++)
		{
			int i = nTmpHits + j;
			trackClustersUnsorted[i] = trackClusters[i];
			clAUnsorted[i] = clA[i];
			clusterIndices[i].i = i;
			clusterIndices[i].x = trackClusters[i].GetX();
			clusterIndices[i].z = trackClusters[i].GetZ();
			clusterIndices[i].q = t->QPt();
		}
		nTmpHits += nTrackHits;
	  }
	  
	  std::sort(clusterIndices, clusterIndices + nHits, AliHLTTPCGMMerger_CompareClusterIds(baseQPt, baseZ));
	  nTmpHits = 0;
	  for (int i = 0;i < nParts;i++)
	  {
		  nTmpHits += trackParts[i]->NClusters();
		  if (nTmpHits > clusterIndices[0].i)
		  {
			firstTrackIndex = i;
			break;
		  }
	  }

	  int nFilteredHits = 0;
	  int indPrev = -1;
	  for (int i = 0;i < nHits;i++)
	    {
	      int ind = clusterIndices[i].i;
	      if(indPrev >= 0 && trackClusters[ind].GetId() == trackClusters[indPrev].GetId()) continue;
	      indPrev = ind;
	      trackClusters[nFilteredHits] = trackClustersUnsorted[ind];
	      clA[nFilteredHits] = clAUnsorted[ind];
	      nFilteredHits++;
	    }
	  nHits = nFilteredHits;
	}
      
      AliHLTTPCGMMergedTrackHit *cl = fClusters + nOutTrackClusters;
      for( int i=0; i<nHits; i++ )
      {
          cl[i].fX = trackClusters[i].GetX();
          cl[i].fY = trackClusters[i].GetY();
          cl[i].fZ = trackClusters[i].GetZ();
          cl[i].fRow = trackClusters[i].GetRow();
          cl[i].fId = trackClusters[i].GetId();
          cl[i].fState = 0;
          cl[i].fSlice = clA[i].x;
          cl[i].fLeg = clA[i].y;
      }

      AliHLTTPCGMMergedTrack &mergedTrack = fOutputTracks[fNOutputTracks];
      mergedTrack.SetOK(1);
      mergedTrack.SetNClusters( nHits );
      mergedTrack.SetFirstClusterRef( nOutTrackClusters );
      AliHLTTPCGMTrackParam &p1 = mergedTrack.Param();
      const AliHLTTPCGMSliceTrack &p2 = *trackParts[firstTrackIndex];
      mergedTrack.SetSide(p2.Slice() < 18);
	  
	  AliHLTTPCGMBorderTrack b;
	  if (p2.TransportToX(cl[0].fX, fSliceParam.ConstBz(), b, 0.999, false))
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

      fNOutputTracks++;
      nOutTrackClusters += nHits;
    }
  }
  fNOutputTrackClusters = nOutTrackClusters;
}

void AliHLTTPCGMMerger::Refit()
{
  //* final refit
#ifdef HLTCA_GPU_MERGER
	if (fGPUTracker && fGPUTracker->IsInitialized())
	{
		fGPUTracker->RefitMergedTracks(this);
	}
	else
#endif
	{
#ifdef HLTCA_STANDALONE
#pragma omp parallel for
#endif
	  for ( int itr = 0; itr < fNOutputTracks; itr++ )
	  {
	    AliHLTTPCGMTrackParam::RefitTrack(fOutputTracks[itr], &fField, fClusters, fSliceParam);
	  }
	}
}
