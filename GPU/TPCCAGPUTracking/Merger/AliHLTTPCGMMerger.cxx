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
#include <string.h>
#include "AliHLTTPCCASliceOutTrack.h"
#include "AliHLTTPCCATracker.h"
#include "AliHLTTPCCAClusterData.h"
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
#include "AliHLTTPCGMSliceTrack.h"
#include "AliHLTTPCGMBorderTrack.h"
#include <cmath>

#include <algorithm>

#include "AliHLTTPCCAGPUConfig.h"
#include "MemoryAssignmentHelpers.h"

#if defined(BUILD_QA) && defined(HLTCA_STANDALONE) && !defined(HLTCA_GPUCODE)
#include "include.h"
#endif

#define DEBUG 0

//#define OFFLINE_FITTER

#if ( defined(HLTCA_STANDALONE) || defined(HLTCA_GPUCODE) )
#undef OFFLINE_FITTER
#endif

#if ( defined(OFFLINE_FITTER) )
#include "AliHLTTPCGMOfflineFitter.h"
AliHLTTPCGMOfflineFitter gOfflineFitter;
#endif

AliHLTTPCGMMerger::AliHLTTPCGMMerger()
  :
  fField(),
  fSliceParam(),
  fNOutputTracks( 0 ),
  fNOutputTrackClusters( 0 ),
  fNMaxOutputTrackClusters( 0 ),
  fOutputTracks( 0 ),
  fSliceTrackInfoMemory( 0 ),
  fMaxSliceTracks(0),
  fClusters(NULL),
  fGlobalClusterIDs(NULL),
  fClusterAttachment(NULL),
  fMaxID(0),
  fTrackOrder(NULL),
  fBorderMemory(0),
  fBorderRangeMemory(0),
  fGPUTracker(NULL),
  fSliceTrackers(NULL),
  fDebugLevel(0),
  fNClusters(0)
{
  //* constructor
  
  for ( int iSlice = 0; iSlice < fgkNSlices; iSlice++ ) {
    fNextSliceInd[iSlice] = iSlice + 1;
    fPrevSliceInd[iSlice] = iSlice - 1;
    fSliceTrackInfo[iSlice] = NULL;
    fSliceTrackGlobalInfo[iSlice] = NULL;
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
  fNMaxOutputTrackClusters( 0 ),
  fOutputTracks( 0 ),
  fSliceTrackInfoMemory( 0 ),
  fMaxSliceTracks(0),
  fClusters(NULL),
  fGlobalClusterIDs(NULL),
  fClusterAttachment(NULL),
  fMaxID(0),
  fTrackOrder(NULL),
  fBorderMemory(0),
  fBorderRangeMemory(0),
  fGPUTracker(NULL),
  fSliceTrackers(NULL),
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

void AliHLTTPCGMMerger::SetSliceParam( const AliHLTTPCCAParam &v, long int TimeStamp, bool isMC  )
{
  fSliceParam = v;
  if (fSliceParam.AssumeConstantBz()) AliHLTTPCGMPolynomialFieldCreator::GetPolynomialField( AliHLTTPCGMPolynomialFieldCreator::kUniform, v.BzkG(), fField );
  else AliHLTTPCGMPolynomialFieldCreator::GetPolynomialField( v.BzkG(), fField );

#if ( defined(OFFLINE_FITTER) )
  gOfflineFitter.Initialize(  fSliceParam, TimeStamp, isMC );
#else
  (void) (TimeStamp + isMC); //Suppress warning
#endif
}

void AliHLTTPCGMMerger::Clear()
{
  for ( int i = 0; i < fgkNSlices; ++i ) {
    fkSlices[i] = 0;
    fSliceNTrackInfos[ i ] = 0;
  }
  ClearMemory();
}
 
void AliHLTTPCGMMerger::ClearMemory()
{
  if (fSliceTrackInfoMemory) delete[] fSliceTrackInfoMemory;
  if (!(fGPUTracker && fGPUTracker->IsInitialized()))
  {
    if (fOutputTracks) delete[] fOutputTracks;
    if (fClusters) delete[] fClusters;
  }
  if (fGlobalClusterIDs) delete[] fGlobalClusterIDs;
  if (fBorderMemory) delete[] fBorderMemory;
  if (fBorderRangeMemory) delete[] fBorderRangeMemory;
  
  if (fTrackOrder) delete[] fTrackOrder;
  if (fClusterAttachment) delete[] fClusterAttachment;

  fNOutputTracks = 0;
  fOutputTracks = NULL;
  fSliceTrackInfoMemory = NULL;
  fMaxSliceTracks = 0;
  fClusters = NULL;
  fGlobalClusterIDs = NULL;
  fBorderMemory = NULL;
  fBorderRangeMemory = NULL;
  fTrackOrder = NULL;
  fClusterAttachment = NULL;
}

void AliHLTTPCGMMerger::SetSliceData( int index, const AliHLTTPCCASliceOutput *sliceData )
{
  fkSlices[index] = sliceData;
}

bool AliHLTTPCGMMerger::Reconstruct(bool resetTimers)
{
  //* main merging routine
  int nIter = 1;
#ifdef HLTCA_STANDALONE
  HighResTimer timer;
  static double times[8] = {};
  static int nCount = 0;
  if (resetTimers || !HLTCA_TIMING_SUM)
  {
    for (unsigned int k = 0;k < sizeof(times) / sizeof(times[0]);k++) times[k] = 0;
    nCount = 0;
  }
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
    MergeCEInit();
#ifdef HLTCA_STANDALONE
    times[3] += timer.GetCurrentElapsedTime(true);
#endif
    CollectMergedTracks();
#ifdef HLTCA_STANDALONE
    times[4] += timer.GetCurrentElapsedTime(true);
#endif
    MergeCE();
#ifdef HLTCA_STANDALONE
    times[3] += timer.GetCurrentElapsedTime(true);
#endif
    PrepareClustersForFit();
#ifdef HLTCA_STANDALONE
    times[5] += timer.GetCurrentElapsedTime(true);
#endif
    Refit(resetTimers);
#ifdef HLTCA_STANDALONE
    times[6] += timer.GetCurrentElapsedTime(true);
    Finalize();
    times[7] += timer.GetCurrentElapsedTime(true);
    nCount++;
    if (fDebugLevel > 0)
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
#endif
  }  
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
  fNMaxOutputTrackClusters = fNClusters * 1.1f + 1000;

  //cout<<"\nMerger: input "<<nTracks<<" tracks, "<<nClusters<<" clusters"<<endl;

  fSliceTrackInfoMemory = new AliHLTTPCGMSliceTrack[nTracks];
  if (fGPUTracker && fGPUTracker->IsInitialized())
  {
    char* basemem = fGPUTracker->MergerHostMemory();
    AssignMemory(fClusters, basemem, fNMaxOutputTrackClusters);
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
    fClusters = new AliHLTTPCGMMergedTrackHit[fNMaxOutputTrackClusters];
  }
  if (!fSliceTrackers) fGlobalClusterIDs = new int[fNMaxOutputTrackClusters];
  fBorderMemory = new AliHLTTPCGMBorderTrack[nTracks];
  fBorderRangeMemory = new AliHLTTPCGMBorderTrack::Range[2 * nTracks];
  nTracks = 0;
  for ( int iSlice = 0; iSlice < fgkNSlices; iSlice++ ) {
      fBorder[iSlice] = fBorderMemory + nTracks;
      fBorderRange[iSlice] = fBorderRangeMemory + 2 * nTracks;
    if ( !fkSlices[iSlice] ) continue;
    nTracks += fkSlices[iSlice]->NTracks();
  }  
  return ( ( fOutputTracks!=NULL )
    && ( fSliceTrackInfoMemory!=NULL )
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

    fSliceTrackInfo[ iSlice ] = fSliceTrackInfoMemory + nTracksCurrent;;

    fSliceNTrackInfos[ iSlice ] = 0;

    if ( !fkSlices[iSlice] ) continue;

    float alpha = fSliceParam.Alpha( iSlice );

    const AliHLTTPCCASliceOutput &slice = *( fkSlices[iSlice] );
    const AliHLTTPCCASliceOutTrack *sliceTr = slice.GetFirstTrack();    
    
    for ( int itr = 0; itr < slice.NLocalTracks(); itr++, sliceTr = sliceTr->GetNextTrack() ) {
      AliHLTTPCGMSliceTrack &track = fSliceTrackInfoMemory[nTracksCurrent];
      track.Set( sliceTr, alpha, iSlice );
      if( !track.FilterErrors( fSliceParam, HLTCA_MAX_SIN_PHI, 0.1f ) ) continue;
      if (DEBUG) printf("INPUT Slice %d, Track %d, QPt %f DzDs %f\n", iSlice, itr, track.QPt(), track.DzDs());
      track.SetPrevNeighbour( -1 );
      track.SetNextNeighbour( -1 );
      track.SetPrevNeighbour( -1, 1 );
      track.SetNextNeighbour( -1, 1 );
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
    fSliceTrackGlobalInfo[iSlice] = fSliceTrackInfoMemory + nTracksCurrent;
    fSliceNGlobalTrackInfos[iSlice] = 0;

    if ( !fkSlices[iSlice] ) continue;
    float alpha = fSliceParam.Alpha( iSlice );

    const AliHLTTPCCASliceOutput &slice = *( fkSlices[iSlice] );
    const AliHLTTPCCASliceOutTrack *sliceTr = firstGlobalTracks[iSlice];
    for (int itr = slice.NLocalTracks();itr < slice.NTracks();itr++, sliceTr = sliceTr->GetNextTrack())
    {
      int localId = TrackIds[(sliceTr->LocalTrackId() >> 24) * maxSliceTracks + (sliceTr->LocalTrackId() & 0xFFFFFF)];
      if (localId == -1) continue;
      AliHLTTPCGMSliceTrack &track = fSliceTrackInfoMemory[nTracksCurrent];
      track.Set( sliceTr, alpha, iSlice );
      track.SetGlobalSectorTrackCov();
      track.SetPrevNeighbour( -1 );
      track.SetNextNeighbour( -1 );
      track.SetPrevNeighbour( -1, 1 );
      track.SetNextNeighbour( -1, 1 );
      track.SetSliceNeighbour( -1 );
      track.SetUsed( 0 );           
      track.SetLocalTrackId(localId);
      nTracksCurrent++;
      fSliceNGlobalTrackInfos[ iSlice ]++;
    }
  }

  delete[] TrackIds;
}

void AliHLTTPCGMMerger::MakeBorderTracks( int iSlice, int iBorder, AliHLTTPCGMBorderTrack B[], int &nB, bool fromOrig )
{
  //* prepare slice tracks for merging with next/previous/same sector
  //* each track transported to the border line

  float fieldBz = fSliceParam.ConstBz();
  
  nB = 0;
  
  float dAlpha = fSliceParam.DAlpha() / 2;
  float x0 = 0;

  if ( iBorder == 0 ) { // transport to the left edge of the sector and rotate horisontally
    dAlpha = dAlpha - CAMath::Pi() / 2 ;
  } else if ( iBorder == 1 ) { //  transport to the right edge of the sector and rotate horisontally
    dAlpha = -dAlpha - CAMath::Pi() / 2 ;
  } else if ( iBorder == 2 ) { // transport to the middle of the sector and rotate vertically to the border on the left
    x0 = fSliceParam.RowX( 63 );
  } else if ( iBorder == 3 ) { // transport to the middle of the sector and rotate vertically to the border on the right
    dAlpha = -dAlpha;
    x0 = fSliceParam.RowX( 63 );
  } else if ( iBorder == 4 ) { // transport to the middle of the sßector, w/o rotation
    dAlpha = 0;
    x0 = fSliceParam.RowX( 63 );
  }

  const float maxSin = CAMath::Sin( 60. / 180.*CAMath::Pi() );
  float cosAlpha = AliHLTTPCCAMath::Cos( dAlpha );
  float sinAlpha = AliHLTTPCCAMath::Sin( dAlpha );

  AliHLTTPCGMSliceTrack trackTmp;
  for ( int itr = 0; itr < fSliceNTrackInfos[iSlice]; itr++ ) {

    const AliHLTTPCGMSliceTrack *track = &fSliceTrackInfo[iSlice][itr];

    if (fromOrig)
    {
        if (fabs(track->QPt()) < MERGE_LOOPER_QPT_LIMIT) continue;
        const AliHLTTPCGMSliceTrack *trackMin = track;
        while (track->SliceNeighbour() >= 0)
        {
            track = &fSliceTrackInfo[iSlice][track->SliceNeighbour()];
            if (track->OrigTrack()->Param().X() < trackMin->OrigTrack()->Param().X()) trackMin = track;
        }
        trackTmp = *trackMin;
        track = &trackTmp;
        trackTmp.Set(trackMin->OrigTrack(), trackMin->Alpha(), trackMin->Slice());
    }
    else
    {
        if( track->Used() ) continue;
        if (fabs(track->QPt()) < MERGE_HORIZONTAL_DOUBLE_QPT_LIMIT)
        {
            if (iBorder == 0 && track->NextNeighbour() >= 0) continue;
            if (iBorder == 1 && track->PrevNeighbour() >= 0) continue;
        }
    }
    AliHLTTPCGMBorderTrack &b = B[nB];

    if(  track->TransportToXAlpha( x0, sinAlpha, cosAlpha, fieldBz, b, maxSin)){
      b.SetTrackID( itr );
      b.SetNClusters( track->NClusters() );
      for (int i = 0;i < 4;i++) if (fabs(b.Cov()[i]) >= 5.0) b.SetCov(i, 5.0);
      if (fabs(b.Cov()[4]) >= 0.5) b.SetCov(4, 0.5);
      nB++; 
    }
  }
}

void AliHLTTPCGMMerger::MergeBorderTracks ( int iSlice1, AliHLTTPCGMBorderTrack B1[], int N1, int iSlice2, AliHLTTPCGMBorderTrack B2[], int N2, int crossCE )
{
  //* merge two sets of tracks
  if (N1 == 0 || N2 == 0) return;
  
  if (DEBUG) printf("\nMERGING Slices %d %d NTracks %d %d CROSS %d\n", iSlice1, iSlice2, N1, N2, crossCE);
  int statAll=0, statMerged=0;
  float factor2ys = 1.5;//1.5;//SG!!!
  float factor2zt = 1.5;//1.5;//SG!!!
  float factor2k = 2.0;//2.2;

  factor2k  = 3.5 * 3.5 * factor2k * factor2k;
  factor2ys = 3.5 * 3.5 * factor2ys * factor2ys;
  factor2zt = 3.5 * 3.5 * factor2zt * factor2zt;
 
  int minNPartHits = 10;//SG!!!
  int minNTotalHits = 20;

  AliHLTTPCGMBorderTrack::Range *range1 = fBorderRange[iSlice1];
  AliHLTTPCGMBorderTrack::Range *range2 = fBorderRange[iSlice2] + N2;

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
      if (crossCE >= 2 && abs(b1.Row() - b2.Row()) > 1) continue;
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
    
    if (DEBUG) printf("Found match %d %d\n", b1.TrackID(), iBest2);
    
    if (crossCE >= 1)
    {
        fOutputTracks[iBest2].SetNeighborTrack(b1.TrackID());
    }
    else
    {
        AliHLTTPCGMSliceTrack &newTrack1 = fSliceTrackInfo[iSlice1][b1.TrackID()];
        AliHLTTPCGMSliceTrack &newTrack2 = fSliceTrackInfo[iSlice2][iBest2];
        
        int neighborType = crossCE < 0 ? 1 : 0;

        int old1 = newTrack2.PrevNeighbour(0);
        int old2 = newTrack1.NextNeighbour(0);
        if (old1 < 0 && old2 < 0) neighborType = 0;
        if (old1 == b1.TrackID()) continue;
        if (crossCE < 0) old1 = newTrack2.PrevNeighbour(1);
        if ( old1 >= 0 ) {
          AliHLTTPCGMSliceTrack &oldTrack1 = fSliceTrackInfo[iSlice1][old1];
          if (sameSlice)
          {
              oldTrack1.SetNextNeighbour(b1.TrackID());
              newTrack1.SetPrevNeighbour(old1);
          }
          else if ( oldTrack1.NClusters() < newTrack1.NClusters() ) {
            newTrack2.SetPrevNeighbour( -1, neighborType );
            oldTrack1.SetNextNeighbour( -1, neighborType );
          } else continue;
        }
        
        if (old2 == iBest2) continue;
        if (crossCE < 0) old2 = newTrack1.NextNeighbour(1);
        if ( old2 >= 0 ) {
          AliHLTTPCGMSliceTrack &oldTrack2 = fSliceTrackInfo[iSlice2][old2];
          if ( oldTrack2.NClusters() < newTrack2.NClusters() ) {
            oldTrack2.SetPrevNeighbour( -1, neighborType );
          } else continue;
        }
        newTrack1.SetNextNeighbour( iBest2, neighborType );
        newTrack2.SetPrevNeighbour( b1.TrackID(), neighborType );
    }
  }
  //printf("STAT: slices %d, %d: all %d merged %d\n", iSlice1, iSlice2, statAll, statMerged);
}


void AliHLTTPCGMMerger::MergeWithingSlices()
{
  float x0 = fSliceParam.RowX( 63 );  
  const float maxSin = CAMath::Sin( 60. / 180.*CAMath::Pi() );

  for ( int iSlice = 0; iSlice < fgkNSlices; iSlice++ ) {
    int nBord = 0;
    for ( int itr = 0; itr < fSliceNTrackInfos[iSlice]; itr++ ) {
      AliHLTTPCGMSliceTrack &track = fSliceTrackInfo[iSlice][itr];
      
      AliHLTTPCGMBorderTrack &b = fBorder[iSlice][nBord];
      if( track.TransportToX( x0, fSliceParam.ConstBz(), b, maxSin) ){
        b.SetTrackID( itr );
        if (DEBUG) {printf("WITHIN SLICE %d Track %d - ", iSlice, itr);for (int i = 0;i < 5;i++) {printf("%8.3f ", b.Par()[i]);} printf(" - ");for (int i = 0;i < 5;i++) {printf("%8.3f ", b.Cov()[i]);} printf("\n");}
        b.SetNClusters( track.NClusters() );
        nBord++;
      }
    }  

    MergeBorderTracks( iSlice, fBorder[iSlice], nBord, iSlice, fBorder[iSlice], nBord );
    
    for ( int itr = 0; itr < fSliceNTrackInfos[iSlice]; itr++ ) {
      AliHLTTPCGMSliceTrack &track = fSliceTrackInfo[iSlice][itr];
      if( track.PrevNeighbour()>=0 || track.Used() ) continue;
      int jtr = track.NextNeighbour();
      track.SetSliceNeighbour( jtr );
      track.SetNextNeighbour(-1);
      while( jtr>=0 ){
        AliHLTTPCGMSliceTrack &trackN = fSliceTrackInfo[iSlice][jtr];
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
  for ( int iSlice = 0; iSlice < fgkNSlices; iSlice++ ) {    
    int jSlice = fNextSliceInd[iSlice];    
    AliHLTTPCGMBorderTrack *bCurr = fBorder[iSlice], *bNext = fBorder[jSlice];
    int nCurr = 0, nNext = 0;
    MakeBorderTracks( iSlice, 2, bCurr, nCurr );
    MakeBorderTracks( jSlice, 3, bNext, nNext );
    MergeBorderTracks( iSlice, bCurr, nCurr, jSlice, bNext, nNext );
    MakeBorderTracks( iSlice, 0, bCurr, nCurr );
    MakeBorderTracks( jSlice, 1, bNext, nNext );
    MergeBorderTracks( iSlice, bCurr, nCurr, jSlice, bNext, nNext );
    MakeBorderTracks( iSlice, 0, bCurr, nCurr, true );
    MakeBorderTracks( jSlice, 1, bNext, nNext, true );
    MergeBorderTracks( iSlice, bCurr, nCurr, jSlice, bNext, nNext, -1 );
  }
}

void AliHLTTPCGMMerger::MergeCEInit()
{
    for (int k = 0;k < 2;k++)
    {
        for (int i = 0;i < fgkNSlices;i++)
        {
            fBorderCETracks[k][i] = 0;
        }
    }
}

void AliHLTTPCGMMerger::MergeCEFill(const AliHLTTPCGMSliceTrack* track, const AliHLTTPCGMMergedTrackHit& cls, int itr)
{
#if defined(HLTCA_STANDALONE) && !defined(HLTCA_GPUCODE) && !defined(HLTCA_BUILD_O2_LIB)  
  if (cls.fRow < MERGE_CE_ROWLIMIT || cls.fRow >= HLTCA_ROW_COUNT - MERGE_CE_ROWLIMIT) return;
  if (!fSliceParam.GetContinuousTracking() && fabs(cls.fZ) > 10) return;
  int slice = track->Slice();
  for (int attempt = 0;attempt < 2;attempt++)
  {
      AliHLTTPCGMBorderTrack &b = attempt == 0 ? fBorder[slice][fBorderCETracks[0][slice]] : fBorder[slice][fkSlices[slice]->NTracks() - 1 - fBorderCETracks[1][slice]];
      const float x0 = attempt == 0 ? fSliceParam.RowX(63) : cls.fX;
      if(track->TransportToX(x0, fSliceParam.ConstBz(), b, HLTCA_MAX_SIN_PHI_LOW))
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

void AliHLTTPCGMMerger::MergeCE()
{
    for (int i = 0;i < fNOutputTracks;i++) fOutputTracks[i].SetNeighborTrack(-1);
    for (int iSlice = 0;iSlice < fgkNSlices / 2;iSlice++)
    {
        int jSlice = iSlice + fgkNSlices / 2;
        MergeBorderTracks(iSlice, fBorder[iSlice], fBorderCETracks[0][iSlice], jSlice, fBorder[jSlice], fBorderCETracks[0][jSlice], 1);
        MergeBorderTracks(iSlice, fBorder[iSlice] + fkSlices[iSlice]->NTracks() - fBorderCETracks[1][iSlice], fBorderCETracks[1][iSlice], jSlice, fBorder[jSlice] + fkSlices[jSlice]->NTracks() - fBorderCETracks[1][jSlice], fBorderCETracks[1][jSlice], 2);
    }
    for (int i = 0;i < fNOutputTracks;i++)
    {
        if (fOutputTracks[i].NeighborTrack() != -1)
        {
            AliHLTTPCGMMergedTrack* trk[2] = {&fOutputTracks[i], &fOutputTracks[fOutputTracks[i].NeighborTrack()]};

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
                trk[0]->SetNeighborTrack(-1);
                trk[1]->SetNeighborTrack(i);
                std::swap(trk[0], trk[1]);
            }                
            
            bool reverse[2] = {false, false};
            if (looper)
            {
                reverse[0] = fabs(fClusters[trk[0]->FirstClusterRef()].fZ) > fabs(fClusters[trk[0]->FirstClusterRef() + trk[0]->NClusters() - 1].fZ);
                reverse[1] = fabs(fClusters[trk[1]->FirstClusterRef()].fZ) < fabs(fClusters[trk[1]->FirstClusterRef() + trk[1]->NClusters() - 1].fZ);
            }
            
            if (fSliceParam.GetContinuousTracking())
            {
                const float z0min = std::min(fabs(fClusters[trk[0]->FirstClusterRef()].fZ), fabs(fClusters[trk[0]->FirstClusterRef() + trk[0]->NClusters() - 1].fZ));
                const float z1min = std::min(fabs(fClusters[trk[1]->FirstClusterRef()].fZ), fabs(fClusters[trk[1]->FirstClusterRef() + trk[1]->NClusters() - 1].fZ));
                float offset = (z0min + z1min) / 2;
                if (!trk[0]->CSide()) offset = -offset;
                trk[0]->Param().Z() += trk[0]->Param().ZOffset() - offset;
                trk[0]->Param().ZOffset() = offset;
            }
            
            int newRef = fNOutputTrackClusters;
            for (int k = 1;k >= 0;k--)
            {
                if (reverse[k]) for (int j = trk[k]->NClusters() - 1;j >= 0;j--) fClusters[fNOutputTrackClusters++] = fClusters[trk[k]->FirstClusterRef() + j];
                else for (int j = 0;j < trk[k]->NClusters();j++) fClusters[fNOutputTrackClusters++] = fClusters[trk[k]->FirstClusterRef() + j];
            }
            trk[0]->SetFirstClusterRef(newRef);
            trk[0]->SetNClusters(trk[0]->NClusters() + trk[1]->NClusters());
            trk[1]->SetNClusters(0);
            trk[1]->SetOK(false);
        }
    }

    //for (int i = 0;i < fNOutputTracks;i++) {if (fOutputTracks[i].NeighborTrack() == -1) {fOutputTracks[i].SetNClusters(0);fOutputTracks[i].SetOK(false);}} //Remove all non-CE tracks
}

struct AliHLTTPCGMMerger_CompareClusterIdsLooper
{
  struct clcomparestruct {int i; float q;};

  const float fQPt, fDzDs;
  float fThresh;
  const AliHLTTPCCASliceOutCluster* const fCmp1;
  const clcomparestruct* const fCmp2;
  AliHLTTPCGMMerger_CompareClusterIdsLooper(float q, float z, const AliHLTTPCCASliceOutCluster* cmp1, const clcomparestruct* cmp2) : fQPt(q), fDzDs(z), fThresh(fabs(0.1f * 3.14f * 666.f * z / q)), fCmp1(cmp1), fCmp2(cmp2)
  {
    if (fThresh < 1.) fThresh = 1.;
    if (fThresh > 4.) fThresh = 4.;
  }
  bool operator()(const int aa, const int bb)
  {
    const clcomparestruct& a = fCmp2[aa];
    const clcomparestruct& b = fCmp2[bb];
    const AliHLTTPCCASliceOutCluster& a1 = fCmp1[aa];
    const AliHLTTPCCASliceOutCluster& b1 = fCmp1[bb];
    float dz = a1.GetZ() - b1.GetZ();
    if (a.q * b.q < 0) return(dz * fDzDs > 0);
    if (fabs(dz) > fThresh) return(dz * fDzDs > 0);
    return((a1.GetX() - b1.GetX()) * a.q * fQPt > 0);
  }
};

struct AliHLTTPCGMMerger_CompareClusterIds
{
  const AliHLTTPCCASliceOutCluster* const fCmp;
  AliHLTTPCGMMerger_CompareClusterIds(const AliHLTTPCCASliceOutCluster* cmp) : fCmp(cmp) {}
  bool operator()(const int aa, const int bb)
  {
      const AliHLTTPCCASliceOutCluster& a = fCmp[aa];
      const AliHLTTPCCASliceOutCluster& b = fCmp[bb];
      return(a.GetX() > b.GetX());
  }
};

struct AliHLTTPCGMMerger_CompareTracks
{
  const AliHLTTPCGMMergedTrack* const fCmp;
  AliHLTTPCGMMerger_CompareTracks(AliHLTTPCGMMergedTrack* cmp) : fCmp(cmp) {}
  bool operator()(const int aa, const int bb)
  {
    const AliHLTTPCGMMergedTrack& a = fCmp[aa];
    const AliHLTTPCGMMergedTrack& b = fCmp[bb];
    return(fabs(a.GetParam().GetQPt()) > fabs(b.GetParam().GetQPt()));
  }
};

bool AliHLTTPCGMMerger_CompareParts(const AliHLTTPCGMSliceTrack* a, const AliHLTTPCGMSliceTrack* b)
{
  return(a->X() > b->X());
}

void AliHLTTPCGMMerger::CollectMergedTracks()
{
  //Resolve connections for global tracks first
  for (int iSlice = 0;iSlice < fgkNSlices;iSlice++)
  {
    for (int itr = 0;itr < fSliceNGlobalTrackInfos[iSlice];itr++)
    {
      AliHLTTPCGMSliceTrack &globalTrack = fSliceTrackGlobalInfo[iSlice][itr];
      AliHLTTPCGMSliceTrack &localTrack = fSliceTrackInfoMemory[globalTrack.LocalTrackId()];
      localTrack.SetGlobalTrackId(localTrack.GlobalTrackId(0) != -1, fSliceTrackGlobalInfo[iSlice] + itr - fSliceTrackInfoMemory);
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

      AliHLTTPCGMSliceTrack &track = fSliceTrackInfo[iSlice][itr];
      
      if ( track.Used() ) continue;
      if ( track.PrevNeighbour() >= 0 ) continue;
      //if ( track.NextNeighbour(1) >= 0 ) continue;
      int nParts = 0;
      int nHits = 0;
      int jSlice = iSlice;
      bool looper = false;
      float looperQPt = 0.f;
      AliHLTTPCGMSliceTrack *trbase = &track, *tr = &track;
      tr->SetUsed( 1 );
      do {
        if( nParts >= kMaxParts ) break;
        if (nHits + tr->NClusters() >= kMaxClusters) break;
        nHits += tr->NClusters();
        
        if (!looper && nParts && fabs(tr->QPt()) > 2 && tr->QPt() * looperQPt < 0) looper = true;
        looperQPt = tr->QPt();
        trackParts[nParts++] = tr;
        for (int i = 0;i < 2;i++) if (tr->GlobalTrackId(i) != -1) trackParts[nParts++] = &fSliceTrackInfoMemory[tr->GlobalTrackId(i)];

        int jtr = tr->SliceNeighbour();
        if( jtr >= 0 ) {
          tr = &(fSliceTrackInfo[jSlice][jtr]);
          tr->SetUsed( 2 );
          continue;
        }
        jtr = trbase->NextNeighbour();
        if (jtr >= 0) jSlice = fNextSliceInd[jSlice];
        else
        {
            jtr = trbase->PrevNeighbour(1);
            if (jtr >= 0) jSlice = fPrevSliceInd[jSlice];
        }
        if( jtr>=0 ){
          trbase = &(fSliceTrackInfo[jSlice][jtr]);
          tr = trbase;
          if( tr->Used() ) break;
          tr->SetUsed( 1 );
          continue;
        }
        break;
      } while(1);

      // unpack and sort clusters
      
      if (nParts > 1 && !looper)
      {
        std::sort(trackParts, trackParts + nParts, AliHLTTPCGMMerger_CompareParts);
      }
      
      AliHLTTPCCASliceOutCluster trackClusters[kMaxClusters];
      uchar2 clA[kMaxClusters];
      nHits = 0;
      int yBase = 0;
      for( int ipart=0; ipart<nParts; ipart++ )
      {
        const AliHLTTPCGMSliceTrack *t = trackParts[ipart];
        if (DEBUG) printf("Collect Track %d Part %d QPt %f DzDs %f\n", fNOutputTracks, ipart, t->QPt(), t->DzDs());
        int nTrackHits = t->NClusters();
        const AliHLTTPCCASliceOutCluster *c= t->OrigTrack()->Clusters();
        AliHLTTPCCASliceOutCluster *c2 = trackClusters + nHits + nTrackHits-1;
        if (ipart && t->Slice() != trackParts[ipart - 1]->Slice()) yBase += 2;
        for( int i=0; i<nTrackHits; i++, c++, c2-- )
        {
          *c2 = *c;
          clA[nHits].x = t->Slice();
          clA[nHits++].y = yBase + (t->QPt() > 0);
        }
      }
      if ( nHits < TRACKLET_SELECTOR_MIN_HITS(track.QPt()) ) continue;

      int ordered = !looper;
      if (!looper)
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
        AliHLTTPCCASliceOutCluster trackClustersUnsorted[kMaxClusters];
        uchar2 clAUnsorted[kMaxClusters];
        int clusterIndices[kMaxClusters];
        for (int i = 0;i < nHits;i++)
        {
            trackClustersUnsorted[i] = trackClusters[i];
            clAUnsorted[i] = clA[i];
            clusterIndices[i] = i;
        }
        
        if (looper)
        {
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

            AliHLTTPCGMMerger_CompareClusterIdsLooper::clcomparestruct clusterSort[kMaxClusters];
            for (int iPart = 0;iPart < nParts;iPart++)
            {
              const AliHLTTPCGMSliceTrack *t = trackParts[iPart];
              int nTrackHits = t->NClusters();
              for (int j = 0;j < nTrackHits;j++)
              {
                int i = nTmpHits + j;
                clusterSort[i].q = t->QPt();
              }
              nTmpHits += nTrackHits;
            }
            
            std::sort(clusterIndices, clusterIndices + nHits, AliHLTTPCGMMerger_CompareClusterIdsLooper(baseQPt, baseZ, trackClusters, clusterSort));
        }
        else
        {
            std::sort(clusterIndices, clusterIndices + nHits, AliHLTTPCGMMerger_CompareClusterIds(trackClusters));
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
      
      AliHLTTPCGMMergedTrackHit *cl = fClusters + nOutTrackClusters;
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
          cl[i].fState = trackClusters[i].GetFlags() & AliHLTTPCGMMergedTrackHit::hwcfFlags; //Only allow edge and deconvoluted flags
          cl[i].fSlice = clA[i].x;
          cl[i].fLeg = clA[i].y;
#ifdef GMPropagatePadRowTime
          cl[i].fPad = trackClusters[i].fPad;
          cl[i].fTime = trackClusters[i].fTime;
#endif
      }

      AliHLTTPCGMMergedTrack &mergedTrack = fOutputTracks[fNOutputTracks];
      mergedTrack.SetOK(1);
      mergedTrack.SetLooper(looper);
      mergedTrack.SetNClusters( nHits );
      mergedTrack.SetFirstClusterRef( nOutTrackClusters );
      AliHLTTPCGMTrackParam &p1 = mergedTrack.Param();
      const AliHLTTPCGMSliceTrack &p2 = *trackParts[firstTrackIndex];
      mergedTrack.SetCSide(p2.CSide());

      AliHLTTPCGMBorderTrack b;
      if (p2.TransportToX(cl[0].fX, fSliceParam.ConstBz(), b, HLTCA_MAX_SIN_PHI, false))
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

#if defined(BUILD_QA) && defined(HLTCA_STANDALONE) && !defined(HLTCA_GPUCODE)
      if (SuppressTrack(fNOutputTracks))
      {
          mergedTrack.SetOK(0);
          mergedTrack.SetNClusters(0);
      }
#endif

      bool CEside = (mergedTrack.CSide() != 0) ^ (cl[0].fZ > cl[nHits - 1].fZ);
      if (mergedTrack.NClusters() && mergedTrack.OK()) MergeCEFill(trackParts[CEside ? lastTrackIndex : firstTrackIndex], cl[CEside ? (nHits - 1) : 0], fNOutputTracks);
      fNOutputTracks++;
      nOutTrackClusters += nHits;
    }
  }
  fNOutputTrackClusters = nOutTrackClusters;
}

void AliHLTTPCGMMerger::PrepareClustersForFit()
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
  unsigned int* trackSort = new unsigned int[fNOutputTracks];
  
#if defined(HLTCA_STANDALONE) && !defined(HLTCA_GPUCODE) && !defined(HLTCA_BUILD_O2_LIB)  
  if (fTrackOrder) delete[] fTrackOrder;
  if (fClusterAttachment) delete[] fClusterAttachment;
  fTrackOrder = new unsigned int[fNOutputTracks];
  fClusterAttachment = new int[maxId];
  fMaxID = maxId;
  for (int i = 0;i < fNOutputTracks;i++) trackSort[i] = i;
  std::sort(trackSort, trackSort + fNOutputTracks, AliHLTTPCGMMerger_CompareTracks(fOutputTracks));
  memset(fClusterAttachment, 0, maxId * sizeof(fClusterAttachment[0]));
  for (int i = 0;i < fNOutputTracks;i++) fTrackOrder[trackSort[i]] = i;
  for (int i = 0;i < fNOutputTrackClusters;i++) fClusterAttachment[fClusters[i].fNum] = attachAttached | attachGood;
#endif
  
  for (unsigned int k = 0;k < maxId;k++) sharedCount[k] = 0;
  for (int k = 0;k < fNOutputTrackClusters;k++)
  {
    sharedCount[fClusters[k].fNum] = (sharedCount[fClusters[k].fNum] << 1) | 1;
  }
  for (int k = 0;k < fNOutputTrackClusters;k++)
  {
    if (sharedCount[fClusters[k].fNum] > 1) fClusters[k].fState |= AliHLTTPCGMMergedTrackHit::flagShared;
  }
  delete[] sharedCount;
  delete[] trackSort;
}

void AliHLTTPCGMMerger::Refit(bool resetTimers)
{
  //* final refit
#ifdef HLTCA_GPU_MERGER
  if (fGPUTracker && fGPUTracker->IsInitialized())
  {
    fGPUTracker->RefitMergedTracks(this, resetTimers);
  }
  else
#endif
  {
#ifdef HLTCA_STANDALONE
#pragma omp parallel for
#endif
    for ( int itr = 0; itr < fNOutputTracks; itr++ )
    {
      AliHLTTPCGMTrackParam::RefitTrack(fOutputTracks[itr], itr, this, fClusters);
#if defined(OFFLINE_FITTER)
      gOfflineFitter.RefitTrack(fOutputTracks[itr], &fField, fClusters);
#endif
    }
  }
}

void AliHLTTPCGMMerger::Finalize()
{
#if defined(HLTCA_STANDALONE) && !defined(HLTCA_GPUCODE) && !defined(HLTCA_BUILD_O2_LIB)
    int* trkOrderReverse = new int[fNOutputTracks];
    for (int i = 0;i < fNOutputTracks;i++) trkOrderReverse[fTrackOrder[i]] = i;
    for (int i = 0;i < fNOutputTrackClusters;i++) fClusterAttachment[fClusters[i].fNum] = 0; //Reset adjacent attachment for attached clusters, set correctly below
    for (int i = 0;i < fNOutputTracks;i++)
    {
      const AliHLTTPCGMMergedTrack& trk = fOutputTracks[i];
      if (!trk.OK() || trk.NClusters() == 0) continue;
      char goodLeg = fClusters[trk.FirstClusterRef() + trk.NClusters() - 1].fLeg;
      for (int j = 0;j < trk.NClusters();j++)
      {
          int id = fClusters[trk.FirstClusterRef() + j].fNum;
          int weight = fTrackOrder[i] | attachAttached;
          if (!(fClusters[trk.FirstClusterRef() + j].fState & AliHLTTPCGMMergedTrackHit::flagReject)) weight |= attachGood;
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
