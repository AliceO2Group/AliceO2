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

#include "AliHLTTPCGMMerger.h"

#include "AliHLTTPCCAMath.h"
#include "TStopwatch.h"

#include "AliHLTTPCCATrackParam.h"
#include "AliHLTTPCCASliceOutput.h"
#include "AliHLTTPCGMMergedTrack.h"
#include "AliHLTTPCCADataCompressor.h"
#include "AliHLTTPCCAParam.h"
#include "AliHLTTPCCATrackLinearisation.h"
#include "AliHLTTPCCADataCompressor.h"

#include "AliHLTTPCGMTrackParam.h"
#include "AliHLTTPCGMTrackLinearisation.h"
#include "AliHLTTPCGMSliceTrack.h"
#include "AliHLTTPCGMBorderTrack.h"
#include <cmath>

#include <algorithm>

#include "AliHLTTPCCAGPUConfig.h"
#include "MemoryAssignmentHelpers.h"

AliHLTTPCGMMerger::AliHLTTPCGMMerger()
  :
  fSliceParam(),
  fNOutputTracks( 0 ),
  fNOutputTrackClusters( 0 ),
  fOutputTracks( 0 ),
  fOutputClusterIds(0),
  fSliceTrackInfos( 0 ),  
  fMaxSliceTracks(0),
  fClusterX(0),
  fClusterY(0),
  fClusterZ(0),
  fClusterRowType(0),
  fClusterAngle(0),
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
  {
    const double kCLight = 0.000299792458;
    double constBz = fSliceParam.BzkG() * kCLight;
    
    fPolinomialFieldBz[0] = constBz * (  0.999286   );
    fPolinomialFieldBz[1] = constBz * ( -4.54386e-7 );
    fPolinomialFieldBz[2] = constBz * (  2.32950e-5 );
    fPolinomialFieldBz[3] = constBz * ( -2.99912e-7 );
    fPolinomialFieldBz[4] = constBz * ( -2.03442e-8 );
    fPolinomialFieldBz[5] = constBz * (  9.71402e-8 );    
  }
  
  Clear();
}


AliHLTTPCGMMerger::AliHLTTPCGMMerger(const AliHLTTPCGMMerger&)
  :
  fSliceParam(),
  fNOutputTracks( 0 ),
  fNOutputTrackClusters( 0 ),
  fOutputTracks( 0 ),
  fOutputClusterIds(0),
  fSliceTrackInfos( 0 ),  
  fMaxSliceTracks(0),
  fClusterX(0),
  fClusterY(0),
  fClusterZ(0),
  fClusterRowType(0),
  fClusterAngle(0),
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
  {
    const double kCLight = 0.000299792458;
    double constBz = fSliceParam.BzkG() * kCLight;

    fPolinomialFieldBz[0] = constBz * (  0.999286   );
    fPolinomialFieldBz[1] = constBz * ( -4.54386e-7 );
    fPolinomialFieldBz[2] = constBz * (  2.32950e-5 );
    fPolinomialFieldBz[3] = constBz * ( -2.99912e-7 );
    fPolinomialFieldBz[4] = constBz * ( -2.03442e-8 );
    fPolinomialFieldBz[5] = constBz * (  9.71402e-8 );    
  }  
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
  delete[] fOutputClusterIds;
  delete[] fSliceTrackInfos;  
  if (!fGPUTracker)
  {
	  delete[] fOutputTracks;
	  delete[] fClusterX;
	  delete[] fClusterY;
	  delete[] fClusterZ;
	  delete[] fClusterRowType;
	  delete[] fClusterAngle;
  }
  delete[] fBorderMemory;
  delete[] fBorderRangeMemory;

  fNOutputTracks = 0;
  fOutputTracks = 0;
  fOutputClusterIds = 0;
  fSliceTrackInfos = 0;
  fMaxSliceTracks = 0;
  fClusterX = 0;
  fClusterY = 0;
  fClusterZ = 0;
  fClusterRowType = 0;
  fClusterAngle = 0;
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

  {
    const double kCLight = 0.000299792458;
    double constBz = fSliceParam.BzkG() * kCLight;

    fPolinomialFieldBz[0] = constBz * (  0.999286   );
    fPolinomialFieldBz[1] = constBz * ( -4.54386e-7 );
    fPolinomialFieldBz[2] = constBz * (  2.32950e-5 );
    fPolinomialFieldBz[3] = constBz * ( -2.99912e-7 );
    fPolinomialFieldBz[4] = constBz * ( -2.03442e-8 );
    fPolinomialFieldBz[5] = constBz * (  9.71402e-8 );    
  }
  
  int nIter = 1;
  TStopwatch timer;
#ifdef HLTCA_STANDALONE
  unsigned long long int a, b, c, d, e, f, g;
  AliHLTTPCCATracker::StandaloneQueryFreq(&g);
#endif
  //cout<<"Merger..."<<endl;
  for( int iter=0; iter<nIter; iter++ ){
    if( !AllocateMemory() ) return 0;
#ifdef HLTCA_STANDALONE
	AliHLTTPCCATracker::StandaloneQueryTime(&a);
#endif
    UnpackSlices();
#ifdef HLTCA_STANDALONE
	AliHLTTPCCATracker::StandaloneQueryTime(&b);
#endif
    MergeWithingSlices();
#ifdef HLTCA_STANDALONE
	AliHLTTPCCATracker::StandaloneQueryTime(&c);
#endif
    MergeSlices();
#ifdef HLTCA_STANDALONE
	AliHLTTPCCATracker::StandaloneQueryTime(&d);
#endif
    CollectMergedTracks();
#ifdef HLTCA_STANDALONE
	AliHLTTPCCATracker::StandaloneQueryTime(&e);
#endif
    Refit();
#ifdef HLTCA_STANDALONE
	AliHLTTPCCATracker::StandaloneQueryTime(&f);
	if (fDebugLevel > 0)
	{
		printf("Merge Time:\tUnpack Slices:\t%lld us\n", (b - a) * 1000000 / g);
		printf("\t\tMerge Within:\t%lld us\n", (c - b) * 1000000 / g);
		printf("\t\tMerge Slices:\t%lld us\n", (d - c) * 1000000 / g);
		printf("\t\tCollect:\t%lld us\n", (e - d) * 1000000 / g);
		printf("\t\tRefit:\t\t%lld us\n", (f - e) * 1000000 / g);
	}
	int newTracks = 0;
	for (int i = 0;i < fNOutputTracks;i++) if (fOutputTracks[i].OK()) newTracks++;
	printf("Output Tracks: %d\n", newTracks);
#endif
  }  
  timer.Stop();  
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

  fOutputClusterIds = new UInt_t[fNClusters];
  fSliceTrackInfos = new AliHLTTPCGMSliceTrack[nTracks];
  if (fGPUTracker)
  {
	char* basemem = fGPUTracker->MergerBaseMemory();
	AssignMemory(fClusterX, basemem, fNClusters);
	AssignMemory(fClusterY, basemem, fNClusters);
	AssignMemory(fClusterZ, basemem, fNClusters);
	AssignMemory(fClusterAngle, basemem, fNClusters);
	AssignMemory(fClusterRowType, basemem, fNClusters);
	AssignMemory(fOutputTracks, basemem, nTracks);
  }
  else
  {
	  fOutputTracks = new AliHLTTPCGMMergedTrack[nTracks];
	  fClusterX = new float[fNClusters];
	  fClusterY = new float[fNClusters];
	  fClusterZ = new float[fNClusters];
	  fClusterRowType = new UInt_t[fNClusters];
	  fClusterAngle = new float[fNClusters];        
  }
  fBorderMemory = new AliHLTTPCGMBorderTrack[fMaxSliceTracks*2];
  fBorderRangeMemory = new AliHLTTPCGMBorderTrack::Range[fMaxSliceTracks*2];  

  return ( ( fOutputTracks!=NULL )
	   && ( fOutputClusterIds!=NULL )
	   && ( fSliceTrackInfos!=NULL )
	   && ( fClusterX!=NULL )
	   && ( fClusterY!=NULL )
	   && ( fClusterZ!=NULL )
	   && ( fClusterRowType!=NULL )
	   && ( fClusterAngle!=NULL )
	   && ( fBorderMemory!=NULL )
	   && ( fBorderRangeMemory!=NULL )
	   );
}



void AliHLTTPCGMMerger::UnpackSlices()
{
  //* unpack the cluster information from the slice tracks and initialize track info array
  
  int nTracksCurrent = 0;
  for ( int iSlice = 0; iSlice < fgkNSlices; iSlice++ ) {

    fSliceTrackInfoStart[ iSlice ] = nTracksCurrent;

    fSliceNTrackInfos[ iSlice ] = 0;

    if ( !fkSlices[iSlice] ) continue;

    float alpha = fSliceParam.Alpha( iSlice );

    const AliHLTTPCCASliceOutput &slice = *( fkSlices[iSlice] );
    const AliHLTTPCCASliceOutTrack *sliceTr = slice.GetFirstTrack();    

    for ( int itr = 0; itr < slice.NTracks(); itr++, sliceTr = sliceTr->GetNextTrack() ) {
      AliHLTTPCGMSliceTrack &track = fSliceTrackInfos[nTracksCurrent];
      track.Set( sliceTr, alpha );
      if( !track.FilterErrors( fSliceParam, .999 ) ) continue;
      track.SetPrevNeighbour( -1 );
      track.SetNextNeighbour( -1 );
      track.SetSliceNeighbour( -1 );
      track.SetUsed( 0 );
      nTracksCurrent++;
      fSliceNTrackInfos[ iSlice ]++;
    }
    
    //std::cout<<"Unpack slice "<<iSlice<<": ntracks "<<slice.NTracks()<<"/"<<fSliceNTrackInfos[iSlice]<<std::endl;
  } 
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
    dAlpha = dAlpha;
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
      nB++; 
    }
  }
}


void AliHLTTPCGMMerger::MergeBorderTracks ( int iSlice1, AliHLTTPCGMBorderTrack B1[],  int N1,
					    int iSlice2, AliHLTTPCGMBorderTrack B2[],  int N2 )
{
  //* merge two sets of tracks

  //std::cout<<" Merge slices "<<iSlice1<<"+"<<iSlice2<<": tracks "<<N1<<"+"<<N2<<std::endl;
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
      //   if( iSlice1==7 && iSlice2==8 ){
      //cout<<b.TrackID()<<": "<<b.Cov()[0]<<" "<<b.Cov()[1]<<endl;
      //}
      float d = 3.5*sqrt(b.Cov()[1]);
      range1[itr].fId = itr;
      range1[itr].fMin = b.Par()[1] - d;
      range1[itr].fMax = b.Par()[1] + d;
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
	float d = 3.5*sqrt(b.Cov()[1]);
	range2[itr].fId = itr;
	range2[itr].fMin = b.Par()[1] - d;
	range2[itr].fMax = b.Par()[1] + d;
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
      if ( b2.NClusters() < lBest2 ) continue;
      
      if( !b1.CheckChi2Y(b2, factor2ys ) ) continue;
      //if( !b1.CheckChi2Z(b2, factor2zt ) ) continue;
      if( !b1.CheckChi2QPt(b2, factor2k ) ) continue;
      if( !b1.CheckChi2YS(b2, factor2ys ) ) continue;
      if( !b1.CheckChi2ZT(b2, factor2zt ) ) continue;
      if ( b2.NClusters() < minNPartHits ) continue;
      if ( b1.NClusters() + b2.NClusters() < minNTotalHits ) continue;      

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





void AliHLTTPCGMMerger::CollectMergedTracks()
{
	//* 

	//for ( int iSlice = 0; iSlice < fgkNSlices; iSlice++ ) {
	//for ( int itr = 0; itr < fSliceNTrackInfos[iSlice]; itr++ ) {
	//AliHLTTPCGMSliceTrack &track = fSliceTrackInfos[ fSliceTrackInfoStart[iSlice] + itr ];
	//if( track.Used()!=2 ) track.SetUsed(0);
	//}
	//}

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

			std::sort(trackParts, trackParts+nParts, CompareTrackParts );

			AliHLTTPCCASliceOutClusterWithAngle tmp[kMaxClusters];
			int nHits = 0;
			for( int ipart=0; ipart<nParts; ipart++ ){
				const AliHLTTPCGMSliceTrack *t = trackParts[ipart];
				int nTrackHits = t->NClusters();
				if( nHits + nTrackHits >= kMaxClusters ) break;
				const AliHLTTPCCASliceOutCluster *c= t->OrigTrack()->Clusters();
				AliHLTTPCCASliceOutClusterWithAngle *c2 = tmp+nHits + nTrackHits-1;
				for( int i=0; i<nTrackHits; i++, c++, c2-- )
				{
					c2->c = *c;
					c2->angle = t->Alpha();
				}
				nHits+=nTrackHits;
			}

			if ( nHits < 30 ) continue;   

			int ordered = 1;
			float *clX = fClusterX + nOutTrackClusters;
			float *clA = fClusterAngle + nOutTrackClusters;
			UInt_t *clId = fOutputClusterIds + nOutTrackClusters;
			UInt_t *clT  = fClusterRowType + nOutTrackClusters;
			float *clY = fClusterY + nOutTrackClusters;
			float *clZ = fClusterZ + nOutTrackClusters;

			clX[0] = tmp[0].c.GetX();
			for( int i=1; i<nHits; i++ )
			{
				clX[i] = tmp[i].c.GetX();
				if (tmp[i].c.GetX() >= tmp[i - 1].c.GetX())
				{
					ordered = 0;
				}
			}
/*			if (ordered == 0)
			{
				bool goodIds[kMaxClusters];
				for (int i = 0;i < nHits;i++)
				{
					tmp[i].r2 = tmp[i].c.GetX() * tmp[i].c.GetX() + tmp[i].c.GetY() * tmp[i].c.GetY();
					//printf("%d: %f %f %f %f\n", i, tmp[i].c.GetX(), tmp[i].c.GetY(), tmp[i].c.GetZ(), (float) sqrt((double) tmp[i].r2));
				}
				printf("Unordered track\n");
				qsort(tmp, nHits, sizeof(AliHLTTPCCASliceOutClusterWithAngle), ClusterSortComparison);

				int nHitsGood = 1;
				float lastAngle = tmp[0].angle;
				float lastr = tmp[0].r2;
				float lastX = tmp[0].c.GetX();
				float lastY = tmp[0].c.GetY();
				float lastZ = tmp[0].c.GetZ();
				float lastId = tmp[0].c.GetId();
				
				goodIds[0] = true;
				for (int i = 1;i < nHits;i++)
				{
					float newx, newy;
					if (tmp[i].angle != lastAngle)
					{
						const float angle = - tmp[i].angle + lastAngle;
						const float cA = CAMath::Cos( -angle );
						const float sA = CAMath::Sin( -angle );
						newx = lastX*cA + lastY*sA;
						newy = -lastX*sA + lastY*cA;
					}
					else
					{
						newx = lastX;
						newy = lastY;
					}

					//printf("%d: Angle %f LastAngle %f  - NewX %f NewY %f, OldX %f OldY %f - X %f Y %f - R %f LastR %f\n", i, tmp[i].angle, lastAngle, newx, newy, lastX, lastY, tmp[i].c.GetX(), tmp[i].c.GetY(), (float) sqrt((double) tmp[i].r2), (float) sqrt((double) lastr));

					const float dy = tmp[i].c.GetY() - newy;
					const float dz = tmp[i].c.GetZ() - lastZ;
					const float dx = tmp[i].c.GetX() - newx;
					const float dr = tmp[i].r2 - lastr;
					if (tmp[i].c.GetId() != lastId && tmp[i].r2 <= lastr && fabs(dx) < 10 && fabs(dy) < 6. * fabs(dr))
					{
						goodIds[i] = true;
						nHitsGood++;
						lastAngle = tmp[i].angle;
						lastr = tmp[i].r2;
						lastX = tmp[i].c.GetX();
						lastY = tmp[i].c.GetY();
						lastZ = tmp[i].c.GetZ();
						lastId = tmp[i].c.GetId();
					}
					else
					{
						goodIds[i] = false;
					}
				}

				for (int i = 0;i < nHits;i++)
				{
					//printf("%d: %f %f %f\n", i, tmp[i].c.GetX(), tmp[i].c.GetY(), tmp[i].c.GetZ());
				}

				int iGood = 0, iBad = 0;
				for( int i=0; i<nHits; i++ )
				{
					if (goodIds[i])
					{
						clX[iGood] = tmp[i].c.GetX();		//Copy X Coordinates again in correct order
						clA[iGood] = tmp[i].angle; 
						clId[iGood] = tmp[i].c.GetId();
						clT[iGood] = tmp[i].c.GetRowType();
						clY[iGood] = tmp[i].c.GetY();
						clZ[iGood] = tmp[i].c.GetZ();
						iGood++;
					}
					else
					{
						clX[nHitsGood + iBad] = tmp[i].c.GetX();		//Copy X Coordinates again in correct order
						clA[nHitsGood + iBad] = tmp[i].angle; 
						clId[nHitsGood + iBad] = tmp[i].c.GetId();
						clT[nHitsGood + iBad] = tmp[i].c.GetRowType();
						clY[nHitsGood + iBad] = tmp[i].c.GetY();
						clZ[nHitsGood + iBad] = tmp[i].c.GetZ();
						iBad++;
					}
				}
			}
			else*/
			{
				for( int i=0; i<nHits; i++ ) clA[i] = tmp[i].angle;      
				for( int i=0; i<nHits; i++ ) clId[i] = tmp[i].c.GetId();      
				for( int i=0; i<nHits; i++ ) clT[i] = tmp[i].c.GetRowType();  
				for( int i=0; i<nHits; i++ ) clY[i] = tmp[i].c.GetY();      
				for( int i=0; i<nHits; i++ ) clZ[i] = tmp[i].c.GetZ();
			}

			AliHLTTPCGMMergedTrack &mergedTrack = fOutputTracks[fNOutputTracks];
			mergedTrack.SetOK(1);
			mergedTrack.SetNClusters( nHits );
			mergedTrack.SetFirstClusterRef( nOutTrackClusters );
			AliHLTTPCGMTrackParam &p1 = mergedTrack.Param();
			const AliHLTTPCGMSliceTrack &p2 = *(trackParts[0]);

			p1.X() = p2.X();
			p1.Y() = p2.Y();
			p1.Z() = p2.Z();
			p1.SinPhi()  = p2.SinPhi();
			p1.DzDs()  = p2.DzDs();
			p1.QPt()  = p2.QPt();
			mergedTrack.SetAlpha( p2.Alpha() );

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
	if (fGPUTracker)
	{
		fGPUTracker->RefitMergedTracks(this);
	}
	else
#endif
	{
#ifdef HLTCA_STANDALONE
#pragma omp parallel for
#endif
	  for ( int itr = 0; itr < fNOutputTracks; itr++ ) {
		AliHLTTPCGMMergedTrack &track = fOutputTracks[itr];
		if( !track.OK() ) continue;    

		int nTrackHits = track.NClusters();
	       
		AliHLTTPCGMTrackParam t = track.Param();
		float Alpha = track.Alpha();  
	    
		t.Fit( fPolinomialFieldBz,
		   fClusterX+track.FirstClusterRef(),
		   fClusterY+track.FirstClusterRef(),
		   fClusterZ+track.FirstClusterRef(),
		   fClusterRowType+track.FirstClusterRef(),
		   fClusterAngle+track.FirstClusterRef(),
		   fSliceParam, nTrackHits, Alpha, 0 );      
	    
		if ( fabs( t.QPt() ) < 1.e-4 ) t.QPt() = 1.e-4 ;
		
		bool ok = nTrackHits >= 30 && t.CheckNumericalQuality() && fabs( t.SinPhi() ) <= .999;
		track.SetOK(ok);
		if( !ok ) continue;

		if( 1 ){//SG!!!
		  track.SetNClusters( nTrackHits );
		  track.Param() = t;
		  track.Alpha() = Alpha;
		}

		{
		  int ind = track.FirstClusterRef();
		  float alpha = fClusterAngle[ind];
		  float x = fClusterX[ind];
		  float y = fClusterY[ind];
		  float z = fClusterZ[ind];
		  float sinA = AliHLTTPCCAMath::Sin( alpha - track.Alpha());
		  float cosA = AliHLTTPCCAMath::Cos( alpha - track.Alpha());
		  track.SetLastX( x*cosA - y*sinA );
		  track.SetLastY( x*sinA + y*cosA );
		  track.SetLastZ( z );
		}
	  }
	}
}

