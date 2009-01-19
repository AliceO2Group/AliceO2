// @(#) $Id: AliHLTTPCCATrackletSelector.cxx 27042 2008-07-02 12:06:02Z richterm $
//***************************************************************************
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
//***************************************************************************

#include "AliHLTTPCCATrackletSelector.h"
#include "AliHLTTPCCATrack.h"
#include "AliHLTTPCCATracker.h"
#include "AliHLTTPCCATrackParam.h"
#include "AliHLTTPCCAMath.h"

GPUd() void AliHLTTPCCATrackletSelector::Thread
( Int_t nBlocks, Int_t nThreads, Int_t iBlock, Int_t iThread, Int_t iSync,
  AliHLTTPCCASharedMemory &s, AliHLTTPCCATracker &tracker )
{
  // select best tracklets and kill clones

  if( iSync==0 )
    {
      if( iThread==0 ){
	if(iBlock==0){
	  CAMath::atomicExch(&(tracker.NTracks()),0);
	  CAMath::atomicExch(tracker.TrackHits(),0);
	}
	s.fNTracklets = tracker.Tracklets()[0];
	s.fNThreadsTotal = nThreads*nBlocks;
	s.fItr0 = nThreads*iBlock;
	//if( iBlock==0 ) tracker.StartHits()[0] = 0;//SG!!!
      }
    }
  else if( iSync==1 )
    {
      AliHLTTPCCATrack tout;
      Int_t trackHits[160];
	
      for( Int_t itr= s.fItr0 + iThread; itr<s.fNTracklets; itr+=s.fNThreadsTotal ){    		
	Int_t *t = tracker.Tracklets() + 1 + itr*(5+ sizeof(AliHLTTPCCATrackParam)/4 + 160 );	
	Int_t tNHits = *t;
	if( tNHits<=0 ) continue;
	
	CAMath::atomicAdd( tracker.StartHits(), 1);//SG!!!
	tout.NHits() = 0;
	Int_t *hitstore = t + 5+ sizeof(AliHLTTPCCATrackParam)/4 ;    
	Int_t w = (tNHits<<16)+itr;	
	Int_t nRows = tracker.Param().NRows();
	Int_t gap = 0;
 	for( Int_t irow=0; irow<nRows; irow++ ){
	  Int_t ih = hitstore[irow];
	  if( ih<0 ) continue;
	  AliHLTTPCCARow &row = tracker.Rows()[irow];
	  Int_t ihTot = row.FirstHit()+ih;      
	  if( tracker.HitIsUsed()[ihTot] > w ){
            if( ++gap>6){ tout.NHits()=0; break; }
            continue;
          }else gap = 0;
	  Int_t th = AliHLTTPCCATracker::IRowIHit2ID(irow,ih);
	  trackHits[tout.NHits()] = th;
	  tout.NHits()++;
	}	
	if( tout.NHits()<10 ) continue;//SG!!!
	Int_t itrout = CAMath::atomicAdd(&(tracker.NTracks()),1);
	tout.FirstHitID() = CAMath::atomicAdd( tracker.TrackHits(), tout.NHits() ) + 1;
	tout.Param() = *( (AliHLTTPCCATrackParam*)( t+5) );
	tout.Alive() = 1;
	tracker.Tracks()[itrout] = tout;
	for( Int_t ih=0; ih<tout.NHits(); ih++ ){//SG
	  tracker.TrackHits()[tout.FirstHitID() + ih] = trackHits[ih];
	}
      }
    }
}
