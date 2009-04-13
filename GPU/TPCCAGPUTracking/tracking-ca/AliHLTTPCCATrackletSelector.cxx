// @(#) $Id: AliHLTTPCCATrackletSelector.cxx 27042 2008-07-02 12:06:02Z richterm $
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


#include "AliHLTTPCCATrackletSelector.h"
#include "AliHLTTPCCATrack.h"
#include "AliHLTTPCCATracker.h"
#include "AliHLTTPCCATrackParam.h"
#include "AliHLTTPCCATracklet.h"
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
	  CAMath::AtomicExch(tracker.NTracks(),0);
	  CAMath::AtomicExch(tracker.NTrackHits(),0);
	}
	s.fNTracklets = *tracker.NTracklets();
	s.fNThreadsTotal = nThreads*nBlocks;
	s.fItr0 = nThreads*iBlock;	
      }
    }
  else if( iSync==1 )
    {
      AliHLTTPCCATrack tout;
      Int_t trackHits[160];
	
      for( Int_t itr= s.fItr0 + iThread; itr<s.fNTracklets; itr+=s.fNThreadsTotal ){    		

	AliHLTTPCCATracklet &tracklet = tracker.Tracklets()[itr];

	Int_t tNHits = tracklet.NHits();
	if( tNHits<=0 ) continue;

	const Int_t kMaxRowGap = 4;
	const Float_t kMaxShared = .1;

	Int_t firstRow = tracklet.FirstRow();
	Int_t lastRow = tracklet.LastRow();

	tout.SetNHits( 0 );
	Int_t kind = 0;
	if(0){ 
	  if( tNHits>=10 && 1./.5 >= CAMath::Abs(tracklet.Param().QPt()) ){ //SG!!!
	    kind = 1;
	  }    
	}

	Int_t w = (kind<<29) + (tNHits<<16)+itr;

	//Int_t w = (tNHits<<16)+itr;	
	//Int_t nRows = tracker.Param().NRows();
	Int_t gap = 0;
	Int_t nShared = 0;
	//std::cout<<" store tracklet: "<<firstRow<<" "<<lastRow<<std::endl;
 	for( Int_t irow=firstRow; irow<=lastRow; irow++ ){
	  gap++;
	  Int_t ih = tracklet.RowHit(irow);
	  if( ih>=0 ){
	    Int_t ihTot = tracker.Row(irow).FirstHit()+ih;
	    Bool_t own = ( tracker.HitWeights()[ihTot] <= w );
	    Bool_t sharedOK = ( (tout.NHits()<0) || (nShared<tout.NHits()*kMaxShared) );
	    if( own || sharedOK ){//SG!!!
	      gap = 0;
	      Int_t th = AliHLTTPCCATracker::IRowIHit2ID(irow,ih);
	      trackHits[tout.NHits()] = th;
	      tout.SetNHits( tout.NHits() + 1 );
	      if( !own ) nShared++;
	    }
	  }
	  
	  if( gap>kMaxRowGap || irow==lastRow ){ // store 
	    if( tout.NHits()>=10 ){ //SG!!!
	      Int_t itrout = CAMath::AtomicAdd(tracker.NTracks(),1);
	      tout.SetFirstHitID( CAMath::AtomicAdd( tracker.NTrackHits(), tout.NHits() ));
	      tout.SetParam( tracklet.Param() );
	      tout.SetAlive( 1 );
	      tracker.Tracks()[itrout] = tout;
	      for( Int_t jh=0; jh<tout.NHits(); jh++ ){
		tracker.TrackHits()[tout.FirstHitID() + jh] = trackHits[jh];
	      }
	    }
	    tout.SetNHits( 0 ); 
	    gap = 0;
	    nShared = 0;
	  }
	}
      }
    }
}
