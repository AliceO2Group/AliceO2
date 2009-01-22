// @(#) $Id: AliHLTTPCCAStartHitsFinder.cxx 27042 2008-07-02 12:06:02Z richterm $
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

#include "AliHLTTPCCAStartHitsFinder.h"
#include "AliHLTTPCCATracker.h"
#include "AliHLTTPCCAMath.h"

GPUd() void AliHLTTPCCAStartHitsFinder::Thread
( Int_t /*nBlocks*/, Int_t nThreads, Int_t iBlock, Int_t iThread, Int_t iSync,
  AliHLTTPCCASharedMemory &s, AliHLTTPCCATracker &tracker )
{
  // find start hits for tracklets

  if( iSync==0 )
    {
      if( iThread==0 ){
	if( iBlock==0 ){
	  CAMath::atomicExch( tracker.NTracklets(),0); 
	}
	s.fNRows = tracker.Param().NRows();
	s.fIRow = iBlock+1;
	s.fNRowStartHits = 0;
	if( s.fIRow <= s.fNRows-4 ){	  
	  s.fNHits = tracker.Rows()[s.fIRow].NHits(); 
	  if( s.fNHits>=1024 ) s.fNHits = 1023;

	  AliHLTTPCCARow &row = tracker.Rows()[s.fIRow];
	  s.fHitLinkUp = ((Short_t*)(tracker.RowData() + row.FullOffset())) + row.FullLinkOffset();
	  s.fHitLinkDown = s.fHitLinkUp + row.NHits();

	} else s.fNHits = -1;
      }
    } 
  else if( iSync==1 )
    {
      for( Int_t ih=iThread; ih<s.fNHits; ih+=nThreads ){      
	if( ( s.fHitLinkDown[ih]<0 ) && ( s.fHitLinkUp[ih]>=0 ) ){
	  Int_t oldNRowStartHits = CAMath::atomicAdd(&s.fNRowStartHits,1);
	  s.fRowStartHits[oldNRowStartHits] = AliHLTTPCCATracker::IRowIHit2ID(s.fIRow, ih);
	}
      }
    }
  else if( iSync==2 )
    {
      if( iThread == 0 ){
	s.fNOldStartHits = CAMath::atomicAdd(tracker.NTracklets(),s.fNRowStartHits);  
      }
    }
  else if( iSync==3 )
    {
      Int_t *startHits = tracker.TrackletStartHits();
      for( Int_t ish=iThread; ish<s.fNRowStartHits; ish+=nThreads ){    
	startHits[s.fNOldStartHits+ish] = s.fRowStartHits[ish];
      }
    }
}

