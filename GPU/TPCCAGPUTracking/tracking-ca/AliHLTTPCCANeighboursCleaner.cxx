// @(#) $Id: AliHLTTPCCANeighboursCleaner.cxx 27042 2008-07-02 12:06:02Z richterm $
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

#include "AliHLTTPCCANeighboursCleaner.h"
#include "AliHLTTPCCAMath.h"
#include "AliHLTTPCCATracker.h"

GPUd() void AliHLTTPCCANeighboursCleaner::Thread 
( Int_t /*nBlocks*/, Int_t nThreads, Int_t iBlock, Int_t iThread, Int_t iSync,
  AliHLTTPCCASharedMemory &s, AliHLTTPCCATracker &tracker )
{
  // *
  // * kill link to the neighbour if the neighbour is not pointed to the hit
  // *

  if( iSync==0 )
    {
      if( iThread==0 ){
	s.fNRows = tracker.Param().NRows();
	s.fIRow = iBlock+1;
	if( s.fIRow <= s.fNRows-2 ){
	  int iRowUp = s.fIRow+1;
	  int iRowDn = s.fIRow-1;  
	  int firstDn = tracker.Rows()[iRowDn].FirstHit();
	  s.fFirstHit = tracker.Rows()[s.fIRow].FirstHit(); 
	  int firstUp = tracker.Rows()[iRowUp].FirstHit();
	  Short_t *hhitLinkUp = tracker.HitLinkUp();
	  Short_t *hhitLinkDown = tracker.HitLinkDown();
	  s.fHitLinkUp = hhitLinkUp + s.fFirstHit;
	  s.fHitLinkDown = hhitLinkDown + s.fFirstHit;    
	  s.fNHits = firstUp - s.fFirstHit;
	  s.fUpHitLinkDown = hhitLinkDown + firstUp;
	  s.fDnHitLinkUp   = hhitLinkUp + firstDn;
	}
      }
    } 
  else if( iSync==1 )
    {
      if( s.fIRow <= s.fNRows-2 ){
	for( int ih=iThread; ih<s.fNHits; ih+=nThreads ){
	  int up = s.fHitLinkUp[ih];
	  int dn = s.fHitLinkDown[ih];
	  if( (up>=0) && ( s.fUpHitLinkDown[up] != ih) ){
	    s.fHitLinkUp[ih]= -1;      
	    //HLTCA_GPU_SUFFIX(CAMath::atomicExch)( tracker.HitLinkUp() + s.fFirstHit+ih, -1 );      
	  }
	  if( (dn>=0) && ( s.fDnHitLinkUp  [dn] != ih) ){
	    s.fHitLinkDown[ih]=-1;
	    //HLTCA_GPU_SUFFIX(CAMath::atomicExch)( tracker.HitLinkDown() + s.fFirstHit+ih,-1);
	  }
	}
      }
    }
}

