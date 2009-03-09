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
	  Int_t iRowUp = s.fIRow+1;
	  Int_t iRowDn = s.fIRow-1;  	  
	  s.fFirstHit = tracker.Row(s.fIRow).FirstHit(); 
	  const AliHLTTPCCARow &row = tracker.Row(s.fIRow);
	  const AliHLTTPCCARow &rowUp = tracker.Row(iRowUp);
	  const AliHLTTPCCARow &rowDn = tracker.Row(iRowDn);
	  s.fHitLinkUp = ((Short_t*)(tracker.RowData() + row.FullOffset())) + row.FullLinkOffset();
	  s.fHitLinkDown = s.fHitLinkUp + row.NHits();
	  s.fDnHitLinkUp = ((Short_t*)(tracker.RowData() + rowDn.FullOffset())) + rowDn.FullLinkOffset();
	  s.fUpHitLinkDown = ((Short_t*)(tracker.RowData() + rowUp.FullOffset())) + rowUp.FullLinkOffset() + rowUp.NHits();

	  s.fNHits = tracker.Row(s.fIRow).NHits();
	}
      }
    } 
  else if( iSync==1 )
    {
      if( s.fIRow <= s.fNRows-2 ){
	for( Int_t ih=iThread; ih<s.fNHits; ih+=nThreads ){
	  Int_t up = s.fHitLinkUp[ih];
	  Int_t dn = s.fHitLinkDown[ih];
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

