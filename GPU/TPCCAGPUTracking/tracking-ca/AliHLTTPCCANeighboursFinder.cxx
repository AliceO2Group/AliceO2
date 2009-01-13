// @(#) $Id: AliHLTTPCCANeighboursFinder.cxx 27042 2008-07-02 12:06:02Z richterm $
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

#include "AliHLTTPCCANeighboursFinder.h"
#include "AliHLTTPCCAMath.h"
#include "AliHLTTPCCAHitArea.h"
#include "AliHLTTPCCATracker.h"

GPUd() void AliHLTTPCCANeighboursFinder::Thread
( Int_t /*nBlocks*/, Int_t nThreads, Int_t iBlock, Int_t iThread, Int_t iSync,
  AliHLTTPCCASharedMemory &s, AliHLTTPCCATracker &tracker )
{
  //* find neighbours

  if( iSync==0 )
    {
      if( iThread==0 ){
	s.fNRows = tracker.Param().NRows();
	s.fIRow = iBlock;
	if( s.fIRow < s.fNRows ){
	  s.fFirst = tracker.Rows()[s.fIRow].FirstHit();      
	  s.fHitLinkUp = tracker.HitLinkUp() + s.fFirst;
	  s.fHitLinkDn = tracker.HitLinkDown() + s.fFirst;
	  s.fNHits = tracker.Rows()[s.fIRow].NHits();
	  if( (s.fIRow>0) && (s.fIRow<s.fNRows-1) ){
	    s.fIRowUp = s.fIRow+1;
	    s.fIRowDn = s.fIRow-1; 
	    s.fFirstDn = tracker.Rows()[s.fIRowDn].FirstHit();
	    s.fFirstUp = tracker.Rows()[s.fIRowUp].FirstHit();
	    float xDn = tracker.Rows()[s.fIRowDn].X();
	    float x = tracker.Rows()[s.fIRow].X();
	    float xUp = tracker.Rows()[s.fIRowUp].X();
	    s.fUpNHits = tracker.Rows()[s.fIRowUp].NHits();
	    s.fDnNHits = s.fFirst - s.fFirstDn;
	    s.fUpDx = xUp - x;
	    s.fDnDx = xDn - x;
	    s.fUpTx = xUp/x;
	    s.fDnTx = xDn/x;
	    s.fGridUp = tracker.Rows()[s.fIRowUp].Grid();
	    s.fGridDn = tracker.Rows()[s.fIRowDn].Grid();
	  }
	}
      }
    } 
  else if( iSync==1 )
    {
      if( s.fIRow < s.fNRows ){
	if( (s.fIRow==0) || (s.fIRow==s.fNRows-1) ){
	  for( int ih=iThread; ih<s.fNHits; ih+=nThreads ){
	    s.fHitLinkUp[ih] = -1;
	    s.fHitLinkDn[ih] = -1;
	  }
	}else if(0){
	  for( UInt_t ih=iThread; ih<s.fGridUp.N()+1; ih+=nThreads ){
	    s.fGridContentUp[ih] = tracker.GetGridContent(s.fGridUp.Offset()+ih);
	  }
	  for( UInt_t ih=iThread; ih<s.fGridDn.N()+1; ih+=nThreads ){
	    s.fGridContentDn[ih] = tracker.GetGridContent(s.fGridDn.Offset()+ih);
	  }
	}
      }
    }
  else if( iSync==2 )
    {
      if( (s.fIRow<=0) || (s.fIRow >= s.fNRows-1) ) return;
      
      const float kAreaSize = 3;     
      float chi2Cut = 3.*3.*4*(s.fUpDx*s.fUpDx + s.fDnDx*s.fDnDx );
      const Int_t kMaxN = 5;

      for( int ih=iThread; ih<s.fNHits; ih+=nThreads ){	

	UShort_t *neighUp = s.fB[iThread];
	float2 *yzUp = s.fA[iThread];
	
	int linkUp = -1;
	int linkDn = -1;
	
	if( s.fDnNHits>=1 && s.fUpNHits>=1 ){
	  
	  int nNeighUp = 0;
	  AliHLTTPCCAHit h0 = tracker.GetHit( s.fFirst + ih );	  
	  float y = h0.Y(); 
	  float z = h0.Z(); 
	  AliHLTTPCCAHitArea areaDn, areaUp;
	  areaUp.Init( tracker, s.fGridUp,  s.fFirstUp, y*s.fUpTx, z*s.fUpTx, kAreaSize, kAreaSize );
	  areaDn.Init(  tracker, s.fGridDn,  s.fFirstDn, y*s.fDnTx, z*s.fDnTx, kAreaSize, kAreaSize );      
	  do{
	    AliHLTTPCCAHit h;
	    Int_t i = areaUp.GetNext( tracker,h );
	    if( i<0 ) break;
	    neighUp[nNeighUp] = (UShort_t) i;	    
	    yzUp[nNeighUp] = make_float2( s.fDnDx*(h.Y()-y), s.fDnDx*(h.Z()-z) );
	    if( ++nNeighUp>=kMaxN ) break;
	  }while(1);

	  int nNeighDn=0;
	  if( nNeighUp>0 ){

	    int bestDn=-1, bestUp=-1;
	    float bestD=1.e10;

	    do{
	      AliHLTTPCCAHit h;
	      Int_t i = areaDn.GetNext( tracker,h );
	      if( i<0 ) break;
	      nNeighDn++;	      
	      float2 yzdn = make_float2( s.fUpDx*(h.Y()-y), s.fUpDx*(h.Z()-z) );
	      
	      for( int iUp=0; iUp<nNeighUp; iUp++ ){
		float2 yzup = yzUp[iUp];
		float dy = yzdn.x - yzup.x;
		float dz = yzdn.y - yzup.y;
		float d = dy*dy + dz*dz;
		if( d<bestD ){
		  bestD = d;
		  bestDn = i;
		  bestUp = iUp;
		}		
	      }
	    }while(1);	 
	  
	    if( bestD <= chi2Cut ){      
	      linkUp = neighUp[bestUp];
	      linkDn = bestDn;      
	    }
	  }
	}
	
	s.fHitLinkUp[ih] = linkUp;
	s.fHitLinkDn[ih] = linkDn;
      }
    }
}

