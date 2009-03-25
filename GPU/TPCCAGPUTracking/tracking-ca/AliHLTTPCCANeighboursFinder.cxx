// @(#) $Id: AliHLTTPCCANeighboursFinder1.cxx 27042 2008-07-02 12:06:02Z richterm $
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

#include "AliHLTTPCCANeighboursFinder.h"
#include "AliHLTTPCCAMath.h"
#include "AliHLTTPCCAHitArea.h"
#include "AliHLTTPCCATracker.h"

//#define DRAW

#ifdef DRAW
#include "AliHLTTPCCADisplay.h"
#endif

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
	  const AliHLTTPCCARow &row = tracker.Row(s.fIRow);
	  s.fFirst = row.FirstHit();
	  s.fNHits = row.NHits();

	  s.fHitLinkUp = ((Short_t*)(tracker.RowData() + row.FullOffset())) + row.FullLinkOffset();
	  s.fHitLinkDn = s.fHitLinkUp + row.NHits();

	  if( (s.fIRow>=2) && (s.fIRow<=s.fNRows-3) ){
	    s.fIRowUp = s.fIRow+2;
	    s.fIRowDn = s.fIRow-2; 
	    s.fFirstDn = tracker.Row(s.fIRowDn).FirstHit();
	    s.fFirstUp = tracker.Row(s.fIRowUp).FirstHit();
	    float xDn = tracker.Row(s.fIRowDn).X();
	    float x = tracker.Row(s.fIRow).X();
	    float xUp = tracker.Row(s.fIRowUp).X();
	    s.fUpNHits = tracker.Row(s.fIRowUp).NHits();
	    s.fDnNHits = s.fFirst - s.fFirstDn;
	    s.fUpDx = xUp - x;
	    s.fDnDx = xDn - x;
	    s.fUpTx = xUp/x;
	    s.fDnTx = xDn/x;
	    s.fGridUp = tracker.Row(s.fIRowUp).Grid();
	    s.fGridDn = tracker.Row(s.fIRowDn).Grid();
	  }
	}
      }
    } 
  else if( iSync==1 )
    {
      if( s.fIRow < s.fNRows ){
	if( (s.fIRow==0) || (s.fIRow==s.fNRows-1) || (s.fIRow==1) || (s.fIRow==s.fNRows-2) ){
	  for( Int_t ih=iThread; ih<s.fNHits; ih+=nThreads ){
	    s.fHitLinkUp[ih] = -1;
	    s.fHitLinkDn[ih] = -1;
	  }
	}else {
	  const AliHLTTPCCARow &rowUp = tracker.Row(s.fIRowUp);
	  const AliHLTTPCCARow &rowDn = tracker.Row(s.fIRowDn);
	  const UShort_t *gContentUp = (reinterpret_cast<const UShort_t*>(tracker.RowData() + rowUp.FullOffset())) + rowUp.FullGridOffset();
	  const UShort_t *gContentDn = (reinterpret_cast<const UShort_t*>(tracker.RowData() + rowDn.FullOffset())) + rowDn.FullGridOffset();
	  
	  for( UInt_t ih=iThread; ih<s.fGridUp.N()+s.fGridUp.Ny()+2; ih+=nThreads ){
	    s.fGridContentUp[ih] = gContentUp[ih];
	  }
	  for( UInt_t ih=iThread; ih<s.fGridDn.N()+s.fGridDn.Ny()+2; ih+=nThreads ){
	    s.fGridContentDn[ih] = gContentDn[ih];
	  }
	}
      }
    }
  else if( iSync==2 )
    {
      if( (s.fIRow<=1) || (s.fIRow >= s.fNRows-2) ) return;
 
      //const float kAreaSize = 3;     
      float chi2Cut = 3.*3.*4*(s.fUpDx*s.fUpDx + s.fDnDx*s.fDnDx );
      const float kAreaSize = 3;     
      //float chi2Cut = 3.*3.*(s.fUpDx*s.fUpDx + s.fDnDx*s.fDnDx ); //SG
      const Int_t kMaxN = 20;
      
      const AliHLTTPCCARow &row = tracker.Row(s.fIRow);
      const AliHLTTPCCARow &rowUp = tracker.Row(s.fIRowUp);
      const AliHLTTPCCARow &rowDn = tracker.Row(s.fIRowDn);
      float y0 = row.Grid().YMin();
      float z0 = row.Grid().ZMin();
      float stepY = row.HstepY();
      float stepZ = row.HstepZ();
      const uint4* tmpint4 = tracker.RowData() + row.FullOffset();
      const ushort2 *hits = reinterpret_cast<const ushort2*>(tmpint4);

      for( Int_t ih=iThread; ih<s.fNHits; ih+=nThreads ){	

	UShort_t *neighUp = s.fB[iThread];
	float2 *yzUp = s.fA[iThread];
	//UShort_t neighUp[5];
	//float2 yzUp[5];

	Int_t linkUp = -1;
	Int_t linkDn = -1;
	
	if( s.fDnNHits>0 && s.fUpNHits>0 ){
	  
	  Int_t nNeighUp = 0;
	  AliHLTTPCCAHit h0;
	  {
	    ushort2 hh = hits[ih];
	    h0.SetY( y0 + hh.x*stepY );
	    h0.SetZ( z0 + hh.y*stepZ );
	  }
	  //h0 = tracker.Hits()[ s.fFirst + ih ];	  

	  float y = h0.Y(); 
	  float z = h0.Z(); 

	  AliHLTTPCCAHitArea areaDn, areaUp;
	  areaUp.Init( s.fGridUp,  s.fGridContentUp,s.fFirstUp, y*s.fUpTx, z*s.fUpTx, kAreaSize, kAreaSize );
	  areaDn.Init( s.fGridDn,  s.fGridContentDn,s.fFirstDn, y*s.fDnTx, z*s.fDnTx, kAreaSize, kAreaSize );
	 
	  if( linkUp>=0 ){
	    AliHLTTPCCAHit h;
	    float y0Up = rowUp.Grid().YMin();
	    float z0Up = rowUp.Grid().ZMin();
	    float stepYUp  = rowUp.HstepY();
	    float stepZUp  = rowUp.HstepZ();
	    const uint4* tmpint4Up  = tracker.RowData() + rowUp.FullOffset();
	    const ushort2 *hitsUp  = reinterpret_cast<const ushort2*>(tmpint4Up );
	    ushort2 hh = hitsUp[linkUp];
	    h.SetY( y0Up  + hh.x*stepYUp  );
	    h.SetZ( z0Up  + hh.y*stepZUp  );    
	    neighUp[0] = linkUp;    
	    yzUp[0] = CAMath::MakeFloat2( s.fDnDx*(h.Y()-y), s.fDnDx*(h.Z()-z) );	    
	    nNeighUp = 1;
	  } else {
	    do{
	      AliHLTTPCCAHit h;
	      Int_t i = areaUp.GetNext( tracker, rowUp,s.fGridContentUp, h );
	      if( i<0 ) break;	      
	      neighUp[nNeighUp] = (UShort_t) i;
	      yzUp[nNeighUp] = CAMath::MakeFloat2( s.fDnDx*(h.Y()-y), s.fDnDx*(h.Z()-z) );
	      if( ++nNeighUp>=kMaxN ) break;
	    }while(1);
	  }
	  
	  Int_t nNeighDn=0;
	  
	  if( nNeighUp>0 ){

	    Int_t bestDn=-1, bestUp=-1;
	    float bestD=1.e10;

	    if( linkDn>=0 ){
	      AliHLTTPCCAHit h;
	      float y0Dn = rowDn.Grid().YMin();
	      float z0Dn  = rowDn.Grid().ZMin();
	      float stepYDn  = rowDn.HstepY();
	      float stepZDn  = rowDn.HstepZ();
	      const uint4* tmpint4Dn  = tracker.RowData() + rowDn.FullOffset();
	      const ushort2 *hitsDn  = reinterpret_cast<const ushort2*>(tmpint4Dn );
	      ushort2 hh = hitsDn [linkDn];
	      h.SetY( y0Dn  + hh.x*stepYDn  );
	      h.SetZ( z0Dn  + hh.y*stepZDn  );
	      
	      nNeighDn = 1;
	      
	      float2 yzdn = CAMath::MakeFloat2( s.fUpDx*(h.Y()-y), s.fUpDx*(h.Z()-z) );

	      for( Int_t iUp=0; iUp<nNeighUp; iUp++ ){
		float2 yzup = yzUp[iUp];
		float dy = yzdn.x - yzup.x;
		float dz = yzdn.y - yzup.y;
		float d = dy*dy + dz*dz;
		if( d<bestD ){
		  bestD = d;
		  bestDn = linkDn;
		  bestUp = iUp;
		}
	      }
	    } else {
	      do{
		AliHLTTPCCAHit h;
		Int_t i = areaDn.GetNext( tracker, rowDn,s.fGridContentDn,h );
		if( i<0 ) break;

		nNeighDn++;
		float2 yzdn = CAMath::MakeFloat2( s.fUpDx*(h.Y()-y), s.fUpDx*(h.Z()-z) );

		for( Int_t iUp=0; iUp<nNeighUp; iUp++ ){
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
	    }

	    if( bestD <= chi2Cut ){      
	      linkUp = neighUp[bestUp];
	      linkDn = bestDn;      
	    }
	  }
#ifdef DRAW
	  std::cout<<"n NeighUp = "<<nNeighUp<<", n NeighDn = "<<nNeighDn<<std::endl;
#endif	  

	}
	
	s.fHitLinkUp[ih] = linkUp;
	s.fHitLinkDn[ih] = linkDn;
#ifdef DRAW
	std::cout<<"Links for row "<<s.fIRow<<", hit "<<ih<<": "<<linkUp<<" "<<linkDn<<std::endl;
	if( s.fIRow==22 && ih==5 )
	  {
	  AliHLTTPCCADisplay::Instance().DrawSliceLink(s.fIRow, ih, -1, -1, 1);  
	  AliHLTTPCCADisplay::Instance().DrawSliceHit(s.fIRow, ih, kBlue, 1.);
	  AliHLTTPCCADisplay::Instance().Ask();
	}
#endif
      }
    }
}

