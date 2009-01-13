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

#include "AliHLTTPCCALinksWriter.h"
#include "AliHLTTPCCATracker.h"

GPUd() void AliHLTTPCCALinksWriter::Thread
( Int_t nBlocks, Int_t nThreads, Int_t iBlock, Int_t iThread, Int_t iSync,
  AliHLTTPCCASharedMemory &/*s*/, AliHLTTPCCATracker &tracker )
{
  // copy constructed links to the new data scheme (temporary)

  if( iSync==0 )
    {
      for( int irow=iBlock; irow<tracker.Param().NRows(); irow+=nBlocks ){
	AliHLTTPCCARow &row = tracker.Rows()[irow];
	Short_t *hitLinkUp = tracker.HitLinkUp() + row.FirstHit() ;
	Short_t *hitLinkDown = tracker.HitLinkDown() + row.FirstHit();
	Short_t *newUp = ((Short_t*)(tracker.TexHitsFullData() + row.FullOffset())) + row.FullLinkOffset();
	Short_t *newDown = newUp + row.NHits();
	
	for( int ih=iThread; ih<row.NHits(); ih+=nThreads ){
	  newUp[ih] = hitLinkUp[ih];
	  newDown[ih] = hitLinkDown[ih];
	}
      }
    }
}
