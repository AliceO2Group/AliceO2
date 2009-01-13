// @(#) $Id: AliHLTTPCCAUsedHitsInitialiser.cxx 27042 2008-07-02 12:06:02Z richterm $
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

#include "AliHLTTPCCAUsedHitsInitialiser.h"
#include "AliHLTTPCCATracker.h"


void AliHLTTPCCAUsedHitsInitialiser::Thread
( Int_t nBlocks, Int_t nThreads, Int_t iBlock, Int_t iThread, Int_t iSync,
  AliHLTTPCCASharedMemory &s, AliHLTTPCCATracker &tracker )
{
  // initialise used hit flags with 0

  if( iSync==0 )
    {
      if( iThread==0 ){
	s.fNHits = tracker.NHitsTotal();
	s.fUsedHits = tracker.HitIsUsed();
	s.fNThreadsTotal = nThreads*nBlocks;
	s.fIh0 = nThreads*iBlock;
      }
    } 
  else if( iSync==1 )
    {
      for( int ih=s.fIh0 + iThread; ih<s.fNHits; ih+=s.fNThreadsTotal ) s.fUsedHits[ih] = 0;	      
    }
}

