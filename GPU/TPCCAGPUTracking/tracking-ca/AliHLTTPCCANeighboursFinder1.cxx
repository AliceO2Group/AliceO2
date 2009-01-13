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

#include "AliHLTTPCCANeighboursFinder1.h"
#include "AliHLTTPCCATracker.h"

GPUd() void AliHLTTPCCANeighboursFinder1::Thread
( Int_t nBlocks, Int_t nThreads, Int_t iBlock, Int_t iThread, Int_t iSync,
  SharedMemory &s, AliHLTTPCCATracker &tracker )
{
  int nRows = 159;
  int iRow = iBlock;
  int first = tracker.Rows()[iRow].FirstHit();      
  Short_t *hitLinkUp = tracker.HitLinkUp() + first;
  Short_t *hitLinkDn = tracker.HitLinkDown() + first;
  int NHits = tracker.Rows()[iRow].NHits();
  for( int ih=iThread; ih<NHits; ih+=nThreads ){	
    hitLinkUp[ih] = -1;
    hitLinkDn[ih] = -1;	
  }
}
