//-*- Mode: C++ -*-
// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        * 
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************

#ifndef ALIHLTTPCCASTARTHITSFINDER_H
#define ALIHLTTPCCASTARTHITSFINDER_H

#include "AliHLTTPCCADef.h"

class AliHLTTPCCATracker;

/**
 * @class AliHLTTPCCAStartHitsFinder
 * 
 */
class AliHLTTPCCAStartHitsFinder
{
 public:
  class AliHLTTPCCASharedMemory
    {
     friend class AliHLTTPCCAStartHitsFinder;
    public:
 #if !defined(HLTCA_GPUCODE)
      AliHLTTPCCASharedMemory()
	: fIRow(0), fNRows(0), fNHits(0), fHitLinkDown(0), fHitLinkUp(0), fNOldStartHits(0), fNRowStartHits(0)
      {}

      AliHLTTPCCASharedMemory( const AliHLTTPCCASharedMemory& /*dummy*/) 
	: fIRow(0), fNRows(0), fNHits(0), fHitLinkDown(0), fHitLinkUp(0), fNOldStartHits(0), fNRowStartHits(0)
      {}
      AliHLTTPCCASharedMemory& operator=(const AliHLTTPCCASharedMemory& /*dummy*/){ return *this; }
#endif
    protected:
      Int_t fIRow; // row index
      Int_t fNRows; // n rows
      Int_t fNHits; // n hits in the row
      Short_t *fHitLinkDown; // pointer to down link array
      Short_t *fHitLinkUp; // pointer to the up link array
      Int_t fRowStartHits[10240]; // temp. array for the start hits
      Int_t fNOldStartHits; // n start hits from other jobs
      Int_t fNRowStartHits; // n start hits for this row
   };

  GPUd() static Int_t NThreadSyncPoints(){ return 3; }  

  GPUd() static void Thread( Int_t nBlocks, Int_t nThreads, Int_t iBlock, Int_t iThread, Int_t iSync,
			     AliHLTTPCCASharedMemory &smem, AliHLTTPCCATracker &tracker );
};


#endif
