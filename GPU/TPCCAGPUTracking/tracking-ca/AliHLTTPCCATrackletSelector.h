//-*- Mode: C++ -*-
// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        * 
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************

#ifndef ALIHLTTPCCATRACKLETSELECTOR_H
#define ALIHLTTPCCATRACKLETSELECTOR_H


#include "AliHLTTPCCADef.h"
class AliHLTTPCCATracker;

/**
 * @class AliHLTTPCCATrackletSelector
 * 
 */
class AliHLTTPCCATrackletSelector
{
 public:
  class AliHLTTPCCASharedMemory
    {
      friend class AliHLTTPCCATrackletSelector;
    protected:
      Int_t fItr0; // index of the first track in the block
      Int_t fNThreadsTotal; // total n threads
      Int_t fNTracklets; // n of tracklets
    };

  GPUd() static Int_t NThreadSyncPoints(){ return 1; }  
  
  GPUd() static void Thread( Int_t nBlocks, Int_t nThreads, Int_t iBlock, Int_t iThread, Int_t iSync,
			     AliHLTTPCCASharedMemory &smem, AliHLTTPCCATracker &tracker );
  
};


#endif
