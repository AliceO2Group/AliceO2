//-*- Mode: C++ -*-

//* This file is property of and copyright by the ALICE HLT Project        * 
//* ALICE Experiment at CERN, All rights reserved.                         *
//* See cxx source for full Copyright notice                               *

#ifndef ALIHLTTPCCALINKSWRITER_H
#define ALIHLTTPCCALINKSWRITER_H


#include "AliHLTTPCCADef.h"

class AliHLTTPCCATracker;

/**
 * 
 */
class AliHLTTPCCALinksWriter
{
 public:
  class AliHLTTPCCASharedMemory{};
  
  GPUd() static Int_t NThreadSyncPoints(){ return 0; }  

  GPUd() static void Thread( Int_t nBlocks, Int_t nThreads, Int_t iBlock, Int_t iThread, Int_t iSync,
			     AliHLTTPCCASharedMemory &smem, AliHLTTPCCATracker &tracker );
  
};


#endif
