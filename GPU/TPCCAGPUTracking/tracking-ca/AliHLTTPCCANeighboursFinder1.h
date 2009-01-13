//-*- Mode: C++ -*-

//* This file is property of and copyright by the ALICE HLT Project        * 
//* ALICE Experiment at CERN, All rights reserved.                         *
//* See cxx source for full Copyright notice                               *

#ifndef ALIHLTTPCCANEIGHBOURSFINDER1_H
#define ALIHLTTPCCANEIGHBOURSFINDER1_H


#include "AliHLTTPCCADef.h"
class AliHLTTPCCATracker;

/**
 * @class AliHLTTPCCANeighboursFinder
 * 
 */
class AliHLTTPCCANeighboursFinder1
{
 public:
  class SharedMemory
    {
    public:
    };
  
  GPUd() static Int_t NThreadSyncPoints(){ return 0; }  

  GPUd() static void Thread( Int_t nBlocks, Int_t nThreads, Int_t iBlock, Int_t iThread, Int_t iSync,
			     SharedMemory &smem, AliHLTTPCCATracker &tracker );
  
};


#endif
