//-*- Mode: C++ -*-
// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************

#ifndef ALIHLTTPCCAUSEDHITSINITIALISER_H
#define ALIHLTTPCCAUSEDHITSINITIALISER_H


#include "AliHLTTPCCADef.h"

class AliHLTTPCCATracker;

/**
 * @class AliHLTTPCCAUsedHitsInitialiser
 *
 */
class AliHLTTPCCAUsedHitsInitialiser
{
  public:
    class AliHLTTPCCASharedMemory
    {
        friend class AliHLTTPCCAUsedHitsInitialiser;
      public:
#if !defined(HLTCA_GPUCODE)
        AliHLTTPCCASharedMemory()
            : fNHits( 0 ), fUsedHits( 0 ), fNThreadsTotal( 0 ), fIh0( 0 ) {}

        AliHLTTPCCASharedMemory( const AliHLTTPCCASharedMemory& /*dummy*/ )
            : fNHits( 0 ), fUsedHits( 0 ), fNThreadsTotal( 0 ), fIh0( 0 ) {}
        AliHLTTPCCASharedMemory& operator=( const AliHLTTPCCASharedMemory& /*dummy*/ ) { return *this; }
#endif
      protected:
        int fNHits; // n hits
        int *fUsedHits; // pointer to the used hits array
        int fNThreadsTotal; // total n threads
        int fIh0; // start hit index for the thread
    };

    GPUd() static int NThreadSyncPoints() { return 3; }

    GPUd() static void Thread( int nBlocks, int nThreads, int iBlock, int iThread, int iSync,
                               AliHLTTPCCASharedMemory &smem, AliHLTTPCCATracker &tracker );

};


#endif
