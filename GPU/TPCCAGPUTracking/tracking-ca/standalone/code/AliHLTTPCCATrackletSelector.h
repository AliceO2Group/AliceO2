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
#include "AliHLTTPCCAHitId.h"
#include "AliHLTTPCCAGPUConfig.h"
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
        int fItr0; // index of the first track in the block
        int fNThreadsTotal; // total n threads
        int fNTracklets; // n of tracklets
#if HLTCA_GPU_TRACKLET_SELECTOR_HITS_REG_SIZE != 0
		AliHLTTPCCAHitId fHits[HLTCA_GPU_THREAD_COUNT][HLTCA_GPU_TRACKLET_SELECTOR_HITS_REG_SIZE];
#endif
	};

    GPUd() static int NThreadSyncPoints() { return 1; }

    GPUd() static void Thread( int nBlocks, int nThreads, int iBlock, int iThread, int iSync,
                               AliHLTTPCCASharedMemory &smem, AliHLTTPCCATracker &tracker );

};


#endif
