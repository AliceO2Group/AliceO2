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
MEM_CLASS_PRE() class AliHLTTPCCATracker;

/**
 * @class AliHLTTPCCATrackletSelector
 *
 */
class AliHLTTPCCATrackletSelector
{
  public:
    MEM_CLASS_PRE() class AliHLTTPCCASharedMemory
    {
        friend class AliHLTTPCCATrackletSelector;

	protected:
        int fItr0; // index of the first track in the block
        int fNThreadsTotal; // total n threads
        int fNTracklets; // n of tracklets
#if HLTCA_GPU_TRACKLET_SELECTOR_HITS_REG_SIZE != 0
		AliHLTTPCCAHitId fHits[HLTCA_GPU_THREAD_COUNT_SELECTOR][HLTCA_GPU_TRACKLET_SELECTOR_HITS_REG_SIZE];
#endif //HLTCA_GPU_TRACKLET_SELECTOR_HITS_REG_SIZE != 0
	};

    GPUd() static int NThreadSyncPoints() { return 1; }

    GPUd() static void Thread( int nBlocks, int nThreads, int iBlock, int iThread, int iSync,
                                MEM_LOCAL(GPUsharedref() AliHLTTPCCASharedMemory) &smem, GPUconstant() MEM_CONSTANT(AliHLTTPCCATracker) &tracker );

};


#endif //ALIHLTTPCCATRACKLETSELECTOR_H
