//-*- Mode: C++ -*-
// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************

#ifndef ALIHLTTPCCASTARTHITSSORTER_H
#define ALIHLTTPCCASTARTHITSSORTER_H

#include "AliHLTTPCCADef.h"
#include "AliHLTTPCCAHitId.h"

MEM_CLASS_PRE class AliHLTTPCCATracker;

/**
 * @class AliHLTTPCCAStartHitsSorter
 *
 */
class AliHLTTPCCAStartHitsSorter
{
  public:
    MEM_CLASS_PRE class AliHLTTPCCASharedMemory
    {
        friend class AliHLTTPCCAStartHitsSorter;
      public:
#if !defined(HLTCA_GPUCODE)
        AliHLTTPCCASharedMemory()
            : fStartRow( 0 ), fNRows( 0 ), fStartOffset( 0 ) {}

        AliHLTTPCCASharedMemory( const AliHLTTPCCASharedMemory& /*dummy*/ )
            : fStartRow( 0 ), fNRows( 0 ), fStartOffset( 0 ) {}
        AliHLTTPCCASharedMemory& operator=( const AliHLTTPCCASharedMemory& /*dummy*/ ) { return *this; }
#endif //!HLTCA_GPUCODE

      protected:
        int fStartRow;		// start row index
        int fNRows;			// number of rows to process
		int fStartOffset;	//start offset for hits sorted by this block
    };

    GPUd() static int NThreadSyncPoints() { return 1; }

    GPUd() static void Thread( int nBlocks, int nThreads, int iBlock, int iThread, int iSync,
                               GPUsharedref() AliHLTTPCCASharedMemory MEM_LOCAL &smem, GPUconstant() AliHLTTPCCATracker MEM_CONSTANT &tracker );
};


#endif //ALIHLTTPCCASTARTHITSSORTER_H
