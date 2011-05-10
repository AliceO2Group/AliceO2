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
#include "AliHLTTPCCAHitId.h"

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
            : fIRow( 0 ), fNRows( 0 ), fNHits( 0 ), fNOldStartHits( 0 ), fNRowStartHits( 0 ) {
	  AliHLTTPCCAHitId tmp;
	  tmp.Set(0,0);
	  for( int i=0; i<ALIHLTTPCCASTARTHITSFINDER_MAX_FROWSTARTHITS; i++)fRowStartHits[i] = tmp;
}

        AliHLTTPCCASharedMemory( const AliHLTTPCCASharedMemory& /*dummy*/ )
	  : fIRow( 0 ), fNRows( 0 ), fNHits( 0 ), fNOldStartHits( 0 ), fNRowStartHits( 0 ) {
	  AliHLTTPCCAHitId tmp;
	  tmp.Set(0,0);
	  for( int i=0; i<ALIHLTTPCCASTARTHITSFINDER_MAX_FROWSTARTHITS; i++)fRowStartHits[i] = tmp;
	}
        AliHLTTPCCASharedMemory& operator=( const AliHLTTPCCASharedMemory& /*dummy*/ ) { return *this; }
#endif //!HLTCA_GPUCODE

      protected:
        int fIRow; // row index
        int fNRows; // n rows
        int fNHits; // n hits in the row
        AliHLTTPCCAHitId fRowStartHits[ALIHLTTPCCASTARTHITSFINDER_MAX_FROWSTARTHITS]; // temp. array for the start hits
        int fNOldStartHits; // n start hits from other jobs
        int fNRowStartHits; // n start hits for this row
    };

    GPUd() static int NThreadSyncPoints() { return 3; }

    GPUd() static void Thread( int nBlocks, int nThreads, int iBlock, int iThread, int iSync,
                               AliHLTTPCCASharedMemory &smem, AliHLTTPCCATracker &tracker );
};


#endif //ALIHLTTPCCASTARTHITSFINDER_H
