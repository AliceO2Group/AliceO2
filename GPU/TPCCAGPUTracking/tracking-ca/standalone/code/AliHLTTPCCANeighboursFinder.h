//-*- Mode: C++ -*-
// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************

#ifndef ALIHLTTPCCANEIGHBOURSFINDER_H
#define ALIHLTTPCCANEIGHBOURSFINDER_H


#include "AliHLTTPCCADef.h"
#include "AliHLTTPCCARow.h"
#include "AliHLTTPCCAGPUConfig.h"
class AliHLTTPCCATracker;

/**
 * @class AliHLTTPCCANeighboursFinder
 *
 */
class AliHLTTPCCANeighboursFinder
{
  public:
    class AliHLTTPCCASharedMemory
    {
        friend class AliHLTTPCCANeighboursFinder;
      public:
#if !defined(HLTCA_GPUCODE)
        AliHLTTPCCASharedMemory()
            : fNHits( 0 ), fUpNHits( 0 ), fDnNHits( 0 ), fUpDx( 0 ), fDnDx( 0 ), fUpTx( 0 ), fDnTx( 0 ), fIRow( 0 ), fIRowUp( 0 ), fIRowDn( 0 ), fNRows( 0 ), fRow(), fRowUp(), fRowDown() {}

        AliHLTTPCCASharedMemory( const AliHLTTPCCASharedMemory& /*dummy*/ )
            : fNHits( 0 ), fUpNHits( 0 ), fDnNHits( 0 ), fUpDx( 0 ), fDnDx( 0 ), fUpTx( 0 ), fDnTx( 0 ), fIRow( 0 ), fIRowUp( 0 ), fIRowDn( 0 ), fNRows( 0 ), fRow(), fRowUp(), fRowDown() {}
        AliHLTTPCCASharedMemory& operator=( const AliHLTTPCCASharedMemory& /*dummy*/ ) { return *this; }
#endif //!HLTCA_GPUCODE

      protected:
        int fNHits; // n hits
        int fUpNHits; // n hits in the next row
        int fDnNHits; // n hits in the prev row
        float fUpDx; // x distance to the next row
        float fDnDx; // x distance to the previous row
        float fUpTx; // normalized x distance to the next row
        float fDnTx; // normalized x distance to the previous row
        int fIRow; // row number
        int fIRowUp; // next row number
        int fIRowDn;// previous row number
        int fNRows; // number of rows
#if ALIHLTTPCCANEIGHBOURS_FINDER_MAX_NNEIGHUP > 0
        float2 fA[HLTCA_GPU_THREAD_COUNT_FINDER][ALIHLTTPCCANEIGHBOURS_FINDER_MAX_NNEIGHUP]; // temp memory
        unsigned short fB[HLTCA_GPU_THREAD_COUNT_FINDER][ALIHLTTPCCANEIGHBOURS_FINDER_MAX_NNEIGHUP]; // temp memory
#endif //ALIHLTTPCCANEIGHBOURS_FINDER_MAX_NNEIGHUP > 0
		AliHLTTPCCARow fRow, fRowUp, fRowDown;
    };

    GPUd() static int NThreadSyncPoints() { return 2; }

    GPUd() static void Thread( int nBlocks, int nThreads, int iBlock, int iThread, int iSync,
                               AliHLTTPCCASharedMemory &smem, AliHLTTPCCATracker &tracker );

};


#endif //ALIHLTTPCCANEIGHBOURSFINDER_H
