//-*- Mode: C++ -*-
// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************

#ifndef ALIHLTTPCCAPROCESS_H
#define ALIHLTTPCCAPROCESS_H


/**
 * Definitions needed for AliHLTTPCCATracker
 *
 */

#include "AliHLTTPCCADef.h"
#include "AliHLTTPCCATrackParam.h"

class AliHLTTPCCATracker;

#if defined(HLTCA_GPUCODE)

template<class TProcess>
GPUg() void AliHLTTPCCAProcess()
{
  AliHLTTPCCATracker &tracker = *( ( AliHLTTPCCATracker* ) gAliHLTTPCCATracker );

  GPUshared() typename TProcess::AliHLTTPCCASharedMemory smem;

  TProcess::Thread( gridDim.x, blockDim.x, blockIdx.x, threadIdx.x, 0, smem, tracker  );

#define GPUPROCESS(iSync) \
  if( TProcess::NThreadSyncPoints()>=iSync ){ \
    GPUsync(); \
    TProcess::Thread( gridDim.x, blockDim.x, blockIdx.x, threadIdx.x, iSync, smem, tracker  ); \
  }

  GPUPROCESS( 1 )
  GPUPROCESS( 2 )
  GPUPROCESS( 3 )

  //for( int iSync=0; iSync<=TProcess::NThreadSyncPoints(); iSync++){
  //__syncthreads();
  //TProcess::ThreadGPU( gridDim.x, blockDim.x, blockIdx.x, threadIdx.x, iSync, smem, tracker  );
  //}

#undef GPUPROCESS
}

#else

template<class TProcess>
GPUg() void AliHLTTPCCAProcess( int nBlocks, int nThreads, AliHLTTPCCATracker &tracker )
{
  for ( int iB = 0; iB < nBlocks; iB++ ) {
    typename TProcess::AliHLTTPCCASharedMemory smem;
    for ( int iS = 0; iS <= TProcess::NThreadSyncPoints(); iS++ )
      for ( int iT = 0; iT < nThreads; iT++ ) {
        TProcess::Thread( nBlocks, nThreads, iB, iT, iS, smem, tracker  );
      }
  }
}

#endif



#if defined(HLTCA_GPUCODE)

template<typename TProcess>
GPUg() void AliHLTTPCCAProcess1()
{
  AliHLTTPCCATracker &tracker = *( ( AliHLTTPCCATracker* ) gAliHLTTPCCATracker );
  AliHLTTPCCATrackParam tParam;

  GPUshared() typename TProcess::AliHLTTPCCASharedMemory sMem;

  typename TProcess::AliHLTTPCCAThreadMemory rMem;

  for ( int iSync = 0; iSync <= TProcess::NThreadSyncPoints(); iSync++ ) {
    GPUsync();
    TProcess::Thread( gridDim.x, blockDim.x, blockIdx.x, threadIdx.x, iSync,
                      sMem, rMem, tracker, tParam  );
  }
}

#else

template<typename TProcess>
GPUg() void AliHLTTPCCAProcess1( int nBlocks, int nThreads, AliHLTTPCCATracker &tracker )
{
  for ( int iB = 0; iB < nBlocks; iB++ ) {
    typename TProcess::AliHLTTPCCASharedMemory smem;
    typename TProcess::AliHLTTPCCAThreadMemory *rMem = new typename TProcess::AliHLTTPCCAThreadMemory[nThreads];
    AliHLTTPCCATrackParam *tParam = new AliHLTTPCCATrackParam[ nThreads ];
    for ( int iS = 0; iS <= TProcess::NThreadSyncPoints(); iS++ ) {
      for ( int iT = 0; iT < nThreads; iT++ )
        TProcess::Thread( nBlocks, nThreads, iB, iT, iS, smem, rMem[iT], tracker, tParam[iT]  );
    }
    delete[] rMem;
    delete[] tParam;
  }
}

#endif

#endif
