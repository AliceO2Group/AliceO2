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
GPUg() void AliHLTTPCCAProcess(int iSlice)
{
  AliHLTTPCCATracker &tracker = ( ( AliHLTTPCCATracker* ) gAliHLTTPCCATracker )[iSlice];
  GPUshared() typename TProcess::AliHLTTPCCASharedMemory smem;

  for( int iSync=0; iSync<=TProcess::NThreadSyncPoints(); iSync++){
    __syncthreads();
    TProcess::Thread( gridDim.x, blockDim.x, blockIdx.x, threadIdx.x, iSync, smem, tracker  );
  }
}

template <class TProcess>
GPUg() void AliHLTTPCCAProcessMultiA(int firstSlice, int nSliceCount, int nVirtualBlocks)
{
	if (blockIdx.x >= nSliceCount) return;
	AliHLTTPCCATracker &tracker = ( ( AliHLTTPCCATracker* ) gAliHLTTPCCATracker )[firstSlice + blockIdx.x];

	GPUshared() typename TProcess::AliHLTTPCCASharedMemory smem;

	for (int i = 0;i < nVirtualBlocks;i++)
	{
		for( int iSync=0; iSync<=TProcess::NThreadSyncPoints(); iSync++){
			__syncthreads();
			TProcess::Thread( nVirtualBlocks, blockDim.x, i, threadIdx.x, iSync, smem, tracker  );
		}		
	}
}

template<class TProcess>
GPUg() void AliHLTTPCCAProcessMulti(int firstSlice, int nSliceCount)
{
  const int iSlice = nSliceCount * (blockIdx.x + (gridDim.x % nSliceCount != 0 && nSliceCount * (blockIdx.x + 1) % gridDim.x != 0)) / gridDim.x;
  const int nSliceBlockOffset = gridDim.x * iSlice / nSliceCount;
  const int sliceBlockId = blockIdx.x - nSliceBlockOffset;
  const int sliceGridDim = gridDim.x * (iSlice + 1) / nSliceCount - gridDim.x * (iSlice) / nSliceCount;
  AliHLTTPCCATracker &tracker = ( ( AliHLTTPCCATracker* ) gAliHLTTPCCATracker )[firstSlice + iSlice];
  GPUshared() typename TProcess::AliHLTTPCCASharedMemory smem;

  for( int iSync=0; iSync<=TProcess::NThreadSyncPoints(); iSync++){
    __syncthreads();
    TProcess::Thread( sliceGridDim, blockDim.x, sliceBlockId, threadIdx.x, iSync, smem, tracker  );
  }
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

#endif //HLTCA_GPUCODE



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

#endif //HLTCA_GPUCODE

#endif //ALIHLTTPCCAPROCESS_H
