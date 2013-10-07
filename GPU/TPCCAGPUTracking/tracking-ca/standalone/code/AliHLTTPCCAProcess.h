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

MEM_CLASS_PRE() class AliHLTTPCCATracker;

#if defined(HLTCA_GPUCODE)

#ifdef __OPENCL__

#else //__OPENCL__

template<class TProcess>
GPUg() void AliHLTTPCCAProcess(int iSlice)
{
  AliHLTTPCCATracker &tracker = ( ( AliHLTTPCCATracker* ) gAliHLTTPCCATracker )[iSlice];
  GPUshared() typename TProcess::AliHLTTPCCASharedMemory smem;

  for( int iSync=0; iSync<=TProcess::NThreadSyncPoints(); iSync++){
    GPUsync();
    TProcess::Thread( get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), iSync, smem, tracker  );
  }
}

template <class TProcess>
GPUg() void AliHLTTPCCAProcessMultiA(int firstSlice, int nSliceCount, int nVirtualBlocks)
{
	if (get_group_id(0) >= nSliceCount) return;
	AliHLTTPCCATracker &tracker = ( ( AliHLTTPCCATracker* ) gAliHLTTPCCATracker )[firstSlice + get_group_id(0)];

	GPUshared() typename TProcess::AliHLTTPCCASharedMemory smem;

	for (int i = 0;i < nVirtualBlocks;i++)
	{
		for( int iSync=0; iSync<=TProcess::NThreadSyncPoints(); iSync++){
			GPUsync();
			TProcess::Thread( nVirtualBlocks, get_local_size(0), i, get_local_id(0), iSync, smem, tracker  );
		}		
	}
}

template<class TProcess>
GPUg() void AliHLTTPCCAProcessMulti(int firstSlice, int nSliceCount)
{
  const int iSlice = nSliceCount * (get_group_id(0) + (get_num_groups(0) % nSliceCount != 0 && nSliceCount * (get_group_id(0) + 1) % get_num_groups(0) != 0)) / get_num_groups(0);
  const int nSliceBlockOffset = get_num_groups(0) * iSlice / nSliceCount;
  const int sliceBlockId = get_group_id(0) - nSliceBlockOffset;
  const int sliceGridDim = get_num_groups(0) * (iSlice + 1) / nSliceCount - get_num_groups(0) * (iSlice) / nSliceCount;
  AliHLTTPCCATracker &tracker = ( ( AliHLTTPCCATracker* ) gAliHLTTPCCATracker )[firstSlice + iSlice];
  GPUshared() typename TProcess::AliHLTTPCCASharedMemory smem;

  for( int iSync=0; iSync<=TProcess::NThreadSyncPoints(); iSync++){
    GPUsync();
    TProcess::Thread( sliceGridDim, get_local_size(0), sliceBlockId, get_local_id(0), iSync, smem, tracker  );
  }
}

template<typename TProcess>
GPUg() void AliHLTTPCCAProcess1()
{
  AliHLTTPCCATracker &tracker = *( ( AliHLTTPCCATracker* ) gAliHLTTPCCATracker );
  AliHLTTPCCATrackParam tParam;

  GPUshared() typename TProcess::AliHLTTPCCASharedMemory sMem;

  typename TProcess::AliHLTTPCCAThreadMemory rMem;

  for ( int iSync = 0; iSync <= TProcess::NThreadSyncPoints(); iSync++ ) {
    GPUsync();
    TProcess::Thread( get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), iSync,
                      sMem, rMem, tracker, tParam  );
  }
}

#endif //__OPENCL__

#else //HLTCA_GPUCODE

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
