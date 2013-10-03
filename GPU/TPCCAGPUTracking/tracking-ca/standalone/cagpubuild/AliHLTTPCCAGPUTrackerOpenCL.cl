#define __OPENCL__

//Disable assertions since they produce errors in GPU Code
#ifdef assert
#undef assert
#endif
#define assert(param)

#include "AliHLTTPCCATrackParam.cxx"
#include "AliHLTTPCCATrack.cxx" 

#include "AliHLTTPCCAHitArea.cxx"
#include "AliHLTTPCCAGrid.cxx"
#include "AliHLTTPCCARow.cxx"
#include "AliHLTTPCCAParam.cxx"
#include "AliHLTTPCCATracker.cxx"

#include "AliHLTTPCCATrackletSelector.cxx"
#include "AliHLTTPCCANeighboursFinder.cxx"
#include "AliHLTTPCCANeighboursCleaner.cxx"
#include "AliHLTTPCCAStartHitsFinder.cxx"
#include "AliHLTTPCCAStartHitsSorter.cxx"
#include "AliHLTTPCCATrackletConstructor.cxx"

__kernel void PreInitRowBlocks(GPUconstant() void* pTrackerTmp, int iSlice)
{
	GPUconstant() AliHLTTPCCATracker MEM_CONSTANT &pTracker = (( GPUconstant() AliHLTTPCCATracker MEM_CONSTANT * ) pTrackerTmp)[iSlice];

	//Initialize GPU RowBlocks and HitWeights
	const int nSliceDataHits = pTracker.Data().NumberOfHitsPlusAlign();
	__global int4* SliceDataHitWeights4 = (__global int4*) pTracker.Data().HitWeights();

	const int stride = get_global_size(0);
	int4 i0;
	i0.x = i0.y = i0.z = i0.w = 0;
	for (int i = get_global_id(0);i < nSliceDataHits * sizeof(int) / sizeof(int4);i += stride)
		SliceDataHitWeights4[i] = i0;
}

GPUg() void AliHLTTPCCAProcess_AliHLTTPCCANeighboursFinder(GPUconstant() void* pTrackerTmp, int iSlice)
{
  GPUconstant() AliHLTTPCCATracker MEM_CONSTANT &pTracker = (( GPUconstant() AliHLTTPCCATracker MEM_CONSTANT * ) pTrackerTmp)[iSlice];
  GPUshared() typename AliHLTTPCCANeighboursFinder::AliHLTTPCCASharedMemory MEM_LOCAL smem;

  for( int iSync=0; iSync<=AliHLTTPCCANeighboursFinder::NThreadSyncPoints(); iSync++){
    GPUsync();
    AliHLTTPCCANeighboursFinder::Thread( get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), iSync, smem, pTracker  );
  }
}

GPUg() void AliHLTTPCCAProcess_AliHLTTPCCANeighboursCleaner(GPUconstant() void* pTrackerTmp, int iSlice)
{
  GPUconstant() AliHLTTPCCATracker MEM_CONSTANT &pTracker = (( GPUconstant() AliHLTTPCCATracker MEM_CONSTANT * ) pTrackerTmp)[iSlice];
  GPUshared() typename AliHLTTPCCANeighboursCleaner::AliHLTTPCCASharedMemory MEM_LOCAL smem;

  for( int iSync=0; iSync<=AliHLTTPCCANeighboursCleaner::NThreadSyncPoints(); iSync++){
    GPUsync();
    AliHLTTPCCANeighboursCleaner::Thread( get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), iSync, smem, pTracker  );
  }
}

GPUg() void AliHLTTPCCAProcess_AliHLTTPCCAStartHitsFinder(GPUconstant() void* pTrackerTmp, int iSlice)
{
  GPUconstant() AliHLTTPCCATracker MEM_CONSTANT &pTracker = (( GPUconstant() AliHLTTPCCATracker MEM_CONSTANT * ) pTrackerTmp)[iSlice];
  GPUshared() typename AliHLTTPCCAStartHitsFinder::AliHLTTPCCASharedMemory MEM_LOCAL smem;

  for( int iSync=0; iSync<=AliHLTTPCCAStartHitsFinder::NThreadSyncPoints(); iSync++){
    GPUsync();
    AliHLTTPCCAStartHitsFinder::Thread( get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), iSync, smem, pTracker  );
  }
}

GPUg() void AliHLTTPCCAProcess_AliHLTTPCCAStartHitsSorter(GPUconstant() void* pTrackerTmp, int iSlice)
{
  GPUconstant() AliHLTTPCCATracker MEM_CONSTANT &pTracker = (( GPUconstant() AliHLTTPCCATracker MEM_CONSTANT * ) pTrackerTmp)[iSlice];
  GPUshared() typename AliHLTTPCCAStartHitsSorter::AliHLTTPCCASharedMemory MEM_LOCAL smem;

  for( int iSync=0; iSync<=AliHLTTPCCAStartHitsSorter::NThreadSyncPoints(); iSync++){
    GPUsync();
    AliHLTTPCCAStartHitsSorter::Thread( get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), iSync, smem, pTracker  );
  }
}

GPUg() void AliHLTTPCCAProcessMulti_AliHLTTPCCATrackletSelector(GPUconstant() void* pTrackerTmp, int firstSlice, int nSliceCount)
{
  const int iSlice = nSliceCount * (get_group_id(0) + (get_num_groups(0) % nSliceCount != 0 && nSliceCount * (get_group_id(0) + 1) % get_num_groups(0) != 0)) / get_num_groups(0);
  const int nSliceBlockOffset = get_num_groups(0) * iSlice / nSliceCount;
  const int sliceBlockId = get_group_id(0) - nSliceBlockOffset;
  const int sliceGridDim = get_num_groups(0) * (iSlice + 1) / nSliceCount - get_num_groups(0) * (iSlice) / nSliceCount;
  GPUconstant() AliHLTTPCCATracker MEM_CONSTANT &pTracker = (( GPUconstant() AliHLTTPCCATracker MEM_CONSTANT * ) pTrackerTmp)[firstSlice + iSlice];
  GPUshared() typename AliHLTTPCCATrackletSelector::AliHLTTPCCASharedMemory MEM_LOCAL smem;

  for( int iSync=0; iSync<=AliHLTTPCCATrackletSelector::NThreadSyncPoints(); iSync++){
    GPUsync();
    AliHLTTPCCATrackletSelector::Thread( sliceGridDim, get_local_size(0), sliceBlockId, get_local_id(0), iSync, smem, pTracker  );
  }
}

GPUg() void AliHLTTPCCATrackletConstructorGPU(GPUconstant() void* pTrackerTmp)
{
	//GPU Wrapper for AliHLTTPCCATrackletConstructor::AliHLTTPCCATrackletConstructorGPU
	GPUconstant() AliHLTTPCCATracker MEM_CONSTANT *pTracker = ( GPUconstant() AliHLTTPCCATracker MEM_CONSTANT * ) pTrackerTmp ;
	GPUshared() AliHLTTPCCATrackletConstructor::AliHLTTPCCASharedMemory MEM_LOCAL sMem;
	AliHLTTPCCATrackletConstructor::AliHLTTPCCATrackletConstructorGPU(pTracker, sMem);
}
