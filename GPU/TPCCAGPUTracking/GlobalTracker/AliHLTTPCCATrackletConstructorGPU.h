#include "AliHLTTPCCAGPUConfig.h"

GPUdi() int AliHLTTPCCATrackletConstructor::FetchTracklet(GPUconstant() MEM_CONSTANT(AliHLTTPCCATracker) &tracker, GPUsharedref() MEM_LOCAL(AliHLTTPCCASharedMemory) &sMem)
{
	const int nativeslice = get_group_id(0) % tracker.GPUParametersConst()->fGPUnSlices;
	const int nTracklets = *tracker.NTracklets();
	GPUsync();
	if (get_local_id(0) == 0)
	{
		if (sMem.fNextTrackletFirstRun == 1)
		{
			sMem.fNextTrackletFirst = (get_group_id(0) - nativeslice) / tracker.GPUParametersConst()->fGPUnSlices * HLTCA_GPU_THREAD_COUNT_CONSTRUCTOR;
			sMem.fNextTrackletFirstRun = 0;
		}
		else
		{
			if (tracker.GPUParameters()->fNextTracklet < nTracklets)
			{
				const int firstTracklet = CAMath::AtomicAdd(&tracker.GPUParameters()->fNextTracklet, HLTCA_GPU_THREAD_COUNT_CONSTRUCTOR);
				if (firstTracklet < nTracklets) sMem.fNextTrackletFirst = firstTracklet;
				else sMem.fNextTrackletFirst = -2;
			}
			else
			{
				sMem.fNextTrackletFirst = -2;
			}
		}
	}
	GPUsync();
	return (sMem.fNextTrackletFirst);
}

GPUdi() void AliHLTTPCCATrackletConstructor::AliHLTTPCCATrackletConstructorGPU(GPUconstant() MEM_CONSTANT(AliHLTTPCCATracker) *pTracker, GPUsharedref() AliHLTTPCCATrackletConstructor::MEM_LOCAL(AliHLTTPCCASharedMemory)& sMem)
{
	const int nSlices = pTracker[0].GPUParametersConst()->fGPUnSlices;
	int mySlice = get_group_id(0) % nSlices;
	int currentSlice = -1;

	if (get_local_id(0) == 0)
	{
		sMem.fNextTrackletFirstRun = 1;
	}

	for (int iSlice = 0;iSlice < nSlices;iSlice++)
	{
		GPUconstant() MEM_CONSTANT(AliHLTTPCCATracker) &tracker = pTracker[mySlice];

		AliHLTTPCCAThreadMemory rMem;

		while ((rMem.fItr = FetchTracklet(tracker, sMem)) != -2)
		{
			if (rMem.fItr >= 0 && get_local_id(0) < HLTCA_GPU_THREAD_COUNT_CONSTRUCTOR)
			{
				rMem.fItr += get_local_id(0);
			}
			else
			{
				rMem.fItr = -1;
			}

			if (mySlice != currentSlice)
			{
				if (get_local_id(0) == 0)
				{
					sMem.fNTracklets = *tracker.NTracklets();
				}

				for (int i = get_local_id(0);i < HLTCA_ROW_COUNT * sizeof(MEM_PLAIN(AliHLTTPCCARow)) / sizeof(int);i += get_local_size(0))
				{
					reinterpret_cast<GPUsharedref() int*>(&sMem.fRows)[i] = reinterpret_cast<GPUglobalref() int*>(tracker.SliceDataRows())[i];
				}
				GPUsync();
				currentSlice = mySlice;
			}

			if (rMem.fItr >= 0 && rMem.fItr < sMem.fNTracklets)
			{
				rMem.fGo = true;
				DoTracklet(tracker, sMem, rMem);
			}
		}
		if (++mySlice >= nSlices) mySlice = 0;
	}
}

GPUdi() void AliHLTTPCCATrackletConstructor::AliHLTTPCCATrackletConstructorSingleSlice(GPUconstant() MEM_CONSTANT(AliHLTTPCCATracker) *pTracker, GPUsharedref() AliHLTTPCCATrackletConstructor::MEM_LOCAL(AliHLTTPCCASharedMemory)& sMem)
{
	GPUconstant() MEM_CONSTANT(AliHLTTPCCATracker) &tracker = *pTracker;

	if (get_local_id(0) == 0) sMem.fNTracklets = *tracker.NTracklets();
	for (int i = get_local_id(0);i < HLTCA_ROW_COUNT * sizeof(MEM_PLAIN(AliHLTTPCCARow)) / sizeof(int);i += get_local_size(0))
	{
		reinterpret_cast<GPUsharedref() int*>(&sMem.fRows)[i] = reinterpret_cast<GPUglobalref() int*>(tracker.SliceDataRows())[i];
	}
	GPUsync();

	AliHLTTPCCAThreadMemory rMem;
	for (rMem.fItr = get_global_id(0);rMem.fItr < sMem.fNTracklets;rMem.fItr += get_global_size(0))
	{
		rMem.fGo = 1;
		DoTracklet(tracker, sMem, rMem);
	}
}

#ifndef __OPENCL__
GPUg() void AliHLTTPCCATrackletConstructorGPU()
{
	//GPU Wrapper for AliHLTTPCCATrackletConstructor::AliHLTTPCCATrackletConstructorGPU
	AliHLTTPCCATracker *pTracker = ( ( AliHLTTPCCATracker* ) gAliHLTTPCCATracker );
	GPUshared() AliHLTTPCCATrackletConstructor::MEM_LOCAL(AliHLTTPCCASharedMemory) sMem;
	AliHLTTPCCATrackletConstructor::AliHLTTPCCATrackletConstructorGPU(pTracker, sMem);
}

GPUg() void AliHLTTPCCATrackletConstructorSingleSlice(int iSlice)
{
	GPUshared() AliHLTTPCCATrackletConstructor::MEM_LOCAL(AliHLTTPCCASharedMemory) sMem;
	AliHLTTPCCATracker *pTracker = ((AliHLTTPCCATracker*) gAliHLTTPCCATracker) + iSlice;
	AliHLTTPCCATrackletConstructor::AliHLTTPCCATrackletConstructorSingleSlice(pTracker, sMem);
}
#endif
